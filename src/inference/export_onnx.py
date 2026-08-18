"""
Export the trained Transformer to ONNX for the browser demo.

The model is exported as TWO graphs, because ONNX represents computation
graphs, not loops:
  encoder.onnx       -- src token ids -> encoder memory (run once per word)
  decoder_step.onnx  -- (memory, prefix so far, causal mask) -> next-token
                        logits (run repeatedly; the JS side loops)

Batch is fixed at 1 (the demo transliterates one word at a time), which
conveniently removes all padding-mask plumbing. The causal mask is built
by the caller and passed in as an input -- generating it inside the graph
bakes the sequence length in as a constant with the legacy tracer.

After export, verifies parity: greedy decode via onnxruntime must exactly
match PyTorch greedy decode on a sample of test words.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab
from models.transformer import Seq2SeqTransformer

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
DEMO_DIR = Path(__file__).resolve().parents[2] / "demo" / "public" / "model"

SOS_IDX, EOS_IDX = 1, 2
MAX_LEN = 40


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        # src: (1, src_len) int64. No padding mask needed at batch=1.
        return self.model.transformer.encoder(self.model._embed_src(src))


class DecoderStepWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory, tgt_so_far, causal_mask):
        # memory: (1, src_len, d_model); tgt_so_far: (1, cur_len) int64
        # causal_mask: (cur_len, cur_len) float, 0 on/below diagonal, -inf above
        tgt_emb = self.model._embed_tgt(tgt_so_far)
        # tgt_is_causal=False: skip nn.TransformerDecoder's causal-mask
        # auto-detection, which inspects mask *contents* (data-dependent,
        # breaks torch.export). The mask we pass is causal; the layer just
        # doesn't need to know that.
        out = self.model.transformer.decoder(
            tgt_emb, memory, tgt_mask=causal_mask, tgt_is_causal=False
        )
        return self.model.generator(out[:, -1, :])  # (1, vocab)


def build_causal_mask(size):
    mask = torch.zeros(size, size)
    mask.masked_fill_(torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1), float("-inf"))
    return mask


def onnx_greedy_decode(enc_sess, dec_sess, src_ids, max_len=MAX_LEN):
    memory = enc_sess.run(None, {"src": np.asarray([src_ids], dtype=np.int64)})[0]
    tokens = [SOS_IDX]
    for _ in range(max_len):
        mask = build_causal_mask(len(tokens)).numpy()
        logits = dec_sess.run(None, {
            "memory": memory,
            "tgt_so_far": np.asarray([tokens], dtype=np.int64),
            "causal_mask": mask,
        })[0]
        next_tok = int(logits[0].argmax())
        if next_tok == EOS_IDX:
            break
        tokens.append(next_tok)
    return tokens[1:]


def main():
    device = torch.device("cpu")  # export from CPU: simplest, and parity-checks against the same backend
    src_vocab = Vocab.load(PROC_DIR / "vocab_src.json")
    tgt_vocab = Vocab.load(PROC_DIR / "vocab_tgt.json")

    ckpt = torch.load(CKPT_DIR / "transformer_1M_best.pt", map_location=device, weights_only=False)
    model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    enc = EncoderWrapper(model).eval()
    dec = DecoderStepWrapper(model).eval()

    # The legacy tracer bakes concrete sequence lengths into the attention
    # reshapes (a 4-char dummy word produced a graph that only accepted
    # 4-char words). The dynamo exporter traces shapes symbolically instead.
    # max=59, not 60: the positional-encoding buffer is exactly 60 long, and
    # slicing it at its full length trips a specialization guard in torch.export.
    src_len_dim = torch.export.Dim("src_len", min=1, max=59)
    cur_len_dim = torch.export.Dim("cur_len", min=1, max=59)

    dummy_src = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    torch.onnx.export(
        enc, (dummy_src,), DEMO_DIR / "encoder.onnx",
        input_names=["src"], output_names=["memory"],
        dynamic_shapes={"src": {1: src_len_dim}},
        dynamo=True,
        external_data=False,  # single self-contained file: simpler to fetch from JS
    )

    with torch.no_grad():
        dummy_memory = enc(dummy_src)
    dummy_tgt = torch.tensor([[SOS_IDX, 10, 11]], dtype=torch.long)
    dummy_mask = build_causal_mask(3)
    torch.onnx.export(
        dec, (dummy_memory, dummy_tgt, dummy_mask), DEMO_DIR / "decoder_step.onnx",
        input_names=["memory", "tgt_so_far", "causal_mask"], output_names=["logits"],
        dynamic_shapes={
            "memory": {1: src_len_dim},
            "tgt_so_far": {1: cur_len_dim},
            "causal_mask": {0: cur_len_dim, 1: cur_len_dim},
        },
        dynamo=True,
        external_data=False,
    )
    print(f"exported -> {DEMO_DIR}/encoder.onnx, decoder_step.onnx")

    # Ship the vocabs with the model so JS can encode/decode.
    for name in ["vocab_src.json", "vocab_tgt.json"]:
        (DEMO_DIR / name).write_text(
            (PROC_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    # ---- parity check ----
    import onnxruntime as ort
    from inference.decode import greedy_decode_transformer

    enc_sess = ort.InferenceSession(str(DEMO_DIR / "encoder.onnx"))
    dec_sess = ort.InferenceSession(str(DEMO_DIR / "decoder_step.onnx"))

    test_words = ["vishesham", "ente", "malayalam", "computer", "nammal",
                  "kozhikode", "thiruvananthapuram", "sneham", "veedu", "pattanam"]
    mismatches = 0
    for word in test_words:
        ids = src_vocab.encode(word, add_sos_eos=False)
        onnx_out = tgt_vocab.decode(onnx_greedy_decode(enc_sess, dec_sess, ids))
        with torch.no_grad():
            pt_out = tgt_vocab.decode(
                greedy_decode_transformer(model, torch.tensor([ids], dtype=torch.long))[0]
            )
        status = "OK " if onnx_out == pt_out else "MISMATCH"
        if onnx_out != pt_out:
            mismatches += 1
        print(f"{status} {word:22s} onnx={onnx_out!r} torch={pt_out!r}")

    if mismatches:
        raise SystemExit(f"{mismatches}/{len(test_words)} words mismatched -- export is NOT faithful")
    print(f"\nparity check passed: {len(test_words)}/{len(test_words)} identical")


if __name__ == "__main__":
    main()
