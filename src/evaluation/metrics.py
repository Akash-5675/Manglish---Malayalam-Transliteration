"""
Word accuracy, CER, and top-3 accuracy for both models across all 3 test sets.

Word accuracy + CER: computed on the FULL test set using batched greedy decode.
Top-3 accuracy: computed on a random subset using beam search, since beam
search here is one-example-at-a-time and doesn't scale to 80k+ pairs in
reasonable time -- this is a deliberate, documented tradeoff, not an
oversight (see README for the exact subset sizes used).
"""
import argparse
import json
import random
from pathlib import Path

import jiwer
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab, PAD_IDX
from models.lstm_attention import Seq2SeqLSTM
from models.transformer import Seq2SeqTransformer
from training.dataset import make_dataloader
from inference.decode import (
    greedy_decode, greedy_decode_transformer,
    beam_search_decode, beam_search_decode_transformer,
)

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

TEST_SETS = {
    "aksharantar": PROC_DIR / "test_aksharantar.jsonl",
    "dakshina_lexicon": PROC_DIR / "test_dakshina_lexicon.jsonl",
    "dakshina_sentences": PROC_DIR / "test_dakshina_sentences.jsonl",
}


def load_model(model_type, ckpt_path, src_vocab, tgt_vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if model_type == "lstm":
        model = Seq2SeqLSTM(len(src_vocab), len(tgt_vocab)).to(device)
    else:
        model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def greedy_predict_all(model, model_type, jsonl_path, src_vocab, tgt_vocab, device, batch_size=256):
    """Batched greedy decode over an entire test set. Returns (preds, golds) word lists."""
    loader = make_dataloader(jsonl_path, src_vocab, tgt_vocab, batch_size, shuffle=False)
    preds, golds = [], []
    with torch.no_grad():
        for src, src_lens, tgt, _ in loader:
            src = src.to(device)
            if model_type == "lstm":
                pred_ids = greedy_decode(model, src, src_lens)
            else:
                pred_ids = greedy_decode_transformer(model, src)
            for i, ids in enumerate(pred_ids):
                preds.append(tgt_vocab.decode(ids, strip_specials=False))
                golds.append(tgt_vocab.decode(tgt[i].tolist(), strip_specials=True))
    return preds, golds


def word_accuracy(preds, golds):
    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / len(golds)


def char_error_rate(preds, golds):
    # jiwer.cer computes edit-distance at the character level; guard against
    # empty gold strings (jiwer errors on an all-empty reference set).
    pairs = [(p, g) for p, g in zip(preds, golds) if g]
    if not pairs:
        return float("nan")
    hyps, refs = zip(*[(p, g) for p, g in pairs])
    return jiwer.cer(list(refs), list(hyps))


def top3_accuracy(model, model_type, jsonl_path, src_vocab, tgt_vocab, device, subset_size, seed=42):
    with open(jsonl_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(subset_size, len(pairs)))

    hits = 0
    with torch.no_grad():
        for row in sample:
            roman, native = row["roman"], row["native"]
            src_ids = torch.tensor([src_vocab.encode(roman, add_sos_eos=False)], dtype=torch.long, device=device)
            if model_type == "lstm":
                src_len = torch.tensor([len(roman)], dtype=torch.long)
                candidates = beam_search_decode(model, src_ids, src_len, beam_size=5)
            else:
                candidates = beam_search_decode_transformer(model, src_ids, beam_size=5)
            top3 = [tgt_vocab.decode(ids, strip_specials=False) for ids, _ in candidates[:3]]
            if native in top3:
                hits += 1
    return hits / len(sample), len(sample)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top3-subset", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    src_vocab = Vocab.load(PROC_DIR / "vocab_src.json")
    tgt_vocab = Vocab.load(PROC_DIR / "vocab_tgt.json")

    models = {
        "lstm": load_model("lstm", CKPT_DIR / "lstm_1M_best.pt", src_vocab, tgt_vocab, device),
        "transformer": load_model("transformer", CKPT_DIR / "transformer_1M_best.pt", src_vocab, tgt_vocab, device),
    }

    results = {}
    for model_name, model in models.items():
        for test_name, test_path in TEST_SETS.items():
            if not test_path.exists():
                print(f"[skip] {test_path} not found")
                continue
            preds, golds = greedy_predict_all(model, model_name, test_path, src_vocab, tgt_vocab, device)
            acc = word_accuracy(preds, golds)
            cer = char_error_rate(preds, golds)
            top3, n_sub = top3_accuracy(model, model_name, test_path, src_vocab, tgt_vocab, device, args.top3_subset)

            key = f"{model_name}/{test_name}"
            results[key] = {
                "n": len(golds), "word_acc": acc, "cer": cer,
                "top3_acc": top3, "top3_n": n_sub,
            }
            print(f"{key:35s} n={len(golds):6d}  word_acc={acc:.4f}  cer={cer:.4f}  "
                  f"top3_acc={top3:.4f} (n={n_sub})")

    out_path = Path(__file__).resolve().parents[2] / "logs" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
