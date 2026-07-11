"""
Greedy decode (fast, used during training for validation accuracy) and
beam search decode (slower, used for final eval + top-3 accuracy).
"""
import torch

SOS_IDX, EOS_IDX = 1, 2


@torch.no_grad()
def greedy_decode_transformer(model, src, max_len=40):
    """src: (batch, src_len). Returns list of predicted id-lists (no specials)."""
    model.eval()
    device = src.device
    batch_size = src.size(0)

    memory, src_key_padding_mask = model.encode(src)
    tgt_so_far = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    outputs = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits = model.decode_step(memory, src_key_padding_mask, tgt_so_far)
        next_token = logits.argmax(dim=1)
        for i in range(batch_size):
            if not finished[i]:
                if next_token[i].item() == EOS_IDX:
                    finished[i] = True
                else:
                    outputs[i].append(next_token[i].item())
        tgt_so_far = torch.cat([tgt_so_far, next_token.unsqueeze(1)], dim=1)
        if finished.all():
            break

    return outputs


@torch.no_grad()
def beam_search_decode_transformer(model, src, beam_size=5, max_len=40, length_penalty=0.7):
    """Single example (src: (1, src_len)). Returns list of (token_ids, score) sorted best-first."""
    model.eval()
    device = src.device

    memory, src_key_padding_mask = model.encode(src)

    beams = [([SOS_IDX], 0.0, False)]
    completed = []

    for _ in range(max_len):
        candidates = []
        active = [b for b in beams if not b[2]]
        if not active:
            candidates.extend(beams)
        else:
            tgt_batch = torch.tensor([b[0] for b in active], dtype=torch.long, device=device)
            mem_rep = memory.expand(len(active), -1, -1)
            mask_rep = src_key_padding_mask.expand(len(active), -1)
            logits = model.decode_step(mem_rep, mask_rep, tgt_batch)  # (n_active, vocab)
            log_probs = torch.log_softmax(logits, dim=1)
            topk_logp, topk_idx = log_probs.topk(beam_size, dim=1)

            for bi, (tokens, score, _) in enumerate(active):
                for lp, idx in zip(topk_logp[bi].tolist(), topk_idx[bi].tolist()):
                    is_eos = idx == EOS_IDX
                    candidates.append((tokens + [idx], score + lp, is_eos))
            candidates.extend(b for b in beams if b[2])

        def norm_score(c):
            tokens, score = c[0], c[1]
            length = len(tokens) - 1
            return score / (length ** length_penalty if length > 0 else 1.0)

        candidates.sort(key=norm_score, reverse=True)

        beams = []
        for cand in candidates:
            if cand[2]:
                completed.append(cand)
            else:
                beams.append(cand)
            if len(beams) == beam_size:
                break
        if not beams:
            break

    completed.extend(beams)
    completed.sort(key=lambda c: c[1] / ((len(c[0]) - 1) ** length_penalty if len(c[0]) > 1 else 1.0), reverse=True)

    results = []
    for tokens, score, *_ in completed[:beam_size]:
        ids = tokens[1:]
        if EOS_IDX in tokens:
            eos_pos = tokens.index(EOS_IDX)
            ids = tokens[1:eos_pos]
        results.append((ids, score))
    return results


@torch.no_grad()
def greedy_decode(model, src, src_lens, max_len=40):
    """src: (batch, src_len). Returns list of predicted id-lists (no specials)."""
    model.eval()
    device = src.device
    batch_size = src.size(0)

    enc_outputs, src_mask, hidden, cell = model.encode(src, src_lens)
    context = torch.zeros(batch_size, model.enc_output_dim, device=device)
    input_token = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    outputs = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits, context, hidden, cell, _ = model.decoder.step(
            input_token, context, hidden, cell, enc_outputs, src_mask
        )
        next_token = logits.argmax(dim=1)
        for i in range(batch_size):
            if not finished[i]:
                if next_token[i].item() == EOS_IDX:
                    finished[i] = True
                else:
                    outputs[i].append(next_token[i].item())
        input_token = next_token
        if finished.all():
            break

    return outputs


@torch.no_grad()
def beam_search_decode(model, src, src_len, beam_size=5, max_len=40, length_penalty=0.7):
    """Single example (src: (1, src_len)). Returns list of (token_ids, score) sorted best-first."""
    model.eval()
    device = src.device

    enc_outputs, src_mask, hidden, cell = model.encode(src, src_len)
    enc_output_dim = model.enc_output_dim

    # Each beam: (tokens, log_prob_sum, context, hidden, cell, finished)
    beams = [([SOS_IDX], 0.0, torch.zeros(1, enc_output_dim, device=device), hidden, cell, False)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for tokens, score, context, h, c, finished in beams:
            if finished:
                candidates.append((tokens, score, context, h, c, True))
                continue
            input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
            logits, new_context, new_h, new_c, _ = model.decoder.step(
                input_token, context, h, c, enc_outputs, src_mask
            )
            log_probs = torch.log_softmax(logits, dim=1).squeeze(0)  # (vocab,)
            topk_logp, topk_idx = log_probs.topk(beam_size)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                is_eos = idx == EOS_IDX
                candidates.append((tokens + [idx], score + lp, new_context, new_h, new_c, is_eos))

        def norm_score(c):
            tokens, score = c[0], c[1]
            length = len(tokens) - 1  # exclude <sos>
            return score / (length ** length_penalty if length > 0 else 1.0)

        candidates.sort(key=norm_score, reverse=True)

        beams = []
        for cand in candidates:
            if cand[5]:  # finished (hit <eos>)
                completed.append(cand)
            else:
                beams.append(cand)
            if len(beams) == beam_size:
                break

        if not beams:
            break

    completed.extend(beams)  # anything still unfinished at max_len, keep as-is
    completed.sort(key=lambda c: c[1] / ((len(c[0]) - 1) ** length_penalty if len(c[0]) > 1 else 1.0), reverse=True)

    results = []
    for tokens, score, *_ in completed[:beam_size]:
        ids = [t for t in tokens[1:] if t != EOS_IDX]  # drop <sos>, stop at/exclude <eos>
        if EOS_IDX in tokens:
            eos_pos = tokens.index(EOS_IDX)
            ids = tokens[1:eos_pos]
        results.append((ids, score))
    return results
