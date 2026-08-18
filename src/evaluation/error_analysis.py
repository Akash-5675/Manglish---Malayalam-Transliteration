"""
Sample wrong predictions from both models and categorize them into the
failure buckets the guide calls out: ambiguous romanizations (digraphs,
gemination, vowel length), chillu endings, English loanwords, and conjunct/
virama construction errors.

Categorization is heuristic (regex/wordlist based), not hand-labeled --
that's a documented simplification, not a claim of perfect labels. It's
meant to surface *patterns*, which is what the write-up needs.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab
from models.lstm_attention import Seq2SeqLSTM
from models.transformer import Seq2SeqTransformer
from training.dataset import make_dataloader
from inference.decode import greedy_decode, greedy_decode_transformer

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# Common English loanwords that show up untransliterated-in-spirit inside
# Malayalam text -- a small curated list, not exhaustive.
LOANWORDS = {
    "computer", "phone", "mobile", "television", "doctor", "bank", "school",
    "college", "hospital", "bus", "car", "train", "ticket", "internet",
    "email", "office", "manager", "company", "market", "hotel", "restaurant",
    "petrol", "diesel", "battery", "camera", "channel", "channel", "cricket",
    "football", "movie", "cinema", "theatre", "party", "committee", "minister",
    "government", "university", "student", "teacher", "software", "printer",
}

CHILLU_CHARS = set("ൺൻർൽൾ")
DIGRAPH_PATTERNS = re.compile(r"nj|zh")
GEMINATION_PATTERN = re.compile(r"(.)\1")  # any doubled letter, e.g. kk, tt, pp
VOWEL_LENGTH_PATTERN = re.compile(r"aa|ee|oo|ii")
VIRAMA = "്"


def categorize(roman, gold, pred):
    cats = []
    if gold and gold[-1] in CHILLU_CHARS:
        cats.append("chillu_ending")
    if roman in LOANWORDS:
        cats.append("english_loanword")
    if DIGRAPH_PATTERNS.search(roman):
        cats.append("digraph_ambiguity_nj_zh")
    if GEMINATION_PATTERN.search(roman):
        cats.append("gemination_ambiguity")
    if VOWEL_LENGTH_PATTERN.search(roman):
        cats.append("vowel_length_ambiguity")
    if VIRAMA in gold or VIRAMA in pred:
        # only flag as a conjunct/virama error if the mismatch actually
        # involves a virama position, not just any error on a word that
        # happens to contain one
        gold_virama_positions = [i for i, c in enumerate(gold) if c == VIRAMA]
        pred_virama_positions = [i for i, c in enumerate(pred) if c == VIRAMA]
        if gold_virama_positions != pred_virama_positions:
            cats.append("conjunct_virama_error")
    if not cats:
        cats.append("other_unclassified")
    return cats


def load_model(model_type, ckpt_path, src_vocab, tgt_vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if model_type == "lstm":
        model = Seq2SeqLSTM(len(src_vocab), len(tgt_vocab)).to(device)
    else:
        model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def collect_errors(model, model_type, jsonl_path, src_vocab, tgt_vocab, device, batch_size=256):
    """Returns list of (roman, gold, pred) for every wrong prediction."""
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    roman_by_line = [r["roman"] for r in rows]

    loader = make_dataloader(jsonl_path, src_vocab, tgt_vocab, batch_size, shuffle=False)
    errors = []
    idx = 0
    with torch.no_grad():
        for src, src_lens, tgt, _ in loader:
            src = src.to(device)
            pred_ids = greedy_decode(model, src, src_lens) if model_type == "lstm" else greedy_decode_transformer(model, src)
            for i, ids in enumerate(pred_ids):
                pred = tgt_vocab.decode(ids, strip_specials=False)
                gold = tgt_vocab.decode(tgt[i].tolist(), strip_specials=True)
                roman = roman_by_line[idx]
                if pred != gold:
                    errors.append((roman, gold, pred))
                idx += 1
    return errors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab = Vocab.load(PROC_DIR / "vocab_src.json")
    tgt_vocab = Vocab.load(PROC_DIR / "vocab_tgt.json")

    models = {
        "lstm": load_model("lstm", CKPT_DIR / "lstm_1M_best.pt", src_vocab, tgt_vocab, device),
        "transformer": load_model("transformer", CKPT_DIR / "transformer_1M_best.pt", src_vocab, tgt_vocab, device),
    }
    # Dakshina sentences: hardest, most realistic test set -- richest source
    # of genuinely interesting errors (in-domain Aksharantar errors are
    # rarer and less illustrative of real failure modes).
    test_path = PROC_DIR / "test_dakshina_sentences.jsonl"

    report = {}
    rng = random.Random(42)

    for model_name, model in models.items():
        print(f"collecting errors for {model_name}...")
        errors = collect_errors(model, model_name, test_path, src_vocab, tgt_vocab, device)
        sample = rng.sample(errors, min(200, len(errors)))

        cat_counts = Counter()
        examples_by_cat = defaultdict(list)
        for roman, gold, pred in sample:
            cats = categorize(roman, gold, pred)
            for c in cats:
                cat_counts[c] += 1
                if len(examples_by_cat[c]) < 5:
                    examples_by_cat[c].append({"roman": roman, "gold": gold, "pred": pred})

        report[model_name] = {
            "total_errors_in_test_set": len(errors),
            "total_test_set_size": sum(1 for _ in open(test_path, encoding="utf-8")),
            "sample_size": len(sample),
            "category_counts": dict(cat_counts),
            "examples": examples_by_cat,
        }

        print(f"\n{model_name}: {len(errors)}/{report[model_name]['total_test_set_size']} wrong "
              f"({len(errors)/report[model_name]['total_test_set_size']:.1%}), sampled {len(sample)}")
        for cat, count in cat_counts.most_common():
            print(f"  {cat:30s} {count:4d}  ({count/len(sample):.1%} of sample)")

    out_path = LOG_DIR / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
