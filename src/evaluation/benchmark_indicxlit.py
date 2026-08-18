"""
Run AI4Bharat's IndicXlit (the published production transliteration model)
on our exact test sets, with the same normalization as our own models,
so the comparison table is apples-to-apples.

IndicXlit is a ~11M-param multilingual transformer trained on Aksharantar
across 21 languages. We expect it to beat our from-scratch models -- the
point is to measure the gap honestly.

Note: IndicXlit inference is CPU-bound through its own engine API and slow
per word, so we evaluate on a random subset per test set (default 2000)
rather than all 82k+ pairs. Subset size is recorded in the output JSON.
"""
import argparse
import json
import random
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess import normalize_ml

import jiwer

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

TEST_SETS = {
    "aksharantar": PROC_DIR / "test_aksharantar.jsonl",
    "dakshina_lexicon": PROC_DIR / "test_dakshina_lexicon.jsonl",
    "dakshina_sentences": PROC_DIR / "test_dakshina_sentences.jsonl",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=int, default=2000, help="pairs sampled per test set")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from ai4bharat.transliteration import XlitEngine
    print("loading IndicXlit engine (downloads model weights on first run)...")
    engine = XlitEngine("ml", beam_width=4, rescore=True)

    rng = random.Random(args.seed)
    results = {}

    for test_name, test_path in TEST_SETS.items():
        if not test_path.exists():
            print(f"[skip] {test_path} not found")
            continue
        with open(test_path, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        sample = rng.sample(pairs, min(args.subset, len(pairs)))

        preds, golds, top3_hits = [], [], 0
        t0 = time.time()
        for i, row in enumerate(sample):
            roman, gold = row["roman"], row["native"]
            try:
                out = engine.translit_word(roman, topk=3)
                # engine returns {"ml": [cand1, cand2, cand3]}
                cands = out["ml"] if isinstance(out, dict) else out
            except Exception:
                cands = [""]
            cands = [normalize_ml(c) for c in cands]
            pred = cands[0] if cands else ""
            preds.append(pred)
            golds.append(gold)
            if gold in cands[:3]:
                top3_hits += 1
            if (i + 1) % 200 == 0:
                print(f"  {test_name}: {i+1}/{len(sample)} ({time.time()-t0:.0f}s)")

        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        nonempty = [(p, g) for p, g in zip(preds, golds) if g]
        cer = jiwer.cer([g for _, g in nonempty], [p for p, _ in nonempty])
        top3 = top3_hits / len(sample)

        results[f"indicxlit/{test_name}"] = {
            "n": len(sample), "word_acc": acc, "cer": cer,
            "top3_acc": top3, "top3_n": len(sample),
        }
        print(f"indicxlit/{test_name:20s} n={len(sample):5d}  word_acc={acc:.4f}  "
              f"cer={cer:.4f}  top3_acc={top3:.4f}  ({time.time()-t0:.0f}s)")

    out_path = LOG_DIR / "indicxlit_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
