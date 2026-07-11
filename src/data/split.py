"""
Leak-free train/val/test split for the Aksharantar pairs.

Groups all (roman, native) pairs by their native Malayalam word, then assigns
whole groups to a split, so no native word appears in more than one split
-- even though it may have several romanized spellings, which all move together.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.96, 0.02  # test gets the remainder (~0.02)


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_by_native(pairs):
    groups = defaultdict(list)
    for p in pairs:
        groups[p["native"]].append(p)
    return groups


def split_groups(groups, seed=SEED):
    native_words = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(native_words)

    n = len(native_words)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_words = set(native_words[:n_train])
    val_words = set(native_words[n_train:n_train + n_val])
    test_words = set(native_words[n_train + n_val:])

    def flatten(words):
        out = []
        for w in words:
            out.extend(groups[w])
        return out

    return flatten(train_words), flatten(val_words), flatten(test_words), (train_words, val_words, test_words)


def assert_no_leak(train_words, val_words, test_words):
    assert train_words.isdisjoint(val_words), "train/val native-word overlap!"
    assert train_words.isdisjoint(test_words), "train/test native-word overlap!"
    assert val_words.isdisjoint(test_words), "val/test native-word overlap!"


def write_jsonl(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    pairs = load_pairs(PROC_DIR / "pairs_aksharantar.jsonl")
    groups = group_by_native(pairs)
    train, val, test, (tw, vw, sw) = split_groups(groups)

    assert_no_leak(tw, vw, sw)
    print("Leak check passed: zero native-word overlap between splits.")

    write_jsonl(train, PROC_DIR / "train.jsonl")
    write_jsonl(val, PROC_DIR / "val.jsonl")
    write_jsonl(test, PROC_DIR / "test_aksharantar.jsonl")

    print(f"native words: train={len(tw)} val={len(vw)} test={len(sw)}")
    print(f"pairs:        train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
