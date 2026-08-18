"""
Normalize, filter, and deduplicate raw (romanized, native) pairs.

Reads data/raw/aksharantar_mal.jsonl and data/raw/dakshina/...
Writes data/processed/pairs_aksharantar.jsonl and data/processed/pairs_dakshina_*.jsonl
"""
import json
import re
import unicodedata
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 40

# Malayalam Unicode block is U+0D00-U+0D7F.
MALAYALAM_BLOCK_RE = re.compile(r"^[ഀ-ൿ\s]+$")
ROMAN_ALLOWED_RE = re.compile(r"^[a-z\s]+$")

# Legacy chillu sequences (base consonant + virama + ZWJ) mapped to their
# atomic Unicode chillu codepoints. Both encodings render identically on
# screen, so scraped web text is a mix of both. If we don't unify them the
# model effectively has to learn two spellings for every chillu ending,
# which quietly wastes capacity and hurts eval (a "correct" prediction in
# the "wrong" encoding would count as a miss).
LEGACY_TO_ATOMIC = {
    "ണ്‍": "ൺ",  # ണ്‍ -> ൺ
    "ന്‍": "ൻ",  # ന്‍ -> ൻ
    "ര്‍": "ർ",  # ര്‍ -> ർ
    "ല്‍": "ൽ",  # ല്‍ -> ൽ
    "ള്‍": "ൾ",  # ള്‍ -> ൾ
}


def normalize_ml(text: str) -> str:
    """Canonicalize Malayalam text so visually-identical strings become byte-identical."""
    text = unicodedata.normalize("NFC", text)
    for legacy, atomic in LEGACY_TO_ATOMIC.items():
        text = text.replace(legacy, atomic)
    # Strip stray joiners that survive outside the chillu patterns above.
    text = text.replace("‌", "").replace("‍", "")
    return text.strip()


def normalize_roman(text: str) -> str:
    """Lowercase, strip accents/diacritics, collapse whitespace."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_pair(roman: str, native: str) -> bool:
    if not roman or not native:
        return False
    if len(roman) > MAX_LEN or len(native) > MAX_LEN:
        return False
    if not ROMAN_ALLOWED_RE.match(roman):
        return False
    if not MALAYALAM_BLOCK_RE.match(native):
        return False
    return True


def clean_pairs(raw_pairs):
    """raw_pairs: iterable of (roman, native) tuples -> deduped, cleaned list."""
    seen = set()
    cleaned = []
    for roman, native in raw_pairs:
        r = normalize_roman(roman)
        n = normalize_ml(native)
        if not is_valid_pair(r, n):
            continue
        key = (r, n)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"roman": r, "native": n})
    return cleaned


def process_aksharantar():
    src = RAW_DIR / "aksharantar_mal.jsonl"
    raw_pairs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_pairs.append((row["roman"], row["native"]))

    cleaned = clean_pairs(raw_pairs)
    out = PROC_DIR / "pairs_aksharantar.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Aksharantar: {len(raw_pairs)} raw -> {len(cleaned)} cleaned pairs -> {out}")


DAKSHINA_DIR = RAW_DIR / "dakshina_dataset_v1.0" / "ml"


def process_dakshina_lexicon():
    """Attested-romanization lexicon, held out entirely as a test set (never
    trained on) -- reports native \\t roman \\t annotator_count per line."""
    src = DAKSHINA_DIR / "lexicons" / "ml.translit.sampled.test.tsv"
    raw_pairs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            native, roman = parts[0], parts[1]
            raw_pairs.append((roman, native))

    cleaned = clean_pairs(raw_pairs)
    out = PROC_DIR / "test_dakshina_lexicon.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Dakshina lexicon: {len(raw_pairs)} raw -> {len(cleaned)} cleaned pairs -> {out}")


def process_dakshina_sentences():
    """Word-level pairs extracted (by Dakshina's own alignment) from real
    romanized Wikipedia sentences -- harder and more realistic than the
    lexicon, since it reflects actual free-text romanization, not elicited
    single-word attestations. </s> sentence-boundary markers are dropped."""
    src = DAKSHINA_DIR / "romanized" / "ml.romanized.rejoined.aligned.tsv"
    raw_pairs = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            native, roman = parts[0], parts[1]
            if native == "</s>" or roman == "</s>":
                continue
            raw_pairs.append((roman, native))

    cleaned = clean_pairs(raw_pairs)
    out = PROC_DIR / "test_dakshina_sentences.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Dakshina sentences (word-aligned): {len(raw_pairs)} raw -> {len(cleaned)} cleaned pairs -> {out}")


if __name__ == "__main__":
    process_aksharantar()
    process_dakshina_lexicon()
    process_dakshina_sentences()
