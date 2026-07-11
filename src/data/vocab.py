"""
Build character-level vocabularies for the source (romanized Latin) and
target (Malayalam) sides, from the training split only (never from val/test
-- that would leak information about which characters appear in eval data).
"""
import json
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = range(4)


class Vocab:
    def __init__(self, chars):
        self.itos = list(SPECIALS) + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text, add_sos_eos=True):
        ids = [self.stoi.get(c, UNK_IDX) for c in text]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids, strip_specials=True):
        chars = [self.itos[i] for i in ids]
        if strip_specials:
            chars = [c for c in chars if c not in SPECIALS]
        return "".join(chars)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            itos = json.load(f)
        v = cls.__new__(cls)
        v.itos = itos
        v.stoi = {c: i for i, c in enumerate(itos)}
        return v


def build_vocabs():
    train_path = PROC_DIR / "train.jsonl"
    roman_chars, native_chars = set(), set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            roman_chars.update(row["roman"])
            native_chars.update(row["native"])

    src_vocab = Vocab(roman_chars)
    tgt_vocab = Vocab(native_chars)

    src_vocab.save(PROC_DIR / "vocab_src.json")
    tgt_vocab.save(PROC_DIR / "vocab_tgt.json")

    print(f"src vocab: {len(src_vocab)} symbols ({len(roman_chars)} chars + {len(SPECIALS)} specials)")
    print(f"tgt vocab: {len(tgt_vocab)} symbols ({len(native_chars)} chars + {len(SPECIALS)} specials)")
    return src_vocab, tgt_vocab


if __name__ == "__main__":
    build_vocabs()
