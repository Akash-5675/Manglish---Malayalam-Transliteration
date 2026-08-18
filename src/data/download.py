"""
Download raw datasets into data/raw/. Run this once.

Sources:
- Aksharantar (AI4Bharat): word-level (romanized, native) pairs for Malayalam.
- Dakshina (Google): lexicon pairs + full romanized sentences, used only as
  held-out test sets (never for training) so we can report out-of-domain accuracy.
"""
import json
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DAKSHINA_URL = (
    "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
)


def download_aksharantar():
    """Download mal.zip directly from the HF repo (load_dataset's script-based
    loader no longer works with recent `datasets` versions), extract, and
    normalize every split's rows into one flat JSONL."""
    out_path = RAW_DIR / "aksharantar_mal.jsonl"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print("Downloading mal.zip from ai4bharat/Aksharantar...")
    zip_path = hf_hub_download(repo_id="ai4bharat/Aksharantar", repo_type="dataset", filename="mal.zip")

    # mal.zip contains mal_train.json / mal_valid.json / mal_test.json, each
    # JSON-lines with fields "english word" / "native word" (verified by
    # inspection -- Aksharantar's schema has drifted across releases so we
    # don't trust the docs blindly).
    rows_written = 0
    with zipfile.ZipFile(zip_path) as zf, open(out_path, "w", encoding="utf-8") as out_f:
        for member in zf.namelist():
            if not member.endswith(".json"):
                continue
            split_name = Path(member).stem  # mal_train / mal_valid / mal_test
            with zf.open(member) as in_f:
                for line in in_f:
                    row = json.loads(line)
                    roman = row["english word"]
                    native = row["native word"]
                    out_f.write(json.dumps({"roman": roman, "native": native, "split_src": split_name}, ensure_ascii=False) + "\n")
                    rows_written += 1
    print(f"Saved {rows_written} rows -> {out_path}")


def download_dakshina():
    """Download and extract the Dakshina v1.0 archive (contains all languages; we keep only ml/)."""
    dest = RAW_DIR / "dakshina_dataset_v1.0"
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    print("Downloading Dakshina dataset (this is a large tar, may take a while)...")
    r = requests.get(DAKSHINA_URL, stream=True, timeout=120)
    r.raise_for_status()
    tar_path = RAW_DIR / "dakshina_dataset_v1.0.tar"
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    import tarfile
    print("Extracting (filtering to ml/ only)...")
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if "/ml/" in m.name or m.name.endswith("/ml")]
        tar.extractall(path=RAW_DIR, members=members)
    tar_path.unlink()
    print(f"Saved -> {dest}")


if __name__ == "__main__":
    download_aksharantar()
    download_dakshina()
