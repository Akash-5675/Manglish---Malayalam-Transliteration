# Manglish → Malayalam Transliteration

**Live demo: [manglish-malayalam-transliteration.vercel.app](https://manglish-malayalam-transliteration.vercel.app/)** — runs entirely in your browser, no server.

Character-level neural transliteration from romanized Malayalam ("Manglish") to Malayalam script — two architectures trained from scratch in PyTorch, evaluated honestly across three test sets, benchmarked against a production system, and deployed as a fully client-side browser demo (ONNX Runtime Web — no server, no API).

```
Input:  enthokke undu vishesham
Output: എന്തൊക്കെ ഉണ്ട് വിശേഷം
```

This is how most Malayalis actually type Malayalam. The task is harder than it looks: romanization is not standardized ("ente", "ende", "enthe" all mean എന്റെ), one Latin letter can map to several Malayalam characters depending on context, and vowel length / gemination / chillu endings are chronically ambiguous.

## Results

Word accuracy (exact match), character error rate, and top-3 accuracy (beam search, k=5), evaluated on three test sets of increasing difficulty:

| Test set | Metric | BiLSTM+Attn | Transformer | IndicXlit* |
|---|---|---|---|---|
| **Aksharantar test** (in-domain, 82k pairs) | Word acc | 77.97% | **78.30%** | — |
| | CER | 2.63% | **2.57%** | — |
| | Top-3 acc | **94.80%** | 94.70% | — |
| **Dakshina lexicon** (out-of-domain, 5.6k pairs) | Word acc | 61.57% | **61.60%** | — |
| | CER | **8.17%** | 8.36% | — |
| | Top-3 acc | **77.40%** | 76.60% | — |
| **Dakshina sentences** (real Wikipedia text, 42k pairs) | Word acc | 50.85% | **52.04%** | — |
| | CER | 9.51% | **9.31%** | — |
| | Top-3 acc | 68.30% | **70.90%** | — |

*IndicXlit column pending (runs on Kaggle; fairseq doesn't build on Windows — see `notebooks/03_kaggle_indicxlit.ipynb`).

| | BiLSTM+Attn | Transformer |
|---|---|---|
| Parameters | 8.2M | **5.6M** |
| Training (1M pairs, 15 epochs, T4) | ~2.1h | ~1.4h |

**Takeaways**

- The Transformer matches or beats the BiLSTM on every test set with **32% fewer parameters**, with the biggest edge on the hardest set (real-text romanization) — consistent with self-attention capturing long-range dependencies more directly than recurrence.
- The in-domain → out-of-domain drop (78% → 51-62%) is the honest number many transliteration writeups omit. Real free-text romanization is far more variable than curated word lists.
- Top-3 accuracy is dramatically higher than top-1 everywhere (94.8% vs 78% in-domain): the right answer is usually *in the model's beam*, which motivates the click-to-choose UI in the demo.
- CER stays under 10% even where word accuracy is ~51% — wrong predictions are typically off by a character or two, not garbage.

## Honest-evaluation details (the part that keeps the numbers real)

- **Leak-free splits.** Aksharantar has many romanizations per Malayalam word. Splitting randomly by *pair* would put the same word in train and test. We group all pairs by native word and assign whole groups to splits, with a hard assertion of zero overlap (`src/data/split.py`).
- **Unicode normalization on both sides.** Malayalam chillu letters have two encodings that render identically (atomic vs. base+virama+ZWJ). Everything — training data, predictions, references, and the baseline's outputs — passes through the same `normalize_ml()` so no prediction is marked wrong for using the "other" encoding.
- **Documented dataset noise.** Dakshina's sentence-derived pairs come from automatic word alignment, which occasionally misaligns (we found pairs like `clokkukalum → ഉത്സവം` in the error sample). Our Dakshina-sentences accuracy is therefore a conservative lower bound; the noise concentrates in the "error" bucket by construction.
- **Fixed seeds** for Python/NumPy/PyTorch (`--seed`, default 42). Two independent training runs landed within ~2 points of each other before seeding; with seeds, runs are reproducible.

## Error analysis

200 sampled errors per model on the hardest test set, categorized heuristically (`src/evaluation/error_analysis.py`):

| Category | BiLSTM | Transformer | Example |
|---|---|---|---|
| Conjunct/virama construction | 55.5% | 61.5% | നല്കിയ vs ൽകിയ (wrong stacking) |
| Gemination (kk/tt/pp…) | 40.0% | 45.5% | ഡയറക്റ്ററി vs ഡയറക്ടറി |
| Chillu endings | 21.0% | 21.0% | അവൻ vs അവന് |
| Vowel length (a/aa, e/ee) | 19.0% | 23.5% | സർക്കാർ vs സക്കാർ |
| Digraphs (nj, zh) | 3.0% | 3.5% | ഴ vs ശ് |

(Categories overlap; a word can hit several.) Most failures are *phonologically plausible* alternatives — often spellings a human might also produce — rather than random noise.

## Browser demo

**[manglish-malayalam-transliteration.vercel.app](https://manglish-malayalam-transliteration.vercel.app/)**

The Transformer runs **entirely client-side**: exported to ONNX (encoder + decoder-step as separate graphs, greedy/beam loop reimplemented in JS), executed by ONNX Runtime Web's WASM backend. ~25MB one-time model download, then every keystroke is processed locally. Type Manglish, get Malayalam, click any word for the model's top-3 candidates.

```bash
cd demo && npm install && npm run dev
```

The export includes a parity check: ONNX output must be byte-identical to PyTorch on a test battery before the export is accepted (`src/inference/export_onnx.py`).

## Reproducing everything

```bash
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Week 1: data (downloads ~600MB, builds 3.9M/82k/82k splits + 2 Dakshina test sets)
python src/data/download.py
python src/data/preprocess.py
python src/data/split.py
python src/data/vocab.py

# Weeks 2-3: training (a T4/P100 is plenty; see notebooks/02_kaggle_training.ipynb)
python src/training/train.py --max-train-samples 1000000 --epochs 15 --run-name lstm_1M
python src/training/train_transformer.py --max-train-samples 1000000 --epochs 15 --run-name transformer_1M

# Evaluation
python src/evaluation/metrics.py                  # both models × three test sets
python src/evaluation/error_analysis.py           # categorized failure analysis
# IndicXlit baseline: notebooks/03_kaggle_indicxlit.ipynb (Linux-only dependency)

# Demo
python src/inference/export_onnx.py               # export + parity check
cd demo && npm install && npm run build
```

## Repo layout

```
src/
  data/         download.py · preprocess.py (Unicode normalization) · split.py (leak-free) · vocab.py
  models/       lstm_attention.py (BiLSTM + Luong attention) · transformer.py
  training/     train.py · train_transformer.py (Noam warmup, label smoothing) · dataset.py (length bucketing)
  inference/    decode.py (greedy + beam, both models) · export_onnx.py
  evaluation/   metrics.py · error_analysis.py · benchmark_indicxlit.py
notebooks/      01_data_stats.ipynb · 02_kaggle_training.ipynb · 03_kaggle_indicxlit.ipynb
demo/           React + onnxruntime-web client-side demo
```

## Data & citations

- **[Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)** (AI4Bharat) — 4.1M Malayalam pairs, training + in-domain test. *Madhani et al., 2022.*
- **[Dakshina](https://github.com/google-research-datasets/dakshina)** (Google Research) — held-out test only. *Roark et al., LREC 2020.*
- **[IndicXlit](https://ai4bharat.iitm.ac.in/indic-xlit/)** (AI4Bharat) — published baseline. *Madhani et al., 2022.*
