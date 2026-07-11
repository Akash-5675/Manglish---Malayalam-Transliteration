# Manglish → Malayalam Transliteration

Character-level neural transliteration from romanized Malayalam (Manglish) to Malayalam script, comparing two from-scratch architectures — a BiLSTM encoder-decoder with attention, and a small Transformer — against AI4Bharat's IndicXlit as a baseline. No external APIs; everything trained and run locally / on free GPU notebooks.

```
Input:  enthokke undu vishesham
Output: എന്തൊക്കെ ഉണ്ട് വിശേഷം
```

## Status

Work in progress. Current state:

- [x] Data pipeline: Aksharantar (4.1M pairs) + Dakshina, Unicode-normalized, deduped, leak-free split by native word (verified zero train/val/test overlap)
- [x] BiLSTM + Luong attention model, training loop, greedy + beam search decoding
- [x] Small Transformer (encoder-decoder, warmup LR schedule, label smoothing)
- [ ] Full training runs on real GPU (in progress)
- [ ] Evaluation vs. IndicXlit across 3 test sets (Aksharantar, Dakshina lexicon, Dakshina sentences)
- [ ] Error analysis
- [ ] ONNX export + browser demo

## Repo layout

```
src/
  data/         download.py, preprocess.py, split.py, vocab.py
  models/       lstm_attention.py, transformer.py
  training/     train.py, train_transformer.py, dataset.py
  inference/    decode.py (greedy + beam search)
  evaluation/   metrics, IndicXlit benchmark, error analysis (WIP)
notebooks/      01_data_stats.ipynb -- pair counts, length distributions, leak check
demo/           ONNX + browser demo (WIP)
```

## Data

- **[Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)** (AI4Bharat) — primary training data, word-level romanized/native pairs.
- **[Dakshina](https://github.com/google-research-datasets/dakshina)** (Google) — held-out test sets only: a lexicon with multiple attested romanizations per word, and full romanized Wikipedia sentences (harder, more realistic).

Splits are grouped by native Malayalam word before assignment to train/val/test, so no word's romanization variants leak across splits — see `src/data/split.py` and the overlap assertion in `notebooks/01_data_stats.ipynb`.

## Reproducing

```bash
python -m venv venv
source venv/Scripts/activate  # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python src/data/download.py
python src/data/preprocess.py
python src/data/split.py
python src/data/vocab.py

python src/training/train.py --epochs 15
python src/training/train_transformer.py --epochs 15
```
