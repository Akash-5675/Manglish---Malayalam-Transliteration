"""
Training loop for the small Transformer.

Usage (quick local smoke test):
    python src/training/train_transformer.py --max-train-samples 2000 --max-val-samples 500 --epochs 2

Usage (real run, e.g. on Kaggle):
    python src/training/train_transformer.py --epochs 15 --batch-size 256
"""
import argparse
import csv
import itertools
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab, PAD_IDX
from models.transformer import Seq2SeqTransformer, count_parameters
from training.dataset import make_dataloader
from inference.decode import greedy_decode_transformer
from utils import set_seed

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=str(PROC_DIR / "train.jsonl"))
    p.add_argument("--val", default=str(PROC_DIR / "val.jsonl"))
    p.add_argument("--src-vocab", default=str(PROC_DIR / "vocab_src.json"))
    p.add_argument("--tgt-vocab", default=str(PROC_DIR / "vocab_tgt.json"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-subset", type=int, default=3000)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--run-name", default="transformer")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def subsample_jsonl(path, n):
    if n is None:
        return path
    out_path = Path(path).with_name(Path(path).stem + f"_head{n}.jsonl")
    with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in itertools.islice(fin, n):
            fout.write(line)
    return str(out_path)


class NoamScheduler:
    """LR = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)

    Ramps LR up linearly for `warmup_steps`, then decays proportional to
    the inverse square root of the step count. Without this, transformers
    at this scale are prone to diverging early in training.
    """

    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def compute_loss(logits, tgt, criterion):
    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, _, tgt, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = compute_loss(logits, tgt, criterion)
            n_tokens = (tgt[:, 1:] != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def word_accuracy(model, loader, tgt_vocab, device, max_examples):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, _, tgt, _ in loader:
            src = src.to(device)
            preds = greedy_decode_transformer(model, src)
            for i, pred_ids in enumerate(preds):
                pred_str = tgt_vocab.decode(pred_ids, strip_specials=False)
                gold_str = tgt_vocab.decode(tgt[i].tolist(), strip_specials=True)
                if pred_str == gold_str:
                    correct += 1
                total += 1
            if total >= max_examples:
                return correct / total
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")

    train_path = subsample_jsonl(args.train, args.max_train_samples)
    val_path = subsample_jsonl(args.val, args.max_val_samples)

    src_vocab = Vocab.load(args.src_vocab)
    tgt_vocab = Vocab.load(args.tgt_vocab)
    print(f"src vocab={len(src_vocab)} tgt vocab={len(tgt_vocab)}")

    train_loader = make_dataloader(train_path, src_vocab, tgt_vocab, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_path, src_vocab, tgt_vocab, args.batch_size, shuffle=False)
    val_acc_loader = make_dataloader(val_path, src_vocab, tgt_vocab, batch_size=64, shuffle=False)

    model = Seq2SeqTransformer(
        len(src_vocab), len(tgt_vocab), d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim, dropout=args.dropout,
    ).to(device)
    print(f"model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)

    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{args.run_name}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_word_acc", "lr", "seconds"])

    best_acc, epochs_no_improve = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_tokens = 0.0, 0

        for step, (src, _, tgt, _) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = compute_loss(logits, tgt, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = scheduler.step()
            optimizer.step()

            n_tokens = (tgt[:, 1:] != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.3f} lr={lr:.2e}")

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        val_acc = word_accuracy(model, val_acc_loader, tgt_vocab, device, args.val_subset)
        elapsed = time.time() - t0

        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_word_acc={val_acc:.4f} lr={lr:.2e} ({elapsed:.0f}s)")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, lr, elapsed])

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch, "val_acc": val_acc},
                CKPT_DIR / f"{args.run_name}_best.pt",
            )
            print(f"  -> new best (val_word_acc={val_acc:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    print(f"done. best val_word_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
