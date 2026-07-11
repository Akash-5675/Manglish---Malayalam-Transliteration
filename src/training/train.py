"""
Training loop for the BiLSTM+attention model.

Usage (quick local smoke test on a tiny slice, CPU or GPU):
    python src/training/train.py --max-train-samples 2000 --max-val-samples 500 --epochs 2

Usage (real run, e.g. on Kaggle):
    python src/training/train.py --epochs 15 --batch-size 256
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab, PAD_IDX
from models.lstm_attention import Seq2SeqLSTM, count_parameters
from training.dataset import make_dataloader
from inference.decode import greedy_decode

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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--enc-hidden", type=int, default=256)
    p.add_argument("--dec-hidden", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--tf-start", type=float, default=1.0, help="teacher forcing ratio at epoch 1")
    p.add_argument("--tf-end", type=float, default=0.7, help="teacher forcing ratio at final epoch")
    p.add_argument("--patience", type=int, default=4, help="early stopping patience (epochs, on val word acc)")
    p.add_argument("--val-subset", type=int, default=3000, help="cap on examples used for greedy val-accuracy check (speed)")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--run-name", default="lstm_attn")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def subsample_jsonl(path, n):
    """Write a truncated copy of a jsonl file for fast smoke tests; returns new path."""
    if n is None:
        return path
    out_path = Path(path).with_name(Path(path).stem + f"_head{n}.jsonl")
    with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in itertools.islice(fin, n):
            fout.write(line)
    return str(out_path)


def compute_loss(logits, tgt, criterion):
    # logits: (batch, tgt_len-1, vocab) ; tgt: (batch, tgt_len) -- predict tgt[:,1:]
    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, src_lens, tgt, _ in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
            loss = compute_loss(logits, tgt, criterion)
            n_tokens = (tgt[:, 1:] != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def word_accuracy(model, loader, tgt_vocab, device, max_examples):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, src_lens, tgt, _ in loader:
            src = src.to(device)
            preds = greedy_decode(model, src, src_lens)
            for i, pred_ids in enumerate(preds):
                pred_str = tgt_vocab.decode(pred_ids, strip_specials=False)
                gold_ids = tgt[i].tolist()
                gold_str = tgt_vocab.decode(gold_ids, strip_specials=True)
                if pred_str == gold_str:
                    correct += 1
                total += 1
            if total >= max_examples:
                return correct / total
    return correct / max(total, 1)


def main():
    args = parse_args()
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

    model = Seq2SeqLSTM(
        len(src_vocab), len(tgt_vocab), emb_dim=args.emb_dim,
        enc_hidden_dim=args.enc_hidden, dec_hidden_dim=args.dec_hidden,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    print(f"model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    CKPT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{args.run_name}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_word_acc", "lr", "seconds"])

    best_acc, epochs_no_improve = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * (epoch - 1) / max(args.epochs - 1, 1)

        total_loss, total_tokens = 0.0, 0
        for step, (src, src_lens, tgt, _) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, src_lens, tgt, teacher_forcing_ratio=tf_ratio)
            loss = compute_loss(logits, tgt, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tokens = (tgt[:, 1:] != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.3f} tf={tf_ratio:.2f}")

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        val_acc = word_accuracy(model, val_acc_loader, tgt_vocab, device, args.val_subset)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
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
