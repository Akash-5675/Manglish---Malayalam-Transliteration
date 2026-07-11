"""
PyTorch Dataset + length-bucketed batch sampler for the transliteration pairs.

Bucketing: sort by source length, chop into large chunks, shuffle batch order
and within-chunk order each epoch. This keeps batches length-similar (less
wasted padding/compute) while still giving each epoch a different batch mix.
"""
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.vocab import Vocab, PAD_IDX


class TransliterationDataset(Dataset):
    def __init__(self, jsonl_path, src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.pairs.append((row["roman"], row["native"]))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        roman, native = self.pairs[idx]
        # src has no <sos>/<eos> -- attention reads the whole thing at once,
        # it doesn't need boundary markers the way the autoregressive
        # decoder does.
        src_ids = torch.tensor(self.src_vocab.encode(roman, add_sos_eos=False), dtype=torch.long)
        tgt_ids = torch.tensor(self.tgt_vocab.encode(native, add_sos_eos=True), dtype=torch.long)
        return src_ids, tgt_ids


class BucketedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, chunk_mult=50, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.chunk_size = batch_size * chunk_mult
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        batches = []
        for i in range(0, len(indices), self.chunk_size):
            chunk = indices[i:i + self.chunk_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)
    return src_padded, src_lens, tgt_padded, tgt_lens


def make_dataloader(jsonl_path, src_vocab, tgt_vocab, batch_size, shuffle=True, num_workers=0):
    ds = TransliterationDataset(jsonl_path, src_vocab, tgt_vocab)
    lengths = [len(p[0]) for p in ds.pairs]
    sampler = BucketedBatchSampler(lengths, batch_size, shuffle=shuffle)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
