"""
Small encoder-decoder Transformer for char-level transliteration.

d_model=256, 4 heads, 3 encoder layers, 3 decoder layers, FFN dim=1024,
dropout=0.1, sinusoidal positional encoding, label smoothing applied in the
training loop (not here). ~6-10M params depending on vocab size.
"""
import math

import torch
import torch.nn as nn

PAD_IDX = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024,
                 dropout=0.1, max_len=60):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def _embed_src(self, src):
        return self.dropout(self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model)))

    def _embed_tgt(self, tgt):
        return self.dropout(self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)))

    def forward(self, src, tgt_in):
        # src: (batch, src_len) ; tgt_in: (batch, tgt_len-1) -- target shifted right (starts with <sos>)
        src_key_padding_mask = (src == PAD_IDX)          # (batch, src_len)
        tgt_key_padding_mask = (tgt_in == PAD_IDX)        # (batch, tgt_len-1)
        tgt_len = tgt_in.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)

        src_emb = self._embed_src(src)
        tgt_emb = self._embed_tgt(tgt_in)

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(out)  # (batch, tgt_len-1, vocab_size)

    def encode(self, src):
        src_key_padding_mask = (src == PAD_IDX)
        memory = self.transformer.encoder(self._embed_src(src), src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode_step(self, memory, src_key_padding_mask, tgt_so_far):
        # tgt_so_far: (batch, cur_len) -- full prefix generated so far (incl <sos>)
        tgt_len = tgt_so_far.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_so_far.device)
        tgt_emb = self._embed_tgt(tgt_so_far)
        out = self.transformer.decoder(
            tgt_emb, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_key_padding_mask
        )
        logits = self.generator(out[:, -1, :])  # only need next-token prediction
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
