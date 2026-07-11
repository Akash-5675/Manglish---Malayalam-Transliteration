"""
BiLSTM encoder + Luong-attention decoder for char-level transliteration.

Encoder: embedding(128) -> 2-layer BiLSTM(hidden=256/direction) -> 512-dim outputs
Attention: Luong "general" (learned bilinear score between decoder state and
           each encoder position, masked at padding, softmax -> context vector)
Decoder: embedding(128) -> 2-layer LSTM(hidden=512), fed input + previous
         attention context (input feeding), context concatenated with the
         LSTM output and projected to vocab logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # outputs: (batch, src_len, hidden_dim*2)
        return outputs, hidden, cell


class Bridge(nn.Module):
    """Projects the encoder's final bidirectional states into the decoder's
    initial (unidirectional) hidden/cell states, per layer."""

    def __init__(self, enc_hidden_dim, dec_hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.linear_h = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.linear_c = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

    def forward(self, hidden, cell):
        # hidden/cell from encoder: (num_layers*2, batch, enc_hidden_dim)
        # reshape to (num_layers, 2, batch, enc_hidden_dim), concat directions
        def merge(x):
            x = x.view(self.num_layers, 2, x.size(1), x.size(2))
            x = torch.cat([x[:, 0], x[:, 1]], dim=-1)  # (num_layers, batch, enc_hidden_dim*2)
            return x

        h = torch.tanh(self.linear_h(merge(hidden)))
        c = torch.tanh(self.linear_c(merge(cell)))
        return h.contiguous(), c.contiguous()


class LuongAttention(nn.Module):
    """General (bilinear) attention: score = dec_hidden^T W enc_output."""

    def __init__(self, dec_hidden_dim, enc_output_dim):
        super().__init__()
        self.W = nn.Linear(enc_output_dim, dec_hidden_dim, bias=False)

    def forward(self, dec_hidden, enc_outputs, src_mask):
        # dec_hidden: (batch, dec_hidden_dim) -- top-layer decoder hidden at this step
        # enc_outputs: (batch, src_len, enc_output_dim)
        # src_mask: (batch, src_len) bool, True at valid (non-pad) positions
        proj = self.W(enc_outputs)  # (batch, src_len, dec_hidden_dim)
        scores = torch.bmm(proj, dec_hidden.unsqueeze(2)).squeeze(2)  # (batch, src_len)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_output_dim)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=512, enc_output_dim=512,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        # input feeding: concat token embedding with previous context vector
        self.lstm = nn.LSTM(
            emb_dim + enc_output_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = LuongAttention(hidden_dim, enc_output_dim)
        self.out_proj = nn.Linear(hidden_dim + enc_output_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(self, input_token, prev_context, hidden, cell, enc_outputs, src_mask):
        # input_token: (batch,) int64 ; prev_context: (batch, enc_output_dim)
        embedded = self.dropout(self.embedding(input_token))  # (batch, emb_dim)
        lstm_input = torch.cat([embedded, prev_context], dim=1).unsqueeze(1)  # (batch, 1, emb+enc)
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        top_hidden = lstm_out.squeeze(1)  # (batch, hidden_dim) -- top layer output

        context, attn_weights = self.attention(top_hidden, enc_outputs, src_mask)
        combined = torch.cat([top_hidden, context], dim=1)
        combined = torch.tanh(self.out_proj(self.dropout(combined)))
        logits = self.classifier(combined)  # (batch, vocab_size)
        return logits, context, hidden, cell, attn_weights


class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=128,
                 enc_hidden_dim=256, dec_hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, enc_hidden_dim, num_layers, dropout)
        self.bridge = Bridge(enc_hidden_dim, dec_hidden_dim, num_layers)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, dec_hidden_dim, enc_hidden_dim * 2, num_layers, dropout)
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_output_dim = enc_hidden_dim * 2

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=1.0):
        # src: (batch, src_len), tgt: (batch, tgt_len) including <sos>...<eos>
        batch_size, tgt_len = tgt.size()
        device = src.device

        enc_outputs, enc_hidden, enc_cell = self.encoder(src, src_lens)
        src_mask = (src != PAD_IDX)  # (batch, src_len)
        hidden, cell = self.bridge(enc_hidden, enc_cell)

        input_token = tgt[:, 0]  # <sos>
        context = torch.zeros(batch_size, self.enc_output_dim, device=device)
        logits_list = []

        for t in range(1, tgt_len):
            logits, context, hidden, cell, _ = self.decoder.step(
                input_token, context, hidden, cell, enc_outputs, src_mask
            )
            logits_list.append(logits)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if use_teacher_forcing else logits.argmax(dim=1)

        return torch.stack(logits_list, dim=1)  # (batch, tgt_len-1, vocab_size)

    def encode(self, src, src_lens):
        enc_outputs, enc_hidden, enc_cell = self.encoder(src, src_lens)
        src_mask = (src != PAD_IDX)
        hidden, cell = self.bridge(enc_hidden, enc_cell)
        return enc_outputs, src_mask, hidden, cell


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
