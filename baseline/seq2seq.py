"""LSTM Seq2Seq baseline with Bahdanau attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (batch, dec_dim), enc_outputs: (batch, src_len, enc_dim)
        score = self.v(torch.tanh(
            self.W_enc(enc_outputs) + self.W_dec(dec_hidden).unsqueeze(1)
        )).squeeze(-1)  # (batch, src_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(score, dim=-1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0, bidirectional=True)
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h, c) = self.rnn(embedded)
        # Combine bidirectional hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)
        c = torch.cat([c[-2], c[-1]], dim=1)
        h = torch.tanh(self.fc_h(h))
        c = torch.tanh(self.fc_c(c))
        return outputs, h, c


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, enc_dim, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = BahdanauAttention(enc_dim, hidden_dim)
        self.rnn = nn.LSTM(emb_dim + enc_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + enc_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_token, h, c, enc_outputs, mask=None):
        # tgt_token: (batch, 1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, emb_dim)
        context, attn_weights = self.attention(h, enc_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (h, c) = self.rnn(rnn_input, (h.unsqueeze(0), c.unsqueeze(0)))
        h = h.squeeze(0)
        c = c.squeeze(0)
        prediction = self.fc_out(torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1))
        return prediction, h, c, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=256, hidden_dim=512,
                 n_enc_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.encoder = LSTMEncoder(src_vocab_size, emb_dim, hidden_dim, n_enc_layers, dropout)
        enc_dim = hidden_dim * 2  # bidirectional
        self.decoder = LSTMDecoder(tgt_vocab_size, emb_dim, hidden_dim, enc_dim, 1, dropout)
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt):
        src_mask = (src != self.pad_idx)
        enc_outputs, h, c = self.encoder(src)

        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)

        for t in range(tgt_len):
            tgt_token = tgt[:, t:t+1]
            prediction, h, c, _ = self.decoder(tgt_token, h, c, enc_outputs, src_mask)
            outputs[:, t] = prediction

        return outputs

    def get_config(self):
        return {
            "src_vocab_size": self.encoder.embedding.num_embeddings,
            "tgt_vocab_size": self.tgt_vocab_size,
            "emb_dim": self.encoder.embedding.embedding_dim,
            "hidden_dim": self.encoder.rnn.hidden_size,
            "n_enc_layers": self.encoder.rnn.num_layers,
            "dropout": self.encoder.dropout.p,
            "pad_idx": self.pad_idx,
            "model_type": "seq2seq_lstm",
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            src_vocab_size=config["src_vocab_size"],
            tgt_vocab_size=config["tgt_vocab_size"],
            emb_dim=config.get("emb_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            n_enc_layers=config.get("n_enc_layers", 2),
            dropout=config.get("dropout", 0.3),
            pad_idx=config.get("pad_idx", 0),
        )
