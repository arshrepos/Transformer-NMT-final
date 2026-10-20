"""Full encoder-decoder Transformer for sequence-to-sequence translation."""

import math
import torch
import torch.nn as nn

from model.layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.tok_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, enc_output, src_mask, tgt_mask)
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=8,
                 d_ff=512, n_enc_layers=3, n_dec_layers=3, dropout=0.1,
                 max_len=5000, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_enc_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_dec_layers, dropout, max_len)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx
        self.d_model = d_model

    def make_src_mask(self, src):
        # (batch, 1, 1, src_len) — mask out <pad> positions
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        # Padding mask
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        # Causal mask
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
        return pad_mask & causal_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output, attn_weights = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.fc_out(dec_output)
        return logits, attn_weights

    def get_config(self):
        """Return a dict that fully describes this model's architecture."""
        return {
            "src_vocab_size": self.encoder.tok_emb.num_embeddings,
            "tgt_vocab_size": self.decoder.tok_emb.num_embeddings,
            "d_model": self.d_model,
            "n_heads": self.encoder.layers[0].self_attn.n_heads,
            "d_ff": self.encoder.layers[0].ff.fc1.out_features,
            "n_enc_layers": len(self.encoder.layers),
            "n_dec_layers": len(self.decoder.layers),
            "dropout": self.encoder.layers[0].dropout1.p,
            "pad_idx": self.pad_idx,
        }

    @classmethod
    def from_config(cls, config):
        """Reconstruct a Transformer from a saved config dict."""
        return cls(
            src_vocab_size=config["src_vocab_size"],
            tgt_vocab_size=config["tgt_vocab_size"],
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 8),
            d_ff=config.get("d_ff", 512),
            n_enc_layers=config.get("n_enc_layers", 3),
            n_dec_layers=config.get("n_dec_layers", 3),
            dropout=config.get("dropout", 0.1),
            pad_idx=config.get("pad_idx", 0),
        )
