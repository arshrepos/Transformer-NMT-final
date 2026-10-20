"""Scaled dot-product attention and multi-head attention (custom, from scratch)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """Compute scaled dot-product attention.

    Args:
        query: (batch, heads, seq_len_q, d_k)
        key:   (batch, heads, seq_len_k, d_k)
        value: (batch, heads, seq_len_k, d_v)
        mask:  broadcastable to (batch, heads, seq_len_q, seq_len_k)

    Returns:
        output: (batch, heads, seq_len_q, d_v)
        attn_weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention built from scratch."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output, attn_weights
