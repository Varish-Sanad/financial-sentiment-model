"""Rotary positional embeddings. Rotates Q/K in pairs of dimensions by an angle
proportional to position, so the dot product between a rotated query and key
depends only on their relative offset -- position never gets added to the
residual stream the way a learned positional embedding would.
"""

import torch


def precompute_rope(head_dim, max_seq_len, base=10000.0):
    assert head_dim % 2 == 0, "RoPE needs an even head_dim to form rotation pairs"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # mirrored to match rotate_half pairing
    return emb.cos(), emb.sin()  # each [max_seq_len, head_dim]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
