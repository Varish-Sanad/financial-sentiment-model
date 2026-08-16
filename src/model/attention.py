import torch.nn as nn
import torch.nn.functional as F

from src.model.rope import apply_rope, precompute_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # is_causal=True applies the look-ahead mask inside the fused kernel
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
