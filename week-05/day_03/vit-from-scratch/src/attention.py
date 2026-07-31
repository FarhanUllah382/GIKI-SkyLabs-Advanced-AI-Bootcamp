"""
Multi-head self-attention, implemented from scratch (no nn.MultiheadAttention).

Run this file directly to see shapes at every step:
    python -m src.attention
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    Steps:
        1. Project input into Q, K, V (one combined linear layer, then split).
        2. Reshape into (B, num_heads, N, head_dim) so each head attends independently.
        3. Compute attention scores: (Q @ K^T) / sqrt(head_dim)
        4. Softmax over the last dim -> attention weights.
        5. Weighted sum of V using those weights.
        6. Concatenate heads back together and project through one more linear layer.
    """

    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5   # 1/sqrt(head_dim) — keeps softmax gradients stable

        # One linear layer producing Q, K, V all at once (more efficient than three separate ones)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, D = x.shape   # batch, sequence length, d_model

        # ---- 1. Project to Q, K, V and split heads ----
        qkv = self.qkv(x)                                          # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)   # (B, N, 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                            # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # each: (B, heads, N, head_dim)

        # ---- 2. Scaled dot-product attention ----
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale        # (B, heads, N, N)
        attn_weights = attn_scores.softmax(dim=-1)                  # (B, heads, N, N)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- 3. Weighted sum of values ----
        out = attn_weights @ v                                      # (B, heads, N, head_dim)

        # ---- 4. Merge heads back together ----
        out = out.transpose(1, 2).reshape(B, N, D)                  # (B, N, D)

        # ---- 5. Final output projection ----
        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attn:
            return out, attn_weights
        return out


if __name__ == "__main__":
    x = torch.randn(2, 65, 192)   # (batch=2, seq_len=65 [64 patches + CLS], d_model=192)

    mhsa = MultiHeadSelfAttention(d_model=192, num_heads=3)
    out, attn = mhsa(x, return_attn=True)

    print(f"Input:            {tuple(x.shape)}")
    print(f"Output:           {tuple(out.shape)}  (should match input shape)")
    print(f"Attention weights:{tuple(attn.shape)}  (batch, heads, seq_len, seq_len)")
    print(f"Each attention row sums to 1 (softmax): {attn[0, 0, 0].sum().item():.4f}")
