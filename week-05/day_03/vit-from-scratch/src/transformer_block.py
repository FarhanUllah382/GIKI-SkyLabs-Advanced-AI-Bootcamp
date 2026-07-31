"""
A single Transformer encoder block: pre-norm attention + pre-norm MLP,
each wrapped in a residual connection. This is the piece that gets
stacked `depth` times to build the full ViT.

Run this file directly to see shapes:
    python -m src.transformer_block
"""

import torch
import torch.nn as nn

from src.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    """Standard transformer feed-forward block: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block (norm BEFORE attention/MLP, not after —
    this is what ViT and most modern transformers use; it trains more
    stably than the original "post-norm" Transformer paper design).

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


if __name__ == "__main__":
    x = torch.randn(2, 65, 192)

    block = TransformerBlock(d_model=192, num_heads=3, mlp_ratio=4.0)
    out = block(x)

    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}  (unchanged shape — this is what lets us stack N of these)")

    num_params = sum(p.numel() for p in block.parameters())
    print(f"Parameters in one block: {num_params:,}")
