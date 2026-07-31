"""
Patch embedding: turns an image (B, C, H, W) into a sequence of tokens
(B, N+1, d_model) — the input format a transformer encoder expects.

Run this file directly to see the shape of every intermediate step:
    python -m src.patch_embedding
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects
    each one into a `d_model`-dimensional embedding.

    Key insight: splitting into patches + flattening + a linear layer is
    mathematically IDENTICAL to a single Conv2d with kernel_size=patch_size
    and stride=patch_size. We implement it as a Conv2d because it's more
    efficient, but the two are the same operation — proving this to
    yourself is a good exercise (see `__main__` below).
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # A single Conv2d does "split into patches + flatten + linear project" in one op:
        # kernel_size == stride == patch_size means each output pixel corresponds to
        # exactly one non-overlapping patch, and out_channels == d_model gives us the
        # linear projection we want.
        self.projection = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)             # (B, d_model, H/patch, W/patch)
        x = x.flatten(2)                   # (B, d_model, num_patches)
        x = x.transpose(1, 2)              # (B, num_patches, d_model)
        return x


class ViTEmbeddings(nn.Module):
    """
    Full embedding stack: patch embedding + [CLS] token + positional embedding.
    This is the very first thing an image passes through in a ViT.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        dropout: float = 0.1,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_patches = self.patch_embed.num_patches

        # Learned [CLS] token, prepended to every sequence. Its output embedding
        # after all transformer blocks is used for classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned positional embedding, one vector per position (including [CLS]).
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Standard ViT initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, num_patches, d_model)

        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)             # (B, num_patches+1, d_model)

        if self.use_pos_embed:
            x = x + self.pos_embed

        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Sanity-check the shapes at every step, using CIFAR-10-sized inputs.
    dummy_images = torch.randn(4, 3, 32, 32)   # batch of 4 CIFAR-10 images

    patch_embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, d_model=192)
    patches = patch_embed(dummy_images)
    print(f"Input images:        {tuple(dummy_images.shape)}")
    print(f"After patch embed:   {tuple(patches.shape)}  "
          f"(expect batch=4, num_patches={(32//4)**2}, d_model=192)")

    vit_embed = ViTEmbeddings(image_size=32, patch_size=4, in_channels=3, d_model=192)
    tokens = vit_embed(dummy_images)
    print(f"After [CLS]+pos:     {tuple(tokens.shape)}  "
          f"(expect batch=4, seq_len={(32//4)**2 + 1}, d_model=192)")
