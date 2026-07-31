"""
Full Vision Transformer: embeddings -> N transformer blocks -> classification head.

Run this file directly to see the full shape trace + parameter count:
    python -m src.vit
"""

import torch
import torch.nn as nn

from src.patch_embedding import ViTEmbeddings
from src.transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_pos_embed: bool = True,
    ):
        super().__init__()

        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            use_pos_embed=use_pos_embed,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self.embeddings(x)                    # (B, N+1, d_model)

        attn_maps = [] if return_attn else None
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]                 # (B, d_model) — take only the [CLS] token
        logits = self.head(cls_token_final)        # (B, num_classes)

        if return_attn:
            return logits, attn_maps
        return logits


def build_vit_from_config(config: dict) -> VisionTransformer:
    m = config["model"]
    return VisionTransformer(
        image_size=m["image_size"],
        patch_size=m["patch_size"],
        in_channels=m["in_channels"],
        num_classes=m["num_classes"],
        d_model=m["d_model"],
        depth=m["depth"],
        num_heads=m["num_heads"],
        mlp_ratio=m["mlp_ratio"],
        dropout=m["dropout"],
        attn_dropout=m["attn_dropout"],
        use_pos_embed=m["use_pos_embed"],
    )


if __name__ == "__main__":
    model = VisionTransformer(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=192, depth=6, num_heads=3,
    )

    dummy_images = torch.randn(4, 3, 32, 32)
    logits = model(dummy_images)

    print(f"Input images: {tuple(dummy_images.shape)}")
    print(f"Output logits: {tuple(logits.shape)}  (batch=4, num_classes=10)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Trace with attention maps too
    logits, attn_maps = model(dummy_images, return_attn=True)
    print(f"\nNumber of attention maps (== depth): {len(attn_maps)}")
    print(f"Shape of one attention map: {tuple(attn_maps[0].shape)}  (batch, heads, seq_len, seq_len)")
