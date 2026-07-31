# Vision Transformer (ViT) — From Scratch

A from-scratch PyTorch implementation of the Vision Transformer
("An Image is Worth 16x16 Words", Dosovitskiy et al., 2020), built to
train on CIFAR-10. Every core component — patch embedding, positional
encoding, multi-head self-attention, the transformer encoder block — is
implemented by hand instead of imported, so you can actually see how an
image becomes a sequence of tokens a transformer can process.

## Folder structure

```
vit-from-scratch/
├── config/
│   └── vit_config.yaml         # model + training hyperparameters
├── src/
│   ├── patch_embedding.py      # image -> patches -> linear projection + [CLS] + pos embed
│   ├── attention.py            # multi-head self-attention from scratch
│   ├── transformer_block.py    # attention + MLP + residuals + layer norm
│   ├── vit.py                  # full model assembly
│   ├── dataset.py              # CIFAR-10 loading + augmentation
│   ├── train.py                # training loop
│   ├── evaluate.py             # accuracy / confusion matrix on test set
│   └── utils.py                # shared helpers
├── scripts/
│   ├── run_train.sh
│   └── run_eval.sh
├── notebooks/
│   └── 01_visualize_attention.ipynb   # visualize attention maps + patch embeddings
└── requirements.txt
```

## How an image becomes a transformer input

This is the part most people gloss over when they just call
`timm.create_model("vit_base_patch16_224")`. Here it's spelled out:

1. **Split into patches** — a `32x32x3` image with patch size 4 becomes
   `(32/4) * (32/4) = 64` patches, each `4x4x3 = 48` values.
2. **Linear projection** — each flattened patch (48 numbers) is projected
   through a learned linear layer into a `d_model`-dim embedding (e.g. 192).
   This is literally a `Conv2d(kernel_size=patch_size, stride=patch_size)`
   — see `patch_embedding.py` for why those are mathematically identical.
3. **Prepend a `[CLS]` token** — a learned embedding vector prepended to
   the patch sequence. Its final-layer representation is what gets used
   for classification.
4. **Add positional embeddings** — transformers have no inherent sense of
   order, so a learned position vector is added to every token (patches
   have no "next patch" relationship the way words have "next word").
5. **Feed through N transformer encoder blocks** — each block is
   `LayerNorm -> Multi-Head Self-Attention -> residual -> LayerNorm -> MLP -> residual`.
6. **Classify from the `[CLS]` token** — after the final block, the
   `[CLS]` token's representation goes through a small MLP head to
   produce class logits.

## Learning order

1. **`src/patch_embedding.py`** — run it standalone, print shapes at every step.
2. **`src/attention.py`** — implement/read multi-head attention, understand Q/K/V splitting.
3. **`src/transformer_block.py`** — see how attention + MLP + residuals combine.
4. **`src/vit.py`** — the full model; trace a tensor's shape from image to logits.
5. **`src/dataset.py` + `src/train.py`** — train on CIFAR-10 (small enough for a laptop GPU, or even patient CPU).
6. **`src/evaluate.py`** — measure accuracy, look at a confusion matrix.
7. **`notebooks/01_visualize_attention.ipynb`** — visualize which patches the `[CLS]` token attends to.

## Quick start

```bash
pip install -r requirements.txt

# Train (downloads CIFAR-10 automatically on first run)
bash scripts/run_train.sh

# Evaluate the trained checkpoint
bash scripts/run_eval.sh
```

## Exercises to actually learn (not just run)

- [ ] Change `patch_size` from 4 to 8. What happens to sequence length and accuracy?
- [ ] Remove the positional embeddings entirely (`use_pos_embed: false`). Does the model still train? How much worse?
- [ ] Swap the learned positional embedding for a fixed sinusoidal one (like the original Transformer paper) — compare results.
- [ ] Print `attn_weights` from one head in one layer for a single image — which patches does `[CLS]` attend to most?
- [ ] Increase `depth` (number of transformer blocks) and watch how much more data/epochs it needs before it stops underfitting.
- [ ] Try mean-pooling all patch tokens instead of using `[CLS]` for classification — does accuracy change?
- [ ] Compare this from-scratch ViT's accuracy against a pretrained `timm` ViT fine-tuned on CIFAR-10 with LoRA (bring back what you learned in the LoRA project!).

## Notes on scale

The default config trains a small ViT (`d_model=192`, `depth=6`, `heads=3`)
on CIFAR-10 (32x32 images) — this is intentionally small so it's practical
to train from scratch without a data center. The original ViT paper's
large models only outperform CNNs when pretrained on hundreds of millions
of images; on small datasets trained from scratch, don't expect it to beat
a well-tuned ResNet. That gap *is* part of the lesson — it's why ViT
pretraining (and later, hybrid/data-efficient variants like DeiT) matters.
