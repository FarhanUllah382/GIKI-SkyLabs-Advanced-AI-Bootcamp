# ResNet from Scratch (PyTorch)

A from-first-principles implementation of ResNet (He et al., 2015),
built as part of the GIKI-SkyLabs Advanced AI Bootcamp 2026 (CNN /
Computer Vision module).

Unlike `torchvision.models.resnet18(pretrained=True)`, this repo
implements the residual blocks, downsampling logic, and full network
architecture manually — the goal is to actually understand *why*
ResNets work, not just call a library.

## Why ResNets matter

Before ResNet, stacking more layers onto a plain CNN made training
*harder*, not easier — deeper networks suffered from vanishing
gradients and degraded accuracy even on the training set. ResNet's
fix: instead of learning a direct mapping `H(x)`, each block learns a
residual `F(x) = H(x) - x`, and the block outputs `F(x) + x`. The
`+ x` "skip connection" gives gradients a direct path back through
the network, making 50, 101, even 152-layer networks trainable.

## Structure

```
resnet-cnn/
├── model.py          # BasicBlock, Bottleneck, ResNet, resnet18/34/50/101/152
├── train.py           # CIFAR-10 training loop
├── requirements.txt
└── README.md
```

## Model variants

| Model      | Block      | Layers per stage | Params (10-class) |
|------------|------------|-------------------|--------------------|
| ResNet-18  | BasicBlock | [2, 2, 2, 2]       | ~11.2M             |
| ResNet-34  | BasicBlock | [3, 4, 6, 3]       | ~21.3M             |
| ResNet-50  | Bottleneck | [3, 4, 6, 3]       | ~23.5M             |
| ResNet-101 | Bottleneck | [3, 4, 23, 3]      | ~42.5M             |
| ResNet-152 | Bottleneck | [3, 8, 36, 3]      | ~58.2M             |

## Usage

Sanity-check the architecture (prints output shape + param count for
all 5 variants):

```bash
python model.py
```

Train on CIFAR-10:

```bash
pip install -r requirements.txt
python train.py --model resnet18 --epochs 20 --batch-size 128
```

Use it in your own code:

```python
from model import resnet18

model = resnet18(num_classes=10)
```

## Key implementation details

- **BasicBlock** (ResNet-18/34): two 3x3 convs with a skip connection.
- **Bottleneck** (ResNet-50/101/152): 1x1 -> 3x3 -> 1x1 convs, squeezing
  channels before the expensive 3x3 conv and expanding again after —
  much cheaper than stacking full-width 3x3 convs.
- **Projection shortcuts**: when spatial size or channel count changes
  between input and output of a block, the skip connection uses a 1x1
  conv + BatchNorm to match dimensions before the addition.
- **CIFAR stem swap** (`train.py`): the original ImageNet stem (7x7
  conv stride 2 + maxpool) over-downsamples 32x32 CIFAR images, so
  `train.py` swaps in a 3x3 stride-1 conv with no maxpool, which is the
  standard adjustment for CIFAR-scale ResNets.

## Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning
for Image Recognition*. https://arxiv.org/abs/1512.03385
