"""
ResNet implemented from first principles in PyTorch.

Covers the two residual block types from the original paper
(He et al., 2015, "Deep Residual Learning for Image Recognition"):

    - BasicBlock    -> used in ResNet-18 / ResNet-34
    - Bottleneck    -> used in ResNet-50 / ResNet-101 / ResNet-152

The key idea in both blocks is the same: instead of learning a direct
mapping H(x), the block learns a residual F(x) = H(x) - x, and the
output is F(x) + x (the "skip connection"). This makes it much easier
to optimize very deep networks, since gradients can flow through the
identity shortcut without vanishing.
"""

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding=1, no bias (bias is redundant before BatchNorm)."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, typically used to change channel dimensions or downsample."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Two stacked 3x3 convolutions with a skip connection.
    Used in the shallower ResNets (18, 34).

    Input  -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+= identity) -> ReLU -> Output
    """
    expansion = 1  # output channels == planes (no channel expansion)

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # If input/output shapes don't match (channels or spatial size),
        # the identity path needs a projection to match dimensions.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity          # the residual / skip connection
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Three stacked convolutions (1x1 -> 3x3 -> 1x1) with a skip connection.
    Used in the deeper ResNets (50, 101, 152).

    The 1x1 convs squeeze then expand channels, so the expensive 3x3
    conv operates on a smaller number of channels -> much cheaper
    computationally than stacking plain 3x3 convs at full width.

    Input -> conv1x1 -> BN -> ReLU -> conv3x3 -> BN -> ReLU -> conv1x1 -> BN
          -> (+= identity) -> ReLU -> Output
    """
    expansion = 4  # output channels == planes * 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Generic ResNet backbone. Instantiate via the resnet18/34/50/101/152
    helper functions below rather than calling this directly.
    """

    def __init__(self, block, layers: list, num_classes: int = 1000, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64

        # Stem: 7x7 conv + maxpool, aggressively downsamples the input
        # before the residual stages (this is what the original paper uses
        # for ImageNet-sized 224x224 inputs).
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual stages, each halving spatial resolution
        # (except the first) and doubling channel width.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        # A projection shortcut is needed whenever the stride != 1
        # (spatial size changes) or the channel count changes.
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # He (Kaiming) initialization -- standard for ReLU networks.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Standard ResNet configurations
# ---------------------------------------------------------------------------

def resnet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def resnet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def resnet152(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through each variant and
    # print output shape + parameter count.
    dummy = torch.randn(2, 3, 224, 224)

    for name, fn in [
        ("resnet18", resnet18), ("resnet34", resnet34),
        ("resnet50", resnet50), ("resnet101", resnet101),
        ("resnet152", resnet152),
    ]:
        model = fn(num_classes=10)
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s} output shape: {tuple(out.shape)}  |  params: {n_params:,}")
