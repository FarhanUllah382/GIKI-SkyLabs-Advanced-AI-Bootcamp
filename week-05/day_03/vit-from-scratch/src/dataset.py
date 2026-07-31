"""
CIFAR-10 dataset loading with standard ViT-friendly augmentation.

Run this file directly to see a batch's shape and view sample images:
    python -m src.dataset
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_transforms(augment: bool = True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if augment else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def build_dataloaders(config: dict):
    d_cfg = config["data"]
    train_transform, test_transform = build_transforms(augment=d_cfg["augment"])

    train_set = datasets.CIFAR10(
        root=d_cfg["data_dir"], train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=d_cfg["data_dir"], train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=d_cfg["batch_size"],
        shuffle=True,
        num_workers=d_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=d_cfg["batch_size"],
        shuffle=False,
        num_workers=d_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, test_loader


def unnormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the normalization for visualization purposes."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return img_tensor * std + mean


if __name__ == "__main__":
    dummy_config = {
        "data": {
            "data_dir": "./data",
            "batch_size": 8,
            "num_workers": 0,
            "augment": True,
        }
    }

    train_loader, test_loader = build_dataloaders(dummy_config)
    images, labels = next(iter(train_loader))

    print(f"Batch of images: {tuple(images.shape)}  (batch, channels, H, W)")
    print(f"Batch of labels: {tuple(labels.shape)}")
    print(f"Sample labels: {[CIFAR10_CLASSES[l] for l in labels[:8]]}")
    print(f"\nTrain set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
