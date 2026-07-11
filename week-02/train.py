"""
Minimal training loop for the from-scratch ResNet on CIFAR-10.

Usage:
    python train.py --model resnet18 --epochs 20 --batch-size 128

Note: the original ResNet stem (7x7 conv, stride 2 + maxpool) is tuned
for 224x224 ImageNet images. For CIFAR-10's 32x32 images we swap in a
lighter stem (3x3 conv, no maxpool) -- this is the standard adjustment
used in most "ResNet for CIFAR" implementations, otherwise the image
gets downsampled to ~1x1 before it even reaches the residual stages.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import ResNet, BasicBlock, Bottleneck


def build_cifar_resnet(variant: str, num_classes: int = 10) -> ResNet:
    configs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs)}")

    block, layers = configs[variant]
    model = ResNet(block, layers, num_classes=num_classes)

    # Swap the ImageNet-style stem for a CIFAR-friendly one.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_dataloaders(batch_size: int, data_dir: str = "./data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_cifar_resnet(args.model, num_classes=10).to(device)
    train_loader, test_loader = get_dataloaders(args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - start

        print(f"Epoch [{epoch+1}/{args.epochs}]  "
              f"train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"time={elapsed:.1f}s")

    torch.save(model.state_dict(), f"{args.model}_cifar10.pth")
    print(f"Saved weights to {args.model}_cifar10.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet18",
                         choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    train(args)
