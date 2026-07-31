"""
Training loop for the from-scratch Vision Transformer on CIFAR-10.

Run:
    python -m src.train --config config/vit_config.yaml
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.vit import build_vit_from_config
from src.dataset import build_dataloaders
from src.utils import load_config, ensure_dir, get_device, count_parameters, save_checkpoint, AverageMeter


def build_optimizer_and_scheduler(model, config: dict, steps_per_epoch: int):
    t_cfg = config["training"]

    optimizer = AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    warmup_steps = t_cfg["warmup_epochs"] * steps_per_epoch
    total_steps = t_cfg["epochs"] * steps_per_epoch

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch, log_interval, grad_clip_norm):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

        if step % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.4f} | LR: {current_lr:.6f}"
            )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


def main(config_path: str):
    config = load_config(config_path)
    t_cfg = config["training"]

    device = get_device(t_cfg["device"])
    print(f"Using device: {device}")

    output_dir = ensure_dir(t_cfg["output_dir"])

    model = build_vit_from_config(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    train_loader, test_loader = build_dataloaders(config)

    optimizer, scheduler = build_optimizer_and_scheduler(model, config, steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg["label_smoothing"])

    best_acc = 0.0
    for epoch in range(1, t_cfg["epochs"] + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            epoch, t_cfg["log_interval"], t_cfg["grad_clip_norm"],
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - start
        print(
            f"\n=== Epoch {epoch}/{t_cfg['epochs']} done in {elapsed:.1f}s | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Test loss: {test_loss:.4f} acc: {test_acc:.4f} ===\n"
        )

        if t_cfg["save_every_epoch"]:
            save_checkpoint(model, optimizer, epoch, f"{output_dir}/checkpoint_epoch{epoch}.pt")

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, f"{output_dir}/best_model.pt")
            print(f"New best model saved (test acc: {best_acc:.4f})")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vit_config.yaml")
    args = parser.parse_args()
    main(args.config)
