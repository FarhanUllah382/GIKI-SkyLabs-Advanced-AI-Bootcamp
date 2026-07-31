"""
Evaluate a trained ViT checkpoint on the CIFAR-10 test set: overall
accuracy, per-class accuracy, and a confusion matrix.

Run:
    python -m src.evaluate --config config/vit_config.yaml --checkpoint outputs/vit-run/best_model.pt
"""

import argparse

import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.vit import build_vit_from_config
from src.dataset import build_dataloaders, CIFAR10_CLASSES
from src.utils import load_config, get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_labels = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def main(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device(config["training"]["device"])

    model = build_vit_from_config(config).to(device)
    epoch = load_checkpoint(model, checkpoint_path, device=str(device))
    print(f"Loaded checkpoint from epoch {epoch}")

    _, test_loader = build_dataloaders(config)

    preds, labels = collect_predictions(model, test_loader, device)

    accuracy = (preds == labels).mean()
    print(f"\nOverall test accuracy: {accuracy:.4f}\n")

    print("Per-class report:")
    print(classification_report(labels, preds, target_names=CIFAR10_CLASSES))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"{'':12s}" + "".join(f"{c[:4]:>6s}" for c in CIFAR10_CLASSES))
    for i, row in enumerate(cm):
        print(f"{CIFAR10_CLASSES[i]:12s}" + "".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vit_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
