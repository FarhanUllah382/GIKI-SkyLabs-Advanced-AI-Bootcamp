"""
Shared helpers used by train_lora.py, train_qlora.py, inference.py, etc.
Keeping these in one place means the training scripts stay short and
readable — that's the whole point of practicing with a real project layout.
"""

import yaml
import torch
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_trainable_parameters(model) -> None:
    """
    Prints the number and percentage of trainable parameters.
    This is the single most useful sanity check when learning LoRA —
    it proves you're training <1% of the model's weights.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    pct = 100 * trainable_params / all_params
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {pct:.4f}"
    )


def get_gpu_memory_summary() -> str:
    """Quick peak-memory readout, useful for comparing LoRA vs QLoRA VRAM usage."""
    if not torch.cuda.is_available():
        return "CUDA not available — running on CPU."
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    return f"Peak GPU memory — allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB"


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_formatted_dataset(config: dict):
    """
    Loads the dataset prepared by data/prepare_dataset.py if it exists on disk,
    otherwise builds it fresh (useful if you skip the explicit prep step).
    """
    from datasets import load_from_disk
    from data.prepare_dataset import build_dataset

    disk_path = Path("./data/formatted_alpaca_subset")
    if disk_path.exists():
        return load_from_disk(str(disk_path))

    return build_dataset(
        dataset_name=config["data"]["dataset_name"],
        subset_size=config["data"]["train_subset_size"],
    )
