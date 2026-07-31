#!/usr/bin/env bash
set -e

echo "=== Running plain LoRA fine-tuning ==="
python -m src.train_lora --config config/lora_config.yaml
