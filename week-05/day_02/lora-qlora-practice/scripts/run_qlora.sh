#!/usr/bin/env bash
set -e

echo "=== Running QLoRA (4-bit) fine-tuning ==="
python -m src.train_qlora --config config/qlora_config.yaml
