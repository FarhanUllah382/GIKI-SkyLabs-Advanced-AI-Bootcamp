#!/usr/bin/env bash
set -e

echo "=== Training Vision Transformer on CIFAR-10 ==="
python -m src.train --config config/vit_config.yaml
