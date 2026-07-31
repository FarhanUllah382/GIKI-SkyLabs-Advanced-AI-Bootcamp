#!/usr/bin/env bash
set -e

echo "=== Evaluating Vision Transformer checkpoint ==="
python -m src.evaluate \
    --config config/vit_config.yaml \
    --checkpoint outputs/vit-run/best_model.pt
