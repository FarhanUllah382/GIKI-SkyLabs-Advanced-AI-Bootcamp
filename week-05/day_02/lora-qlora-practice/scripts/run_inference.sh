#!/usr/bin/env bash
set -e

echo "=== Running inference with fine-tuned adapter ==="
python -m src.inference \
    --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --adapter_path "outputs/lora-run/final_adapter" \
    --prompt "Explain the difference between LoRA and QLoRA in two sentences."
