# LoRA / QLoRA Practice Project

A hands-on project to internalize LoRA and QLoRA fine-tuning using Hugging Face
`transformers`, `peft`, `bitsandbytes`, and `trl`.

## Folder structure

```
lora-qlora-practice/
├── config/
│   ├── lora_config.yaml        # LoRA hyperparameters
│   └── qlora_config.yaml       # QLoRA hyperparameters
├── data/
│   └── prepare_dataset.py      # Downloads/formats a small instruction dataset
├── src/
│   ├── utils.py                 # Shared helpers (config loading, formatting, logging)
│   ├── train_lora.py            # Plain LoRA fine-tuning (fp16/bf16 base model)
│   ├── train_qlora.py           # QLoRA fine-tuning (4-bit base model)
│   ├── inference.py             # Load base model + adapter, run generation
│   ├── merge_adapter.py         # Merge LoRA weights into base model
│   └── compare_methods.py       # Full FT vs LoRA vs QLoRA comparison harness
├── scripts/
│   ├── run_lora.sh
│   ├── run_qlora.sh
│   └── run_inference.sh
├── notebooks/
│   └── 01_inspect_adapter.ipynb # Inspect adapter weights, rank, param counts
└── requirements.txt
```

## Suggested learning order

1. **`data/prepare_dataset.py`** — understand the data format LoRA training expects.
2. **`config/lora_config.yaml`** — read every hyperparameter, know what it does.
3. **`src/train_lora.py`** — run this first (no quantization, simpler to debug).
4. **`src/train_qlora.py`** — same idea + 4-bit quantization. Compare the diffs.
5. **`src/inference.py`** — load your adapter, generate text, verify it changed behavior.
6. **`src/merge_adapter.py`** — bake the adapter into the base model.
7. **`src/compare_methods.py`** — run all three approaches, log VRAM/time/loss side by side.
8. **`notebooks/01_inspect_adapter.ipynb`** — visually inspect what LoRA actually trained.

## Quick start

```bash
pip install -r requirements.txt

# Step 1: prepare data
python data/prepare_dataset.py

# Step 2: train with LoRA (no quantization)
bash scripts/run_lora.sh

# Step 3: train with QLoRA (4-bit quantization)
bash scripts/run_qlora.sh

# Step 4: run inference with your trained adapter
bash scripts/run_inference.sh
```

## Exercises to actually learn (not just run)

- [ ] Change `r` (rank) from 8 → 16 → 64 in `lora_config.yaml`. Plot loss curves. What changes?
- [ ] Change `target_modules` — train only `q_proj`/`v_proj` vs all attention + MLP layers.
- [ ] Compare trainable parameter count between LoRA and QLoRA runs — should be similar.
- [ ] Compare peak VRAM usage between `train_lora.py` and `train_qlora.py`.
- [ ] Merge the adapter, then try to fine-tune a NEW LoRA on top of the merged model.
- [ ] Try `lora_alpha = 2*r` vs `lora_alpha = r` — does the effective learning rate scaling matter?
- [ ] Break something on purpose: set `target_modules` to a layer name that doesn't exist. Read the error.
