"""
Plain LoRA fine-tuning — base model loaded in bf16, NOT quantized.

This is the version to run first. Compare it side by side with
train_qlora.py to see exactly what quantization adds/changes.

Run:
    python src/train_lora.py --config config/lora_config.yaml
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from src.utils import load_config, print_trainable_parameters, load_formatted_dataset, get_gpu_memory_summary


def main(config_path: str):
    config = load_config(config_path)

    # ---- 1. Load tokenizer + base model (full precision / bf16, no quantization) ----
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, config["model"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        torch_dtype=dtype,
        device_map=config["model"]["device_map"],
    )

    # ---- 2. Wrap the model with LoRA adapters ----
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    print("\n=== Trainable parameter summary ===")
    print_trainable_parameters(model)

    # ---- 3. Load data ----
    dataset = load_formatted_dataset(config)
    print(f"\nTraining on {len(dataset)} examples")

    # ---- 4. Training arguments ----
    t_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        num_train_epochs=t_cfg["num_train_epochs"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        optim=t_cfg["optim"],
        report_to=t_cfg["report_to"],
    )

    # ---- 5. Train ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["data"]["max_seq_length"],
    )

    trainer.train()

    # ---- 6. Save adapter only (small file, not the full model) ----
    model.save_pretrained(f"{t_cfg['output_dir']}/final_adapter")
    tokenizer.save_pretrained(f"{t_cfg['output_dir']}/final_adapter")

    print("\n" + get_gpu_memory_summary())
    print(f"\nAdapter saved to {t_cfg['output_dir']}/final_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/lora_config.yaml")
    args = parser.parse_args()
    main(args.config)
