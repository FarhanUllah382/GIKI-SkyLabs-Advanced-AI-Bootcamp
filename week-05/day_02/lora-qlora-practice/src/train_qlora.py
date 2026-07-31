"""
QLoRA fine-tuning — base model loaded in 4-bit via bitsandbytes, then
LoRA adapters trained on top. Compare this file line-by-line against
train_lora.py — the diffs ARE the QLoRA lesson.

Key differences from plain LoRA:
    1. BitsAndBytesConfig loads the base model in 4-bit.
    2. prepare_model_for_kbit_training() is required before adding LoRA —
       it casts norms to fp32 and enables gradient checkpointing-friendly
       settings so training a quantized model is numerically stable.
    3. optimizer is usually "paged_adamw_8bit" to save extra memory.

Run:
    python src/train_qlora.py --config config/qlora_config.yaml
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from src.utils import load_config, print_trainable_parameters, load_formatted_dataset, get_gpu_memory_summary


def main(config_path: str):
    config = load_config(config_path)

    # ---- 1. Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Load base model in 4-bit ----
    q_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        quantization_config=bnb_config,
        device_map=config["model"]["device_map"],
    )

    # ---- 3. Prepare quantized model for training (QLoRA-specific step) ----
    model = prepare_model_for_kbit_training(model)

    # ---- 4. Wrap with LoRA adapters ----
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

    # ---- 5. Load data ----
    dataset = load_formatted_dataset(config)
    print(f"\nTraining on {len(dataset)} examples")

    # ---- 6. Training arguments ----
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

    # ---- 7. Train ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["data"]["max_seq_length"],
    )

    trainer.train()

    # ---- 8. Save adapter only ----
    model.save_pretrained(f"{t_cfg['output_dir']}/final_adapter")
    tokenizer.save_pretrained(f"{t_cfg['output_dir']}/final_adapter")

    print("\n" + get_gpu_memory_summary())
    print(f"\nAdapter saved to {t_cfg['output_dir']}/final_adapter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/qlora_config.yaml")
    args = parser.parse_args()
    main(args.config)
