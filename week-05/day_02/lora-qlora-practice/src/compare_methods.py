"""
Runs a lightweight comparison across three fine-tuning strategies on the
SAME tiny model + dataset, and reports trainable params + peak VRAM for
each. This is the exercise that builds real intuition about why QLoRA
exists: run this on a model that barely fits your GPU in full precision
and watch full fine-tuning fail while QLoRA succeeds.

This does NOT run full training loops (that would take too long for a
comparison script) — it does ONE forward+backward step per method and
measures memory, which is enough to see the pattern clearly.

Run:
    python src/compare_methods.py --model gpt2
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.utils import print_trainable_parameters


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def one_training_step(model, tokenizer, device):
    """Run a single forward+backward pass to trigger realistic memory usage."""
    text = "The quick brown fox jumps over the lazy dog. " * 20
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()
    return outputs.loss.item()


def run_full_finetune(model_name: str, device: str):
    print("\n" + "=" * 60)
    print("METHOD 1: Full fine-tuning (all params trainable)")
    print("=" * 60)
    reset_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    print_trainable_parameters(model)  # 100% trainable
    loss = one_training_step(model, tokenizer, device)
    print(f"Loss after 1 step: {loss:.4f}")
    print(f"Peak memory: {peak_memory_gb():.2f} GB")
    del model
    reset_memory_stats()


def run_lora(model_name: str, device: str):
    print("\n" + "=" * 60)
    print("METHOD 2: LoRA (bf16 base model + adapters)")
    print("=" * 60)
    reset_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    loss = one_training_step(model, tokenizer, device)
    print(f"Loss after 1 step: {loss:.4f}")
    print(f"Peak memory: {peak_memory_gb():.2f} GB")
    del model
    reset_memory_stats()


def run_qlora(model_name: str, device: str):
    print("\n" + "=" * 60)
    print("METHOD 3: QLoRA (4-bit base model + adapters)")
    print("=" * 60)
    reset_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    loss = one_training_step(model, tokenizer, device)
    print(f"Loss after 1 step: {loss:.4f}")
    print(f"Peak memory: {peak_memory_gb():.2f} GB")
    del model
    reset_memory_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2",
                         help="Use a small model here — this script runs all 3 methods in sequence")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comparison on device: {device}")

    run_full_finetune(args.model, device)
    run_lora(args.model, device)
    if device == "cuda":
        run_qlora(args.model, device)  # bitsandbytes 4-bit requires CUDA
    else:
        print("\nSkipping QLoRA comparison — 4-bit quantization requires a CUDA GPU.")

    print("\n" + "=" * 60)
    print("Compare the 'Peak memory' and 'trainable%' numbers above.")
    print("That gap IS the reason QLoRA exists.")
    print("=" * 60)
