"""
Merge a trained LoRA adapter into the base model weights and save a
standalone model — no `peft` dependency needed to load it afterward.

This is what you'd do before deploying a fine-tuned model as a normal
Hugging Face checkpoint (e.g. for vLLM serving).

Note: you can merge a LoRA adapter trained via QLoRA too, but the base
model must first be reloaded in full precision (NOT 4-bit) — you cannot
merge float LoRA deltas into 4-bit quantized weights directly.

Run:
    python src/merge_adapter.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_path outputs/lora-run/final_adapter \
        --output_path outputs/merged-model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main(base_model_name: str, adapter_path: str, output_path: str):
    print(f"Loading base model in full precision: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()  # <-- this is the key call to understand

    print(f"Saving merged standalone model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\nDone. You can now load this with plain AutoModelForCausalLM.from_pretrained(),")
    print("no `peft` import required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/merged-model")
    args = parser.parse_args()
    main(args.base_model, args.adapter_path, args.output_path)
