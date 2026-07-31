"""
Load the base model + your trained LoRA/QLoRA adapter and generate text.
Also prints a before/after comparison so you can SEE that fine-tuning
changed the model's behavior, not just trust that training "ran".

Run:
    python src/inference.py --adapter_path outputs/lora-run/final_adapter \
                             --base_model Qwen/Qwen2.5-1.5B-Instruct \
                             --prompt "Explain LoRA in one sentence."
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main(base_model_name: str, adapter_path: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n"
    )

    print("=== Loading base model (no adapter) ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_output = generate(base_model, tokenizer, formatted_prompt)
    print("\n--- BASE MODEL OUTPUT ---")
    print(base_output)

    print("\n=== Loading fine-tuned model (base + adapter) ===")
    tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    tuned_output = generate(tuned_model, tokenizer, formatted_prompt)
    print("\n--- FINE-TUNED MODEL OUTPUT ---")
    print(tuned_output)

    print("\n=== Diff summary ===")
    if base_output.strip() == tuned_output.strip():
        print("WARNING: outputs are identical — adapter may not have loaded, or training had no effect.")
    else:
        print("Outputs differ — adapter is having an effect on generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Explain LoRA fine-tuning in one sentence.")
    args = parser.parse_args()
    main(args.base_model, args.adapter_path, args.prompt)
