"""
Downloads a small instruction-tuning dataset and formats it into a single
`text` field the way SFTTrainer expects. This is the file to study if you
want to understand *what data LoRA fine-tuning actually consumes*.

Run:
    python data/prepare_dataset.py
"""

from datasets import load_dataset


ALPACA_PROMPT_TEMPLATE_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_example(example: dict) -> dict:
    """Turn a raw Alpaca-style example into a single formatted text field."""
    if example.get("input", "").strip():
        text = ALPACA_PROMPT_TEMPLATE_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        text = ALPACA_PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    return {"text": text}


def build_dataset(dataset_name: str = "tatsu-lab/alpaca", subset_size: int = 1000):
    raw = load_dataset(dataset_name, split=f"train[:{subset_size}]")
    formatted = raw.map(format_example, remove_columns=raw.column_names)
    return formatted


if __name__ == "__main__":
    ds = build_dataset()
    print(f"Loaded {len(ds)} examples")
    print("\n--- Example formatted training sample ---\n")
    print(ds[0]["text"])

    # Save locally so training scripts can load from disk without re-downloading
    ds.save_to_disk("./data/formatted_alpaca_subset")
    print("\nSaved to ./data/formatted_alpaca_subset")
