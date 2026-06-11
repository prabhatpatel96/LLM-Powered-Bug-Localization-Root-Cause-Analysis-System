from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import build_prompt, load_bug_dataset
from utils.metrics import hallucination_rate, localization_accuracy

DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "bug-localizer"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned bug localization model")
    parser.add_argument("--adapter_dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    return parser.parse_args()


def load_model(base_model: str, adapter_dir: Path):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def extract_buggy_lines(text: str):
    match = re.search(r"Buggy Lines:\s*(.*)", text)
    if not match:
        return []
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


def extract_fix(text: str) -> str:
    match = re.search(r"Fix:\s*(.*)", text)
    return match.group(1).strip() if match else ""


def generate_answer(model, tokenizer, code: str, error: str, max_new_tokens: int) -> str:
    prompt = build_prompt(code, error)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output[len(prompt):].strip() if output.startswith(prompt) else output.strip()


def main():
    args = parse_args()
    dataset = load_bug_dataset(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model, args.adapter_dir)

    count = min(args.num_samples, len(dataset))
    acc_scores = []
    hall_scores = []

    for sample in dataset.select(range(count)):
        prediction = generate_answer(model, tokenizer, sample["code"], sample["error"], args.max_new_tokens)
        predicted_lines = extract_buggy_lines(prediction)
        actual_lines = sample["output"]["buggy_lines"]
        acc_scores.append(localization_accuracy(predicted_lines, actual_lines))
        hall_scores.append(hallucination_rate(extract_fix(prediction), sample["output"]["fix"]))

    print(f"Samples: {count}")
    print(f"Localization accuracy: {sum(acc_scores) / len(acc_scores):.4f}")
    print(f"Hallucination rate: {sum(hall_scores) / len(hall_scores):.4f}")


if __name__ == "__main__":
    main()
