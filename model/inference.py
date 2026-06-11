from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import build_prompt

DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "bug-localizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned bug localization model")
    parser.add_argument("--adapter_dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--error", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
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


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model, args.adapter_dir)
    prompt = build_prompt(args.code, args.error)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output[len(prompt):].strip() if output.startswith(prompt) else output.strip())


if __name__ == "__main__":
    main()
