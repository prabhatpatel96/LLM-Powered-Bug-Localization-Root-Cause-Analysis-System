from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import format_bug_example, load_bug_dataset

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "bug-localizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM for bug localization")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


def pick_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(model_name: str, use_4bit: bool):
    dtype = pick_compute_dtype()
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if use_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    return model


def prepare_datasets(dataset_path: Path, tokenizer, eval_split: float, seed: int):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run data/synthetic/generate_bugs.py first."
        )

    dataset = load_bug_dataset(dataset_path)
    if len(dataset) < 2:
        raise ValueError("Dataset is too small to train on.")

    split = dataset.train_test_split(test_size=eval_split, seed=seed) if eval_split > 0 else {"train": dataset}
    eos_token = tokenizer.eos_token or ""

    train_dataset = split["train"].map(
        lambda example: format_bug_example(example, eos_token=eos_token),
        remove_columns=split["train"].column_names,
    )

    eval_dataset = None
    if eval_split > 0:
        eval_dataset = split["test"].map(
            lambda example: format_bug_example(example, eos_token=eos_token),
            remove_columns=split["test"].column_names,
        )

    return train_dataset, eval_dataset


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.model_name)
    train_dataset, eval_dataset = prepare_datasets(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        eval_split=args.eval_split,
        seed=args.seed,
    )

    model = build_model(args.model_name, use_4bit=not args.no_4bit)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not torch.cuda.is_available():
        print("CUDA is not available; training will run on CPU and may be very slow.")

    if not args.no_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = dict(train_result.metrics)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] is not None:
            metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])

    print("Training complete.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Model artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
