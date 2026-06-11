
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

DEFAULT_INSTRUCTION = "Identify buggy lines, root cause, and fix."


def load_bug_dataset(path: str | Path):
    dataset_path = str(path)
    return load_dataset("json", data_files=dataset_path)["train"]


def build_prompt(code: str, error: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Code:\n"
        f"{code}\n\n"
        "Error:\n"
        f"{error}\n\n"
        "Response:\n"
    )


def build_completion(output: Dict[str, Any]) -> str:
    buggy_lines = output.get("buggy_lines", [])
    root_cause = output.get("root_cause", "")
    fix = output.get("fix", "")
    return (
        f"Buggy Lines: {buggy_lines}\n"
        f"Root Cause: {root_cause}\n"
        f"Fix: {fix}\n"
    )


def format_bug_example(example: Dict[str, Any], eos_token: str = "") -> Dict[str, str]:
    text = build_prompt(example["code"], example["error"])
    text += build_completion(example["output"])
    if eos_token:
        text += eos_token
    return {"text": text}
