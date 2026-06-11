
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
OUTPUT_FILE = PROCESSED_DIR / "train.jsonl"

TEMPLATES = [
    {
        "code": """def sum_list(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total""",
        "buggy_lines": [3],
        "error": "IndexError: list index out of range",
        "root_cause": "Off-by-one error in loop",
        "fix": "Change range(len(arr)) to range(len(arr) - 1)",
    },
    {
        "code": """def safe_divide(a, b):
    return a / b""",
        "buggy_lines": [2],
        "error": "ZeroDivisionError: division by zero",
        "root_cause": "Missing guard against zero divisor",
        "fix": "Add a check for b == 0 before dividing",
    },
    {
        "code": """def get_user_name(user):
    if user is None:
        return None
    return user['name']""",
        "buggy_lines": [4],
        "error": "KeyError: 'name'",
        "root_cause": "Assumes the name key always exists",
        "fix": "Use user.get('name') or validate the key before accessing it",
    },
    {
        "code": """def append_item(items, value):
    for item in items:
        items.append(item)
    return items""",
        "buggy_lines": [3],
        "error": "RuntimeError: list changed size during iteration",
        "root_cause": "Modifying a list while iterating over it",
        "fix": "Append to a new list or iterate over a copy",
    },
]

def generate():
    template = random.choice(TEMPLATES)
    return {
        "instruction": "Identify buggy lines, root cause, and fix",
        "code": template["code"],
        "error": template["error"],
        "output": {
            "buggy_lines": template["buggy_lines"],
            "root_cause": template["root_cause"],
            "fix": template["fix"],
        }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000, help="Number of synthetic samples to write")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for _ in range(args.count):
            f.write(json.dumps(generate()) + "\n")

    print(f"Dataset generated at {OUTPUT_FILE.as_posix()}")


if __name__ == "__main__":
    main()
