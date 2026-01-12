
import json, random, os

os.makedirs("../processed", exist_ok=True)

TEMPLATE = {
    "code": """def sum_list(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total""",
    "buggy_line": 3,
    "error": "IndexError: list index out of range",
    "fix": "Change range(len(arr)) to range(len(arr)-1)"
}

def generate():
    return {
        "instruction": "Identify buggy lines, root cause, and fix",
        "code": TEMPLATE["code"],
        "error": TEMPLATE["error"],
        "output": {
            "buggy_lines": [TEMPLATE["buggy_line"]],
            "root_cause": "Off-by-one error in loop",
            "fix": TEMPLATE["fix"]
        }
    }

with open("../processed/train.jsonl", "w") as f:
    for _ in range(2000):
        f.write(json.dumps(generate()) + "\n")

print("Dataset generated at data/processed/train.jsonl")
