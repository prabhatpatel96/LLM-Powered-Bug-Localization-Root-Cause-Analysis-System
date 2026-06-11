
# LLM-Powered Bug Localization & Root Cause Analysis System

This project fine-tunes TinyLlama-1.1B using QLoRA (4-bit) to automatically:
- Localize buggy lines
- Explain root cause
- Suggest fixes for Python bugs

## Features
- Synthetic bug dataset generation
- Instruction fine-tuning with QLoRA
- Bug localization + fix suggestion
- Evaluation metrics for accuracy & hallucination

## Setup
```bash
pip install -r requirements.txt
python data/synthetic/generate_bugs.py --count 2000
python model/fine_tune.py --output_dir checkpoints/bug-localizer
python model/evaluate.py --adapter_dir checkpoints/bug-localizer
```

## Inference
```bash
python model/inference.py --adapter_dir checkpoints/bug-localizer --code "def sum_list(arr):\n    total = 0\n    for i in range(len(arr)):\n        total += arr[i]\n    return total" --error "IndexError: list index out of range"
```
