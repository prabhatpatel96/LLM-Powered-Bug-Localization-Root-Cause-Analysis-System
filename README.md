
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
python data/synthetic/generate_bugs.py
python model/fine_tune.py
```
