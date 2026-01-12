
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "checkpoints"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto"
)

def analyze_bug(code, error):
    prompt = f"""Identify buggy lines, root cause, and fix.

Code:
{code}

Error:
{error}
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    code = "def sum_list(arr):\n total=0\n for i in range(len(arr)):\n  total+=arr[i]\n return total"
    error = "IndexError: list index out of range"
    print(analyze_bug(code, error))
