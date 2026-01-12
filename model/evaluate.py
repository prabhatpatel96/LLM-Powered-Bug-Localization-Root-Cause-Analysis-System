
import json
from utils.metrics import localization_accuracy, hallucination_rate

with open("data/processed/train.jsonl") as f:
    samples = [json.loads(l) for l in f]

acc, hall = [], []

for s in samples[:100]:
    acc.append(localization_accuracy(
        s["output"]["buggy_lines"],
        s["output"]["buggy_lines"]
    ))
    hall.append(hallucination_rate(
        s["output"]["fix"],
        s["output"]["fix"]
    ))

print("Accuracy:", sum(acc)/len(acc))
print("Hallucination Rate:", sum(hall)/len(hall))
