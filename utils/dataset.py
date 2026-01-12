
from datasets import load_dataset

def load_bug_dataset(path):
    return load_dataset("json", data_files=path)["train"]
