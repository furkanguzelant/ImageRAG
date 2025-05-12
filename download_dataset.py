from datasets import load_dataset

dataset = load_dataset("laion/laion400m", split="train[:10%]")

print(dataset[2107])