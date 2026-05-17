from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("colin/PrimeVul", "paired")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

lengths = [len(tokenizer(x["func"])["input_ids"]) for x in ds["test"]]
print(f"Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}")

over_8192 = sum(1 for l in lengths if l > 26653)
print(f"{over_8192}/{len(lengths)} = {over_8192/len(lengths)*100:.1f}%")