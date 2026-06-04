import json
import random

input_path = "train.json"
output_path = "train_shuffle_2.json"

with open(input_path, "r") as f:
    data = json.load(f)

for entry in data:
    scores = entry["scores"]
    items = list(scores.items())
    random.shuffle(items)
    entry["scores"] = dict(items)

with open(output_path, "w") as f:
    json.dump(data, f)

print(f"Done. {len(data)} entries written to {output_path}")
