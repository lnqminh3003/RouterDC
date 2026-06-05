import json
import random

input_path = "train.json"

with open(input_path) as f:
    data = json.load(f)

original_order = list(data[0]["scores"].keys())
print(f"Original model order: {original_order}\n")


def apply_order(data, order):
    """Rewrite every sample's scores dict using the given model order."""
    result = []
    for entry in data:
        reordered = {model: entry["scores"][model] for model in order}
        result.append({**entry, "scores": reordered})
    return result


def shuffle_order(original, seed):
    order = original.copy()
    random.seed(seed)
    random.shuffle(order)
    return order


# Shuffle 1
order1 = shuffle_order(original_order, seed=1)
data1 = apply_order(data, order1)
with open("train_shuffle1.json", "w") as f:
    json.dump(data1, f)
print(f"Shuffle 1 (seed=1): {order1}")
print(f"  Saved {len(data1)} samples to train_shuffle1.json\n")

# Shuffle 2
order2 = shuffle_order(original_order, seed=2)
data2 = apply_order(data, order2)
with open("train_shuffle2.json", "w") as f:
    json.dump(data2, f)
print(f"Shuffle 2 (seed=2): {order2}")
print(f"  Saved {len(data2)} samples to train_shuffle2.json\n")

# Verify consistency — all samples in each file must have the same key order
for fname, shuffled_data, expected_order in [
    ("train_shuffle1.json", data1, order1),
    ("train_shuffle2.json", data2, order2),
]:
    for i, entry in enumerate(shuffled_data):
        assert list(entry["scores"].keys()) == expected_order, \
            f"{fname} sample {i} has wrong order!"
print("Consistency check passed: all samples have identical model order within each file.")
