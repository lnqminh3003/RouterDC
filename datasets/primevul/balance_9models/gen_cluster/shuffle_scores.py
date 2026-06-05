import json
import random

train_path = "train.json"
test_path  = "../gen/test.json"

with open(train_path) as f:
    train_data = json.load(f)
with open(test_path) as f:
    test_data = json.load(f)

original_order = list(train_data[0]["scores"].keys())
print(f"Original model order: {original_order}\n")


def apply_order(data, order):
    """Rewrite every sample's scores dict using the given model order."""
    return [{**entry, "scores": {model: entry["scores"][model] for model in order}}
            for entry in data]


def shuffle_order(original, seed):
    order = original.copy()
    random.seed(seed)
    random.shuffle(order)
    return order


def verify(data, expected_order, fname):
    for i, entry in enumerate(data):
        assert list(entry["scores"].keys()) == expected_order, \
            f"{fname} sample {i} has wrong order!"


# Shuffle 1
order1 = shuffle_order(original_order, seed=1)
train1 = apply_order(train_data, order1)
test1  = apply_order(test_data,  order1)
with open("train_shuffle1.json", "w") as f:
    json.dump(train1, f)
with open("test_shuffle1.json", "w") as f:
    json.dump(test1, f)
print(f"Shuffle 1 (seed=1): {order1}")
print(f"  train_shuffle1.json: {len(train1)} samples")
print(f"  test_shuffle1.json:  {len(test1)} samples\n")

# Shuffle 2
order2 = shuffle_order(original_order, seed=2)
train2 = apply_order(train_data, order2)
test2  = apply_order(test_data,  order2)
with open("train_shuffle2.json", "w") as f:
    json.dump(train2, f)
with open("test_shuffle2.json", "w") as f:
    json.dump(test2, f)
print(f"Shuffle 2 (seed=2): {order2}")
print(f"  train_shuffle2.json: {len(train2)} samples")
print(f"  test_shuffle2.json:  {len(test2)} samples\n")

# Verify consistency
for fname, data, order in [
    ("train_shuffle1.json", train1, order1),
    ("test_shuffle1.json",  test1,  order1),
    ("train_shuffle2.json", train2, order2),
    ("test_shuffle2.json",  test2,  order2),
]:
    verify(data, order, fname)
print("Consistency check passed: all samples have identical model order within each file.")
