import random
import json
import os
from datasets import load_dataset, concatenate_datasets

random.seed(42)

PATCHED_RATIO = 2   # 1:2 (vulnerable:patched)
TRAIN_RATIO   = 0.8

ds = load_dataset('colin/PrimeVul', 'default')

# combine all splits
all_data = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

vulnerable = [all_data[i] for i in range(len(all_data)) if all_data[i]['target'] == 1]
patched    = [all_data[i] for i in range(len(all_data)) if all_data[i]['target'] == 0]

print(f"Total vulnerable: {len(vulnerable)}")
print(f"Total patched pool: {len(patched)}")

n_patched = len(vulnerable) * PATCHED_RATIO
sampled_patched = random.sample(patched, n_patched)

combined = vulnerable + sampled_patched
random.shuffle(combined)

n_train = int(len(combined) * TRAIN_RATIO)
train_data = combined[:n_train]
test_data  = combined[n_train:]

print(f"\nTrain: total={len(train_data)}, vulnerable={sum(r['target'] for r in train_data)}, patched={sum(1-r['target'] for r in train_data)}")
print(f"Test:  total={len(test_data)},  vulnerable={sum(r['target'] for r in test_data)},  patched={sum(1-r['target'] for r in test_data)}")

os.makedirs('./datasets/primevul_1to2', exist_ok=True)

for filename, data in [('train', train_data), ('test', test_data)]:
    path = f'./datasets/primevul_1to2/{filename}.jsonl'
    with open(path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    print(f"Saved {path}")
