import random
import json
import os
from datasets import load_dataset, concatenate_datasets

random.seed(42)

PATCHED_RATIO = 2  # 1:2 (vulnerable:patched)

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

print(f"\nTotal: {len(combined)}, vulnerable={sum(r['target'] for r in combined)}, patched={sum(1-r['target'] for r in combined)}")

os.makedirs('./datasets/primevul_1to2', exist_ok=True)

path = './datasets/primevul_1to2/all.jsonl'
with open(path, 'w') as f:
    for row in combined:
        f.write(json.dumps(row) + '\n')
print(f"Saved {path}")
