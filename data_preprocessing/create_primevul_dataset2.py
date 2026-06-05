import random
import json
import os
from datasets import load_dataset, concatenate_datasets

random.seed(123)

PATCHED_RATIO = 1

ds = load_dataset('colin/PrimeVul', 'default')

all_data = concatenate_datasets([ds['train'], ds['validation'], ds['test']])

vulnerable = [all_data[i] for i in range(len(all_data)) if all_data[i]['target'] == 1]
patched    = [all_data[i] for i in range(len(all_data)) if all_data[i]['target'] == 0]

print(f"Total vulnerable: {len(vulnerable)}")
print(f"Total patched pool: {len(patched)}")

# load already-used benign functions from all.jsonl and all2.jsonl
used_ids = set()
for fname in ['./datasets/primevul_balance/all.jsonl',
              './datasets/primevul_balance/all2.jsonl']:
    if not os.path.exists(fname):
        print(f"[skip] {fname} not found")
        continue
    with open(fname) as f:
        for line in f:
            row = json.loads(line)
            if row['target'] == 0:
                used_ids.add(row['idx'])

remaining_patched = [r for r in patched if r['idx'] not in used_ids]
print(f"Remaining patched pool (unused): {len(remaining_patched)}")

n_patched = len(vulnerable) * PATCHED_RATIO
sampled_patched = random.sample(remaining_patched, n_patched)

combined = vulnerable + sampled_patched
random.shuffle(combined)

print(f"\nTotal: {len(combined)}, vulnerable={sum(r['target'] for r in combined)}, patched={sum(1-r['target'] for r in combined)}")

# --- overlap check against all existing datasets ---
new_ids = {r['idx'] for r in combined}
print(f"\nOverlap check:")
for fname in ['./datasets/primevul_balance/all.jsonl',
              './datasets/primevul_balance/all2.jsonl']:
    if not os.path.exists(fname):
        continue
    existing_ids = set()
    with open(fname) as f:
        for line in f:
            existing_ids.add(json.loads(line)['idx'])
    overlap = existing_ids & new_ids
    overlap_vuln    = sum(1 for r in combined if r['idx'] in overlap and r['target'] == 1)
    overlap_patched = sum(1 for r in combined if r['idx'] in overlap and r['target'] == 0)
    print(f"  vs {fname}:")
    print(f"    overlapping: {len(overlap)}  ({100*len(overlap)/len(new_ids):.1f}%)  vuln={overlap_vuln}  patched={overlap_patched}")

os.makedirs('./datasets/primevul_balance', exist_ok=True)

path = './datasets/primevul_balance/all3.jsonl'
with open(path, 'w') as f:
    for row in combined:
        f.write(json.dumps(row) + '\n')
print(f"\nSaved {path}")
