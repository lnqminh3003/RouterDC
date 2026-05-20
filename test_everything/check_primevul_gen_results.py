import json
import os
import re
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE, "output/primevul_gen_1to2/CodeLlama-7b-Instruct/codellama__CodeLlama-7b-Instruct-hf/samples_primevul_gen_1to2_2026-05-19T19-44-00.595573.jsonl")

with open(path) as f:
    data = [json.loads(line) for line in f]

targets, predicted, scores = [], [], []

for s in data:
    # target is "Yes" or "No" string
    t = 1 if s['target'].strip().lower() == 'yes' else 0

    # resps: [[resp1, resp2, resp3, resp4, resp5]]
    resps = s['resps'][0]
    correct_count = sum(1 for r in resps if re.search(r'(?i)(yes|no)', r) and
                        re.search(r'(?i)(yes|no)', r).group(1).lower() == s['target'].strip().lower())
    score = correct_count / len(resps)

    # predicted from filtered_resps (majority/first)
    filtered = s['filtered_resps'][0].strip().lower() if s['filtered_resps'] else ''
    p = 1 if 'yes' in filtered else 0

    targets.append(t)
    predicted.append(p)
    scores.append(score)

acc = accuracy_score(targets, predicted)
f1  = f1_score(targets, predicted, zero_division=0)
cm  = confusion_matrix(targets, predicted)
total   = len(data)
correct = sum(t == p for t, p in zip(targets, predicted))
avg_score = sum(scores) / len(scores)

print(f"Total samples : {total}")
print(f"Correct       : {correct}")
print(f"Wrong         : {total - correct}")
print(f"Accuracy      : {acc*100:.2f}%")
print(f"F1 Score      : {f1:.4f}")
print(f"Avg score/5   : {avg_score:.4f}  (RouterDC score)")
print()
print("Confusion Matrix:")
print(f"              Predicted No  Predicted Yes")
print(f"Actual No   : {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Yes  : {cm[1][0]:<14} {cm[1][1]}")
