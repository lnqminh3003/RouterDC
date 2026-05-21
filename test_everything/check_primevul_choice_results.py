import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


 
# path = os.path.join(BASE, "output/primevul_choice_1to2/CodeLlama-7b-Instruct/codellama__CodeLlama-7b-Instruct-hf/samples_primevul_choice_1to2_2026-05-19T19-09-21.587827.jsonl")
path = os.path.join(BASE, "output/primevul_choice_1to2/CodeLlama-13b-Instruct/codellama__CodeLlama-13b-Instruct-hf/samples_primevul_choice_1to2_2026-05-20T18-20-27.697534.jsonl")


with open(path) as f:
    data = [json.loads(line) for line in f]

targets   = [int(s['doc']['target']) for s in data]
predicted = [1 if float(s['resps'][1][0][0]) > float(s['resps'][0][0][0]) else 0 for s in data]

acc = accuracy_score(targets, predicted)
f1  = f1_score(targets, predicted)
cm  = confusion_matrix(targets, predicted)

total   = len(data)
correct = sum(1 for t, p in zip(targets, predicted) if t == p)

print(f"Total samples : {total}")
print(f"Correct       : {correct}")
print(f"Wrong         : {total - correct}")
print(f"Accuracy      : {acc*100:.2f}%")
print(f"F1 Score      : {f1:.4f}")
print()
print("Confusion Matrix:")
print(f"              Predicted No  Predicted Yes")
print(f"Actual No   : {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual Yes  : {cm[1][0]:<14} {cm[1][1]}")
