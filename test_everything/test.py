"""
1. For each model, show accuracy bucket distribution (0.0 – 1.0) from raw JSONL.
2. Read generated train/test JSON and print per-query scores for all models.
3. Show unique accuracy values present in train and test sets.
"""

import json
import glob
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent.parent

model_list = [
    ["codellama",      "CodeLlama-7b-Instruct-hf",        "CodeLlama-7b-Instruct"],
    ["codellama",      "CodeLlama-13b-Instruct-hf",       "CodeLlama-13b-Instruct"],
    ["deepseek-ai",    "DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek-Coder-V2-Lite-Instruct"],
    ["Qwen",           "Qwen2.5-Coder-14B-Instruct",      "Qwen2.5-Coder-14B-Instruct"],
    ["bigcode",        "starcoder2-15b-instruct-v0.1",    "starcoder2-15b-instruct"],
    ["Virtue-AI-HUB",  "VulnLLM-R-7B",                   "VulnLLM-R-7B"],
]

BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DATASET_DIR = BASE / "datasets/split2_primevul"


def extract_answer(raw: str) -> str | None:
    m = re.search(r"\\boxed\{(Yes|No)", raw, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m = re.search(r"\b(Yes|No)\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None


# ── Section 1: bucket distribution from raw JSONL ────────────────────────────

def section_bucket_distribution():
    print("=" * 90)
    print("SECTION 1 — Accuracy bucket distribution per model (from raw JSONL)")
    print("=" * 90)

    header = f"{'Model':<50}" + "".join(f"  {b:.1f}" for b in BUCKETS) + "   total"
    print(header)
    print("─" * len(header))

    for org, model_name, dir_name in model_list:
        pattern = str(BASE / f"output/primevul_gen_1to2/{dir_name}/*/samples_primevul_gen_1to2_*.jsonl")
        files = glob.glob(pattern)
        if not files:
            print(f"{org}/{model_name:<45}  [no file found]")
            continue

        counts: Counter = Counter()

        with open(files[0]) as f:
            for line in f:
                if not line.strip():
                    continue
                entry  = json.loads(line)
                resps  = entry.get("resps", [[]])[0]
                target = entry["target"].strip()

                answers = [extract_answer(r) for r in resps]
                valid   = [a for a in answers if a is not None]
                correct = sum(1 for a in valid if a == target)
                acc     = round(correct / len(resps), 1) if resps else 0.0

                counts[acc] += 1

        total = sum(counts.values())
        row = f"{org}/{model_name:<45}"
        for b in BUCKETS:
            row += f"  {counts[b]:>5}"
        row += f"   {total:>6,}"
        print(row)


# ── Section 2: per-query scores from generated JSON files ────────────────────

def print_split(name: str, data: list, n: int = 20):
    model_ids = list(data[0]["scores"].keys())

    print(f"\n{'─'*90}")
    print(f"{name}  ({len(data):,} queries total, showing first {n})")
    print(f"{'─'*90}")

    header = f"  {'#':>5}  {'query (first 60 chars)':<62}"
    for mid in model_ids:
        short = mid.split("/")[-1][:10]
        header += f"  {short:>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for idx, item in enumerate(data[:n]):
        preview = item["question"].replace("\n", " ").strip()[:60]
        row = f"  {idx:>5}  {preview:<62}"
        for mid in model_ids:
            score = item["scores"].get(mid, float("nan"))
            row += f"  {score:>10.2f}"
        print(row)


def section_per_query_scores():
    print("\n\n" + "=" * 90)
    print("SECTION 2 — Per-query scores from generated train/test JSON")
    print("=" * 90)

    for split_file in ["primevul_train.json", "primevul_test.json"]:
        path = DATASET_DIR / split_file
        if not path.exists():
            print(f"[skip] {split_file} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        print_split(split_file, data, n=20)


# ── Section 3: unique accuracy values in train and test ──────────────────────

def section_unique_scores():
    print("\n\n" + "=" * 90)
    print("SECTION 3 — Unique accuracy values in train / test sets")
    print("=" * 90)

    for split_file in ["primevul_train.json", "primevul_test.json"]:
        path = DATASET_DIR / split_file
        if not path.exists():
            print(f"[skip] {split_file} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        # unique values globally and per model
        global_values: set = set()
        per_model: dict[str, set] = {}

        for item in data:
            for model_id, score in item["scores"].items():
                val = round(score, 4)
                global_values.add(val)
                per_model.setdefault(model_id, set()).add(val)

        print(f"\n{split_file}  ({len(data):,} queries)")
        print(f"  Global unique values ({len(global_values)}): "
              f"{sorted(global_values)}")
        print(f"  Per model:")
        for model_id, vals in per_model.items():
            short = model_id.split("/")[-1]
            print(f"    {short:<45}  {sorted(vals)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    section_bucket_distribution()
    section_per_query_scores()
    section_unique_scores()
