"""
Check how many PrimeVul samples exceed max_model_len for choice and gen tasks.
Counts tokens using the actual model tokenizer.

Usage:
    # For Gemma-family models (choice task):
    python check_token_lengths.py --tokenizer google/codegemma-7b-it

    # For Qwen (choice task):
    python check_token_lengths.py --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct

    # For GPT via tiktoken (gen task):
    python check_token_lengths.py --tokenizer gpt-4o-mini

    # Custom thresholds:
    python check_token_lengths.py --tokenizer google/codegemma-7b-it --limits 2048 4096 8192
"""

import argparse
import json
from pathlib import Path
from collections import Counter

DATASET_PATH = "../data_preprocessing/datasets/primevul_balance/all.jsonl"

# Prompt templates matching the YAML tasks
CHOICE_PROMPT = (
    "You are a security expert specializing in C/C++ vulnerability detection. "
    "Your task is to analyze the given function carefully and determine whether it "
    "contains a security vulnerability. Answer Yes if you can identify a vulnerability."
    "\n\nFunction:\n```c\n{func}\n```\n\nIs this function vulnerable?"
)
# For choice task lm_eval appends " No" or " Yes" as the continuation — include the longer one
CHOICE_SUFFIX = " Yes"

GEN_PROMPT = (
    "You are a security expert specializing in C/C++ vulnerability detection. "
    "Your task is to analyze the given function carefully and determine whether it "
    "contains a security vulnerability. Answer Yes if you can identify a vulnerability."
    "\n\nFunction:\n```c\n{func}\n```\n\n"
    "Is this function vulnerable? Respond is only allowed to be with this format "
    "\\boxed{{Yes}} or \\boxed{{No}}. No explanation, no other text. \n\nAnswer:"
)
GEN_TRUNCATE_CHARS = 280000  # current truncation applied in primevul_gen.yaml


def load_dataset(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_tokenizer(name):
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        import tiktoken
        enc = tiktoken.encoding_for_model(name)
        return lambda text: len(enc.encode(text))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, local_files_only=True)
        return lambda text: len(tok.encode(text, add_special_tokens=False))


def count_tokens_choice(sample, tokenizer_fn):
    func = sample["func"]
    full_text = CHOICE_PROMPT.format(func=func) + CHOICE_SUFFIX
    return tokenizer_fn(full_text)


def count_tokens_gen(sample, tokenizer_fn):
    func = sample["func"][:GEN_TRUNCATE_CHARS]  # truncation applied in YAML
    full_text = GEN_PROMPT.format(func=func)
    return tokenizer_fn(full_text)


def report(token_counts, limits, task_name):
    total = len(token_counts)
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  Total samples: {total}")
    print(f"{'='*60}")

    sorted_counts = sorted(token_counts)
    percentiles = [50, 75, 90, 95, 99, 100]
    print("\nToken length percentiles:")
    for p in percentiles:
        idx = min(int(p / 100 * total), total - 1)
        print(f"  p{p:3d}: {sorted_counts[idx]:>8,}")

    print(f"\nSamples exceeding max_model_len thresholds (truncated, not skipped):")
    print(f"  {'Limit':>8}  {'Truncated':>10}  {'% of total':>10}")
    print(f"  {'-'*32}")
    for limit in limits:
        n_over = sum(1 for t in token_counts if t > limit)
        pct = 100 * n_over / total
        print(f"  {limit:>8,}  {n_over:>10,}  {pct:>9.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="google/codegemma-7b-it",
                        help="HuggingFace model name or GPT model name (for tiktoken)")
    parser.add_argument("--limits", nargs="+", type=int, default=[2048, 4096, 8192],
                        help="max_model_len thresholds to check")
    parser.add_argument("--task", choices=["choice", "gen", "both"], default="both",
                        help="Which task prompt format to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N samples (for quick testing)")
    args = parser.parse_args()

    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run from the RouterDC root directory.")
        return

    print(f"Loading dataset from {dataset_path}...")
    samples = load_dataset(dataset_path)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples):,} samples.")

    print(f"Loading tokenizer: {args.tokenizer} ...")
    tokenizer_fn = build_tokenizer(args.tokenizer)

    from tqdm import tqdm

    if args.task in ("choice", "both"):
        print("\nCounting tokens for choice task...")
        choice_counts = [
            count_tokens_choice(s, tokenizer_fn)
            for s in tqdm(samples, unit="sample")
        ]
        report(choice_counts, args.limits, f"primevul_choice  [{args.tokenizer}]")

    if args.task in ("gen", "both"):
        print("\nCounting tokens for gen task (with 280K char truncation)...")
        gen_counts = [
            count_tokens_gen(s, tokenizer_fn)
            for s in tqdm(samples, unit="sample")
        ]
        report(gen_counts, args.limits, f"primevul_gen     [{args.tokenizer}]")


if __name__ == "__main__":
    main()
