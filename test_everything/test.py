"""
Estimate total tokens and cost for running gpt-4o-mini on primevul_balance/all.jsonl.

Accounts for:
  - repeats: 5  (each sample is run 5 times, per primevul_gen.yaml)
  - max_gen_toks: 20  (max output tokens per generation)
  - apply_chat_template adds a system turn overhead
"""

import json
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # o200k_base
    token_method = "tiktoken (gpt-4o-mini / o200k_base)"
except ImportError:
    enc = None
    token_method = "char/4 fallback (install tiktoken for exact counts)"

# ── Pricing (gpt-4o-mini) ────────────────────────────────────────────────────
INPUT_PRICE_PER_M  = 0.150   # USD per 1M input tokens
OUTPUT_PRICE_PER_M = 0.600   # USD per 1M output tokens

# ── Task config (from custom_tasks_lm_eval/primevul_gen.yaml) ─────────────────
REPEATS       = 5
MAX_GEN_TOKS  = 20

PROMPT_TEMPLATE = (
    "You are a security expert specializing in C/C++ vulnerability detection. "
    "Your task is to analyze the given function carefully and determine whether "
    "it contains a security vulnerability. "
    "Answer Yes if you can identify a vulnerability.\n\n"
    "Function:\n```c\n{func}\n```\n\n"
    "Is this function vulnerable? "
    "Answer in the format \\boxed{{Yes}} or \\boxed{{No}}.\n\nAnswer:"
)

# --apply_chat_template wraps the prompt in a user turn; the chat wrapper itself
# adds a small fixed overhead (role tokens + delimiters). We approximate it here.
CHAT_WRAPPER_OVERHEAD = 10  # tokens for role markers / delimiters per message


def count_tokens(text: str) -> int:
    if enc is not None:
        return len(enc.encode(text))
    return len(text) // 4


BASE      = Path(__file__).parent.parent
JSONL     = BASE / "data_preprocessing/datasets/primevul_balance/all.jsonl"


def main():
    print(f"Tokenizer : {token_method}")
    print(f"Dataset   : {JSONL}")
    print(f"Repeats   : {REPEATS}  |  max_gen_toks: {MAX_GEN_TOKS}")
    print(f"Pricing   : ${INPUT_PRICE_PER_M}/1M input  |  ${OUTPUT_PRICE_PER_M}/1M output")
    print()

    if not JSONL.exists():
        print(f"[ERROR] File not found: {JSONL}")
        return

    total_samples       = 0
    total_input_tokens  = 0
    total_output_tokens = 0

    with open(JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            func  = entry.get("func", "")

            prompt       = PROMPT_TEMPLATE.format(func=func)
            prompt_toks  = count_tokens(prompt) + CHAT_WRAPPER_OVERHEAD

            # each sample is evaluated REPEATS times
            total_input_tokens  += prompt_toks * REPEATS
            total_output_tokens += MAX_GEN_TOKS * REPEATS
            total_samples       += 1

    total_tokens = total_input_tokens + total_output_tokens

    input_cost  = total_input_tokens  / 1_000_000 * INPUT_PRICE_PER_M
    output_cost = total_output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    total_cost  = input_cost + output_cost

    print(f"Samples              : {total_samples:>12,}")
    print(f"Total API calls      : {total_samples * REPEATS:>12,}  ({total_samples:,} × {REPEATS} repeats)")
    print()
    print(f"Input  tokens        : {total_input_tokens:>12,}")
    print(f"Output tokens        : {total_output_tokens:>12,}  ({MAX_GEN_TOKS} × {total_samples * REPEATS:,} calls)")
    print(f"Total  tokens        : {total_tokens:>12,}")
    print()
    print(f"Input  cost          : ${input_cost:>10.2f}")
    print(f"Output cost          : ${output_cost:>10.2f}")
    print(f"Total  cost          : ${total_cost:>10.2f}")


if __name__ == "__main__":
    main()
