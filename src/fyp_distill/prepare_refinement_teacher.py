"""Save train-side MATH original-rationale requests without spending API credit."""
from __future__ import annotations

import argparse
import json

from .common import load_config, read_jsonl, resolve_path
from .refinement import digest, unique_index, write_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refinement_math.yaml")
    parser.add_argument("--input", default="data/refinement_math/processed/math_pilot.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    pool = unique_index(read_jsonl(config["data"]["train_file"]))
    rows = read_jsonl(args.input)
    if args.limit <= 0 or not rows:
        raise ValueError("Need a nonempty, positively bounded training subset")
    requests = []
    for row in rows[:args.limit]:
        if row["id"] not in pool or row != pool[row["id"]]:
            raise ValueError("Original rationale generation is restricted to frozen TRAIN rows")
        prompt = (
            "Solve this mathematics problem accurately. Give a concise, self-contained public "
            "solution in your final response. Put each meaningful reasoning step on a separate line. "
            "Do not force a short word limit or omit essential derivations. Use at most roughly "
            "1800 output tokens. Do not include a boxed final answer earlier in the derivation. "
            "End with one line `Final answer: <answer>`; the answer may be a fraction, expression, "
            "tuple or other mathematical object, not necessarily a number.\n\nProblem:\n" + row["question"]
        )
        requests.append({"id": row["id"] + ":original:0", "source_id": row["id"],
                         "source_hash": digest(row), "kind": "original", "dataset": "math",
                         "question": row["question"], "gold_answer": row["gold_answer"],
                         "model": "deepseek-ai/DeepSeek-R1", "prompt": prompt,
                         "request_hash": digest(prompt), "max_output_tokens": 2048})
    output = resolve_path(config["refinement"]["root"]) / "original_requests.jsonl"
    write_frozen(output, requests)
    print(json.dumps({"requests": len(requests), "output": str(output), "api_calls_made": 0}))


if __name__ == "__main__":
    main()
