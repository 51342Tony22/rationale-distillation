from __future__ import annotations

import argparse
import json
import statistics

from transformers import AutoTokenizer

from .common import configured_variants, load_config, read_jsonl, resolve_path


def count_row(tokenizer, row: dict) -> tuple[int, int]:
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        add_generation_prompt=True,
        tokenize=True,
    )
    full_tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ],
        add_generation_prompt=False,
        tokenize=True,
    )
    return len(full_tokens), max(len(full_tokens) - len(prompt_tokens), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit token budgets across training conditions.")
    parser.add_argument("--config", default="configs/qwen3_8b_main.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    model = args.model or config["mlx_training"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(model), local_files_only=True)
    max_length = config["mlx_training"]["max_length"]
    variants_dir = resolve_path(config["data"]["variants_dir"])
    report = {
        "tokenizer": str(resolve_path(model)),
        "max_length": max_length,
        "variants": {},
    }
    for variant in configured_variants(config):
        rows = read_jsonl(variants_dir / f"{variant}.jsonl")
        if not rows:
            raise SystemExit(f"Missing variant data for {variant}.")
        counts = [count_row(tokenizer, row) for row in rows]
        total = [item[0] for item in counts]
        assistant = [item[1] for item in counts]
        report["variants"][variant] = {
            "rows": len(rows),
            "unique_questions": len({row["source_id"] for row in rows}),
            "total_tokens": sum(total),
            "assistant_tokens": sum(assistant),
            "mean_total_tokens": statistics.mean(total),
            "mean_assistant_tokens": statistics.mean(assistant),
            "median_assistant_tokens": statistics.median(assistant),
            "max_total_tokens": max(total),
            "rows_over_max_length": sum(value > max_length for value in total),
        }
    assistant_totals = [
        value["assistant_tokens"] for value in report["variants"].values()
    ]
    report["assistant_token_budget_ratio_max_to_min"] = max(assistant_totals) / min(
        assistant_totals
    )
    output = resolve_path(
        args.output
        or config.get("artifacts", {}).get(
            "token_audit", "reports_qwen3_8b/data/token_audit.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
