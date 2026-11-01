from __future__ import annotations

import argparse
import json
import statistics

from .build_variants import mean_pairwise_distance, select_diverse, select_low_diversity
from .common import load_config, read_jsonl, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize teacher rationale quality and cost.")
    parser.add_argument("--config", default="configs/qwen3_8b_main.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--input-price", type=float, default=4.0, help="CNY per M tokens")
    parser.add_argument("--output-price", type=float, default=16.0, help="CNY per M tokens")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_jsonl(args.input or config["data"]["teacher_file"])
    if not rows:
        raise SystemExit("No teacher records found.")
    candidates = [candidate for row in rows for candidate in row.get("candidates", [])]
    usable = [candidate for candidate in candidates if candidate.get("is_usable", False)]
    usage = [candidate["usage"] for candidate in candidates if candidate.get("usage")]
    input_tokens = sum(item.get("prompt_tokens") or 0 for item in usage)
    output_tokens = sum(item.get("completion_tokens") or 0 for item in usage)
    k = config["data"]["selected_rationales"]
    eligible = [
        [candidate for candidate in row.get("candidates", []) if candidate.get("is_usable", False)]
        for row in rows
    ]
    eligible = [group for group in eligible if len(group) >= k]
    low_distances = [
        mean_pairwise_distance(select_low_diversity(group, k)) for group in eligible
    ]
    high_distances = [mean_pairwise_distance(select_diverse(group, k)) for group in eligible]
    payload = {
        "questions_present": len(rows),
        "questions_complete": sum(
            len(row.get("candidates", [])) == config["data"]["rationale_candidates"]
            for row in rows
        ),
        "candidates_present": len(candidates),
        "usable_candidates": len(usable),
        "usable_rate": len(usable) / max(len(candidates), 1),
        "eligible_questions_k": len(eligible),
        "per_question_usable": [
            sum(candidate.get("is_usable", False) for candidate in row.get("candidates", []))
            for row in rows
        ],
        "api_errors": sum(bool(candidate.get("error")) for candidate in candidates),
        "length_finishes": sum(
            candidate.get("finish_reason") == "length" for candidate in candidates
        ),
        "wrong_or_quality_failed": sum(
            not candidate.get("is_usable", False) and not candidate.get("error")
            for candidate in candidates
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "mean_output_tokens": statistics.mean(
            item.get("completion_tokens") or 0 for item in usage
        ) if usage else None,
        "estimated_cost_cny": (
            input_tokens * args.input_price + output_tokens * args.output_price
        ) / 1_000_000,
        "price_cny_per_m_tokens": {
            "input": args.input_price,
            "output": args.output_price,
        },
        "mean_low_diversity_distance": statistics.mean(low_distances)
        if low_distances else None,
        "mean_high_diversity_distance": statistics.mean(high_distances)
        if high_distances else None,
    }
    output = resolve_path(
        args.output
        or config.get("artifacts", {}).get(
            "teacher_summary", "reports_qwen3_8b/data/teacher_summary.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
