from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from .common import configured_variants, load_config, read_jsonl, resolve_path


def bootstrap_difference(left: dict, right: dict, samples: int, seed: int) -> dict:
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"difference": None, "ci_low": None, "ci_high": None, "n": 0}
    deltas = np.array([float(left[key]) - float(right[key]) for key in shared])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return {
        "difference": float(deltas.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": len(shared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine evaluation results into final tables.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("dev", "test", "svamp", "math500"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    variants = configured_variants(config)
    seed = args.seed if args.seed is not None else config["seed"]
    rows = []
    correctness = {}
    model_names = ("base_student", *variants)
    for variant in model_names:
        directory = resolve_path("reports") / args.split / variant / f"seed_{seed}"
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(metrics)
        predictions = read_jsonl(directory / "predictions.jsonl")
        correctness[variant] = {row["id"]: row["correct"] for row in predictions}
    teacher_dir = resolve_path("reports") / args.split / "teacher"
    teacher_metrics_path = teacher_dir / "metrics.json"
    if teacher_metrics_path.exists():
        with teacher_metrics_path.open(encoding="utf-8") as handle:
            teacher_metrics = json.load(handle)
        rows.append(teacher_metrics)
        teacher_predictions = read_jsonl(teacher_dir / "predictions.jsonl")
        correctness["teacher"] = {row["id"]: row["correct"] for row in teacher_predictions}
    if not rows:
        raise SystemExit("No evaluation metrics found.")

    comparisons = {}
    pairs = [
        ("single_rationale", "answer_only"),
        ("multi_rationale", "single_rationale"),
        ("diverse_multi_rationale", "multi_rationale"),
    ]
    for left, right in pairs:
        if left in correctness and right in correctness:
            comparisons[f"{left}_minus_{right}"] = bootstrap_difference(
                correctness[left],
                correctness[right],
                samples=config["evaluation"]["bootstrap_samples"],
                seed=seed,
            )

    by_variant = {row["variant"]: row for row in rows}
    gap_closure = {}
    if "teacher" in by_variant and "base_student" in by_variant:
        teacher_accuracy = by_variant["teacher"]["exact_match"]
        base_accuracy = by_variant["base_student"]["exact_match"]
        denominator = teacher_accuracy - base_accuracy
        for variant in variants:
            if variant in by_variant:
                gap_closure[variant] = (
                    (by_variant[variant]["exact_match"] - base_accuracy) / denominator
                    if denominator
                    else None
                )

    report_dir = resolve_path("reports")
    fieldnames = sorted({key for row in rows for key in row})
    with (report_dir / f"summary_{args.split}_seed_{seed}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (report_dir / f"comparisons_{args.split}_seed_{seed}.json").open("w", encoding="utf-8") as handle:
        json.dump(comparisons, handle, indent=2)

    headers = ["Variant", "N", "Exact match", "Invalid", "Words", "Sample agreement", "Prompt agreement"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for variant in ("teacher", *model_names):
        if variant not in by_variant:
            continue
        row = by_variant[variant]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(row["n"]),
                    f"{row['exact_match']:.3f}",
                    f"{row['invalid_answer_rate']:.3f}",
                    f"{row.get('mean_rationale_words', float('nan')):.1f}",
                    f"{row.get('mean_sample_agreement', float('nan')):.3f}",
                    f"{row.get('mean_prompt_agreement', float('nan')):.3f}",
                ]
            )
            + " |"
        )
    markdown_path = report_dir / f"summary_{args.split}_seed_{seed}.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(comparisons, indent=2))
    if gap_closure:
        print("Gap closure:")
        print(json.dumps(gap_closure, indent=2))


if __name__ == "__main__":
    main()
