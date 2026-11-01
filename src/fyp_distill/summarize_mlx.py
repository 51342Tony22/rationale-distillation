from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from .common import artifact_dir, configured_variants, load_config, read_jsonl, resolve_path
from .summarize import bootstrap_difference


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all MLX seeds for one split.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=("dev", "test", "svamp", "math500"), required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    variants = configured_variants(config)
    seeds = config["seeds"]
    root = artifact_dir(config, "mlx_report_dir", "reports_mlx")
    aggregate_rows = []
    correctness: dict[str, dict[str, list[float]]] = {}
    required_ids = None
    for variant in variants:
        metrics = []
        by_question: dict[str, list[float]] = {}
        for seed in seeds:
            run_dir = root / args.split / variant / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                if config["evaluation"].get("require_all_seeds"):
                    raise SystemExit(f"Missing required seed result: {metrics_path}")
                continue
            metric = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = read_jsonl(run_dir / "predictions.jsonl")
            if config["evaluation"].get("require_all_seeds"):
                ids = {prediction["id"] for prediction in predictions}
                if not ids or len(ids) != len(predictions) or len(ids) != metric["n"]:
                    raise SystemExit(f"Incomplete/duplicate predictions: {run_dir}")
                if required_ids is None:
                    required_ids = ids
                elif ids != required_ids:
                    raise SystemExit(f"Question coverage differs between groups/seeds: {run_dir}")
            metrics.append(metric)
            for prediction in predictions:
                by_question.setdefault(prediction["id"], []).append(float(prediction["correct"]))
        if not metrics:
            continue
        accuracies = np.array([row["exact_match"] for row in metrics])
        aggregate_rows.append(
            {
                "variant": variant,
                "seeds": len(metrics),
                "n_per_seed": metrics[0]["n"],
                "exact_match_mean": float(accuracies.mean()),
                "exact_match_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
                "invalid_answer_rate_mean": float(
                    np.mean([row["invalid_answer_rate"] for row in metrics])
                ),
                "mean_rationale_words": float(
                    np.mean([row["mean_rationale_words"] for row in metrics])
                ),
            }
        )
        correctness[variant] = {
            question_id: values
            for question_id, values in by_question.items()
            if len(values) == len(metrics)
        }

    if not aggregate_rows:
        raise SystemExit(f"No completed MLX metrics for split {args.split}.")

    comparisons = {}
    pairs = (
        ("single_rationale", "answer_only"),
        ("multi_rationale", "single_rationale"),
        ("multi_rationale", "low_diversity_rationale"),
        ("diverse_multi_rationale", "multi_rationale"),
        ("diverse_multi_rationale", "low_diversity_rationale"),
        ("compatible_rationale", "multi_rationale"),
        ("compatible_rationale", "diverse_multi_rationale"),
        ("diverse_multi_rationale", "answer_only"),
    )
    pairs = config["evaluation"].get("comparison_pairs", pairs)
    for left, right in pairs:
        if left not in correctness or right not in correctness:
            continue
        left_means = {key: float(np.mean(value)) for key, value in correctness[left].items()}
        right_means = {key: float(np.mean(value)) for key, value in correctness[right].items()}
        comparisons[f"{left}_minus_{right}"] = bootstrap_difference(
            left_means,
            right_means,
            samples=config["evaluation"]["bootstrap_samples"],
            seed=config["seed"],
        )

    base_metrics_path = root / args.split / "base_student" / f"seed_{seeds[0]}" / "metrics.json"
    teacher_root = config.get("artifacts", {}).get("teacher_report_dir")
    teacher_metrics_path = (
        resolve_path(teacher_root) / args.split / "metrics.json" if teacher_root
        else resolve_path("reports") / args.split / "teacher" / "metrics.json"
    )
    gap_closure = {}
    if base_metrics_path.exists() and teacher_metrics_path.exists():
        base_accuracy = json.loads(base_metrics_path.read_text(encoding="utf-8"))["exact_match"]
        teacher_metrics = json.loads(teacher_metrics_path.read_text(encoding="utf-8"))
        teacher_accuracy = teacher_metrics["exact_match"]
        denominator = teacher_accuracy - base_accuracy
        if denominator:
            gap_closure = {
                row["variant"]: (row["exact_match_mean"] - base_accuracy) / denominator
                for row in aggregate_rows
            }

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"summary_{args.split}_all_seeds.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    payload = {"comparisons": comparisons, "gap_closure": gap_closure}
    (root / f"statistics_{args.split}_all_seeds.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "| Variant | Seeds | N/seed | Exact match mean | Std | Invalid | Words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {variant} | {seeds} | {n_per_seed} | {exact_match_mean:.3f} | "
            "{exact_match_std:.3f} | {invalid_answer_rate_mean:.3f} | "
            "{mean_rationale_words:.1f} |".format(**row)
        )
    markdown = "\n".join(lines) + "\n"
    (root / f"summary_{args.split}_all_seeds.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
