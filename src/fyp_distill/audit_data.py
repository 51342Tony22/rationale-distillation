from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import configured_variants, load_config, read_jsonl, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experimental data before training.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    variant_names = configured_variants(config)
    data = config["data"]
    train = read_jsonl(data["train_file"])
    dev = read_jsonl(data["dev_file"])
    test = read_jsonl(data["test_file"])
    variants_dir = resolve_path(data["variants_dir"])
    variants = {name: read_jsonl(variants_dir / f"{name}.jsonl") for name in variant_names}

    train_questions = {row["question"].strip() for row in train}
    test_questions = {row["question"].strip() for row in test}
    dev_questions = {row["question"].strip() for row in dev}
    id_sets = {name: {row["source_id"] for row in rows} for name, rows in variants.items()}
    train_by_id = {row["id"]: row for row in train}
    reference = id_sets["answer_only"]
    report = {
        "source": {"train": len(train), "dev": len(dev), "test": len(test)},
        "train_test_question_overlap": len(train_questions & test_questions),
        "train_dev_question_overlap": len(train_questions & dev_questions),
        "dev_test_question_overlap": len(dev_questions & test_questions),
        "variants": {
            name: {
                "rows": len(rows),
                "unique_questions": len(id_sets[name]),
                "approx_characters": sum(len(row["prompt"] + row["completion"]) for row in rows),
                "duplicate_completions": sum(
                    count - 1 for count in Counter(row["completion"] for row in rows).values() if count > 1
                ),
            }
            for name, rows in variants.items()
        },
        "same_question_set_across_variants": all(ids == reference for ids in id_sets.values()),
        "same_row_count_across_variants": len({len(rows) for rows in variants.values()}) == 1,
        "unknown_or_mismatched_variant_rows": {
            name: sum(
                row["source_id"] not in train_by_id
                or row["question"].strip() != train_by_id[row["source_id"]]["question"].strip()
                or row["gold_answer"] != train_by_id[row["source_id"]]["gold_answer"]
                for row in rows
            )
            for name, rows in variants.items()
        },
    }
    failures = []
    if report["train_test_question_overlap"]:
        failures.append("train/test leakage detected")
    if report["train_dev_question_overlap"] or report["dev_test_question_overlap"]:
        failures.append("development split leakage detected")
    if not report["same_question_set_across_variants"]:
        failures.append("variants do not share the same source question set")
    if not report["same_row_count_across_variants"]:
        failures.append("variants do not have matched question exposure counts")
    for name, count in report["unknown_or_mismatched_variant_rows"].items():
        if count:
            failures.append(f"{name} contains {count} stale or mismatched rows")
    for name, rows in variants.items():
        if not rows:
            failures.append(f"{name} is empty")
    report["status"] = "FAIL" if failures else "PASS"
    report["failures"] = failures
    output = resolve_path("reports/data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
