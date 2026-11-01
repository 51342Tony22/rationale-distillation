"""Freeze a small exploratory subset without changing the running baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import baseline


def prepare():
    root = baseline.ROOT
    output = root / "data/math_distill_v1/pilot50"
    manifest = {
        "stage": "scope_frozen_training_spec_pending",
        "purpose": "Exploratory direction check, not formal held-out evidence",
        "selection": "First 50 of frozen training order; first 100 of frozen development order; no performance-based selection",
        "seed": 7,
        "conditions": ["student_base_reuse", "ordinary_cot", "difficulty_only", "difficulty_attention"],
        "deferred": ["hard_answer", "soft_answer", "random_and_budget_controls", "official_test", "multiple_seeds"],
        "splits": {},
        "training_started": False,
        "limitations": [
            "Development outputs and some errors have already been inspected; this is an exploratory development comparison.",
            "Attention extraction is functional, but localization and teacher-editing implementation still require validation.",
            "Training hyperparameters and target eligibility must be fixed before training; failed targets cannot be silently replaced.",
            "All trained conditions must use the same eligible source IDs, base initialization and training settings.",
            "Original v1 scores and source gold remain; apply the separately versioned sensitivity parser equally to every condition.",
            "More supervised tokens or improved answer formatting can explain changes; report them explicitly.",
            "A small single-seed comparison cannot establish stable improvement or isolate attention from all confounders."
        ],
    }
    selected = {}
    for split, count in (("train", 50), ("dev", 100)):
        source = root / f"data/math_distill_v1/{split}.jsonl"
        lines = source.read_bytes().splitlines(keepends=True)[:count]
        assert len(lines) == count and all(line.endswith(b"\n") for line in lines)
        payload = b"".join(lines)
        rows = [json.loads(line) for line in lines]
        ids = [row["source_id"] for row in rows]
        assert len(set(ids)) == count
        selected[split] = rows
        # This JSON envelope retains exact parent bytes as SHA evidence; the
        # selected row objects are reproduced by the existing freeze helper.
        baseline.freeze(output / f"{split}.json", rows)
        manifest["splits"][split] = {
            "count": count, "source_file": str(source.relative_to(root)),
            "parent_sha256": baseline.sha(source),
            "selected_parent_lines_sha256": hashlib.sha256(payload).hexdigest(),
            "source_ids": ids,
            "frozen_rows_sha256": baseline.sha(output / f"{split}.json"),
        }
    assert not {r["source_id"] for r in selected["train"]} & {r["source_id"] for r in selected["dev"]}
    predictions = root / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl"
    saved = {row["source_id"]: row for row in map(json.loads, predictions.read_text().splitlines())}
    for row in selected["dev"]:
        assert row["source_id"] in saved
        original = saved[row["source_id"]]
        assert original["gold"] == row["gold"]
        assert hashlib.sha256(original["output"].encode()).hexdigest() == original["output_sha256"]
    manifest["baseline_predictions_sha256"] = baseline.sha(predictions)
    manifest["sensitivity_parser_sha256"] = baseline.sha(Path(__file__).with_name("score_sensitivity_v2.py"))
    baseline.freeze(output / "manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "train": 50, "dev": 100,
                      "conditions": manifest["conditions"], "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
