"""Freeze one level 3-5 pilot question per MATH subject before observing outputs."""
from __future__ import annotations

import argparse
import json

from .common import load_config, read_jsonl, resolve_path
from .refinement import stable_seed, write_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    pool = read_jsonl("data/refinement_math/processed/math_pilot.jsonl")
    billed = {r["id"].split(":original:")[0] for r in read_jsonl(config["refinement"]["budget_ledger"])
              if ":original:" in r["id"]}
    subjects = sorted({r["subject"] for r in pool})
    levels = (3, 4, 5)
    selected = []
    for index, subject in enumerate(subjects):
        level = levels[index % len(levels)]
        eligible = [r for r in pool if r["subject"] == subject and r["level"] == level
                    and r["id"] not in billed]
        if not eligible:
            raise ValueError(f"No unused pilot question for {subject}, level {level}")
        selected.append(min(eligible, key=lambda r: stable_seed(config["seed"], "calibration-v3", r["id"])))
    write_frozen(root / "selected_questions.jsonl", selected)
    manifest = {"seed": config["seed"], "selection": "one per subject; levels 3/4/5 cyclic in sorted subject order; stable hash tie-break",
                "questions": [{"id":r["id"], "subject":r["subject"], "level":r["level"]} for r in selected],
                "selected_before_teacher_or_student_outputs": True, "train_only": True,
                "teacher_api_calls": 0}
    write_frozen(root / "selection_manifest.jsonl", [manifest])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
