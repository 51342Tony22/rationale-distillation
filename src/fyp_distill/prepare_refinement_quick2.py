"""Freeze two already-paid, reviewed teacher solutions for a mechanism check."""
from __future__ import annotations

import json

from .common import load_config, read_jsonl, resolve_path
from .refinement import digest, stable_seed, write_frozen


def main():
    parent = "data/refinement_math_calibration7_v3"
    root = "data/refinement_math_quick2_v3"
    config_path = resolve_path("configs/refinement_math_quick2_v3.yaml")
    config = load_config("configs/refinement_math_calibration7_v3.yaml")
    config["refinement"].update(root=root, question_limit=2)
    config["data"].update(teacher_file=root + "/teacher_candidates.jsonl",
                          variants_dir=root + "/variants")
    for key, value in config["artifacts"].items():
        config["artifacts"][key] = value.replace("calibration7", "quick2")
    # The config itself is immutable after the first local rollout.
    if config_path.exists():
        if load_config(config_path) != config:
            raise ValueError("Existing quick config differs")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    ends = {
        "math_train_algebra_2425.json": [
            "\\[8x^2 - 65x + 8 = 0\\]  \n",
            "and \\(c = 8\\).  \n",
            "= 4225 - 256 = 3969\\]  \n",
        ],
        "math_train_counting_and_probability_393.json": [
            "4 red, 5 white, and 6 blue.\n\n",
            "= \\frac{2730}{6} = 455.\n\\]\n\n",
            "all red, all white, or all blue.\n\n",
        ],
    }
    rows = [r for r in read_jsonl(parent + "/teacher_candidates.jsonl") if r["id"] in ends]
    if len(rows) != 2:
        raise ValueError("Both reviewed original solutions are required")
    write_frozen(resolve_path(root + "/teacher_candidates.jsonl"), rows)
    bases = []
    for row in rows:
        candidate = row["candidates"][0]
        original = candidate["text"]
        boundaries = sorted(original.index(anchor) + len(anchor) for anchor in ends[row["id"]])
        record = dict(id=row["id"], split="train", dataset="math", question=row["question"],
                      gold_answer=row["gold_answer"], teacher_model=row["teacher_model"],
                      candidate_id=candidate["candidate_id"], original=original,
                      boundaries=boundaries,
                      random_boundary=boundaries[stable_seed(config["seed"], row["id"], "quick_random") % 3],
                      review={"correct_steps": True, "prefixes_do_not_reveal_answer": True})
        record["source_hash"] = digest({k: v for k, v in record.items() if k != "review"})
        bases.append(record)
    write_frozen(resolve_path(root + "/bases.jsonl"), bases)
    write_frozen(resolve_path(root + "/selection_manifest.jsonl"), [{
        "ids": list(ends), "selection_reason": "first two available manually checked teacher originals",
        "scope": "mechanism check only; convenience subset, not a representative accuracy benchmark",
        "originals_reused": True, "boundary_review": "complete steps before any final roots or final ratio",
        "teacher_hint_namespace": "quick2_v3", "training_started": False,
    }])
    print(json.dumps({"config": str(config_path), "questions": len(bases), "boundaries": 6}))


if __name__ == "__main__":
    main()
