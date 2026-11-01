"""Summarize a complete baseline/local mechanism check without training claims."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import digest, hardest, screen_index, unique_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["refinement"]
    root = resolve_path(settings["root"])
    bases = read_jsonl(root / "bases.jsonl")
    screens = screen_index(bases, read_jsonl(root / "screening.jsonl"), settings["screening_samples"])
    hints = read_jsonl(root / "reviewed_hints.jsonl")
    verified = unique_index(read_jsonl(root / "verification.jsonl"))
    all_trials, rows = [], []
    for base in bases:
        score = verified[base["id"]]
        local = sorted([h for h in hints if h["source_id"] == base["id"] and h["kind"] == "local"], key=lambda h: h["id"])
        assert len(local) == settings["local_candidates"]
        assert score["source_hash"] == base["source_hash"] and score["hint_hash"] == digest(local)
        baseline = score["baseline"]
        n = settings["verification_samples"]
        assert len(baseline) == n and len({x["seed"] for x in baseline}) == n
        screen_trials = [t for values in screens[base["id"]]["boundaries"].values() for t in values]
        assert not {t["seed"] for t in screen_trials} & {t["seed"] for t in baseline}
        b = sum(t["correct"] for t in baseline)
        candidates = []
        for h in local:
            trials = score["hints"][h["id"]]
            assert len(trials) == n and [t["seed"] for t in trials] == [t["seed"] for t in baseline]
            correct = sum(t["correct"] for t in trials)
            candidates.append({"id": h["id"], "correct": correct, "n": n,
                               "gain": (correct-b)/n,
                               "wrong_to_right": sum(not x["correct"] and y["correct"] for x,y in zip(baseline,trials)),
                               "right_to_wrong": sum(x["correct"] and not y["correct"] for x,y in zip(baseline,trials))})
            all_trials.extend(trials)
        all_trials.extend(screen_trials + baseline)
        rows.append({"id": base["id"], "selected_boundary": hardest(base, screens[base["id"]]),
                     "screen_correct_by_offset": {k: sum(t["correct"] for t in ts) for k,ts in screens[base["id"]]["boundaries"].items()},
                     "baseline_correct": b, "n": n, "candidates": candidates,
                     "accepted": any(c["gain"] >= settings["min_gain"] for c in candidates)})
    ledger = read_jsonl(settings["budget_ledger"])
    ids = {r["id"] for r in read_jsonl(root / "local_requests.jsonl")}
    now = datetime.now(UTC)
    summary = {"scope": "two-question convenience subset; hint-assisted mechanism check only",
               "training_started": False, "six_groups_built": False,
               "answer_protocol": config["evaluation"]["answer_protocol"], "questions": rows,
               "accepted_questions": sum(r["accepted"] for r in rows),
               "screen_rollouts": len(bases)*settings["max_boundaries"]*settings["screening_samples"],
               "verify_rollouts": len(bases)*settings["verification_samples"]*(settings["local_candidates"]+1),
               "student_elapsed_seconds": sum(t["elapsed_seconds"] for t in all_trials),
               "generation_token_limit_hits": sum(t["generation_tokens"] >= settings["max_new_tokens"] for t in all_trials),
               "max_prompt_tokens": max(t["prompt_tokens"] for t in all_trials),
               "quick_hint_cost_or_reserve_cny": sum(r["charged_cny"] for r in ledger if r["id"] in ids),
               "shared_new_charged_or_reserved_cny": sum(r["charged_cny"] for r in ledger),
               "shared_unknown_usage_reserve_cny": sum(r["charged_cny"] for r in ledger if not r.get("usage")),
               "historical_cost_cny": 2.549316,
               "combined_cost_or_reserved_cny": 2.549316 + sum(r["charged_cny"] for r in ledger),
               "finished_at": now.isoformat(),
               "wall_seconds_since_config_snapshot": now.timestamp()-resolve_path(args.config).stat().st_mtime}
    assert summary["shared_new_charged_or_reserved_cny"] <= settings["new_api_budget_cny"]
    write_jsonl(root / "quick_summary.jsonl", [summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
