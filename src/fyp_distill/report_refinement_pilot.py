"""Summarize a completed, built engineering pilot without performance claims."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from statistics import mean, median

from .common import load_config, read_jsonl, resolve_path
from .refinement import GROUPS, digest, unique_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    bases = read_jsonl(root / "bases.jsonl")
    audits = unique_index(read_jsonl(root / "selection_audit.jsonl"))
    checks = unique_index(read_jsonl(root / "verification.jsonl"))
    screens = unique_index(read_jsonl(root / "screening.jsonl"))
    hints = read_jsonl(root / "hints.jsonl")
    if not bases or {b["id"] for b in bases} != set(audits):
        raise ValueError("Pilot has not been built completely")
    variants = {g: read_jsonl(resolve_path(config["data"]["variants_dir"]) / f"{g}.jsonl") for g in GROUPS}
    for group, rows in variants.items():
        if {r["source_id"] for r in rows} != set(audits) or len(rows) != len(bases):
            raise ValueError(f"Incomplete group: {group}")
    trials = [t for r in screens.values() for ts in r["boundaries"].values() for t in ts]
    verified_trials = []
    table = []
    n = config["refinement"]["verification_samples"]
    for base in bases:
        r = checks[base["id"]]
        local = sorted([h for h in hints if h["source_id"] == base["id"] and h["kind"] == "local"], key=lambda h: h["id"])
        if r["hint_hash"] != digest(local):
            raise ValueError("Reviewed hints changed after verification")
        arms = [r["baseline"], *[r["hints"][h["id"]] for h in local]]
        if any(len(ts) != n for ts in arms):
            raise ValueError("Incomplete independent verification")
        verified_trials.extend(t for ts in arms for t in ts)
        counts = [sum(t["correct"] for t in ts) for ts in arms]
        table.append({"id": base["id"], "correct_per_arm": counts,
                      "max_gain": max(audits[base["id"]]["gains"].values()), "accepted": audits[base["id"]]["accepted"]})
    ledger = read_jsonl(config["refinement"]["budget_ledger"])
    charged = sum(r["charged_cny"] for r in ledger)
    uncertain = [r for r in ledger if r["status"] != "complete"]
    all_trials = trials + verified_trials
    summary = {"created_at": datetime.now(UTC).isoformat(), "questions": len(bases), "hints": len(hints),
               "accepted": sum(r["accepted"] for r in table), "per_question": table,
               "groups": {g: {"records": len(rows), "changed_records": sum(r["hint_id"] is not None for r in rows),
                              "added_tokens": sum(r["added_tokens"] for r in rows)} for g, rows in variants.items()},
               "screen_trials": len(trials), "verification_trials": len(verified_trials),
               "rollout_seconds": sum(t["elapsed_seconds"] for t in all_trials),
               "rollout_mean_seconds": mean(t["elapsed_seconds"] for t in all_trials),
               "rollout_median_seconds": median(t["elapsed_seconds"] for t in all_trials),
               "generation_limit_hits": sum(t["finish_reason"] == "length" for t in all_trials),
               "new_ledger_total_cny": charged, "uncertain_reservations_cny": sum(r["charged_cny"] for r in uncertain),
               "uncertain_request_ids": [r["id"] for r in uncertain], "historical_cny": 2.549316,
               "combined_estimated_or_reserved_cny": 2.549316 + charged,
               "budget_mode": "question_exposure_pilot; not token-matched", "training_started": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
