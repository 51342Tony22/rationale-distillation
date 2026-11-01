"""Freeze complete manually reviewed originals after the final paid queue ends."""
from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from datetime import UTC, datetime

from .common import load_config, read_jsonl, resolve_path
from .refinement import digest, question_key, unique_index, write_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    status = read_jsonl(root / "run_status.jsonl")
    if not status or status[0].get("stage") != "original_repair_queue_finished":
        raise ValueError("Wait for the final original queue to finish before freezing")
    final_requests = read_jsonl(root / "original_repair_queue_2.jsonl")
    selected = unique_index(read_jsonl(root / "selected_questions.jsonl"))
    drafts = unique_index(read_jsonl(root / "draft_bases_reviewed.jsonl"))
    teacher = unique_index(read_jsonl(config["data"]["teacher_file"]))
    corrected = unique_index(read_jsonl(root / "manual_corrected_candidates.jsonl"))
    source_excluded = unique_index(read_jsonl(root / "data_quality_exclusions.jsonl"))
    rejected = unique_index(read_jsonl(root / "rejected_original_response_reviews.jsonl"), "source_id")
    ledger_path = resolve_path(config["refinement"]["budget_ledger"])
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = unique_index(read_jsonl(ledger_path))
        for request in final_requests:
            if ledger.get(request["id"], {}).get("status") not in {"complete", "failed_or_uncertain"}:
                raise ValueError("Every final request must have a terminal ledger entry")
        charged = sum(r["charged_cny"] for r in ledger.values())
        if charged > config["refinement"]["new_api_budget_cny"] or charged + 2.549316 > 42:
            raise ValueError("Shared budget exceeds authorized cap")
        exclusions = []
        for source_id in selected:
            if source_id in drafts:
                if source_id in source_excluded or source_id in rejected:
                    raise ValueError("A rejected source cannot enter the frozen training set")
                continue
            if source_id in teacher:
                raise ValueError("A returned original still needs actual review")
            if source_id in source_excluded:
                exclusions.append({"id": source_id, "reason": "invalid_source_math", "detail": source_excluded[source_id]})
                continue
            request = next((r for r in final_requests if r["source_id"] == source_id), None)
            if request is None:
                raise ValueError("Missing source has no reviewed final resolution")
            entry = ledger[request["id"]]
            review = rejected.get(source_id)
            if entry.get("response_content") and review is None:
                raise ValueError("Inspect/recover paid response before excluding the source")
            if review:
                exclusions.append({"id": source_id, "reason": review["review_category"], "detail": review})
            else:
                exclusions.append({"id": source_id, "reason": "final_api_failure_without_saved_response",
                                   "request_id": request["id"], "error_type": entry.get("error_type")})
        if set(drafts) - set(selected):
            raise ValueError("Draft outside frozen selection")
        heldout = {question_key(r["question"]) for r in read_jsonl(root / "eval_questions.jsonl")}
        for source_id, base in drafts.items():
            if base["question"] != selected[source_id]["question"] or question_key(base["question"]) in heldout:
                raise ValueError("Source mismatch or exact evaluation overlap")
            if base["source_hash"] != digest({k: v for k, v in base.items() if k not in {"review", "source_hash"}}):
                raise ValueError("Audited source hash mismatch")
            if not all(base.get("review", {}).get(k) is True for k in ("correct_steps", "prefixes_do_not_reveal_answer")):
                raise ValueError("Actual semantic review required")
            bounds = base["boundaries"]
            if len(bounds) != 3 or bounds != sorted(set(bounds)) or any(not 0 < b < len(base["original"]) for b in bounds):
                raise ValueError("Require three unique complete-step reviewed boundaries")
            candidate_ids = {r["candidate_id"] for r in teacher[source_id]["candidates"]}
            candidate_ids |= {r["candidate_id"] for r in corrected.get(source_id, {}).get("candidates", [])}
            if base["candidate_id"] not in candidate_ids:
                raise ValueError("Original candidate provenance missing")
        ordered = [drafts[k] for k in selected if k in drafts]
        if not ordered:
            raise ValueError("No valid reviewed originals")
        write_frozen(root / "original_processing_exclusions.jsonl", exclusions)
        write_frozen(root / "bases.jsonl", ordered)
        coverage = dict(Counter(selected[b["id"]]["subject"] for b in ordered))
        manifest = {"frozen_at": datetime.now(UTC).isoformat(),
                     "selected": len(selected), "valid": len(ordered), "excluded": len(exclusions),
                     "by_subject": coverage, "seven_subjects_covered": len(coverage) == 7,
                     "selection_changed": False, "no_further_paid_original_repair": True,
                     "source_hashes": {b["id"]: b["source_hash"] for b in ordered}}
        prior_manifest = read_jsonl(root / "base_freeze_manifest.jsonl")
        if prior_manifest:
            manifest["frozen_at"] = prior_manifest[0]["frozen_at"]
        write_frozen(root / "base_freeze_manifest.jsonl", [manifest])
        print({"frozen_valid": len(ordered), "excluded": len(exclusions), "by_subject": coverage})


if __name__ == "__main__":
    main()
