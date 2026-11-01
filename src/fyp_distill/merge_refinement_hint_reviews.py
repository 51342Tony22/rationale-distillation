"""Merge an explicitly reviewed snapshot after the paid queue releases its lock.

Preserves the pre-merge paid candidates and ledger as frozen artifacts. This
utility performs provenance checks; it never decides mathematical validity.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
from datetime import UTC, datetime

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import digest, reviewed_hint, unique_index, write_frozen
from .request_refinement_hints import parse_hint


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.checkpoint):
        parser.error("Checkpoint must be an artifact-safe name")
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    ledger_path = resolve_path(config["refinement"]["budget_ledger"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_path(config["mlx_training"]["model"]), local_files_only=True)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    # Fail promptly; never edit a worker's in-memory ledger/candidate set.
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        paid = read_jsonl(root / "hints.jsonl")
        ledger_rows = read_jsonl(ledger_path)
        ledger = unique_index(ledger_rows)
        bases = unique_index(read_jsonl(root / "bases.jsonl"))
        reviewed = read_jsonl(root / "reviewed_hints.jsonl")
        unique_index(reviewed)
        if len(reviewed) != args.expected_count:
            raise ValueError("Unexpected reviewed snapshot size")
        requests = unique_index(read_jsonl(root / "teacher_requests.jsonl"))
        supplemental_requests = list(root.glob("local_repair_requests_*.jsonl"))
        supplemental_requests += list(root.glob("control_repair_requests_*.jsonl"))
        if (root / "control_requests.jsonl").exists():
            supplemental_requests.append(root / "control_requests.jsonl")
        for path in sorted(supplemental_requests):
            for request in read_jsonl(path):
                if request["id"] in requests and requests[request["id"]] != request:
                    raise ValueError("Conflicting saved replacement request")
                requests[request["id"]] = request
        recoveries = {}
        for path in sorted(root.glob("hint_response_recovery_reviews*.jsonl")):
            for record in read_jsonl(path):
                if record["id"] in recoveries:
                    raise ValueError("Duplicated recovery provenance")
                recoveries[record["id"]] = record
        paid_index = unique_index(paid)
        rejected = unique_index(read_jsonl(root / "rejected_local_hint_reviews.jsonl")
                                + read_jsonl(root / "rejected_control_hint_reviews.jsonl"))
        # No candidate may disappear except an explicitly rejected one, whose
        # original record remains in the frozen paid snapshot and shared ledger.
        missing = set(paid_index) - {hint["id"] for hint in reviewed}
        if missing - set(rejected):
            raise ValueError("Unreviewed paid candidates would be omitted")
        updates = []
        for hint in reviewed:
            reviewed_hint(hint, bases[hint["source_id"]], count,
                          config["refinement"]["max_hint_tokens"])
            if hint["review"].get("student_tokens") != count(hint["hint"]):
                raise ValueError("Incorrect recorded token count")
            request = requests[hint["id"]]
            entry = ledger[hint["id"]]
            if request["request_hash"] != hint["request_hash"] or entry["request_hash"] != hint["request_hash"]:
                raise ValueError("Request/response provenance mismatch")
            if hint["id"] in paid_index:
                if ({k: v for k, v in hint.items() if k != "review"} !=
                        {k: v for k, v in paid_index[hint["id"]].items() if k != "review"}):
                    raise ValueError("Reviewed candidate text/provenance changed")
            elif hint["id"] not in recoveries:
                raise ValueError("Missing saved response recovery review")
            if hint["id"] in recoveries:
                recovery = recoveries[hint["id"]]
                raw = entry["response_content"]
                if entry.get("finish_reason") != "stop" or not entry.get("usage") or sha(raw) != recovery["response_sha256"]:
                    raise ValueError("Incomplete or changed recovery response")
                block = raw[recovery["extraction_start"]:recovery["extraction_end"]]
                if sha(block) != recovery["extracted_json_sha256"]:
                    raise ValueError("Changed extracted public JSON")
                if "canonical_json_sha256" in recovery:
                    match = re.fullmatch(r'\{\s*"hint"\s*:\s*"([^"\n]*)"\s*,\s*"offset"\s*:\s*(\d+)\s*\}', block)
                    if not match:
                        raise ValueError("Ambiguous raw field recovery")
                    block = json.dumps({"hint": match.group(1), "offset": int(match.group(2))})
                    if sha(block) != recovery["canonical_json_sha256"]:
                        raise ValueError("Changed canonical public JSON")
                parsed = parse_hint(request, block)
                if {k: v for k, v in parsed.items() if k != "review"} != {
                        k: v for k, v in hint.items()
                        if k not in {"review", "response_parse", "recovered_public_response_sha256"}}:
                    raise ValueError("Recovery changed mathematical text")
                updates.append((entry, recovery))
        for record in read_jsonl(root / "verification.jsonl"):
            if record.get("hint_hash"):
                local = sorted([h for h in reviewed if h["source_id"] == record["id"] and h["kind"] == "local"], key=lambda h: h["id"])
                if record["hint_hash"] != digest(local):
                    raise ValueError("Snapshot changed after verification")
        # All validation precedes mutations. Keep every paid original and error.
        write_frozen(root / f"paid_hints_before_{args.checkpoint}.jsonl", paid)
        write_frozen(root / f"ledger_before_{args.checkpoint}.jsonl", ledger_rows)
        for entry, recovery in updates:
            previous_status = entry["status"]
            entry["status"] = "complete"
            if "recovery" not in entry:
                entry["recovery"] = {"method": recovery["recovery_method"],
                    "previous_status": previous_status, "previous_error": entry.get("error_type"),
                    "response_sha256": recovery["response_sha256"],
                    "api_calls": 0, "at": datetime.now(UTC).isoformat()}
        write_jsonl(ledger_path, ledger.values())
        write_jsonl(root / "hints.jsonl", reviewed)
        write_frozen(root / f"hint_merge_{args.checkpoint}.jsonl", [{
            "reviewed_count": len(reviewed), "removed_from_active_pool": sorted(missing),
            "preserved_paid_snapshot": f"paid_hints_before_{args.checkpoint}.jsonl",
            "recovered_ids": [e["id"] for e, _ in updates], "teacher_api_calls": 0,
            "reviewed_snapshot_sha256": hashlib.sha256((root / "reviewed_hints.jsonl").read_bytes()).hexdigest()}])
    print(json.dumps({"merged": len(reviewed), "recovered": len(updates),
                      "preserved_rejections": len(missing), "teacher_api_calls": 0}))


if __name__ == "__main__":
    main()
