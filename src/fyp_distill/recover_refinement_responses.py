"""Recover parseable saved hint responses, without sending any API requests."""
from __future__ import annotations

import argparse
import fcntl
import json
from datetime import UTC, datetime

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import unique_index
from .request_refinement_hints import parse_hint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    ledger_path = resolve_path(config["refinement"]["budget_ledger"])
    requests = unique_index(read_jsonl(root / "teacher_requests.jsonl"))
    # Fail promptly if a paid queue still owns the ledger. Never interfere with
    # its in-memory copy or issue a retry for unknown-billing failures.
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = unique_index(read_jsonl(ledger_path))
        hints = unique_index(read_jsonl(root / "hints.jsonl"))
        recovered = []
        for key, request in requests.items():
            entry = ledger.get(key, {})
            if key in hints or request["kind"] == "original":
                continue
            if (entry.get("finish_reason") == "length" or not entry.get("response_content")
                    or entry.get("request_hash") != request["request_hash"]):
                continue
            try:
                hint = parse_hint(request, entry["response_content"])
            except ValueError:
                continue
            hints[key] = hint
            entry["status"] = "complete"
            entry["recovery"] = {"method": "parse_saved_public_response", "previous_error": entry.get("error_type"),
                                 "at": datetime.now(UTC).isoformat(), "api_calls": 0}
            recovered.append(key)
        write_jsonl(ledger_path, ledger.values())
        write_jsonl(root / "hints.jsonl", hints.values())
    print(json.dumps({"recovered": recovered, "teacher_api_calls": 0}))


if __name__ == "__main__":
    main()
