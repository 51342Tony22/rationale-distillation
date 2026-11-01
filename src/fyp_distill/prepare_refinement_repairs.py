"""Save explicit repair requests for failed hints; does not send API calls."""
from __future__ import annotations

import argparse
import json

from .common import load_config, read_jsonl, resolve_path
from .refinement import unique_index, write_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()
    if args.revision < 1:
        parser.error("revision must be positive")
    config = load_config(args.config)
    root = resolve_path(config["refinement"]["root"])
    hints = unique_index(read_jsonl(root / "hints.jsonl"))
    ledger = unique_index(read_jsonl(config["refinement"]["budget_ledger"]))
    requests = []
    for row in read_jsonl(root / "teacher_requests.jsonl"):
        if row["id"] in hints:
            continue
        failed = ledger.get(row["id"], {})
        if failed.get("status") != "failed_or_uncertain":
            raise ValueError("Only explicitly failed requests may enter repair queue")
        if failed.get("response_content"):
            raise ValueError("Recover or review saved response before paying for replacement")
        requests.append({**row, "id": row["id"] + f":repair:{args.revision}",
                         "replaces_request_id": row["id"], "repair_reason": failed.get("error_type")})
    path = root / f"repair_requests_{args.revision}.jsonl"
    write_frozen(path, requests)
    print(json.dumps({"requests": len(requests), "output": str(path), "teacher_api_calls": 0}))


if __name__ == "__main__":
    main()
