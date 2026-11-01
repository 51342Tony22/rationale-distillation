from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime

from .common import normalize_answer, resolve_path, write_jsonl


URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and freeze the official SVAMP challenge set.")
    parser.add_argument("--output", default="data/processed/svamp_test.jsonl")
    args = parser.parse_args()
    raw_path = resolve_path("data/raw/SVAMP.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print("Downloading official SVAMP challenge set...")
        urllib.request.urlretrieve(URL, raw_path)
    raw = raw_path.read_bytes()
    source = json.loads(raw)
    rows = [
        {
            "id": f"svamp_{item['ID']}",
            "split": "svamp",
            "question": f"{item['Body'].strip()}\n{item['Question'].strip()}",
            "gold_answer": normalize_answer(str(item["Answer"])),
            "gold_equation": item["Equation"],
            "problem_type": item["Type"],
        }
        for item in source
    ]
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1000 SVAMP examples, found {len(rows)}")
    write_jsonl(args.output, rows)
    manifest = {
        "dataset": "SVAMP",
        "source_url": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(rows),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = resolve_path("data/processed/svamp_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
