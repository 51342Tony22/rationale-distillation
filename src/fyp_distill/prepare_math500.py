from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime

from .common import resolve_path, write_jsonl


URL = (
    "https://media.githubusercontent.com/media/openai/prm800k/"
    "7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/test.jsonl"
)
SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and freeze OpenAI's MATH-500 split.")
    parser.add_argument("--output", default="data/processed/math500_test.jsonl")
    parser.add_argument("--source", help="Optional already-downloaded official JSONL file.")
    args = parser.parse_args()

    raw_path = resolve_path(args.source or "data/raw/math500_test.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        print("Downloading OpenAI PRM800K MATH-500 split...")
        urllib.request.urlretrieve(URL, raw_path)
    raw = raw_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"MATH-500 checksum mismatch: expected {SHA256}, got {digest}")

    source = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(source) != 500:
        raise RuntimeError(f"Expected 500 MATH-500 examples, found {len(source)}")
    rows = [
        {
            "id": f"math500_{index:04d}",
            "split": "math500",
            "source_id": item["unique_id"],
            "question": item["problem"].strip(),
            "gold_solution": item["solution"].strip(),
            "gold_answer": item["answer"].strip(),
            "subject": item["subject"],
            "level": item["level"],
        }
        for index, item in enumerate(source)
    ]
    write_jsonl(args.output, rows)
    manifest = {
        "dataset": "OpenAI PRM800K MATH-500",
        "source_url": URL,
        "source_sha256": digest,
        "count": len(rows),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = resolve_path("data/processed/math500_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
