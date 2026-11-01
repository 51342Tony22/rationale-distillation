"""Freeze original MATH train-only subsets from the pinned PRM800K export."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict

from .common import read_jsonl, resolve_path
from .refinement import question_key, stable_seed, write_frozen

SOURCE_URL = "https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/train.jsonl"
SOURCE_SHA256 = "90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076"


def stratified_take(rows, size, seed):
    if not 0 <= size <= len(rows):
        raise ValueError("Requested split size exceeds eligible source")
    buckets = defaultdict(list)
    for row in rows:
        buckets[(str(row["subject"]), str(row["level"]))].append(row)
    quotas = {key: math.floor(size * len(bucket) / len(rows)) for key, bucket in buckets.items()}
    ranking = sorted(buckets, key=lambda key: (-(size * len(buckets[key]) / len(rows) - quotas[key]), key))
    for key in ranking[:size - sum(quotas.values())]:
        quotas[key] += 1
    selected, remaining = [], []
    for key, bucket in sorted(buckets.items()):
        bucket = sorted(bucket, key=lambda row: row["id"])
        random.Random(stable_seed(seed, key)).shuffle(bucket)
        selected.extend(bucket[:quotas[key]])
        remaining.extend(bucket[quotas[key]:])
    return selected, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="data/refinement_math/processed")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = resolve_path(args.source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source differs from pinned PRM800K export")
    source_rows = read_jsonl(source)
    test = read_jsonl("data/processed/math500_test.jsonl")
    if len(test) != 500:
        raise ValueError("Prepare the checked MATH500 split first")
    test_ids = {r["source_id"] for r in test}
    test_questions = {question_key(r["question"]) for r in test}
    eligible, seen, excluded = [], set(), defaultdict(int)
    for row in source_rows:
        source_id = row["unique_id"]
        if not source_id.startswith("train/"):
            excluded["not_original_math_train"] += 1
            continue
        key = question_key(row["problem"])
        if source_id in test_ids or key in test_questions:
            excluded["math500_overlap"] += 1
            continue
        if key in seen:
            excluded["duplicate_question"] += 1
            continue
        seen.add(key)
        eligible.append({"id": "math_" + source_id.replace("/", "_"), "source_id": source_id,
                         "question": row["problem"], "gold_solution": row["solution"],
                         "gold_answer": row["answer"], "subject": row["subject"], "level": row["level"]})
    dev, remainder = stratified_take(eligible, args.dev_size, args.seed)
    train, _ = stratified_take(remainder, args.train_size, args.seed + 1)
    pilot, _ = stratified_take(train, min(100, len(train)), args.seed + 2)
    root = resolve_path(args.output_dir)
    for name, rows in (("train", train), ("dev", dev), ("pilot", pilot)):
        write_frozen(root / f"math_{name}.jsonl", [{**r, "split": "dev" if name == "dev" else "train"} for r in rows])
    manifest = {"source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_rows": len(source_rows), "eligible_original_train": len(eligible),
                "excluded": dict(excluded), "train": len(train), "dev": len(dev), "pilot": len(pilot),
                "seed": args.seed, "status": "problem_splits_only; R1 rationales not generated"}
    write_frozen(root / "manifest.jsonl", [manifest])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
