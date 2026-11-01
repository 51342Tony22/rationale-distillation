from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .common import gsm8k_gold, load_config, resolve_path, write_jsonl


URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def download_jsonl(split: str) -> list[dict]:
    destination = resolve_path(f"data/raw/gsm8k_{split}.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading official GSM8K {split} split...")
        urllib.request.urlretrieve(URLS[split], destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != SHA256[split]:
        raise RuntimeError(
            f"Checksum mismatch for {destination}: expected {SHA256[split]}, got {digest}. "
            "Delete the partial file and retry."
        )
    with destination.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_split(dataset: list[dict], split: str, limit: int | None) -> list[dict]:
    selected = dataset[:limit] if limit else dataset
    rows = []
    for index, item in enumerate(selected):
        rows.append(
            {
                "id": f"gsm8k_{split}_{index:05d}",
                "source_index": item.get("_source_index", index),
                "split": split,
                "question": item["question"].strip(),
                "gold_solution": item["answer"].strip(),
                "gold_answer": gsm8k_gold(item["answer"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and freeze the GSM8K splits.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--fixture", type=Path, help="Use a local JSONL fixture instead.")
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]

    if args.fixture:
        from .common import read_jsonl

        fixture = read_jsonl(args.fixture)
        train_rows = [row for row in fixture if row.get("split") == "train"]
        dev_rows = [row for row in fixture if row.get("split") == "dev"]
        test_rows = [row for row in fixture if row.get("split") == "test"]
    else:
        source_train = [
            {**row, "_source_index": index}
            for index, row in enumerate(download_jsonl("train"))
        ]
        source_test = [
            {**row, "_source_index": index}
            for index, row in enumerate(download_jsonl("test"))
        ]
        indices = list(range(len(source_train)))
        random.Random(config["seed"]).shuffle(indices)
        dev_size = args.dev_limit if args.dev_limit is not None else data_config["validation_size"]
        dev_source = [source_train[index] for index in indices[:dev_size]]
        train_source = [source_train[index] for index in indices[dev_size:]]
        train_rows = convert_split(train_source, "train", args.train_limit)
        dev_rows = convert_split(dev_source, "dev", args.dev_limit)
        test_rows = convert_split(source_test, "test", args.test_limit)

    write_jsonl(data_config["train_file"], train_rows)
    write_jsonl(data_config["dev_file"], dev_rows)
    write_jsonl(data_config["test_file"], test_rows)
    print(f"Wrote {len(train_rows)} train rows to {resolve_path(data_config['train_file'])}")
    print(f"Wrote {len(dev_rows)} dev rows to {resolve_path(data_config['dev_file'])}")
    print(f"Wrote {len(test_rows)} test rows to {resolve_path(data_config['test_file'])}")
    manifest = {
        "dataset": "openai/gsm8k",
        "source_urls": URLS if not args.fixture else None,
        "source_sha256": SHA256 if not args.fixture else None,
        "fixture": str(args.fixture) if args.fixture else None,
        "split_seed": config["seed"],
        "validation_size": len(dev_rows),
        "counts": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = resolve_path("data/processed/manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
