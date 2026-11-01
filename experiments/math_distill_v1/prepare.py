"""Fetch pinned public assets, verify them, and freeze disjoint source splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).absolute().parents[2]
CONFIG = Path(__file__).with_name("config.json")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def freeze(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, f"Frozen asset differs: {path}"
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(path)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def normalized(question):
    return " ".join(re.findall(r"[a-z0-9]+", question.lower()))


def prepare():
    from huggingface_hub import HfApi
    c = json.loads(CONFIG.read_text())
    root = ROOT / c["data_dir"]
    source = root / "source"
    sources = {}
    # The official test is fetched solely for a separate lexical-overlap audit.
    # No test generation or score is permitted by this phase's config.
    for split, expected in [("train", 7473), ("test", 1319)]:
        name = f"{split}.jsonl"
        url = (f'https://raw.githubusercontent.com/{c["dataset_repo"]}/'
               f'{c["dataset_revision"]}/grade_school_math/data/{name}')
        path = source / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                freeze(path, response.read())
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == expected
        assert all(set(row) == {"question", "answer"} for row in rows)
        sources[split] = {"url": url, "sha256": sha(path), "count": len(rows)}
    train = [json.loads(line) for line in (source / "train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (source / "test.jsonl").read_text().splitlines()]
    order = list(range(len(train)))
    random.Random(c["seed"]).shuffle(order)
    parts = {"dev": order[:c["dev_count"]],
             "train": order[c["dev_count"]:c["dev_count"] + c["train_count"]]}
    assert not set(parts["dev"]) & set(parts["train"])
    stats = {}
    for split, indexes in parts.items():
        rows = []
        for index in indexes:
            row = train[index]
            assert row["answer"].count("####") == 1
            rationale, final = row["answer"].split("####")
            rows.append({"source_id": f"gsm8k:train:{index}", "source_index": index,
                         "question": row["question"], "source_answer": row["answer"],
                         "reference_rationale": re.sub(r"<<[^>]*>>", "", rationale).strip(),
                         "gold": final.strip(),
                         "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest()})
        path = root / f"{split}.jsonl"
        freeze(path, ("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)).encode())
        stats[split] = {"count": len(rows), "sha256": sha(path),
                        "source_ids": [r["source_id"] for r in rows]}
    normalized_sets = {s: {normalized(train[i]["question"]) for i in idx} for s, idx in parts.items()}
    test_questions = {normalized(r["question"]) for r in test}
    overlap = {"train_dev": len(normalized_sets["train"] & normalized_sets["dev"]),
               "train_test": len(normalized_sets["train"] & test_questions),
               "dev_test": len(normalized_sets["dev"] & test_questions)}
    # Record, never silently substitute questions if source contains a duplicate.
    freeze(root / "source_manifest.json", encoded({"dataset_repo": c["dataset_repo"],
           "dataset_revision": c["dataset_revision"], "seed": c["seed"],
           "sources": sources, "splits": stats, "normalized_exact_overlap": overlap,
           "split_selection": "Python Random(seed).shuffle original train indices; first 200 dev, next 500 train",
           "test_usage": "Question-only lexical overlap; no model generation or score",
           "limitation": "Exact lexical audit does not rule out templates, paraphrases or pretraining contamination."}))
    # Save remote student file expectations independently of local cache metadata.
    info = HfApi(token=False).model_info(c["student_repo"], revision=c["student_revision"], files_metadata=True)
    files = {}
    for item in info.siblings:
        files[item.rfilename] = {"bytes": item.size, "git_blob": item.blob_id}
        if item.lfs:
            files[item.rfilename]["sha256"] = item.lfs.sha256
    freeze(root / "student_manifest.json", encoded({"repo": info.id, "revision": info.sha, "files": files}))
    print(json.dumps({"stage": "data_splits_frozen", "dev": c["dev_count"],
                      "train": c["train_count"], "overlap": overlap}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    prepare()
