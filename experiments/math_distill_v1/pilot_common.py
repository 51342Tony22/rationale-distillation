"""Shared immutable inputs and durable append-only pilot records."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import baseline

ROOT = baseline.ROOT
HERE = Path(__file__).absolute().parent
CONFIG = HERE / "pilot_config.json"
C = json.loads(CONFIG.read_text())
BASE = json.loads(baseline.CONFIG.read_text())
OUT = ROOT / C["output_dir"]
DATA = ROOT / C["data_dir"]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sources(split="train"):
    manifest = json.loads((DATA / "manifest.json").read_text())
    path = DATA / f"{split}.json"
    assert baseline.sha(path) == manifest["splits"][split]["frozen_rows_sha256"]
    result = json.loads(path.read_text())
    assert [r["source_id"] for r in result] == manifest["splits"][split]["source_ids"]
    return result


def sign(stage, files, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [CONFIG, baseline.CONFIG, DATA / "manifest.json", Path(__file__), *files]
    value = {"stage": stage, "inputs": {str(p.relative_to(ROOT)): baseline.sha(p) for p in inputs},
             "versions": {name: importlib.metadata.version(name) for name in ["mlx", "mlx-lm", "transformers"]},
             "extra": extra or {}}
    baseline.freeze(OUT / f"{stage}_signature.json", value)


@contextlib.contextmanager
def gpu(stage, wait=False):
    """Wait without loading models; retain the same global exclusive lock."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "output/local_math_gpu.lock").open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise
                baseline.save(OUT / "material_status.json", {"stage": "waiting_existing_gpu_job", "next": stage,
                                                            "pid": os.getpid(), "model_loaded": False})
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reviewed(name, generated, required="accepted"):
    audits = rows(OUT / f"{name}_reviews.jsonl")
    assert len({r["source_id"] for r in audits}) == len(audits), "Duplicate human audit"
    lookup = {r["source_id"]: r for r in audits}
    for row in generated:
        assert row["source_id"] in lookup, f'Missing full-text audit: {row["source_id"]}'
        audit = lookup[row["source_id"]]
        assert audit["output_sha256"] == row["output_sha256"]
        assert isinstance(audit[required], bool) and audit.get("reason")
    return lookup
