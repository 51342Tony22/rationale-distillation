"""One sequential, locked training/evaluation pipeline after target audit."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import baseline
from pilot_common import C, HERE, OUT, append


def pipeline():
    assert (OUT / "training_protocol.json").is_file(), "Build audited common targets first"
    preflight = json.loads((OUT / "cpu_training_preflight.json").read_text())
    assert preflight["status"] == "passed"
    assert preflight["training_runner_sha256"] == baseline.sha(HERE / "pilot_train.py"), "Repeat CPU preflight after runner changes"
    for stage in ["train", "evaluate"]:
        for name in C["conditions"]:
            status = {"stage": stage, "condition": name, "pid": os.getpid()}
            baseline.save(OUT / "pipeline_status.json", status)
            append(OUT / "stage_events.jsonl", status)
            subprocess.run([sys.executable, "-u", str(HERE / "pilot_train.py"), stage, "--condition", name], check=True)
    subprocess.run([sys.executable, str(HERE / "pilot_report.py")], check=True)
    baseline.save(OUT / "pipeline_status.json", {"stage": "needs_final_full_text_audit", "trained_conditions": 3,
                                                "formal_test_evaluated": False, "pid": os.getpid()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (OUT / "pipeline.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (OUT / "pipeline.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).absolute()), "--lock-fd", str(lock.fileno())],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),), start_new_session=True)
            baseline.save(OUT / "launch.json", {"pid": process.pid, "log": str(OUT / "pipeline.log")})
            print(json.dumps({"pid": process.pid, "log": str(OUT / "pipeline.log")}))
    else:
        handle = os.fdopen(args.lock_fd, "a+") if args.lock_fd is not None else (OUT / "pipeline.lock").open("a+")
        with handle as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                pipeline()
            except BaseException as exc:
                baseline.save(OUT / "pipeline_status.json", {"stage": "failed", "error_type": type(exc).__name__, "message": str(exc)})
                raise
