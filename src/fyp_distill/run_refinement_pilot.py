"""Run the reviewed pilot through screening and a capped teacher hint queue.

Stops before hint verification/training so explanations can be reviewed. The
background worker owns a log, status file, and a per-run process lock.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime

from .common import ROOT, load_config, read_jsonl, resolve_path, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    root = resolve_path(config["refinement"]["root"])
    bases = read_jsonl(root / "bases.jsonl")
    if not bases or not all(all(row.get("review", {}).get(key) is True for key in
                               ("correct_steps", "prefixes_do_not_reveal_answer")) for row in bases):
        raise SystemExit("Only reviewed original steps/prefixes may enter this pilot")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "run_status.jsonl"
    log_path = root / "pipeline.log"
    if args.background:
        previous = read_jsonl(status_path)
        if previous and previous[0].get("state") == "running":
            try:
                os.kill(previous[0]["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                print(json.dumps({"state": "already_running", "pid": previous[0]["pid"], "log": str(log_path)}))
                return
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "fyp_distill.run_refinement_pilot", "--config", str(config_path)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        write_jsonl(status_path, [{"state": "running", "stage": "starting", "pid": process.pid,
                                  "updated_at": datetime.now(UTC).isoformat(), "log": str(log_path)}])
        # Sleep inhibition is attached only to this worker and ends with it.
        try:
            subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except FileNotFoundError:
            pass
        print(json.dumps({"state": "started", "pid": process.pid, "questions": len(bases),
                          "status": str(status_path), "log": str(log_path)}))
        return

    import fcntl
    PROTOCOL_VERSION = config.get("evaluation", {}).get("answer_protocol", "first_final_line_v2")
    if PROTOCOL_VERSION not in ("first_final_line_v2", "first_answer_v3"):
        raise ValueError("Unsupported pilot answer protocol")

    started = datetime.now(UTC).isoformat()

    def status(state, stage, **extra):
        record = {"state": state, "stage": stage, "pid": os.getpid(), "questions": len(bases),
                  "config": str(config_path), "answer_protocol": PROTOCOL_VERSION,
                  "started_at": started, "updated_at": datetime.now(UTC).isoformat(), **extra}
        write_jsonl(status_path, [record])
        print(json.dumps(record), flush=True)

    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        commands = [
            ("student_screening", [sys.executable, "-u", "-m", "fyp_distill.diagnose_refinement_mlx",
                                   "screen", "--config", str(config_path)]),
            ("prepare_hint_requests", [str(ROOT / ".venv/bin/python"), "-u", "-m", "fyp_distill.refinement",
                                       "requests", "--config", str(config_path)]),
            ("teacher_hint_generation", [str(ROOT / ".venv/bin/python"), "-u", "-m",
                                         "fyp_distill.request_refinement_hints", "--config", str(config_path),
                                         "--limit", str(len(bases) * (config["refinement"]["local_candidates"] + 2))]),
        ]
        for stage, command in commands:
            status("running", stage)
            result = subprocess.run(command, cwd=ROOT, env=env)
            if result.returncode:
                status("failed", stage, returncode=result.returncode)
                raise SystemExit(result.returncode)
        requests = read_jsonl(root / "teacher_requests.jsonl")
        hints = read_jsonl(root / "hints.jsonl")
        request_ids = {row["id"] for row in requests}
        hint_ids = {row["id"] for row in hints}
        ledger = read_jsonl(config["refinement"]["budget_ledger"])
        trials = [t for row in read_jsonl(root / "screening.jsonl")
                  for values in row["boundaries"].values() for t in values]
        status("needs_explanation_review" if request_ids <= hint_ids else "needs_response_repair",
               "hint_generation_finished", hints=len(hints), expected_hints=len(requests),
               estimated_or_reserved_spend_cny=sum(row["charged_cny"] for row in ledger),
               rollouts=len(trials), total_rollout_seconds=sum(t["elapsed_seconds"] for t in trials),
               missing_hint_ids=sorted(request_ids - hint_ids))


if __name__ == "__main__":
    main()
