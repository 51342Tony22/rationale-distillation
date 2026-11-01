"""Frozen development baseline runner; no training or formal-test entry point."""
from __future__ import annotations

import argparse
import contextlib
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
HERE = Path(__file__).absolute().parent
CONFIG = HERE / "config.json"
NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def freeze(path, obj):
    content = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == content, f"Signature changed: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


@contextlib.contextmanager
def gpu_lock(output):
    with (ROOT / "output/local_math_gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scan = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True,
                              text=True, check=True)
        for line in scan.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
                continue
            if re.match(r"^\S*(?:/Python|python[\d.]*)\s", parts[1]) and any(
                name in parts[1] for name in ["fyp_distill", "mlx_lm", "teacher_trial.py run", "baseline.py evaluate", "attention_probe.py"]):
                raise RuntimeError("Another model process is active; no second GPU job started")
        yield


def number(text):
    text = text.strip().replace("−", "-").replace(",", "")
    text = re.sub(r"^\$", "", text)
    if not re.fullmatch(NUM, text):
        return None
    try:
        if "/" in text:
            a, b = text.split("/")
            return Decimal(a.strip()) / Decimal(b.strip())
        return Decimal(text)
    except (InvalidOperation, ZeroDivisionError):
        return None


def parse_answer(text):
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if "\\boxed" in text:
        if not boxes:
            return None, "malformed_box"
        values = [number(s) for s in boxes]
        if any(v is None for v in values) or len(set(values)) != 1:
            return None, "ambiguous_box"
        # Duplicate boxes of the same answer are allowed for math EM, not strict format EM.
        return values[-1], "boxed"
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        match = re.match(rf"^\$?({NUM})(?:\s|$|[.!])", tail)
        return (number(match.group(1)), "hash_final") if match else (None, "malformed_hash")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "empty"
    line = lines[-1]
    marker = re.search(r"(?:final\s+answer\s*(?:is|:)?|answer\s+is|therefore|thus|so\b)", line, re.I)
    candidate = line[marker.end():] if marker else line
    candidate = candidate.replace(r"\(", "").replace(r"\)", "")
    candidate = candidate.replace(r"\[", "").replace(r"\]", "")
    # An answer must terminate the final non-empty line, apart from units/punctuation.
    # No arbitrary expression evaluation or last-number search across the whole output.
    match = re.search(rf"(?<![\w.])\$?({NUM})(?:\s*(?:[A-Za-z]+(?:\^?\d)?|%)){{0,3}}[.!]?$", candidate.strip())
    if not match:
        return None, "no_clear_final_number"
    return number(match.group(1)), "final_marker" if marker else "final_numeric_line"


def score(text, gold):
    parsed, rule = parse_answer(text)
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    strict = False
    if len(boxes) == 1 and number(boxes[0]) is not None:
        strict = number(boxes[0]) == number(gold)
    elif "\\boxed" not in text and "####" in text:
        strict = parsed is not None and rule == "hash_final" and parsed == number(gold)
    return {"parsed_answer": str(parsed) if parsed is not None else None,
            "parse_rule": rule, "invalid": parsed is None,
            "correct": parsed is not None and parsed == number(gold),
            "strict_format_correct": strict}


def selfcheck():
    examples = [
        (r"A=15*10=150. Therefore, \boxed{150}", "150", True, False),
        ("Thus, the total cost is $1,250.", "1250", True, False),
        (r"\boxed{16} then \boxed{18}", "18", False, True),
        (r"\boxed{16} and \boxed{16}", "16", True, False),
        ("#### -3/2", "-1.5", True, False),
        ("Therefore the answer is not known.", "3", False, True),
        ("We have 3 +", "3", False, True),
        ("The answer is 18 students.", "18", True, False),
        ("8 + 3x = 35", "35", True, False),
        (r"\boxed{3", "3", False, True),
    ]
    for text, gold, correct, invalid in examples:
        result = score(text, gold)
        assert result["correct"] == correct and result["invalid"] == invalid, (text, result)
    print("Scorer checks passed; terminal-number EM can accept an unfinished equation, requiring separate completion/math audit.")


def verify_student(config):
    folder = ROOT / config["student_dir"]
    manifest = json.loads((ROOT / config["data_dir"] / "student_manifest.json").read_text())
    assert manifest["revision"] == config["student_revision"]
    checked = {}
    for name, expected in manifest["files"].items():
        path = folder / name
        assert path.is_file() and path.stat().st_size == expected["bytes"], f"Missing/incomplete {name}"
        digest = sha(path)
        if "sha256" in expected:
            assert digest == expected["sha256"], f"Weight SHA256 mismatch: {name}"
        else:
            data = path.read_bytes()
            assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected["git_blob"]
        checked[name] = digest
    return checked


def summary(rows):
    total = len(rows)
    return {"completed": total, "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / total if total else None,
            "strict_format_correct": sum(r["strict_format_correct"] for r in rows),
            "invalid": sum(r["invalid"] for r in rows),
            "length_stops": sum(r["finish_reason"] == "length" for r in rows),
            "next_question_stops": sum(r["finish_reason"] == "next_question" for r in rows),
            "generated_tokens": sum(r["generation_tokens"] for r in rows),
            "generation_seconds": sum(r["seconds"] for r in rows),
            "mean_seconds": sum(r["seconds"] for r in rows) / total if total else None,
            "peak_mlx_memory_bytes": max((r["peak_mlx_memory_bytes"] for r in rows), default=0),
            "parse_rule_counts": {key: sum(r["parse_rule"] == key for r in rows)
                                  for key in sorted({r["parse_rule"] for r in rows})},
            "full_math_review": "pending; numeric EM alone does not assess reasoning"}


def evaluate(config, condition):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    data = ROOT / config["data_dir"]
    out = ROOT / config["output_dir"] / "dev_baseline" / condition
    out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.jsonl"
    sources = [json.loads(line) for line in (data / "dev.jsonl").read_text().splitlines()]
    assert len(sources) == config["dev_count"] and not config["formal_test_evaluation_enabled"]
    with gpu_lock(out):
        if condition == "student_base":
            checked = verify_student(config)
            folder = ROOT / config["student_dir"]
        else:
            sys.path.insert(0, str(ROOT / "experiments/local_math_teacher"))
            import teacher_trial
            teacher_cfg = json.loads(teacher_trial.CONFIG.read_text())
            checked, _ = teacher_trial.validate_files(teacher_cfg)
            folder = ROOT / config["teacher_dir"]
        versions = {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "transformers"]}
        signature = {"config_sha256": sha(CONFIG), "runner_sha256": sha(Path(__file__)),
                     "dev_sha256": sha(data / "dev.jsonl"), "condition": condition,
                     "versions": versions, "model_files": checked,
                     "source_ids": [r["source_id"] for r in sources]}
        freeze(out / "signature.json", signature)
        old = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        assert len(old) <= len(sources)
        for row, source in zip(old, sources):
            assert row["source_id"] == source["source_id"] and row["question_sha256"] == source["question_sha256"]
            assert hashlib.sha256(row["output"].encode()).hexdigest() == row["output_sha256"]
            expected_score = score(row["scoring_text"], source["gold"])
            assert all(row[k] == v for k, v in expected_score.items()), "Stored score changed"
        if len(old) == len(sources):
            print(f"{condition}: all {len(old)} predictions already exist, reused after validation.", flush=True)
            save(out / "summary.json", summary(old))
            return
        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(1024**3)
        mx.random.seed(config["seed"])
        start = time.perf_counter()
        model, tokenizer = load(str(folder), tokenizer_config={"local_files_only": True, "trust_remote_code": False})
        model.eval()
        save(out / "load.json", {"seconds": time.perf_counter() - start,
                                 "resumed_after": len(old), "pid": os.getpid(),
                                 "weight_dtype": str(model.model.embed_tokens.weight.dtype),
                                 "precision": config["student_precision"] if condition == "student_base" else config["teacher_precision"]})
        print(f"{condition}: loaded, starting at {len(old)}/{len(sources)}", flush=True)
        for source in sources[len(old):]:
            tick = time.perf_counter()
            if condition == "student_base":
                prompt = config["student_prompt"].format(question=source["question"])
                ids = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": config["teacher_system"]},
                     {"role": "user", "content": source["question"]}],
                    tokenize=True, return_dict=False, add_generation_prompt=True)
                if hasattr(ids, "keys"):
                    ids = ids["input_ids"]
                assert isinstance(ids, list)
            assert len(ids) + config["max_new_tokens"] <= config["context_tokens"], "Would truncate context"
            text, scored, last, boundary = "", "", None, None
            for part in stream_generate(model, tokenizer, ids, max_tokens=config["max_new_tokens"],
                                        sampler=make_sampler(temp=0.0)):
                text += part.text
                last = part
                if condition == "student_base" and config["stop_at_next_question_student"]:
                    boundary = re.search(r"(?:^|\n)\s*(?:###\s*)?Question:", text)
                    if boundary:
                        break
            assert last is not None
            scored = text[:boundary.start()] if boundary else text
            row = {"source_id": source["source_id"], "question_sha256": source["question_sha256"],
                   "gold": source["gold"], "output": text, "scoring_text": scored,
                   "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "prompt_tokens": len(ids), "generation_tokens": last.generation_tokens,
                   "seconds": time.perf_counter() - tick, "decode_tokens_per_second": last.generation_tps,
                   "finish_reason": "next_question" if boundary else last.finish_reason,
                   "peak_mlx_memory_bytes": mx.get_peak_memory(), **score(scored, source["gold"])}
            with path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            old.append(row)
            save(out / "progress.json", {"stage": "evaluating", **summary(old), "total": len(sources)})
            if len(old) % config["report_interval"] == 0 or len(old) <= 1:
                print(f'{condition}: {len(old)}/{len(sources)}, mean {summary(old)["mean_seconds"]:.1f}s/question', flush=True)
        final = summary(old)
        final["signature_sha256"] = sha(out / "signature.json")
        final["predictions_sha256"] = sha(path)
        save(out / "summary.json", final)
        save(out / "progress.json", {"stage": "generation_complete_needs_audit", **final})
        print(f"{condition}: 200/200 saved. Scores require output audit.", flush=True)


def pipeline(config):
    out = ROOT / config["output_dir"]
    for condition in ["student_base", "teacher_reference"]:
        save(out / "pipeline_status.json", {"stage": "dev_baselines", "active": condition, "pid": os.getpid()})
        subprocess.run([sys.executable, "-u", str(Path(__file__).absolute()), "evaluate", "--condition", condition], check=True)
    save(out / "pipeline_status.json", {"stage": "needs_dev_output_audit_and_training_protocol", "completed_conditions": 2,
                                       "training_started": False, "formal_test_evaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["selfcheck", "verify-student", "evaluate", "pipeline"])
    parser.add_argument("--condition", choices=["student_base", "teacher_reference"])
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    out = ROOT / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if args.background:
        assert args.stage == "pipeline"
        # Reserve the pipeline lock before spawning to avoid a duplicate launch.
        with (out / "pipeline.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Transfer the held file description, avoiding an unlock/relock race.
            with (out / "baseline_pipeline.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).absolute()), "pipeline",
                                          "--lock-fd", str(lock.fileno())],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
            save(out / "launch.json", {"pid": child.pid, "command": "pipeline", "training_started": False})
            print(json.dumps({"pid": child.pid, "log": str(out / "baseline_pipeline.log")}), flush=True)
        return
    if args.stage == "selfcheck":
        selfcheck()
    elif args.stage == "verify-student":
        checked = verify_student(config)
        save(out / "student_asset_verification.json", {"status": "passed", "files": checked})
        print("Official student assets verified.", flush=True)
    elif args.stage == "evaluate":
        assert args.condition
        evaluate(config, args.condition)
    else:
        lock_handle = os.fdopen(args.lock_fd, "a+") if args.lock_fd is not None else (out / "pipeline.lock").open("a+")
        with lock_handle as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                pipeline(config)
            except Exception as error:
                save(out / "pipeline_status.json", {"stage": "failed", "error_type": type(error).__name__,
                                                    "message": str(error), "training_started": False})
                raise


if __name__ == "__main__":
    main()
