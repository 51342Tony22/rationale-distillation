"""Offline preflight and explicitly started local teacher feasibility trial.

Preflight uses a tiny random quantized Qwen2, never the teacher's weight tensors.
The run command is the only entry point that loads the mathematical teacher.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).absolute().parents[2]
CONFIG = Path(__file__).with_name("config.json")
OUT = ROOT / "output/local_math_teacher"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


@contextlib.contextmanager
def exclusive():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trial.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prompt_tokens(tokenizer, config, question):
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": config["system_prompt"]},
         {"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    return ids


def references(config):
    """Fetch only public metadata and the student's small tokenizer/config files."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=False)
    teacher = api.model_info(config["teacher_repo"], revision=config["teacher_revision"],
                             files_metadata=True)
    student = api.model_info(config["student_reference_repo"], files_metadata=True)
    student_dir = ROOT / "models/Qwen2.5-Math-1.5B-tokenizer-reference"
    snapshot_download(config["student_reference_repo"], revision=student.sha,
                      local_dir=student_dir, token=False, max_workers=2,
                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "config.json"])
    assets = {}
    for item in teacher.siblings:
        assets[item.rfilename] = {"bytes": item.size, "git_blob": item.blob_id}
        if item.lfs:
            assets[item.rfilename]["sha256"] = item.lfs.sha256
    save(OUT / "reference_manifest.json", {
        "created_at": stamp(), "teacher_repo": teacher.id, "teacher_revision": teacher.sha,
        "teacher_assets": assets, "upstream_teacher": config["upstream_teacher"],
        "student_repo": student.id, "student_revision": student.sha,
        "student_dir": str(student_dir),
        "student_assets": {p.name: digest(p) for p in student_dir.glob("*.json")},
        "provenance_limit": "Community conversion; upstream weight revision is not documented in its card."
    })
    print("Public reference assets saved; no inference performed.", flush=True)


def validate_files(config):
    folder = ROOT / config["model_dir"]
    manifest = json.loads((OUT / "reference_manifest.json").read_text())
    assert manifest["teacher_revision"] == config["teacher_revision"]
    for name, sha in manifest["student_assets"].items():
        assert digest(Path(manifest["student_dir"]) / name) == sha, f"Student reference changed: {name}"
    checked = {}
    for name, spec in manifest["teacher_assets"].items():
        path = folder / name
        assert path.is_file(), f"Missing asset: {path}"
        assert path.stat().st_size == spec["bytes"], f"Incomplete asset: {name}"
        sha = digest(path)
        if "sha256" in spec:
            assert sha == spec["sha256"], f"SHA256 mismatch: {name}"
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            assert blob == spec["git_blob"], f"Git blob mismatch: {name}"
        checked[name] = {"bytes": path.stat().st_size, "sha256": sha}
    weight = folder / "model.safetensors"
    assert checked[weight.name]["sha256"] == config["weight_sha256"]
    assert weight.stat().st_size == config["weight_bytes"]
    with weight.open("rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < header_len < 10_000_000
        header = json.loads(stream.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    assert set(tensors) == set(index["weight_map"]), "Tensor index mismatch"
    offsets = sorted(v["data_offsets"] for v in tensors.values())
    assert offsets[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(offsets, offsets[1:]))
    assert offsets[-1][1] + 8 + header_len == weight.stat().st_size
    actual = json.loads((folder / "config.json").read_text())
    assert actual["model_type"] == "qwen2"
    assert actual["quantization"] == {"bits": 4, "group_size": 64}
    assert actual["max_position_embeddings"] == config["context_tokens"]
    return checked, len(tensors)


def soft_probe(model, tokenizer, config, target_dir):
    """Teacher-forced logits: input position P-1 predicts target position P."""
    import mlx.core as mx
    import numpy as np
    question = "What is 1 + 2?"
    prompt = prompt_tokens(tokenizer, config, question)
    target = tokenizer.encode(r"\boxed{3}", add_special_tokens=False)
    supplied = prompt + target
    assert len(supplied) <= config["context_tokens"]
    hidden = model.model(mx.array([supplied[:-1]]))
    selected_hidden = hidden[:, len(prompt) - 1:, :]
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(selected_hidden)
    else:
        logits = model.lm_head(selected_hidden)
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    assert logits.shape == (1, len(target), model.args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())
    manifest = json.loads((OUT / "reference_manifest.json").read_text())
    student_tokenizer = json.loads((Path(manifest["student_dir"]) / "tokenizer.json").read_text())
    common_ids = sorted(student_tokenizer["model"]["vocab"].values())
    assert common_ids == list(range(len(common_ids))), "Shared vocabulary must be contiguous"
    assert all(t < len(common_ids) for t in target)
    temperatures = []
    for temperature in config["soft_probe_temperatures"]:
        scaled = logits / temperature
        full = mx.softmax(scaled, axis=-1)
        common = mx.softmax(scaled[..., :len(common_ids)], axis=-1)
        mass = mx.sum(full[..., :len(common_ids)], axis=-1)
        sum_error = mx.max(mx.abs(mx.sum(common, axis=-1) - 1.0)).item()
        assert sum_error < 1e-5
        log_probs = scaled[..., :len(common_ids)] - mx.logsumexp(
            scaled[..., :len(common_ids)], axis=-1, keepdims=True)
        nll = -mx.mean(mx.take_along_axis(log_probs, mx.array(target)[None, :, None], axis=-1))
        temperatures.append({"temperature": temperature, "normalization_max_error": sum_error,
                             "minimum_common_vocab_mass": mx.min(mass).item(),
                             "target_nll_nats": nll.item()})
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_dir / "teacher_forced_logits.npz",
                        logits=np.array(logits)[0], target_ids=np.array(target),
                        input_ids=np.array(supplied[:-1]),
                        predictor_positions=np.arange(len(prompt) - 1, len(supplied) - 1))
    return {"question": question, "supplied_target": r"\boxed{3}",
            "target_is_supplied_not_generated": True,
            "logits_shape": list(logits.shape), "finite": True,
            "shared_ordinary_vocab": len(common_ids), "temperatures": temperatures,
            "alignment": "logits[i] predicts target_ids[i]; first predictor is last prompt token",
            "limitation": "No student forward/backward, KL training, or EOS mapping tested. Shared ordinary vocabulary only."}


def preflight(config, verify_weights=True):
    from transformers import AutoTokenizer
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    start = time.perf_counter()
    checked, tensor_count = validate_files(config) if verify_weights else ({}, None)
    folder = ROOT / config["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    actual = json.loads((folder / "config.json").read_text())
    manifest = json.loads((OUT / "reference_manifest.json").read_text())
    student = json.loads((Path(manifest["student_dir"]) / "tokenizer.json").read_text())
    teacher = json.loads((folder / "tokenizer.json").read_text())
    assert teacher["model"]["vocab"] == student["model"]["vocab"], "Ordinary token mapping differs"
    prompts = []
    for case in config["cases"]:
        ids = prompt_tokens(tokenizer, config, case["question"])
        rendered = tokenizer.decode(ids)
        assert rendered.startswith("<|im_start|>system\n"), "Wrong chat-template newline"
        assert rendered.endswith("<|im_start|>assistant\n"), "Wrong assistant boundary"
        assert len(ids) + config["max_new_tokens"] <= config["context_tokens"]
        prompts.append({"id": case["id"], "prompt_tokens": len(ids)})
    assert mx.metal.is_available(), "Metal backend unavailable"
    mx.random.seed(config["seed"])
    tiny = Model(ModelArgs(model_type="qwen2", hidden_size=64, num_hidden_layers=1,
                           intermediate_size=128, num_attention_heads=4,
                           num_key_value_heads=2, rms_norm_eps=1e-6,
                           vocab_size=actual["vocab_size"], tie_word_embeddings=False,
                           max_position_embeddings=config["context_tokens"]))
    nn.quantize(tiny, group_size=64, bits=4)
    tiny.eval()
    mx.eval(tiny.parameters())
    wrapped = load_tokenizer(folder, eos_token_ids=actual["eos_token_id"])
    assert prompt_tokens(wrapped, config, config["cases"][0]["question"]) == prompt_tokens(
        tokenizer, config, config["cases"][0]["question"])
    # A tiny random model tests the same generation API without teacher inference.
    chunks = list(stream_generate(tiny, wrapped, [1, 2, 3], max_tokens=3,
                                  sampler=make_sampler(temp=0.0)))
    assert chunks and chunks[-1].finish_reason in ("stop", "length")
    with tempfile.TemporaryDirectory(prefix="qmath-preflight-") as temp:
        probe = soft_probe(tiny, tokenizer, config, Path(temp))
        # Independently compare the batched target logits with autoregressive KV-cache
        # predictions. This catches a one-token alignment error in offline KD export.
        import numpy as np
        saved = np.load(Path(temp) / "teacher_forced_logits.npz")
        target = saved["target_ids"].tolist()
        ids = prompt_tokens(tokenizer, config, probe["question"])
        cache = make_prompt_cache(tiny)
        rows = [tiny(mx.array([ids]), cache=cache)[:, -1:, :]]
        for token in target[:-1]:
            rows.append(tiny(mx.array([[token]]), cache=cache))
        sequential = mx.concatenate(rows, axis=1).astype(mx.float32)
        batched = mx.array(saved["logits"])[None]
        delta = mx.max(mx.abs(sequential - batched)).item()
        shifted_gap = min(mx.max(mx.abs(sequential[:, 1:] - batched[:, :-1])).item(),
                          mx.max(mx.abs(sequential[:, :-1] - batched[:, 1:])).item())
        probability_delta = mx.max(mx.abs(mx.softmax(sequential, axis=-1)
                                         - mx.softmax(batched, axis=-1))).item()
        print(json.dumps({"quantized_cache_diagnostic": {"max_logit_delta": delta,
                         "min_shifted_position_gap": shifted_gap,
                         "max_probability_delta": probability_delta}}), flush=True)
        # Quantized GEMV/GEMM kernels need not be bit-identical. Check probability
        # agreement and explicitly show that either one-token shift is much worse.
        assert delta < 1e-2 and probability_delta < 1e-6, "Cached logits disagree numerically"
        assert shifted_gap > 100 * max(delta, 1e-8), "Prediction positions may be shifted"
        probe["cached_vs_batched_logits_max_abs_error"] = delta
        probe["cached_vs_batched_probability_max_abs_error"] = probability_delta
        probe["one_token_shift_minimum_max_abs_error"] = shifted_gap
    versions = {p: importlib.metadata.version(p) for p in
                ["mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy"]}
    report = {"created_at": stamp(), "status": ("ready_for_explicit_teacher_run" if verify_weights
                                               else "interfaces_passed_weights_pending"),
              "teacher_inference_performed": False, "config_sha256": digest(CONFIG),
              "script_sha256": digest(Path(__file__)),
              "reference_manifest_sha256": digest(OUT / "reference_manifest.json"),
              "platform": platform.platform(), "device": mx.device_info(),
              "versions": versions, "model_files": checked, "tensor_count": tensor_count,
              "prompt_checks": prompts, "shared_ordinary_vocab": len(student["model"]["vocab"]),
              "teacher_output_dim": actual["vocab_size"],
              "student_output_dim": json.loads((Path(manifest["student_dir"]) / "config.json").read_text())["vocab_size"],
              "teacher_eos_token_id": actual["eos_token_id"],
              "student_eos_token_id": json.loads((Path(manifest["student_dir"]) / "config.json").read_text())["eos_token_id"],
              "tiny_random_quantized_model": {"generation_api": "passed", "soft_probe": probe},
              "peak_mlx_memory_bytes": mx.get_peak_memory(),
              "disk_free_bytes": shutil.disk_usage(ROOT).free,
              "seconds": time.perf_counter() - start,
              "limits": ["Real teacher loading, quality, speed and peak memory remain unmeasured.",
                         "Quantized probabilities differ from the original BF16 teacher.",
                         "Student attention extraction and end-to-end distillation are separate work."]}
    save(OUT / ("preflight.json" if verify_weights else "interface_preflight.json"), report)
    print(json.dumps({k: report[k] for k in ["status", "teacher_inference_performed", "seconds"]}), flush=True)


def run(config):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    pre = json.loads((OUT / "preflight.json").read_text())
    assert pre["status"] == "ready_for_explicit_teacher_run"
    assert pre["config_sha256"] == digest(CONFIG), "Config changed; rerun preflight"
    assert pre["script_sha256"] == digest(Path(__file__)), "Script changed; rerun preflight"
    assert pre["reference_manifest_sha256"] == digest(OUT / "reference_manifest.json"), "References changed"
    assert pre["versions"] == {p: importlib.metadata.version(p) for p in pre["versions"]}, "Runtime changed"
    print("Rechecking local assets before teacher load...", flush=True)
    validate_files(config)
    # Catch other known MLX jobs; do not terminate them or start a second one.
    scan = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True, check=True)
    for line in scan.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        if re.match(r"^\S*(?:/Python|python[\d.]*)\s", fields[1]) and any(
            key in fields[1] for key in ["fyp_distill", "mlx_lm", "teacher_trial.py run"]):
            raise RuntimeError("Another model job is active; wait for it to finish.")
    job = OUT / ("run_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    job.mkdir(exist_ok=False)
    save(job / "config.json", config)
    save(job / "status.json", {"stage": "loading", "created_at": stamp(),
                               "preflight_sha256": digest(OUT / "preflight.json")})
    start = time.perf_counter()
    try:
        mx.random.seed(config["seed"])
        # Soft cap well below physical memory; a failure is logged, never auto-retried.
        mx.set_memory_limit(12 * 1024**3)
        mx.set_cache_limit(1024**3)
        model, tokenizer = load(str(ROOT / config["model_dir"]))
        model.eval()
        loaded = time.perf_counter() - start
        predictions = []
        for case in config["cases"]:
            ids = prompt_tokens(tokenizer, config, case["question"])
            assert len(ids) + config["max_new_tokens"] <= config["context_tokens"]
            text = ""
            tick = time.perf_counter()
            last = None
            for part in stream_generate(model, tokenizer, ids,
                                        max_tokens=config["max_new_tokens"],
                                        sampler=make_sampler(temp=0.0)):
                text += part.text
                last = part
            assert last is not None
            boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
            exact = len(boxes) == 1 and boxes[0].strip() == case["expected"]
            row = {**case, "output": text, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "seconds": time.perf_counter() - tick, "generated_tokens": last.generation_tokens,
                   "tokens_per_second": last.generation_tps, "finish_reason": last.finish_reason,
                   "single_box_exact_match": exact, "full_math_review": "pending"}
            with (job / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            predictions.append(row)
            save(job / "status.json", {"stage": "generating", "completed": len(predictions), "total": len(config["cases"])})
            print(f'{case["id"]}: saved ({len(predictions)}/{len(config["cases"])}), {row["seconds"]:.1f}s', flush=True)
        probe = soft_probe(model, tokenizer, config, job)
        save(job / "soft_probe.json", probe)
        save(job / "status.json", {"stage": "needs_full_math_review", "completed": len(predictions),
                                   "load_seconds": loaded, "total_seconds": time.perf_counter() - start,
                                   "peak_mlx_memory_bytes": mx.get_peak_memory(),
                                   "single_box_exact_matches": sum(p["single_box_exact_match"] for p in predictions),
                                   "teacher_revision": config["teacher_revision"], "weight_bits": 4,
                                   "soft_probe": "passed", "api_generation_cost": 0})
        print(f"Saved trial to {job}; full mathematical review is still required.", flush=True)
    except Exception as exc:
        failure = {"created_at": stamp(), "error_type": type(exc).__name__,
                   "message": str(exc), "seconds": time.perf_counter() - start}
        save(job / "failure.json", failure)
        save(job / "status.json", {"stage": "failed", **failure})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["references", "interfaces", "preflight", "run"])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    with exclusive():
        {"references": references, "interfaces": lambda c: preflight(c, verify_weights=False),
         "preflight": preflight, "run": run}[args.stage](config)


if __name__ == "__main__":
    main()
