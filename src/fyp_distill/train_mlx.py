from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime

import yaml

from .common import (
    artifact_dir,
    configured_variants,
    load_config,
    read_jsonl,
    resolve_path,
    write_jsonl,
)


def training_iterations(rows: int, batch_size: int, epochs: int, accumulation: int) -> int:
    micro_batches = math.ceil(rows / batch_size) * epochs
    return max(accumulation, micro_batches - micro_batches % accumulation)


def supervision_signature(rows: list[dict]) -> str:
    """Condition labels alone cannot make identical supervision a new treatment."""
    content = [(r["source_id"], r["prompt"], r["completion"]) for r in rows]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX QLoRA fine-tuning for one condition.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--iters", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-from", help="Reuse a completed identical-supervision condition explicitly")
    args = parser.parse_args()

    config = load_config(args.config)
    variants = configured_variants(config)
    if args.variant not in variants:
        raise SystemExit(f"Variant {args.variant!r} is not enabled in {args.config}.")
    settings = config["mlx_training"]
    model = args.model or settings["model"]
    source_path = resolve_path(config["data"]["variants_dir"]) / f"{args.variant}.jsonl"
    rows = read_jsonl(source_path)
    if not rows:
        raise SystemExit(f"No training rows found at {source_path}.")
    training_data_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    # Refinement targets are longer than the old GSM8K targets. Never silently
    # train on truncated reasoning; mirror MLX CompletionsDataset tokenization.
    if settings.get("reject_truncation"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolve_path(model), local_files_only=True)
        too_long = []
        for row in rows:
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]},
                 {"role": "assistant", "content": row["completion"]}], return_dict=False,
            )
            if len(token_ids) > settings["max_length"]:
                too_long.append(row["source_id"])
        if too_long:
            raise SystemExit(f"Training would truncate {len(too_long)} rows; audit all groups first.")

    data_dir = artifact_dir(config, "mlx_data_dir", "data/mlx") / args.variant
    write_jsonl(
        data_dir / "train.jsonl",
        ({"prompt": row["prompt"], "completion": row["completion"]} for row in rows),
    )
    output_dir = artifact_dir(config, "mlx_output_dir", "outputs_mlx") / args.variant / f"seed_{args.seed}"
    adapter_dir = output_dir / "adapter"
    adapter_file = adapter_dir / "adapters.safetensors"
    metadata_path = output_dir / "run_metadata.json"
    if adapter_file.exists() and metadata_path.exists() and not args.force:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "base_model": model,
            "batch_size": settings["batch_size"],
            "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
            "training_data_sha256": training_data_sha256,
        }
        if settings.get("reject_truncation"):
            expected["training_signature"] = hashlib.sha256(json.dumps(
                {"settings": settings, "training": config["training"], "seed": args.seed,
                 "iters_override": args.iters}, sort_keys=True,
            ).encode()).hexdigest()
        if all(existing_metadata.get(key) == value for key, value in expected.items()):
            print(f"Completed adapter already exists: {adapter_file}")
            return
    if adapter_dir.exists() and settings.get("reject_truncation") and not args.force:
        raise SystemExit("Existing adapter does not match this run; choose a new output directory.")
    if args.reuse_from:
        if args.force or args.reuse_from == args.variant or args.reuse_from not in variants:
            raise SystemExit("Invalid explicit reuse condition")
        representative_rows = read_jsonl(source_path.with_name(args.reuse_from + ".jsonl"))
        if supervision_signature(rows) != supervision_signature(representative_rows):
            raise SystemExit("Cannot reuse different supervision")
        representative_dir = output_dir.parent.parent / args.reuse_from / f"seed_{args.seed}"
        representative_metadata = representative_dir / "run_metadata.json"
        representative_adapter = representative_dir / "adapter"
        if not (representative_adapter / "adapters.safetensors").exists() or not representative_metadata.exists():
            raise SystemExit("Representative training has not completed")
        metadata = json.loads(representative_metadata.read_text())
        expected_signature = hashlib.sha256(json.dumps(
            {"settings": settings, "training": config["training"], "seed": args.seed,
             "iters_override": args.iters}, sort_keys=True,
        ).encode()).hexdigest()
        if metadata.get("training_signature") != expected_signature or metadata["base_model"] != model:
            raise SystemExit("Representative training settings differ")
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.symlink_to(representative_adapter.resolve(), target_is_directory=True)
        metadata.update(variant=args.variant, training_data_sha256=training_data_sha256,
                        reused_from=args.reuse_from, actual_new_training=False,
                        supervision_sha256=supervision_signature(rows),
                        reuse_recorded_at=datetime.now(UTC).isoformat())
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(json.dumps({"variant":args.variant,"reused_from":args.reuse_from,
                          "reason":"identical ordered question/prompt/completion and training settings"}))
        return
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    iterations = args.iters or training_iterations(
        len(rows),
        settings["batch_size"],
        settings["num_train_epochs"],
        settings["gradient_accumulation_steps"],
    )
    optimizer_updates = iterations // settings["gradient_accumulation_steps"]
    warmup_steps = max(1, round(optimizer_updates * config["training"]["warmup_ratio"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": model,
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {"weight_decay": config["training"]["weight_decay"]}},
        "data": str(data_dir),
        "seed": args.seed,
        "num_layers": settings["num_layers"],
        "batch_size": settings["batch_size"],
        "iters": iterations,
        "learning_rate": settings["learning_rate"],
        "steps_per_report": settings["logging_steps"],
        "steps_per_eval": iterations + 1,
        "grad_accumulation_steps": settings["gradient_accumulation_steps"],
        "adapter_path": str(adapter_dir),
        "save_every": settings["save_steps"],
        "max_seq_length": settings["max_length"],
        "mask_prompt": True,
        "lora_parameters": {
            "rank": settings["lora_r"],
            "scale": settings["lora_scale"],
            "dropout": settings["lora_dropout"],
        },
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [settings["learning_rate"], optimizer_updates],
            "warmup": warmup_steps,
            "warmup_init": 0.0,
        },
    }
    config_path = output_dir / "mlx_config.yaml"
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")

    started_at = datetime.now(UTC)
    start = time.monotonic()
    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    log_path = output_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=resolve_path("."),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)

    metadata = {
        "training_signature": hashlib.sha256(json.dumps(
            {"settings": settings, "training": config["training"], "seed": args.seed,
             "iters_override": args.iters}, sort_keys=True,
        ).encode()).hexdigest(),
        "budget_mode": settings.get("budget_mode", "legacy_question_exposure"),
        "backend": "mlx-lm",
        "variant": args.variant,
        "seed": args.seed,
        "base_model": model,
        "quantization": "4-bit",
        "batch_size": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "training_data_sha256": training_data_sha256,
        "supervision_sha256": supervision_signature(rows),
        "actual_new_training": True,
        "source_rows": len(rows),
        "unique_questions": len({row["source_id"] for row in rows}),
        "iterations": iterations,
        "optimizer_updates": optimizer_updates,
        "warmup_updates": warmup_steps,
        "effective_epochs": iterations * settings["batch_size"] / len(rows),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
