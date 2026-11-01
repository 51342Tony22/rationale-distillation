from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler

from .common import (
    artifact_dir,
    configured_variants,
    extract_answer,
    load_config,
    prompt_for,
    read_jsonl,
    resolve_path,
    write_jsonl,
)
from .math_grader import extract_math_answer, grade_math_answer
from .answer_protocol import PROTOCOL_VERSION, first_final_answer, generate_single_solution


def generation_response(model, tokenizer, prompts, max_tokens, new_protocol=False, **kwargs):
    if not new_protocol:
        return batch_generate(model, tokenizer, prompts, max_tokens=max_tokens, **kwargs)
    from types import SimpleNamespace
    generator = generate_single_solution
    if new_protocol == "first_answer_v3":
        from .answer_protocol_v3 import generate_single_solution as generator
    records = [generator(model, tokenizer, prompt, max_tokens, kwargs.get("sampler"))
               for prompt in prompts]
    return SimpleNamespace(texts=[record["text"] for record in records], records=records)


def rationale_words(text: str) -> int:
    rationale = re.split(r"(?i)final\s+answer", text, maxsplit=1)[0]
    return len(re.findall(r"\b\w+\b", rationale))


def agreement(answers: list[str | None]) -> float:
    return Counter(answers).most_common(1)[0][1] / len(answers) if answers else 0.0


def extract_for_split(text: str, split: str) -> str | None:
    return extract_math_answer(text) if split == "math500" else extract_answer(text)


def is_correct(answer: str | None, gold_answer: str, split: str) -> bool:
    return grade_math_answer(answer, gold_answer) if split == "math500" else answer == gold_answer


def encode_prompt(tokenizer, question: str, template: str) -> list[int]:
    message = {"role": "user", "content": prompt_for(question, template)}
    return tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, return_dict=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MLX QLoRA condition.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "test", "svamp", "math500"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--skip-behavioural", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    variants = configured_variants(config)
    if args.variant != "base_student" and args.variant not in variants:
        raise SystemExit(f"Variant {args.variant!r} is not enabled in {args.config}.")
    model_name = config["mlx_training"]["model"]
    settings = config["evaluation"]
    grading_split = "math500" if settings.get("grading") == "math" else args.split
    protocol_name = settings.get("answer_protocol")
    if protocol_name not in (None, "legacy", PROTOCOL_VERSION, "first_answer_v3"):
        raise ValueError(f"Unsupported evaluation protocol: {protocol_name}")
    new_protocol = protocol_name if protocol_name in (PROTOCOL_VERSION, "first_answer_v3") else False
    extractor = first_final_answer
    if new_protocol == "first_answer_v3":
        from .answer_protocol_v3 import first_final_answer as extractor

    def extract(text):
        return extractor(text) if new_protocol else extract_for_split(text, grading_split)
    adapter_dir = None
    adapter_signature = "base_student"
    if args.variant != "base_student":
        run_dir = artifact_dir(config, "mlx_output_dir", "outputs_mlx") / args.variant / f"seed_{args.seed}"
        adapter_dir = run_dir / "adapter"
        metadata_path = run_dir / "run_metadata.json"
        if not adapter_dir.exists() or not metadata_path.exists():
            raise SystemExit(f"Completed adapter not found at {adapter_dir}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_name = metadata["base_model"]
        adapter_signature = hashlib.sha256(
            (adapter_dir / "adapters.safetensors").read_bytes()
        ).hexdigest()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(model_name, adapter_path=str(adapter_dir) if adapter_dir else None)
    rows = read_jsonl(config["data"][f"{args.split}_file"])
    rows = rows[: args.limit] if args.limit else rows
    max_tokens = args.max_new_tokens or settings["max_new_tokens"]
    batch_size = args.batch_size or settings.get("batch_size", 4)
    report_dir = artifact_dir(config, "mlx_report_dir", "reports_mlx") / args.split / args.variant / f"seed_{args.seed}"
    output_path = report_dir / "predictions.jsonl"
    run_signature = {
        "model": model_name,
        "adapter_sha256": adapter_signature,
        "max_new_tokens": max_tokens,
        "batch_size": batch_size,
        "split": args.split,
        "grading_split": grading_split,
        "answer_protocol": protocol_name if new_protocol else "legacy",
        "prompt_templates": settings["prompt_templates"],
        "evaluation_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode("utf-8")
        ).hexdigest(),
        "behavioural": not args.skip_behavioural,
    }
    if new_protocol == "first_answer_v3":
        from pathlib import Path
        protocol_path = Path(__file__).with_name("answer_protocol_v3.py")
        run_signature["protocol_source_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        run_signature["evaluation_rows_sha256"] = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    signature_path = report_dir / "evaluation_signature.json"
    old_signature = (
        json.loads(signature_path.read_text(encoding="utf-8"))
        if signature_path.exists()
        else None
    )
    if old_signature is not None and old_signature != run_signature:
        raise SystemExit("Evaluation signature changed; preserve prior outputs and use a new run directory.")
    if output_path.exists() and old_signature is None:
        raise SystemExit("Unsigned prior predictions; use a new run directory rather than overwrite.")
    report_dir.mkdir(parents=True, exist_ok=True)
    # Bind signatures before the first generation so interruption can resume.
    signature_path.write_text(json.dumps(run_signature, indent=2), encoding="utf-8")
    existing = (
        {row["id"]: row for row in read_jsonl(output_path)}
        if old_signature == run_signature
        else {}
    )
    completed = {
        key
        for key, row in existing.items()
        if args.skip_behavioural or "sampled_answers" in row
    }

    remaining = [row for row in rows if row["id"] not in completed]
    row_positions = {row["id"]: index for index, row in enumerate(rows, start=1)}
    for offset in range(0, len(remaining), batch_size):
        batch = remaining[offset : offset + batch_size]
        prompts = [
            encode_prompt(tokenizer, row["question"], settings["prompt_templates"][0])
            for row in batch
        ]
        response = generation_response(
            model,
            tokenizer,
            prompts,
            max_tokens=max_tokens,
            new_protocol=new_protocol,
            prefill_batch_size=batch_size,
            completion_batch_size=batch_size,
        )
        for row, text in zip(batch, response.texts):
            answer = extract(text)
            record = {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "primary_text": text,
                "primary_answer": answer,
                "correct": is_correct(answer, row["gold_answer"], grading_split),
                "rationale_words": rationale_words(text),
            }
            if new_protocol:
                generation = response.records[batch.index(row)]
                record.update({key: generation[key] for key in
                               ("generation_tokens", "finish_reason", "elapsed_seconds", "answer_protocol")})
            position = row_positions[row["id"]]
            if not args.skip_behavioural and position <= settings["behavioural_limit"]:
                sample_prompt = prompts[batch.index(row)]
                sample_response = generation_response(
                    model,
                    tokenizer,
                    [sample_prompt] * settings["stochastic_samples"],
                    max_tokens=max_tokens,
                    new_protocol=new_protocol,
                    sampler=make_sampler(temp=settings["temperature"], top_p=0.95),
                )
                prompt_response = generation_response(
                    model,
                    tokenizer,
                    [
                        encode_prompt(tokenizer, row["question"], template)
                        for template in settings["prompt_templates"]
                    ],
                    max_tokens=max_tokens,
                    new_protocol=new_protocol,
                )
                record.update(
                    {
                        "sampled_texts": sample_response.texts,
                        "sampled_answers": [
                            extract(text) for text in sample_response.texts
                        ],
                        "prompt_texts": prompt_response.texts,
                        "prompt_answers": [
                            extract(text) for text in prompt_response.texts
                        ],
                    }
                )
            existing[row["id"]] = record
        write_jsonl(output_path, [existing[row["id"]] for row in rows if row["id"] in existing])
        print(f"[{min(offset + len(batch), len(remaining))}/{len(remaining)}] evaluated")

    predictions = [existing[row["id"]] for row in rows if row["id"] in existing]
    total = max(len(predictions), 1)
    metrics = {
        "variant": args.variant,
        "seed": args.seed,
        "split": args.split,
        "backend": "mlx-lm",
        "base_model": model_name,
        "n": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / total,
        "metric": "math_equivalence" if grading_split == "math500" else "exact_match",
        "invalid_answer_rate": sum(row["primary_answer"] is None for row in predictions) / total,
        "mean_rationale_words": float(np.mean([row["rationale_words"] for row in predictions]))
        if predictions
        else 0.0,
    }
    metrics["exact_match"] = metrics["accuracy"]
    behavioural_rows = [row for row in predictions if "sampled_answers" in row]
    if behavioural_rows:
        metrics.update(
            {
                "behavioural_n": len(behavioural_rows),
                "mean_sample_agreement": float(
                    np.mean([agreement(row["sampled_answers"]) for row in behavioural_rows])
                ),
                "mean_prompt_agreement": float(
                    np.mean([agreement(row["prompt_answers"]) for row in behavioural_rows])
                ),
                "prompt_robust_accuracy": float(
                    np.mean(
                        [
                            is_correct(answer, row["gold_answer"], grading_split)
                            for row in behavioural_rows
                            for answer in row["prompt_answers"]
                        ]
                    )
                ),
            }
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(json.dumps(run_signature, indent=2), encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
