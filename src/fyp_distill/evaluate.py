from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    configured_variants,
    extract_answer,
    load_config,
    prompt_for,
    read_jsonl,
    resolve_path,
    write_jsonl,
)
from .math_grader import extract_math_answer, grade_math_answer


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int, **kwargs) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    generated = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def rationale_words(text: str) -> int:
    rationale = re.split(r"(?i)final\s+answer", text, maxsplit=1)[0]
    return len(re.findall(r"\b\w+\b", rationale))


def agreement(answers: list[str | None]) -> float:
    if not answers:
        return 0.0
    return Counter(answers).most_common(1)[0][1] / len(answers)


def extract_for_split(text: str, split: str) -> str | None:
    return extract_math_answer(text) if split == "math500" else extract_answer(text)


def is_correct(answer: str | None, gold_answer: str, split: str) -> bool:
    return grade_math_answer(answer, gold_answer) if split == "math500" else answer == gold_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one trained rationale condition.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("dev", "test", "svamp", "math500"), default="test")
    parser.add_argument("--skip-behavioural", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    config = load_config(args.config)
    variants = configured_variants(config)
    if args.variant != "base_student" and args.variant not in variants:
        raise SystemExit(f"Variant {args.variant!r} is not enabled in {args.config}.")
    seed = args.seed if args.seed is not None else config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    student = config["student"]
    settings = config["evaluation"]
    if args.variant == "base_student":
        adapter_dir = None
        base_model_name = student["model"]
    else:
        adapter_dir = resolve_path("outputs") / args.variant / f"seed_{seed}" / "adapter"
        if not adapter_dir.exists():
            raise SystemExit(f"Adapter not found at {adapter_dir}. Train this variant first.")
        metadata_path = adapter_dir.parent / "run_metadata.json"
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        base_model_name = metadata["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"dtype": "auto"}
    if torch.backends.mps.is_available():
        model_kwargs["attn_implementation"] = "eager"
    base = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    device = choose_device()
    model.to(device).eval()

    rows = read_jsonl(config["data"][f"{args.split}_file"])
    rows = rows[: args.limit] if args.limit else rows
    predictions = []
    for position, row in enumerate(rows, start=1):
        primary_text = generate(
            model,
            tokenizer,
            device,
            prompt_for(row["question"], settings["prompt_templates"][0]),
            settings["max_new_tokens"],
            do_sample=False,
        )
        primary_answer = extract_for_split(primary_text, args.split)
        record = {
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "primary_text": primary_text,
            "primary_answer": primary_answer,
            "correct": is_correct(primary_answer, row["gold_answer"], args.split),
            "rationale_words": rationale_words(primary_text),
        }
        behavioural = not args.skip_behavioural and position <= settings["behavioural_limit"]
        if behavioural:
            sampled_texts = [
                generate(
                    model,
                    tokenizer,
                    device,
                    prompt_for(row["question"], settings["prompt_templates"][0]),
                    settings["max_new_tokens"],
                    do_sample=True,
                    temperature=settings["temperature"],
                    top_p=0.95,
                )
                for _ in range(settings["stochastic_samples"])
            ]
            prompt_texts = [
                generate(
                    model,
                    tokenizer,
                    device,
                    prompt_for(row["question"], template),
                    settings["max_new_tokens"],
                    do_sample=False,
                )
                for template in settings["prompt_templates"]
            ]
            record.update(
                {
                    "sampled_texts": sampled_texts,
                    "sampled_answers": [extract_for_split(text, args.split) for text in sampled_texts],
                    "prompt_texts": prompt_texts,
                    "prompt_answers": [extract_for_split(text, args.split) for text in prompt_texts],
                }
            )
        predictions.append(record)
        print(f"[{position}/{len(rows)}] {row['id']}: {primary_answer} vs {row['gold_answer']}")

    report_dir = resolve_path("reports") / args.split / args.variant / f"seed_{seed}"
    write_jsonl(report_dir / "predictions.jsonl", predictions)
    total = max(len(predictions), 1)
    metrics = {
        "variant": args.variant,
        "seed": seed,
        "split": args.split,
        "n": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / total,
        "metric": "math_equivalence" if args.split == "math500" else "exact_match",
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
                            is_correct(answer, row["gold_answer"], args.split)
                            for row in behavioural_rows
                            for answer in row["prompt_answers"]
                        ]
                    )
                ),
            }
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
