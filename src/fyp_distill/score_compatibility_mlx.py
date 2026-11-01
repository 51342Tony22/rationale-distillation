from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from .common import load_config, rationale_body, read_jsonl, resolve_path, write_jsonl


def completion_nll(model, tokenizer, question: str, rationale: str) -> tuple[float, int]:
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following problem step by step. "
                "End with `Final answer: <number>`.\n\n" + question
            ),
        },
        {"role": "assistant", "content": rationale_body(rationale)},
    ]
    tokens = tokenizer.apply_chat_template(messages, return_dict=False)
    offset = len(
        tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False
        )
    )
    if len(tokens) <= offset:
        raise ValueError("Rationale produced no assistant tokens.")
    token_array = mx.array([tokens])
    inputs = token_array[:, :-1]
    targets = token_array[:, 1:]
    logits = model(inputs)
    start = max(offset - 1, 0)
    completion_logits = logits[:, start:, :]
    completion_targets = targets[:, start:]
    losses = nn.losses.cross_entropy(completion_logits, completion_targets)
    mean_loss = losses.astype(mx.float32).mean()
    mx.eval(mean_loss)
    return float(mean_loss.item()), int(completion_targets.size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score each correct teacher rationale under the untrained MLX student."
    )
    parser.add_argument("--config", default="configs/qwen3_8b_main.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    input_path = args.input or data["teacher_file"]
    output_path = args.output or data["scored_teacher_file"]
    model_name = args.model or config["compatibility"]["model"]
    rows = read_jsonl(input_path)
    rows = rows[: args.limit] if args.limit else rows
    existing = {row["id"]: row for row in read_jsonl(output_path)}

    model, tokenizer = load(model_name)
    scored_rows = []
    total_candidates = 0
    newly_scored = 0
    for row_index, source in enumerate(rows, start=1):
        record = existing.get(source["id"], source)
        if record.get("teacher_model") != source.get("teacher_model"):
            record = source
        for candidate in record.get("candidates", []):
            if not candidate.get("is_usable", candidate.get("is_correct", False)):
                continue
            total_candidates += 1
            score_meta = candidate.get("compatibility", {})
            reusable = (
                score_meta.get("model") == model_name
                and score_meta.get("nll") is not None
                and not args.force
            )
            if reusable:
                candidate["compatibility_nll"] = score_meta["nll"]
                continue
            nll, token_count = completion_nll(
                model, tokenizer, record["question"], candidate["text"]
            )
            candidate["compatibility_nll"] = nll
            candidate["compatibility"] = {
                "model": model_name,
                "nll": nll,
                "rationale_tokens": token_count,
                "scored_at": datetime.now(UTC).isoformat(),
            }
            newly_scored += 1
        scored_rows.append(record)
        write_jsonl(output_path, scored_rows)
        print(f"[{row_index}/{len(rows)}] {record['id']} scored", flush=True)

    report = {
        "teacher_rows": len(rows),
        "usable_candidates": total_candidates,
        "newly_scored": newly_scored,
        "model": model_name,
        "output": str(resolve_path(output_path)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
