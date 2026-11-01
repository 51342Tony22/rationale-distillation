from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .common import extract_answer, load_config, prompt_for, read_jsonl, resolve_path, write_jsonl


async def evaluate_one(client, model, row, settings, teacher_settings, semaphore):
    last_error = None
    for attempt in range(teacher_settings.get("max_retries", 2)):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_for(row["question"], settings["prompt_templates"][0]),
                        }
                    ],
                    temperature=0,
                    max_tokens=settings["max_new_tokens"],
                    reasoning_effort=teacher_settings.get("reasoning_effort", "low"),
                )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None) or ""
            content = message.content or ""
            text = content.strip() or reasoning.strip()
            answer = extract_answer(text)
            usage = response.usage
            return {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "primary_text": text,
                "schema_version": 3,
                "supervision_source": (
                    "response_content" if content.strip() else "reasoning_fallback"
                ),
                "reasoning_word_count": len(reasoning.split()),
                "primary_answer": answer,
                "correct": answer == row["gold_answer"],
                "response_id": response.id,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                },
                "generated_at": datetime.now(UTC).isoformat(),
            }
        except Exception as error:
            last_error = error
            await asyncio.sleep(min(2**attempt, 20))
    return {
        "id": row["id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "primary_text": "",
        "primary_answer": None,
        "correct": False,
        "error": repr(last_error),
        "generated_at": datetime.now(UTC).isoformat(),
    }


async def run(args) -> None:
    load_dotenv()
    config = load_config(args.config)
    settings = config["evaluation"]
    api_key = os.getenv("TEACHER_API_KEY")
    base_url = os.getenv("TEACHER_BASE_URL") or None
    model = os.getenv("TEACHER_MODEL")
    if not api_key or not model:
        raise SystemExit("Set TEACHER_API_KEY and TEACHER_MODEL in .env first.")
    rows = read_jsonl(config["data"][f"{args.split}_file"])
    rows = rows[: args.limit] if args.limit else rows
    report_dir = resolve_path("reports") / args.split / "teacher"
    output_path = report_dir / "predictions.jsonl"
    existing = {row["id"]: row for row in read_jsonl(output_path)} if args.resume else {}
    completed = {
        key
        for key, row in existing.items()
        if not row.get("error") and row.get("schema_version") == 3
    }
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=config["teacher"].get("request_timeout_seconds", 120),
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    remaining = [row for row in rows if row["id"] not in completed]
    finished = len(rows) - len(remaining)
    for offset in range(0, len(remaining), args.concurrency):
        batch = remaining[offset : offset + args.concurrency]
        tasks = [
            asyncio.create_task(
                evaluate_one(client, model, row, settings, config["teacher"], semaphore)
            )
            for row in batch
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            existing[result["id"]] = result
            finished += 1
            write_jsonl(
                output_path,
                [existing[row["id"]] for row in rows if row["id"] in existing],
            )
            print(f"Teacher evaluated {finished}/{len(rows)}")
    await client.close()
    predictions = read_jsonl(output_path)
    selected_ids = {row["id"] for row in rows}
    predictions = [row for row in predictions if row["id"] in selected_ids]
    total = max(len(predictions), 1)
    successful = [row for row in predictions if not row.get("error")]
    evaluated_total = max(len(successful), 1)
    usage_rows = [row for row in predictions if row.get("usage")]
    metrics = {
        "variant": "teacher",
        "teacher_model": model,
        "split": args.split,
        "n": len(predictions),
        "evaluated_n": len(successful),
        "coverage": len(successful) / total,
        "exact_match": sum(row["correct"] for row in successful) / evaluated_total,
        "end_to_end_exact_match": sum(row["correct"] for row in predictions) / total,
        "invalid_answer_rate": (
            sum(row["primary_answer"] is None for row in successful) / evaluated_total
        ),
        "api_error_rate": sum(bool(row.get("error")) for row in predictions) / total,
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] or 0 for row in usage_rows),
        "completion_tokens": sum(row["usage"]["completion_tokens"] or 0 for row in usage_rows),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paid teacher with pass@1.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", choices=("dev", "test", "svamp"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
