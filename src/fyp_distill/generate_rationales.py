from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .common import extract_answer, load_config, read_jsonl, resolve_path, write_jsonl


STRATEGY_HINTS = [
    "Use a direct arithmetic derivation.",
    "Work backwards from what the question asks when useful.",
    "Introduce a variable and form short equations.",
    "Break the problem into small quantities and combine them.",
    "Use an alternative valid route rather than copying an obvious stock explanation.",
    "Check the result with a second calculation before concluding.",
]


def teacher_prompt(question: str, candidate_index: int, max_words: int = 180) -> str:
    hint = STRATEGY_HINTS[candidate_index % len(STRATEGY_HINTS)]
    return f"""Solve this grade-school mathematics problem accurately.
After reasoning privately, put a concise, self-contained derivation in the final response.
The final response must contain the actual derivation, not only the answer, and must be no more than {max_words} words. {hint}
Do not mention being a teacher or this instruction.
Your last line must have exactly this form: Final answer: <number>

Problem:
{question}"""


def estimated_cost_cny(records, input_price: float, output_price: float) -> float:
    prompt_tokens = 0
    completion_tokens = 0
    for record in records:
        for candidate in record.get("candidates", []):
            usage = candidate.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


async def request_candidate(client, model, row, index, settings, semaphore):
    last_error = None
    for attempt in range(settings["max_retries"]):
        try:
            async with semaphore:
                request = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": teacher_prompt(
                                row["question"], index, settings.get("max_words", 180)
                            ),
                        }
                    ],
                    "temperature": settings["temperature"],
                    "top_p": settings.get("top_p", 0.95),
                    "max_tokens": settings["max_tokens"],
                }
                if settings.get("reasoning_effort"):
                    request["reasoning_effort"] = settings["reasoning_effort"]
                if settings.get("thinking_budget"):
                    request["extra_body"] = {
                        "enable_thinking": True,
                        "thinking_budget": settings["thinking_budget"]
                    }
                response = await client.chat.completions.create(**request)
            message = response.choices[0].message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            source = settings.get("supervision_source", "response_content")
            if source == "reasoning_content":
                text = reasoning.strip()
            elif source == "response_or_reasoning":
                text = content.strip() or reasoning.strip()
            else:
                text = content.strip()
            parsed = extract_answer(text)
            word_count = len(text.split())
            passes_quality = (
                bool(text)
                and word_count >= settings.get("min_words", 8)
                and word_count <= settings.get("max_words", 180)
                and response.choices[0].finish_reason != "length"
            )
            usage = response.usage
            return {
                "candidate_id": f"{row['id']}_r{index:02d}",
                "text": text,
                "supervision_source": source,
                "word_count": word_count,
                "passes_quality": passes_quality,
                "reasoning_word_count": len(reasoning.split()),
                "response_content_word_count": len(content.split()),
                "parsed_answer": parsed,
                "is_correct": parsed == row["gold_answer"],
                "is_usable": parsed == row["gold_answer"] and passes_quality,
                "strategy_hint": STRATEGY_HINTS[index % len(STRATEGY_HINTS)],
                "response_id": response.id,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                },
            }
        except Exception as error:  # API errors vary across compatible providers.
            last_error = error
            await asyncio.sleep(min(2**attempt, 20))
    return {
        "candidate_id": f"{row['id']}_r{index:02d}",
        "text": "",
        "parsed_answer": None,
        "is_correct": False,
        "error": repr(last_error),
    }


async def process_row(
    client,
    model,
    row,
    n,
    settings,
    semaphore,
    existing_row=None,
    on_candidate=None,
):
    existing_by_id = {
        candidate.get("candidate_id"): candidate
        for candidate in (existing_row or {}).get("candidates", [])
    }

    generated_at = datetime.now(UTC).isoformat()

    def make_record(candidates):
        return {
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "teacher_model": model,
            "generated_at": generated_at,
            "generation": {
                "schema_version": 4,
                "supervision_source": settings.get("supervision_source", "response_content"),
                "reasoning_effort": settings.get("reasoning_effort"),
                "thinking_budget": settings.get("thinking_budget"),
                "temperature": settings["temperature"],
                "top_p": settings.get("top_p", 0.95),
                "max_tokens": settings["max_tokens"],
                "min_words": settings.get("min_words", 8),
                "max_words": settings.get("max_words", 180),
                "requested_candidates": n,
            },
            "candidates": candidates,
        }

    async def get_candidate(index):
        candidate_id = f"{row['id']}_r{index:02d}"
        existing = existing_by_id.get(candidate_id)
        if existing and existing.get("text", "").strip() and not existing.get("error"):
            return index, existing
        candidate = await request_candidate(client, model, row, index, settings, semaphore)
        return index, candidate

    candidates_by_index: dict[int, dict] = {
        index: candidate
        for index in range(n)
        if (candidate := existing_by_id.get(f"{row['id']}_r{index:02d}")) is not None
    }
    tasks = [asyncio.create_task(get_candidate(index)) for index in range(n)]
    for task in asyncio.as_completed(tasks):
        index, candidate = await task
        candidates_by_index[index] = candidate
        record = make_record([candidates_by_index[key] for key in sorted(candidates_by_index)])
        if on_candidate is not None:
            await on_candidate(record)
    return make_record([candidates_by_index[index] for index in range(n)])


async def run(args) -> None:
    load_dotenv()
    config = load_config(args.config)
    settings = config["teacher"]
    data_config = config["data"]
    input_path = args.input or data_config["train_file"]
    output_path = args.output or data_config["teacher_file"]
    n = args.n or data_config["rationale_candidates"]

    api_key = os.getenv("TEACHER_API_KEY")
    model = args.model or settings.get("model") or os.getenv("TEACHER_MODEL")
    base_url = os.getenv("TEACHER_BASE_URL") or None
    if not api_key or not model:
        raise SystemExit("Set TEACHER_API_KEY and TEACHER_MODEL in .env before generation.")

    rows = read_jsonl(input_path)
    limit = args.limit if args.limit is not None else data_config.get("teacher_question_limit")
    rows = rows[:limit] if limit else rows
    existing_rows = read_jsonl(output_path) if args.resume else []
    if args.seed_from:
        existing_ids = {row["id"] for row in existing_rows}
        existing_rows.extend(
            row for row in read_jsonl(args.seed_from) if row["id"] not in existing_ids
        )
    records_by_id = {row["id"]: row for row in existing_rows}
    completed = {
        row_id
        for row_id, row in records_by_id.items()
        if len(row.get("candidates", [])) == n
        and row.get("generation", {}).get("schema_version") == 4
        and not any(candidate.get("error") for candidate in row["candidates"])
    }
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.get("request_timeout_seconds", 120),
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(settings["concurrency"])
    write_lock = asyncio.Lock()

    async def persist(record):
        async with write_lock:
            records_by_id[record["id"]] = record
            write_jsonl(
                output_path,
                [records_by_id[row["id"]] for row in rows if row["id"] in records_by_id],
            )

    remaining = [row for row in rows if row["id"] not in completed]
    batch_size = settings.get("question_concurrency", 4)
    input_price = settings.get("input_price_cny_per_m_tokens", 4.0)
    output_price = settings.get("output_price_cny_per_m_tokens", 16.0)
    cost_cap = settings.get("max_total_estimated_cost_cny")
    finished = len(rows) - len(remaining)
    for offset in range(0, len(remaining), batch_size):
        current_cost = estimated_cost_cny(
            records_by_id.values(), input_price, output_price
        )
        if cost_cap is not None and current_cost >= cost_cap:
            print(
                f"Cost cap reached: estimated ¥{current_cost:.2f} >= ¥{cost_cap:.2f}. "
                "Stopping before the next question batch.",
                flush=True,
            )
            break
        batch = remaining[offset : offset + batch_size]
        tasks = [
            asyncio.create_task(
                process_row(
                    client,
                    model,
                    row,
                    n,
                    settings,
                    semaphore,
                    records_by_id.get(row["id"]),
                    persist,
                )
            )
            for row in batch
        ]
        for task in asyncio.as_completed(tasks):
            record = await task
            await persist(record)
            finished += 1
            accepted = sum(candidate.get("is_usable", False) for candidate in record["candidates"])
            print(f"[{finished}/{len(rows)}] {record['id']}: accepted {accepted}/{n}")
    await client.close()
    final_cost = estimated_cost_cny(records_by_id.values(), input_price, output_price)
    print(f"Estimated recorded API cost: ¥{final_cost:.2f}")
    print(f"Teacher data: {resolve_path(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate teacher rationales.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--n", type=int)
    parser.add_argument("--model", help="Override the teacher model for this run.")
    parser.add_argument(
        "--seed-from",
        help="Reuse compatible pilot records when initializing a new output file.",
    )
    parser.add_argument("--limit", type=int, help="Generate only the first N questions for a pilot.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
