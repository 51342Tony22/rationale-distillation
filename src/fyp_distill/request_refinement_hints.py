"""Sequential teacher queue with durable conservative cost reservations.

Only saved public response content is converted into hint candidates. Credentials
are loaded from the existing project .env and never logged. This is an estimated
spend limit at configured prices, not a provider-enforced billing guarantee.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime

from .common import ROOT, load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import correct, unique_index


def reservation_cny(prompt: str, max_output: int, thinking_budget: int) -> float:
    # UTF-8 bytes conservatively upper-bound input tokens with envelope allowance.
    # Reserve output + reasoning separately even when a provider counts them jointly.
    return ((len(prompt.encode()) + 1024) * 4 + (max_output + thinking_budget) * 16) / 1e6


def parse_hint(request: dict, text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # R1 occasionally returns a valid leading JSON object followed by prose.
    # Keep the billed raw response in the ledger, recover only an unambiguous
    # leading object, and still require a separate explanation review.
    repaired_delimiters = False
    try:
        obj, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        # A lone LaTeX math delimiter is an invalid JSON escape. Repair only
        # these four delimiters, leaving valid JSON escapes and the hint itself
        # untouched. The original public response remains saved in the ledger.
        escaped = re.sub(r"(?<!\\)\\([()\[\]])", r"\\\\\1", cleaned)
        if escaped == cleaned:
            raise
        cleaned = escaped
        obj, end = json.JSONDecoder().raw_decode(cleaned)
        repaired_delimiters = True
    suffix = cleaned[end:].strip()
    if suffix and any(c in suffix for c in "{}"):
        raise ValueError("Ambiguous additional JSON object in teacher response")
    if not isinstance(obj, dict):
        raise ValueError("Teacher response must start with a JSON object")
    if type(obj.get("offset")) is not int or obj["offset"] not in request["allowed_offsets"]:
        raise ValueError("Teacher chose an invalid offset")
    if not isinstance(obj.get("hint"), str) or not obj["hint"].strip():
        raise ValueError("Teacher returned an empty explanation")
    record = {"id": request["id"], "source_id": request["source_id"],
            "source_hash": request["source_hash"], "kind": request["kind"],
            "offset": obj["offset"], "hint": obj["hint"].strip(),
            "request_hash": request["request_hash"],
            "review": {"valid_explanation": False, "no_answer_leak": False}}
    if suffix:
        record["response_parse"] = "leading_json_with_trailing_text"
    if repaired_delimiters:
        record["latex_delimiter_escape_repair"] = True
    if request.get("replaces_request_id"):
        record["replaces_request_id"] = request["replaces_request_id"]
    return record


def parse_original(request: dict, text: str) -> dict:
    usable = correct(text, request["gold_answer"], request["dataset"],
                     request.get("answer_protocol", "first_final_line_v2"))
    return {"id": request["source_id"], "question": request["question"],
            "gold_answer": request["gold_answer"], "teacher_model": request["model"],
            "request_hash": request["request_hash"], "candidates": [{
                "candidate_id": request["id"], "text": text,
                "is_correct": usable, "is_usable": usable,
                "quality_status": "final_answer_only; step_review_required",
            }]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/refinement_pilot.yaml")
    parser.add_argument("--limit", type=int, default=4, help="Maximum newly issued requests this invocation")
    parser.add_argument("--requests", help="Saved request JSONL; defaults to teacher_requests.jsonl")
    parser.add_argument("--check-connection", action="store_true",
                        help="Read model inventory without generating, spending, or changing the ledger")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["refinement"]
    cap = settings["new_api_budget_cny"]
    if cap <= 0:
        raise SystemExit("No teacher budget allocated")
    root = resolve_path(settings["root"])
    requests = read_jsonl(resolve_path(args.requests) if args.requests else root / "teacher_requests.jsonl")
    if not requests:
        raise SystemExit("No saved teacher requests")
    unique_index(requests)
    ledger_path = resolve_path(settings["budget_ledger"])
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = unique_index(read_jsonl(ledger_path))
    hints_path = root / "hints.jsonl"
    hints = unique_index(read_jsonl(hints_path))
    originals_path = resolve_path(config["data"]["teacher_file"])
    originals = unique_index(read_jsonl(originals_path))

    def save_response(request, text):
        if request["kind"] == "original":
            record = parse_original(request, text)
            previous = originals.get(record["id"])
            if previous and previous != record:
                raise ValueError("Original rationale already exists; refusing overwrite")
            originals[record["id"]] = record
            write_jsonl(originals_path, originals.values())
        else:
            hints[request["id"]] = parse_hint(request, text)
            write_jsonl(hints_path, hints.values())
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["TEACHER_API_KEY"], base_url=os.environ.get("TEACHER_BASE_URL"),
                    timeout=config.get("teacher_request_timeout_seconds", 240), max_retries=0)
    if args.check_connection:
        try:
            inventory = client.with_options(timeout=20, max_retries=0).models.list()
            print(json.dumps({"connection_check": "ok", "returned_models": len(inventory.data),
                              "teacher_generation_calls": 0, "ledger_modified": False}))
        finally:
            client.close()
        return
    calls = 0
    # A lock prevents two queue workers from independently spending the same cap.
    import fcntl
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = unique_index(read_jsonl(ledger_path))
        hints = unique_index(read_jsonl(hints_path))
        for request in requests:
            if request["id"] in ledger:
                previous = ledger[request["id"]]
                if previous["request_hash"] != request["request_hash"]:
                    raise ValueError("Request changed after it was billed/reserved")
                # Recovery of a completed response never issues another paid call.
                missing = (request["source_id"] not in originals if request["kind"] == "original"
                           else request["id"] not in hints)
                if previous.get("status") == "complete" and missing:
                    save_response(request, previous["response_content"])
                continue
            if calls >= args.limit:
                break
            max_tokens, thinking_budget = request.get("max_output_tokens", 1024), 4096
            reservation = reservation_cny(request["prompt"], max_tokens, thinking_budget)
            charged = sum(row["charged_cny"] for row in ledger.values())
            if charged + reservation > cap:
                print("Estimated budget cap reached; no request sent.", flush=True)
                break
            entry = {"id": request["id"], "request_hash": request["request_hash"],
                     "status": "reserved", "charged_cny": reservation,
                     "reserved_cny": reservation, "model": request["model"],
                     "started_at": datetime.now(UTC).isoformat()}
            ledger[request["id"]] = entry
            write_jsonl(ledger_path, ledger.values())
            calls += 1
            started = time.monotonic()
            try:
                # A reviewed request may explicitly allow a longer transport wait.
                # This does not enable retries or alter generation/model settings.
                request_client = client.with_options(timeout=request.get("request_timeout_seconds", config.get("teacher_request_timeout_seconds", 240)), max_retries=0)
                response = request_client.chat.completions.create(
                    model=request["model"], messages=[{"role": "user", "content": request["prompt"]}],
                    temperature=0.6, top_p=0.95, max_tokens=max_tokens,
                    extra_body={"enable_thinking": True, "thinking_budget": thinking_budget},
                )
                text = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage else None
                entry.update(response_content=text, response_id=response.id,
                             returned_model=response.model, usage=usage,
                             finish_reason=response.choices[0].finish_reason,
                             elapsed_seconds=time.monotonic() - started,
                             finished_at=datetime.now(UTC).isoformat())
                if usage and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                    entry["charged_cny"] = (usage["prompt_tokens"] * 4 + usage["completion_tokens"] * 16) / 1e6
                if entry["finish_reason"] == "length":
                    raise ValueError("Truncated teacher response; retained in ledger, not reused")
                # Validate before marking complete; incorrect original answers are
                # retained as unusable candidates rather than silently retried.
                if request["kind"] == "original":
                    parse_original(request, text)
                else:
                    parse_hint(request, text)
                entry["status"] = "complete"
                write_jsonl(ledger_path, ledger.values())
                save_response(request, text)
            except Exception as error:
                # No automatic retry. Unknown-billing failures retain reservation.
                entry["status"] = "failed_or_uncertain"
                entry["error_type"] = type(error).__name__
                entry["elapsed_seconds"] = time.monotonic() - started
                entry["finished_at"] = datetime.now(UTC).isoformat()
                write_jsonl(ledger_path, ledger.values())
                print(f"{request['id']}: {type(error).__name__}; reservation retained if usage unknown", flush=True)
                # Advance through other queued IDs without retrying this request.
                continue
            print(f"{request['id']}: recorded; explanation review still required", flush=True)
    client.close()
    print(json.dumps({"new_calls": calls, "estimated_or_reserved_cny": sum(x["charged_cny"] for x in ledger.values()),
                      "cap_cny": cap, "hint_candidates": len(hints)}))


if __name__ == "__main__":
    main()
