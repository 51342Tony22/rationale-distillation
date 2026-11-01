"""Separate, gold-independent expansion of final natural-language answer parsing.

Never mutates the baseline predictions or their frozen v1 scores.
This is a development sensitivity analysis, not a new generation experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import baseline

ROOT = baseline.ROOT
REJECTION = re.compile(r"\b(?:unknown|unclear|impossible|cannot|can't|no\s+(?:answer|solution)|not\s+(?:known|possible|enough))\b", re.I)
MARKER = re.compile(r"\b(?:so|thus|therefore|hence|finally|answer|altogether|total|will\s+(?:have|need|take|get|cost)|needs?|equals?)\b", re.I)
TOKEN_NUM = rf"(?<![\w.])({baseline.NUM})(?![\w.]|,\d)"


def parse(text):
    old, old_rule = baseline.parse_answer(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "empty"
    # Explicit uncertainty in unstructured prose overrides v1's loose terminal-number
    # fallback; keep this correction visible in the secondary analysis.
    if "\\boxed" not in text and "####" not in text and REJECTION.search(lines[-1]):
        return None, "explicit_uncertainty_in_final_claim"
    if old is not None:
        return old, "v1_" + old_rule
    if "\\boxed" in text or "####" in text:
        return None, "preserved_structured_answer_rejection"
    last = lines[-1]
    # Avoid rescuing code, simulated output or unfinished code fences as prose answers.
    if "```" in last or re.match(r"^(?:#|print\s*\(|return\b)", last):
        return None, "not_final_prose"
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", last)
    candidate = sentences[-1]
    if REJECTION.search(candidate) or not MARKER.search(candidate):
        return None, "no_clear_final_claim"
    candidate = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", candidate)
    candidate = candidate.replace(r"\(", "").replace(r"\)", "")
    numbers = [match.group(1) for match in re.finditer(TOKEN_NUM, candidate)]
    if len(numbers) != 1:
        return None, "multiple_or_no_numbers_in_final_claim"
    value = baseline.number(numbers[0])
    return value, "single_number_in_final_claim" if value is not None else "invalid_number"


def selfcheck():
    cases = [
        ("So, it will take George and Harry 22 rounds to fill the pool.", "22"),
        ("Therefore, Cary will need 6 more weekends before he can buy the shoes.", "6"),
        ("So, Sandy will have 375,000 more tokens than any sibling.", "375000"),
        ("Therefore, Oshea needs 30 small planters to plant all the seeds.", "30"),
        ("The answer is unknown, although we initially guessed 3.", None),
        ("So, either 6 or 8 could be the answer in this problem.", None),
        (r"\boxed{6} but \boxed{8}", None),
        ("```output", None),
        ("Therefore, the task is impossible even with 7 extra workers.", None),
        ("The answer is 3.00 dollars in this problem.", "3.00"),
    ]
    for text, expected in cases:
        value, _ = parse(text)
        assert value == (baseline.number(expected) if expected is not None else None), (text, value)


def analyse():
    selfcheck()
    config = json.loads(baseline.CONFIG.read_text())
    root = ROOT / config["output_dir"]
    report = {"analysis_version": "v2_final_claim_sensitivity",
              "script_sha256": baseline.sha(Path(__file__)),
              "v1_runner_sha256": baseline.sha(Path(baseline.__file__)),
              "gold_independent_rule": "Reject explicit uncertainty in unstructured final prose, otherwise preserve v1 valid parses; rescue only an explicit final prose claim with exactly one number; preserve boxed/hash ambiguity rejections.",
              "primary_scores_changed": False, "generations_repeated": False,
              "conditions": {},
              "limitations": ["Expanded EM does not validate reasoning or source gold correctness.",
                             "v1 may accept an intermediate terminal number; v2 preserves those parses.",
                             "This version was designed during development, after inspecting the first 20 student outputs."]}
    for condition in ["student_base", "teacher_reference"]:
        source = root / "dev_baseline" / condition / "predictions.jsonl"
        if not source.exists():
            continue
        # A newline-terminated snapshot avoids scoring a line mid-append.
        payload = source.read_bytes()
        payload = payload[:payload.rfind(b"\n") + 1]
        rows = [json.loads(line) for line in payload.splitlines()]
        results = []
        for row in rows:
            assert hashlib.sha256(row["output"].encode()).hexdigest() == row["output_sha256"]
            value, rule = parse(row["scoring_text"])
            correct = value is not None and value == baseline.number(row["gold"])
            results.append({"source_id": row["source_id"], "output_sha256": row["output_sha256"],
                            "v1_correct": row["correct"], "v1_invalid": row["invalid"],
                            "v2_parsed_answer": str(value) if value is not None else None,
                            "v2_parse_rule": rule, "v2_correct": correct, "v2_invalid": value is None})
        report["conditions"][condition] = {
            "snapshot_predictions_sha256": hashlib.sha256(payload).hexdigest(),
            "completed": len(rows), "expected": config["dev_count"],
            "complete_generation": len(rows) == config["dev_count"],
            "v1_correct": sum(r["v1_correct"] for r in results),
            "v2_correct": sum(r["v2_correct"] for r in results),
            "v1_invalid": sum(r["v1_invalid"] for r in results),
            "v2_invalid": sum(r["v2_invalid"] for r in results),
            "rescued_correct_format_rejections": sum(r["v1_invalid"] and r["v2_correct"] for r in results),
            "rows": results}
    baseline.save(root / "scoring_sensitivity_v2.json", report)
    print(json.dumps({condition: {key: value for key, value in data.items() if key != "rows"}
                      for condition, data in report["conditions"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        print("Final-claim parsing checks passed.")
    else:
        analyse()
