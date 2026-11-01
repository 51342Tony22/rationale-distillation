from __future__ import annotations

import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"
LEGACY_VARIANTS = (
    "answer_only",
    "single_rationale",
    "multi_rationale",
    "diverse_multi_rationale",
)
VARIANTS = (
    *LEGACY_VARIANTS,
    "low_diversity_rationale",
    "compatible_rationale",
    "original_rationale",
    "generic_expansion",
    "random_expansion",
    "hint_at_start",
    "local_refinement",
    "local_unfiltered",
)


def configured_variants(config: dict[str, Any]) -> tuple[str, ...]:
    variants = tuple(config.get("variants", LEGACY_VARIANTS))
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown experiment variants: {', '.join(unknown)}")
    return variants


def artifact_dir(config: dict[str, Any], key: str, default: str) -> Path:
    return resolve_path(config.get("artifacts", {}).get(key, default))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return []
    with resolved.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(resolved)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gsm8k_gold(answer: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", answer)
    return normalize_answer(match.group(1)) if match else extract_answer(answer)


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.replace("\\,", "").replace("%", "")
    cleaned = cleaned.strip(" .。!！")
    if cleaned.startswith("\\boxed{") and cleaned.endswith("}"):
        cleaned = cleaned[7:-1].strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.lower() or None
    if number == number.to_integral_value():
        # Avoid the context precision limit for long integers produced by a model.
        return format(number, "f").split(".", 1)[0]
    return format(number.normalize(), "f")


def extract_answer(text: str) -> str | None:
    patterns = [
        r"(?i)final\s+answer\s*(?:is|:)\s*[:=]?\s*\\?boxed\{([^}]+)\}",
        r"(?i)final\s+answer\s*(?:is|:)\s*[:=]?\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
        r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)",
        r"\\boxed\{([^}]+)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return normalize_answer(matches[-1])
    numbers = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", text)
    return normalize_answer(numbers[-1]) if numbers else None


def prompt_for(question: str, template: str | None = None) -> str:
    default = (
        "Solve the following problem step by step. "
        "End with `Final answer: <number>`.\n\n{question}"
    )
    return (template or default).format(question=question)


def completion_for(rationale: str | None, answer: str) -> str:
    if rationale:
        body = rationale_body(rationale)
        return f"{body}\nFinal answer: {answer}" if body else f"Final answer: {answer}"
    return f"Final answer: {answer}"


def rationale_body(text: str) -> str:
    body = text.strip()
    return re.sub(
        r"(?is)\s*(?:final\s+answer\s*(?:is|:)|####)\s*[:=]?\s*.*$", "", body
    ).strip()


def seeded_rng(seed: int, namespace: str = "") -> random.Random:
    stable = sum((index + 1) * ord(char) for index, char in enumerate(namespace))
    return random.Random(seed + stable)
