"""One-solution answer protocol shared by new diagnosis and evaluation runs."""
from __future__ import annotations

import re

PROTOCOL_VERSION = "first_final_line_v2"
FINAL_MARKER = re.compile(
    r"(?i)(?<![\w`])(?:\*\*)?Final[ \t]+answer(?:\*\*)?[ \t]*"
    r"(?::|is)[ \t]*(?:\*\*)?[ \t]*([^\r\n]*)"
)


def completed_answer_end(text: str) -> int | None:
    """Stop only after the first nonempty answer line has actually ended."""
    match = FINAL_MARKER.search(text)
    if not match or not match.group(1).strip():
        return None
    if match.end() < len(text) and text[match.end()] in "\r\n":
        return match.end()
    return None


def first_final_answer(text: str) -> str | None:
    match = FINAL_MARKER.search(text)
    if not match:
        return None
    answer = re.split(r"(?i)\s+(?:User|Assistant)\s*:", match.group(1), maxsplit=1)[0]
    answer = answer.strip().strip("* ").rstrip(".。")
    # An explicit mathematical span is the answer. Later prose on the same
    # line (e.g. 'Step-by-step reasoning process:') is not part of the value.
    for opener, closer in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
        if answer.startswith(opener):
            end = answer.find(closer, len(opener))
            if end >= 0:
                answer = answer[len(opener):end].strip()
                break
    for token in (r"\(", r"\)", r"\[", r"\]"):
        answer = answer.replace(token, "")
    answer = answer.strip().strip("$<>").strip()
    if not answer or answer.casefold() in ("answer", "<answer>"):
        return None
    from .math_grader import _last_balanced_command
    return _last_balanced_command(answer) or answer


def generate_single_solution(model, tokenizer, prompt, max_tokens: int, sampler=None) -> dict:
    """New-run generation only; preserve the exact answer and stop after its line."""
    import time
    from mlx_lm import stream_generate
    started = time.monotonic()
    text, last, stopped = "", None, False
    kwargs = {"sampler": sampler} if sampler is not None else {}
    for last in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        text += last.text
        end = completed_answer_end(text)
        if end is not None:
            text, stopped = text[:end], True
            break
    return {"text": text, "generation_tokens": last.generation_tokens if last else 0,
            "finish_reason": "final_answer" if stopped else last.finish_reason if last else "empty",
            "elapsed_seconds": time.monotonic() - started, "answer_protocol": PROTOCOL_VERSION}
