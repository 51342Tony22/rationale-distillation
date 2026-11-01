"""Calibration protocol: first answer, bounded to the original solution.

The frozen v2 implementation stays available for historical runs. No gold answer
or arbitrary number extraction is used here. Ambiguous plain prose stays invalid.
"""
from __future__ import annotations

import re

PROTOCOL_VERSION = "first_answer_v3"
FINAL_MARKER = re.compile(
    r"(?i)(?<![\w`])(?:\*\*)?Final[ \t]+answer(?:\*\*)?[ \t]*"
    r"(?::|is)[ \t]*(?:\*\*)?[ \t]*(?:\r?\n[ \t]*)?([^\r\n]*)"
)
NEW_TURN = re.compile(
    r"(?im)(?:^[ \t]*Solve the following problem step by step|"
    r"(?<!\w)User(?: Query)?\s*:)"
)


def original_solution(text: str) -> str:
    # A repeated instruction or user turn bounds the original task. An
    # 'Assistant:' label alone can be a continuation of the same solution.
    for match in NEW_TURN.finditer(text):
        if text[:match.start()].strip():
            return text[:match.start()]
    return text


def boxed_spans(text: str):
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    yield match.start(), index + 1, text[match.end():index].strip()
                    break


def clean_answer(answer: str) -> str | None:
    answer = re.split(r"(?i)\s+(?:User|Assistant)\s*:", answer, maxsplit=1)[0]
    answer = answer.strip().rstrip(".。").strip("*` ").rstrip(".。").strip()
    for opener, closer in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
        if answer.startswith(opener):
            end = answer.find(closer, len(opener))
            if end >= 0:
                answer = answer[len(opener):end].strip()
                break
    boxes = list(boxed_spans(answer))
    if boxes:
        # Do not replace an earlier unboxed value by a contradictory later box.
        prefix = answer[:boxes[0][0]]
        return None if re.search(r"\d", prefix) else boxes[0][2] or None
    for token in (r"\(", r"\)", r"\[", r"\]"):
        answer = answer.replace(token, "")
    answer = answer.strip().strip("$<>*` ")
    if not answer or re.search(r"(?i)\b(?:answer|assistant|user)\b", answer):
        return None
    return answer


def first_final_answer(text: str) -> str | None:
    body = original_solution(text)
    match = FINAL_MARKER.search(body)
    boxes = list(boxed_spans(body))
    # A box is the model's explicit answer commitment, even if it later
    # contradicts it. Never choose a later answer because it matches the gold.
    if boxes and (match is None or boxes[0][0] < match.start()):
        return boxes[0][2] or None
    return clean_answer(answer_text(body, match)) if match else None


def answer_text(body: str, match) -> str:
    value = match.group(1).strip()
    for opener, closer in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$")):
        if value.startswith(opener):
            tail = body[match.start(1):].lstrip()
            end = tail.find(closer, len(opener))
            if end >= 0:
                return tail[:end + len(closer)]
            boxes = list(boxed_spans(tail))
            if boxes:
                return tail[:boxes[0][1]]
            return ""
    return value


def completed_answer_end(text: str) -> int | None:
    body = original_solution(text)
    match = FINAL_MARKER.search(body)
    boxes = list(boxed_spans(body))
    if boxes and (match is None or boxes[0][0] < match.start()):
        return boxes[0][1]
    if match and clean_answer(answer_text(body, match)) is not None:
        if boxes:
            return boxes[0][1]
        if match.end() < len(body) and body[match.end()] in "\r\n":
            return match.end()
    # Stop when the model switches to a new task, even without a valid answer.
    if len(body) < len(text):
        return len(body)
    return None


def generate_single_solution(model, tokenizer, prompt, max_tokens: int, sampler=None) -> dict:
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
            "finish_reason": "answer_or_turn_boundary" if stopped else last.finish_reason if last else "empty",
            "elapsed_seconds": time.monotonic() - started, "answer_protocol": PROTOCOL_VERSION}
