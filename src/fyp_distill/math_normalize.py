"""MATH-answer string normalization.

Adapted from OpenAI PRM800K's ``grading/math_normalize.py``, whose logic is
itself based on the Hendrycks MATH release. Keeping this separate makes the
MATH-500 metric reproducible and avoids treating equivalent LaTeX strings as
different answers.
"""

from __future__ import annotations

import re


def _fix_fracs(value: str) -> str:
    pieces = value.split("\\frac")
    result = pieces[0]
    for piece in pieces[1:]:
        result += "\\frac"
        if not piece:
            continue
        if piece[0] == "{":
            result += piece
            continue
        if len(piece) < 2:
            return value
        numerator, denominator = piece[0], piece[1]
        suffix = piece[2:]
        if denominator == "{":
            result += "{" + numerator + "}" + denominator + suffix
        else:
            result += "{" + numerator + "}{" + denominator + "}" + suffix
    return result


def _fix_simple_fraction(value: str) -> str:
    pieces = value.split("/")
    if len(pieces) != 2:
        return value
    try:
        numerator, denominator = (int(piece) for piece in pieces)
    except ValueError:
        return value
    if value != f"{numerator}/{denominator}":
        return value
    return f"\\frac{{{numerator}}}{{{denominator}}}"


def _fix_sqrt(value: str) -> str:
    if "\\sqrt" not in value:
        return value
    pieces = value.split("\\sqrt")
    result = pieces[0]
    for piece in pieces[1:]:
        if piece and piece[0] != "{":
            result += "\\sqrt{" + piece[0] + "}" + piece[1:]
        else:
            result += "\\sqrt" + piece
    return result


def normalize_math_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    value = answer.strip()
    match = re.fullmatch(r"\\text\{(.+?)\}", value)
    if match:
        value = match.group(1).strip()
    try:
        value = value.replace("\n", "").replace("\\!", "")
        value = value.replace("\\\\", "\\")
        value = value.replace("tfrac", "frac").replace("dfrac", "frac")
        value = value.replace("\\left", "").replace("\\right", "")
        value = value.replace("^{\\circ}", "").replace("^\\circ", "")
        value = value.replace("\\$", "")
        if "\\text{ " in value:
            value = value.split("\\text{ ", maxsplit=1)[0]
        value = value.replace("\\%", "").replace("\%", "")
        value = value.replace(" .", " 0.").replace("{.", "{0.")
        if not value:
            return value
        if value[0] == ".":
            value = "0" + value
        if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
            value = value.split("=", maxsplit=1)[1]
        value = _fix_sqrt(value).replace(" ", "")
        value = _fix_fracs(value)
        if value == "0.5":
            value = "\\frac{1}{2}"
        return _fix_simple_fraction(value)
    except (AssertionError, IndexError):
        return answer.strip()
