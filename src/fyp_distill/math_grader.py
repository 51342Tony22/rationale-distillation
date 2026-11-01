"""MATH-500 answer extraction and mathematical-equivalence grading.

The equivalence rules are adapted from OpenAI PRM800K's official grader:
https://github.com/openai/prm800k/tree/main/prm800k/grading
"""

from __future__ import annotations

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from .math_normalize import normalize_math_answer


BAD_SUBSTRINGS = ("^{", "^(")
BAD_REGEXES = (r"\^[0-9]+\^", r"\^[0-9][0-9]+")
TUPLE_CHARS = "()[]"


def _last_balanced_command(text: str, commands: tuple[str, ...] = ("\\boxed", "\\fbox")) -> str | None:
    """Return the contents of the last balanced LaTeX box command."""
    starts: list[tuple[int, str]] = []
    for command in commands:
        starts.extend((match.start(), command) for match in re.finditer(re.escape(command), text))
    for start, command in sorted(starts, reverse=True):
        brace = text.find("{", start + len(command))
        if brace < 0:
            continue
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1 : index].strip()
    return None


def extract_math_answer(text: str) -> str | None:
    boxed = _last_balanced_command(text)
    if boxed:
        return boxed
    final_matches = re.findall(
        r"(?im)(?:final\s+answer|answer)\s*(?:is|:)\s*[:=]?\s*([^\n]+)", text
    )
    if final_matches:
        value = final_matches[-1].strip().strip("$ ").rstrip(".。")
        return value or None
    dollar_matches = re.findall(r"\$([^$\n]+)\$", text)
    if dollar_matches:
        return dollar_matches[-1].strip()
    return None


def _sympy_parse(expr: str):
    return sympy_parser.parse_expr(
        expr.replace("^", "**"),
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    value = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    value = value.replace("\\frac", " \\frac")
    value = latex2text.LatexNodes2Text().latex_to_text(value)
    replacements = {
        "√": "sqrt",
        "π": "pi",
        "∞": "inf",
        "∪": "U",
        "·": "*",
        "×": "*",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value.strip()


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_int(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except (OverflowError, ValueError):
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        updated = pattern.sub(r"\1\3\4", expr)
        if updated == expr:
            return updated
        expr = updated


def _str_is_int(value: str) -> bool:
    try:
        number = float(_strip_properly_formatted_commas(value))
        return abs(number - int(round(number))) <= 1e-7
    except (OverflowError, ValueError):
        return False


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    match = re.fullmatch(r"\\text\{(.+?)\}", expr)
    if match:
        expr = match.group(1)
    expr = expr.replace("\\%", "%").replace("\\$", "$")
    expr = expr.replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in (
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ):
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub(r"- *", "-", expr)
    expr = re.sub(r"([0-9]) +([0-9])", r"\1+\2", expr)
    expr = expr.replace(" ", "").replace("{", "").replace("}", "").lower()
    if _str_is_int(expr):
        expr = str(int(float(expr.replace(",", ""))))
    return expr


def _count_unknown_letters(expr: str) -> int:
    value = expr.replace("sqrt", "").replace("frac", "")
    return len({character for character in value if character.isalpha()})


def _allow_eval(expr: str) -> bool:
    return (
        _count_unknown_letters(expr) <= 2
        and not any(value in expr for value in BAD_SUBSTRINGS)
        and not any(re.search(pattern, expr) for pattern in BAD_REGEXES)
    )


def _equal_under_sympy(ground_truth: str, prediction: str) -> bool:
    try:
        difference = f"({ground_truth})-({prediction})"
        return _allow_eval(difference) and sympy.simplify(_sympy_parse(difference)) == 0
    except Exception:
        return False


def _split_tuple(expr: str) -> list[str]:
    value = _strip_properly_formatted_commas(expr)
    if not value:
        return []
    if (
        len(value) > 2
        and value[0] in TUPLE_CHARS
        and value[-1] in TUPLE_CHARS
        and all(character not in value[1:-1] for character in TUPLE_CHARS)
    ):
        return [element.strip() for element in value[1:-1].split(",")]
    return [value]


def grade_math_answer(prediction: str | None, ground_truth: str) -> bool:
    if prediction is None:
        return False
    truth_mathd = normalize_math_answer(ground_truth)
    prediction_mathd = normalize_math_answer(prediction)
    if truth_mathd == prediction_mathd:
        return True
    truth = _normalize(ground_truth)
    predicted = _normalize(prediction)
    if truth is None or predicted is None:
        return False
    if truth == predicted:
        return True
    if not predicted:
        return False
    truth_elements = _split_tuple(truth)
    predicted_elements = _split_tuple(predicted)
    if len(truth_elements) > 1 and (
        truth[0] != predicted[0] or truth[-1] != predicted[-1]
    ):
        return False
    if len(truth_elements) != len(predicted_elements):
        return False
    for truth_element, predicted_element in zip(truth_elements, predicted_elements):
        if _is_frac(truth_element) and _is_frac(predicted_element):
            equal = truth_element == predicted_element
        elif _str_is_int(truth_element) != _str_is_int(predicted_element):
            equal = False
        else:
            equal = _equal_under_sympy(truth_element, predicted_element)
        if not equal:
            return False
    return True
