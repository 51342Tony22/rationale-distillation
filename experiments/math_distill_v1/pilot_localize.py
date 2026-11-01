"""Restrict pilot NLL candidates to complete mathematical lines/display blocks.

Reuse the already tested sparse capture implementation without editing the
signed teacher-generation source. This adapter and rule are signed explicitly.
"""
from __future__ import annotations

from pathlib import Path
import re

import pilot_materials as material

MATH = re.compile(r"=|\\(?:frac|dfrac|times|div)|\b(?:add|subtract|multiply|divide|twice|half|sum|difference|total|remaining)\b", re.I)


def mathematical_steps(text):
    blocks = [(m.start(), m.end()) for m in re.finditer(r"\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$", text)]
    spans = list(blocks)
    spans += [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)
              if not any(m.start() < end and m.end() > start for start, end in blocks)]
    selected = []
    for start, end in sorted(spans):
        value = text[start:end]
        without_answer = re.sub(r"\\boxed\{[^{}]*\}", "", value)
        if MATH.search(without_answer) and (re.search(r"\d", without_answer) or "=" in without_answer):
            selected.append({"start": start, "end": end, "text": value})
    return selected


def main():
    original_sign = material.sign

    def signed(stage, files, extra=None):
        rules = dict(extra or {})
        rules["candidates"] = "Complete math-bearing display blocks or full lines with numbers/relations; boxed-only answers, introductions, standalone headings and TeX delimiter-only lines excluded"
        rules["excluded_lexical_noise"] = "Original English stopwords plus TeX control names"
        return original_sign(stage, [*files, Path(__file__)], rules)

    material.steps = mathematical_steps
    material.STOP.update("text frac dfrac times div boxed left right begin end mathrm textbf".split())
    material.sign = signed
    material.locate()


if __name__ == "__main__":
    main()
