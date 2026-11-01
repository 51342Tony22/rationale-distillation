"""Engineering correction: enumerators and headings are not math transitions."""
from __future__ import annotations

from pathlib import Path

import pilot_materials as material
import pilot_localize as initial


def mathematical_steps(text):
    # Every candidate contains an actual equality. This deliberately limits the
    # pilot to explicit equality-bearing steps, not all possible reasoning gaps.
    return [span for span in initial.mathematical_steps(text) if "=" in span["text"]]


def main():
    original_sign = material.sign

    def signed(stage, files, extra=None):
        rules = dict(extra or {})
        rules["candidates"] = "Complete equality-bearing mathematical display blocks or full lines; numbered task plans, headings, introductions without equations and boxed-only answers excluded"
        rules["revision_reason"] = "Initial heuristic admitted list enumerators as mathematical numbers; initial diagnostics archived without training or edit generation"
        rules["excluded_lexical_noise"] = "English stopwords plus TeX control names"
        return original_sign(stage, [*files, Path(initial.__file__), Path(__file__)], rules)

    material.steps = mathematical_steps
    material.STOP.update("text frac dfrac times div boxed left right begin end mathrm textbf".split())
    material.sign = signed
    material.locate()


if __name__ == "__main__":
    main()
