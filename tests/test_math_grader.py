from fyp_distill.math_grader import extract_math_answer, grade_math_answer


def test_extracts_nested_boxed_answer() -> None:
    text = r"Work gives x=3. Final answer: \boxed{\left(3, \frac{\pi}{2}\right)}."
    assert extract_math_answer(text) == r"\left(3, \frac{\pi}{2}\right)"


def test_math_equivalence_handles_latex_and_spacing() -> None:
    assert grade_math_answer(r"\frac{1}{2}", "0.5")
    assert grade_math_answer("p-q", "p - q")
    assert grade_math_answer(r"\left(3,\frac{\pi}{2}\right)", r"\left( 3, \frac{\pi}{2} \right)")


def test_math_equivalence_rejects_different_answer() -> None:
    assert not grade_math_answer(r"\frac{2}{3}", r"\frac{1}{2}")
