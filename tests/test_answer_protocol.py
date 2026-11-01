import pytest

from fyp_distill.answer_protocol import completed_answer_end, first_final_answer
from fyp_distill.refinement import correct


@pytest.mark.parametrize("text", [
    r"Final answer: \(x = 2\)",
    r"**Final answer:** \(2\)",
    r"Final answer: $\boxed{2}$",
    "Therefore it follows. Final answer: 2",
    r"Final answer: \(2\) Step-by-step reasoning process:",
])
def test_equivalent_math_answer_formats_are_not_reasoning_failures(text):
    assert correct(text, "2", "math")


def test_later_invented_problem_does_not_replace_original_answer():
    text = "Final answer: 20 miles.\nUser: Another problem.\nFinal answer: 7"
    assert correct(text, r"20\text{ miles}", "math")
    assert not correct("Final answer: 7\nFinal answer: 20", "20", "math")


def test_answer_stop_waits_for_complete_nonempty_line():
    assert completed_answer_end("Final answer: ") is None
    assert completed_answer_end("Final answer: 2") is None
    text = "work\nFinal answer: 20\nUser: another problem"
    end = completed_answer_end(text)
    assert text[:end] == "work\nFinal answer: 20"
    assert completed_answer_end("Final answer:\n") is None


def test_expression_answer_preserves_operators_and_no_numeric_fallback():
    assert first_final_answer(r"Final answer: \(2xy + 10x + 20y + 100\)") == "2xy + 10x + 20y + 100"
    assert first_final_answer("The calculation gave 20") is None
    assert not correct("The calculation gave 20", "20", "math")
