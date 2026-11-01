import pytest

from fyp_distill.answer_protocol_v3 import completed_answer_end, first_final_answer
from fyp_distill.refinement import correct


@pytest.mark.parametrize("text", [
    "Final answer: `8`", "**Final answer:**\n8", "Final answer: **8**.",
    r"Thus the answer is \(\boxed{8}\).", r"Final answer: \(x=8\)",
    r"Final answer: \(x=\boxed{8}\)", "Assistant: Final answer: 8",
    "### Final Answer:\n\\[\n\\boxed{8}\n\\]",
])
def test_unambiguous_answers_survive_markup_and_newlines(text):
    assert correct(text, "8", "math", "first_answer_v3")


def test_first_box_or_answer_commitment_wins():
    assert first_final_answer(r"\boxed{7} then Final answer: 8") == "7"
    assert first_final_answer(r"Final answer: 7 then \boxed{8}") is None
    assert not correct("Final answer: 7\nFinal answer: 8", "8", "math", "first_answer_v3")
    assert not correct(r"\boxed{7} then Final answer: 8", "8", "math", "first_answer_v3")


def test_new_problem_does_not_supply_missing_answer():
    assert first_final_answer("Work gives x=8.\nSolve the following problem step by step.\nFinal answer: 8") is None
    assert first_final_answer("Work.\nUser: new task\nFinal answer: 8") is None
    assert first_final_answer(r"\boxed{8}" + "\nUser: new task\nFinal answer: 7") == "8"
    assert first_final_answer("Final answer: 8. User: new task Final answer: 7") == "8"


def test_stop_is_not_triggered_by_incomplete_answer_or_box():
    assert completed_answer_end("Final answer: ") is None
    assert completed_answer_end("Final answer: 8") is None
    assert completed_answer_end(r"\boxed{\frac{1}{2}") is None
    assert completed_answer_end(r"\boxed{\frac{1}{2}}") == len(r"\boxed{\frac{1}{2}}")
    assert completed_answer_end("**Final answer:**\n8\nmore") == len("**Final answer:**\n8")
    assert completed_answer_end("Final answer:\n\\[\n") is None
    assert first_final_answer("Final answer:\n\\[\n\\boxed{8}") == "8"


def test_no_arbitrary_number_or_gold_guided_fallback():
    for text in ("Calculation gave 8", "Final answer: <answer>", "Final answer: 8jectoryassistant"):
        assert not correct(text, "8", "math", "first_answer_v3")
