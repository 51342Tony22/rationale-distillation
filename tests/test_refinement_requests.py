import pytest

from fyp_distill.request_refinement_hints import parse_hint, parse_original, reservation_cny


def request():
    return {"id": "r", "source_id": "s", "source_hash": "frozen", "kind": "local",
            "request_hash": "q", "allowed_offsets": [5]}


def test_response_requires_review_and_preserves_provenance():
    hint = parse_hint(request(), '```json\n{"hint": "Use the definition.", "offset": 5}\n```')
    assert hint["source_hash"] == "frozen"
    assert hint["review"] == {"valid_explanation": False, "no_answer_leak": False}


@pytest.mark.parametrize("text", ['{"hint":"h","offset":3}', '{"hint":"","offset":5}',
                                  '{"hint":"h","offset":true}', 'not json'])
def test_malformed_teacher_output_is_not_accepted(text):
    with pytest.raises(ValueError):
        parse_hint(request(), text)


def test_reservation_accounts_for_reasoning_and_utf8_input():
    assert reservation_cny("数学", 1024, 4096) > reservation_cny("math", 1024, 0)
    assert reservation_cny("数学", 1024, 4096) >= (1024 + 4096) * 16 / 1e6


def test_leading_json_can_be_recovered_without_accepting_a_second_object():
    hint = parse_hint(request(), '{"hint":"Use the definition.","offset":5}\n</think>\nExplanation of the choice.')
    assert hint["hint"] == "Use the definition."
    assert hint["response_parse"] == "leading_json_with_trailing_text"
    assert not hint["review"]["valid_explanation"]
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_hint(request(), '{"hint":"Use the definition.","offset":5}\n{"hint":"Different.","offset":5}')


def test_invalid_latex_delimiter_escapes_are_recovered_literally():
    hint = parse_hint(request(), r'{"hint":"Use \(a+b\) and \[a=b\].","offset":5}')
    assert hint["hint"] == r"Use \(a+b\) and \[a=b\]."
    assert hint["latex_delimiter_escape_repair"]
    with pytest.raises(ValueError):
        parse_hint(request(), r'{"hint":"Unknown \z escape.","offset":5}')


def test_original_math_answer_is_graded_as_expression_not_numeric_fallback():
    source = {**request(), "kind": "original", "dataset": "math", "question": "Half of one?",
              "gold_answer": "0.5", "model": "teacher"}
    correct = parse_original(source, "Divide by two.\nFinal answer: \\frac{1}{2}")
    wrong = parse_original(source, "Divide by two.\nFinal answer: 2")
    assert correct["candidates"][0]["is_usable"]
    assert not wrong["candidates"][0]["is_usable"]
