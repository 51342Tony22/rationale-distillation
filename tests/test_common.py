from fyp_distill.common import completion_for, extract_answer, gsm8k_gold, normalize_answer, read_jsonl


def test_answer_extraction_formats():
    assert extract_answer("Reasoning.\nFinal answer: 1,234") == "1234"
    assert extract_answer("Therefore \\boxed{2.50}") == "2.5"
    assert extract_answer("work\n#### -7") == "-7"


def test_gsm8k_gold_and_normalization():
    assert gsm8k_gold("work\n#### 42") == "42"
    assert normalize_answer("$1,000.00") == "1000"
    assert normalize_answer("9" * 100) == "9" * 100


def test_completion_has_one_canonical_final_answer():
    completion = completion_for("Calculate 2+3.\nFinal answer: 5", "5")
    assert completion == "Calculate 2+3.\nFinal answer: 5"


def test_jsonl_reader_preserves_unicode_line_separator(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "before\\u2028after"}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"text": "before\u2028after"}]
