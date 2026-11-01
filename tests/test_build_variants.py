import json
from pathlib import Path

from fyp_distill.build_variants import (
    build,
    mean_pairwise_distance,
    select_compatible,
    select_diverse,
    select_low_diversity,
)


FIXTURE = Path(__file__).parent / "fixtures" / "teacher_tiny.jsonl"


def fixture_rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def test_four_variants_use_same_question_set():
    outputs, report = build(fixture_rows(), k=3, seed=42)
    expected = {"gsm8k_train_00000", "gsm8k_train_00001"}
    for rows in outputs.values():
        assert {row["source_id"] for row in rows} == expected
    assert len(outputs["answer_only"]) == 6
    assert len(outputs["single_rationale"]) == 6
    assert len(outputs["multi_rationale"]) == 6
    assert len(outputs["diverse_multi_rationale"]) == 6
    assert report["eligible_questions"] == 2


def test_ineligible_question_is_excluded_from_all_variants():
    rows = fixture_rows()
    rows[0]["candidates"] = rows[0]["candidates"][:2]
    outputs, _ = build(rows, k=3, seed=42)
    for variant_rows in outputs.values():
        assert {row["source_id"] for row in variant_rows} == {"gsm8k_train_00001"}


def test_diverse_selection_has_requested_size():
    candidates = [candidate for candidate in fixture_rows()[0]["candidates"] if candidate["is_correct"]]
    selected = select_diverse(candidates, 2)
    assert len(selected) == 2
    assert len({row["candidate_id"] for row in selected}) == 2


def test_low_and_high_diversity_form_ordered_extremes():
    candidates = [candidate for candidate in fixture_rows()[0]["candidates"] if candidate["is_correct"]]
    low = select_low_diversity(candidates, 3)
    high = select_diverse(candidates, 3)

    assert mean_pairwise_distance(low) <= mean_pairwise_distance(high)


def test_compatible_selection_uses_lowest_student_nll():
    candidates = fixture_rows()[0]["candidates"][:]
    for index, candidate in enumerate(candidates):
        candidate["compatibility_nll"] = float(len(candidates) - index)

    selected = select_compatible(candidates, 3)

    assert [row["compatibility_nll"] for row in selected] == [1.0, 2.0, 3.0]
