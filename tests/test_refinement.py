from copy import deepcopy

import pytest

from fyp_distill.refinement import (
    GROUPS, build_conditions, correct, digest, insert, segment, write_frozen,
)


def fixture():
    original = "Distance is 120.\nTime is 3.\nDivide distance by time.\nFinal answer: 40"
    boundaries = [original.index("Time"), original.index("Divide")]
    base = {"id": "train_1", "source_hash": "frozen", "question": "What is average speed?",
            "gold_answer": "40", "original": original, "boundaries": boundaries,
            "random_boundary": boundaries[0], "review": {
                "correct_steps": True, "prefixes_do_not_reveal_answer": True}}
    screen = {"id": base["id"], "source_hash": "frozen", "boundaries": {
        str(boundaries[0]): [{"correct": True, "seed": 1}, {"correct": False, "seed": 2}],
        str(boundaries[1]): [{"correct": False, "seed": 3}, {"correct": False, "seed": 4}],
    }}
    hints = []
    for index, kind in enumerate(("local", "local", "generic", "random")):
        hints.append({"id": f"h{index}", "kind": kind, "source_id": base["id"],
                      "source_hash": "frozen", "offset": boundaries[0] if kind == "random" else boundaries[1],
                      "hint": "Average speed uses total distance and total time." if index != 1 else
                              "Use total distance divided by total travel time.",
                      "review": {"valid_explanation": True, "no_answer_leak": True}})
    verify = {"id": base["id"], "source_hash": "frozen", "hint_hash": digest(hints[:2]),
              "baseline": [{"correct": False, "seed": 101}, {"correct": False, "seed": 102}],
              "hints": {
                  "h0": [{"correct": True, "seed": 101}, {"correct": True, "seed": 102}],
                  "h1": [{"correct": False, "seed": 101}, {"correct": True, "seed": 102}]}}
    settings = {"require_review": True, "max_hint_tokens": 128, "control_token_tolerance": 2,
                "local_candidates": 2, "screening_samples": 2, "verification_samples": 2, "min_gain": 0.25}
    return [base], [screen], hints, [verify], settings


def build(items):
    return build_conditions(*items, seed=7, count_tokens=lambda text: len(text.split()))


def test_paired_groups_change_only_hint_position():
    items = fixture()
    groups, audit = build(items)
    assert set(groups) == set(GROUPS)
    assert all(len(rows) == 1 and rows[0]["source_id"] == "train_1" for rows in groups.values())
    local, start = groups["local_refinement"][0], groups["hint_at_start"][0]
    assert local["hint_id"] == start["hint_id"] == "h0"
    assert local["added_tokens"] == start["added_tokens"]
    assert local["hint_offset"] == items[0][0]["boundaries"][1]
    assert start["hint_offset"] == 0
    assert groups["original_rationale"][0]["completion"] == items[0][0]["original"]
    assert all(row[0]["completion"].endswith("Final answer: 40") for row in groups.values())
    assert audit[0]["accepted"]


def test_rejected_hint_preserves_question_in_all_groups():
    items = fixture()
    items[4]["min_gain"] = 1.1
    groups, audit = build(items)
    for name in GROUPS[:-1]:
        assert groups[name][0]["completion"] == items[0][0]["original"]
    assert groups["local_unfiltered"][0]["hint_id"] is not None
    assert not audit[0]["accepted"]


@pytest.mark.parametrize("problem", ["unreviewed", "stale", "reuse_seed", "wrong_random", "length", "missing_control", "partial_screen"])
def test_invalid_experiments_fail_closed(problem):
    items = deepcopy(fixture())
    if problem == "unreviewed":
        items[0][0]["review"]["correct_steps"] = False
    elif problem == "stale":
        items[3][0]["hint_hash"] = "old"
    elif problem == "reuse_seed":
        items[3][0]["baseline"][0]["seed"] = 1
    elif problem == "wrong_random":
        items[2][3]["offset"] = items[0][0]["boundaries"][1]
    elif problem == "length":
        items[2][2]["hint"] = "short"
    elif problem == "missing_control":
        items[2].pop()
    else:
        items[1][0]["boundaries"][str(items[0][0]["boundaries"][0])].pop()
    with pytest.raises(ValueError):
        build(items)


def test_segmentation_preserves_decimals_and_excludes_final_answer():
    text = "First 3.14 is used.\nThen divide by 2.\nFinal answer: 1.57"
    offsets = segment(text, 3, 7)
    assert offsets == [text.index("Then")]
    assert all("Final answer" not in text[:x] for x in offsets)
    assert insert(text, "", offsets[0]) == text


def test_answer_requires_explicit_marker():
    assert not correct("I computed 40", "40", "gsm8k")
    assert correct("The result follows.\nFinal answer: 40", "40", "gsm8k")


def test_frozen_artifact_rejects_silent_overwrite(tmp_path):
    path = tmp_path / "frozen.jsonl"
    write_frozen(path, [{"id": "a"}])
    write_frozen(path, [{"id": "a"}])
    with pytest.raises(ValueError):
        write_frozen(path, [{"id": "b"}])
