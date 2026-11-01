from fyp_distill.prepare_refinement_math import stratified_take


def test_stratified_partitions_are_disjoint_reproducible_and_exact_size():
    rows = [{"id": str(i), "subject": "algebra" if i % 2 else "geometry", "level": 1 + i % 3}
            for i in range(40)]
    selected, remaining = stratified_take(rows, 11, 7)
    assert len(selected) == 11
    assert len(remaining) == 29
    assert not {r["id"] for r in selected} & {r["id"] for r in remaining}
    assert stratified_take(list(reversed(rows)), 11, 7) == (selected, remaining)
