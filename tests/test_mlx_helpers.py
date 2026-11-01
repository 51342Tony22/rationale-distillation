from fyp_distill.train_mlx import supervision_signature, training_iterations


def test_training_iterations_preserve_effective_batch_updates() -> None:
    iterations = training_iterations(rows=585, batch_size=4, epochs=3, accumulation=2)

    assert iterations == 440
    assert iterations % 2 == 0
    assert iterations // 2 == 220


def test_identical_condition_names_do_not_create_a_distinct_treatment():
    rows = [{"id":"question:A", "source_id":"question", "prompt":"problem", "completion":"reasoning"}]
    renamed = [{**rows[0], "id":"question:E", "hint_id":None}]
    assert supervision_signature(rows) == supervision_signature(renamed)
    assert supervision_signature(rows) != supervision_signature([{**renamed[0], "completion":"reasoning plus hint"}])


def test_tiny_pilot_training_has_26_complete_updates():
    assert training_iterations(21,1,10,8)==208
