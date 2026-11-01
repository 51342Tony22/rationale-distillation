from fyp_distill.generate_rationales import estimated_cost_cny


def test_estimated_cost_uses_recorded_usage() -> None:
    records = [
        {
            "candidates": [
                {"usage": {"prompt_tokens": 100_000, "completion_tokens": 200_000}},
                {"error": "timeout"},
            ]
        }
    ]
    assert estimated_cost_cny(records, input_price=4.0, output_price=16.0) == 3.6
