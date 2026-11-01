import unittest

from selector_continuation import continuation_prompt
from selector_continuation_report import position_results


class ContinuationChecks(unittest.TestCase):
    def test_student_prompt_has_prefix_not_current_or_future_teacher_steps(self):
        item = {"source": {"question": "How many?", "gold": "SECRET_GOLD"},
                "prefix": "KNOWN_PREFIX\n", "step": {"text": "CURRENT_GOLD_STEP"},
                "teacher": {"output": "KNOWN_PREFIX\nCURRENT_GOLD_STEP\nFUTURE_GOLD"}}
        without = continuation_prompt(item)
        with_explanation = continuation_prompt(item, "LOCAL_EXPLANATION")
        self.assertTrue(with_explanation.startswith(without))
        self.assertTrue(with_explanation.endswith("LOCAL_EXPLANATION\n"))
        for forbidden in ["CURRENT_GOLD_STEP", "FUTURE_GOLD", "SECRET_GOLD"]:
            self.assertNotIn(forbidden, with_explanation)
            self.assertNotIn(forbidden, without)

    def paired(self):
        return [{"position_id": "q:step:0", "source_id": "q", "step_index": 0,
                 "replicate": rep, "condition": condition, "seed": rep + 7,
                 "correct": condition == "explanation", "v2_correct": True}
                for rep in range(2) for condition in ["no_explanation", "explanation"]]

    def test_position_gain_is_paired_difference_not_raw_accuracy(self):
        result = position_results(self.paired(), 2)["q:step:0"]
        self.assertEqual(result["delta"], 1)
        self.assertEqual(result["delta_v2"], 0)

    def test_partial_or_duplicate_pairs_are_rejected(self):
        rows = self.paired()
        for bad in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(AssertionError):
                position_results(bad, 2)

    def test_mismatched_pair_seeds_are_rejected(self):
        rows = self.paired()
        rows[0]["seed"] = 99
        with self.assertRaises(AssertionError):
            position_results(rows, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
