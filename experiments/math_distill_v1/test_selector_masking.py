"""Meaningful alignment, leakage, control and real-kernel CPU checks."""
import statistics
import unittest

from selector_masking_core import (attention_units, delete_context, diagnose_step,
                                   exploratory_choices, percentile_ranks)


class SyntheticBackend:
    def __init__(self, sign=1):
        self.calls = []
        self.sign = sign

    def score(self, ids, targets):
        self.calls.append((list(ids), list(targets)))
        # Original token id 2 is the predeclared relevant premise.
        loss = 2 + self.sign * (2 not in ids)
        return [loss] * len(targets)

    def observe(self, ids, targets, queries):
        vector = [0.0] * len(ids)
        vector[1] = .8
        for index in range(2, 6):
            vector[index] = .05
        return self.score(ids, targets), [list(vector) for _ in queries]


def sample():
    return (list(range(1, 19)),
            {"step_index": 0, "text": "a = b", "char_start": 0, "char_end": 5,
             "start_position": 10, "targets": [12, 14]},
            [{"kind": "question", "text": "relevant", "positions": [1]},
             {"kind": "question", "text": "other", "positions": [2, 3, 4, 5]}])


class MaskingChecks(unittest.TestCase):
    def test_remapping_preserves_targets_and_predictor_shift(self):
        ids, targets = list(range(20)), [12, 14]
        changed, mapped = delete_context(ids, targets, [2, 5, 7], 10)
        self.assertEqual(mapped, [9, 11])
        self.assertEqual([changed[p] for p in mapped], [12, 14])
        self.assertEqual([changed[p - 1] for p in mapped], [11, 13])

    def test_never_delete_current_step_even_if_not_scored(self):
        for removed in [[10], [11], [12], [15], [-1]]:
            with self.assertRaises(ValueError):
                delete_context(list(range(20)), [12, 14], removed, 10)

    def test_diagnosis_uses_only_prefix_and_has_matched_distinct_controls(self):
        ids, step, units = sample()
        backend = SyntheticBackend()
        record = diagnose_step(ids, step, units, backend, seed=7, key="a")
        self.assertEqual(record["status"], "diagnostic_complete")
        self.assertEqual(record["mask_excess"], 1)
        self.assertEqual(record["predictor_queries"], [11, 13])
        self.assertEqual(record["prefix_tokens"], 15)
        self.assertEqual(len(record["random_masks"]), 3)
        self.assertEqual(len({tuple(r["removed_positions"]) for r in record["random_masks"]}), 3)
        for changed, targets in backend.calls:
            self.assertLessEqual(len(changed), 15)
            self.assertEqual([changed[p] for p in targets], [13, 15])
            self.assertNotIn(16, changed)  # gold text following this step absent
        for control in record["random_masks"]:
            self.assertEqual(control["removed_tokens"], record["attended_mask"]["removed_tokens"])
            self.assertNotIn(1, control["removed_positions"])

    def test_negative_effect_is_not_clipped_or_called_gain(self):
        record = diagnose_step(*sample(), SyntheticBackend(-1), seed=7, key="a")
        self.assertEqual(record["mask_excess"], -1)
        self.assertNotIn("explanation_gain", record)

    def test_repeatability(self):
        a = diagnose_step(*sample(), SyntheticBackend(), seed=7, key="a")
        b = diagnose_step(*sample(), SyntheticBackend(), seed=7, key="a")
        self.assertEqual(a, b)

    def test_missing_controls_keeps_candidate_and_abstains(self):
        ids, step, units = sample()
        record = diagnose_step(ids, step, units[:1], SyntheticBackend(), seed=7, key="a")
        self.assertEqual(record["status"], "insufficient_matched_controls")
        self.assertIsNone(record["mask_excess"])
        choices = exploratory_choices([record], seed=7, key="a")
        self.assertEqual(choices["nll"], 0)
        self.assertIsNone(choices["combined"])

    def test_future_or_unnormalized_attention_rejected(self):
        for vector in [[0, 1, 0, 0, 0, 0, 0, .1], [0, .9, 0, 0, 0, 0, 0, 0]]:
            with self.assertRaises(ValueError):
                attention_units([vector], [5], [{"positions": [1], "kind": "question"}], 4, 8)

    def test_attention_observation_must_not_change_prediction(self):
        class Broken(SyntheticBackend):
            def observe(self, ids, targets, queries):
                losses, weights = super().observe(ids, targets, queries)
                return [v + .01 for v in losses], weights
        with self.assertRaisesRegex(ValueError, "changed"):
            diagnose_step(*sample(), Broken(), seed=7, key="a")

    def test_rank_ties_and_empty_cases(self):
        self.assertEqual(percentile_ranks([2, 2, 4]), [.25, .25, 1])
        self.assertEqual(exploratory_choices([], seed=7, key="a")["status"], "no_candidates")


class TinyMLXChecks(unittest.TestCase):
    def test_actual_causal_attention_and_nll_against_dense_logits(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.qwen2 import Model, ModelArgs
        from selector_masking_probe import MLXBackend
        mx.set_default_device(mx.cpu)
        mx.random.seed(7)
        model = Model(ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=2,
                      intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                      rms_norm_eps=1e-6, vocab_size=64, max_position_embeddings=128,
                      tie_word_embeddings=True))
        model.eval()
        backend = MLXBackend(model, 2)
        ids, step, units = sample()
        ids = ids[:15]
        positions = step["targets"]
        dense = model(mx.array([ids[:-1]]))[:, [p - 1 for p in positions]]
        reference = nn.losses.cross_entropy(dense, mx.array([[ids[p] for p in positions]]), reduction="none")[0].tolist()
        actual = backend.score(ids, positions)
        self.assertLess(max(abs(a - b) for a, b in zip(reference, actual)), 1e-6)
        losses, weights = backend.observe(ids, positions, [p - 1 for p in positions])
        self.assertLess(max(abs(a - b) for a, b in zip(actual, losses)), 1e-6)
        attention_units(weights, [p - 1 for p in positions], units, step["start_position"], len(ids))
        record = diagnose_step(ids, step, units, backend, seed=7, key="tiny")
        self.assertTrue(record["status"] in ["diagnostic_complete", "insufficient_matched_controls"])
        self.assertTrue(all(type(layer.self_attn).__name__ != "Capture" for layer in model.layers))
        # A deletion intervention must score exactly the remapped original targets.
        removed = [1, 3]
        changed, remapped = delete_context(ids, positions, removed, step["start_position"])
        dense2 = model(mx.array([changed[:-1]]))[:, [p - 1 for p in remapped]]
        reference2 = nn.losses.cross_entropy(dense2, mx.array([[ids[p] for p in positions]]), reduction="none")[0].tolist()
        self.assertLess(abs(statistics.mean(reference2) - statistics.mean(backend.score(changed, remapped))), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
