"""No model loading: cross-selector identity and immutable pilot reuse checks."""
import copy
import unittest
from unittest.mock import patch

import explanation_teacher_cal34 as diagnostic


class BoundedExplainerTests(unittest.TestCase):
    def request(self, cid='root:step0:relation', rank=0):
        return {**{k: k for k in diagnostic.IDENTITY},
                'candidate_id': cid, 'position_rank': rank}

    def test_cross_selector_ranks_share_one_output_without_losing_unique_positions(self):
        first = self.request(); other_rank = self.request(rank=1)
        unique = self.request('root:step1:relation')
        merged = diagnostic.merge_requests([[first], [other_rank, unique]])
        self.assertEqual(merged, [first, unique])

    def test_same_candidate_cannot_reuse_changed_question_step_type_or_prompt(self):
        first = self.request()
        for key in diagnostic.IDENTITY:
            changed = copy.deepcopy(first); changed[key] = 'changed'
            with self.assertRaises(ValueError):
                diagnostic.merge_requests([[first], [changed]])

    def test_pilot_reuse_rejects_hash_drift_before_reading_any_generated_rows(self):
        reads = [[], [], {'inputs': {'review.json': 'expected'}}]
        with patch.object(diagnostic, 'read', side_effect=reads), \
                patch.object(diagnostic.baseline, 'sha', return_value='changed'), \
                patch.object(diagnostic, 'checked_rows') as generated:
            with self.assertRaisesRegex(ValueError, 'semantic evidence changed'):
                diagnostic.pilot_reuse([])
            generated.assert_not_called()


if __name__ == '__main__':
    unittest.main()
