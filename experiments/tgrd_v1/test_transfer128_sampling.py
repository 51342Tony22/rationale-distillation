"""Training-only proportional sampling without consulting success outcomes."""
import copy
import json
import unittest

from budgeted_materials import PROTOCOL
from transfer128_sampling import draw


class TransferSamplingTests(unittest.TestCase):
    def fixtures(self):
        ids = [f'gsm8k:train:{i}' for i in range(220)]
        questions = ['What percent remains?', 'How many per day?', 'How many more than before?',
                     'Share equally into groups.', 'How many total?']
        sources = {sid: {'question': questions[i % 5]} for i, sid in enumerate(ids)}
        rows = [{'source_id': sid, 'selector': 'nll', 'protocol': PROTOCOL,
                 'steps': [{'mean_nll': i / 100}] if i % 7 else [],
                 'proposal': {'attention_used': False, 'masking_used': False,
                              'mask_control_filter_used': False}}
                for i, sid in enumerate(ids)]
        return sources, rows, ids, ids[:20]

    def test_roundtrip_reproducibility_exact_budget_exclusions_and_proportions(self):
        args = self.fixtures(); result = draw(*args)
        roundtripped = json.loads(json.dumps(args))
        roundtripped[0] = dict(reversed(list(roundtripped[0].items())))
        self.assertEqual(result, draw(*roundtripped))
        self.assertEqual(len(set(result['source_ids'])), 128)
        self.assertFalse(set(result['source_ids']) & set(args[3]))
        self.assertEqual(sum(s['allocated'] for s in result['strata'].values()), 128)
        for s in result['strata'].values():
            self.assertLess(abs(s['allocated'] - 128*s['available']/200), 1)

    def test_empty_candidates_remain_a_sampling_stratum(self):
        result = draw(*self.fixtures())
        self.assertTrue(any('/no_eligible_step' in key for key in result['strata']))
        self.assertTrue(any(result['labels'][sid]['difficulty'] == 'no_eligible_step'
                            for sid in result['source_ids']))

    def test_trial_success_and_joint_control_fields_do_not_change_the_draw(self):
        args = self.fixtures(); expected = draw(*args)
        for i, row in enumerate(args[1]):
            row.update(joint_control_valid=i % 2 == 0, trial_success=False,
                       variant_available=False, explanation_accepted=False)
        self.assertEqual(expected, draw(*args))

    def test_missing_nll_control_filtered_scores_bad_numbers_and_small_inventory_fail(self):
        for modify in (lambda a: a[1].pop(),
                       lambda a: a[1][30]['proposal'].update(mask_control_filter_used=True),
                       lambda a: a[1][30]['steps'][0].update(mean_nll=float('nan')),
                       lambda a: a.__setitem__(3, a[2][:100])):
            args = list(copy.deepcopy(self.fixtures())); modify(args)
            with self.assertRaises(ValueError): draw(*args)


if __name__ == '__main__': unittest.main()
