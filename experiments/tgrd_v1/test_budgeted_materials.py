"""Budget isolation checks; all synthetic objects remain in test memory."""
import copy
import unittest

from budgeted_materials import (PROTOCOL, selector_root_materials,
    transfer_root_materials, validate_roster, validate_transfer_coverage)
import test_material_builder as fixtures
from runtime_full import digest


class BudgetedMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MaterialBuilderTests()
        self.fixture.setUp()
        self.sid = self.fixture.source['source_id']
        self.ids = [f'gsm8k:train:{i}' for i in range(300) if f'gsm8k:train:{i}' != self.sid]
        self.common = [self.sid] + self.ids
        self.roster = {'protocol': PROTOCOL, 'seed': 42, 'root_budget': 128,
            'source_ids': self.ids[:128], 'excluded_development_ids': [],
            'frozen_before_transfer_results': True, 'public_test_used': False,
            'replacement_on_failure': False, 'sampling_rule': 'synthetic_test_only',
            'sampling_inputs_sha256': {'synthetic_input': 'not_an_experiment'}}

    def args(self):
        f = self.fixture
        return (f.source, f.teacher, f.review, f.pool, f.steps)

    def transfer(self, **kwargs):
        return transfer_root_materials(*self.args(), nll_pool=self.fixture.nll_pool,
            roster=self.roster, common_root_ids=self.common, **kwargs)

    def test_selector_stage_needs_no_variant_or_trial_results(self):
        cid = self.fixture.add_candidate('relation')
        rows, choice = selector_root_materials(*self.args(), nll_pool=self.fixture.nll_pool)
        self.assertEqual(set(rows), {'M1', 'NLL', 'M3'})
        self.assertEqual(choice['selected_candidate_ids']['M3'], cid)
        self.assertFalse(choice['trial_inputs_required'])

    def test_outside_budget_is_exact_G3_and_rejects_extra_trial_work(self):
        self.fixture.add_candidate('relation')
        baseline, _ = selector_root_materials(*self.args(), nll_pool=self.fixture.nll_pool)
        row, receipt = self.transfer()
        self.assertEqual(row, baseline['M3'])
        self.assertEqual(row['epoch_materials'][0]['kind'], 'explained_CoT')
        self.assertFalse(receipt['transfer_budget_member'])
        with self.assertRaises(ValueError):
            self.transfer(variant_bundle=self.fixture.variant)

    def test_inside_budget_keeps_real_trials_and_original_abstention(self):
        self.roster['source_ids'][0] = self.sid
        self.fixture.add_candidate('relation')
        with self.assertRaises(ValueError):
            self.transfer(variant_bundle=self.fixture.variant)
        inputs, decisions = self.fixture.trial()
        row, receipt = self.transfer(variant_bundle=self.fixture.variant,
                                     trial_inputs=inputs, trial_decisions=decisions)
        self.assertEqual(row['epoch_materials'][0]['kind'], 'explained_CoT')
        self.assertTrue(receipt['transfer_budget_member'])
        row, receipt = self.transfer()
        self.assertEqual(row['epoch_materials'][0]['kind'], 'ordinary_CoT')
        self.assertEqual(receipt['fallback_reason'], 'no_qualified_complete_6plus6_probes')

    def test_roster_cannot_shrink_replace_or_include_public_or_development_roots(self):
        for change in (lambda r: r['source_ids'].pop(),
                       lambda r: r['source_ids'].__setitem__(0, 'gsm8k:test:1'),
                       lambda r: r['source_ids'].__setitem__(0, r['source_ids'][1]),
                       lambda r: r.update(public_test_used=True),
                       lambda r: r.update(replacement_on_failure=True),
                       lambda r: r.update(excluded_development_ids=[r['source_ids'][0]])):
            bad = copy.deepcopy(self.roster)
            change(bad)
            with self.assertRaises(ValueError): validate_roster(bad, self.common)

    def test_complete_bundle_rejects_any_off_budget_completion_change(self):
        def bundle(group):
            rows = []
            for sid in self.common:
                question = 'Synthetic question for ' + sid
                m = {'root_source_id': sid, 'question': question,
                    'question_sha256': digest(question), 'completion': 'Synthetic answer',
                    'completion_sha256': digest('Synthetic answer'), 'kind': 'ordinary_CoT'}
                rows.append({'source_id': sid, 'question_sha256': digest(question),
                             'epoch_materials': [copy.deepcopy(m) for _ in range(4)]})
            return {'schema': 'tgrd_training_materials_v1', 'protocol': PROTOCOL,
                    'condition': group, 'candidate_root_count': 7473, 'rows': rows}
        g3, g4 = bundle('M3'), bundle('M7')
        validate_transfer_coverage(g3, g4, self.roster)
        for m in g4['rows'][0]['epoch_materials']:
            m['completion'] = 'Changed outside the budget'
            m['completion_sha256'] = digest(m['completion'])
        with self.assertRaises(ValueError): validate_transfer_coverage(g3, g4, self.roster)


if __name__ == '__main__':
    unittest.main()
