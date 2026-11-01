"""Real-tokenizer material assembly, without any variant/trial dependency."""
import copy
import unittest

from full_selector_materials import build_bundles, PROTOCOL
from runtime_full import ROOT, BASE
import test_material_builder as fixtures


class FullSelectorMaterialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
            local_files_only=True, trust_remote_code=False)

    def setUp(self):
        self.f = fixtures.MaterialBuilderTests(); self.f.setUp()
        sid = self.f.source['source_id']
        self.sources = {sid: self.f.source}; self.teachers = {sid: self.f.teacher}
        self.reviews = {sid: self.f.review}; self.cases = [self.f.case]
        self.pools = {'joint': {sid: self.f.pool}, 'nll': {sid: self.f.nll_pool}}

    def build(self):
        return build_bundles(self.sources, self.teachers, self.reviews,
                             self.cases, self.pools, self.tokenizer)

    def test_real_tokenizer_three_bundles_no_transfer_or_variant_requirements(self):
        nll = self.f.add_candidate('relation', selector='nll')
        joint = self.f.add_candidate('contrastive', step_slot=1)
        bundles, choices, coverage = self.build()
        self.assertEqual(set(bundles), {'M1', 'NLL', 'M3'})
        self.assertEqual(choices[0]['selected_candidate_ids'],
                         {'M1': 'e0', 'NLL': nll, 'M3': joint})
        for group, bundle in bundles.items():
            self.assertEqual(bundle['protocol'], PROTOCOL)
            self.assertEqual(coverage['groups'][group]['root_exposures'], 4)
            epochs = bundle['rows'][0]['epoch_materials']
            self.assertTrue(all(m == epochs[0] for m in epochs))
            self.assertGreater(epochs[0]['completion_tokens_including_eos'], 0)
        self.assertFalse(coverage['variant_or_trial_inputs_read'])
        self.assertFalse(coverage['formal_training_enabled'])

    def test_missing_root_or_invalid_NLL_scores_cannot_become_selective_exclusions(self):
        self.pools['nll'] = {}
        with self.assertRaises(ValueError): self.build()
        self.setUp()
        self.f.nll_pool['position_proposal']['scores'].pop()
        with self.assertRaises(ValueError): self.build()
        self.setUp()
        self.cases[0]['teacher_output_sha256'] = 'different_teacher'
        with self.assertRaises(ValueError): self.build()


if __name__ == '__main__':
    unittest.main()
