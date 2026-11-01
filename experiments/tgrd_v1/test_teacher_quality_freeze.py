"""Scientific quality gate rejection checks; synthetic decisions only."""
import copy
import unittest

from freeze_teacher_quality import assemble, collect_reviews, decision_digest, PROTOCOL
import test_full_selector as fixtures


class TeacherQualityFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FullSelectorTests.setUpClass()
        cls.sources = fixtures.FullSelectorTests.sources
        cls.teachers = fixtures.FullSelectorTests.teachers
        cls.reviews = fixtures.FullSelectorTests.reviews

    def policy(self, reviews=None):
        return {'schema': 'tgrd_common_teacher_quality_policy_v1',
            'status': 'frozen', 'protocol': PROTOCOL, 'candidate_root_count': 7473,
            'all_candidate_decisions_consistency_reviewed': True, 'public_test_used': False,
            'numeric_match_substitutes_for_full_text_review': False,
            'teacher_or_gold_silently_edited': False,
            'decision_set_sha256': decision_digest(self.reviews if reviews is None else reviews),
            'rules': {k: 'Synthetic policy fixture only, never a real quality review.' for k in (
                'arithmetic_and_units', 'all_text_and_self_correction',
                'source_ambiguity_and_entity_identity', 'calendar_conventions',
                'gold_units_and_reference_errors', 'truncation_and_context',
                'shared_root_exclusions')}}

    def assemble(self, reviews=None, policy=None, teachers=None):
        return assemble(self.sources, self.teachers if teachers is None else teachers,
            [self.reviews if reviews is None else reviews], self.policy() if policy is None else policy,
            {'synthetic_policy.json': 'synthetic_digest'}, 'synthetic_policy.json')

    def test_complete_source_order_and_numeric_match_do_not_promote_other_roots(self):
        value = self.assemble(list(reversed(self.reviews)))
        self.assertEqual(value['accepted_root_ids'], ['gsm8k:train:0', 'gsm8k:train:1'])
        self.assertEqual([r['source_id'] for r in value['reviews']],
                         [s['source_id'] for s in self.sources])
        self.assertFalse(value['legacy_reviews_auto_promoted'])

    def test_missing_review_or_legacy_or_duplicate_cannot_fill_coverage(self):
        with self.assertRaises(ValueError):
            self.assemble(self.reviews[:-1])
        legacy = dict(self.reviews[-1]); legacy.pop('all_text_read')
        with self.assertRaises(ValueError):
            self.assemble([*self.reviews[:-1], legacy])
        with self.assertRaises(ValueError):
            collect_reviews([self.reviews, [self.reviews[0]]])

    def test_policy_must_bind_exact_decisions_and_actual_consistency_review(self):
        for change in (lambda p: p.update(status='draft'),
                       lambda p: p.update(all_candidate_decisions_consistency_reviewed=False),
                       lambda p: p.update(public_test_used=True),
                       lambda p: p['rules'].pop('calendar_conventions')):
            policy = self.policy(); change(policy)
            with self.assertRaises(ValueError): self.assemble(policy=policy)
        changed = copy.deepcopy(self.reviews); changed[0]['accepted'] = False
        with self.assertRaises(ValueError): self.assemble(changed)

    def test_valid_policy_cannot_override_original_gold_or_teacher_binding(self):
        teachers = copy.deepcopy(self.teachers); teachers[0]['correct'] = False
        with self.assertRaises(ValueError): self.assemble(teachers=teachers)
        changed = copy.deepcopy(self.reviews); changed[0]['output_sha256'] = 'wrong_output'
        with self.assertRaises(ValueError): self.assemble(changed, self.policy(changed))


if __name__ == '__main__':
    unittest.main()
