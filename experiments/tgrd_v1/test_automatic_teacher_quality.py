"""Automatic eligibility cannot impersonate a semantic review or change roots."""
import copy
import unittest
from unittest.mock import patch

import automatic_teacher_quality as auto
from runtime_full import digest


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class AutomaticTeacherQualityTests(unittest.TestCase):
    def setUp(self):
        self.source = {'source_id': 'gsm8k:train:1', 'question': 'What is two plus three?',
            'gold': '5', 'question_sha256': digest('What is two plus three?')}
        self.teacher = {**{k: self.source[k] for k in ('source_id', 'question_sha256', 'gold')},
            'output': r'Two plus three is five. \boxed{5}', 'finish_reason': 'stop', 'correct': True}
        self.teacher['output_sha256'] = digest(self.teacher['output'])
        self.tok = TinyTokenizer()

    def test_numeric_pass_is_explicitly_not_a_semantic_claim(self):
        self.teacher['output'] = r'Incorrect intermediate reasoning, then \boxed{5}'
        self.teacher['output_sha256'] = digest(self.teacher['output'])
        row = auto.decision(self.source, self.teacher, self.tok)
        self.assertTrue(row['accepted'])
        self.assertFalse(row['all_text_read'])
        self.assertFalse(row['semantic_correctness_verified'])
        auto.validate_decision(self.source, self.teacher, row, self.tok)
        for key in ('all_text_read', 'semantic_correctness_verified'):
            bad = {**row, key: True}
            with self.assertRaises(ValueError):
                auto.validate_decision(self.source, self.teacher, bad, self.tok)

    def test_truncated_wrong_numeric_multiple_box_and_long_context_are_excluded(self):
        for output, finish, correct in ((r'Answer \boxed{5}', 'length', True),
                (r'Answer \boxed{6}', 'stop', False),
                (r'Answer \boxed{5} or \boxed{5}', 'stop', True),
                ('x' * 4100 + r'\boxed{5}', 'stop', True)):
            t = {**self.teacher, 'output': output, 'output_sha256': digest(output),
                'finish_reason': finish, 'correct': correct}
            row = auto.decision(self.source, t, self.tok)
            self.assertFalse(row['accepted'])
            auto.validate_decision(self.source, t, row, self.tok)

    def test_scores_original_labels_and_token_counts_are_recomputed(self):
        row = auto.decision(self.source, self.teacher, self.tok)
        for source, teacher, saved in (({**self.source, 'gold': '6'}, self.teacher, row),
                (self.source, {**self.teacher, 'correct': False}, row),
                (self.source, self.teacher, {**row, 'prompt_tokens': 1}),
                (self.source, self.teacher, {**row, 'output_sha256': 'changed'})):
            with self.assertRaises(ValueError):
                auto.validate_decision(source, teacher, saved, self.tok)

    def test_automatic_material_builder_still_requires_real_explanation_reviews(self):
        import test_material_builder as fixtures
        from material_builder import root_materials
        f = fixtures.MaterialBuilderTests(); f.setUp()
        row = auto.decision(f.source, f.teacher, self.tok)
        # Actual reviewed train fixture is shorter than the context limit.
        self.assertTrue(row['accepted'])
        args = (f.source, f.teacher, row, f.pool, f.steps)
        rows, choices = root_materials(*args, nll_pool=f.nll_pool)
        self.assertEqual(choices['selected_candidate_ids']['M1'], 'e0')
        self.assertEqual(rows['M1']['epoch_materials'][0]['completion'], f.teacher['output'])
        f.add_candidate('relation')
        f.pool['candidates'][-1]['review']['accepted'] = False
        with self.assertRaises(ValueError):
            root_materials(*args, nll_pool=f.nll_pool)

    def test_old_full_text_schema_cannot_accept_automatic_rows(self):
        import test_full_selector as fixtures
        from full_selector import validate_quality
        fixtures.FullSelectorTests.setUpClass()
        f = fixtures.FullSelectorTests()
        quality = f.quality()
        quality['reviews'][0]['all_text_read'] = False
        quality['reviews'][0]['review_kind'] = auto.MODE
        with self.assertRaises(ValueError):
            validate_quality(f.sources, f.teachers, quality)

    def spotcheck_fixture(self):
        ids = [f'gsm8k:train:{i}' for i in range(128)]
        sources = {sid: {**self.source, 'source_id': sid} for sid in ids}
        teachers = {sid: {**self.teacher, 'source_id': sid} for sid in ids}
        reviews = [{'source_id': sid, 'question_sha256': self.source['question_sha256'],
            'output_sha256': self.teacher['output_sha256'], 'all_text_read': True,
            'review_kind': 'assistant_full_text_mathematical_review',
            'accepted': i % 2 == 0, 'reason': 'Synthetic decision only'} for i, sid in enumerate(ids)]
        roster = {'source_ids': ids, 'root_count': 128, 'protocol': auto.PROTOCOL, 'seed': 42,
            'replacement_on_failure': False, 'transfer128_roster': False, 'public_test_used': False}
        return sources, teachers, roster, reviews

    def test_spotcheck_requires_all_fixed_roots_including_negative_reviews(self):
        from teacher_spotcheck_v2 import validate_reviews
        sources, teachers, roster, reviews = self.spotcheck_fixture()
        value = validate_reviews(sources, teachers, roster, reviews)
        self.assertEqual(value['semantic_flagged'], 64)
        self.assertFalse(value['changes_automatic_training_roster'])
        for changed in (reviews[:-1], list(reversed(reviews)), [r for r in reviews if r['accepted']]):
            with self.assertRaises(ValueError): validate_reviews(sources, teachers, roster, changed)

    def test_spotcheck_rejects_automatic_rows_and_stale_output_binding(self):
        from teacher_spotcheck_v2 import validate_reviews
        sources, teachers, roster, reviews = self.spotcheck_fixture()
        for changes in ({'all_text_read': False, 'review_kind': auto.MODE},
                        {'output_sha256': 'changed'}, {'reason': ''}):
            bad = copy.deepcopy(reviews); bad[0].update(changes)
            with self.assertRaises(ValueError): validate_reviews(sources, teachers, roster, bad)


if __name__ == '__main__':
    unittest.main()
