"""Recipe isolation and actual-review guard checks; synthetic fixtures only."""
import copy
import unittest
from unittest.mock import patch

import auxiliary_explainer_recipe as recipe
from review_general_explainer import validate_reviews
from explanation_pool import REVIEW_CHECKS, mechanical_checks
from runtime_full import digest
import test_explanation_pool as fixtures


class AuxiliaryExplainerTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.ExplanationPoolTests(); self.f.setUp()
        from explanation_pool import make_request
        self.request = make_request(self.f.source, self.f.teacher, self.f.step, 'relation', 0)
        self.cid = self.request['candidate_id']
        self.output = {'candidate_id': self.cid, 'source_id': self.request['source_id'],
            'output': self.f.text, 'output_sha256': digest(self.f.text), 'finish_reason': 'stop',
            'engineering_only': True, 'scientific_pool_exported': False,
            'record_sha256': 'synthetic_digest'}
        self.output['mechanical_checks'] = mechanical_checks(self.request, self.output, self.f.tokenizer)
        self.review = {'candidate_id': self.cid, 'output_sha256': digest(self.f.text),
            'accepted': True, 'kind': 'assistant_mathematical_review', 'all_text_read': True,
            'reason': 'Synthetic test only.', 'mechanical_hard_rejection_reasons': [],
            **{k: True for k in REVIEW_CHECKS}}

    def validate(self, reviews=None, outputs=None):
        return validate_reviews([self.request], [self.output] if outputs is None else outputs,
                                [self.review] if reviews is None else reviews, self.f.tokenizer)

    def test_missing_or_stale_full_text_decision_cannot_be_promoted(self):
        self.validate()
        with self.assertRaises(ValueError): self.validate(reviews=[])
        for changes in ({'output_sha256': 'changed'}, {'all_text_read': False},
                        {'no_new_result_or_final_answer': False}):
            with self.assertRaises(ValueError): self.validate(reviews=[{**self.review, **changes}])

    def test_mechanical_failure_cannot_be_accepted_even_with_positive_review(self):
        bad = {**self.output, 'output': 'Short', 'output_sha256': digest('Short')}
        bad['mechanical_checks'] = mechanical_checks(self.request, bad, self.f.tokenizer)
        review = {**self.review, 'output_sha256': bad['output_sha256'],
            'mechanical_hard_rejection_reasons': bad['mechanical_checks']['hard_rejection_reasons']}
        with self.assertRaises(ValueError): self.validate([review], [bad])
        review['accepted'] = False
        self.validate([review], [bad])

    def test_recipe_cache_reads_only_fixed_general_diagnostic_namespace(self):
        with patch.object(recipe, 'load_recipe'), \
                patch.object(recipe, 'read', return_value=[self.request]) as read, \
                patch.object(recipe, 'checked_rows', return_value=[self.output]) as rows, \
                patch.object(recipe.baseline, 'sha', return_value='synthetic'):
            registry, _ = recipe.diagnostic_registry([self.request])
            self.assertEqual(registry[self.cid]['recipe_id'], recipe.RECIPE_ID)
            read.assert_called_once_with(recipe.DIAGNOSTIC / 'requests.json')
            rows.assert_called_once_with(recipe.DIAGNOSTIC / 'generated.jsonl')

    def test_changed_visible_step_or_prompt_cannot_reuse_general_diagnostic(self):
        changed = copy.deepcopy(self.request); changed['user'] += 'Different prompt'
        with patch.object(recipe, 'load_recipe'), \
                patch.object(recipe, 'read', return_value=[self.request]), \
                patch.object(recipe, 'checked_rows', return_value=[self.output]):
            with self.assertRaises(ValueError): recipe.diagnostic_registry([changed])

    def test_full_generated_resume_rejects_other_teacher_recipe(self):
        from full_explanations import validate_generated, PROTOCOL
        output = {**self.output, 'protocol': PROTOCOL, 'recipe_id': 'old_math_teacher',
            'request_sha256': digest(self.request['system'] + '\n' + self.request['user'])}
        with self.assertRaises(ValueError):
            validate_generated([output], [self.request], self.f.tokenizer, recipe_id=recipe.RECIPE_ID)
        output['recipe_id'] = recipe.RECIPE_ID
        validate_generated([output], [self.request], self.f.tokenizer, recipe_id=recipe.RECIPE_ID)

    def test_recipe_label_cannot_disguise_an_old_teacher_cache_path(self):
        from full_explanations import load_reused_outputs
        entry = {'recipe_id': recipe.RECIPE_ID,
            'requests_file': 'output/old_math_teacher/requests.json',
            'generated_file': 'output/old_math_teacher/generated.jsonl'}
        with patch('full_explanations.read') as read:
            with self.assertRaises(ValueError):
                load_reused_outputs({'previous_generation_registry': {self.cid: entry}}, [self.request])
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
