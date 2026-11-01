"""Actual-review ledger guards tested with synthetic in-memory judgments."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_full import append, save, digest
from auxiliary_explainer_recipe import RECIPE_ID
from explanation_pool import REVIEW_CHECKS
import review_full_explanations as audit


class StreamingExplanationReviewTests(unittest.TestCase):
    def fixture(self):
        request = {'candidate_id': 'synthetic:relation', 'system': 'system', 'user': 'visible prompt',
            'question_sha256': 'question', 'teacher_output_sha256': 'teacher', 'explanation_type': 'relation'}
        output = {'candidate_id': request['candidate_id'], 'output_sha256': digest('synthetic text'),
            'mechanical_checks': {'hard_rejection_reasons': []}}
        review = {'candidate_id': request['candidate_id'], 'kind': 'assistant_mathematical_review',
            'all_text_read': True, 'accepted': True, 'reason': 'Synthetic explicit judgment only.',
            'output_sha256': output['output_sha256'], 'request_sha256': digest('system\nvisible prompt'),
            'question_sha256': 'question', 'teacher_output_sha256': 'teacher', 'recipe_id': RECIPE_ID,
            'mechanical_hard_rejection_reasons': [], **{k: True for k in REVIEW_CHECKS}}
        return request, output, review

    def test_active_file_partial_tail_is_ignored_but_corrupt_complete_row_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'active.jsonl'; append(path, {'output': 'complete'})
            with path.open('ab') as stream: stream.write(b'{"incomplete":')
            self.assertEqual(len(audit.complete_prefix(path)), 1)
            with path.open('ab') as stream: stream.write(b'0}\n')
            with self.assertRaises((ValueError, KeyError)): audit.complete_prefix(path)

    def test_missing_wrong_output_or_invalid_contrastive_cannot_become_acceptance(self):
        req, out, review = self.fixture()
        result = audit.coverage([req], [out], [])
        self.assertFalse(result['ready_for_final_export'])
        self.assertEqual(result['mechanically_usable_pending_actual_review'], 1)
        for change in (lambda r: r.update(output_sha256='other'),
                       lambda r: r.update(all_text_read=False),
                       lambda r: r.update(no_new_result_or_final_answer=False)):
            bad = copy.deepcopy(review); change(bad)
            with self.assertRaises(ValueError): audit.validate_decisions([req], [out], [bad])
        req['explanation_type'] = 'contrastive'
        with self.assertRaises(ValueError): audit.validate_decisions([req], [out], [review])
        review.update(contrastive_alternative_valid_and_plausible=True)
        audit.validate_decisions([req], [out], [review])

    def test_mechanical_rejection_needs_no_invented_review_but_generation_must_finish(self):
        req, out, review = self.fixture(); out['mechanical_checks']['hard_rejection_reasons'] = ['too_short']
        self.assertTrue(audit.coverage([req], [out], [])['ready_for_final_export'])
        self.assertFalse(audit.coverage([req, {**req, 'candidate_id': 'future'}], [out], [])['ready_for_final_export'])
        with self.assertRaises(ValueError): audit.validate_decisions([req], [out], [review])

    def test_interrupted_actual_ledger_replays_frozen_judgments_preserving_fragment(self):
        req, out, review = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); ledger = folder / 'reviews.jsonl'; batches = folder / 'batches'
            save(batches / 'actual.json', {'reviews': [review]})
            ledger.write_bytes(b'{"partial":')
            with patch.object(audit, 'LEDGER', ledger), patch.object(audit, 'BATCHES', batches), \
                    patch.object(audit, 'FULL', folder):
                first = audit.reconciled_reviews([req], [out])
                self.assertEqual(first, audit.reconciled_reviews([req], [out]))
                self.assertEqual(len(first), 1)
                self.assertEqual(next((folder / 'interrupted_review_fragments_v1').glob('*.bin')).read_bytes(), b'{"partial":')
                changed = {**review, 'accepted': False, 'reason': 'changed'}
                save(batches / 'actual.json', {'reviews': [changed]})
                with self.assertRaises(ValueError): audit.reconciled_reviews([req], [out])

    def test_invalid_decision_never_creates_a_frozen_review_batch(self):
        req, out, review = self.fixture(); review['output_sha256'] = 'wrong-output'
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(audit, 'LEDGER', folder/'ledger.jsonl'), \
                    patch.object(audit, 'BATCHES', folder/'batches'), patch.object(audit, 'FULL', folder), \
                    patch.object(audit, 'context', return_value=({}, [req], [out])):
                with self.assertRaises(ValueError): audit.commit_batch('invalid', [review])
                self.assertFalse((folder/'batches/invalid.json').exists())

    def test_all_mechanically_rejected_outputs_can_finish_without_invented_reviews(self):
        req, out, _ = self.fixture(); out['mechanical_checks']['hard_rejection_reasons'] = ['too_short']
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); save(folder/'requests_manifest.json', {'synthetic': True})
            append(folder/'generated.jsonl', out)
            with patch.object(audit, 'ROOT', folder), patch.object(audit, 'FULL', folder), \
                    patch.object(audit, 'LEDGER', folder/'ledger.jsonl'), \
                    patch.object(audit, 'BATCHES', folder/'batches'), \
                    patch.object(audit, 'context', return_value=({}, [req], [out])):
                self.assertTrue(audit.progress()['ready_for_final_export'])
                self.assertEqual((folder/'ledger.jsonl').read_bytes(), b'')
                self.assertTrue((folder/'reviews_ready_v1.json').exists())


if __name__ == '__main__': unittest.main()
