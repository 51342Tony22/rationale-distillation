"""Full-train explanation boundary, shared-output and abstention checks.

All synthetic records stay in test memory. They never become actual reviews.
"""
import copy
import unittest

from full_explanations import (PROTOCOL, request_views, canonical_requests,
    validate_generated, assemble_views, reuse_entry)
from positions_nll import propose_nll
from positions_v2 import propose
from runtime_full import digest
from explanation_pool import mechanical_checks, REVIEW_CHECKS
import test_explanation_pool as fixtures


class FullExplanationsTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.ExplanationPoolTests(); self.f.setUp()
        self.sid = self.f.source['source_id']
        self.sources = {self.sid: self.f.source}
        self.teachers = {self.sid: self.f.teacher}
        self.cases = [{'source_id': self.sid,
            'teacher_output_sha256': self.f.teacher['output_sha256'],
            'candidates': [{'step': self.f.step}]}]
        self.diagnostics = {}
        for selector in ('joint', 'nll'):
            steps = [{'step_index': 0, 'mean_nll': 2., 'mask_excess': .5,
                      'status': 'diagnostic_complete'}]
            self.diagnostics[selector] = [{'source_id': self.sid, 'protocol': PROTOCOL,
                'selector': selector, 'teacher_output_sha256': self.f.teacher['output_sha256'],
                'steps': steps,
                'proposal': (propose if selector == 'joint' else propose_nll)(steps)}]

    def views(self):
        return request_views(self.sources, self.teachers, self.cases, self.diagnostics)

    def generated(self, requests, text=None):
        text = self.f.text if text is None else text
        rows = []
        for request in requests:
            row = {'candidate_id': request['candidate_id'], 'source_id': self.sid,
                'protocol': PROTOCOL, 'output': text, 'output_sha256': digest(text),
                'finish_reason': 'stop',
                'request_sha256': digest(request['system'] + '\n' + request['user'])}
            row['mechanical_checks'] = mechanical_checks(request, row, self.f.tokenizer)
            rows.append(row)
        return rows

    def reviews(self, generated):
        return [{'candidate_id': r['candidate_id'], 'output_sha256': r['output_sha256'],
            'accepted': True, 'kind': 'assistant_mathematical_review',
            'reason': 'Synthetic in-memory test review only.',
            **{k: True for k in REVIEW_CHECKS}} for r in generated]

    def assemble(self, views, generated, reviews):
        return assemble_views(self.sources, self.teachers, self.cases, self.diagnostics,
                               views, generated, reviews, self.f.tokenizer)

    def test_NLL_explanations_survive_missing_joint_mask_controls(self):
        joint = self.diagnostics['joint'][0]
        joint['steps'][0]['status'] = 'no_matched_random_controls'
        joint['proposal'] = propose(joint['steps'])
        views = self.views()
        self.assertEqual(views['joint'], [])
        self.assertEqual(len(views['nll']), 2)
        rows = self.generated(canonical_requests(views))
        pools = self.assemble(views, rows, self.reviews(rows))
        self.assertEqual(len(pools['joint'][0]['candidates']), 1)
        self.assertEqual(len(pools['nll'][0]['candidates']), 3)
        self.assertFalse(pools['nll'][0]['G4_full_trial_pool'])

    def test_shared_position_has_one_teacher_attempt_and_exact_shared_text(self):
        views = self.views(); requests = canonical_requests(views)
        self.assertEqual(sum(map(len, views.values())), 4)
        self.assertEqual(len(requests), 2)
        for request in requests:
            self.assertNotIn('UNSEEN_FUTURE_SUFFIX', request['user'])
            self.assertFalse(request['development_only'])
        rows = self.generated(requests)
        pools = self.assemble(views, rows, self.reviews(rows))
        self.assertEqual(pools['joint'][0]['candidates'], pools['nll'][0]['candidates'])
        bad = copy.deepcopy(views); bad['nll'][0]['user'] += 'Changed prompt'
        with self.assertRaises(ValueError): canonical_requests(bad)

    def test_incomplete_diagnostics_and_NLL_step_filtering_are_errors(self):
        self.diagnostics['nll'] = []
        with self.assertRaises(ValueError): self.views()
        self.setUp()
        self.diagnostics['nll'][0]['steps'] = []
        self.diagnostics['nll'][0]['proposal'] = propose_nll([])
        with self.assertRaises(ValueError): self.views()

    def test_prior_calibration_output_reused_despite_different_selector_rank(self):
        request = canonical_requests(self.views())[0]
        cid = request['candidate_id']
        previous_request = copy.deepcopy(request)
        previous_request.update(position_rank=1, development_only=True)
        row = self.generated([request])[0]
        row['record_sha256'] = 'synthetic_old_signature'
        reused = reuse_entry(request, {cid: previous_request}, {cid: row})
        self.assertEqual(reused['output'], self.f.text)
        self.assertNotIn('record_sha256', reused)
        previous_request['user'] += 'Different visible context'
        with self.assertRaises(ValueError):
            reuse_entry(request, {cid: previous_request}, {cid: row})
        with self.assertRaises(ValueError):
            reuse_entry(request, {cid: request}, {cid: {**row, 'output': 'Corrupt'}})

    def test_generated_resume_binds_order_prompt_output_and_mechanical_evidence(self):
        requests = canonical_requests(self.views()); rows = self.generated(requests)
        validate_generated(rows[:1], requests, self.f.tokenizer)
        with self.assertRaises(ValueError):
            validate_generated(rows[:1], requests, self.f.tokenizer, complete=True)
        for change in (lambda r: r.update(request_sha256='changed'),
                       lambda r: r.update(output='changed'),
                       lambda r: r.update(candidate_id='outside_plan'),
                       lambda r: r['mechanical_checks'].update(hard_rejection_reasons=[])):
            bad = self.generated(requests, 'Too short.')
            change(bad[0])
            with self.assertRaises(ValueError):
                validate_generated(bad, requests, self.f.tokenizer)

    def test_missing_semantic_review_is_error_and_rejection_is_durable_fallback(self):
        views = self.views(); requests = canonical_requests(views)
        rows = self.generated(requests)
        with self.assertRaises(ValueError): self.assemble(views, rows, [])
        reviews = self.reviews(rows)
        for review in reviews:
            review.update(accepted=False, reason='Synthetic semantic rejection.')
        pools = self.assemble(views, rows, reviews)
        for selector in ('joint', 'nll'):
            self.assertEqual(len(pools[selector][0]['candidates']), 1)
            self.assertEqual(len(pools[selector][0]['rejected']), 2)
        # Mechanically bad outputs need no invented semantic acceptance.
        bad = self.generated(requests, 'Too short.')
        pools = self.assemble(views, bad, [])
        self.assertEqual(len(pools['joint'][0]['rejected']), 2)
        with self.assertRaises(ValueError): self.assemble(views, bad, reviews)

    def test_second_positions_are_retained_in_proposals_but_not_generated_for_G3(self):
        # Add a different eligible step; both diagnostics still report it.
        step = copy.deepcopy(self.f.step); step['step_index'] = 1
        self.cases[0]['candidates'].append({'step': step})
        for selector in ('joint', 'nll'):
            row = self.diagnostics[selector][0]
            row['steps'].append({'step_index': 1, 'mean_nll': 1., 'mask_excess': .1,
                                 'status': 'diagnostic_complete'})
            row['proposal'] = (propose if selector == 'joint' else propose_nll)(row['steps'])
            self.assertEqual(row['proposal']['positions'], [0, 1])
        requests = canonical_requests(self.views())
        self.assertEqual(len(requests), 2)
        self.assertEqual({r['step']['step_index'] for r in requests}, {0})


if __name__ == '__main__':
    unittest.main()
