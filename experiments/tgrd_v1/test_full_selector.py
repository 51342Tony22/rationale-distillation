"""Full-cohort quality boundaries and real-tokenizer selector parity."""
import copy
import unittest

from runtime_full import DATA, OUT, BASE, ROOT, read, checked_rows, digest
from full_selector import (PROTOCOL, validate_quality, prepare_training_case,
                           validate_resume)
from positions_nll import propose_nll
from calibration_cohort import cohort_paths


class FullSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = []
        cls.teachers = []
        cls.reviews = []
        for i in range(7473):
            sid = f'gsm8k:train:{i}'; question = f'Synthetic test question {i}'
            output = f'Synthetic test answer {i}'
            source = {'source_id': sid, 'question': question,
                'question_sha256': digest(question), 'gold': str(i)}
            teacher = {'source_id': sid, 'question_sha256': digest(question),
                'gold': str(i), 'output': output, 'output_sha256': digest(output),
                'correct': True, 'finish_reason': 'stop'}
            review = {'source_id': sid, 'question_sha256': digest(question),
                'output_sha256': digest(output), 'accepted': i < 2, 'all_text_read': True,
                'review_kind': 'assistant_full_text_mathematical_review',
                'reason': 'Synthetic fixture only, not an actual review.'}
            cls.sources.append(source); cls.teachers.append(teacher); cls.reviews.append(review)

    def quality(self):
        return {'schema': 'tgrd_full_teacher_quality_v1', 'protocol': PROTOCOL,
            'status': 'frozen', 'candidate_root_count': 7473, 'public_test_used': False,
            'review_policy': {'path': 'synthetic_policy', 'sha256': 'synthetic'},
            'inputs': {'synthetic_policy': 'synthetic'},
            'reviews': copy.deepcopy(self.reviews),
            'accepted_root_ids': ['gsm8k:train:0', 'gsm8k:train:1']}

    def test_common_accepted_order_requires_complete_actual_decision_coverage(self):
        quality = self.quality()
        self.assertEqual(validate_quality(self.sources, self.teachers, quality),
                         quality['accepted_root_ids'])
        quality['reviews'].pop()
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers, quality)
        quality = self.quality(); quality['accepted_root_ids'].reverse()
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers, quality)
        quality = self.quality(); quality['reviews'][5] = copy.deepcopy(quality['reviews'][6])
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers, quality)

    def test_numeric_match_does_not_substitute_for_review_or_finish_or_original_gold(self):
        quality = self.quality(); quality['reviews'][0]['all_text_read'] = False
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers, quality)
        quality = self.quality(); quality['public_test_used'] = True
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers, quality)
        for change in (lambda t: t.update(correct=False),
                       lambda t: t.update(finish_reason='length'),
                       lambda t: t.update(gold='silently_changed'),
                       lambda t: t.update(output='changed without a matching review')):
            teachers = list(self.teachers); teachers[0] = dict(teachers[0]); change(teachers[0])
            with self.assertRaises(ValueError):
                validate_quality(self.sources, teachers, self.quality())

    def test_public_or_missing_teacher_roots_cannot_enter_training(self):
        sources = list(self.sources); sources[0] = dict(sources[0]); sources[0]['source_id'] = 'gsm8k:test:0'
        with self.assertRaises(ValueError): validate_quality(sources, self.teachers, self.quality())
        with self.assertRaises(ValueError): validate_quality(self.sources, self.teachers[:-1], self.quality())

    def test_actual_cal34_tokenized_cases_are_identical_without_variant_dependency(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
            local_files_only=True, trust_remote_code=False)
        sources = {s['source_id']: s for s in read(DATA / 'cal40.json')}
        teachers = {t['source_id']: t for t in checked_rows(OUT / 'materials/teacher_targets_cal40.jsonl')}
        cases = read(cohort_paths('cal40_reviewed34_v2').cases)
        self.assertEqual(len(cases), 34)
        for case in cases:
            sid = case['source_id']
            self.assertEqual(prepare_training_case(sources[sid], teachers[sid], tokenizer), case)

    def test_resume_preserves_every_NLL_candidate_and_exact_proposals(self):
        cases = [{'source_id': 'gsm8k:train:0', 'teacher_output_sha256': 'test',
                  'candidates': [{'step': {'step_index': i}} for i in (0, 1, 2)]}]
        scores = [{'step_index': i, 'mean_nll': float(i + 1)} for i in (0, 1, 2)]
        row = {'source_id': 'gsm8k:train:0', 'teacher_output_sha256': 'test',
            'protocol': PROTOCOL, 'selector': 'nll', 'steps': scores,
            'proposal': propose_nll(scores)}
        validate_resume([row], cases, 'nll')
        for change in (lambda r: r['steps'].pop(),
                       lambda r: r.update(selector='joint'),
                       lambda r: r['proposal'].update(positions=[0, 1]),
                       lambda r: r.update(source_id='gsm8k:train:1')):
            bad = copy.deepcopy(row); change(bad)
            with self.assertRaises(ValueError): validate_resume([bad], cases, 'nll')


if __name__ == '__main__':
    unittest.main()
