import unittest

from explanation_pool import (make_request, mechanical_checks, reviewed_candidate,
                              REVIEW_CHECKS, digest)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class ExplanationPoolTests(unittest.TestCase):
    def setUp(self):
        self.question = 'Each round two workers carry 2 and 3 buckets. They need 110 buckets. How many rounds?'
        self.source = {'source_id':'gsm8k:train:3146', 'question': self.question,
                       'question_sha256': digest(self.question)}
        self.original = 'They work together.\nTogether they carry 2 + 3 = 5 buckets.\nUNSEEN_FUTURE_SUFFIX: 110/5=22.'
        self.teacher = {'output': self.original, 'output_sha256': digest(self.original),
                        'question_sha256': digest(self.question)}
        line = 'Together they carry 2 + 3 = 5 buckets.'
        start = self.original.index(line)
        self.step = {'char_start':start, 'char_end':start+len(line), 'text':line, 'step_index':0}
        self.request = make_request(self.source, self.teacher, self.step, 'relation', 0)
        self.text = ('The workers act during the same round, so their contributions belong to one combined '
                     'delivery. Adding these contributions measures the amount delivered together; multiplying '
                     'them would combine two counts without representing the described action.')
        self.generated = {'output':self.text, 'output_sha256':digest(self.text), 'finish_reason':'stop'}
        self.review = {'candidate_id':self.request['candidate_id'], 'output_sha256':digest(self.text),
                       'accepted':True, 'kind':'assistant_mathematical_review',
                       'reason':'Synthetic fixture: concurrent contributions add.',
                       **{key:True for key in REVIEW_CHECKS}}
        self.tokenizer = WordTokenizer()

    def test_prompt_never_contains_later_solution_or_probe_labels(self):
        self.assertNotIn('UNSEEN_FUTURE_SUFFIX', self.request['user'])
        self.assertNotIn('110/5=22', self.request['user'])
        self.assertNotIn('gold', self.request)
        self.assertIn(self.step['text'], self.request['user'])
        with self.assertRaises(ValueError):
            make_request(self.source, self.teacher, {**self.step,'char_end':0}, 'relation', 0)

    def test_review_is_mandatory_and_bound_to_exact_generated_output(self):
        self.assertEqual(mechanical_checks(self.request,self.generated,self.tokenizer)['hard_rejection_reasons'], [])
        with self.assertRaises(ValueError):
            reviewed_candidate(self.request,self.generated,None,self.source,self.teacher,self.tokenizer)
        with self.assertRaises(ValueError):
            reviewed_candidate(self.request,self.generated,{**self.review,'output_sha256':'wrong'},
                               self.source,self.teacher,self.tokenizer)
        with self.assertRaises(ValueError):
            reviewed_candidate(self.request,self.generated,{**self.review,'no_new_result_or_final_answer':False},
                               self.source,self.teacher,self.tokenizer)

    def test_insertion_preserves_entire_original_and_does_not_truncate(self):
        candidate, reasons = reviewed_candidate(self.request,self.generated,self.review,
                                                 self.source,self.teacher,self.tokenizer)
        self.assertFalse(reasons)
        addition = '\nExplanation: ' + self.text + '\n'
        self.assertEqual(candidate['completion'].replace(addition,'',1), self.original)
        self.assertLess(candidate['completion'].index(addition),candidate['completion'].index('UNSEEN_FUTURE_SUFFIX'))

    def test_length_stop_and_hidden_numeric_result_are_rejected(self):
        for text, finish, reason in [(self.text,'length','not_a_complete_stop'),
                                    (self.text+' The answer is 22.','stop','answer_marker'),
                                    (self.text+' 32','stop','new_numeric_literal_requires_rejection'),
                                    ('Too short.','stop','outside_32_to_64_student_tokens')]:
            row = {'output':text,'output_sha256':digest(text),'finish_reason':finish}
            checks = mechanical_checks(self.request,row,self.tokenizer)
            self.assertIn(reason, checks['hard_rejection_reasons'])
            candidate, reasons = reviewed_candidate(self.request,row,None,self.source,self.teacher,self.tokenizer)
            self.assertIsNone(candidate)
            self.assertTrue(reasons)


if __name__ == '__main__':
    unittest.main()
