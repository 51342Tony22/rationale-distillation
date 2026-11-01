"""Meaningful v3 policy regressions; synthetic reviews stay in test memory."""
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import explanation_quality_v3 as q
from runtime_full import digest, read, freeze, ROOT
from full_selector_materials_quality_v3 import build_pools
import test_full_explanations as fixtures

class QualityV3Tests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.FullExplanationsTests();self.f.setUp();self.f.f.tokenizer.eos_token_id=0
        self.views=self.f.views();self.request=self.views['joint'][0]
        self.row=self.f.generated([self.request])[0]
        self.row['record_sha256']='synthetic_in_memory_only'
    def decision(self,review=None):
        return q.automatic_decision(self.request,self.row,self.f.f.source,self.f.f.teacher,self.f.f.tokenizer,review)
    def test_unread_pass_is_eligible_without_a_semantic_review(self):
        d=self.decision();self.assertTrue(d['eligible'])
        self.assertFalse(d['all_text_read']);self.assertFalse(d['semantic_correctness_verified'])
        self.assertIsNone(d['actual_review_accepted']);self.assertNotIn('review',d)
    def test_actual_negative_cannot_become_eligible_or_be_silently_dropped(self):
        d=self.decision({'accepted':False,'reason':'Synthetic invalid operation'})
        self.assertFalse(d['eligible']);self.assertIn('actual_semantic_rejection',d['reasons'])
        self.assertTrue(d['all_text_read']);self.assertFalse(d['semantic_correctness_verified'])
    def test_existing_accept_still_must_pass_additional_automatic_checks(self):
        self.row['output']+=' Another sentence.';self.row['output_sha256']=digest(self.row['output'])
        d=self.decision({'accepted':True,'reason':'Synthetic earlier decision'})
        self.assertFalse(d['eligible']);self.assertIn('not_one_or_two_sentences_v3',d['reasons'])
    def test_decimal_and_abbreviation_do_not_create_false_sentence_splits(self):
        self.assertEqual(q.sentence_count('Mr. Green earns 0.25 per unit. That is the rate.'),2)
        self.assertEqual(q.sentence_count('One. Two. Three.'),3)
        self.assertEqual(q.sentence_count(''),0)
    def test_context_overflow_rejected_without_truncation(self):
        with patch('learning.tokenize',side_effect=ValueError('would truncate')):
            d=self.decision()
        self.assertFalse(d['eligible']);self.assertIn('inserted_context_exceeds4096_including_EOS_v3',d['reasons'])
    def test_output_hash_tamper_is_fatal(self):
        self.row['output']+=' altered'
        with self.assertRaises(ValueError):self.decision()
    def test_shared_text_and_eligibility_and_NLL_only_isolation(self):
        from full_explanations import canonical_requests
        outputs=self.f.generated(canonical_requests(self.views))
        for r in outputs:r['record_sha256']='synthetic_in_memory_only'
        decisions=[q.automatic_decision(req,r,self.f.f.source,self.f.f.teacher,self.f.f.tokenizer)
                   for req,r in zip(canonical_requests(self.views),outputs)]
        args=(self.f.cases,self.f.sources,self.f.teachers,self.f.diagnostics,self.views,outputs,decisions)
        pools=build_pools(*args);sid=self.f.sid
        self.assertEqual(pools['joint'][sid]['candidates'],pools['nll'][sid]['candidates'])
        views=copy.deepcopy(self.views);views['joint']=[]
        pools=build_pools(*args[:4],views,outputs,decisions)
        self.assertEqual(len(pools['joint'][sid]['candidates']),1)
        self.assertEqual(len(pools['nll'][sid]['candidates']),3)
        self.assertFalse(pools['joint'][sid]['G4_full_trial_pool'])
    def test_missing_fixed_sample_cannot_produce_ready_report(self):
        with patch.object(q,'read',side_effect=[[],{'sample':[{'candidate_id':'unread'}]}]):
            with self.assertRaises(ValueError):q.sample_report([],[],[])
    def test_frozen_input_tamper_is_fatal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/'proof';p.write_text('original')
            value={'inputs':{str(p.relative_to(ROOT)):q.baseline.sha(p)}}
            q.verify_inputs(value);p.write_text('changed')
            with self.assertRaises(ValueError):q.verify_inputs(value)

if __name__=='__main__':unittest.main()
