import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import transfer128_explanations_v1 as m
from runtime_full import digest,checked_rows
import test_explanation_quality_v3 as fixtures


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.QualityV3Tests();f.setUp();self.f=f;self.q=f.request;self.t=f.f.f.tokenizer
        self.output=f.row
    def test_crash_consumes_attempt_without_regeneration_then_cached_resume(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'check_time'):
            def fail(_):raise RuntimeError('simulated interrupted generation')
            with self.assertRaises(RuntimeError):m.generate_pending([self.q],temp,self.t,fail)
            def forbidden(_):raise AssertionError('duplicate generation')
            rows=m.generate_pending([self.q],temp,self.t,forbidden)
            self.assertEqual(rows[0]['finish_reason'],'interrupted_attempt_excluded_no_retry')
            self.assertIn('not_a_complete_stop',rows[0]['mechanical_checks']['hard_rejection_reasons'])
            self.assertEqual(m.generate_pending([self.q],temp,self.t,forbidden),rows)
            self.assertEqual(len(checked_rows(Path(temp)/'generated.jsonl')),1)
    def test_complete_attempt_resumes_without_calling_model_again(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(m,'check_time'):
            out=m.generate_pending([self.q],temp,self.t,lambda q:self.output)
            self.assertFalse(out[0]['semantic_correctness_verified']);self.assertFalse(out[0]['all_text_read'])
            m.validate_prefix(out,[self.q],self.t)
            changed=copy.deepcopy(self.q);changed['user']+=' changed'
            with self.assertRaises(ValueError):m.validate_prefix(out,[changed],self.t)
    def test_new_unread_automatic_output_is_not_a_semantic_accept(self):
        row=m.output_row(self.q,self.output,self.t);row['record_sha256']='in-memory-only'
        decisions,_=m.combine([self.q],[row],[],[],[],self.f.f.sources,self.f.f.teachers,self.t)
        self.assertTrue(decisions[0]['eligible']);self.assertIsNone(decisions[0]['actual_review_accepted'])
        self.assertFalse(decisions[0]['all_text_read'])
    def reused(self):
        import json
        row=self.output;d=self.f.decision({'accepted':False,'reason':'synthetic negative'})
        r={'candidate_id':self.q['candidate_id'],'generated_record_sha256':row['record_sha256'],
            'quality_decision_sha256':digest(json.dumps(d,sort_keys=True)),
            'output_sha256':row['output_sha256'],'automatic_eligible':False}
        return d,r
    def test_historical_negative_is_preserved_exactly(self):
        d,r=self.reused()
        out,_=m.combine([self.q],[],[self.output],[d],[r],self.f.f.sources,self.f.f.teachers,self.t)
        self.assertEqual(out,[d]);self.assertFalse(out[0]['eligible'])
        d['eligible']=True
        with self.assertRaises(ValueError):m.combine([self.q],[],[self.output],[d],[r],self.f.f.sources,self.f.f.teachers,self.t)
    def test_joint_pool_rejects_nll_only_position_and_keeps_empty_root(self):
        d=self.f.decision();q=self.q
        roots=[{'source_id':q['source_id'],'question_sha256':q['question_sha256'],
            'candidate_ids':[q['candidate_id']],'joint_positions':[q['step']['step_index']]}]
        args=([q],{q['candidate_id']:self.output},[d],self.f.f.teachers)
        pools=m.joint_pools(roots,*args);self.assertEqual(len(pools[0]['candidates']),2)
        roots[0]['joint_positions']=[]
        with self.assertRaises(ValueError):m.joint_pools(roots,*args)
        roots[0]['candidate_ids']=[]
        self.assertEqual(len(m.joint_pools(roots,*args)[0]['candidates']),1)
    def test_missing_and_duplicate_output_coverage_is_fatal(self):
        with self.assertRaises(ValueError):m.combine([self.q],[],[],[],[],{}, {},self.t)
        row=m.output_row(self.q,self.output,self.t)
        with self.assertRaises(ValueError):m.validate_prefix([row,row],[self.q],self.t)


if __name__=='__main__':unittest.main()
