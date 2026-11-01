"""Independent NLL ablation: coverage, prefix isolation and prompt reuse."""
import copy
import unittest
from positions_nll import propose_nll,score_case
from explanation_pool import reusable_output


class NLLPositionTests(unittest.TestCase):
    def test_rank_uses_all_steps_even_if_joint_controls_are_missing(self):
        scores=[{'step_index':3,'mean_nll':1.,'status':'diagnostic_complete'},
                {'step_index':8,'mean_nll':9.,'status':'insufficient_matched_controls'},
                {'step_index':2,'mean_nll':1.,'status':'no_prior_evidence'}]
        result=propose_nll(scores)
        self.assertEqual(result['positions'],[8,2])
        self.assertFalse(result['attention_used']);self.assertFalse(result['masking_used'])
        self.assertFalse(result['mask_control_filter_used'])
        self.assertEqual(propose_nll(list(reversed(scores))),result)
        self.assertEqual(propose_nll([])['positions'],[])
        for bad in (float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):propose_nll([{'step_index':0,'mean_nll':bad}])
        with self.assertRaises(ValueError):propose_nll([scores[0],scores[0]])

    def test_only_unperturbed_score_calls_and_no_later_solution_tokens(self):
        calls=[]
        class Backend:
            def score(self,ids,targets):
                calls.append((ids,targets));return [float(p) for p in targets]
            def observe(self,*args):raise AssertionError('NLL ablation must never read attention')
        case={'source_id':'synthetic:unit-test','teacher_output_sha256':'fixture',
            'input_ids':[10,11,12,13,14,15,999],
            'candidates':[{'step':{'step_index':0,'targets':[2,3]}},
                          {'step':{'step_index':1,'targets':[4,5]}}]}
        result=score_case(case,Backend())
        self.assertEqual(calls,[([10,11,12,13],[2,3]),([10,11,12,13,14,15],[4,5])])
        self.assertEqual(result['proposal']['positions'],[1,0])
        self.assertEqual(result['steps'][0]['target_ids'],[12,13])

    def test_overlap_reuses_exact_prompt_regardless_of_position_rank(self):
        request={'candidate_id':'fixture:step0:relation','source_id':'fixture',
            'question_sha256':'q','teacher_output_sha256':'t','step':{'step_index':0},
            'system':'same system','user':'same prompt','position_rank':0}
        previous={**request,'position_rank':1}
        generated={'candidate_id':request['candidate_id'],'output':'same explanation',
                   'record_sha256':'previous-container-signature'}
        reused=reusable_output(request,{request['candidate_id']:previous},{request['candidate_id']:generated})
        self.assertEqual(reused['output'],'same explanation');self.assertNotIn('record_sha256',reused)
        changed=copy.deepcopy(request);changed['user']+=' changed'
        with self.assertRaises(ValueError):reusable_output(changed,{request['candidate_id']:previous},{request['candidate_id']:generated})
        self.assertIsNone(reusable_output(request,{},{}))


if __name__=='__main__':unittest.main()
