"""Frozen128 request identity checks; no model, semantic review or real output."""
import copy
import unittest
from runtime_full import digest
from budgeted_materials import PROTOCOL
from auxiliary_explainer_recipe import RECIPE_ID
from explanation_pool import make_request
from transfer128_prepare_v1 import request_plan


class TransferPreparationTests(unittest.TestCase):
    def setUp(self):
        self.ids=[f'gsm8k:train:{i}' for i in range(129)]
        self.roster={'protocol':PROTOCOL,'seed':42,'root_budget':128,'source_ids':self.ids[:128],
            'excluded_development_ids':[],'frozen_before_transfer_results':True,'public_test_used':False,
            'replacement_on_failure':False,'sampling_rule':'synthetic_test_only',
            'sampling_inputs_sha256':{'synthetic':'not_an_experiment'}}
        self.sources={};self.teachers={};self.cases=[];self.joint=[]
        text='First step.\nSecond step.\nThird step.'
        self.steps=[]
        for i,s in enumerate(text.splitlines()):
            a=text.index(s);self.steps.append({'step_index':i,'char_start':a,'char_end':a+len(s),'text':s})
        for sid in self.ids:
            question='Synthetic root '+sid
            s={'source_id':sid,'question':question,'question_sha256':digest(question),'gold':'7'}
            t={'source_id':sid,'question_sha256':s['question_sha256'],'output':text,'output_sha256':digest(text)}
            self.sources[sid]=s;self.teachers[sid]=t
            self.cases.append({'source_id':sid,'teacher_output_sha256':t['output_sha256'],
                'candidates':[{'step':copy.deepcopy(step)} for step in self.steps]})
            self.joint.append({'source_id':sid,'teacher_output_sha256':t['output_sha256'],
                'selector':'joint','proposal':{'positions':[0,1]}})
        self.joint[127]['proposal']['positions']=[]
        sid=self.ids[0];s=self.sources[sid];t=self.teachers[sid]
        # Joint rank1 was previously an NLL rank0 request, whose prompt is identical.
        self.old_requests=[make_request(s,t,self.steps[index],kind,0) for index,kind in
            [(0,'relation'),(1,'contrastive'),(2,'relation')]]
        self.outputs=[];self.decisions=[]
        for request in self.old_requests:
            cid=request['candidate_id'];text='Synthetic explanation for '+cid
            row={'candidate_id':cid,'recipe_id':RECIPE_ID,'output':text,'output_sha256':digest(text),
                'request_sha256':digest(request['system']+'\n'+request['user']),
                'record_sha256':digest('synthetic_record '+cid)}
            self.outputs.append(row);self.decisions.append({'candidate_id':cid,
                'output_sha256':row['output_sha256'],'request_sha256':row['request_sha256'],
                'eligible':request['step']['step_index']!=0,'all_text_read':False})

    def plan(self):
        return request_plan(self.sources,self.teachers,self.cases,self.joint,self.roster,self.ids,
            self.old_requests,self.outputs,self.decisions)

    def test_fixed128_two_joint_positions_preserve_empty_root_and_no_nll_only_position(self):
        requests,pending,reused,roots=self.plan()
        self.assertEqual([r['source_id'] for r in roots],self.ids[:128])
        self.assertEqual(len(requests),127*4);self.assertEqual(len(pending),127*4-2)
        self.assertEqual(len(reused),2);self.assertEqual(roots[-1]['candidate_ids'],[])
        self.assertTrue(all(r['step']['step_index'] in [0,1] for r in requests))
        self.assertTrue(all(r['source_id']!=self.ids[-1] for r in requests))
        self.assertTrue(all(r['probe_preparation_status']=='awaiting_actual_template_review' for r in roots))

    def test_exact_old_nll_output_reused_only_when_joint_position_prompt_matches(self):
        requests,_,registry,_=self.plan();cid=self.old_requests[1]['candidate_id']
        request=next(r for r in requests if r['candidate_id']==cid)
        self.assertEqual(request['position_rank'],1);self.assertEqual(self.old_requests[1]['position_rank'],0)
        row=next(r for r in registry if r['candidate_id']==cid)
        self.assertEqual(row['generated_record_sha256'],self.outputs[1]['record_sha256'])
        self.assertFalse(row['semantic_correctness_verified']);self.assertFalse(registry[0]['automatic_eligible'])

    def test_changed_prompt_output_or_quality_cannot_reuse(self):
        original=copy.deepcopy((self.old_requests,self.outputs,self.decisions))
        for kind in range(3):
            self.old_requests,self.outputs,self.decisions=copy.deepcopy(original)
            if kind==0:self.old_requests[0]['user']+=' Changed'
            if kind==1:self.outputs[0]['output']+=' changed'
            if kind==2:self.decisions[0]['request_sha256']='wrong'
            with self.assertRaises(ValueError):self.plan()

    def test_old_math_recipe_and_incomplete_historical_output_are_rejected(self):
        self.outputs[0]['recipe_id']='old_math7b'
        with self.assertRaises(ValueError):self.plan()
        self.outputs[0]['recipe_id']=RECIPE_ID;self.decisions.pop(0)
        with self.assertRaises(ValueError):self.plan()

    def test_roster_and_common_order_cannot_change_or_replace_failed_root(self):
        self.roster['replacement_on_failure']=True
        with self.assertRaises(ValueError):self.plan()
        self.roster['replacement_on_failure']=False;self.joint.reverse()
        with self.assertRaises(ValueError):self.plan()

    def test_teacher_or_joint_identity_change_rejected_before_requests(self):
        self.joint[0]['selector']='nll'
        with self.assertRaises(ValueError):self.plan()
        self.joint[0]['selector']='joint';self.teachers[self.ids[0]]['output']+=' altered'
        with self.assertRaises(ValueError):self.plan()


if __name__=='__main__':unittest.main()
