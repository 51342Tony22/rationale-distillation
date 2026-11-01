"""Five-group assembly using real reviewed roots and synthetic candidate scores.

Synthetic candidate/review objects exist only inside these unit tests. They are
never written to a shared-pool artifact or used as scientific observations.
"""
import copy
import json
import unittest
from runtime_full import DATA,OUT,read,checked_rows,digest
from calibration_cohort import cohort_paths
from material_builder import root_materials
from explanation_pool import REVIEW_CHECKS,insert_explanation
from trial_selection import choose,validate_probes
from variants import build
from training_plan import CONDITIONS,validate_materials


class MaterialBuilderTests(unittest.TestCase):
    def setUp(self):
        paths=cohort_paths('cal40_reviewed34_v2')
        self.case=next(c for c in read(paths.cases) if len(c['candidates'])>=2)
        sid=self.case['source_id']
        self.source=next(r for r in read(DATA/'cal40.json') if r['source_id']==sid)
        self.teacher=next(r for r in checked_rows(OUT/'materials/teacher_targets_cal40.jsonl') if r['source_id']==sid)
        self.review=next(r for r in read(paths.reviews) if r['source_id']==sid)
        self.variant=next(r for r in read(paths.variants) if r['source_id']==sid)
        self.steps={c['step']['step_index']:c['step'] for c in self.case['candidates']}
        self.pool={'source_id':sid,'question_sha256':self.source['question_sha256'],
            'teacher_output_sha256':self.teacher['output_sha256'],'candidates':[{
                'candidate_id':'e0','completion':self.teacher['output'],
                'completion_sha256':self.teacher['output_sha256'],'student_tokens':0}]}

        from positions_nll import propose_nll
        self.nll_pool=copy.deepcopy(self.pool)
        self.nll_pool.update(selector='nll_only',position_proposal=propose_nll([
            {'step_index':index,'mean_nll':float(len(self.steps)-rank)}
            for rank,index in enumerate(self.steps)]))

    def add_candidate(self,kind,rank=0,*,selector='joint',step_slot=None):
        slot=rank if step_slot is None else step_slot
        step=list(self.steps.values())[slot];sid=self.source['source_id'];cid=f'{sid}:step{step["step_index"]}:{kind}'
        text='Synthetic unit-test explanation for '+kind+'.'
        completion=insert_explanation(self.teacher['output'],step,text)
        c={'candidate_id':cid,'source_id':sid,'teacher_output_sha256':self.teacher['output_sha256'],
            'explanation_type':kind,'position_rank':rank,'step_index':step['step_index'],
            'text':text,'text_sha256':digest(text),'student_tokens':40,
            'completion':completion,'completion_sha256':digest(completion),
            'review':{'candidate_id':cid,'output_sha256':digest(text),'accepted':True,
                'kind':'assistant_mathematical_review','reason':'Synthetic test review, not an actual candidate.',
                **{key:True for key in REVIEW_CHECKS}}}
        (self.pool if selector=='joint' else self.nll_pool)['candidates'].append(c);return cid

    def trial(self):
        probes=validate_probes(self.variant,self.source,build)
        inputs={'source_id':self.source['source_id'],'question_sha256':self.source['question_sha256'],
            'reviewed_pool_sha256':digest(json.dumps(self.pool,sort_keys=True)),
            'probe_bundle_sha256':digest(json.dumps(probes,sort_keys=True)),
            'seed':701,'updates':3,'thresholds':[0,.01,.03,.10]}
        scores={c['candidate_id']:{'accuracy':1. if c.get('explanation_type')=='relation' else .5,
            'answer_nll':.5 if c.get('explanation_type')=='relation' else 1.} for c in self.pool['candidates']}
        decisions={'source_id':self.source['source_id'],'Vsel_scores':scores,'untrained_Vsel':{'accuracy':0.,'answer_nll':2.},
            'decisions':[choose(scores,self.pool['candidates'],t) for t in (0,.01,.03,.10)],
            'audit_can_change_selection':False}
        return inputs,decisions

    def run_root(self,**kwargs):
        kwargs.setdefault('nll_pool',self.nll_pool)
        return root_materials(self.source,self.teacher,self.review,self.pool,self.steps,**kwargs)

    def check_bundles(self,rows):
        self.assertEqual(set(rows),{'M1','NLL','M3','M7','M6'})
        for group in CONDITIONS:
            validate_materials({'schema':'tgrd_training_materials_v1','condition':group,
                'candidate_root_count':7473,'rows':[rows[group]]},[self.source['source_id']])

    def test_independent_NLL_can_choose_position_absent_from_joint_pool(self):
        nll=self.add_candidate('relation',selector='nll')
        joint=self.add_candidate('relation',rank=0,step_slot=1)
        inputs,decisions=self.trial()
        rows,choices=self.run_root(variant_bundle=self.variant,trial_inputs=inputs,trial_decisions=decisions)
        self.check_bundles(rows);selected=choices['selected_candidate_ids']
        self.assertEqual(selected,{'M1':'e0','NLL':nll,'M3':joint,'M7':joint,'M6':'e0'})
        self.assertNotIn(nll,[c['candidate_id'] for c in self.pool['candidates']])
        self.assertFalse(choices['audit_predictions_read'])
        self.assertEqual(choices['variant_labels_in_formal_gradients_only_for'],['M6'])

    def test_empty_pools_fallback_but_missing_NLL_computation_is_error(self):
        rows,choices=self.run_root();self.check_bundles(rows)
        self.assertTrue(all(cid=='e0' for cid in choices['selected_candidate_ids'].values()))
        with self.assertRaises(ValueError):self.run_root(nll_pool=None)
        bad=copy.deepcopy(self.nll_pool);bad['position_proposal']['scores'].pop()
        with self.assertRaises(ValueError):self.run_root(nll_pool=bad)
        bad=copy.deepcopy(self.nll_pool);bad['position_proposal']['attention_used']=True
        with self.assertRaises(ValueError):self.run_root(nll_pool=bad)

    def test_missing_trial_result_is_not_scientific_abstention(self):
        self.add_candidate('relation')
        with self.assertRaises(ValueError):self.run_root(variant_bundle=self.variant)
        _,choices=self.run_root()
        self.assertEqual(choices['fallback_reasons']['M7'],'no_qualified_complete_6plus6_probes')

    def test_fixed_choices_never_substitute_second_position(self):
        self.add_candidate('relation',rank=1)
        self.add_candidate('relation',rank=1,selector='nll')
        rows,choices=self.run_root();self.check_bundles(rows)
        for group in ('NLL','M3'):
            self.assertEqual(choices['selected_candidate_ids'][group],'e0')
            self.assertEqual(choices['fallback_reasons'][group],'top_proposed_position_has_no_qualified_explanation')

    def test_overlap_must_have_identical_explanation_and_prefer_same_type(self):
        relation=self.add_candidate('relation');self.add_candidate('contrastive')
        self.add_candidate('relation',selector='nll');self.add_candidate('contrastive',selector='nll')
        _,choices=self.run_root()
        self.assertEqual(choices['selected_candidate_ids']['NLL'],relation)
        self.assertEqual(choices['selected_candidate_ids']['M3'],relation)
        bad=copy.deepcopy(self.nll_pool)
        c=bad['candidates'][1];c['text']+=' changed';c['text_sha256']=digest(c['text'])
        c['completion']=insert_explanation(self.teacher['output'],self.steps[c['step_index']],c['text'])
        c['completion_sha256']=digest(c['completion']);c['review']['output_sha256']=digest(c['text'])
        with self.assertRaises(ValueError):self.run_root(nll_pool=bad)

    def test_tampered_review_and_before_audit_choices_fail(self):
        self.add_candidate('relation');inputs,decisions=self.trial()
        bad=copy.deepcopy(decisions);bad['audit_scores']={'e0':{'accuracy':1.}}
        with self.assertRaises(ValueError):self.run_root(variant_bundle=self.variant,trial_inputs=inputs,trial_decisions=bad)
        bad=copy.deepcopy(decisions);bad['decisions'][0]['selected_id']='e0'
        with self.assertRaises(ValueError):self.run_root(variant_bundle=self.variant,trial_inputs=inputs,trial_decisions=bad)
        self.pool['candidates'][1]['review']['mathematics_valid']=False
        with self.assertRaises(ValueError):self.run_root()

    def test_M6_uses_first_two_verified_Vsel_instances_and_uniform_fallback(self):
        rows,choices=self.run_root(variant_bundle=self.variant);self.check_bundles(rows)
        self.assertEqual(choices['M6_variant_ids'],[p['id'] for p in self.variant['variants']['select'][:2]])
        for epoch in (0,2):self.assertEqual(rows['M6']['epoch_materials'][epoch],rows['M1']['epoch_materials'][epoch])
        for epoch,probe in zip((1,3),self.variant['variants']['select'][:2]):
            material=rows['M6']['epoch_materials'][epoch]
            self.assertEqual(material['variant_id'],probe['id']);self.assertEqual(material['split'],'select')
            self.assertEqual(material['question_sha256'],probe['question_sha256'])
        rows,choices=self.run_root();self.check_bundles(rows)
        self.assertEqual(rows['M6']['epoch_materials'],rows['M1']['epoch_materials'])
        self.assertEqual(choices['fallback_reasons']['M6'],'no_qualified_complete_6plus6_probes')

    def test_rejected_teacher_never_survives_as_group_specific_exclusion(self):
        self.review['accepted']=False
        with self.assertRaises(ValueError):self.run_root()


if __name__=='__main__':unittest.main()
