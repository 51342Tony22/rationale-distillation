import copy
import unittest
from runtime_full import read,OUT,ROOT,digest
from budgeted_materials import PROTOCOL
from transfer128_materials_v1 import build_bundles
from full_selector_materials_quality_v3 import FOLDER as SELECTOR
import test_transfer128_context_v1 as fixtures


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=read(SELECTOR/'materials_manifest.json')
        cls.g3=read(ROOT/manifest['materials']['M3']['path'])
        cls.roster=read(OUT/'transfer128/stratified_roster_v1.json')
        cls.sources=read(OUT/'transfer128/preparation_v1/root_preparation_v1.json')
        # Use the real frozen7179 G3 inventory, but all artificial trial evidence
        # and pool completions remain in test memory; never export experiment data.
        from full_selector import SOURCE,TEACHERS
        from runtime_full import checked_rows
        cls.all_sources=read(SOURCE);cls.teachers=checked_rows(TEACHERS)
        cls.rationales=read(OUT/'transfer128/variants_v1/qualified_Vsel_rationales_v2.json')['roots']
    def setUp(self):
        teachers={t['source_id']:t for t in self.teachers};self.pools=[];self.outcomes=[]
        qualified={r['selection_rationales'][0]['root_source_id'] for r in self.rationales}
        for sid in self.roster['source_ids']:
            t=teachers[sid]
            e0={'candidate_id':'e0','completion':t['output'],'completion_sha256':t['output_sha256'],'student_tokens':0}
            self.pools.append({'source_id':sid,'question_sha256':t['question_sha256'],
                'teacher_output_sha256':t['output_sha256'],'candidates':[e0]})
            self.outcomes.append({'source_id':sid,'selected_candidate_id':'e0','trial_executed':False,
                'reason':'no_automatically_eligible_joint_explanation' if sid in qualified else 'actual_template_rejected'})
    def build(self,outcomes=None,rationales=None):
        return build_bundles(self.g3,self.all_sources,self.teachers,self.roster,self.pools,
            self.outcomes if outcomes is None else outcomes,
            self.rationales if rationales is None else rationales,fixtures.Tokenizer())
    def test_full7179_coverage_G4_exact_outside128_G5_fixed_Vsel_only(self):
        bundles,choices,coverage=self.build();selected=set(self.roster['source_ids'])
        for original,g4,g5 in zip(self.g3['rows'],bundles['M7']['rows'],bundles['M6']['rows']):
            if original['source_id'] not in selected:self.assertEqual(original,g4)
            self.assertEqual(g5['epoch_materials'][0],g5['epoch_materials'][2])
        changed=[r for r in bundles['M6']['rows'] if r['epoch_materials'][1]['kind']=='verified_Vsel_variant']
        self.assertEqual(len(changed),124)
        self.assertEqual(coverage['common_roots'],7179)
        self.assertTrue(all(not c['audit_used_to_choose_material'] for c in choices))
    def test_missing_outcome_is_an_error_not_scientific_fallback(self):
        with self.assertRaises(ValueError):self.build(outcomes=self.outcomes[:-1])
    def test_false_template_rejection_and_audit_gradient_cannot_pass(self):
        outcomes=copy.deepcopy(self.outcomes);outcomes[0]['reason']='actual_template_rejected'
        with self.assertRaises(ValueError):self.build(outcomes=outcomes)
        rationales=copy.deepcopy(self.rationales);rationales[0]['selection_rationales'][0]['split']='audit'
        with self.assertRaises(ValueError):self.build(rationales=rationales)
    def test_negative_audit_cannot_change_the_frozen_Vsel_winner(self):
        pool=self.pools[0];outcome=self.outcomes[0]
        completion=pool['candidates'][0]['completion']+'\nSynthetic explained material in memory only.'
        candidate={'candidate_id':'test_explanation','completion':completion,'completion_sha256':digest(completion),
            'quality_provenance':{'eligible':True,'actual_review_accepted':None,'semantic_correctness_verified':False}}
        pool['candidates'].append(candidate)
        outcome.update(trial_executed=True,selected_candidate_id='test_explanation',Vaudit_accuracy_gain=-1.)
        first,_,_=self.build()
        outcome['Vaudit_accuracy_gain']=1.
        second,_,_=self.build()
        self.assertEqual(first,second)
        row=next(r for r in first['M7']['rows'] if r['source_id']==pool['source_id'])
        self.assertEqual(row['epoch_materials'][0]['selected_candidate_id'],'test_explanation')


if __name__=='__main__':unittest.main()
