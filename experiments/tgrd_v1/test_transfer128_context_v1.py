import copy
import unittest
from runtime_full import OUT,read,digest
from transfer128_context_v1 import check_context


class Tokenizer:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):return list(range(len(text.split())))


class ContextTests(unittest.TestCase):
    def setUp(self):
        bundle=read(OUT/'transfer128/variants_v1/qualified_bundles_v2.json')['roots'][0]
        self.sid=bundle['source_id'];self.bundles=[copy.deepcopy(bundle)]
        rationale=read(OUT/'transfer128/variants_v1/qualified_Vsel_rationales_v2.json')['roots'][0]
        self.rationales=[copy.deepcopy(rationale)]
        self.source=next(r for r in read(OUT/'transfer128/preparation_v1/root_preparation_v1.json') if r['source_id']==self.sid)
        self.teacher={'source_id':self.sid,'question_sha256':self.source['question_sha256'],
            'output':'Synthetic CoT context test.','output_sha256':digest('Synthetic CoT context test.')}
    def run_check(self,tokenizer=None):
        return check_context([self.sid],[self.source],[self.teacher],self.bundles,self.rationales,tokenizer or Tokenizer())
    def test_actual_frozen_probe_and_rationale_identities_replay(self):
        result=self.run_check()
        self.assertEqual(result['probes_checked'],12);self.assertEqual(result['Vsel_rationales_checked'],6)
        self.assertFalse(result['truncation_used']);self.assertFalse(result['scientific_trial_enabled'])
    def test_audit_rationale_and_changed_labels_or_completions_cannot_pass(self):
        original=copy.deepcopy(self.rationales)
        for key,value in [('split','audit'),('gold','wrong'),('completion','changed')]:
            self.rationales=copy.deepcopy(original)
            self.rationales[0]['selection_rationales'][0][key]=value
            with self.assertRaises(ValueError):self.run_check()
        self.rationales=original;self.bundles[0]['variants']['select'][0]['gold']='tampered'
        with self.assertRaises(ValueError):self.run_check()
    def test_full1024_generation_budget_blocks_context_overflow(self):
        class Long(Tokenizer):
            def encode(self,text,add_special_tokens=False):return [0]*3200
        with self.assertRaises(ValueError):self.run_check(Long())
    def test_missing_bundle_and_rationale_cannot_qualify(self):
        self.rationales=[]
        with self.assertRaises(ValueError):self.run_check()


if __name__=='__main__':unittest.main()
