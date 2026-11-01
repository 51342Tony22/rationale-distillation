"""Actual training-side authored templates and hidden-parameter leakage guards."""
import copy
import unittest
from runtime_full import OUT,read
from transfer128_variants_v1 import validate_entry


class FiniteVariantReviewTests(unittest.TestCase):
    def setUp(self):
        self.entries=read(OUT/'transfer128/variants_v1/authored_batches/batch01_actual12.json')
        self.sources={r['source_id']:r for r in read(OUT/'transfer128/preparation_v1/root_preparation_v1.json')}

    def test_missing_exact_question_binding_or_actual_read_flag_cannot_accept(self):
        original=self.entries[0]
        for change in ({'question_sha256':'wrong'},{'original_question_text_read':False},
                       {'accepted':'automatic'},{'all_six_templates_text_read':False}):
            with self.assertRaises(ValueError):validate_entry({**original,**change},self.sources[original['source_id']])

    def test_correct_original_answer_cannot_justify_hidden_parameter_as_given(self):
        entry=copy.deepcopy(self.entries[1]);source=self.sources[entry['source_id']]
        # B itself is a hidden generation input: visible numeric questions
        # supply A and the percent. Using B matches gold but leaks the unknown.
        entry['spec']['templates']['numeric'][0]['answer']='b'
        with self.assertRaisesRegex(ValueError,'hidden generation parameter'):
            validate_entry(entry,source)

    def test_rejected_dimensional_ambiguity_never_gets_a_qualified_template(self):
        entry=next(e for e in self.entries if not e['accepted']);source=self.sources[entry['source_id']]
        result=validate_entry(entry,source)
        self.assertFalse(result['qualified']);self.assertNotIn('bundle',result)
        bad={**entry,'spec':self.entries[0]['spec']}
        with self.assertRaises(ValueError):validate_entry(bad,source)

    def test_exact_original_match_does_not_bypass_actual_template_review(self):
        entry=copy.deepcopy(self.entries[0]);entry['spec']['review']['accepted']=False
        with self.assertRaisesRegex(ValueError,'semantic review'):
            validate_entry(entry,self.sources[entry['source_id']])


if __name__=='__main__':unittest.main()
