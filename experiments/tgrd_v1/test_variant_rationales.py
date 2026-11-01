"""M6 observed-only supervision and generator-answer leakage regression checks."""
import copy
import unittest
from fractions import Fraction
from runtime_full import DATA,read,digest
from variant_rationales import root_rationales,derivation,observable_expression,symbolic
from variants import exact


class VariantRationaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundles=read(DATA/'cal40_reviewed_variants_v2.json')
        cls.sources={r['source_id']:r for r in read(DATA/'cal40.json')}

    def test_all204_selection_derivations_and_fixed_M6_order(self):
        count=0
        for bundle in self.bundles:
            result=root_rationales(bundle,self.sources[bundle['source_id']])
            self.assertEqual(result['M6_fixed_variant_ids'],[p['id'] for p in bundle['variants']['select'][:2]])
            self.assertFalse(result['Vaudit_rationales_generated'])
            for material,probe in zip(result['selection_rationales'],bundle['variants']['select']):
                count+=1
                self.assertEqual(material['variant_id'],probe['id'])
                self.assertEqual(material['completion_sha256'],digest(material['completion']))
                self.assertEqual(exact(material['observable_expression'],material['observed_parameters']),Fraction(probe['gold']))
                self.assertIn('\\boxed{'+probe['gold']+'}',material['completion'])
        self.assertEqual(count,204)

    def test_inverse_does_not_print_hidden_answer_parameter_as_given(self):
        bundle=next(b for b in self.bundles if b['source_id']=='gsm8k:train:3146')
        material=root_rationales(bundle,self.sources[bundle['source_id']])['selection_rationales'][0]
        # Generator base has rounds=16. Question gives buckets=192 and two
        # rates=5/7, then asks rounds. A generator-tuple dump would leak it.
        self.assertIn('rounds',bundle['variants']['select'][0]['base_parameters'])
        self.assertEqual(set(material['observed_parameters']),{'a','b','total'})
        self.assertNotIn('rounds',material['completion'])
        self.assertEqual(material['arithmetic_trace'][-1]['result'],'16')
        self.assertEqual(material['arithmetic_trace'][0]['result'],'12')

    def test_hidden_base_reference_cycles_and_unsupported_code_fail(self):
        for expression,visible,derived in [
            ('answer',{'total'},{'label':'total/2'}),
            ('a',{'given'},{'a':'b+1','b':'a+1'}),
            ('__import__("os")',set(),{}),
            ('x ** 2',{'x'},{}),
        ]:
            with self.assertRaises(ValueError):observable_expression(expression,visible,derived)
        tree=observable_expression('alias',{'total','rate'},{'alias':'total/rate'})
        self.assertEqual(exact(symbolic(tree),{'total':3,'rate':2}),Fraction(3,2))

    def test_audit_mislabeled_question_and_gold_cannot_be_supervision(self):
        bundle=self.bundles[0];spec=bundle['spec'];template=spec['templates']['numeric'][0]
        original=bundle['variants']['select'][0]
        audit=bundle['variants']['audit'][0]
        with self.assertRaises(ValueError):derivation(spec,template,audit)
        for field,value in [('gold','999999'),('question',original['question']+' Changed?'),('answer_expression','0')]:
            altered=copy.deepcopy(original);altered[field]=value
            with self.assertRaises(ValueError):derivation(spec,template,altered)
        altered=copy.deepcopy(bundle);altered['variants']['audit'][0]['gold']='999999'
        with self.assertRaises(ValueError):root_rationales(altered,self.sources[bundle['source_id']])

    def test_fractional_intermediates_are_exact_not_rounded(self):
        bundle=next(b for b in self.bundles if b['source_id']=='gsm8k:train:4488')
        material=root_rationales(bundle,self.sources[bundle['source_id']])['selection_rationales'][0]
        # 60/10 - 60/36 = 6 - 5/3 = 13/3, without decimal rounding.
        self.assertEqual(material['gold'],'13/3')
        self.assertEqual([s['result'] for s in material['arithmetic_trace']],['6','5/3','13/3'])
        self.assertEqual(material['arithmetic_trace'][-1]['substitution'],'6 - (5/3)')
        self.assertNotIn('1.666',material['completion'])


if __name__=='__main__':unittest.main()
