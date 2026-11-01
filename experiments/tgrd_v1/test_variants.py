"""Meaningful validation for exact labels, instance isolation and code rejection."""
import unittest
import json
import copy
from fractions import Fraction
from variants import exact,number,render,build,validate_spec,solve
from trial_selection import validate_probes


class VariantTests(unittest.TestCase):
    def spec(self):
        return {'source_id':'root:1','original_parameters':{'a':3,'b':4,'rounds':5},
                'derived':{'total':'(a+b)*rounds'},'original_answer':'total/(a+b)',
                'domains':{'a':[2,3,4,5,6],'b':[3,4,5,6,7],'rounds':[5,6,7,8,9]},
                'review':{'accepted':True,'kind':'assistant_mathematical_review','reason':'Two carriers add capacities; total divided by joint capacity yields rounds.'},
                'templates':{
                  'numeric':[{'id':'n_sel','question':'A carries {a} buckets and B carries {b} each round. How many rounds to move {total} buckets?','answer':'total/(a+b)'},
                             {'id':'n_audit','question':'Two people move {a} and {b} buckets per round. How many rounds for {total} buckets?','answer':'total/(a+b)'}],
                  'wording':[{'id':'w_sel','question':'Two workers jointly carry {a} plus {b} buckets each trip. How many trips for {total} buckets?','answer':'rounds'},
                             {'id':'w_audit','question':'A pool needs {total} buckets. Partners carry {a} and {b} each trip. How many trips?','answer':'rounds'}],
                  'structural':[{'id':'s_sel','question':'A carries {a} and B carries {b} buckets per round for {rounds} rounds. How many buckets together?','answer':'total'},
                                {'id':'s_audit','question':'Two workers make {rounds} rounds carrying {a} and {b} buckets. How many buckets in total?','answer':'total'}]}}

    def test_exact_decimal_and_fraction(self):
        self.assertEqual(exact('a/3 + b',{'a':'1','b':'0.1'}),Fraction(13,30))
        self.assertEqual(number(Fraction(1,8)),'0.125')
        self.assertEqual(number(Fraction(1,3)),'1/3')
        self.assertEqual(number(Fraction(120)),'120')

    def test_rejects_code_and_unknown_parameters(self):
        for expression in ['__import__("os").system("echo bad")','a.__class__','a[0]','2**1000000','[x for x in a]','float(a)','unknown + 1']:
            with self.assertRaises((ValueError,SyntaxError)):
                exact(expression,{'a':3})
        with self.assertRaises(ZeroDivisionError):exact('1/0',{})
        with self.assertRaises(ValueError):render('{a.__class__}',{'a':3})

    def test_split_isolation_and_structural_recomputation(self):
        source={'source_id':'root:1','question':'Original question?','gold':'5'}
        a=build(self.spec(),source)
        self.assertEqual(a,build(self.spec(),source))
        self.assertEqual([len(a[k]) for k in ['select','audit']],[6,6])
        select={tuple(sorted(x['base_parameters'].items())) for x in a['select']}
        audit={tuple(sorted(x['base_parameters'].items())) for x in a['audit']}
        self.assertFalse(select & audit)
        for split in a.values():
            for row in split:
                p=row['base_parameters']
                expected=(p['a']+p['b'])*p['rounds'] if row['kind']=='structural' else p['rounds']
                self.assertEqual(Fraction(row['gold']),expected)

    def test_bad_gold_and_missing_review_fail_closed(self):
        with self.assertRaises(ValueError):validate_spec(self.spec(),{'source_id':'root:1','gold':'99'})
        spec=self.spec();spec['review']['accepted']=False
        with self.assertRaises(ValueError):build(spec,{'source_id':'root:1','gold':'5','question':'Original?'})

    def test_integer_constraints_apply_to_given_counts_too(self):
        spec=self.spec()
        spec['derived']['count']='a/b'
        spec['integer_constraints']=['count']
        template={'question':'There are {count} buckets. How many buckets?','answer':'a+b'}
        with self.assertRaises(ValueError):solve(spec,template,{'a':3,'b':4,'rounds':5})

    def test_declared_sampling_order_survives_sorted_json_round_trip(self):
        spec=self.spec();spec['sampling_parameter_order']=list(spec['domains'])
        source={'source_id':'root:1','question':'Original question?','gold':'5'}
        bundle={'source_id':'root:1','spec':spec,'variants':build(spec,source)}
        serialized=json.loads(json.dumps(bundle,sort_keys=True))
        self.assertEqual(build(serialized['spec'],source),bundle['variants'])
        self.assertEqual(validate_probes(serialized,source,build),bundle['variants'])
        for order in [['a','b'],['a','a','rounds'],['a','b','unknown']]:
            with self.assertRaises(ValueError):
                build({**spec,'sampling_parameter_order':order},source)

    def test_legacy_frozen_instances_are_replayed_and_tampering_is_rejected(self):
        spec=self.spec();source={'source_id':'root:1','question':'Original question?','gold':'5'}
        bundle=json.loads(json.dumps({'source_id':'root:1','spec':spec,
                                    'variants':build(spec,source)},sort_keys=True))
        validate_probes(bundle,source,build)
        for field,value in [('gold','999'),('question','A different question?'),
                            ('question_sha256','wrong'),('answer_expression','999'),
                            ('template_id','wrong'),('id','other:select:numeric:0'),
                            ('parameters',{}),('base_parameters',{'a':999,'b':4,'rounds':5})]:
            tampered=copy.deepcopy(bundle);tampered['variants']['select'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_probes(tampered,source,build)
        tampered=copy.deepcopy(bundle)
        duplicate=copy.deepcopy(tampered['variants']['select'][0]);duplicate['split']='audit'
        duplicate['id']='root:1:audit:numeric:0';duplicate['template_id']='n_audit'
        tampered['variants']['audit'][0]=duplicate
        with self.assertRaises(ValueError):validate_probes(tampered,source,build)


if __name__=='__main__':unittest.main()
