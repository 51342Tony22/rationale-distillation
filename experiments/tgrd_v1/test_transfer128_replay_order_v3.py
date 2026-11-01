"""Order-only compatibility tests; no model or official test loading."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import transfer128_replay_order_v3 as m


class ReplayOrderTests(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'manifest.json'
        value = {'schema':'fixture','inputs':{'source':'sha'},'roots':[
            {'source_id':'a','value':1},{'source_id':'b','value':2}]}
        path.write_text(json.dumps(value))
        return path,value

    def test_permutation_is_allowed_without_writing_bytes(self):
        path,value=self.fixture();before=path.read_bytes()
        reversed_value={**value,'roots':list(reversed(value['roots']))}
        self.assertEqual(m.verify_equivalent(path,reversed_value),value)
        self.assertEqual(before,path.read_bytes())

    def test_entry_content_change_is_rejected(self):
        path,value=self.fixture();value=copy.deepcopy(value);value['roots'][0]['value']=3
        with self.assertRaises(ValueError):m.verify_equivalent(path,value)

    def test_duplicate_and_missing_roots_are_rejected(self):
        path,value=self.fixture()
        for rows in ([value['roots'][0]]*2,value['roots'][:1]):
            with self.assertRaises(ValueError):m.verify_equivalent(path,{**value,'roots':rows})

    def test_metadata_and_input_hash_changes_are_rejected(self):
        path,value=self.fixture()
        for change in ({'schema':'changed'},{'inputs':{'source':'changed'}},{'extra':True}):
            with self.assertRaises(ValueError):m.verify_equivalent(path,{**value,**change})

    def test_non_root_manifest_requires_exact_equality(self):
        path,value=self.fixture();value={'outputs':{'file':'sha'}};path.write_text(json.dumps(value))
        self.assertEqual(m.verify_equivalent(path,value),value)
        with self.assertRaises(ValueError):m.verify_equivalent(path,{'outputs':{'file':'changed'}})

    def test_rationale_root_key_is_explicit_and_duplicate_safe(self):
        r={'selection_rationales':[{'root_source_id':'a','completion':'x'}]}
        self.assertEqual(m.keyed([r]),{'a':r})
        with self.assertRaises(ValueError):m.keyed([r,r])

    def test_canonical_index_preserves_roster_and_all_values(self):
        ids=[str(i) for i in range(128)]
        report={'roots':[{'source_id':i,'qualified':True} for i in reversed(ids)],'metadata':'same'}
        actual=m.canonical_report(report,ids)
        self.assertEqual([r['source_id'] for r in actual['roots']],ids)
        self.assertEqual(m.keyed(actual['roots']),m.keyed(report['roots']))
        self.assertEqual(actual['metadata'],'same')
        with self.assertRaises(ValueError):m.canonical_report(report,ids[:-1])

    def test_actual_frozen_batches_replay_identically_across_hash_seeds(self):
        script='''import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('experiments/tgrd_v1').resolve()))
import transfer128_variant_manifest_v2 as legacy
import transfer128_replay_order_v3 as adapter
adapter._original_prepare=legacy.prepare
files=sorted(legacy.FOLDER.glob('*v2.json'))
before={str(p):p.read_bytes() for p in files}
report=adapter.replay_variants()
assert before=={str(p):p.read_bytes() for p in files}
assert report['qualified_roots']==124 and report['rejected_roots']==4
print(json.dumps({'roots':report['frozen_roots'],'qualified':report['qualified_roots']}))
'''
        outputs=[]
        for seed in ('1','73'):
            env={**os.environ,'PYTHONHASHSEED':seed}
            result=subprocess.run([sys.executable,'-c',script],cwd=m.ROOT,env=env,
                                  text=True,capture_output=True,check=True)
            outputs.append(json.loads(result.stdout.splitlines()[-1]))
        self.assertEqual(outputs,[{'roots':128,'qualified':124}]*2)


if __name__ == '__main__':unittest.main()
