"""Root exposure, update budgets and actual tiny MLX CPU continuation checks."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from runtime_full import digest
from training_plan import schedule,validate_materials,CONDITIONS,SEEDS,DEFERRED_SEEDS
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint,verify_checkpoint


class ScheduleTests(unittest.TestCase):
    def test_same_root_order_equal_updates_and_final_partial_batch(self):
        for count in (1,5,7,7473):
            plan=schedule(count,42)
            self.assertEqual(len(plan),4*((count+4)//5))
            self.assertEqual(plan,schedule(count,42))
            for epoch in range(1,5):
                steps=[s for s in plan if s['epoch']==epoch]
                self.assertEqual(Counter(i for s in steps for i in s['root_indices']),Counter(range(count)))
                self.assertTrue(steps[-1]['epoch_end'])
                self.assertTrue(all(not s['epoch_end'] for s in steps[:-1]))
                self.assertEqual(len(steps[-1]['root_indices']),count%5 or 5)
        self.assertEqual(SEEDS,(42,))
        self.assertEqual(DEFERRED_SEEDS,(7,17))
        for undeclared_seed in (7,17,27):
            with self.assertRaises(ValueError):schedule(20,undeclared_seed)

    def fixture(self,condition):
        q='A box contains 3 red and 4 blue pencils. How many pencils?';text='3 + 4 = 7. The answer is 7.'
        material={'root_source_id':'gsm8k:train:0','question':q,'question_sha256':digest(q),
            'completion':text,'completion_sha256':digest(text),'kind':'ordinary_CoT'}
        return {'schema':'tgrd_training_materials_v1','condition':condition,'candidate_root_count':7473,
            'rows':[{'source_id':'gsm8k:train:0','question_sha256':digest(q),
                     'epoch_materials':[copy.deepcopy(material) for _ in range(4)]}]}

    def test_five_conditions_share_roots_and_only_M6_uses_two_distinct_Vsel_instances(self):
        self.assertEqual(CONDITIONS,('M1','NLL','M3','M7','M6'))
        for condition in CONDITIONS:validate_materials(self.fixture(condition),['gsm8k:train:0'])
        bundle=self.fixture('M6')
        for epoch in (1,3):
            m=bundle['rows'][0]['epoch_materials'][epoch]
            m.update(kind='verified_Vsel_variant',variant_id=f'root:select:numeric:{epoch}',split='select')
            m['question']=f'A box contains {epoch} pencils. How many pencils?'
            m['question_sha256']=digest(m['question'])
        validate_materials(bundle,['gsm8k:train:0'])
        bundle['rows'][0]['epoch_materials'][3]['variant_id']=bundle['rows'][0]['epoch_materials'][1]['variant_id']
        with self.assertRaises(ValueError):validate_materials(bundle)
        for condition in CONDITIONS:
            if condition=='M6':continue
            invalid=copy.deepcopy(bundle);invalid['condition']=condition
            with self.assertRaises(ValueError):validate_materials(invalid)

    def test_removed_groups_and_audit_variant_labels_are_rejected(self):
        for group in ('M2','M4','M5'):
            with self.assertRaises(ValueError):validate_materials(self.fixture(group))
        bundle=self.fixture('M6')
        for epoch in (1,3):
            material=bundle['rows'][0]['epoch_materials'][epoch]
            material.update(kind='verified_Vsel_variant',variant_id=f'root:audit:{epoch}',split='audit')
        with self.assertRaises(ValueError):validate_materials(bundle)

    def test_no_test_roots_wrong_hashes_or_group_specific_root_exclusions(self):
        for sid in ('gsm8k:test:0','gsm8k:train:7473','gsm8k:train:00'):
            b=self.fixture('M1');b['rows'][0]['source_id']=sid
            with self.assertRaises(ValueError):validate_materials(b)
        b=self.fixture('M1');b['rows'][0]['epoch_materials'][0]['completion']='changed'
        with self.assertRaises(ValueError):validate_materials(b)
        with self.assertRaises(ValueError):validate_materials(self.fixture('M1'),['gsm8k:train:1'])


class CheckpointMLXCPUTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        self.device=mx.default_device();mx.set_default_device(mx.cpu)
        self.addCleanup(lambda:mx.set_default_device(self.device))
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)

    def learner(self,seed):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(3,1);self.dropout=nn.Dropout(.25)
            def __call__(self,x):return self.linear(self.dropout(x))
        class Learner:
            def __init__(self):
                mx.random.seed(seed);self.model=Model();self.updates=0;self.tokenizer=None
                self.optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
                def loss(model,target=1.):
                    x=mx.array([[1.,2.,3.],[2.,0.,1.]])
                    return mx.mean((model(x)-mx.array([[target],[target+1.]]))**2)
                self.grad=nn.value_and_grad(self.model,loss)
            def step(self):
                self.model.train();loss,grads=self.grad(self.model)
                self.optimizer.update(self.model,grads)
                mx.eval(loss,self.model.parameters(),self.optimizer.state)
                self.updates+=1;self.model.eval();return float(loss.item())
            def weights(self):return dict(tree_flatten(self.model.trainable_parameters()))
            def save(self,path):mx.save_safetensors(str(path),self.weights())
            def update_batch(self,samples):
                self.model.train();loss,grads=self.grad(self.model,sum(samples)/len(samples))
                self.optimizer.update(self.model,grads)
                mx.eval(loss,self.model.parameters(),self.optimizer.state)
                self.updates+=1;self.model.eval()
                return {'updates':self.updates,'loss':float(loss.item()),'tokens':len(samples)*3}
        return Learner()

    def test_adapter_optimizer_and_random_state_restore_exact_next_update(self):
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        first=self.learner(7);first.step()
        folder=self.folder/'update_00000001'
        save_checkpoint(first,folder,{'update':1,'input_sha256':'frozen-input'})
        expected_loss=first.step()
        expected={k:np.asarray(v).copy() for k,v in first.weights().items()}
        expected_opt={k:np.asarray(v).copy() for k,v in tree_flatten(first.optimizer.state)}
        expected_random=mx.random.uniform(shape=(5,));mx.eval(expected_random)
        second=self.learner(999)
        self.assertEqual(restore_checkpoint(second,folder),{'update':1,'input_sha256':'frozen-input'})
        self.assertEqual(second.updates,1)
        self.assertEqual(second.step(),expected_loss)
        for k,v in second.weights().items():self.assertTrue(np.array_equal(np.asarray(v),expected[k]))
        for k,v in tree_flatten(second.optimizer.state):self.assertTrue(np.array_equal(np.asarray(v),expected_opt[k]))
        actual_random=mx.random.uniform(shape=(5,));mx.eval(actual_random)
        self.assertTrue(bool(mx.all(actual_random==expected_random).item()))

    def test_incomplete_checkpoint_is_ignored_but_corrupt_commit_is_not(self):
        learner=self.learner(7);learner.step()
        first=self.folder/'update_00000001';save_checkpoint(learner,first,{'update':1})
        (self.folder/'update_00000002').mkdir()  # no commit, never treated as resumable
        self.assertEqual(latest_checkpoint(self.folder),first)
        with self.assertRaises(FileExistsError):save_checkpoint(learner,first,{'update':1})
        learner.step();second=self.folder/'update_00000003'
        save_checkpoint(learner,second,{'update':2})
        (second/'state.safetensors').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):latest_checkpoint(self.folder)

    def engine_bundle(self,condition='M1'):
        bundle=ScheduleTests().fixture(condition);original=bundle['rows'][0]
        bundle['rows']=[]
        for index in range(7):
            row=copy.deepcopy(original);row['source_id']=f'gsm8k:train:{index}'
            for m in row['epoch_materials']:
                m['root_source_id']=row['source_id'];m['question']+=f' Fixture {index}.'
                m['question_sha256']=digest(m['question'])
            row['question_sha256']=row['epoch_materials'][0]['question_sha256']
            if condition=='M6':continue
            invalid=copy.deepcopy(bundle);invalid['condition']=condition
            with self.assertRaises(ValueError):validate_materials(invalid)

    def test_no_test_roots_wrong_hashes_or_group_specific_root_exclusions(self):
        for sid in ('gsm8k:test:0','gsm8k:train:7473','gsm8k:train:00'):
            b=self.fixture('M1');b['rows'][0]['source_id']=sid
            with self.assertRaises(ValueError):validate_materials(b)
        b=self.fixture('M1');b['rows'][0]['epoch_materials'][0]['completion']='changed'
        with self.assertRaises(ValueError):validate_materials(b)
        with self.assertRaises(ValueError):validate_materials(self.fixture('M1'),['gsm8k:train:1'])


class CheckpointMLXCPUTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        self.device=mx.default_device();mx.set_default_device(mx.cpu)
        self.addCleanup(lambda:mx.set_default_device(self.device))
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)

    def learner(self,seed):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(3,1);self.dropout=nn.Dropout(.25)
            def __call__(self,x):return self.linear(self.dropout(x))
        class Learner:
            def __init__(self):
                mx.random.seed(seed);self.model=Model();self.updates=0;self.tokenizer=None
                self.optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
                def loss(model,target=1.):
                    x=mx.array([[1.,2.,3.],[2.,0.,1.]])
                    return mx.mean((model(x)-mx.array([[target],[target+1.]]))**2)
                self.grad=nn.value_and_grad(self.model,loss)
            def step(self):
                self.model.train();loss,grads=self.grad(self.model)
                self.optimizer.update(self.model,grads)
                mx.eval(loss,self.model.parameters(),self.optimizer.state)
                self.updates+=1;self.model.eval();return float(loss.item())
            def weights(self):return dict(tree_flatten(self.model.trainable_parameters()))
            def save(self,path):mx.save_safetensors(str(path),self.weights())
            def update_batch(self,samples):
                self.model.train();loss,grads=self.grad(self.model,sum(samples)/len(samples))
                self.optimizer.update(self.model,grads)
                mx.eval(loss,self.model.parameters(),self.optimizer.state)
                self.updates+=1;self.model.eval()
                return {'updates':self.updates,'loss':float(loss.item()),'tokens':len(samples)*3}
        return Learner()

    def test_adapter_optimizer_and_random_state_restore_exact_next_update(self):
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        first=self.learner(7);first.step()
        folder=self.folder/'update_00000001'
        save_checkpoint(first,folder,{'update':1,'input_sha256':'frozen-input'})
        expected_loss=first.step()
        expected={k:np.asarray(v).copy() for k,v in first.weights().items()}
        expected_opt={k:np.asarray(v).copy() for k,v in tree_flatten(first.optimizer.state)}
        expected_random=mx.random.uniform(shape=(5,));mx.eval(expected_random)
        second=self.learner(999)
        self.assertEqual(restore_checkpoint(second,folder),{'update':1,'input_sha256':'frozen-input'})
        self.assertEqual(second.updates,1)
        self.assertEqual(second.step(),expected_loss)
        for k,v in second.weights().items():self.assertTrue(np.array_equal(np.asarray(v),expected[k]))
        for k,v in tree_flatten(second.optimizer.state):self.assertTrue(np.array_equal(np.asarray(v),expected_opt[k]))
        actual_random=mx.random.uniform(shape=(5,));mx.eval(actual_random)
        self.assertTrue(bool(mx.all(actual_random==expected_random).item()))

    def test_incomplete_checkpoint_is_ignored_but_corrupt_commit_is_not(self):
        learner=self.learner(7);learner.step()
        first=self.folder/'update_00000001';save_checkpoint(learner,first,{'update':1})
        (self.folder/'update_00000002').mkdir()  # no commit, never treated as resumable
        self.assertEqual(latest_checkpoint(self.folder),first)
        with self.assertRaises(FileExistsError):save_checkpoint(learner,first,{'update':1})
        learner.step();second=self.folder/'update_00000003'
        save_checkpoint(learner,second,{'update':2})
        (second/'state.safetensors').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):latest_checkpoint(self.folder)

    def engine_bundle(self,condition='M1'):
        bundle=ScheduleTests().fixture(condition);original=bundle['rows'][0]
        bundle['rows']=[]
        for index in range(7):
            row=copy.deepcopy(original);row['source_id']=f'gsm8k:train:{index}'
            for m in row['epoch_materials']:
                m['root_source_id']=row['source_id'];m['question']+=f' Fixture {index}.'
                m['question_sha256']=digest(m['question'])
            row['question_sha256']=row['epoch_materials'][0]['question_sha256']
            if condition=='M6':
                for epoch in (1,3):
                    m=row['epoch_materials'][epoch];m['question']+=f' Variant {epoch}.'
                    m.update(question_sha256=digest(m['question']),kind='verified_Vsel_variant',
                             variant_id=f'{index}:select:{epoch}',split='select')
            bundle['rows'].append(row)
        return bundle

    def run_engine(self,learner,bundle,folder,**kwargs):
        from training_engine import run_training
        def weight_hash(value):
            return hashlib.sha256(b''.join(k.encode()+memoryview(v).tobytes() for k,v in sorted(value.weights().items()))).hexdigest()
        def tokenize(_,question,completion):return int(digest(question+completion)[:6],16)/2**24
        return run_training(learner,bundle,folder,seed=42,tokenize=tokenize,weight_hash=weight_hash,
            save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,
            latest_checkpoint=latest_checkpoint,checkpoint_interval=3,**kwargs)

    def test_engine_replays_uncommitted_update_and_exactly_restores_trajectory(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        import numpy as np
        bundle=self.engine_bundle();reference=self.learner(42)
        expected=self.run_engine(reference,bundle,self.folder/'reference')
        expected_state={k:np.asarray(v).copy() for k,v in tree_flatten(reference.optimizer.state)}
        expected_rng=[np.asarray(v).copy() for v in mx.random.state]
        interrupted=self.learner(42)
        def stop(step):
            if step['update']==5:raise InterruptedError('Uncommitted update5;epoch2 committed at4')
        with self.assertRaises(InterruptedError):
            self.run_engine(interrupted,bundle,self.folder/'resumed',after_update=stop)
        self.assertEqual(verify_checkpoint(latest_checkpoint(self.folder/'resumed/checkpoints'))['progress']['update'],4)
        # Simulate termination before writing an epoch index; resume must repair it.
        (self.folder/'resumed/epochs/epoch2.json').unlink()
        resumed=self.learner(42);actual=self.run_engine(resumed,bundle,self.folder/'resumed')
        self.assertEqual(expected['adapter_sha256'],actual['adapter_sha256'])
        self.assertEqual(actual['root_exposures'],28);self.assertEqual(actual['updates'],8)
        self.assertEqual(actual['supervised_tokens_including_eos'],84)
        for k,v in tree_flatten(resumed.optimizer.state):self.assertTrue(np.array_equal(np.asarray(v),expected_state[k]))
        for a,b in zip(mx.random.state,expected_rng):self.assertTrue(np.array_equal(np.asarray(a),b))
        events=[json.loads(line) for line in (self.folder/'resumed/events.jsonl').read_text().splitlines()]
        updates=[e['update'] for e in events if e['event']=='optimizer_update']
        self.assertEqual(updates.count(5),2)  # replayed compute is retained, not double-counted as exposure
        for epoch in range(1,5):
            index=json.loads((self.folder/f'resumed/epochs/epoch{epoch}.json').read_text())
            self.assertEqual(index['update'],2*epoch);self.assertEqual(index['primary'],epoch==4)
        cached=self.run_engine(self.learner(42),bundle,self.folder/'resumed')
        self.assertEqual(actual,cached)
        self.assertEqual(events,[json.loads(line) for line in (self.folder/'resumed/events.jsonl').read_text().splitlines()])

    def test_engine_NLL_keeps_original_roots_and_rejects_changed_inputs(self):
        bundle=self.engine_bundle('NLL');folder=self.folder/'NLL'
        self.run_engine(self.learner(42),bundle,folder)
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        for epoch in range(1,5):
            updates=[e for e in events if e.get('event')=='optimizer_update' and e['epoch']==epoch]
            self.assertEqual(Counter(sid for e in updates for sid in e['root_ids']),
                             Counter(r['source_id'] for r in bundle['rows']))
            for event in updates:
                self.assertEqual(set(event['material_kinds']),
                    {'ordinary_CoT'})
                for sid,checksum in zip(event['root_ids'],event['question_sha256']):
                    row=next(r for r in bundle['rows'] if r['source_id']==sid)
                    self.assertEqual(checksum,row['epoch_materials'][epoch-1]['question_sha256'])
        altered=copy.deepcopy(bundle)
        for m in altered['rows'][0]['epoch_materials']:
            m['completion']+=' changed';m['completion_sha256']=digest(m['completion'])
        with self.assertRaises((ValueError,AssertionError)):self.run_engine(self.learner(42),altered,folder)

    def test_engine_M6_switches_epoch_materials_and_rejects_changed_inputs(self):
        bundle=self.engine_bundle('M6');folder=self.folder/'M6'
        self.run_engine(self.learner(42),bundle,folder)
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        for epoch in range(1,5):
            updates=[e for e in events if e.get('event')=='optimizer_update' and e['epoch']==epoch]
            self.assertEqual(Counter(sid for e in updates for sid in e['root_ids']),
                             Counter(r['source_id'] for r in bundle['rows']))
            for event in updates:
                self.assertEqual(set(event['material_kinds']),
                    {'ordinary_CoT' if epoch in (1,3) else 'verified_Vsel_variant'})
                for sid,checksum in zip(event['root_ids'],event['question_sha256']):
                    row=next(r for r in bundle['rows'] if r['source_id']==sid)
                    self.assertEqual(checksum,row['epoch_materials'][epoch-1]['question_sha256'])
        altered=copy.deepcopy(bundle)
        for m in altered['rows'][0]['epoch_materials']:
            m['completion']+=' changed';m['completion_sha256']=digest(m['completion'])
        with self.assertRaises((ValueError,AssertionError)):self.run_engine(self.learner(42),altered,folder)


if __name__=='__main__':unittest.main()
