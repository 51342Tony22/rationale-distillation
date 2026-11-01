"""Finite trial state-machine checks; synthetic scores are not learning evidence."""
import copy
import unittest
from runtime_full import read, freeze, digest, baseline
import test_trial_engine as fixtures
from test_trial_engine import restore, weight_hash
from fixed_trial_engine_v1 import run_fixed_root


class FixedTrialTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.TrialEngineTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.folder=self.f.folder;self.events=[];self.interrupt=False
        self.threshold=.03

    def evaluate(self,learner,probes,path):
        split=probes[0]['split'];self.events.append((split,learner.state))
        if split=='audit':
            self.assertTrue((self.folder/'decision_before_audit.json').exists())
            self.assertFalse((self.folder/'trial_result.json').exists() and not path.exists())
        if self.interrupt and path.parent.name=='branch1' and split=='select':
            self.interrupt=False;raise InterruptedError('After trained adapter commit')
        if path.exists():return read(path)
        scores=({'S0':(0.,2.),'ordinary':(1/6,1.),'explanation':(.5,.5)} if split=='select'
                else {'ordinary':(.5,.8),'explanation':(0.,1.5)})
        a,n=scores[learner.state];result={'accuracy':a,'answer_nll':n};freeze(path,result);return result

    def run_engine(self,**kwargs):
        return run_fixed_root(self.f.learner,self.f.source,self.f.pool,self.f.probes,self.folder,
            **dict(seed=701,updates=3,initial_hash=digest('S0'),weight_hash=weight_hash,
                restore=restore,evaluate=self.evaluate,threshold=self.threshold,
                configuration_sha256=digest('synthetic_configuration'),**kwargs))

    def test_one_threshold_only_choice_is_frozen_before_worse_audit(self):
        r=self.run_engine();self.assertEqual(r['decision']['selected_id'],'explain')
        self.assertLess(r['Vaudit_accuracy_gain'],0);self.assertFalse(r['other_thresholds_evaluated'])
        self.assertEqual(set(r['audit_scores']),{'e0','explain'})
        inputs=read(self.folder/'inputs.json');self.assertNotIn('thresholds',inputs)
        self.assertEqual(inputs['threshold'],.03)
        first=next(i for i,(split,_) in enumerate(self.events) if split=='audit')
        self.assertTrue(all(split=='select' for split,_ in self.events[:first]))
        self.assertEqual(len(self.f.learner.training_calls),2)
        self.assertEqual({c[0] for c in self.f.learner.training_calls},{self.f.source['question']})
        self.assertEqual(self.f.learner.updates,0)

    def test_resume_does_not_repeat_original_root_updates_or_modify_commits(self):
        self.interrupt=True
        with self.assertRaises(InterruptedError):self.run_engine()
        self.assertEqual(len(self.f.learner.training_calls),2)
        r=self.run_engine();before={str(p):baseline.sha(p) for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(self.run_engine(),r);self.assertEqual(len(self.f.learner.training_calls),2)
        self.assertEqual(before,{str(p):baseline.sha(p) for p in self.folder.rglob('*') if p.is_file()})

    def test_changed_frozen_configuration_or_corrupt_adapter_cannot_resume(self):
        self.run_engine();self.threshold=.1
        with self.assertRaises(AssertionError):self.run_engine()
        self.threshold=.03;(self.folder/'branch0/adapter.safetensors').write_text('tampered')
        with self.assertRaises(ValueError):self.run_engine()

    def test_invalid_configuration_rejected_before_training_or_disk(self):
        defaults=dict(seed=701,updates=3,initial_hash=digest('S0'),weight_hash=weight_hash,
            restore=restore,evaluate=self.evaluate,threshold=.03,configuration_sha256=digest('c'))
        for change in ({'seed':1701},{'updates':2},{'threshold':True},{'configuration_sha256':''}):
            args={**defaults,**change}
            with self.assertRaises(ValueError):
                run_fixed_root(self.f.learner,self.f.source,self.f.pool,self.f.probes,self.folder,**args)
        self.assertFalse(self.folder.exists());self.assertFalse(self.f.learner.training_calls)

    def test_no_nonempty_branch_audits_only_equal_budget_e0(self):
        self.f.pool['candidates']=self.f.pool['candidates'][:1];r=self.run_engine()
        self.assertEqual(r['decision']['selected_id'],'e0');self.assertEqual(set(r['audit_scores']),{'e0'})
        self.assertEqual(len(self.f.learner.training_calls),1)

    def test_probe_split_overlap_rejected(self):
        self.f.probes['audit'][0]['id']=self.f.probes['select'][0]['id']
        with self.assertRaises(ValueError):self.run_engine()
        self.assertFalse(self.folder.exists())


class FixedTrialMLXCPUIntegration(unittest.TestCase):
    def test_actual_tiny_CPU_adapter_cold_resume_with_fresh_optimizer(self):
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np
        import test_training as training
        fixture=fixtures.TrialEngineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        device=mx.default_device();mx.set_default_device(mx.cpu)
        self.addCleanup(lambda:mx.set_default_device(device))
        factory=training.CheckpointMLXCPUTests()
        class Tiny:
            def __init__(self):
                self.inner=factory.learner(701);mx.eval(self.inner.weights())
                self.initial={k:mx.array(v) for k,v in self.inner.weights().items()}
                mx.eval(self.initial);self.reset(701);self.trained_calls=0
            @property
            def updates(self):return self.inner.updates
            def reset(self,seed):
                self.inner.model.load_weights(list(self.initial.items()),strict=False)
                mx.eval(self.inner.model.parameters());mx.random.seed(seed)
                self.inner.optimizer=optim.AdamW(learning_rate=1e-4,weight_decay=.01)
                self.inner.updates=0;self.inner.model.eval()
            def trial(self,question,completion,updates):
                self.trained_calls+=1
                return [self.inner.update_batch([1. if completion=='ordinary' else 2.]) for _ in range(updates)]
            def weights(self):return self.inner.weights()
            def save(self,path):self.inner.save(path)
        def tensor_hash(learner):
            mx.eval(learner.weights())
            return digest(''.join(k+np.asarray(v).tobytes().hex() for k,v in sorted(learner.weights().items())))
        def load(learner,path):
            learner.inner.model.load_weights(str(path),strict=False)
            mx.eval(learner.inner.model.parameters());learner.inner.model.eval()
        interrupted=[False]
        def evaluate(learner,probes,path):
            if path.parent.name=='branch1' and probes[0]['split']=='select' and not interrupted[0]:
                interrupted[0]=True;raise InterruptedError('Actual CPU adapter already committed')
            checksum=tensor_hash(learner)
            score={'accuracy':.5 if path.parent.name=='branch1' else 0.,
                'answer_nll':.5 if path.parent.name=='branch1' else 1.}
            if path.exists():self.assertEqual(read(path),{'tensor_sha256':checksum,**score})
            else:freeze(path,{'tensor_sha256':checksum,**score})
            if probes[0]['split']=='audit':self.assertTrue((fixture.folder/'decision_before_audit.json').exists())
            return score
        first=Tiny();initial=tensor_hash(first)
        args=dict(seed=701,updates=1,initial_hash=initial,weight_hash=tensor_hash,
            restore=load,evaluate=evaluate,threshold=.03,configuration_sha256=digest('CPU fixture config'))
        with self.assertRaises(InterruptedError):
            run_fixed_root(first,fixture.source,fixture.pool,fixture.probes,fixture.folder,**args)
        self.assertEqual(first.trained_calls,2)
        second=Tiny();self.assertEqual(tensor_hash(second),initial)
        result=run_fixed_root(second,fixture.source,fixture.pool,fixture.probes,fixture.folder,**args)
        self.assertEqual(second.trained_calls,0);self.assertEqual(result['decision']['selected_id'],'explain')
        self.assertEqual(tensor_hash(second),initial)


if __name__=='__main__':unittest.main()
