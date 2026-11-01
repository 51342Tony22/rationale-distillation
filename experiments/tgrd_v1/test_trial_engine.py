"""Filesystem/state-machine tests; fake arithmetic is not evidence of learning."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from runtime_full import digest, read, freeze, baseline
from trial_engine import run_root


class FakeLearner:
    def __init__(self):
        self.reset(701);self.training_calls=[];self.optimizer_generation=0

    def reset(self,seed):
        self.state='S0';self.updates=0
        self.optimizer_generation=getattr(self,'optimizer_generation',0)+1

    def trial(self,question,completion,updates):
        if self.state!='S0' or self.updates!=0:
            raise AssertionError('Training inherited an earlier branch')
        self.training_calls.append((question,completion,updates,self.optimizer_generation))
        self.state=completion;self.updates=updates
        return [{'updates':i+1,'loss':1.} for i in range(updates)]

    def save(self,path):
        Path(path).write_text(json.dumps({'state':self.state}))


def restore(learner,path):
    learner.state=read(path)['state']


def weight_hash(learner):
    return digest(learner.state)


class TrialEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)/'root'
        q='A box has 3 red and 4 blue pencils. How many pencils?'
        self.source={'source_id':'synthetic:1','question':q,'question_sha256':digest(q)}
        self.pool={'source_id':'synthetic:1','question_sha256':digest(q),
            'teacher_output_sha256':digest('ordinary'),'candidates':[
                {'candidate_id':'e0','completion':'ordinary','completion_sha256':digest('ordinary'),'student_tokens':0},
                {'candidate_id':'explain','completion':'explanation','completion_sha256':digest('explanation'),'student_tokens':40}]}
        self.probes={split:[{'id':f'{split}:{i}','split':split} for i in range(6)]
                     for split in ('select','audit')}
        self.learner=FakeLearner();self.events=[];self.interrupt_once=False

    def evaluate(self,learner,probes,path):
        split=probes[0]['split'];self.events.append((split,learner.state))
        if split=='audit':
            self.assertTrue((self.folder/'decisions_before_audit.json').exists())
        if self.interrupt_once and path.parent.name=='branch1' and split=='select':
            self.interrupt_once=False
            raise InterruptedError('Synthetic interruption after trained adapter commit')
        if path.exists():return read(path)
        # The winner on Vsel is deliberately worse on Vaudit. A driver that
        # reselects from audit performance would choose a different branch.
        if split=='select':
            score={'S0':(0.,2.),'ordinary':(1/6,1.),'explanation':(.5,.5)}[learner.state]
        else:
            score={'ordinary':(.5,.8),'explanation':(0.,1.5)}[learner.state]
        result={'accuracy':score[0],'answer_nll':score[1]};freeze(path,result)
        return result

    def run_engine(self):
        return run_root(self.learner,self.source,self.pool,self.probes,self.folder,
            seed=701,updates=3,initial_hash=digest('S0'),weight_hash=weight_hash,
            restore=restore,evaluate=self.evaluate)

    def test_equal_fresh_branches_decisions_before_audit_and_no_audit_reselection(self):
        result=self.run_engine()
        self.assertEqual(len(self.learner.training_calls),2)
        self.assertEqual([call[2] for call in self.learner.training_calls],[3,3])
        self.assertNotEqual(*[call[3] for call in self.learner.training_calls])
        self.assertTrue(all(r['selected_id']=='explain' for r in result['configurations']))
        self.assertTrue(all(r['Vaudit_accuracy_gain']<0 for r in result['configurations']))
        first_audit=next(i for i,e in enumerate(self.events) if e[0]=='audit')
        self.assertTrue(all(e[0]=='select' for e in self.events[:first_audit]))
        self.assertEqual(self.learner.state,'S0');self.assertEqual(self.learner.updates,0)
        hashes={str(p):baseline.sha(p) for p in self.folder.rglob('*') if p.is_file()}
        self.assertEqual(self.run_engine(),result)
        self.assertEqual(len(self.learner.training_calls),2)
        self.assertEqual(hashes,{str(p):baseline.sha(p) for p in self.folder.rglob('*') if p.is_file()})

    def test_interrupted_generation_resumes_without_repeating_training(self):
        self.interrupt_once=True
        with self.assertRaises(InterruptedError):self.run_engine()
        self.assertEqual(len(self.learner.training_calls),2)
        self.assertTrue((self.folder/'branch1/trained.json').exists())
        self.assertFalse((self.folder/'decisions_before_audit.json').exists())
        self.run_engine()
        self.assertEqual(len(self.learner.training_calls),2)

    def test_corrupted_adapter_or_changed_material_cannot_resume(self):
        self.run_engine();adapter=self.folder/'branch0/adapter.safetensors'
        original=adapter.read_bytes();adapter.write_text('corrupt')
        with self.assertRaises(ValueError):self.run_engine()
        adapter.write_bytes(original)
        self.pool['candidates'][1]['completion']='changed'
        self.pool['candidates'][1]['completion_sha256']=digest('changed')
        with self.assertRaises(AssertionError):self.run_engine()

    def test_uncommitted_adapter_is_preserved_before_fresh_training(self):
        branch=self.folder/'branch0';branch.mkdir(parents=True)
        (branch/'adapter.safetensors').write_text('partial prior save')
        self.run_engine()
        archived=list(branch.glob('uncommitted_adapter_*.safetensors'))
        self.assertEqual(len(archived),1)
        self.assertEqual(archived[0].read_text(),'partial prior save')

    def test_no_qualified_explanation_uses_only_empty_branch(self):
        self.pool['candidates']=self.pool['candidates'][:1]
        result=self.run_engine()
        self.assertEqual(len(self.learner.training_calls),1)
        self.assertTrue(all(row['selected_id']=='e0' for row in result['configurations']))
        self.assertEqual(set(result['audit_scores']),{'e0'})


if __name__=='__main__':unittest.main()
