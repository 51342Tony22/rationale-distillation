"""Public evaluation guards tested only with synthetic local artifact files."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluation_inputs import validate_request
from runtime_full import save,read,baseline
from training_plan import CONDITIONS,SEEDS,PROTOCOL
from learning import LORA


class EvaluationInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'output/tgrd_v1'
        self.training={'enabled':True,'full_teacher_quality_review_complete':True,'development_accepted':True,
            'protocol':PROTOCOL,
            'conditions':list(CONDITIONS),'seeds':list(SEEDS),'common_root_ids':['gsm8k:train:0']}
        save(self.out/'gates/final_training_frozen.json',self.training)
        stub=self.root/'synthetic_inference_preflight_stub.txt';stub.write_text('unit-test provenance stub only')
        proof=self.out/'signatures/adapter_loading_gpu_v1.json'
        save(proof,{'inputs':{stub.name:baseline.sha(stub)}})
        save(self.out/'preflight/adapter_loading_gpu_v1.json',{'status':'passed',
            'signature_file_sha256':baseline.sha(proof),'synthetic_unit_test_only':True})
        self.gate={'enabled':True,'primary_dataset':'GSM8K','protocol_frozen_before_public_test':True,
            'protocol':PROTOCOL,
            'development_accepted':True,'full_teacher_quality_review_complete':True,'formal_training_complete':True,
            'conditions':list(CONDITIONS)+['student_base','teacher_reference'],
            'seeds':{**{g:list(SEEDS) for g in CONDITIONS},'student_base':[42],'teacher_reference':[42]},
            'training_gate_sha256':baseline.sha(self.out/'gates/final_training_frozen.json'),'artifacts':{}}
        for group in CONDITIONS:
            self.gate['artifacts'][group]={}
            for seed in SEEDS:self.fixture(group,seed)
        self.write_gate()

    def fixture(self,group,seed):
        folder=self.out/'training'/group/f'seed{seed}';folder.mkdir(parents=True)
        checkpoint=folder/'checkpoints/update_00000004';checkpoint.mkdir(parents=True)
        # These are deliberately fake binary artifacts, never loaded as a model.
        weight=(group+str(seed)+' synthetic unit-test adapter').encode()
        (folder/'adapters.safetensors').write_bytes(weight)
        (checkpoint/'adapter.safetensors').write_bytes(weight)
        (checkpoint/'state.safetensors').write_bytes(b'synthetic optimizer state')
        save(checkpoint/'state_structure.json',{'synthetic_unit_test_only':True})
        save(folder/'adapter_config.json',{'fine_tune_type':'lora','num_layers':16,'lora_parameters':LORA})
        save(folder/'inputs.json',{'condition':group,'seed':seed,'ordered_root_ids':['gsm8k:train:0'],
            'primary_checkpoint':'last_epoch4','updates':4})
        sha=baseline.sha(folder/'adapters.safetensors');input_sha=baseline.sha(folder/'inputs.json')
        save(folder/'metadata.json',{'completed':True,'condition':group,'seed':seed,'epochs':4,
            'accepted_root_count':1,'candidate_root_count':7473,'updates':4,'root_exposures':4,
            'primary_checkpoint':'last_epoch4','public_test_used_to_select_checkpoint':False,
            'adapter_sha256':sha,'input_file_sha256':input_sha})
        save(folder/'epochs/epoch4.json',{'epoch':4,'update':4,'primary':True,'adapter_sha256':sha,
            'checkpoint_directory':'checkpoints/update_00000004'})
        save(checkpoint/'commit.json',{'schema':'tgrd_training_checkpoint_v1','learner_updates':4,
            'progress':{'update':4,'epoch':4,'epoch_end':True,'root_exposures':4,'input_file_sha256':input_sha},
            'files':{name:baseline.sha(checkpoint/name) for name in
                ('adapter.safetensors','state.safetensors','state_structure.json')}})
        files=['adapters.safetensors','adapter_config.json','metadata.json','inputs.json','epochs/epoch4.json',
            'checkpoints/update_00000004/commit.json']
        self.gate['artifacts'][group][str(seed)]={'adapter_directory':str(folder.relative_to(self.root)),
            'files':{name:baseline.sha(folder/name) for name in files}}

    def write_gate(self):save(self.out/'gates/public_evaluation_frozen.json',self.gate)

    def validate(self,condition='M7',seed=42,adapter=None):
        return validate_request(condition,seed,adapter,root=self.root,out=self.out)

    def test_adapter_directory_and_all_loaded_files_are_bound_to_epoch4(self):
        result=self.validate()
        self.assertEqual(result['adapter_directory'],(self.out/'training/M7/seed42').resolve())
        self.assertTrue(all(p.is_file() for p in result['signature_paths']))
        self.assertEqual(result['training_artifact']['primary_checkpoint'],'last_epoch4')
        self.assertEqual(result['training_artifact']['accepted_root_count'],1)
        with self.assertRaises(ValueError):self.validate(adapter=self.out/'training/M1/seed42')
        with self.assertRaises(ValueError):self.validate(adapter=self.out/'training/M7/seed42/adapters.safetensors')

    def test_weight_configuration_and_commit_corruption_fail(self):
        folder=self.out/'training/M7/seed42'
        for name in ('adapters.safetensors','adapter_config.json','checkpoints/update_00000004/state.safetensors'):
            original=(folder/name).read_bytes();(folder/name).write_bytes(b'corrupted')
            with self.assertRaises((ValueError,json.JSONDecodeError)):self.validate()
            (folder/name).write_bytes(original)
        bad=read(folder/'epochs/epoch4.json');bad['primary']=False
        save(folder/'epochs/epoch4.json',bad)
        self.gate['artifacts']['M7']['42']['files']['epochs/epoch4.json']=baseline.sha(folder/'epochs/epoch4.json');self.write_gate()
        with self.assertRaises(ValueError):self.validate()

    def test_all_five_seed42_runs_are_required_even_for_reference_and_no_adapter_is_allowed(self):
        self.assertIsNone(self.validate('student_base')['adapter_directory'])
        with self.assertRaises(ValueError):self.validate('teacher_reference',adapter=self.out/'training/M7/seed42')
        path=self.out/'training/M1/seed42/metadata.json';meta=read(path);meta['completed']=False;save(path,meta)
        with self.assertRaises(ValueError):self.validate('student_base')

    def test_roster_seed_protocol_and_identity_cannot_be_silently_changed(self):
        original=copy.deepcopy(self.gate)
        for modify in (lambda g:g['conditions'].remove('NLL'),
                       lambda g:g['conditions'].remove('M6'),
                       lambda g:g['conditions'].append('M4'),
                       lambda g:g['seeds'].__setitem__('M7',[7]),
                       lambda g:g['seeds'].__setitem__('M7',[42,7,17]),
                       lambda g:g['artifacts']['M3'].pop('42'),
                       lambda g:g.__setitem__('development_accepted',False)):
            self.gate=copy.deepcopy(original);modify(self.gate);self.write_gate()
            with self.assertRaises(ValueError):self.validate()
        self.gate=original;self.write_gate()
        path=self.out/'training/M7/seed42/metadata.json';meta=read(path);meta['seed']=17;save(path,meta)
        with self.assertRaises(ValueError):self.validate()

    def test_failed_guard_occurs_before_any_public_test_loader_call(self):
        import evaluate
        with patch.object(evaluate,'validate_request',side_effect=ValueError('Gate unavailable')),\
                patch.object(evaluate,'test_items') as loader:
            with self.assertRaises(ValueError):evaluate.main('M7',42,None)
            loader.assert_not_called()


if __name__=='__main__':unittest.main()
