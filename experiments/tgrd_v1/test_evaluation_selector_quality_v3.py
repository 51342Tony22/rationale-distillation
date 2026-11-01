"""v3 evaluator integration guards; tiny CPU engine only, no public dataset."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evaluation_selector_quality_v3 as m
import evaluate_selector_quality_v3 as evaluator
import train_selector_quality_v3 as staged
import test_training as fixtures
from runtime_full import read,save,digest,BASE,baseline
from training_engine import run_training
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from learning import LORA

class V3EvaluationTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        device=mx.default_device();mx.set_default_device(mx.cpu)
        self.addCleanup(lambda:mx.set_default_device(device))
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.out=self.root/'output/tgrd_v1'
        self.gate=self.out/'gates/actual_training_v3.json'
        self.bundles={};self.paths=[];self.ids=['gsm8k:train:0']
        def weight_hash(learner):
            import numpy as np
            return digest(''.join(np.asarray(v).tobytes().hex() for v in learner.weights().values()))
        self.materials={}
        for group in m.SELECTOR_CONDITIONS:
            bundle=fixtures.ScheduleTests().fixture(group)
            bundle.update(protocol=m.PROTOCOL,stage='selector_first',development_preview_only=False,
                          explanation_quality_version='v3')
            self.bundles[group]=bundle
            material=self.out/f'materials/{group}.json';save(material,bundle)
            self.materials[group]=dict(path=str(material.relative_to(self.root)),sha256=baseline.sha(material))
            learner=fixtures.CheckpointMLXCPUTests().learner(42);initial=weight_hash(learner)
            folder=self.folder(group)
            save(folder/'adapter_config.json',dict(fine_tune_type='lora',num_layers=16,lora_parameters=LORA))
            run_training(learner,bundle,folder,seed=42,tokenize=lambda *a:1.,weight_hash=weight_hash,
                save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,latest_checkpoint=latest_checkpoint)
        initialization=self.out/'training'/m.PROTOCOL/'initializations/seed42.json'
        save(initialization,dict(protocol=m.PROTOCOL,seed=42,initial_adapter_sha256=initial,
             student_revision=BASE['student_revision'],LoRA=LORA,layers=16))
        manifest=self.out/'materials/manifest.json'
        save(manifest,dict(schema='tgrd_selector_stage_materials_v1',protocol=m.PROTOCOL,stage='selector_first',
            explanation_quality_version='v3',conditions=list(m.SELECTOR_CONDITIONS),formal_seeds=[42],epochs=4,
            candidate_root_count=7473,common_root_ids=self.ids,public_test_used=False,
            G4_or_G5_materials_required=False,materials=self.materials))
        save(self.gate,dict(common_root_ids=self.ids,materials_manifest=str(manifest.relative_to(self.root))))
        code=self.root/'experiments/tgrd_v1/train_selector_quality_v3.py'
        code.parent.mkdir(parents=True);code.write_text('Synthetic code hash fixture only')
        for g in m.SELECTOR_CONDITIONS:
            material=self.root/self.materials[g]['path']
            save(self.signed(g),dict(extra=dict(protocol=m.PROTOCOL,stage='selector_first',condition=g,
                seed=42,primary_checkpoint='last_epoch4',public_test_used=False),
                inputs={str(p.relative_to(self.root)):baseline.sha(p) for p in (self.gate,code,material)}))
    def folder(self,g):return self.out/'training'/m.PROTOCOL/g/'seed42'
    def signed(self,g):return self.out/'signatures'/f'{m.PROTOCOL}_training_{g}_s42.json'
    def collect(self):
        with patch.object(m,'ROOT',self.root),patch.object(m,'OUT',self.out), \
             patch.object(staged,'GATE',self.gate), \
             patch.object(staged,'validate_gate',return_value=(self.bundles['M1'],[self.gate])), \
             patch.object(m,'quality_evidence',return_value=[]), \
             patch.object(m,'verify_inputs',side_effect=lambda v:m._test_verify(v)):
            return m.collect_training()
    def test_new_collect_accepts_all_three_actual_CPU_engine_epoch4_artifacts(self):
        # verify_inputs default root is fixed at definition time; use explicit
        # temporary root, retaining actual file/hash validation in this test.
        with patch.object(m,'_test_verify',lambda v:self.verify(v),create=True):
            training,artifacts,paths=self.collect()
        self.assertEqual(set(artifacts),set(m.SELECTOR_CONDITIONS))
        self.assertEqual(training['common_root_ids'],self.ids)
        self.assertTrue(all(p.is_file() for p in paths))
    def verify(self,v):
        from evaluation_selector_inputs import verify_inputs
        return verify_inputs(v,root=self.root)
    def test_legacy_training_entry_signature_cannot_be_relabelled(self):
        path=self.signed('M3');signed=read(path)
        signed['inputs'].pop('experiments/tgrd_v1/train_selector_quality_v3.py');save(path,signed)
        with patch.object(m,'_test_verify',lambda v:self.verify(v),create=True):
            with self.assertRaises(ValueError):self.collect()
    def test_material_version_and_unrequested_final_artifact_are_checked(self):
        path=self.out/'materials/manifest.json';manifest=read(path)
        save(path,{**manifest,'explanation_quality_version':'v1'})
        with patch.object(m,'_test_verify',lambda v:self.verify(v),create=True):
            with self.assertRaises(ValueError):self.collect()
        save(path,manifest)
        path=self.folder('NLL')/'metadata.json';save(path,{**read(path),'completed':False})
        with patch.object(m,'_test_verify',lambda v:self.verify(v),create=True):
            with self.assertRaises(ValueError):self.collect()
    def test_missing_v3_gate_stops_before_public_loader_or_model(self):
        with patch.object(m,'OUT',self.out),patch.object(evaluator,'test_items') as data, \
             patch.object(evaluator,'gpu') as model:
            with self.assertRaises(FileNotFoundError):evaluator.main('M1')
            data.assert_not_called();model.assert_not_called()
    def test_legacy_public_gate_is_rejected_before_evidence_and_public_loader(self):
        path=self.out/'gates'/m.GATE_NAME
        save(path,{**m.gate_fields(),'schema':'tgrd_selector_public_gate_v1'})
        with patch.object(m,'OUT',self.out),patch.object(m,'current_evidence') as evidence, \
             patch.object(evaluator,'test_items') as data:
            with self.assertRaises(ValueError):evaluator.main('M1')
            evidence.assert_not_called();data.assert_not_called()
    def test_v3_disclosure_must_bind_actual_noise_report(self):
        disclosures=self.root/'disclosures.json';save(disclosures,{'automatic_explanation_quality_noise':'Synthetic disclosure only'})
        with patch.object(m,'legacy_pretest_evidence',return_value=({},[])), \
             patch.object(m,'quality_evidence',return_value=[]):
            with self.assertRaises(ValueError):m.pretest_evidence('roster','rules',disclosures,self.ids)
    def test_incomplete_actual_grid_cannot_freeze_pretest_or_create_public_gate(self):
        with patch('full_selector.load_quality',return_value=([],[],{},self.ids)), \
             patch.object(m,'OUT',self.out), \
             patch.object(m,'legacy_pretest_evidence',side_effect=ValueError('actual grid incomplete')):
            with self.assertRaises(ValueError):m.freeze_pretest('roster','rules','disclosures')
        self.assertFalse((self.out/'gates'/m.PRETEST_NAME).exists())
        self.assertFalse((self.out/'gates'/m.GATE_NAME).exists())

if __name__=='__main__':unittest.main()
