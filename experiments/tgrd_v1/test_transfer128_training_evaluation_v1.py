"""Actual tiny MLX CPU artifacts; no production training or public dataset."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
import train_transfer128_v1 as training
import evaluation_transfer128_v1 as evaluation
import evaluate_transfer128_v1 as evaluator
import learning
import transfer128_production_v1 as production
from runtime_full import read,save,digest,baseline,BASE
from learning import LORA
from training_engine import run_training
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from trial_calibration import adapter_hash
import test_training as fixtures


class TransferTrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        device=mx.default_device();mx.set_default_device(mx.cpu)
        self.addCleanup(lambda:mx.set_default_device(device))
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.out=self.root/'output/tgrd_v1'
        self.ids=['gsm8k:train:0'];self.bundles={}
        self.factory=fixtures.CheckpointMLXCPUTests().learner
        for g in training.CONDITIONS:
            bundle=fixtures.ScheduleTests().fixture(g)
            bundle.update(protocol=training.PROTOCOL,stage='finite_transfer128',development_preview_only=False)
            if g=='M6':
                for epoch in (1,3):
                    m=bundle['rows'][0]['epoch_materials'][epoch]
                    m.update(kind='verified_Vsel_variant',variant_id=f'root:select:numeric:{epoch}',split='select',
                        question=f'A box has {epoch} pencils. How many pencils?')
                    m['question_sha256']=digest(m['question'])
            self.bundles[g]=bundle
        self.init=dict(protocol=training.PROTOCOL,seed=42,initial_adapter_sha256=adapter_hash(self.factory(42)),
            student_revision=BASE['student_revision'],LoRA=LORA,layers=16)
        save(self.out/'training'/training.PROTOCOL/'initializations/seed42.json',self.init)

    def folder(self,g):return self.out/'training'/training.PROTOCOL/g/'seed42'
    def run_engine(self,g,after_update=None):
        folder=self.folder(g)
        save(folder/'adapter_config.json',dict(fine_tune_type='lora',num_layers=16,lora_parameters=LORA))
        return run_training(self.factory(42),self.bundles[g],folder,seed=42,tokenize=lambda *a:1.,
            weight_hash=adapter_hash,save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,
            latest_checkpoint=latest_checkpoint,checkpoint_interval=1,after_update=after_update)
    def artifacts(self):
        return evaluation.training_artifacts(self.bundles,self.ids,self.init,root=self.root,out=self.out)
    def finish_both(self):
        for g in training.CONDITIONS:self.run_engine(g)

    def test_actual_both_CPU_epoch4_artifacts_share_formal_S0_schedule(self):
        self.finish_both();artifacts,paths=self.artifacts()
        self.assertEqual(set(artifacts),{'M7','M6'});self.assertTrue(all(p.is_file() for p in paths))
        inputs=[read(self.folder(g)/'inputs.json') for g in training.CONDITIONS]
        self.assertEqual(inputs[0]['schedule_sha256'],inputs[1]['schedule_sha256'])
        self.assertEqual(inputs[0]['initial_adapter_sha256'],inputs[1]['initial_adapter_sha256'])
        self.assertEqual(inputs[0]['updates'],4)

    def test_partial_unrequested_group_blocks_collection(self):
        self.finish_both();p=self.folder('M6')/'epochs/epoch4.json';p.unlink()
        with self.assertRaises(FileNotFoundError):self.artifacts()

    def test_trial701_S0_cannot_replace_formal42_S0(self):
        training.verify_initialization(self.factory(42),self.init)
        with self.assertRaises(ValueError):training.verify_initialization(self.factory(701),self.init)
        self.finish_both();initial={**self.init,'initial_adapter_sha256':adapter_hash(self.factory(701))}
        with self.assertRaises(ValueError):evaluation.training_artifacts(self.bundles,self.ids,initial,root=self.root,out=self.out)

    def test_epoch4_adapter_or_schedule_tampering_rejected(self):
        self.finish_both();p=self.folder('M7')/'inputs.json';v=read(p)
        save(p,{**v,'schedule_sha256':'altered'})
        with self.assertRaises(ValueError):self.artifacts()

    def test_material_stage_and_Vaudit_gradients_rejected(self):
        self.finish_both();self.bundles['M7']['stage']='selector_first'
        with self.assertRaises(ValueError):self.artifacts()
        self.bundles['M7']['stage']='finite_transfer128'
        self.bundles['M6']['rows'][0]['epoch_materials'][1]['split']='audit'
        with self.assertRaises(ValueError):self.artifacts()

    def test_new_trainer_calls_actual_CPU_engine_and_reuses_cached_completed_run(self):
        with patch.object(training,'OUT',self.out),patch.object(training,'validate_gate',return_value=(self.bundles['M6'],[])), \
             patch.object(training,'signature') as signed,patch.object(training,'gpu',return_value=nullcontext()), \
             patch.object(learning,'Learner',side_effect=self.factory),patch.object(learning,'tokenize',return_value=1.), \
             patch.object(baseline,'verify_student'):
            first=training.train('M6');second=training.train('M6')
        self.assertEqual(first,second);self.assertEqual(first['updates'],4)
        self.assertEqual(first['root_exposures'],4)
        self.assertEqual(signed.call_args.args[2]['stage'],'finite_transfer128')
        from runtime_full import checked_rows
        self.assertEqual(sum(r['event']=='optimizer_update' for r in checked_rows(self.folder('M6')/'events.jsonl')),4)

    def test_new_trainer_interruption_cold_load_matches_uninterrupted_adapter_optimizer_PRNG(self):
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        with patch.object(training,'OUT',self.out),patch.object(training,'validate_gate',return_value=(self.bundles['M7'],[])), \
             patch.object(training,'signature'),patch.object(training,'gpu',side_effect=lambda:nullcontext()), \
             patch.object(learning,'Learner',side_effect=self.factory),patch.object(learning,'tokenize',return_value=1.), \
             patch.object(baseline,'verify_student'):
            def stop(step):
                if step['update']==2:raise InterruptedError('intentional CPU checkpoint2')
            def interrupted(*a,**k):return run_training(*a,**k,checkpoint_interval=1,after_update=stop)
            with patch.object(training,'run_training',side_effect=interrupted):
                with self.assertRaises(InterruptedError):training.train('M7')
            with patch.object(training,'run_training',side_effect=lambda *a,**k:run_training(*a,**k,checkpoint_interval=1)):
                resumed=training.train('M7')
        actual=self.factory(42);restore_checkpoint(actual,latest_checkpoint(self.folder('M7')/'checkpoints'))
        weights=adapter_hash(actual);optimizer={k:np.asarray(v).copy() for k,v in tree_flatten(actual.optimizer.state)}
        prng=[np.asarray(v).copy() for v in mx.random.state]
        reference=self.root/'reference'
        run_training(self.factory(42),self.bundles['M7'],reference,seed=42,tokenize=lambda *a:1.,weight_hash=adapter_hash,
            save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,latest_checkpoint=latest_checkpoint)
        actual=self.factory(42);restore_checkpoint(actual,latest_checkpoint(reference/'checkpoints'))
        self.assertEqual(weights,adapter_hash(actual));self.assertEqual(resumed['updates'],4)
        self.assertTrue(all(np.array_equal(np.asarray(v),optimizer[k]) for k,v in tree_flatten(actual.optimizer.state)))
        self.assertTrue(all(np.array_equal(np.asarray(v),p) for v,p in zip(mx.random.state,prng)))

    def test_negative_development_queue_action_does_not_make_materials_or_train(self):
        with patch.object(production,'load_configuration',return_value={'scientific_trial_enabled':False}), \
             patch.object(production,'run') as unsupported,patch.object(training,'freeze_gate') as gate, \
             patch.object(training,'train') as train:
            value=training.advance('M7')
        unsupported.assert_called_once();gate.assert_not_called();train.assert_not_called()
        self.assertFalse(value['fixed128_trials_complete']);self.assertFalse(value['formal_training_executed'])

    def test_gate_rejects_legacy_schema_and_omitted_actual_evidence(self):
        gate=self.out/'gate.json';file=self.root/'concrete.json';save(file,{'proof':'fixture'})
        fields=training.gate_fields(self.ids)
        value={**fields,'selector_final_artifacts':{},'inputs':{'concrete.json':baseline.sha(file)}};save(gate,value)
        with patch.object(training,'GATE',gate),patch.object(training,'evidence',return_value=(self.bundles,self.ids,{},[file])), \
             patch.object(training,'verify_inputs',side_effect=lambda v:evaluation.verify_inputs(v,root=self.root)):
            training.validate_gate('M7')
            save(gate,{**value,'schema':'tgrd_selector_training_gate_v1'})
            with self.assertRaises(ValueError):training.validate_gate('M7')
            save(gate,value)
            with patch.object(training,'evidence',return_value=(self.bundles,self.ids,{},[file,self.root/'missing'])):
                with self.assertRaises(ValueError):training.validate_gate('M7')

    def test_missing_training_and_public_gates_block_before_model_and_public_loader(self):
        with patch.object(training,'GATE',self.out/'missing.json'),patch.object(training,'gpu') as gpu:
            with self.assertRaises(FileNotFoundError):training.train('M7')
            gpu.assert_not_called()
        with patch.object(evaluation,'OUT',self.out),patch.object(evaluator,'test_items') as data,patch.object(evaluator,'gpu') as gpu:
            with self.assertRaises(FileNotFoundError):evaluator.main('M7')
            data.assert_not_called();gpu.assert_not_called()

    def test_legacy_public_gate_rejected_before_actual_evidence_and_test(self):
        save(self.out/'gates'/evaluation.GATE_NAME,{**evaluation.gate_fields(),'schema':'tgrd_selector_public_gate_quality_v3'})
        with patch.object(evaluation,'OUT',self.out),patch.object(evaluation,'current_evidence') as current, \
             patch.object(evaluator,'test_items') as data:
            with self.assertRaises(ValueError):evaluator.main('M7')
            current.assert_not_called();data.assert_not_called()

    def test_missing_actual_grid_prevents_transfer_gate_without_creating_it(self):
        import evaluation_selector_quality_v3 as selector
        with patch.object(evaluation,'OUT',self.out),patch.object(selector,'current_evidence',side_effect=ValueError('actual grid incomplete')):
            with self.assertRaises(ValueError):evaluation.freeze_gate()
        self.assertFalse((self.out/'gates'/evaluation.GATE_NAME).exists())

    def test142_population_cannot_be_claimed_covered_by_original_fixed_sample(self):
        export=dict(complete=True,requests=412,new_outputs=142,exact_output_reuses=270,
            automatically_eligible=200,excluded=212,new_automatically_eligible=60,
            all_text_read=False,semantic_correctness_verified=False,new142_covered_by_original_descriptive_sample=False)
        disclosures=dict(finite128_confirmation_scope='Synthetic disclosure only',new142_explanation_population='Separate unsampled outputs',
            new142_covered_by_original_descriptive_sample=False,transfer128_explanation_export_sha256='frozen',
            transfer128_explanation_counts={k:export[k] for k in ('requests','new_outputs','exact_output_reuses','automatically_eligible','excluded','new_automatically_eligible')})
        evaluation.extension_disclosures(disclosures,export,'frozen')
        with self.assertRaises(ValueError):evaluation.extension_disclosures({**disclosures,'new142_covered_by_original_descriptive_sample':True},export,'frozen')
        with self.assertRaises(ValueError):evaluation.extension_disclosures(disclosures,{**export,'semantic_correctness_verified':True},'frozen')

    def test_new_actual_GPU_route_receipt_is_required_and_not_created_by_CPU_checks(self):
        with patch.object(training,'OUT',self.out):
            with self.assertRaises(FileNotFoundError):training.verify_transfer_engineering()
        self.assertFalse((self.out/'preflight/transfer128_training_gpu_v1.json').exists())

    def test_new_engineering_material_routes_are_synthetic_and_obey_Vsel_only_epoch_switch(self):
        from preflight_transfer128_training_v1 import fixtures as engineering_fixtures
        from training_plan import validate_materials
        source=fixtures.ScheduleTests().fixture('M1')
        bundles=engineering_fixtures(source)
        for bundle in bundles.values():validate_materials(bundle,self.ids)
        materials=bundles['M6']['rows'][0]['epoch_materials']
        self.assertEqual([m['kind'] for m in materials],['ordinary_CoT','verified_Vsel_variant','ordinary_CoT','verified_Vsel_variant'])
        self.assertNotEqual(materials[1]['question'],materials[3]['question'])
        self.assertEqual(bundles['M7']['purpose'],'synthetic_engineering_not_qualified_scientific_materials')

    def test_new_all_five_gate_cannot_first_freeze_after_predictions(self):
        folder=self.out/'evaluation'/training.PROTOCOL/'M1/seed42'
        save(folder/'predictions.jsonl',{'synthetic':'presence guard only'})
        with patch.object(evaluation,'OUT',self.out),patch.object(evaluation,'current_evidence',return_value=({'common_root_ids':self.ids},{},[])):
            with self.assertRaises(ValueError):evaluation.freeze_gate()
        self.assertFalse((self.out/'gates'/evaluation.GATE_NAME).exists())

    def test_public_order_requires_preceding_complete_evaluations(self):
        with patch.object(evaluation,'OUT',self.out):
            evaluation.validate_order('M1',{})
            with self.assertRaises(FileNotFoundError):evaluation.validate_order('NLL',{})


if __name__=='__main__':unittest.main()
