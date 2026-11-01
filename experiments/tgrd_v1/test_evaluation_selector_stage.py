"""Current-stage evaluator contracts, with no public dataset or large model."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_full import read, save, digest, baseline, BASE
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from learning import LORA
from evaluation_selector_inputs import training_artifacts, pretest_contents, TRANSFER_RULES
from training_checkpoint import save_checkpoint, restore_checkpoint, latest_checkpoint
from training_engine import run_training
import test_training as fixtures


class CurrentTrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        device = mx.default_device(); mx.set_default_device(mx.cpu)
        self.addCleanup(lambda: mx.set_default_device(device))
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.out = self.root / 'output/tgrd_v1'
        self.bundles = {}; self.ids = ['gsm8k:train:0']
        # Produce the ACTUAL engine format using a tiny CPU learner, so this
        # test catches changes to engine artifacts (including primary naming).
        def weight_hash(learner):
            import numpy as np
            return digest(''.join(np.asarray(v).tobytes().hex() for v in learner.weights().values()))
        for group in SELECTOR_CONDITIONS:
            bundle = fixtures.ScheduleTests().fixture(group)
            bundle.update(protocol=PROTOCOL, stage='selector_first', development_preview_only=False)
            self.bundles[group] = bundle
            learner = fixtures.CheckpointMLXCPUTests().learner(42)
            self.initial = weight_hash(learner)
            folder = self.folder(group)
            save(folder / 'adapter_config.json', {
                'fine_tune_type': 'lora', 'num_layers': 16, 'lora_parameters': LORA})
            run_training(learner, bundle, folder, seed=42,
                tokenize=lambda *a: 1., weight_hash=weight_hash,
                save_checkpoint=save_checkpoint, restore_checkpoint=restore_checkpoint,
                latest_checkpoint=latest_checkpoint)
        self.initialization = {'protocol': PROTOCOL, 'seed': 42,
            'initial_adapter_sha256': self.initial, 'student_revision': BASE['student_revision'],
            'LoRA': LORA, 'layers': 16}

    def folder(self, group):
        return self.out / 'training' / PROTOCOL / group / 'seed42'

    def validate(self):
        return training_artifacts(self.bundles, self.ids, self.initialization,
            root=self.root, out=self.out)

    def test_accepts_real_engine_epoch4_without_waiting_for_G4_or_G5(self):
        artifacts, paths = self.validate()
        self.assertEqual(set(artifacts), set(SELECTOR_CONDITIONS))
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertEqual(read(self.folder('NLL') / 'inputs.json')['primary_checkpoint'],
                         'last_epoch4_not_selected_on_test')
        self.assertFalse((self.out / 'training' / PROTOCOL / 'M7').exists())

    def test_unrequested_run_must_also_be_complete_and_initialized_identically(self):
        path = self.folder('NLL') / 'metadata.json'; original = read(path)
        for key, value in [('completed', False), ('seed', 7), ('initial_adapter_sha256', 'other'),
                           ('public_test_used_to_select_checkpoint', True)]:
            changed = {**original, key: value}; save(path, changed)
            with self.assertRaises(ValueError): self.validate()
        save(path, original)
        self.initialization['initial_adapter_sha256'] = 'another-S0'
        with self.assertRaises(ValueError): self.validate()

    def test_training_text_schedule_and_root_provenance_cannot_change(self):
        path = self.folder('M3') / 'inputs.json'; original = read(path)
        for key, value in [('materials_sha256', 'other'), ('schedule_sha256', 'other'),
                           ('ordered_root_ids', ['gsm8k:train:1']), ('updates', 8),
                           ('primary_checkpoint', 'best_test_score')]:
            save(path, {**original, key: value})
            # Update linked metadata to test semantics beyond just its hash.
            meta_path = self.folder('M3') / 'metadata.json'; meta = read(meta_path)
            save(meta_path, {**meta, 'input_file_sha256': baseline.sha(path)})
            with self.assertRaises(ValueError): self.validate()
        save(path, original)

    def test_final_weights_optimizer_and_final_epoch_commit_are_verified(self):
        for name in ('adapters.safetensors', 'checkpoints/update_00000004/state.safetensors'):
            path = self.folder('M1') / name; original = path.read_bytes(); path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): self.validate()
            path.write_bytes(original)
        path = self.folder('M1') / 'epochs/epoch4.json'; epoch = read(path)
        save(path, {**epoch, 'checkpoint_directory': '../NLL/seed42/checkpoints/update_00000004'})
        with self.assertRaises(ValueError): self.validate()


class PretestAndResumeTests(unittest.TestCase):
    def fixtures(self):
        ids = [f'gsm8k:train:{i}' for i in range(200)]
        roster = {'protocol': PROTOCOL, 'seed': 42, 'root_budget': 128,
            'source_ids': ids[:128], 'excluded_development_ids': ids[128:],
            'frozen_before_transfer_results': True, 'public_test_used': False,
            'replacement_on_failure': False, 'sampling_rule': 'synthetic unit-test rule',
            'sampling_inputs_sha256': {'synthetic': 'not actual evidence'}}
        summary = {'protocol': PROTOCOL, 'cohort': 'cal34_general7b_v2',
            'recipe_id': 'local_explanation_general7b_v2', 'actual_model_results_analyzed': True,
            'all24_seed_configuration_cells_complete': True, 'public_test_used': False,
            'root_ids': ids[128:162], 'status': 'positive_development_signal_requires_confirmation',
            'selected_configuration': {'trial_seed': 701, 'updates': 3, 'threshold': .03}}
        disclosures = {'protocol': PROTOCOL, **{k: 'synthetic disclosure for guard test only' for k in
            ('historical_public_test_exposure', 'development_root_reuse',
             'automatic_teacher_quality_noise', 'single_seed_limitation', 'explanation_length_difference')}}
        return roster, copy.deepcopy(TRANSFER_RULES), summary, disclosures, ids

    def test_pretest_does_not_treat_negative_development_as_positive_or_expand(self):
        args = self.fixtures(); self.assertEqual(pretest_contents(*args)['selected_configuration']['updates'], 3)
        args[2].update(status='no_positive_development_support', selected_configuration=None)
        outcome = pretest_contents(*args)
        self.assertIsNone(outcome['selected_configuration'])
        self.assertIn('no_automatic_expansion', outcome['transfer_execution'])
        args[2]['selected_configuration'] = {'trial_seed': 701, 'updates': 3, 'threshold': .03}
        with self.assertRaises(ValueError): pretest_contents(*args)

    def test_old_pool_partial_grid_development_overlap_or_missing_disclosure_cannot_unlock(self):
        for modify in (
            lambda a: a[2].update(recipe_id='failed_math7b'),
            lambda a: a[2].update(all24_seed_configuration_cells_complete=False),
            lambda a: a[2].update(public_test_used=True),
            lambda a: a[0].update(source_ids=a[0]['source_ids'][1:]+['gsm8k:train:128']),
            lambda a: a[1].update(audit_can_reselect=True),
            lambda a: a[3].pop('historical_public_test_exposure')):
            args = self.fixtures(); modify(args)
            with self.assertRaises(ValueError): pretest_contents(*args)

    def test_current_guard_fails_before_public_loader_and_model(self):
        import evaluate_selector_stage as evaluator
        with patch.object(evaluator, 'validate_request', side_effect=ValueError('missing real freeze')), \
                patch.object(evaluator, 'test_items') as loader, patch.object(evaluator, 'gpu') as model:
            with self.assertRaises(ValueError): evaluator.main('M1')
            loader.assert_not_called(); model.assert_not_called()

    def prediction(self):
        import evaluate_selector_stage as evaluator
        item = {'source_id': 'synthetic:test:0', 'question': 'Synthetic 3 + 4?', 'gold': '7',
                'question_sha256': digest('Synthetic 3 + 4?')}
        output = '3 + 4 = 7.\nThe answer is 7.'
        v2, _ = evaluator.sensitivity.parse(output)
        row = {**item, 'output': output, 'output_sha256': digest(output), 'scoring_text': output,
            'protocol': PROTOCOL, 'condition': 'M1', 'seed': 42,
            'evaluation_signature_file_sha256': 'synthetic',
            'prompt_sha256': digest(BASE['student_prompt'].format(question=item['question'])),
            **baseline.score(output, item['gold']), 'v2_correct': v2 == baseline.number(item['gold'])}
        return item, row

    def test_resume_recomputes_scores_order_gold_model_and_prompt(self):
        from evaluate_selector_stage import validate_saved
        item, row = self.prediction()
        validate_saved([row], [item], 'M1', 42, 'synthetic')
        for key, value in [('correct', not row['correct']), ('source_id', 'synthetic:test:1'),
                           ('gold', '8'), ('condition', 'M3'), ('prompt_sha256', 'other'),
                           ('evaluation_signature_file_sha256', 'other'), ('scoring_text', '7')]:
            with self.assertRaises(ValueError):
                validate_saved([{**row, key: value}], [item], 'M1', 42, 'synthetic')
        with self.assertRaises(ValueError): validate_saved([row, row], [item], 'M1', 42, 'synthetic')


if __name__ == '__main__': unittest.main()
