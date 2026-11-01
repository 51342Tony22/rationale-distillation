"""Scientific stage boundaries; synthetic evidence never enables real training."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_full import ROOT, OUT, HERE, read, save, baseline
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from positions_v2 import propose
from positions_nll import propose_nll
from train_selector_stage import (validate_authority, verify_inputs, verify_engineering,
    validate_stage_bundle, development_observations)
from full_selector_materials import build_bundles
import test_material_builder as fixtures


class SelectorTrainingStageTests(unittest.TestCase):
    def test_current_authority_retains_later_groups_but_only_three_are_primary(self):
        original = read(OUT / 'transfer128_amendment_v1.json')
        validate_authority(original)
        for key, value in [('formal_training_seeds', [7, 17, 27]),
                           ('epochs_per_run', 1), ('formal_transfer_root_budget', 7473),
                           ('primary_conditions', ['M1', 'M3']), ('protocol', 'five_groups_seed42_v1')]:
            altered = copy.deepcopy(original); altered[key] = value
            with self.assertRaises(ValueError): validate_authority(altered)

    def test_hashed_development_evidence_rejects_late_code_or_review_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'review.json'; save(path, {'decision': 'synthetic'})
            evidence = {'inputs': {'review.json': baseline.sha(path)}}
            self.assertEqual(verify_inputs(evidence, root=root), [path])
            save(path, {'decision': 'changed after freeze'})
            with self.assertRaises(ValueError): verify_inputs(evidence, root=root)
            with self.assertRaises(ValueError): verify_inputs({'inputs': {}}, root=root)

    def test_both_actual_model_preflights_and_staged_entry_signature_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'output/tgrd_v1'
            name = str((HERE / 'train_selector_stage.py').relative_to(ROOT))
            code = root / name; code.parent.mkdir(parents=True); code.write_text('unit-test stub')
            for stage, checks in (
                    ('formal_training', ['cold_resume_exact_final_adapter', 'cold_resume_exact_optimizer',
                                         'cold_resume_exact_global_PRNG']),
                    ('adapter_loading', ['exact_saved_LoRA_tensors_restored', 'adapter_directory_API_validated'])):
                signed = out / f'signatures/{stage}_gpu_v1.json'
                save(signed, {'inputs': {name: baseline.sha(code)}})
                save(out / f'preflight/{stage}_gpu_v1.json', {
                    'status': 'passed', 'device': 'actual_student_GPU', 'public_test_used': False,
                    'signature_file_sha256': baseline.sha(signed), **{k: True for k in checks},
                    'synthetic_unit_test_only': True})
            self.assertTrue(verify_engineering(root=root, out=out))
            path = out / 'preflight/formal_training_gpu_v1.json'; original = read(path)
            for changes in ({'device': 'tiny_actual_MLX_CPU_and_state_machine_only'},
                            {'cold_resume_exact_optimizer': False}, {'public_test_used': True}):
                save(path, {**original, **changes})
                with self.assertRaises(ValueError): verify_engineering(root=root, out=out)
            save(path, original)
            code.write_text('stale entry after real-model check')
            with self.assertRaises(ValueError): verify_engineering(root=root, out=out)

    def test_zero_joint_coverage_is_recorded_while_NLL_still_scores_every_step(self):
        sid = 'gsm8k:train:0'; sha = 'synthetic teacher'
        case = {'source_id': sid, 'teacher_output_sha256': sha,
                'candidates': [{'step': {'step_index': i}} for i in (0, 1)]}
        joint = {'source_id': sid, 'teacher_output_sha256': sha, 'steps': [
            {'step_index': i, 'status': 'no_valid_controls'} for i in (0, 1)]}
        joint['proposal'] = propose(joint['steps'])
        nll = {'source_id': sid, 'teacher_output_sha256': sha,
               'steps': [{'step_index': i, 'mean_nll': v} for i, v in ((0, .2), (1, 2.))]}
        nll['proposal'] = propose_nll(nll['steps'])
        pools = {s: {sid: {'position_proposal': r['proposal'], 'teacher_output_sha256': sha}}
                 for s, r in (('joint', joint), ('nll', nll))}
        result = development_observations([case], [joint], [nll], pools)
        self.assertEqual(result['joint_rank0_available'], 0)
        self.assertEqual(result['nll_rank0_available'], 1)
        self.assertEqual(result['candidate_steps'], 2)
        incomplete = copy.deepcopy(nll); incomplete['steps'].pop(0)
        incomplete['proposal'] = propose_nll(incomplete['steps'])
        with self.assertRaises(ValueError): development_observations([case], [joint], [incomplete], pools)
        with self.assertRaises(ValueError): development_observations([case], [], [nll], pools)
        tampered = copy.deepcopy(pools); tampered['nll'][sid]['teacher_output_sha256'] = 'other output'
        with self.assertRaises(ValueError): development_observations([case], [joint], [nll], tampered)

    def test_current_three_materials_pass_without_G4_or_G5_and_old_protocol_is_rejected(self):
        f = fixtures.MaterialBuilderTests(); f.setUp(); sid = f.source['source_id']
        from transformers import AutoTokenizer
        from runtime_full import BASE
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
            local_files_only=True, trust_remote_code=False)
        bundles, _, _ = build_bundles({sid: f.source}, {sid: f.teacher}, {sid: f.review}, [f.case],
            {'joint': {sid: f.pool}, 'nll': {sid: f.nll_pool}}, tokenizer)
        manifest = {'schema': 'tgrd_selector_stage_materials_v1', 'protocol': PROTOCOL,
            'stage': 'selector_first', 'conditions': list(SELECTOR_CONDITIONS), 'formal_seeds': [42],
            'epochs': 4, 'candidate_root_count': 7473, 'common_root_ids': [sid],
            'public_test_used': False, 'G4_or_G5_materials_required': False,
            'materials': {g: {'synthetic_unit_test_only': True} for g in SELECTOR_CONDITIONS}}
        validate_stage_bundle(manifest, bundles, [sid])
        for change in ('preview', 'protocol', 'root', 'missing_group'):
            altered = copy.deepcopy(bundles)
            if change == 'preview': altered['M3']['development_preview_only'] = True
            elif change == 'protocol': altered['M3']['protocol'] = 'five_groups_seed42_v1'
            elif change == 'root': altered['NLL']['rows'][0]['source_id'] = 'gsm8k:train:1'
            else: altered.pop('NLL')
            with self.assertRaises(ValueError): validate_stage_bundle(manifest, altered, [sid])

    def test_train_guard_precedes_GPU_and_learner_creation(self):
        import train_selector_stage as entry
        with patch.object(entry, 'validate_gate', side_effect=ValueError('real evidence unavailable')), \
                patch.object(entry, 'gpu') as gpu, patch('learning.Learner') as learner:
            with self.assertRaises(ValueError): entry.train('M3', 42)
            gpu.assert_not_called(); learner.assert_not_called()
        with self.assertRaises(ValueError): entry.validate_gate('M7', 42)
        with self.assertRaises(ValueError): entry.validate_gate('NLL', 7)


if __name__ == '__main__':
    unittest.main()
