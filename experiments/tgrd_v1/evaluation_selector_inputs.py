"""Current selector-stage artifacts and pretest freezes; never opens test data.

The old five-group evaluator and its actual GPU loading evidence stay intact.
This module accepts the real training engine's epoch4 artifacts, not a renamed
legacy gate. A pretest freeze needs actual development results and a separately
frozen transfer roster; neither is inferred from engineering checks.
"""
from __future__ import annotations

import json
from pathlib import Path

from runtime_full import ROOT, HERE, OUT, BASE, read, freeze, baseline, digest
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS, validate_roster
from training_plan import schedule, validate_materials
from training_checkpoint import verify_checkpoint
from learning import LORA

GATE_NAME = PROTOCOL + '_selector_public_evaluation_v1.json'
PRETEST_NAME = PROTOCOL + '_pretest_transfer_v1.json'
EVALUATION_ORDER = ['M1', 'NLL', 'M3', 'M7', 'M6']
TRANSFER_RULES = {
    'protocol': PROTOCOL, 'seed': 42, 'root_budget': 128,
    'max_nonempty_candidates': 4, 'include_equally_trained_e0': True,
    'select_probes': 6, 'audit_probes': 6,
    'trial_gradients': 'original_root_only', 'audit_can_reselect': False,
    'identical_initial_adapter_and_empty_optimizer': True,
    'selection': 'Vsel_accuracy_and_answer_NLL',
    'outside_roster': 'exact_G3_material_reuse',
    'inside_failure': 'ordinary_CoT', 'replace_failed_roots': False,
    'G5_epochs': ['ordinary_CoT', 'fixed_Vsel_1', 'ordinary_CoT', 'fixed_Vsel_2'],
    'G5_missing_full_probes': 'ordinary_CoT_all_epochs',
    'G5_trained_Vsel_is_generalization_evidence': False,
    'public_test_order': EVALUATION_ORDER,
    'public_test_checkpoint_selection': False, 'public_test_tuning': False,
}


def verify_inputs(value, *, root=ROOT):
    if not value.get('inputs'):
        raise ValueError('Concrete immutable evidence hashes required')
    paths = []
    for name, checksum in value['inputs'].items():
        path = (Path(root) / name).resolve()
        path.relative_to(Path(root).resolve())
        if baseline.sha(path) != checksum:
            raise ValueError('Evaluation evidence changed: ' + name)
        paths.append(path)
    return paths


def training_artifacts(bundles, ids, initialization, *, root=ROOT, out=OUT):
    """Validate every G1-G3 final checkpoint, including runs not being scored."""
    root = Path(root); out = Path(out)
    if (set(bundles) != set(SELECTOR_CONDITIONS) or not ids
            or initialization.get('protocol') != PROTOCOL
            or initialization.get('seed') != 42
            or initialization.get('student_revision') != BASE['student_revision']
            or initialization.get('LoRA') != LORA or initialization.get('layers') != 16
            or not initialization.get('initial_adapter_sha256')):
        raise ValueError('Complete staged roster and common actual initialization required')
    initial = initialization['initial_adapter_sha256']
    plan = schedule(len(ids), 42); updates = len(plan)
    artifacts = {}; paths = []
    for group in SELECTOR_CONDITIONS:
        bundle = bundles[group]
        if (bundle.get('condition') != group or bundle.get('protocol') != PROTOCOL
                or bundle.get('stage') != 'selector_first'
                or bundle.get('development_preview_only') is not False):
            raise ValueError('Current-protocol formal selector materials required')
        validate_materials(bundle, ids)
        folder = out / 'training' / PROTOCOL / group / 'seed42'
        inputs = read(folder / 'inputs.json'); meta = read(folder / 'metadata.json')
        epoch = read(folder / 'epochs/epoch4.json')
        checkpoint = folder / 'checkpoints' / f'update_{updates:08d}'
        if ((folder / epoch['checkpoint_directory']).resolve() != checkpoint.resolve()
                or read(folder / 'adapter_config.json') != {
                    'fine_tune_type': 'lora', 'num_layers': 16, 'lora_parameters': LORA}):
            raise ValueError('Final checkpoint or inference adapter configuration differs')
        commit = verify_checkpoint(checkpoint)
        weights = baseline.sha(folder / 'adapters.safetensors')
        input_sha = baseline.sha(folder / 'inputs.json')
        expected_inputs = {
            'schema': 'tgrd_training_run_v1', 'condition': group, 'seed': 42,
            'ordered_root_ids': ids, 'epochs': 4, 'updates': updates,
            'initial_adapter_sha256': initial,
            'materials_sha256': digest(json.dumps(bundle, sort_keys=True)),
            'schedule_sha256': digest(json.dumps(plan, sort_keys=True)),
            'primary_checkpoint': 'last_epoch4_not_selected_on_test',
            'equal_budget': 'one_root_exposure_per_epoch;five_microbatches_per_full_update',
        }
        expected_meta = {
            'completed': True, 'condition': group, 'seed': 42, 'epochs': 4,
            'accepted_root_count': len(ids), 'candidate_root_count': 7473,
            'updates': updates, 'root_exposures': 4 * len(ids),
            'initial_adapter_sha256': initial, 'input_file_sha256': input_sha,
            'adapter_sha256': weights, 'primary_checkpoint': 'last_epoch4',
            'public_test_used_to_select_checkpoint': False,
        }
        point = commit['progress']
        if (any(inputs.get(k) != v for k, v in expected_inputs.items())
                or any(meta.get(k) != v for k, v in expected_meta.items())
                or epoch != {'epoch': 4, 'update': updates,
                    'checkpoint_directory': str(checkpoint.relative_to(folder)),
                    'adapter_sha256': weights, 'primary': True}
                or baseline.sha(checkpoint / 'adapter.safetensors') != weights
                or commit['learner_updates'] != updates
                or any(point.get(k) != v for k, v in {
                    'epoch': 4, 'update': updates, 'epoch_end': True,
                    'input_file_sha256': input_sha, 'root_exposures': 4 * len(ids)}.items())):
            raise ValueError('Not the complete common-root seed42 final-epoch4 artifact: ' + group)
        names = ['inputs.json', 'metadata.json', 'epochs/epoch4.json',
                 'adapters.safetensors', 'adapter_config.json']
        names += [str((checkpoint / n).relative_to(folder)) for n in
                  ('commit.json', 'adapter.safetensors', 'state.safetensors', 'state_structure.json')]
        files = {name: baseline.sha(folder / name) for name in names}
        paths += [folder / name for name in names]
        artifacts[group] = {'adapter_directory': str(folder.relative_to(root)),
            'files': files, 'adapter_sha256': weights, 'accepted_root_count': len(ids),
            'primary_checkpoint': 'last_epoch4', 'seed': 42}
    return artifacts, paths


def collect_training():
    from train_selector_stage import validate_gate, GATE, validate_stage_bundle
    _, paths = validate_gate('M1', 42)  # reconstructs all three materials and real quality/development
    training = read(GATE); manifest = read(ROOT / training['materials_manifest'])
    bundles = {g: read(ROOT / manifest['materials'][g]['path']) for g in SELECTOR_CONDITIONS}
    ids = training['common_root_ids']; validate_stage_bundle(manifest, bundles, ids)
    initialization = OUT / 'training' / PROTOCOL / 'initializations/seed42.json'
    artifacts, files = training_artifacts(bundles, ids, read(initialization))
    paths += [initialization, *files]
    for group in SELECTOR_CONDITIONS:
        path = OUT / 'signatures' / f'{PROTOCOL}_training_{group}_s42.json'
        signed = read(path)
        if (signed.get('extra') != {'protocol': PROTOCOL, 'stage': 'selector_first',
                'condition': group, 'seed': 42, 'primary_checkpoint': 'last_epoch4',
                'public_test_used': False}
                or str(GATE.relative_to(ROOT)) not in signed.get('inputs', {})
                or manifest['materials'][group]['path'] not in signed.get('inputs', {})):
            raise ValueError('Missing current-protocol formal training signature')
        paths += [path, *verify_inputs(signed)]
    return training, artifacts, list(dict.fromkeys(paths))


def pretest_contents(roster, rules, summary, disclosures, ids):
    """Pure content checks; production also rebuilds the actual development grid."""
    validate_roster(roster, ids)
    if rules != TRANSFER_RULES:
        raise ValueError('Transfer, supplement, or test rules differ from the declared study')
    if (summary.get('protocol') != PROTOCOL or summary.get('cohort') != 'cal34_general7b_v2'
            or summary.get('recipe_id') != 'local_explanation_general7b_v2'
            or summary.get('actual_model_results_analyzed') is not True
            or summary.get('all24_seed_configuration_cells_complete') is not True
            or summary.get('public_test_used') is not False):
        raise ValueError('Complete adopted general-teacher development results required')
    configuration = summary.get('selected_configuration')
    status = summary.get('status')
    if status == 'no_positive_development_support':
        if configuration is not None:
            raise ValueError('Negative development cannot select or expand a transfer configuration')
        execution = 'unsupported_no_automatic_expansion_scope_unresolved'
    elif status == 'positive_development_signal_requires_confirmation':
        if (not isinstance(configuration, dict) or configuration.get('trial_seed') != 701
                or configuration.get('updates') not in (1, 3, 5)
                or configuration.get('threshold') not in (0, .01, .03, .1)):
            raise ValueError('A development-selected declared configuration is required')
        execution = 'frozen_configuration_requires_confirmation'
    else:
        raise ValueError('Missing complete development outcome')
    if (not summary.get('root_ids')
            or not set(summary['root_ids']) <= set(roster['excluded_development_ids'])
            or set(roster['source_ids']) & set(summary['root_ids'])):
        raise ValueError('Transfer roster must exclude the declared calibration roots')
    required_disclosures = ('historical_public_test_exposure', 'development_root_reuse',
        'automatic_teacher_quality_noise', 'single_seed_limitation', 'explanation_length_difference')
    if (disclosures.get('protocol') != PROTOCOL
            or any(not isinstance(disclosures.get(k), str) or not disclosures[k].strip()
                   for k in required_disclosures)):
        raise ValueError('Concrete historical exposure and comparison limitations must be recorded')
    return {'selected_configuration': configuration, 'transfer_execution': execution}


def pretest_evidence(roster_path, rules_path, disclosures_path, ids):
    from calibration_general_v2 import GeneralContext, COHORT
    from calibration_summary import main
    roster_path, rules_path, disclosures_path = (
        Path(p).resolve() for p in (roster_path, rules_path, disclosures_path))
    context = GeneralContext()
    # Rebuild from exact actual predictions/reset signatures; loose booleans or
    # the old Math7B summary cannot unlock this entry.
    main(COHORT, context=context)
    summary_path = context.paths.trials_folder / 'development_grid_summary_v1.json'
    summary = read(summary_path); roster = read(roster_path)
    result = pretest_contents(roster, read(rules_path), summary, read(disclosures_path), ids)
    paths = [Path(roster_path), Path(rules_path), Path(disclosures_path), summary_path]
    paths += verify_inputs(summary)
    if not isinstance(roster.get('sampling_inputs_sha256'), dict):
        raise ValueError('Frozen sampling must bind the actual training-side diagnostic inputs')
    required = {'output/tgrd_v1/full_selector_v1/positions_nll_v1.jsonl',
        'output/tgrd_v1/full_selector_v1/positions_joint_v1.jsonl',
        'output/tgrd_v1/full_selector_v1/cases_manifest.json',
        'output/tgrd_v1/gates/full_teacher_automatic_quality_v2.json',
        'data/tgrd_v1/train7473.json', 'data/tgrd_v1/cal40.json',
        'data/tgrd_v1/historical_exposure.json'}
    if not required <= set(roster['sampling_inputs_sha256']):
        raise ValueError('Transfer sampling lacks full training diagnostics and development exclusions')
    excluded = {r['source_id'] for r in read(ROOT / 'data/tgrd_v1/cal40.json')}
    excluded |= set(read(ROOT / 'data/tgrd_v1/historical_exposure.json')['source_ids'])
    excluded |= {'gsm8k:train:0', 'gsm8k:train:2'}
    if not excluded <= set(roster['excluded_development_ids']):
        raise ValueError('Sampling omitted known historical or engineering development roots')
    paths += verify_inputs({'inputs': roster['sampling_inputs_sha256']})
    from transfer128_sampling import verify_roster
    paths += verify_roster(roster, ids)
    return result, list(dict.fromkeys(paths))


def freeze_pretest(roster_path, rules_path, disclosures_path):
    from full_selector import load_quality
    _, _, _, ids = load_quality()
    destination = OUT / 'gates' / PRETEST_NAME
    folder = OUT / 'evaluation' / PROTOCOL
    if not destination.exists() and any(folder.glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot declare a pretest freeze after this protocol has public predictions')
    result, paths = pretest_evidence(roster_path, rules_path, disclosures_path, ids)
    paths += [Path(__file__)]
    value = {'schema': 'tgrd_transfer_pretest_v1', 'protocol': PROTOCOL,
        'frozen_before_current_public_test': True, 'public_test_used': False,
        'roster': str(Path(roster_path).resolve().relative_to(ROOT)),
        'rules': str(Path(rules_path).resolve().relative_to(ROOT)),
        'disclosures': str(Path(disclosures_path).resolve().relative_to(ROOT)),
        **result, 'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    freeze(destination, value)
    return value


def verify_pretest(ids):
    path = OUT / 'gates' / PRETEST_NAME; value = read(path)
    if (value.get('schema') != 'tgrd_transfer_pretest_v1' or value.get('protocol') != PROTOCOL
            or value.get('frozen_before_current_public_test') is not True
            or value.get('public_test_used') is not False):
        raise ValueError('Actual transfer protocol must be frozen before public evaluation')
    paths = verify_inputs(value)
    result, evidence = pretest_evidence(*(ROOT / value[k] for k in ('roster', 'rules', 'disclosures')), ids)
    if any(value.get(k) != v for k, v in result.items()):
        raise ValueError('Transfer configuration no longer matches actual development results')
    if not set(evidence) <= set(paths):
        raise ValueError('Pretest freeze omitted actual roster/development evidence')
    return [path, *paths]


def current_evidence():
    training, artifacts, paths = collect_training()
    paths += verify_pretest(training['common_root_ids'])
    return training, artifacts, list(dict.fromkeys(paths))


def freeze_gate():
    training, artifacts, paths = current_evidence()
    paths += [Path(__file__), HERE / 'evaluate_selector_stage.py']
    value = {'schema': 'tgrd_selector_public_gate_v1', 'enabled': True,
        'protocol': PROTOCOL, 'stage': 'selector_first', 'primary_dataset': 'GSM8K',
        'conditions': list(SELECTOR_CONDITIONS), 'seed': 42, 'epochs': 4,
        'test_count': 1319, 'max_new_tokens': 1024, 'greedy': True,
        'primary_checkpoint': 'last_epoch4', 'public_test_used_for_freeze': False,
        'all_five_groups_completed': False, 'artifacts': artifacts,
        'accepted_root_count': len(training['common_root_ids']),
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    destination = OUT / 'gates' / GATE_NAME
    if not destination.exists() and any((OUT / 'evaluation' / PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot backdate the public gate after predictions')
    freeze(destination, value)
    return value


def validate_request(condition, seed=42, adapter=None):
    if condition not in SELECTOR_CONDITIONS or type(seed) is not int or seed != 42:
        raise ValueError('This public entry only evaluates G1-G3 at seed42')
    gate_path = OUT / 'gates' / GATE_NAME; gate = read(gate_path)
    expected = {'schema': 'tgrd_selector_public_gate_v1', 'enabled': True,
        'protocol': PROTOCOL, 'stage': 'selector_first', 'primary_dataset': 'GSM8K',
        'conditions': list(SELECTOR_CONDITIONS), 'seed': 42, 'epochs': 4,
        'test_count': 1319, 'max_new_tokens': 1024, 'greedy': True,
        'primary_checkpoint': 'last_epoch4', 'public_test_used_for_freeze': False,
        'all_five_groups_completed': False}
    if any(gate.get(k) != v for k, v in expected.items()):
        raise ValueError('A current-protocol selector-stage public gate is required')
    paths = verify_inputs(gate)
    training, artifacts, evidence = current_evidence()
    if (gate.get('artifacts') != artifacts
            or gate.get('accepted_root_count') != len(training['common_root_ids'])
            or not set(evidence) <= set(paths)):
        raise ValueError('Public gate omits or differs from current training/pretest evidence')
    entry = artifacts[condition]; directory = ROOT / entry['adapter_directory']
    if adapter is not None and Path(adapter).resolve() != directory.resolve():
        raise ValueError('Adapter argument differs from the frozen final group/seed directory')
    return {'adapter_directory': directory, 'training_artifact': entry,
            'signature_paths': [gate_path, *paths]}
