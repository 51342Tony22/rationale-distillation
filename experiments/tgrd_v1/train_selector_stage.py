"""Current-protocol G1-G3 training, independent of the later transfer trials.

freeze-gate reconstructs reviewed materials and actual development evidence.
It does not infer scientific success from CPU checks or require positive
transfer gains. No entry point in this module reads public test questions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import ROOT, HERE, OUT, BASE, read, freeze, save, baseline, signature, gpu, now
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from training_plan import validate_materials
from training_engine import run_training
from training_checkpoint import save_checkpoint, restore_checkpoint, latest_checkpoint
from trial_calibration import adapter_hash

GATE = OUT / 'gates' / (PROTOCOL + '_selector_training_v1.json')
DEVELOPMENT = OUT / 'preflight' / (PROTOCOL + '_selector_development_v1.json')


def verify_inputs(value, *, root=ROOT):
    if not value.get('inputs'):
        raise ValueError('A frozen evidence file needs concrete input hashes')
    paths = []
    for name, checksum in value['inputs'].items():
        path = Path(root) / name
        if baseline.sha(path) != checksum:
            raise ValueError('Frozen selector-stage input changed: ' + name)
        paths.append(path)
    return paths


def validate_authority(authority):
    if (authority.get('protocol') != PROTOCOL
            or authority.get('primary_conditions') != list(SELECTOR_CONDITIONS)
            or authority.get('execution_stages') != [list(SELECTOR_CONDITIONS), ['M7'], ['M6']]
            or authority.get('conditions') != ['M1', 'NLL', 'M3', 'M7', 'M6']
            or authority.get('formal_training_seeds') != [42]
            or authority.get('epochs_per_run') != 4
            or authority.get('official_train_candidates') != 7473
            or authority.get('official_test_count') != 1319
            or authority.get('formal_transfer_root_budget') != 128
            or authority.get('G4_outside_roster') != 'Exact G3 material reuse for all four epochs'):
        raise ValueError('Expected the authorized selector-first, transfer128 seed42 study')


def verify_engineering(*, root=ROOT, out=OUT):
    paths = []
    requirements = {
        'formal_training': ('cold_resume_exact_final_adapter', 'cold_resume_exact_optimizer',
                            'cold_resume_exact_global_PRNG'),
        'adapter_loading': ('exact_saved_LoRA_tensors_restored', 'adapter_directory_API_validated'),
    }
    for stage, checks in requirements.items():
        report_path = Path(out) / f'preflight/{stage}_gpu_v1.json'
        signed_path = Path(out) / f'signatures/{stage}_gpu_v1.json'
        report = read(report_path)
        if (report.get('status') != 'passed' or report.get('device') != 'actual_student_GPU'
                or report.get('public_test_used') is not False
                or any(report.get(k) is not True for k in checks)
                or report.get('signature_file_sha256') != baseline.sha(signed_path)):
            raise ValueError('Actual-model training and adapter-loading preflights required')
        signed = read(signed_path)
        code_name = str((HERE / 'train_selector_stage.py').relative_to(ROOT))
        if code_name not in signed.get('inputs', {}):
            raise ValueError('Engineering preflight must bind the current staged training entry')
        paths += [report_path, signed_path, *verify_inputs(signed, root=root)]
    return paths


def development_observations(cases, joint, nll, pools):
    """Check actual development coverage without demanding selector wins."""
    from positions_v2 import propose
    from positions_nll import propose_nll
    ids = [c['source_id'] for c in cases]
    if (not ids or len(ids) != len(set(ids))
            or any([r['source_id'] for r in rows] != ids for rows in (joint, nll))
            or any(set(pools[s]) != set(ids) for s in ('joint', 'nll'))):
        raise ValueError('Development diagnostics and both reviewed pools need complete common coverage')
    observations = {'root_count': len(ids), 'candidate_steps': 0,
                    'joint_rank0_available': 0, 'nll_rank0_available': 0,
                    'rank0_disagreements': 0, 'no_joint_control_roots': 0}
    for case, j, n in zip(cases, joint, nll):
        indices = [c['step']['step_index'] for c in case['candidates']]
        if (any(r['teacher_output_sha256'] != case['teacher_output_sha256'] for r in (j, n))
                or any([s['step_index'] for s in r['steps']] != indices for r in (j, n))
                or j['proposal'] != propose(j['steps']) or n['proposal'] != propose_nll(n['steps'])):
            raise ValueError('Development all-step diagnostics/proposals do not reproduce')
        for selector, row in (('joint', j), ('nll', n)):
            pool = pools[selector][case['source_id']]
            if (pool['position_proposal'] != row['proposal']
                    or pool['teacher_output_sha256'] != case['teacher_output_sha256']):
                raise ValueError('Reviewed development pool differs from its diagnostic')
        jp = j['proposal']['positions']; np = n['proposal']['positions']
        observations['candidate_steps'] += len(indices)
        observations['joint_rank0_available'] += bool(jp)
        observations['nll_rank0_available'] += bool(np)
        observations['rank0_disagreements'] += bool(jp and np and jp[0] != np[0])
        observations['no_joint_control_roots'] += not bool(jp)
    return observations


def freeze_development():
    """Actual 34-root model runs + full-text pool review, no transfer dependency."""
    from calibration_cohort import cohort_paths
    from explanation_pool import selector_paths, unique
    from auxiliary_explainer_recipe import (assemble_development, DEVELOPMENT as ADOPTED,
        RECIPE, RECIPE_ID)
    from full_explanations import student_tokenizer
    from runtime_full import checked_rows
    cohort = 'cal40_reviewed34_v2'; paths = cohort_paths(cohort)
    files = [Path(__file__), paths.cases, paths.reviews,
             HERE / 'positions_v2.py', HERE / 'positions_nll.py',
             HERE / 'explanation_pool.py', HERE / 'calibration_cohort.py']
    cases = read(paths.cases)
    if len(cases) != 34:
        raise ValueError('The declared development cohort must retain all 34 roots')
    adopted_pools, adopted_manifest = assemble_development(student_tokenizer())
    files += [RECIPE, ADOPTED / 'manifest.json', *verify_inputs(adopted_manifest)]
    pools = {}; diagnostics = {}
    for selector, stage in (
            ('joint', 'positions_v2_' + paths.signature_prefix),
            ('nll', 'positions_nll_' + paths.signature_prefix + '_v1')):
        signed_path = OUT / 'signatures' / (stage + '.json')
        files += [signed_path, *verify_inputs(read(signed_path))]
        # The explicit adopted recipe reconstructs all134 reviewed attempts;
        # old failed mathematical-teacher pools remain preserved and unused.
        selected = selector_paths(cohort, selector); folder = selected.pool_folder
        diagnostics[selector] = checked_rows(selected.diagnostics)
        pools[selector] = unique(adopted_pools[selector], 'source_id')
        files += [selected.diagnostics, folder / 'requests.json',
                  ADOPTED / f'reviewed_pools_{selector}.json']
    observations = development_observations(cases, diagnostics['joint'], diagnostics['nll'], pools)
    value = {'schema': 'tgrd_selector_stage_development_v1', 'protocol': PROTOCOL,
        'status': 'passed', 'purpose': 'selector_coverage_and_reviewed_material_integrity',
        'auxiliary_explainer_recipe_id': RECIPE_ID,
        'all_step_independent_NLL_checked': True, 'full_text_explanation_review_enforced': True,
        'positive_selector_or_transfer_gain_required': False,
        'method_effectiveness_established': False, 'public_test_used': False,
        'development_root_ids': [c['source_id'] for c in cases], 'observations': observations,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}}
    freeze(DEVELOPMENT, value)
    return value


def validate_stage_bundle(manifest, bundles, ids):
    if (manifest.get('schema') != 'tgrd_selector_stage_materials_v1'
            or manifest.get('protocol') != PROTOCOL or manifest.get('stage') != 'selector_first'
            or manifest.get('conditions') != list(SELECTOR_CONDITIONS)
            or manifest.get('formal_seeds') != [42] or manifest.get('epochs') != 4
            or manifest.get('candidate_root_count') != 7473
            or manifest.get('common_root_ids') != ids
            or manifest.get('public_test_used') is not False
            or manifest.get('G4_or_G5_materials_required') is not False
            or set(bundles) != set(SELECTOR_CONDITIONS)
            or set(manifest.get('materials', {})) != set(SELECTOR_CONDITIONS)):
        raise ValueError('A complete current-protocol G1-G3 stage is required')
    for group, bundle in bundles.items():
        if (bundle.get('protocol') != PROTOCOL or bundle.get('condition') != group
                or bundle.get('stage') != 'selector_first'
                or bundle.get('development_preview_only') is not False):
            raise ValueError('Legacy or development-preview materials cannot enter this stage')
        validate_materials(bundle, ids)


def freeze_gate():
    from full_selector import load_quality
    from full_selector_materials import prepare, FOLDER
    validate_authority(read(OUT / 'transfer128_amendment_v1.json'))
    engineering = verify_engineering()
    _, _, quality, ids = load_quality()
    sample_evidence = []
    from automatic_teacher_quality import SCHEMA as AUTO_SCHEMA
    if quality.get('schema') == AUTO_SCHEMA:
        from teacher_spotcheck_v2 import verify
        sample_evidence = verify()
    freeze_development()
    # Rebuild from exact teacher outputs, all full-quality decisions, signed
    # selector diagnostics and reviewed shared explanation requests.
    manifest = prepare()
    bundles = {g: read(FOLDER / f'{g}.json') for g in SELECTOR_CONDITIONS}
    validate_stage_bundle(manifest, bundles, ids)
    paths = [Path(__file__), FOLDER / 'materials_manifest.json', DEVELOPMENT,
             OUT / 'transfer128_amendment_v1.json', *engineering, *sample_evidence]
    paths += [ROOT / p for p in manifest['inputs']]
    paths += [FOLDER / f'{g}.json' for g in SELECTOR_CONDITIONS]
    freeze(GATE, {'schema': 'tgrd_selector_training_gate_v1', 'enabled': True,
        'protocol': PROTOCOL, 'primary_dataset': 'GSM8K', 'stage': 'selector_first',
        'conditions': list(SELECTOR_CONDITIONS), 'seeds': [42], 'epochs': 4,
        'candidate_root_count': 7473, 'common_root_ids': ids,
        'materials_manifest': str((FOLDER / 'materials_manifest.json').relative_to(ROOT)),
        'development_evidence': str(DEVELOPMENT.relative_to(ROOT)),
        'G4_G5_trials_or_positive_transfer_not_prerequisites': True,
        'public_evaluation_enabled': False, 'public_test_used': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}})
    return read(GATE)


def validate_gate(condition, seed):
    from full_selector import load_quality
    if condition not in SELECTOR_CONDITIONS or type(seed) is not int or seed != 42:
        raise ValueError('This stage only runs G1/G2/G3 at seed42')
    gate = read(GATE)
    if (gate.get('schema') != 'tgrd_selector_training_gate_v1' or gate.get('enabled') is not True
            or gate.get('protocol') != PROTOCOL or gate.get('primary_dataset') != 'GSM8K'
            or gate.get('stage') != 'selector_first' or gate.get('conditions') != list(SELECTOR_CONDITIONS)
            or gate.get('seeds') != [42] or gate.get('epochs') != 4
            or gate.get('candidate_root_count') != 7473 or gate.get('public_test_used') is not False
            or gate.get('public_evaluation_enabled') is not False):
        raise ValueError('Current-protocol staged training gate required')
    paths = verify_inputs(gate)
    validate_authority(read(OUT / 'transfer128_amendment_v1.json'))
    paths += verify_engineering()
    development = read(ROOT / gate['development_evidence'])
    if (development.get('schema') != 'tgrd_selector_stage_development_v1'
            or development.get('protocol') != PROTOCOL or development.get('status') != 'passed'
            or development.get('public_test_used') is not False
            or development.get('all_step_independent_NLL_checked') is not True
            or development.get('full_text_explanation_review_enforced') is not True
            or development.get('observations', {}).get('root_count') != 34):
        raise ValueError('Actual complete selector development evidence required')
    paths += verify_inputs(development)
    sources, _, _, ids = load_quality()
    if gate['common_root_ids'] != ids:
        raise ValueError('Training roots differ from the full common teacher-quality roster')
    manifest = read(ROOT / gate['materials_manifest']); paths += verify_inputs(manifest)
    bundles = {}
    for group, entry in manifest['materials'].items():
        path = ROOT / entry['path']
        if baseline.sha(path) != entry['sha256']:
            raise ValueError('Frozen staged training materials changed')
        bundles[group] = read(path); paths.append(path)
    validate_stage_bundle(manifest, bundles, ids)
    sources = {s['source_id']: s for s in sources}
    for bundle in bundles.values():
        for row in bundle['rows']:
            if row['question_sha256'] != sources[row['source_id']]['question_sha256']:
                raise ValueError('Training root/source question changed')
    return bundles[condition], list(dict.fromkeys([GATE, *paths]))


def train(condition, seed=42):
    bundle, paths = validate_gate(condition, seed)
    folder = OUT / 'training' / PROTOCOL / condition / f'seed{seed}'
    signature(f'{PROTOCOL}_training_{condition}_s{seed}', paths + [
        Path(__file__), HERE / 'training_plan.py', HERE / 'training_engine.py',
        HERE / 'training_checkpoint.py', HERE / 'learning.py', HERE / 'trial_calibration.py',
        HERE / 'runtime.py', HERE / 'runtime_full.py'],
        {'protocol': PROTOCOL, 'stage': 'selector_first', 'condition': condition, 'seed': seed,
         'primary_checkpoint': 'last_epoch4', 'public_test_used': False})
    with gpu():
        from learning import Learner, LORA, tokenize
        baseline.verify_student(BASE); learner = Learner(seed)
        freeze(OUT / 'training' / PROTOCOL / 'initializations' / f'seed{seed}.json', {
            'protocol': PROTOCOL, 'seed': seed, 'initial_adapter_sha256': adapter_hash(learner),
            'student_revision': BASE['student_revision'], 'LoRA': LORA, 'layers': 16})
        freeze(folder / 'adapter_config.json', {
            'fine_tune_type': 'lora', 'num_layers': 16, 'lora_parameters': LORA})
        metadata = run_training(learner, bundle, folder, seed=seed, tokenize=tokenize,
            weight_hash=adapter_hash, save_checkpoint=save_checkpoint,
            restore_checkpoint=restore_checkpoint, latest_checkpoint=latest_checkpoint)
        save(folder / 'completion_status.json', {'at_utc': now(), **metadata})
        print(metadata, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('freeze-gate')
    runner = sub.add_parser('train')
    runner.add_argument('--condition', choices=SELECTOR_CONDITIONS, required=True)
    runner.add_argument('--seed', type=int, choices=(42,), default=42)
    args = parser.parse_args()
    if args.command == 'freeze-gate':
        gate = freeze_gate()
        print({'accepted_roots': len(gate['common_root_ids']), 'conditions': gate['conditions'],
               'public_evaluation_enabled': False}, flush=True)
    else:
        train(args.condition, args.seed)
