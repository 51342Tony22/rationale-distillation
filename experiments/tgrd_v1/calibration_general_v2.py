"""Versioned cal34 trials using the adopted, actually reviewed general teacher.

This namespace cannot resume the old Math7B e0-only grid. Preparation performs
tokenization and provenance checks on CPU. Running still needs a measured cost
plan and the existing actual-model branch/reset engineering evidence.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, DATA, BASE, read, freeze, save,
    checked_rows, signature, baseline, gpu, now)
from budgeted_materials import PROTOCOL
from calibration_cohort import cohort_paths
from auxiliary_explainer_recipe import (RECIPE_ID, RECIPE, DEVELOPMENT,
    assemble_development, load_recipe, verify_inputs)
from calibration_summary import RULE, SEEDS, UPDATES
from trial_engine import THRESHOLDS, run_root
from trial_selection import validate_probes
from trial_calibration import adapter_hash, restore_adapter, evaluate_probes
from material_builder import unique, validate_pool
from variants import build

COHORT = 'cal34_general7b_v2'
FOLDER = OUT / 'calibration' / COHORT
COST_PLAN = FOLDER / 'cost_plan_frozen_v2.json'
CPU_REPORT = OUT / 'preflight/calibration_general_v2_cpu.json'
MEASUREMENT = FOLDER / 'cost_measurement_v2'


def verify_hashes(value):
    if not value.get('inputs'):
        raise ValueError('Concrete signed calibration inputs are required')
    for name, checksum in value['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('General calibration evidence changed: ' + name)


def validate_pools(pools, cases, sources, teachers, reviews, bundles):
    """Retain all34 predeclared roots, including empty/rejected candidates."""
    ids = [c['source_id'] for c in cases]
    if (len(ids) != 34 or len(set(ids)) != 34
            or [p['source_id'] for p in pools] != ids or set(bundles) != set(ids)):
        raise ValueError('General calibration must retain the complete ordered34 cohort')
    for pool, case in zip(pools, cases):
        sid = pool['source_id']; source = sources[sid]; teacher = teachers[sid]
        review = reviews[sid]
        if (pool.get('selector') != 'joint' or pool.get('recipe_id') != RECIPE_ID
                or any(c.get('recipe_id') != RECIPE_ID for c in pool['candidates'][1:])
                or review.get('accepted') is not True or review.get('all_text_read') is not True
                or review.get('question_sha256') != source['question_sha256']
                or review.get('output_sha256') != teacher['output_sha256']):
            raise ValueError('Only the adopted joint pool and actual root reviews may enter trials')
        validate_pool(source, teacher, pool,
            {c['step']['step_index']: c['step'] for c in case['candidates']})
        validate_probes(bundles[sid], source, build)
    return ids


class GeneralContext:
    def __init__(self):
        original = cohort_paths('cal40_reviewed34_v2')
        self.paths = replace(original, name=COHORT,
            pool_folder=FOLDER / 'pool', trials_folder=FOLDER / 'trials',
            signature_prefix=COHORT)

    def prepare(self):
        load_recipe()
        from full_explanations import student_tokenizer
        pools, manifest = assemble_development(student_tokenizer())
        verify_inputs(manifest)
        cases = read(self.paths.cases)
        sources = unique(read(DATA / 'cal40.json'), 'source_id')
        teachers = unique(checked_rows(OUT / 'materials/teacher_targets_cal40.jsonl'), 'source_id')
        reviews = unique(read(self.paths.reviews), 'source_id')
        bundles = unique(read(self.paths.variants), 'source_id')
        ids = validate_pools(pools['joint'], cases, sources, teachers, reviews, bundles)
        # An exact versioned copy, never an overwrite or promotion of old pools.
        pool_file = self.paths.pool_folder / 'reviewed_pools.json'
        freeze(pool_file, pools['joint'])
        branch_count = sum(len(p['candidates']) for p in pools['joint'])
        nonempty = sum(len(p['candidates']) > 1 for p in pools['joint'])
        paths = [Path(__file__), RECIPE, DEVELOPMENT / 'manifest.json', pool_file,
            self.paths.cases, self.paths.variants, self.paths.reviews, DATA / 'cal40.json',
            OUT / 'materials/teacher_targets_cal40.jsonl', HERE / 'calibration_summary.py',
            HERE / 'trial_engine.py', HERE / 'trial_selection.py', HERE / 'trial_calibration.py']
        paths += [ROOT / name for name in manifest['inputs']]
        value = {'schema': 'tgrd_general_calibration_materials_v2',
            **self.report_identity(), 'root_ids': ids, 'roots': 34,
            'branches_per_grid_job': branch_count, 'roots_with_nonempty_candidate': nonempty,
            'generation_upper_bound_per_grid_job': 6 * len(ids) + 12 * branch_count,
            'grid_job_count': 6, 'seeds': list(SEEDS), 'updates': list(UPDATES),
            'thresholds': list(THRESHOLDS),
            'upper_bound_note': 'Untrained6Vsel per root plus6Vsel and at most6Vaudit per branch; shared choices may reduce audits',
            'old_failed_math_teacher_pool_used': False, 'NLL_only_candidates_enter_joint': False,
            'GPU_loaded': False, 'actual_grid_results_analyzed': False,
            'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
        freeze(FOLDER / 'materials_manifest.json', value)
        freeze_summary_rule(self.paths)
        return value

    def report_identity(self):
        return {'protocol': PROTOCOL, 'cohort': COHORT, 'recipe_id': RECIPE_ID,
            'development_only': True, 'Vaudit_role': 'hyperparameter_development_not_independent_confirmation',
            'formal_training_enabled': False, 'public_test_used': False}

    def evidence_paths(self):
        path = FOLDER / 'materials_manifest.json'; manifest = read(path)
        verify_hashes(manifest)
        return [path, Path(__file__), *[ROOT / name for name in manifest['inputs']]]

    def verify_job(self, status, signed, seed, updates):
        identity = self.report_identity()
        if (any(status.get(k) != v for k, v in identity.items())
                or any(signed.get('extra', {}).get(k) != v for k, v in identity.items())
                or signed['extra'].get('seed') != seed
                or signed['extra'].get('updates') != updates
                or signed['extra'].get('thresholds') != list(THRESHOLDS)
                or signed['extra'].get('variant_labels_in_gradients') is not False):
            raise ValueError('Old or mislabelled trial results cannot become the adopted general grid')
        required = {str(p.relative_to(ROOT)) for p in (
            Path(__file__), FOLDER / 'materials_manifest.json', COST_PLAN, CPU_REPORT,
            self.paths.pool_folder / 'reviewed_pools.json')}
        if not required <= set(signed.get('inputs', {})):
            raise ValueError('General-grid signature lacks recipe/material/cost/CPU provenance')


def freeze_summary_rule(paths):
    plan = paths.trials_folder / 'summary_rule_v1.json'
    if not plan.exists() and any(paths.trials_folder.glob('seed*/gsm8k_train_*/inputs.json')):
        raise ValueError('Cannot declare the rule after general-grid trials')
    # Same selection/tie/negative rules as predeclared before the original grid.
    freeze(plan, {**RULE, 'cohort': COHORT, 'expected_roots': 34,
        'predeclared_before_any_actual_grid_result': True})


def validate_cost_plan(plan, manifest, material_sha):
    expected = {'schema': 'tgrd_transfer_cost_plan_v2', 'protocol': PROTOCOL,
        'cohort': COHORT, 'recipe_id': RECIPE_ID, 'materials_manifest_sha256': material_sha,
        'grid_seeds': list(SEEDS), 'grid_updates': list(UPDATES),
        'thresholds': list(THRESHOLDS), 'root_ids': manifest['root_ids'],
        'frozen_before_any_actual_grid_result': True, 'public_test_used': False}
    if (any(plan.get(k) != v for k, v in expected.items())
            or len(manifest['root_ids']) != 34 or len(set(manifest['root_ids'])) != 34):
        raise ValueError('Measured cost freeze must bind this exact general34 grid')
    measurement = plan.get('actual_cost_measurement', {})
    if (measurement.get('actual_student_GPU') is not True
            or measurement.get('generation_limit') != 1024
            or type(measurement.get('generations')) is not int or measurement['generations'] != 6
            or type(measurement.get('elapsed_seconds')) not in (int, float)
            or not 0 < measurement['elapsed_seconds'] < float('inf')
            or not plan.get('inputs')):
        raise ValueError('Cannot invent a measured GPU cost plan from CPU preparation')


def verify_driver():
    gate = OUT / 'preflight/trial_driver_gpu_v1.json'
    signed = OUT / 'signatures/trial_driver_gpu_v1.json'
    report = read(gate)
    if (read(OUT / 'preflight/learning.json').get('status') != 'passed'
            or report.get('status') != 'passed'
            or report.get('signature_file_sha256') != baseline.sha(signed)):
        raise ValueError('Actual-model branch/reset and driver preflights required')
    verify_hashes(read(signed))
    return [gate, signed, *[ROOT / name for name in read(signed)['inputs']]]


def prepare_cost_measurement(context, manifest):
    plan_path = MEASUREMENT / 'plan.json'
    paths = context.paths
    pools = read(paths.pool_folder / 'reviewed_pools.json')
    # Fixed first declared development root; never pick a fast/successful one.
    sid = manifest['root_ids'][0]
    sources = unique(read(DATA / 'cal40.json'), 'source_id')
    bundles = unique(read(paths.variants), 'source_id')
    probes = validate_probes(bundles[sid], sources[sid], build)['select']
    value = {'schema': 'tgrd_general_grid_cost_measurement_plan_v2',
        **context.report_identity(), 'source_id': sid, 'seed': 701, 'updates': 1,
        'generation_limit': 1024, 'probes': probes,
        'completion_sha256': pools[0]['candidates'][0]['completion_sha256'],
        'timings_only_not_configuration_selection': True,
        'grid_predictions_or_adapters_reused': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in (
            Path(__file__), FOLDER / 'materials_manifest.json', paths.variants,
            paths.pool_folder / 'reviewed_pools.json')}}
    if not plan_path.exists() and (MEASUREMENT / 'predictions.jsonl').exists():
        raise ValueError('Cannot backdate a timing plan after measurements')
    freeze(plan_path, value)
    return value


def measure_cost():
    """Six bounded actual generations; metrics cannot choose any grid setting."""
    context = GeneralContext(); manifest = context.prepare()
    plan = prepare_cost_measurement(context, manifest)
    cpu = read(CPU_REPORT)
    if cpu.get('status') != 'passed':
        raise ValueError('CPU validation required before the timing-only GPU pilot')
    verify_hashes(cpu); evidence = verify_driver()
    report_path = MEASUREMENT / 'measurement.json'
    if report_path.exists():
        report = read(report_path); verify_hashes(report)
        return report
    signature('cal34_general7b_v2_cost_measurement', [*context.evidence_paths(),
        MEASUREMENT / 'plan.json', CPU_REPORT, *evidence],
        {'purpose': 'timing_only_not_grid_selection', 'generation_limit': 1024})
    sid = plan['source_id']
    source = unique(read(DATA / 'cal40.json'), 'source_id')[sid]
    pool = read(context.paths.pool_folder / 'reviewed_pools.json')[0]
    update_file = MEASUREMENT / 'training_timing.json'
    with gpu():
        from learning import Learner
        baseline.verify_student(BASE); learner = Learner(701)
        initial = adapter_hash(learner)
        begin = time.perf_counter()
        learner.trial(source['question'], pool['candidates'][0]['completion'], updates=1)
        after = adapter_hash(learner); seconds = time.perf_counter() - begin
        if learner.updates != 1 or initial == after:
            raise ValueError('Timing update did not train the declared ordinary root')
        if not update_file.exists():
            freeze(update_file, {'source_id': sid, 'updates': 1,
                'elapsed_seconds': seconds, 'initial_adapter_sha256': initial,
                'after_adapter_sha256': after, 'scientific_trial_result': False})
        else:
            timing = read(update_file)
            if timing['initial_adapter_sha256'] != initial or timing['after_adapter_sha256'] != after:
                raise ValueError('Resumed timing pilot changed its measured adapter')
        # Same reset/update is replayed on interruption; completed predictions
        # resume only after the exact measured adapter identity was restored.
        evaluate_probes(learner, plan['probes'], MEASUREMENT / 'predictions.jsonl', generation_limit=1024)
    predictions = checked_rows(MEASUREMENT / 'predictions.jsonl')
    if [r['id'] for r in predictions] != [r['id'] for r in plan['probes']]:
        raise ValueError('Timing needs the complete fixed six generations')
    durations = [r['seconds'] for r in predictions]
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError('Invalid actual timing measurements')
    inputs = [MEASUREMENT / 'plan.json', MEASUREMENT / 'predictions.jsonl', update_file,
        OUT / 'signatures/cal34_general7b_v2_cost_measurement.json']
    inputs += [ROOT / name for name in read(inputs[-1])['inputs']]
    report = {'schema': 'tgrd_general_grid_actual_cost_measurement_v2',
        **context.report_identity(), 'actual_student_GPU': True,
        'generation_limit': 1024, 'generations': len(predictions),
        'elapsed_seconds': sum(durations),
        'generation_tokens': sum(r['generation_tokens'] for r in predictions),
        'training_update_seconds': read(update_file)['elapsed_seconds'],
        'actual_grid_result_or_configuration_selected': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in inputs}}
    freeze(report_path, report)
    return report


def freeze_cost_plan():
    context = GeneralContext(); manifest = context.prepare()
    if any(context.paths.trials_folder.glob('seed*/gsm8k_train_*/inputs.json')):
        if not COST_PLAN.exists():
            raise ValueError('Cannot backdate a cost plan after actual grid results')
    measured_path = MEASUREMENT / 'measurement.json'; measured = read(measured_path)
    verify_hashes(measured)
    if (measured.get('schema') != 'tgrd_general_grid_actual_cost_measurement_v2'
            or measured.get('recipe_id') != RECIPE_ID
            or measured.get('actual_grid_result_or_configuration_selected') is not False):
        raise ValueError('Actual timing-only measurement required')
    # Freeze only a budget, not a scientific threshold or a successful setting.
    upper_generations = manifest['grid_job_count'] * manifest['generation_upper_bound_per_grid_job']
    updates = manifest['branches_per_grid_job'] * len(SEEDS) * sum(UPDATES)
    paths = [measured_path, FOLDER / 'materials_manifest.json', MEASUREMENT / 'plan.json']
    paths += [ROOT / name for name in measured['inputs']]
    value = {'schema': 'tgrd_transfer_cost_plan_v2', **context.report_identity(),
        'materials_manifest_sha256': baseline.sha(FOLDER / 'materials_manifest.json'),
        'grid_seeds': list(SEEDS), 'grid_updates': list(UPDATES),
        'thresholds': list(THRESHOLDS), 'root_ids': manifest['root_ids'],
        'frozen_before_any_actual_grid_result': True,
        'actual_cost_measurement': {k: measured[k] for k in
            ('actual_student_GPU', 'generation_limit', 'generations', 'elapsed_seconds')},
        'estimated_grid_generations_upper_bound': upper_generations,
        'estimated_grid_updates': updates,
        'estimated_seconds_from_six_generation_mean':
            upper_generations * measured['elapsed_seconds'] / measured['generations']
            + updates * measured['training_update_seconds'],
        'estimate_is_not_deadline_or_guarantee': True,
        'estimate_excludes': ['probe_answer_NLL_scoring', 'branch_reset', 'model_load',
            'adapter_checkpoint_IO', 'thermal_or_content_variability'],
        'settings_selected_using_timing_accuracy_or_NLL': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    validate_cost_plan(value, manifest, baseline.sha(FOLDER / 'materials_manifest.json'))
    freeze(COST_PLAN, value)
    return value


def run(updates, seed):
    if type(updates) is not int or updates not in UPDATES or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the original updates1/3/5 and development seeds701/1701 are permitted')
    context = GeneralContext(); manifest = context.prepare(); paths = context.paths
    material = FOLDER / 'materials_manifest.json'
    plan = read(COST_PLAN); validate_cost_plan(plan, manifest, baseline.sha(material)); verify_hashes(plan)
    cpu = read(CPU_REPORT)
    if cpu.get('status') != 'passed' or cpu.get('public_test_used') is not False:
        raise ValueError('General-grid CPU integration checks required')
    verify_hashes(cpu)
    evidence = [*context.evidence_paths(), *verify_driver(), COST_PLAN, CPU_REPORT,
        paths.trials_folder / 'summary_rule_v1.json', HERE / 'learning.py',
        HERE / 'variants.py', HERE / 'runtime.py', HERE / 'runtime_full.py',
        ROOT / 'experiments/math_distill_v1/config.json']
    pools = read(paths.pool_folder / 'reviewed_pools.json')
    sources = unique(read(DATA / 'cal40.json'), 'source_id')
    bundles = unique(read(paths.variants), 'source_id')
    signature(f'trial_{paths.signature_prefix}_s{seed}_u{updates}_v1', evidence,
        {**context.report_identity(), 'updates': updates, 'seed': seed,
         'thresholds': list(THRESHOLDS), 'variant_labels_in_gradients': False})
    folder = paths.trials_folder / f'seed{seed}_updates{updates}'
    with gpu():
        from learning import Learner
        baseline.verify_student(BASE); learner = Learner(seed)
        initial_hash = adapter_hash(learner)
        for pool in pools:
            sid = pool['source_id']; source = sources[sid]
            probes = validate_probes(bundles[sid], source, build)
            result = run_root(learner, source, pool, probes, folder / sid.replace(':', '_'),
                seed=seed, updates=updates, initial_hash=initial_hash, weight_hash=adapter_hash,
                restore=restore_adapter, evaluate=evaluate_probes)
            print(sid, [(r['threshold'], r['selected_id'], r['Vaudit_accuracy_gain'])
                for r in result['configurations']], flush=True)
        save(folder / 'status.json', {**context.report_identity(), 'at_utc': now(),
            'roots': len(pools), 'updates': updates, 'seed': seed,
            'complete_for_declared_cohort': True, 'Dcal40_complete': False,
            'formal_training_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'measure-cost', 'freeze-cost', 'run', 'summarize', 'check'])
    parser.add_argument('--updates', type=int, choices=UPDATES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args(); context = GeneralContext()
    if args.action == 'prepare':
        v = context.prepare()
        print({k: v[k] for k in ('cohort', 'roots', 'branches_per_grid_job', 'generation_upper_bound_per_grid_job', 'GPU_loaded')})
    elif args.action == 'run':
        run(args.updates, args.seed)
    elif args.action == 'measure-cost':
        value = measure_cost()
        print({k:value[k] for k in ('generations','elapsed_seconds','training_update_seconds')})
    elif args.action == 'freeze-cost':
        value = freeze_cost_plan()
        print({k:value[k] for k in ('estimated_grid_generations_upper_bound','estimated_grid_updates',
            'estimated_seconds_from_six_generation_mean')})
    else:
        from calibration_summary import main
        main(COHORT, check_only=args.action == 'check', context=context)
