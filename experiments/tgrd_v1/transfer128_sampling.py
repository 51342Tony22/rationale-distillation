"""Predeclared train-only type/difficulty strata; no trial or test outcomes."""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from runtime_full import ROOT, OUT, read, freeze, checked_rows, baseline
from budgeted_materials import PROTOCOL, validate_roster

FOLDER = OUT / 'transfer128'
PLAN = FOLDER / 'stratified_sampling_plan_v1.json'
ROSTER = FOLDER / 'stratified_roster_v1.json'
TYPE_PATTERNS = [
    ['percentage', r'percent|percentage|%'],
    ['rate', r'\bper\b|\beach\b|\bspeed\b|\brate\b'],
    ['comparison', r'\bmore than\b|\bless than\b|\bfewer\b|\btimes as\b|\btwice\b|\bhalf\b'],
    ['grouping', r'\bequally\b|\bdivide\w*\b|\bshared\b|\bgroups?\b|\bpackages?\b'],
]
RULE = {'version': 1, 'protocol': PROTOCOL, 'seed': 42, 'root_budget': 128,
    'type_patterns_in_priority_order': TYPE_PATTERNS, 'default_type': 'other',
    'type_labels_are': 'coarse_lexical_heuristics_not_semantic_ground_truth',
    'difficulty': 'maximum_step_mean_NLL_rank_terciles_over_non_development_roots',
    'difficulty_ties': 'numeric_source_index', 'empty_step_stratum': 'no_eligible_step',
    'allocation': 'proportional_largest_remainder_ties_lexical_stratum',
    'draw': 'random.Random(42);sorted_strata;numeric_ID_order_then_shuffle;final_numeric_ID_order',
    'joint_control_or_explanation_success_filters': False,
    'replacement_on_failure': False, 'public_test_used': False}


def index(sid):
    if not re.fullmatch(r'gsm8k:train:(?:0|[1-9]\d*)', sid):
        raise ValueError('Only official train source IDs are allowed')
    value = int(sid.rsplit(':', 1)[1])
    if value >= 7473:
        raise ValueError('Training source ID out of range')
    return value


def draw(sources, diagnostics, common_ids, excluded, budget=128):
    """No filtering by joint control availability or variant/explanation success."""
    if (len(common_ids) != len(set(common_ids)) or budget != 128
            or [r['source_id'] for r in diagnostics] != common_ids
            or any(sid not in sources for sid in common_ids)):
        raise ValueError('Full ordered independent NLL/common source coverage required')
    available = [sid for sid in common_ids if sid not in set(excluded)]
    if len(available) < budget:
        raise ValueError('Insufficient roots; cannot replace development roots or shrink the budget')
    nll = {}; types = {}; byid = {r['source_id']: r for r in diagnostics}
    for sid in available:
        index(sid); row = byid[sid]; steps = row['steps']
        if (row.get('selector') != 'nll' or row.get('protocol') != PROTOCOL
                or row['proposal'].get('attention_used') is not False
                or row['proposal'].get('masking_used') is not False
                or row['proposal'].get('mask_control_filter_used') is not False):
            raise ValueError('Difficulty must use the independent all-step NLL diagnostics')
        values = [s['mean_nll'] for s in steps]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid training-side NLL')
        nll[sid] = max(values) if values else None
        question = sources[sid]['question'].lower()
        types[sid] = next((label for label, pattern in TYPE_PATTERNS if re.search(pattern, question)), 'other')
    ranked = sorted((sid for sid in available if nll[sid] is not None), key=lambda sid: (nll[sid], index(sid)))
    difficulty = {sid: f'tercile_{min(2, 3 * rank // len(ranked))}' for rank, sid in enumerate(ranked)}
    strata = defaultdict(list)
    for sid in sorted(available, key=index):
        strata[types[sid] + '/' + difficulty.get(sid, 'no_eligible_step')].append(sid)
    total = len(available)
    allocations = {key: budget * len(values) // total for key, values in strata.items()}
    remainder = budget - sum(allocations.values())
    order = sorted(strata, key=lambda key: (-(budget * len(strata[key]) % total), key))
    for key in order[:remainder]: allocations[key] += 1
    rng = random.Random(42); selected = []
    for key in sorted(strata):
        candidates = strata[key][:]; rng.shuffle(candidates)
        selected += candidates[:allocations[key]]
    return {'source_ids': sorted(selected, key=index),
        'available_root_count': total,
        'strata': {key: {'available': len(strata[key]), 'allocated': allocations[key]} for key in sorted(strata)},
        'labels': {sid: {'type': types[sid], 'difficulty': difficulty.get(sid, 'no_eligible_step'),
                         'max_step_mean_NLL': nll[sid]} for sid in sorted(available, key=index)}}


def verify_hashes(value):
    if not value.get('inputs'):
        raise ValueError('Sampling evidence needs concrete input hashes')
    paths = []
    for name, checksum in value['inputs'].items():
        path = ROOT / name
        if baseline.sha(path) != checksum:
            raise ValueError('Frozen sampling input changed: ' + name)
        paths.append(path)
    return paths


def training_inputs():
    from full_selector import load_quality, FOLDER as SELECTORS
    sources, teachers, _, ids = load_quality()
    nll_path = SELECTORS / 'positions_nll_v1.jsonl'; diagnostics = checked_rows(nll_path)
    if [r['source_id'] for r in diagnostics] != ids:
        raise ValueError('Full independent NLL diagnostics must finish before sampling')
    byid = {r['source_id']: r for r in teachers}
    if any(r['teacher_output_sha256'] != byid[r['source_id']]['output_sha256'] for r in diagnostics):
        raise ValueError('Sampling NLL diagnostics belong to different teacher outputs')
    paths = [Path(__file__), nll_path, SELECTORS / 'cases_manifest.json',
        OUT / 'gates/full_teacher_automatic_quality_v2.json',
        ROOT / 'data/tgrd_v1/train7473.json', ROOT / 'data/tgrd_v1/cal40.json',
        ROOT / 'data/tgrd_v1/historical_exposure.json']
    sig = OUT / 'signatures/full_selector_nll_v1.json'
    paths += [sig, *verify_hashes(read(sig))]
    excluded = set(read(ROOT / 'data/tgrd_v1/historical_exposure.json')['source_ids'])
    excluded |= {r['source_id'] for r in read(ROOT / 'data/tgrd_v1/cal40.json')}
    excluded |= {'gsm8k:train:0', 'gsm8k:train:2'}
    return {s['source_id']: s for s in sources}, diagnostics, ids, sorted(excluded, key=index), paths


def assert_no_transfer_results():
    # Formal 128 trials must use this namespace. Existing cal34 scientific
    # results are development and do not change the declared sampling rule.
    if any((FOLDER / 'trials').rglob('inputs.json')):
        raise ValueError('Cannot declare a new sampling plan after transfer128 trial inputs')


def prepare_plan():
    if not PLAN.exists(): assert_no_transfer_results()
    sources, diagnostics, ids, excluded, paths = training_inputs()
    result = draw(sources, diagnostics, ids, excluded)
    value = {'schema': 'tgrd_transfer_sampling_plan_v1', 'protocol': PROTOCOL,
        'rule': RULE, 'common_root_ids': ids, 'excluded_development_ids': excluded,
        'sampling_result': result, 'public_test_used': False,
        'frozen_before_transfer_results': True,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    freeze(PLAN, value)
    return value


def freeze_roster():
    plan = prepare_plan(); paths = verify_hashes(plan)
    from full_selector import FOLDER as SELECTORS, load_prepared, validate_resume
    joint_path = SELECTORS / 'positions_joint_v1.jsonl'
    rows = checked_rows(joint_path)
    if [r['source_id'] for r in rows] != plan['common_root_ids']:
        raise ValueError('Joint diagnostics are unfinished; sampling plan preserved without a final roster')
    _, cases = load_prepared()
    validate_resume(rows, cases, 'joint')
    signature = OUT / 'signatures/full_selector_joint_v1.json'
    paths += [PLAN, joint_path, signature, *verify_hashes(read(signature))]
    result = plan['sampling_result']
    value = {'schema': 'tgrd_transfer_stratified_roster_v1', 'protocol': PROTOCOL,
        'seed': 42, 'root_budget': 128, 'source_ids': result['source_ids'],
        'excluded_development_ids': plan['excluded_development_ids'],
        'sampling_rule': RULE, 'sampling_plan': str(PLAN.relative_to(ROOT)),
        'sampling_inputs_sha256': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'strata': result['strata'], 'replacement_on_failure': False,
        'frozen_before_transfer_results': True, 'public_test_used': False}
    validate_roster(value, plan['common_root_ids'])
    if not ROSTER.exists(): assert_no_transfer_results()
    freeze(ROSTER, value)
    return value


def verify_roster(roster, ids):
    if (roster.get('schema') != 'tgrd_transfer_stratified_roster_v1'
            or roster.get('sampling_plan') != str(PLAN.relative_to(ROOT))
            or roster.get('sampling_rule') != RULE):
        raise ValueError('Expected the reproducible predeclared sampling rule')
    validate_roster(roster, ids)
    plan = read(PLAN); verify_hashes(plan)
    if plan != prepare_plan() or roster != freeze_roster():
        raise ValueError('Transfer128 sampling no longer reproduces exactly')
    return [ROSTER, PLAN, *verify_hashes({'inputs': roster['sampling_inputs_sha256']})]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'freeze-roster'))
    args = parser.parse_args()
    value = prepare_plan() if args.command == 'prepare' else freeze_roster()
    print({'sampling_plan_frozen': True, 'final_roster_frozen': args.command == 'freeze-roster',
           'selected_count': len(value.get('source_ids', value.get('sampling_result', {}).get('source_ids', []))),
           'public_test_used': False}, flush=True)
