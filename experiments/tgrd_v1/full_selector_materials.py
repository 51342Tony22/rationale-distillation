"""Assemble the full common-root G1/G2/G3 bundles without transfer inputs.

This is a material preparation step. The actual-model engineering preflights
and separate staged formal-training gate must still pass before model updates.
"""
from __future__ import annotations

import copy
from pathlib import Path

from runtime_full import ROOT, HERE, OUT, read, freeze, baseline
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS, selector_root_materials
from full_selector import load_prepared, load_quality, quality_decisions
from full_explanations import (FOLDER as EXPLANATIONS, assemble as assemble_explanations,
                               student_tokenizer)
from explanation_pool import unique
from training_plan import validate_materials
from learning import tokenize

FOLDER = OUT / 'formal_materials' / PROTOCOL / 'selector_first'


def build_bundles(sources, teachers, reviews, cases, pools, tokenizer):
    ids = [c['source_id'] for c in cases]
    if (len(ids) != len(set(ids)) or any(set(x) != set(ids) for x in
            (sources, teachers, reviews, pools['joint'], pools['nll']))):
        raise ValueError('Every selector group needs the same accepted root coverage')
    bundles = {g: {'schema': 'tgrd_training_materials_v1', 'condition': g,
        'protocol': PROTOCOL, 'candidate_root_count': 7473, 'rows': [],
        'stage': 'selector_first', 'development_preview_only': False,
        'formal_training_gate_required': True} for g in SELECTOR_CONDITIONS}
    choices = []
    for case in cases:
        sid = case['source_id']
        if case['teacher_output_sha256'] != teachers[sid]['output_sha256']:
            raise ValueError('Candidate steps belong to a different teacher output')
        steps = {c['step']['step_index']: c['step'] for c in case['candidates']}
        rows, decision = selector_root_materials(sources[sid], teachers[sid], reviews[sid],
            pools['joint'][sid], steps, nll_pool=pools['nll'][sid])
        # Tokenize once per group/root; all four epochs have identical material.
        for group in SELECTOR_CONDITIONS:
            material = rows[group]['epoch_materials'][0]
            sample = tokenize(tokenizer, material['question'], material['completion'])
            material['completion_tokens_including_eos'] = len(sample['target_ids'])
            rows[group]['epoch_materials'] = [copy.deepcopy(material) for _ in range(4)]
            bundles[group]['rows'].append(rows[group])
        choices.append(decision)
    for bundle in bundles.values():
        validate_materials(bundle, ids)
    coverage = {'candidate_root_count': 7473, 'accepted_root_ids': ids,
        'accepted_root_count': len(ids), 'common_root_order': True,
        'groups': {g: {
            'root_exposures': 4 * len(ids),
            'roots_with_explanation': sum(r['epoch_materials'][0]['kind'] == 'explained_CoT'
                                          for r in bundles[g]['rows']),
            'ordinary_only_roots': sum(r['epoch_materials'][0]['kind'] == 'ordinary_CoT'
                                       for r in bundles[g]['rows']),
            'supervised_tokens_including_eos': sum(m['completion_tokens_including_eos']
                for r in bundles[g]['rows'] for m in r['epoch_materials'])
        } for g in SELECTOR_CONDITIONS},
        'budget_equivalence': 'same_roots_order_initialization_and_optimizer_updates;token_totals_reported',
        'variant_or_trial_inputs_read': False, 'formal_training_enabled': False,
        'public_test_used': False}
    return bundles, choices, coverage


def prepare():
    # Reconstruct reviewed pools from signed generations, exact requests and
    # semantic decisions, rather than trusting a standalone pool JSON.
    assemble_explanations()
    parent, cases = load_prepared()
    source_rows, teacher_rows, quality, ids = load_quality(ROOT / parent['quality_file'])
    selected = set(ids)
    def accepted(rows):
        return unique([r for r in rows if r['source_id'] in selected], 'source_id')
    pools_manifest_path = EXPLANATIONS / 'reviewed_pools_manifest.json'
    pools_manifest = read(pools_manifest_path)
    pools = {s: unique(read(EXPLANATIONS / f'reviewed_pools_{s}.json'), 'source_id')
             for s in ('joint', 'nll')}
    if pools_manifest['accepted_root_ids'] != ids:
        raise ValueError('Explanation pools differ from the common full-quality roster')
    bundles, choices, coverage = build_bundles(accepted(source_rows), accepted(teacher_rows),
        accepted(quality_decisions(quality)), cases, pools, student_tokenizer())
    for group, bundle in bundles.items():
        freeze(FOLDER / f'{group}.json', bundle)
    freeze(FOLDER / 'choices.json', choices)
    freeze(FOLDER / 'coverage.json', coverage)
    paths = [Path(__file__), HERE / 'budgeted_materials.py', HERE / 'material_builder.py',
        HERE / 'training_plan.py', HERE / 'learning.py', HERE / 'explanation_pool.py',
        HERE / 'full_explanations.py', HERE / 'full_selector.py',
        HERE / 'runtime.py', HERE / 'runtime_full.py',
        OUT / 'transfer128_amendment_v1.json', OUT / 'execution_priority.json',
        ROOT / parent['quality_file'], pools_manifest_path,
        HERE.parent / 'math_distill_v1/config.json',
        FOLDER / 'choices.json', FOLDER / 'coverage.json']
    paths += [EXPLANATIONS / f'reviewed_pools_{s}.json' for s in ('joint', 'nll')]
    manifest = {'schema': 'tgrd_selector_stage_materials_v1', 'protocol': PROTOCOL,
        'stage': 'selector_first', 'conditions': list(SELECTOR_CONDITIONS),
        'formal_seeds': [42], 'epochs': 4, 'candidate_root_count': 7473,
        'common_root_ids': ids,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'materials': {g: {'path': str((FOLDER / f'{g}.json').relative_to(ROOT)),
                         'sha256': baseline.sha(FOLDER / f'{g}.json')} for g in SELECTOR_CONDITIONS},
        'G4_or_G5_materials_required': False, 'formal_training_enabled': False,
        'public_test_used': False}
    freeze(FOLDER / 'materials_manifest.json', manifest)
    print({'groups': list(SELECTOR_CONDITIONS), 'accepted_roots': len(ids),
           'formal_training_enabled': False}, flush=True)
    return manifest


if __name__ == '__main__':
    prepare()
