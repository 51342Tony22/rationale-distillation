"""Selector-first material assembly with a bounded, frozen transfer roster.

No public-test access or model work. Legacy development assembly remains intact;
these entry points neither invent reviews nor enable formal training gates.
"""
from __future__ import annotations

import copy
import re

from material_builder import root_materials


PROTOCOL = 'selector_first_transfer128_seed42_v1'
SELECTOR_CONDITIONS = ('M1', 'NLL', 'M3')
TRANSFER_ROOT_BUDGET = 128


def validate_roster(roster, common_root_ids):
    ids = roster.get('source_ids', [])
    excluded = roster.get('excluded_development_ids', [])
    if (roster.get('protocol') != PROTOCOL
            or roster.get('seed') != 42
            or roster.get('root_budget') != TRANSFER_ROOT_BUDGET
            or len(ids) != TRANSFER_ROOT_BUDGET
            or len(set(ids)) != TRANSFER_ROOT_BUDGET
            or len(common_root_ids) != len(set(common_root_ids))
            or not set(ids) <= set(common_root_ids)
            or set(ids) & set(excluded)
            or roster.get('frozen_before_transfer_results') is not True
            or roster.get('public_test_used') is not False
            or roster.get('replacement_on_failure') is not False
            or not roster.get('sampling_rule')
            or not roster.get('sampling_inputs_sha256')):
        raise ValueError('Transfer needs the complete frozen 128-root training-only roster')
    for sid in common_root_ids:
        if (not re.fullmatch(r'gsm8k:train:(?:0|[1-9]\d*)', sid)
                or int(sid.rsplit(':', 1)[1]) >= 7473):
            raise ValueError('Only official GSM8K training roots may enter the roster')
    return frozenset(ids)


def selector_root_materials(source, teacher, review, pool, steps, *, nll_pool):
    """Build G1-G3 without requiring or reading variants or trial results."""
    rows, choices = root_materials(source, teacher, review, pool, steps,
                                  nll_pool=nll_pool)
    return ({g: rows[g] for g in SELECTOR_CONDITIONS}, {
        **choices, 'protocol': PROTOCOL, 'stage': 'selector_first',
        'selected_candidate_ids': {g: choices['selected_candidate_ids'][g]
                                   for g in SELECTOR_CONDITIONS},
        'fallback_reasons': {g: why for g, why in choices['fallback_reasons'].items()
                             if g in SELECTOR_CONDITIONS},
        'trial_inputs_required': False,
        'variant_labels_in_formal_gradients_only_for': [],
        'M6_variant_ids': [],
    })


def transfer_root_materials(source, teacher, review, pool, steps, *, nll_pool,
                            roster, common_root_ids, variant_bundle=None,
                            trial_inputs=None, trial_decisions=None,
                            trial_seed=701, updates=3, threshold=.03):
    """G4 equals G3 outside the roster; inside retains the real Vsel rule.

    Missing required trial results remain errors. Missing qualified probes or
    no acceptable candidate uses the historical e0 fallback inside the roster.
    The caller must bind this roster and its inputs by hash in the formal gate.
    """
    ids = validate_roster(roster, common_root_ids)
    sid = source['source_id']
    if sid not in common_root_ids:
        raise ValueError('Material root is outside the common frozen root set')
    if sid not in ids:
        if any(x is not None for x in (variant_bundle, trial_inputs, trial_decisions)):
            raise ValueError('Transfer work is forbidden outside the frozen budget')
        rows, choices = selector_root_materials(source, teacher, review, pool, steps,
                                                nll_pool=nll_pool)
        return copy.deepcopy(rows['M3']), {
            'source_id': sid, 'protocol': PROTOCOL, 'condition': 'M7',
            'transfer_budget_member': False, 'policy': 'exact_G3_material_reuse',
            'selected_candidate_id': choices['selected_candidate_ids']['M3'],
            'audit_predictions_read': False,
        }
    rows, choices = root_materials(source, teacher, review, pool, steps,
        nll_pool=nll_pool, variant_bundle=variant_bundle, trial_inputs=trial_inputs,
        trial_decisions=trial_decisions, trial_seed=trial_seed, updates=updates,
        threshold=threshold)
    return rows['M7'], {
        'source_id': sid, 'protocol': PROTOCOL, 'condition': 'M7',
        'transfer_budget_member': True,
        'policy': 'original_real_trial_Vsel_accuracy_and_NLL_rule',
        'selected_candidate_id': choices['selected_candidate_ids']['M7'],
        'fallback_reason': choices['fallback_reasons'].get('M7'),
        'trial_seed': trial_seed, 'updates': updates, 'threshold': threshold,
        'audit_predictions_read': False,
    }


def validate_transfer_coverage(g3_bundle, g4_bundle, roster):
    """Reject any coverage/order change or off-budget material modification."""
    from training_plan import validate_materials
    if (g3_bundle.get('protocol') != PROTOCOL or g4_bundle.get('protocol') != PROTOCOL
            or g3_bundle.get('condition') != 'M3' or g4_bundle.get('condition') != 'M7'):
        raise ValueError('Expected budgeted-study G3 and G4 bundles')
    g3 = validate_materials(g3_bundle)
    ids = [r['source_id'] for r in g3]
    selected = validate_roster(roster, ids)
    g4 = validate_materials(g4_bundle, ids)
    for before, after in zip(g3, g4):
        if before['source_id'] not in selected and before != after:
            raise ValueError('G4 must reuse exact G3 materials outside its 128 roots')
