"""Compile actual full-text decisions into the common full-train quality gate.

Inspection never accepts a sample. Legacy numeric/format reviews are excluded.
A separate frozen policy must bind the exact complete decision ledger and an
explicit consistency review before the gate can be created. No test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, DATA, read, checked_rows, freeze, save,
                          baseline, digest, now)
from full_selector import validate_quality, QUALITY, SOURCE, TEACHERS
from budgeted_materials import PROTOCOL

AUDITS = OUT / 'material_audits'
POLICY = AUDITS / 'common_teacher_quality_policy_frozen_v1.json'


def decision_digest(reviews):
    """Review list identity independent of file formatting and append envelopes."""
    normalized = [{k: v for k, v in r.items() if k != 'record_sha256'} for r in reviews]
    ordered = sorted(normalized, key=lambda r: r['source_id'])
    return digest(json.dumps(ordered, sort_keys=True, ensure_ascii=False))


def collect_reviews(groups):
    byid = {}
    for rows in groups:
        for review in rows:
            sid = review.get('source_id')
            if (not isinstance(sid, str) or not sid.startswith('gsm8k:train:')
                    or review.get('all_text_read') is not True
                    or review.get('review_kind') != 'assistant_full_text_mathematical_review'
                    or type(review.get('accepted')) is not bool
                    or not review.get('reason') or not review.get('question_sha256')
                    or not review.get('output_sha256')):
                raise ValueError('An actual output-bound mathematical full-text review is required')
            if sid in byid:
                raise ValueError('Duplicate review must be resolved with preserved history: ' + sid)
            byid[sid] = review
    return byid


def validate_policy(policy, reviews):
    required = ('arithmetic_and_units', 'all_text_and_self_correction',
                'source_ambiguity_and_entity_identity', 'calendar_conventions',
                'gold_units_and_reference_errors', 'truncation_and_context',
                'shared_root_exclusions')
    if (policy.get('schema') != 'tgrd_common_teacher_quality_policy_v1'
            or policy.get('status') != 'frozen'
            or policy.get('protocol') != PROTOCOL
            or policy.get('candidate_root_count') != 7473
            or policy.get('all_candidate_decisions_consistency_reviewed') is not True
            or policy.get('public_test_used') is not False
            or policy.get('numeric_match_substitutes_for_full_text_review') is not False
            or policy.get('teacher_or_gold_silently_edited') is not False
            or policy.get('decision_set_sha256') != decision_digest(reviews)
            or any(not isinstance(policy.get('rules', {}).get(k), str)
                   or not policy['rules'][k].strip() for k in required)):
        raise ValueError('A concrete consistent policy must bind this exact complete decision set')


def assemble(sources, teachers, groups, policy, inputs, policy_path):
    byid = collect_reviews(groups)
    expected = {f'gsm8k:train:{i}' for i in range(7473)}
    if set(byid) != expected:
        raise ValueError('Need all7473 actual full-text decisions; legacy imports cannot fill gaps')
    reviews = [byid[s['source_id']] for s in sources]
    validate_policy(policy, reviews)
    if not inputs.get(policy_path):
        raise ValueError('The concrete policy file must be included in frozen input hashes')
    value = {'schema': 'tgrd_full_teacher_quality_v1', 'protocol': PROTOCOL,
        'status': 'frozen', 'candidate_root_count': 7473,
        'review_policy': {'path': policy_path, 'sha256': inputs[policy_path]},
        'inputs': inputs, 'reviews': reviews,
        'accepted_root_ids': [s['source_id'] for s in sources if byid[s['source_id']]['accepted']],
        'legacy_reviews_auto_promoted': False, 'teacher_or_gold_edited': False,
        'public_test_used': False, 'decision_set_sha256': decision_digest(reviews)}
    validate_quality(sources, teachers, value)
    return value


def load_ledgers():
    paths = [AUDITS / 'cal40_teacher_full_text_reviews_v1.json',
             AUDITS / 'teacher_additional_reviews_v1.jsonl']
    return paths, [read(paths[0]), checked_rows(paths[1])]


def inspect():
    paths, groups = load_ledgers()
    reviews = collect_reviews(groups)
    sources = read(SOURCE)
    teachers = {r['source_id']: r for r in checked_rows(TEACHERS)}
    for source in sources:
        sid = source['source_id']
        if sid not in reviews:
            continue
        review, teacher = reviews[sid], teachers[sid]
        if (review['question_sha256'] != source['question_sha256']
                or review['output_sha256'] != teacher['output_sha256']
                or digest(teacher['output']) != teacher['output_sha256']):
            raise ValueError('Review does not bind the actual source/output: ' + sid)
        if review['accepted'] and (not teacher['correct'] or teacher['finish_reason'] != 'stop'):
            raise ValueError('Accepted review conflicts with original gold or completeness: ' + sid)
    missing = [s['source_id'] for s in sources if s['source_id'] not in reviews]
    value = {'at_utc': now(), 'actual_modern_full_text_decisions': len(reviews),
        'accepted_provisional': sum(r['accepted'] for r in reviews.values()),
        'quarantined': sum(not r['accepted'] for r in reviews.values()),
        'missing_actual_full_text_decisions': len(missing), 'missing_ids': missing,
        'decision_set_sha256': decision_digest(list(reviews.values())),
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in [*paths, SOURCE, TEACHERS]},
        'legacy_imports_excluded': True, 'quality_gate_created_by_inspection': False,
        'common_policy_frozen': POLICY.exists(), 'public_test_used': False}
    save(AUDITS / 'actual_full_text_readiness_v1.json', value)
    return value


def freeze_gate():
    paths, groups = load_ledgers()
    policy = read(POLICY)
    paths += [POLICY, Path(__file__), HERE / 'full_selector.py', SOURCE, TEACHERS,
              OUT / 'signatures/teacher_full7473.json', OUT / 'transfer128_amendment_v1.json']
    value = assemble(read(SOURCE), checked_rows(TEACHERS), groups, policy,
        {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}, str(POLICY.relative_to(ROOT)))
    freeze(QUALITY, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'freeze'])
    args = parser.parse_args()
    if args.action == 'inspect':
        result = inspect()
        print({k: result[k] for k in ('actual_modern_full_text_decisions',
            'accepted_provisional', 'quarantined', 'missing_actual_full_text_decisions',
            'quality_gate_created_by_inspection')})
    else:
        result = freeze_gate()
        print({'accepted_roots': len(result['accepted_root_ids']), 'candidates': 7473})
