"""Compile actual output-bound cal34 reviews; never infer review decisions.

This report is a bounded development diagnostic, not a material export or gate.
Original generation reports and failed mathematical-teacher pools stay intact.
"""
from __future__ import annotations

import collections
from pathlib import Path

from runtime_full import ROOT, HERE, BASE, read, checked_rows, freeze, baseline, now
from explanation_teacher_cal34 import FOLDER, COHORT, merge_requests
from explanation_pool import (TYPES, REVIEW_CHECKS, selector_paths, unique, inputs,
                              mechanical_checks, reviewed_candidate)
from calibration_cohort import cohort_paths


def validate_reviews(requests, generated, reviews, tokenizer):
    req = unique(requests, 'candidate_id')
    out = unique(generated, 'candidate_id')
    rev = unique(reviews, 'candidate_id')
    if set(req) != set(out) or set(req) != set(rev):
        raise ValueError('Every unique attempt needs an actual complete full-text decision')
    for cid, request in req.items():
        row, review = out[cid], rev[cid]
        checks = mechanical_checks(request, row, tokenizer)
        if (row.get('source_id') != request['source_id']
                or row.get('mechanical_checks') != checks
                or row.get('engineering_only') is not True
                or row.get('scientific_pool_exported') is not False
                or review.get('output_sha256') != row['output_sha256']
                or review.get('all_text_read') is not True
                or review.get('kind') != 'assistant_mathematical_review'
                or type(review.get('accepted')) is not bool or not review.get('reason')
                or review.get('mechanical_hard_rejection_reasons') != checks['hard_rejection_reasons']):
            raise ValueError('Actual review/generation/mechanical evidence differs: ' + cid)
        if review['accepted'] and (checks['hard_rejection_reasons']
                or any(review.get(k) is not True for k in REVIEW_CHECKS)
                or review.get('contrastive_alternative_valid_and_plausible') is False):
            raise ValueError('Acceptance conflicts with rejection or semantic evidence: ' + cid)
    return req, out, rev


def compile_report():
    plan = read(FOLDER / 'plan.json')
    for name, checksum in plan['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Bounded diagnostic input changed: ' + name)
    if baseline.sha(FOLDER / 'requests.json') != plan['request_sha256']:
        raise ValueError('Original cal34 requests changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
        local_files_only=True, trust_remote_code=False)
    requests = read(FOLDER / 'requests.json')
    generated = checked_rows(FOLDER / 'generated.jsonl')
    reviews = checked_rows(FOLDER / 'full_text_reviews_v1.jsonl')
    req, out, rev = validate_reviews(requests, generated, reviews, tokenizer)
    sources, teachers = inputs()
    cases = read(cohort_paths(COHORT).cases)
    ids = [c['source_id'] for c in cases]
    if len(ids) != 34 or len(requests) != 134:
        raise ValueError('This report is bounded to the original134 attempts/34 roots')
    views = {s: read(selector_paths(COHORT, s).pool_folder / 'requests.json')
             for s in ('joint', 'nll')}
    if merge_requests(list(views.values())) != requests:
        raise ValueError('Diagnostic requests must exactly reconstruct both selectors')
    by_selector = {}; choices = {}; usable_by_selector = {}
    for selector, view in views.items():
        groups = collections.defaultdict(list)
        accepted = {}
        for request in view:
            cid, sid = request['candidate_id'], request['source_id']
            candidate, reasons = reviewed_candidate(request, out[cid], rev[cid],
                                                    sources[sid], teachers[sid], tokenizer)
            if bool(candidate) != rev[cid]['accepted']:
                raise ValueError('Semantic decision cannot be used as a training candidate: ' + cid)
            groups[sid].append((request, candidate, reasons))
            if candidate:
                accepted[cid] = candidate
        usable_by_selector[selector] = accepted
        decisions = []; ranges = {}
        for sid in ids:
            available = [c for r, c, _ in groups[sid] if c]
            rank0 = [c for c in available if c['position_rank'] == 0]
            chosen = next((c for kind in TYPES for c in rank0
                           if c['explanation_type'] == kind), None)
            lengths = [c['student_tokens'] for c in available]
            ranges[sid] = max(lengths) - min(lengths) if lengths else 0
            decisions.append({'source_id': sid,
                'selected_id': chosen['candidate_id'] if chosen else 'e0',
                'type': chosen['explanation_type'] if chosen else 'empty',
                'student_tokens': chosen['student_tokens'] if chosen else 0,
                'rank0_only': True})
        choices[selector] = decisions
        selected_types = collections.Counter(d['type'] for d in decisions)
        by_selector[selector] = {'requests': len(view), 'usable_explanations': len(accepted),
            'rank0_fixed_selection': dict(selected_types),
            'roots_with_nonempty_fixed_explanation': sum(d['selected_id'] != 'e0' for d in decisions),
            'fallback_roots': [d['source_id'] for d in decisions if d['selected_id'] == 'e0'],
            'pool_token_ranges': ranges,
            'roots_exceeding_8_token_matching_target': [sid for sid, span in ranges.items() if span > 8],
            'length_target_is_not_silently_relaxed_or_filtered': True}
    shared = set(usable_by_selector['joint']) & set(usable_by_selector['nll'])
    for cid in shared:
        j, n = (usable_by_selector[s][cid] for s in ('joint', 'nll'))
        if any(j[k] != n[k] for k in ('text_sha256', 'completion_sha256', 'student_tokens', 'review')):
            raise ValueError('Cross-selector shared explanation differs: ' + cid)
    # NLL-only positions remain visible only to the NLL view.
    joint_ids = {r['candidate_id'] for r in views['joint']}
    nll_ids = {r['candidate_id'] for r in views['nll']}
    if set(usable_by_selector['joint']) & (nll_ids - joint_ids):
        raise ValueError('NLL-only candidates contaminated the joint pool')
    files = [Path(__file__), HERE / 'explanation_pool.py', FOLDER / 'plan.json',
        FOLDER / 'requests.json', FOLDER / 'generated.jsonl', FOLDER / 'full_text_reviews_v1.jsonl',
        cohort_paths(COHORT).cases, cohort_paths(COHORT).reviews]
    files += [selector_paths(COHORT, s).pool_folder / 'requests.json' for s in views]
    files += sorted(FOLDER.glob('full_text_review_batch_*_v1.json'))
    # Check that append-ledger entries exactly match the actually written review batches.
    batch_reviews = unique([r for p in sorted(FOLDER.glob('full_text_review_batch_*_v1.json'))
                            for r in read(p)['reviews']], 'candidate_id')
    normalized = {cid: {k: v for k, v in r.items() if k != 'record_sha256'} for cid, r in rev.items()}
    if batch_reviews != normalized:
        raise ValueError('Frozen review batches and signed decision ledger disagree')
    hard = collections.Counter(reason for r in generated
                               for reason in r['mechanical_checks']['hard_rejection_reasons'])
    report = {'schema': 'tgrd_general_explainer_cal34_semantic_report_v1', 'at_utc': now(),
        'completed_unique': len(req), 'all_outputs_actually_full_text_reviewed': True,
        'mechanical_pass': sum(not r['mechanical_checks']['hard_rejection_reasons'] for r in generated),
        'semantic_accepted': sum(r['accepted'] for r in reviews),
        'semantic_rejected': sum(not r['accepted'] for r in reviews),
        'semantic_rejected_after_mechanical_pass': sum(not rev[cid]['accepted'] and
            not row['mechanical_checks']['hard_rejection_reasons'] for cid, row in out.items()),
        'accepted_by_type': {kind: sum(rev[cid]['accepted'] for cid, r in req.items()
                                      if r['explanation_type'] == kind) for kind in TYPES},
        'mechanical_rejection_reasons': dict(hard), 'selectors': by_selector,
        'fixed_rank0_choices': choices, 'identical_usable_shared_candidates': len(shared),
        'nll_only_request_count': len(nll_ids - joint_ids), 'nll_only_contaminates_joint': False,
        'fixed_material_disagreements': sum(j['selected_id'] != n['selected_id']
            for j, n in zip(choices['joint'], choices['nll'])),
        'generation_seconds_new128': sum(r['seconds'] for r in generated
                                          if not r.get('reused_from_pilot_candidate_id')),
        'learning_or_transfer_evidence': False, 'formal_teacher_changed': False,
        'scientific_pool_exported': False, 'readiness_gate_created': False,
        'method_effectiveness_established': False, 'public_test_used': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}}
    freeze(FOLDER / 'semantic_review_v1.json', report)
    return report


if __name__ == '__main__':
    report = compile_report()
    print({k: report[k] for k in ('completed_unique', 'semantic_accepted',
        'semantic_rejected', 'accepted_by_type', 'fixed_material_disagreements')})
