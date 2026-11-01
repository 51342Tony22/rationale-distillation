"""One bounded development-cohort check of the fixed general explainer.

Original prompts, selectors, token limits and outputs remain unchanged. The six
diagnostic outputs are reused, including failures. This runner exports no
scientific pool, changes no formal teacher and creates no readiness gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, BASE, read, checked_rows, freeze, save,
    append, baseline, signature, gpu, check_time, now)
from explanation_pool import prepare as prepare_original, generate_one, mechanical_checks
from explanation_teacher_pilot import FOLDER as PILOT, MODEL, REPO, REVISION

FOLDER = OUT / 'calibration/explanation_teacher_cal34_v1'
COHORT = 'cal40_reviewed34_v2'
IDENTITY = ('source_id', 'question_sha256', 'teacher_output_sha256', 'step',
            'system', 'user', 'visible_context', 'explanation_type')


def merge_requests(groups):
    """Position ranks can differ; identical positions/types share one output."""
    canonical = {}
    for requests in groups:
        for request in requests:
            cid = request['candidate_id']
            if cid in canonical:
                if any(canonical[cid][k] != request[k] for k in IDENTITY):
                    raise ValueError('Cross-selector prompt differs: ' + cid)
            else:
                canonical[cid] = dict(request)
    return list(canonical.values())


def pilot_reuse(requests):
    canonical = {r['candidate_id']: r for r in requests}
    originals = {r['candidate_id']: r for r in read(PILOT / 'requests.json')}
    reviews = {r['candidate_id']: r for r in read(PILOT / 'full_text_reviews_v1.json')}
    semantic = read(PILOT / 'semantic_review_v1.json')
    for name, checksum in semantic['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Six-output semantic evidence changed: ' + name)
    if (semantic.get('all_six_actually_read') is not True
            or semantic.get('completed') != 6 or semantic.get('semantic_accepted') != 2
            or semantic.get('scientific_pool_exported') is not False
            or semantic.get('formal_teacher_changed') is not False):
        raise ValueError('Expected the actual fixed six-output diagnostic and review')
    reused = {}
    for row in checked_rows(PILOT / 'generated.jsonl'):
        old = originals[row['candidate_id']]
        cid = row['original_candidate_id']; request = canonical[cid]
        if (any(request[k] != old[k] for k in IDENTITY)
                or reviews[row['candidate_id']]['output_sha256'] != row['output_sha256']
                or reviews[row['candidate_id']].get('all_text_read') is not True):
            raise ValueError('Pilot output/review does not match the original development prompt')
        value = dict(row); value.pop('record_sha256', None)
        value.update(candidate_id=cid, reused_from_pilot_candidate_id=row['candidate_id'])
        reused[cid] = value
    if len(reused) != 6:
        raise ValueError('All six pilot attempts, including failures, must be reused')
    return reused


def prepare():
    joint = prepare_original(COHORT, 'joint')
    nll = prepare_original(COHORT, 'nll')
    requests = merge_requests([joint, nll]); reused = pilot_reuse(requests)
    if (len(joint), len(nll), len(requests)) != (110, 120, 134):
        raise ValueError('This bounded check is only the already declared 34-root cohort')
    manifest = read(PILOT / 'model_manifest.json')
    if manifest['repo'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('A different diagnostic teacher is not authorized by this plan')
    freeze(FOLDER / 'requests.json', requests)
    paths = [Path(__file__), HERE / 'explanation_pool.py', HERE / 'explanation_teacher_pilot.py',
        PILOT / 'model_manifest.json', PILOT / 'plan.json', PILOT / 'requests.json',
        PILOT / 'generated.jsonl', PILOT / 'full_text_reviews_v1.json', PILOT / 'semantic_review_v1.json']
    for name in ('shared_pool_v1', 'nll_pool_v1'):
        paths.append(OUT / f'calibration/{COHORT}/{name}/requests.json')
    plan = {'schema': 'tgrd_general_explainer_cal34_diagnostic_v1',
        'teacher_repo': REPO, 'teacher_revision': REVISION, 'root_count': 34,
        'joint_requests': len(joint), 'nll_requests': len(nll),
        'unique_requests': len(requests), 'exact_cross_selector_reuses': 96,
        'pilot_outputs_reused_including_failures': len(reused),
        'maximum_new_generations': len(requests) - len(reused),
        'temperature': 0, 'max_new_tokens': 256, 'attempts_per_unique_prompt': 1,
        'student_token_range_unchanged': [32, 64], 'prompt_changed': False,
        'only_generation_factor_changed': 'general_instruction_teacher',
        'formal_teacher_changed': False, 'scientific_pool_export': False,
        'full_data_expansion_authorized_by_this_plan': False,
        'automatic_prompt_or_model_search': False, 'learning_or_transfer_evidence': False,
        'requires_actual_full_text_review_after_generation': True, 'public_test_used': False,
        'request_sha256': baseline.sha(FOLDER / 'requests.json'),
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    freeze(FOLDER / 'plan.json', plan)
    return plan, requests, reused


def run():
    plan, requests, reused = prepare()
    manifest = read(PILOT / 'model_manifest.json')
    for name, spec in manifest['files'].items():
        if baseline.sha(MODEL / name) != spec['sha256']:
            raise ValueError('Fixed diagnostic model changed: ' + name)
    signature('explanation_teacher_cal34_v1',
        [ROOT / name for name in plan['inputs']] + [FOLDER / 'plan.json', FOLDER / 'requests.json'],
        {'engineering_only': True, 'no_scientific_pool_export': True})
    path = FOLDER / 'generated.jsonl'; saved = checked_rows(path)
    if [r['candidate_id'] for r in saved] != [r['candidate_id'] for r in requests[:len(saved)]]:
        raise ValueError('Bounded diagnostic resume order changed')
    if len(saved) < len(requests):
        with gpu():
            import mlx.core as mx
            from mlx_lm import load
            from transformers import AutoTokenizer
            mx.set_memory_limit(14 * 1024**3); mx.set_cache_limit(512 * 1024**2)
            student = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
                local_files_only=True, trust_remote_code=False)
            model, tokenizer = load(str(MODEL), tokenizer_config={
                'local_files_only': True, 'trust_remote_code': False})
            model.eval()
            for request in requests[len(saved):]:
                check_time(); cid = request['candidate_id']
                if cid in reused:
                    row = dict(reused[cid])
                    row['mechanical_checks'] = mechanical_checks(request, row, student)
                else:
                    output = generate_one(model, tokenizer, request)
                    row = {'candidate_id': cid, 'source_id': request['source_id'], **output,
                        'mechanical_checks': mechanical_checks(request, output, student)}
                row.update(engineering_only=True, scientific_pool_exported=False)
                append(path, row)
                print(cid, row['mechanical_checks']['hard_rejection_reasons'], flush=True)
    rows = checked_rows(path)
    save(FOLDER / 'report.json', {'at_utc': now(), 'completed_unique': len(rows),
        'total_unique': len(requests), 'pilot_reuses': len(reused),
        'mechanical_pass': sum(not r['mechanical_checks']['hard_rejection_reasons'] for r in rows),
        'new_generation_seconds': sum(r['seconds'] for r in rows
                                      if not r.get('reused_from_pilot_candidate_id')),
        'semantic_review_pending': True, 'scientific_pool_exported': False,
        'formal_teacher_changed': False, 'readiness_gate_created': False,
        'method_effectiveness_established': False, 'public_test_used': False,
        'plan_sha256': baseline.sha(FOLDER / 'plan.json'),
        'generated_sha256': baseline.sha(path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare':
        plan, _, _ = prepare(); print(plan, flush=True)
    else:
        run()
