"""Isolated six-output check of a general instruction model as an explainer.

Only the teacher changes relative to the original local-explanation recipe.
No automatic adoption, retries, material export, learning, or public-test use.
The existing mathematical teacher and its signed outputs remain unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, BASE, read, checked_rows, freeze, save,
    append, baseline, signature, gpu, now)
from explanation_pool import generate_one, mechanical_checks
from explanation_repair_pilot import prepare as prepare_failed_prompt_pilot

REPO = 'mlx-community/Qwen2.5-7B-Instruct-4bit'
REVISION = 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'
MODEL = ROOT / 'models/Qwen2.5-7B-Instruct-4bit-explainer-pilot'
FOLDER = OUT / 'calibration/explanation_teacher_pilot_v1'
ORIGINAL = OUT / 'calibration/cal40_reviewed34_v2/shared_pool_v1'


def prepare():
    failed = read(OUT / 'calibration/explanation_repair_pilot_v2/report.json')
    if failed['completed'] != 6 or failed['mechanical_pass'] != 0:
        raise ValueError('This teacher diagnostic follows the fixed failed prompt repair')
    _, pilot = prepare_failed_prompt_pilot()
    original = {r['candidate_id']: r for r in read(ORIGINAL / 'requests.json')}
    requests = []
    for selected in pilot:
        request = dict(original[selected['original_candidate_id']])
        request.update(candidate_id=request['candidate_id'] + ':teacher_pilot_v1',
            original_candidate_id=selected['original_candidate_id'],
            engineering_only=True, export_to_scientific_pool=False)
        requests.append(request)
    freeze(FOLDER / 'requests.json', requests)
    paths = [Path(__file__), HERE / 'explanation_pool.py',
        ORIGINAL / 'requests.json', OUT / 'calibration/explanation_repair_pilot_v2/report.json']
    plan = {'schema': 'tgrd_explanation_teacher_diagnostic_v1',
        'teacher_repo': REPO, 'teacher_revision': REVISION,
        'teacher_dir': str(MODEL.relative_to(ROOT)), 'request_count': 6,
        'selection_rule': 'same_first3_roots_and_both_types_as_failed_prompt_pilot',
        'only_generation_factor_changed': 'general_instruction_teacher',
        'prompt': 'original_unmodified_local_explanation_system_and_user',
        'request_sha256': baseline.sha(FOLDER / 'requests.json'),
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'temperature': 0, 'max_new_tokens': 256, 'attempts_per_request': 1,
        'student_token_range_unchanged': [32, 64], 'automatic_adoption': False,
        'scientific_pool_export': False, 'public_test_used': False,
        'learning_or_transfer_evidence': False}
    freeze(FOLDER / 'plan.json', plan)
    return plan, requests


def download():
    prepare()
    from huggingface_hub import snapshot_download, model_info
    info = model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise ValueError('Remote model revision differs from the fixed pilot')
    snapshot_download(REPO, revision=REVISION, local_dir=str(MODEL),
        ignore_patterns=['.gitattributes'], token=False, max_workers=4)
    files = {}
    for blob in info.siblings:
        if blob.rfilename == '.gitattributes':
            continue
        path = MODEL / blob.rfilename
        checksum = baseline.sha(path)
        if path.stat().st_size != blob.size:
            raise ValueError('Downloaded file size mismatch: ' + blob.rfilename)
        if blob.lfs is not None and checksum != blob.lfs.sha256:
            raise ValueError('Downloaded weight LFS digest mismatch: ' + blob.rfilename)
        files[blob.rfilename] = {'sha256': checksum, 'size': blob.size,
            'remote_lfs_digest_verified': blob.lfs is not None}
    freeze(FOLDER / 'model_manifest.json', {'repo': REPO, 'revision': REVISION,
        'files': files, 'role': 'engineering_only_candidate_explainer'})
    print('Pinned explainer pilot model downloaded and hashes verified', flush=True)


def run():
    plan, requests = prepare()
    manifest = read(FOLDER / 'model_manifest.json')
    if manifest['repo'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Expected the fixed diagnostic teacher')
    for name, spec in manifest['files'].items():
        if baseline.sha(MODEL / name) != spec['sha256']:
            raise ValueError('Pilot model file changed: ' + name)
    signature('explanation_teacher_pilot_v1', [Path(__file__), HERE / 'explanation_pool.py',
        FOLDER / 'requests.json', FOLDER / 'plan.json', FOLDER / 'model_manifest.json'],
        {'engineering_only': True, 'no_automatic_adoption': True})
    path = FOLDER / 'generated.jsonl'
    rows = checked_rows(path)
    if [r['candidate_id'] for r in rows] != [r['candidate_id'] for r in requests[:len(rows)]]:
        raise ValueError('Pilot resume request order changed')
    if len(rows) < len(requests):
        with gpu():
            import mlx.core as mx
            from mlx_lm import load
            from transformers import AutoTokenizer
            mx.set_memory_limit(14 * 1024**3)
            mx.set_cache_limit(512 * 1024**2)
            student = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
                local_files_only=True, trust_remote_code=False)
            model, tokenizer = load(str(MODEL), tokenizer_config={
                'local_files_only': True, 'trust_remote_code': False})
            model.eval()
            for request in requests[len(rows):]:
                output = generate_one(model, tokenizer, request)
                checks = mechanical_checks(request, output, student)
                append(path, {'candidate_id': request['candidate_id'],
                    'original_candidate_id': request['original_candidate_id'],
                    'source_id': request['source_id'], **output,
                    'mechanical_checks': checks, 'engineering_only': True})
                print(request['candidate_id'], checks['hard_rejection_reasons'], flush=True)
    rows = checked_rows(path)
    save(FOLDER / 'report.json', {'at_utc': now(), 'completed': len(rows),
        'mechanical_pass': sum(not r['mechanical_checks']['hard_rejection_reasons'] for r in rows),
        'semantic_review_pending': True, 'scientific_pool_exported': False,
        'automatic_adoption': False, 'method_effectiveness_established': False,
        'public_test_used': False, 'plan_sha256': baseline.sha(FOLDER / 'plan.json'),
        'generated_sha256': baseline.sha(path), 'total_seconds': sum(r['seconds'] for r in rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'download', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'download': download, 'run': run}[args.action]()
