"""Full G2/G3 rank-zero explanations, with one shared generation per prompt.

No variants or trial results are prerequisites. Mechanical checks only reject;
an explicit exact-output semantic review is required for every usable output.
This module does not grant formal-training or public-test permission.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, BASE, read, checked_rows, freeze, save,
    append, digest, baseline, signature, gpu, load_model, check_time, now)
from budgeted_materials import PROTOCOL
from full_selector import (FOLDER as SELECTORS, load_prepared, load_quality,
                           validate_resume)
from explanation_pool import (TYPES, unique, make_request, reusable_output,
    generate_one, mechanical_checks, reviewed_candidate, selector_paths)
from auxiliary_explainer_recipe import (FULL, RECIPE, READY, RECIPE_ID, MODEL, DIAGNOSTIC,
    load_recipe, diagnostic_registry, verify_inputs)

FOLDER = FULL
SELECTORS_NAMES = ('joint', 'nll')


def request_views(sources, teachers, cases, diagnostics):
    """Rank zero is the fixed G2/G3 policy, including independent NLL roots.

    The second joint position is not needed for G3; G4 must add it later on its
    frozen 128 roots, preserving its original up-to-four-explanation trial pool.
    """
    ids = [case['source_id'] for case in cases]
    if (len(ids) != len(set(ids)) or set(sources) != set(ids)
            or set(teachers) != set(ids) or set(diagnostics) != set(SELECTORS_NAMES)):
        raise ValueError('Explanation inputs must cover the same accepted training roots')
    views = {}
    for selector in SELECTORS_NAMES:
        rows = diagnostics[selector]
        if len(rows) != len(cases):
            raise ValueError('All full-train position diagnostics must finish first')
        validate_resume(rows, cases, selector)
        requests = []
        for case, diagnostic in zip(cases, rows):
            sid = case['source_id']
            steps = {c['step']['step_index']: c['step'] for c in case['candidates']}
            for index in diagnostic['proposal']['positions'][:1]:
                for kind in TYPES:
                    request = make_request(sources[sid], teachers[sid], steps[index], kind, 0)
                    request.update(development_only=False, protocol=PROTOCOL,
                                   phase='full_train_fixed_explanation')
                    requests.append(request)
        views[selector] = requests
    return views


def canonical_requests(views):
    """Same position/type/prompt has exactly one immutable teacher attempt."""
    result = {}
    for selector in SELECTORS_NAMES:
        for request in views[selector]:
            cid = request['candidate_id']
            if cid in result:
                # Reuse the existing exact prompt comparison, including all step
                # boundaries, without generating any synthetic teacher output.
                reusable_output(request, {cid: result[cid]}, {cid: {}})
            else:
                result[cid] = request
    return list(result.values())


def full_inputs():
    manifest, cases = load_prepared()
    sources, teachers, quality, ids = load_quality(ROOT / manifest['quality_file'])
    selected = set(ids)
    sources = unique([s for s in sources if s['source_id'] in selected], 'source_id')
    teachers = unique([t for t in teachers if t['source_id'] in selected], 'source_id')
    diagnostics = {s: checked_rows(SELECTORS / f'positions_{s}_v1.jsonl')
                   for s in SELECTORS_NAMES}
    views = request_views(sources, teachers, cases, diagnostics)
    return manifest, cases, sources, teachers, diagnostics, views


def reuse_entry(request, old_requests, old_outputs):
    """Return an earlier exact prompt's signed output, never a new attempt."""
    reused = reusable_output(request, old_requests, old_outputs)
    if reused is not None and digest(reused['output']) != reused['output_sha256']:
        raise ValueError('Earlier explanation output hash changed')
    return reused


def previous_generation_registry(requests):
    return diagnostic_registry(requests)


def load_reused_outputs(manifest, requests):
    cache = {}; result = {}
    for request in requests:
        cid = request['candidate_id']
        entry = manifest.get('previous_generation_registry', {}).get(cid)
        if entry is None:
            continue
        if (entry.get('recipe_id') != RECIPE_ID
                or entry.get('requests_file') != str((DIAGNOSTIC / 'requests.json').relative_to(ROOT))
                or entry.get('generated_file') != str((DIAGNOSTIC / 'generated.jsonl').relative_to(ROOT))):
            raise ValueError('An explanation output from a different teacher recipe cannot be reused')
        key = (entry['requests_file'], entry['generated_file'])
        if key not in cache:
            cache[key] = (unique(read(ROOT / key[0]), 'candidate_id'),
                          unique(checked_rows(ROOT / key[1]), 'candidate_id'))
        old_requests, old_outputs = cache[key]
        if old_outputs[cid]['record_sha256'] != entry['record_sha256']:
            raise ValueError('Earlier explanation record signature changed')
        result[cid] = reuse_entry(request, old_requests, old_outputs)
        result[cid]['reuse_source'] = entry
    return result


def prepare():
    recipe = load_recipe()
    ready = read(READY); verify_inputs(ready)
    if (ready.get('recipe_id') != RECIPE_ID
            or ready.get('ready_for_full_reviewed_generation') is not True
            or ready.get('training_enabled') is not False):
        raise ValueError('Current auxiliary recipe needs actual material readiness')
    parent, cases, _, _, _, views = full_inputs()
    requests = canonical_requests(views)
    for selector in SELECTORS_NAMES:
        freeze(FOLDER / f'requests_{selector}.json', views[selector])
    freeze(FOLDER / 'requests.json', requests)
    paths = [Path(__file__), HERE / 'explanation_pool.py', HERE / 'full_selector.py',
        HERE / 'positions_nll.py', HERE / 'positions_v2.py', HERE / 'budgeted_materials.py',
        HERE / 'runtime.py', HERE / 'runtime_full.py',
        ROOT / 'experiments/math_distill_v1/config.json',
        ROOT / parent['quality_file'], SELECTORS / 'cases_manifest.json',
        FOLDER / 'requests.json', RECIPE, READY, HERE / 'auxiliary_explainer_recipe.py']
    paths += [SELECTORS / f'positions_{s}_v1.jsonl' for s in SELECTORS_NAMES]
    paths += [FOLDER / f'requests_{s}.json' for s in SELECTORS_NAMES]
    registry, prior_inputs = previous_generation_registry(requests)
    manifest = {'schema': 'tgrd_full_fixed_explanation_requests_v1', 'protocol': PROTOCOL,
        'recipe_id': recipe['recipe_id'], 'recipe_file': str(RECIPE.relative_to(ROOT)),
        'accepted_root_ids': [c['source_id'] for c in cases], 'candidate_root_count': 7473,
        'inputs': {**{str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}, **prior_inputs},
        'previous_generation_registry': registry,
        'request_count': len(requests),
        'selector_requests': {s: len(views[s]) for s in SELECTORS_NAMES},
        'duplicate_teacher_calls_avoided': sum(map(len, views.values())) - len(requests),
        'position_ranks': [0], 'explanation_types': list(TYPES),
        'one_attempt_no_retries': True, 'variant_dependency': False,
        'G4_second_position': 'separate_frozen128_extension_required',
        'public_test_used': False}
    freeze(FOLDER / 'requests_manifest.json', manifest)
    return manifest


def load_requests():
    load_recipe()
    manifest = read(FOLDER / 'requests_manifest.json')
    if (manifest.get('schema') != 'tgrd_full_fixed_explanation_requests_v1'
            or manifest.get('protocol') != PROTOCOL
            or manifest.get('candidate_root_count') != 7473
            or manifest.get('public_test_used') is not False
            or manifest.get('one_attempt_no_retries') is not True
            or manifest.get('position_ranks') != [0]
            or manifest.get('recipe_id') != RECIPE_ID
            or manifest.get('recipe_file') != str(RECIPE.relative_to(ROOT))):
        raise ValueError('Expected the frozen full-train fixed-explanation plan')
    for name, checksum in manifest['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Frozen explanation input changed: ' + name)
    _, cases, sources, teachers, diagnostics, views = full_inputs()
    requests = canonical_requests(views)
    if (manifest['accepted_root_ids'] != [c['source_id'] for c in cases]
            or read(FOLDER / 'requests.json') != requests
            or manifest['request_count'] != len(requests)
            or any(read(FOLDER / f'requests_{s}.json') != views[s] for s in SELECTORS_NAMES)):
        raise ValueError('Full explanation requests no longer reconstruct from diagnostics')
    return manifest, cases, sources, teachers, diagnostics, views, requests


def validate_generated(saved, requests, tokenizer, *, complete=False, recipe_id=None):
    if len(saved) > len(requests) or (complete and len(saved) != len(requests)):
        raise ValueError('Explanation generation coverage is incomplete or excessive')
    for row, request in zip(saved, requests):
        if (row.get('candidate_id') != request['candidate_id']
                or row.get('source_id') != request['source_id']
                or row.get('protocol') != PROTOCOL
                or (recipe_id is not None and row.get('recipe_id') != recipe_id)
                or row.get('request_sha256') != digest(request['system'] + '\n' + request['user'])
                or row.get('mechanical_checks') != mechanical_checks(request, row, tokenizer)):
            raise ValueError('Generated explanation order, prompt or rejection evidence changed')


def student_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
        local_files_only=True, trust_remote_code=False)


def generate():
    manifest, _, _, _, _, _, requests = load_requests()
    tokenizer = student_tokenizer()
    path = FOLDER / 'generated.jsonl'
    saved = checked_rows(path)
    validate_generated(saved, requests, tokenizer, recipe_id=RECIPE_ID)
    reused_outputs = load_reused_outputs(manifest, requests)
    if len(saved) == len(requests):
        if not path.exists():
            path.touch()
        return
    with gpu():
        signature('full_fixed_explanations_general7b_v2',
            [ROOT / p for p in manifest['inputs']] + [FOLDER / 'requests_manifest.json'],
            {'protocol': PROTOCOL, 'temperature': 0, 'max_new_tokens': 256,
             'recipe_id': RECIPE_ID, 'one_attempt_no_retries': True,
             'same_prompt_shared_across_selectors': True})
        load_recipe(verify_model=True)
        model = teacher_tokenizer = None
        for request in requests[len(saved):]:
            check_time()
            output = reused_outputs.get(request['candidate_id'])
            if output is None:
                if model is None:
                    import mlx.core as mx
                    from mlx_lm import load
                    mx.set_memory_limit(14 * 1024**3); mx.set_cache_limit(512 * 1024**2)
                    model, teacher_tokenizer = load(str(MODEL), tokenizer_config={
                        'local_files_only': True, 'trust_remote_code': False})
                output = generate_one(model, teacher_tokenizer, request)
            row = {**output, 'candidate_id': request['candidate_id'], 'source_id': request['source_id'],
                'protocol': PROTOCOL, 'recipe_id': RECIPE_ID,
                'engineering_only': False, 'full_text_review_pending': True,
                'explicit_recipe_adoption': str(RECIPE.relative_to(ROOT)),
                'request_sha256': digest(request['system'] + '\n' + request['user']),
                'current_full_run_generation_seconds': (0 if request['candidate_id'] in reused_outputs
                                                        else output['seconds']),
                'mechanical_checks': mechanical_checks(request, output, tokenizer)}
            saved.append(append(path, row))
            save(FOLDER / 'generation_status.json', {'at_utc': now(),
                'completed': len(saved), 'total': len(requests),
                'full_text_review_pending': True, 'formal_training_started': False,
                'public_test_used': False})
            print(len(saved), '/', len(requests), request['candidate_id'],
                  row['mechanical_checks']['hard_rejection_reasons'] or 'review_pending', flush=True)


def assemble_views(sources, teachers, cases, diagnostics, views, generated, reviews, tokenizer):
    """Only the applicable selector sees a candidate, even though text is shared."""
    requests = canonical_requests(views)
    validate_generated(generated, requests, tokenizer, complete=True)
    outputs = unique(generated, 'candidate_id')
    review_byid = unique(reviews, 'candidate_id')
    if not set(review_byid) <= set(outputs):
        raise ValueError('Review refers to a candidate outside this generation plan')
    for cid, review in review_byid.items():
        if review.get('output_sha256') != outputs[cid]['output_sha256']:
            raise ValueError('Review refers to a different exact teacher output')
    result = {}
    for selector in SELECTORS_NAMES:
        pools = {}
        for case, diagnostic in zip(cases, diagnostics[selector]):
            sid = case['source_id']; teacher = teachers[sid]
            pools[sid] = {'source_id': sid, 'question_sha256': sources[sid]['question_sha256'],
                'teacher_output_sha256': teacher['output_sha256'], 'candidates': [{
                    'candidate_id': 'e0', 'completion': teacher['output'], 'student_tokens': 0,
                    'completion_sha256': teacher['output_sha256'], 'explanation_type': 'empty'}],
                'rejected': [], 'development_only': False, 'protocol': PROTOCOL,
                'selector': 'nll_only' if selector == 'nll' else 'joint',
                'position_proposal': diagnostic['proposal'], 'position_ranks_generated': [0],
                'G4_full_trial_pool': False}
        for request in views[selector]:
            cid = request['candidate_id']; sid = request['source_id']
            candidate, reasons = reviewed_candidate(request, outputs[cid], review_byid.get(cid),
                                                    sources[sid], teachers[sid], tokenizer)
            if candidate is None:
                pools[sid]['rejected'].append({'candidate_id': cid, 'reasons': reasons})
            else:
                pools[sid]['candidates'].append(candidate)
        result[selector] = list(pools.values())
    return result


def assemble():
    _, cases, sources, teachers, diagnostics, views, _ = load_requests()
    review_path = FOLDER / 'full_text_reviews.json'
    reviews = read(review_path) if review_path.exists() else []
    generated = checked_rows(FOLDER / 'generated.jsonl')
    validate_generated(generated, canonical_requests(views), student_tokenizer(),
                       complete=True, recipe_id=RECIPE_ID)
    pools = assemble_views(sources, teachers, cases, diagnostics, views, generated,
                           reviews, student_tokenizer())
    for selector in SELECTORS_NAMES:
        freeze(FOLDER / f'reviewed_pools_{selector}.json', pools[selector])
    paths = [FOLDER / 'requests_manifest.json', FOLDER / 'generated.jsonl']
    if review_path.exists():
        paths.append(review_path)
    freeze(FOLDER / 'reviewed_pools_manifest.json', {
        'schema': 'tgrd_full_fixed_explanation_pools_v1', 'protocol': PROTOCOL,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'pools': {s: {'path': str((FOLDER / f'reviewed_pools_{s}.json').relative_to(ROOT)),
                     'sha256': baseline.sha(FOLDER / f'reviewed_pools_{s}.json')} for s in SELECTORS_NAMES},
        'accepted_root_ids': [c['source_id'] for c in cases],
        'all_mechanically_usable_outputs_semantically_reviewed': True,
        'G4_full_trial_pool': False, 'formal_training_enabled': False, 'public_test_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'generate', 'assemble'))
    args = parser.parse_args()
    result = {'prepare': prepare, 'generate': generate, 'assemble': assemble}[args.action]()
    if result is not None:
        print({k: result[k] for k in ('request_count', 'duplicate_teacher_calls_avoided')}, flush=True)
