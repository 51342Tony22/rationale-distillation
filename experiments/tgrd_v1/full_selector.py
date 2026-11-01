"""Full accepted-train position diagnostics, without a variant dependency.

Requires an explicitly frozen 7473-root eligibility decision set. This module never
creates that evidence, loads public test data, or enables formal training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, DATA, BASE, read, checked_rows,
    freeze, save, digest, baseline, signature, gpu, load_model, check_time, append)
from budgeted_materials import PROTOCOL
from positions_v2 import genuine_transition, usable_evidence, propose
from positions_nll import score_case, propose_nll
from selector_masking_probe import prepare_case, MLXBackend
from selector_masking_core import diagnose_step


FOLDER = OUT / 'full_selector_v1'
QUALITY = OUT / 'gates/full_teacher_quality_frozen_v1.json'
TEACHERS = OUT / 'materials/teacher_targets_train7473.jsonl'
SOURCE = DATA / 'train7473.json'


def validate_quality(sources, teachers, quality):
    from automatic_teacher_quality import SCHEMA as AUTO_SCHEMA, validate as validate_automatic
    if quality.get('schema') == AUTO_SCHEMA:
        return validate_automatic(sources, teachers, quality)
    ids = [s['source_id'] for s in sources]
    expected = {f'gsm8k:train:{i}' for i in range(7473)}
    if (len(ids) != 7473 or set(ids) != expected
            or [t['source_id'] for t in teachers] != ids
            or quality.get('schema') != 'tgrd_full_teacher_quality_v1'
            or quality.get('protocol') != PROTOCOL
            or quality.get('status') != 'frozen'
            or quality.get('candidate_root_count') != 7473
            or quality.get('public_test_used') is not False
            or not quality.get('review_policy')
            or not quality.get('inputs')):
        raise ValueError('Complete official training candidates and frozen quality policy required')
    reviews = quality['reviews']
    if (len(reviews) != 7473 or len({r['source_id'] for r in reviews}) != 7473
            or {r['source_id'] for r in reviews} != expected):
        raise ValueError('Every candidate needs its own actual full-text quality decision')
    byid = {r['source_id']: r for r in reviews}
    accepted = []
    for source, teacher in zip(sources, teachers):
        sid = source['source_id']; review = byid[sid]
        if (digest(source['question']) != source['question_sha256']
                or digest(teacher['output']) != teacher['output_sha256']
                or teacher['question_sha256'] != source['question_sha256']
                or teacher['gold'] != source['gold']
                or type(review.get('accepted')) is not bool
                or review.get('all_text_read') is not True
                or review.get('review_kind') != 'assistant_full_text_mathematical_review'
                or not review.get('reason')
                or review.get('question_sha256') != source['question_sha256']
                or review.get('output_sha256') != teacher['output_sha256']):
            raise ValueError('Teacher/source/full-text review provenance mismatch: ' + sid)
        if review['accepted']:
            if teacher.get('correct') is not True or teacher.get('finish_reason') != 'stop':
                raise ValueError('Accepted source must preserve the original gold and complete teacher output')
            accepted.append(sid)
    if not accepted or quality.get('accepted_root_ids') != accepted:
        raise ValueError('Accepted roots must retain the common frozen source order')
    return accepted


def active_quality_path():
    from automatic_teacher_quality import AMENDMENT, GATE, MODE
    if AMENDMENT.exists():
        amendment = read(AMENDMENT)
        if (amendment.get('protocol') != PROTOCOL or amendment.get('quality_mode') != MODE
                or amendment.get('all_candidate_full_text_review_required') is not False):
            raise ValueError('Unknown quality workflow amendment')
        return GATE
    return QUALITY


def quality_decisions(quality):
    from automatic_teacher_quality import SCHEMA as AUTO_SCHEMA
    return quality['decisions'] if quality.get('schema') == AUTO_SCHEMA else quality['reviews']


def load_quality(path=None):
    path = active_quality_path() if path is None else path
    path = Path(path)
    quality = read(path)
    required = {str(p.relative_to(ROOT)) for p in (
        SOURCE, TEACHERS, OUT / 'signatures/teacher_full7473.json',
        OUT / 'transfer128_amendment_v1.json')}
    if not required <= set(quality.get('inputs', {})):
        raise ValueError('Quality freeze must bind the full source, teacher and current authorization')
    for name, checksum in quality['inputs'].items():
        # No public-test dataset is a legitimate quality-freeze dependency.
        if name.startswith('data/') and name not in {
                str(SOURCE.relative_to(ROOT)), 'data/tgrd_v1/full_gsm8k_manifest.json'}:
            raise ValueError('Only official training inputs belong in this quality freeze')
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Frozen quality input changed: ' + name)
    policy = quality['review_policy']
    if (not policy.get('path') or not policy.get('sha256')
            or quality['inputs'].get(policy['path']) != policy['sha256']):
        raise ValueError('A concrete quality policy file and hash are required')
    sources, teachers = read(SOURCE), checked_rows(TEACHERS)
    ids = validate_quality(sources, teachers, quality)
    return sources, teachers, quality, ids


def prepare_training_case(source, teacher, tokenizer):
    """Use exactly the existing v2 equation/evidence filters for both selectors."""
    completion = teacher['output']
    row = {'source_id': source['source_id'],
        'prompt': BASE['student_prompt'].format(question=source['question']),
        'completion': completion, 'completion_sha256': teacher['output_sha256']}
    row['prompt_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
    row['target_ids'] = tokenizer.encode(completion, add_special_tokens=False)
    # Formal completions also need EOS and may never be silently truncated.
    if len(row['prompt_ids']) + len(row['target_ids']) + 1 > 4096:
        raise ValueError('Reviewed ordinary CoT exceeds the untruncated training context')
    case = prepare_case(row, source, tokenizer)
    kept = []
    for candidate in case['candidates']:
        if genuine_transition(candidate['step']['text']):
            candidate['units'] = [u for u in candidate['units']
                                  if usable_evidence(u['text'], u['kind'])]
            kept.append(candidate)
        else:
            case['omitted_candidates'].append({**candidate['step'],
                'reason': 'v2_pure_assignment_or_nontransition'})
    case['candidates'] = kept
    case['candidate_rule_version'] = 'v2_stricter_equation_heuristic'
    return case


def prepare(quality_path=None):
    quality_path = (ROOT / (active_quality_path() if quality_path is None else quality_path)).resolve()
    sources, teachers, quality, ids = load_quality(quality_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
        local_files_only=True, trust_remote_code=False)
    selected = set(ids)
    cases = [prepare_training_case(s, t, tokenizer)
             for s, t in zip(sources, teachers) if s['source_id'] in selected]
    path = FOLDER / 'cases.json'
    freeze(path, cases)
    inputs = [Path(__file__), HERE / 'positions_v2.py', HERE / 'positions_nll.py',
        HERE / 'budgeted_materials.py', HERE / 'runtime.py', HERE / 'runtime_full.py',
        HERE.parent / 'math_distill_v1/selector_masking_probe.py',
        HERE.parent / 'math_distill_v1/selector_masking_core.py',
        HERE.parent / 'math_distill_v1/pilot_localize_v2.py',
        ROOT / 'experiments/math_distill_v1/config.json', Path(quality_path), path]
    manifest = {'schema': 'tgrd_full_selector_cases_v1', 'protocol': PROTOCOL,
        'candidate_root_count': 7473, 'accepted_root_ids': ids,
        'accepted_root_count': len(cases), 'cases_sha256': baseline.sha(path),
        'quality_file': str(Path(quality_path).relative_to(ROOT)),
        'quality_file_sha256': baseline.sha(quality_path),
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in inputs},
        'variant_dependency': False, 'public_test_used': False,
        'candidate_steps': sum(len(c['candidates']) for c in cases),
        'zero_candidate_roots': sum(not c['candidates'] for c in cases)}
    freeze(FOLDER / 'cases_manifest.json', manifest)
    return manifest


def load_prepared():
    manifest = read(FOLDER / 'cases_manifest.json')
    if (manifest.get('schema') != 'tgrd_full_selector_cases_v1'
            or manifest.get('protocol') != PROTOCOL
            or manifest.get('candidate_root_count') != 7473
            or manifest.get('public_test_used') is not False
            or manifest.get('variant_dependency') is not False):
        raise ValueError('Expected the reviewed full-train selector manifest')
    for name, checksum in manifest['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Frozen full-selector input changed: ' + name)
    if (manifest['inputs'].get(manifest['quality_file']) != manifest['quality_file_sha256']
            or baseline.sha(ROOT / manifest['quality_file']) != manifest['quality_file_sha256']):
        raise ValueError('Quality manifest identity differs from frozen selector inputs')
    _, _, _, ids = load_quality(ROOT / manifest['quality_file'])
    cases = read(FOLDER / 'cases.json')
    if ([c['source_id'] for c in cases] != ids or manifest['accepted_root_ids'] != ids
            or manifest['accepted_root_count'] != len(cases)
            or baseline.sha(FOLDER / 'cases.json') != manifest['cases_sha256']):
        raise ValueError('Full selector common-root coverage changed')
    return manifest, cases


def validate_resume(saved, cases, selector):
    if selector not in ('joint', 'nll') or len(saved) > len(cases):
        raise ValueError('Unknown selector or invalid resumed coverage')
    for row, case in zip(saved, cases):
        indices = [c['step']['step_index'] for c in case['candidates']]
        if (row['source_id'] != case['source_id']
                or row['teacher_output_sha256'] != case['teacher_output_sha256']
                or row.get('protocol') != PROTOCOL or row.get('selector') != selector
                or [s['step_index'] for s in row['steps']] != indices):
            raise ValueError('Full selector resume source/candidate coverage changed')
        expected = propose_nll(row['steps']) if selector == 'nll' else propose(row['steps'])
        if row['proposal'] != expected:
            raise ValueError('Saved full selector proposal no longer reproduces')


def run(selector):
    manifest, cases = load_prepared()
    path = FOLDER / f'positions_{selector}_v1.jsonl'
    saved = checked_rows(path)
    validate_resume(saved, cases, selector)
    if len(saved) == len(cases):
        return
    with gpu():
        signature('full_selector_' + selector + '_v1',
            [ROOT / p for p in manifest['inputs']] + [FOLDER / 'cases_manifest.json'],
            {'selector': selector, 'protocol': PROTOCOL, 'all_accepted_roots': True,
             'queries': 8, 'layers': 4, 'random_controls': 3, 'diagnostic_seed': 7})
        baseline.verify_student(BASE)
        model, _ = load_model('student')
        backend = MLXBackend(model, 4)
        import mlx.core as mx
        for case in cases[len(saved):]:
            check_time(); start = time.perf_counter()
            if selector == 'nll':
                row = score_case(case, backend)
            else:
                steps = []
                for candidate in case['candidates']:
                    steps.append(diagnose_step(case['input_ids'], candidate['step'],
                        candidate['units'], backend, seed=7,
                        key=f'{case["source_id"]}:{candidate["step"]["step_index"]}',
                        controls=3, queries=8))
                    mx.clear_cache()
                row = {'source_id': case['source_id'], 'steps': steps,
                    'teacher_output_sha256': case['teacher_output_sha256'],
                    'proposal': propose(steps)}
            row.update(protocol=PROTOCOL, selector=selector, development_only=False,
                phase='full_train_material_selection', seconds=time.perf_counter() - start)
            saved.append(append(path, row))
            save(FOLDER / f'progress_{selector}.json', {'completed': len(saved),
                'accepted_total': len(cases), 'candidate_root_count': 7473,
                'selector': selector, 'formal_training_started': False,
                'public_test_used': False})
            mx.clear_cache()
            print(selector, len(saved), '/', len(cases), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--quality', type=Path)
    diag = sub.add_parser('run'); diag.add_argument('--selector', choices=('nll', 'joint'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.quality)
        print(json.dumps({k: result[k] for k in (
            'accepted_root_count', 'candidate_steps', 'zero_candidate_roots')}), flush=True)
    else:
        run(args.selector)
