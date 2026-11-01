"""Shared explanation pool: bounded prompts, immutable outputs, review before use.

Development bootstrap only. Generation does not imply mathematical acceptance.
This runner deliberately has its own explanation-specific system prompt rather
than the main teacher prompt which asks for a complete solution and final answer.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from dataclasses import replace

from runtime_full import (ROOT, HERE, OUT, DATA, BASE, baseline, read, freeze, save,
                          append, checked_rows, digest, signature, gpu, load_model,
                          check_time, now)
from calibration_cohort import COHORTS, cohort_paths

SYSTEM = (
    'Explain one already stated step in a mathematical solution. '
    'Return only a short explanation in one or two sentences. '
    'Do not solve the rest of the problem, give a final answer, introduce new '
    'numerical results, or repeat a full chain of thought. Explain the relation '
    'or condition in words; do not add headings or answer markers.'
)
TYPES = ('relation', 'contrastive')
INSTRUCTIONS = {
    'relation': 'Explain why this relation or operation follows from the stated quantities and conditions.',
    'contrastive': 'Explain why this operation applies and why one plausible alternative operation would not apply. Do not invent a new example.',
}
REVIEW_CHECKS = ('mathematics_valid', 'units_and_entities_valid', 'explains_selected_step',
                 'no_new_result_or_final_answer', 'no_circular_explanation',
                 'preserves_original_meaning', 'one_or_two_sentences')
MIN_TOKENS, MAX_TOKENS = 32, 64
FOLDER = OUT / 'calibration/shared_pool_v1'


def selector_paths(cohort,selector='joint'):
    paths=cohort_paths(cohort)
    if selector=='joint':return paths
    if selector!='nll':raise ValueError('Unknown position selector')
    from positions_nll import diagnostic_path
    return replace(paths,pool_folder=paths.pool_folder.parent/'nll_pool_v1',
        diagnostics=diagnostic_path(cohort),signature_prefix=paths.signature_prefix+'_nll')


def reusable_output(request,other_requests,other_generated):
    """Identical step/prompt gets one teacher output across both selectors."""
    cid=request['candidate_id'];old=other_requests.get(cid)
    if old is None or cid not in other_generated:return None
    for key in ('source_id','question_sha256','teacher_output_sha256','step','system','user'):
        if old[key]!=request[key]:raise ValueError('Overlapping candidate prompt changed: '+cid)
    row=dict(other_generated[cid]);row.pop('record_sha256',None)
    return row


def make_request(source, teacher, step, explanation_type, position_rank):
    """Only the question, *preceding* text and current step enter the user prompt."""
    if explanation_type not in TYPES:
        raise ValueError('Unknown explanation type')
    text = teacher['output']
    start, end = step['char_start'], step['char_end']
    if not (0 <= start < end <= len(text)) or text[start:end] != step['text']:
        raise ValueError('Step character boundary does not match the reviewed teacher')
    if teacher['question_sha256'] != source['question_sha256'] or digest(text) != teacher['output_sha256']:
        raise ValueError('Teacher/source hashes do not match')
    prefix = text[:start]
    visible_context = ('Problem:\n' + source['question'] + '\n\nSolution before this step:\n' +
                       prefix + '\n\nStep to explain:\n' + step['text'])
    user = (visible_context + '\n\nTask:\n' +
            INSTRUCTIONS[explanation_type] +
            '\nUse approximately 32 to 64 student tokens. Return only the explanation.')
    return {
        'candidate_id': f'{source["source_id"]}:step{step["step_index"]}:{explanation_type}',
        'source_id': source['source_id'], 'question_sha256': source['question_sha256'],
        'teacher_output_sha256': teacher['output_sha256'], 'step': step,
        'position_rank': position_rank, 'explanation_type': explanation_type,
        'system': SYSTEM, 'user': user, 'visible_context': visible_context, 'prefix_sha256': digest(prefix),
        'prompt_boundary': 'question_plus_prior_prefix_plus_current_step_only',
        'development_only': True,
    }


def mechanical_checks(request, generated, tokenizer):
    """Rejection checks only; passing them never establishes explanation quality."""
    raw = generated['output']
    if digest(raw) != generated['output_sha256']:
        raise ValueError('Generated output hash mismatch')
    text = raw.strip()
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    reasons = []
    if generated['finish_reason'] != 'stop':
        reasons.append('not_a_complete_stop')
    if not MIN_TOKENS <= tokens <= MAX_TOKENS:
        reasons.append('outside_32_to_64_student_tokens')
    if re.search(r'\\boxed|####|\bfinal answer\b|\bthe answer is\b', text, re.I):
        reasons.append('answer_marker')
    # Catch new written numbers; this does not catch semantic result leakage.
    visible = set(re.findall(r'\d+(?:\.\d+)?', request['visible_context']))
    if set(re.findall(r'\d+(?:\.\d+)?', text)) - visible:
        reasons.append('new_numeric_literal_requires_rejection')
    if not text:
        reasons.append('empty_output')
    return {'text': text, 'text_sha256': digest(text), 'student_tokens': tokens,
            'hard_rejection_reasons': reasons, 'requires_full_text_review': not reasons}


def insert_explanation(original, step, text):
    end = step['char_end']
    if original[step['char_start']:end] != step['text']:
        raise ValueError('Insertion boundary changed')
    # Insertion is after the selected reference step, before any later step.
    return original[:end] + '\nExplanation: ' + text + '\n' + original[end:]


def reviewed_candidate(request, generated, review, source, teacher, tokenizer):
    checks = mechanical_checks(request, generated, tokenizer)
    if checks['hard_rejection_reasons']:
        return None, checks['hard_rejection_reasons']
    if review is None:
        raise ValueError('Unreviewed candidate cannot enter any selection method')
    if review['candidate_id'] != request['candidate_id'] or review['output_sha256'] != generated['output_sha256']:
        raise ValueError('Review does not bind this exact output')
    if review.get('kind') != 'assistant_mathematical_review' or not review.get('reason'):
        raise ValueError('Explicit full-text mathematical review required')
    if review.get('accepted') is not True:
        if review.get('accepted') is not False:
            raise ValueError('Review needs a boolean decision')
        return None, ['full_text_review_rejected', review['reason']]
    if any(review.get(key) is not True for key in REVIEW_CHECKS):
        raise ValueError('Accepted review lacks a required semantic/leakage check')
    if (request['teacher_output_sha256'] != teacher['output_sha256']
            or digest(teacher['output']) != teacher['output_sha256']
            or source['source_id'] != request['source_id']
            or digest(source['question']) != request['question_sha256']):
        raise ValueError('Reviewed candidate root changed')
    completion = insert_explanation(teacher['output'], request['step'], checks['text'])
    prompt = BASE['student_prompt'].format(question=source['question'])
    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False)) + 1
    if len(tokenizer.encode(prompt, add_special_tokens=False)) + completion_tokens > 4096:
        return None, ['training_context_overflow_no_truncation']
    return {
        **{key: request[key] for key in ('candidate_id', 'source_id', 'explanation_type', 'position_rank')},
        'step_index': request['step']['step_index'], 'insertion': 'after_selected_step',
        'text': checks['text'], 'text_sha256': checks['text_sha256'],
        'student_tokens': checks['student_tokens'], 'completion': completion,
        'completion_sha256': digest(completion), 'completion_tokens_including_eos': completion_tokens,
        'teacher_output_sha256': teacher['output_sha256'], 'review': review,
    }, []


def unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError('Duplicate ' + key)
        result[row[key]] = row
    return result


def inputs():
    return (unique(read(DATA/'cal40.json'), 'source_id'),
            unique(checked_rows(OUT/'materials/teacher_targets_cal40.jsonl'), 'source_id'))


def prepare(cohort='bootstrap10',selector='joint'):
    paths = selector_paths(cohort,selector)
    sources, teachers = inputs()
    cases = read(paths.cases)
    diagnostics = unique(checked_rows(paths.diagnostics), 'source_id')
    reviews = unique(read(paths.reviews), 'source_id')
    if len(cases) != paths.expected_roots:
        raise ValueError('Unexpected calibration case coverage')
    if set(diagnostics) != {case['source_id'] for case in cases}:
        raise ValueError('Position diagnostic stage is not complete')
    requests = []
    for case in cases:
        sid = case['source_id']; teacher = teachers[sid]
        if (not reviews[sid]['accepted'] or reviews[sid]['output_sha256'] != teacher['output_sha256']
                or case['teacher_output_sha256'] != teacher['output_sha256']
                or diagnostics[sid]['teacher_output_sha256'] != teacher['output_sha256']):
            raise ValueError('Teacher review/diagnostic provenance mismatch')
        steps = {candidate['step']['step_index']: candidate['step'] for candidate in case['candidates']}
        if selector=='nll':
            from positions_nll import propose_nll
            diagnostic=diagnostics[sid]
            if (set(steps)!={s['step_index'] for s in diagnostic['steps']}
                    or diagnostic['proposal']!=propose_nll(diagnostic['steps'])):
                raise ValueError('NLL must score ALL eligible steps, independently of joint proposals')
        positions = diagnostics[sid]['proposal']['positions']
        if len(positions) > 2 or len(set(positions)) != len(positions):
            raise ValueError('Proposal must contain at most two distinct positions')
        for rank, index in enumerate(positions):
            for kind in TYPES:
                requests.append(make_request(sources[sid], teacher, steps[index], kind, rank))
    freeze(paths.pool_folder/'requests.json', requests)
    return requests


def generate_one(model, tokenizer, request):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': request['system']},
        {'role': 'user', 'content': request['user']}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) + 256 > 4096:
        return {'output': '', 'output_sha256': digest(''), 'prompt_sha256': digest(prompt),
                'prompt_tokens': len(ids), 'generation_tokens': 0, 'seconds': 0,
                'finish_reason': 'context_overflow_not_generated', 'peak_mlx_memory_bytes': 0}
    text = ''; last = None; start = time.perf_counter()
    model.eval()
    for part in stream_generate(model, tokenizer, ids, max_tokens=256, sampler=make_sampler(temp=0.0)):
        check_time()
        last = part; text += part.text
    if last is None:
        raise ValueError('No teacher generation result')
    return {'output': text, 'output_sha256': digest(text), 'prompt_sha256': digest(prompt),
            'prompt_tokens': len(ids), 'generation_tokens': last.generation_tokens,
            'finish_reason': last.finish_reason, 'seconds': time.perf_counter()-start,
            'peak_mlx_memory_bytes': mx.get_peak_memory()}


def generate_pool(cohort='bootstrap10',selector='joint'):
    paths = selector_paths(cohort,selector); folder = paths.pool_folder
    requests = prepare(cohort,selector)
    signature(paths.signature_prefix+'_shared_pool_v1', [Path(__file__), HERE/'runtime_full.py', HERE/'runtime.py',
        HERE/'calibration_cohort.py',HERE/'positions_nll.py', folder/'requests.json', paths.diagnostics, paths.reviews],
        {'temperature': 0, 'max_new_tokens': 256, 'student_token_range': [32,64],
         'one_attempt_no_retries': True, 'requires_semantic_review_after_generation': True})
    path = folder/'generated.jsonl'; saved = checked_rows(path)
    if [r['candidate_id'] for r in saved] != [r['candidate_id'] for r in requests[:len(saved)]]:
        raise ValueError('Resume candidate order mismatch')
    if len(saved) == len(requests):
        return
    other=selector_paths(cohort,'nll' if selector=='joint' else 'joint').pool_folder
    other_requests=unique(read(other/'requests.json'),'candidate_id') if (other/'requests.json').exists() else {}
    other_generated=unique(checked_rows(other/'generated.jsonl'),'candidate_id')
    with gpu():
        signed = read(ROOT/'output/math_distill_v1/dev_baseline/teacher_reference/signature.json')
        for name, spec in signed['model_files'].items():
            if baseline.sha(ROOT/BASE['teacher_dir']/name) != spec['sha256']:
                raise ValueError('Teacher weights changed')
        from transformers import AutoTokenizer
        student_tokenizer = AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']), local_files_only=True,
                                                          trust_remote_code=False)
        model, tokenizer = load_model('teacher')
        for request in requests[len(saved):]:
            check_time()
            reused=reusable_output(request,other_requests,other_generated)
            if reused is not None:
                reused['reuse_from_selector']='joint' if selector=='nll' else 'nll'
                append(path,reused)
                print(request['candidate_id'],'reused_identical_teacher_prompt',flush=True)
                continue
            output = generate_one(model, tokenizer, request)
            checks = mechanical_checks(request, output, student_tokenizer)
            append(path, {'candidate_id': request['candidate_id'], 'source_id': request['source_id'],
                          **output, 'mechanical_checks': checks})
            print(request['candidate_id'], checks['hard_rejection_reasons'] or 'full_text_review_pending', flush=True)
    save(folder/'generation_status.json', {'completed': len(requests), 'at_utc': now(),
        'accepted_explanations': 0, 'quality_gate_pending': True, 'public_test_used': False})


def assemble(cohort='bootstrap10',selector='joint'):
    paths = selector_paths(cohort,selector); folder = paths.pool_folder
    from transformers import AutoTokenizer
    sources, teachers = inputs()
    requests = prepare(cohort,selector)
    diagnostics=unique(checked_rows(paths.diagnostics),'source_id')
    generated = unique(checked_rows(folder/'generated.jsonl'), 'candidate_id')
    review_path = folder/'full_text_reviews.json'
    reviews = unique(read(review_path), 'candidate_id') if review_path.exists() else {}
    other=selector_paths(cohort,'nll' if selector=='joint' else 'joint').pool_folder
    other_reviews=unique(read(other/'full_text_reviews.json'),'candidate_id') if (other/'full_text_reviews.json').exists() else {}
    other_requests=unique(read(other/'requests.json'),'candidate_id') if (other/'requests.json').exists() else {}
    other_generated=unique(checked_rows(other/'generated.jsonl'),'candidate_id')
    if set(generated) != {r['candidate_id'] for r in requests}:
        raise ValueError('All requests need durable generation results before pool assembly')
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']), local_files_only=True,
                                              trust_remote_code=False)
    pools = {}
    for case in read(paths.cases):
        sid = case['source_id']; teacher = teachers[sid]
        pools[sid] = {'source_id': sid, 'question_sha256': sources[sid]['question_sha256'],
            'teacher_output_sha256': teacher['output_sha256'], 'candidates': [{
                'candidate_id': 'e0', 'completion': teacher['output'], 'student_tokens': 0,
                'completion_sha256': teacher['output_sha256'], 'explanation_type': 'empty'}],
            'rejected': [], 'development_only': True,
            'selector':'nll_only' if selector=='nll' else 'joint',
            'position_proposal':diagnostics[sid]['proposal']}
    for request in requests:
        cid = request['candidate_id']; sid = request['source_id']
        review=reviews.get(cid)
        if review is None and cid in other_reviews:
            reused=reusable_output(request,other_requests,other_generated)
            if reused is not None and reused['output_sha256']==generated[cid]['output_sha256']:
                review=other_reviews[cid]
        candidate, reasons = reviewed_candidate(request, generated[cid], review,
                                                sources[sid], teachers[sid], tokenizer)
        if candidate is None:
            pools[sid]['rejected'].append({'candidate_id': cid, 'reasons': reasons})
        else:
            pools[sid]['candidates'].append(candidate)
    for pool in pools.values():
        lengths = [c['student_tokens'] for c in pool['candidates'][1:]]
        pool['nonempty_count'] = len(lengths)
        pool['nonempty_token_range'] = max(lengths)-min(lengths) if lengths else 0
        pool['length_matching_target_max8_satisfied'] = pool['nonempty_token_range'] <= 8
    freeze(folder/'reviewed_pools.json', list(pools.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare','generate','assemble'])
    parser.add_argument('--cohort',choices=COHORTS,default='bootstrap10')
    parser.add_argument('--selector',choices=('joint','nll'),default='joint')
    args = parser.parse_args()
    {'prepare': prepare, 'generate': generate_pool, 'assemble': assemble}[args.action](args.cohort,args.selector)
