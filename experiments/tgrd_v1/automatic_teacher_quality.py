"""Uniform automatic CoT eligibility, explicitly distinct from semantic review.

The user's latest question challenges the all7473 full-text bottleneck. This
version records a lighter implementation choice without pretending that any
unread solution has been semantically reviewed. Historical decisions remain
descriptive evidence; they do not create a source-order-dependent filter.
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from runtime_full import ROOT, HERE, OUT, DATA, BASE, read, checked_rows, freeze, baseline, digest

PROTOCOL = 'selector_first_transfer128_seed42_v1'
MODE = 'automatic_numeric_complete_context_v2'
SCHEMA = 'tgrd_teacher_automatic_eligibility_v2'
FOLDER = OUT / 'teacher_auto_quality_v2'
GATE = OUT / 'gates/full_teacher_automatic_quality_v2.json'
AMENDMENT = OUT / 'teacher_quality_workflow_amendment_v2.json'
CHECK_NAMES = ('numeric_answer_matches_original_gold', 'complete_stop',
               'one_boxed_answer', 'nonempty_output', 'untruncated_context_fits')


def checks(source, teacher, tokenizer):
    output = teacher['output']
    prompt = BASE['student_prompt'].format(question=source['question'])
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    target_tokens = len(tokenizer.encode(output, add_special_tokens=False)) + 1
    numeric = baseline.score(output, source['gold'])['correct']
    if teacher.get('correct') is not numeric:
        raise ValueError('Original numeric score does not reproduce: ' + source['source_id'])
    return {
        'numeric_answer_matches_original_gold': numeric,
        'complete_stop': teacher.get('finish_reason') == 'stop',
        'one_boxed_answer': len(re.findall(r'\\boxed\{[^{}]+\}', output)) == 1,
        'nonempty_output': bool(output.strip()),
        'untruncated_context_fits': prompt_tokens + target_tokens <= 4096,
    }, {'prompt_tokens': prompt_tokens, 'completion_tokens_including_eos': target_tokens}


def decision(source, teacher, tokenizer):
    values, counts = checks(source, teacher, tokenizer)
    return {'source_id': source['source_id'],
        'question_sha256': source['question_sha256'], 'output_sha256': teacher['output_sha256'],
        'review_kind': MODE, 'all_text_read': False,
        'semantic_correctness_verified': False, 'accepted': all(values.values()),
        'automatic_checks': values, **counts,
        'reason': 'Automatic eligibility only; numeric agreement is not a full-text correctness proof.'}


def validate_decision(source, teacher, row, tokenizer=None):
    if (row.get('source_id') != source['source_id'] or teacher['source_id'] != source['source_id']
            or digest(source['question']) != source['question_sha256']
            or teacher['question_sha256'] != source['question_sha256']
            or teacher['gold'] != source['gold']
            or digest(teacher['output']) != teacher['output_sha256']
            or row.get('question_sha256') != source['question_sha256']
            or row.get('output_sha256') != teacher['output_sha256']
            or row.get('review_kind') != MODE or row.get('all_text_read') is not False
            or row.get('semantic_correctness_verified') is not False
            or type(row.get('accepted')) is not bool
            or not row.get('reason')
            or set(row.get('automatic_checks', {})) != set(CHECK_NAMES)
            or any(type(v) is not bool for v in row['automatic_checks'].values())
            or row['accepted'] != all(row['automatic_checks'].values())):
        raise ValueError('Invalid automatic eligibility, not a full-text review: ' + source['source_id'])
    if tokenizer is not None:
        if row != decision(source, teacher, tokenizer):
            raise ValueError('Automatic checks/token counts do not reproduce: ' + source['source_id'])
    elif row['accepted']:
        # Material assembly still independently checks the original answer and
        # completeness. Exact token counts are recomputed by the quality loader.
        if (teacher.get('correct') is not True or teacher.get('finish_reason') != 'stop'
                or not baseline.score(teacher['output'], source['gold'])['correct']
                or len(re.findall(r'\\boxed\{[^{}]+\}', teacher['output'])) != 1
                or type(row.get('prompt_tokens')) is not int
                or type(row.get('completion_tokens_including_eos')) is not int
                or min(row['prompt_tokens'], row['completion_tokens_including_eos']) <= 0
                or row['prompt_tokens'] + row['completion_tokens_including_eos'] > 4096):
            raise ValueError('Accepted automatic root fails its elementary checks')


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
        local_files_only=True, trust_remote_code=False)


def validate(sources, teachers, quality):
    expected = {f'gsm8k:train:{i}' for i in range(7473)}
    ids = [s['source_id'] for s in sources]
    if (quality.get('schema') != SCHEMA or quality.get('protocol') != PROTOCOL
            or quality.get('status') != 'frozen' or quality.get('candidate_root_count') != 7473
            or quality.get('public_test_used') is not False
            or quality.get('all_candidate_full_text_reviewed') is not False
            or quality.get('quality_mode') != MODE or not quality.get('inputs')
            or len(ids) != 7473 or set(ids) != expected
            or [r['source_id'] for r in teachers] != ids
            or [r['source_id'] for r in quality.get('decisions', [])] != ids):
        raise ValueError('Expected explicit complete7473 automatic eligibility, never fabricated reviews')
    tok = tokenizer()
    for source, teacher, row in zip(sources, teachers, quality['decisions']):
        validate_decision(source, teacher, row, tok)
    accepted = [r['source_id'] for r in quality['decisions'] if r['accepted']]
    if not accepted or accepted != quality.get('accepted_root_ids'):
        raise ValueError('Automatic common root roster changed')
    return accepted


def prepare():
    sources = read(DATA / 'train7473.json')
    teachers = checked_rows(OUT / 'materials/teacher_targets_train7473.jsonl')
    tok = tokenizer()
    # Explicitly record that this is an implementation simplification after a
    # user objection, not a verbatim user confirmation of these numeric details.
    freeze(AMENDMENT, {'schema': 'tgrd_teacher_quality_workflow_amendment_v2',
        'protocol': PROTOCOL, 'quality_mode': MODE,
        'user_trigger': '为什么要审核，直接使用不行吗',
        'implementation_decision': 'Replace mandatory7473 semantic reviews with uniform automatic eligibility and fixed128 descriptive spotcheck',
        'exact_automatic_rules_chosen_by_assistant': True,
        'all_candidate_full_text_review_required': False,
        'historical_reviews_preserved': True, 'historical_reviews_used_as_selective_veto': False,
        'new_explanation_semantic_leakage_checks_preserved': True,
        'transfer_probe_verification_preserved': True,
        'formal_conditions_or_seeds_changed': False, 'public_test_used': False})
    policy_path = FOLDER / 'policy.json'
    freeze(policy_path, {'schema': 'tgrd_automatic_teacher_policy_v2', 'protocol': PROTOCOL,
        'quality_mode': MODE, 'checks': list(CHECK_NAMES), 'max_context_tokens_including_eos': 4096,
        'source_or_gold_edited': False, 'numeric_agreement_is_semantic_proof': False,
        'historical_nonrandom_reviews': 'descriptive_only_not_selective_training_filter',
        'spotcheck': {'roots': 128, 'seed': 42, 'sampling': 'uniform_without_replacement_from_automatically_eligible_roots_in_source_order',
            'purpose': 'descriptive_error_estimate_not_root_selection_or_hyperparameter_tuning',
            'failure_policy': 'report_actual_error_rate;no_hindsight_redraw_or_threshold_search'},
        'new_local_explanation_semantic_review_required': True,
        'formal_training_enabled': False, 'public_test_used': False})
    decisions = [decision(s, t, tok) for s, t in zip(sources, teachers)]
    ids = [r['source_id'] for r in decisions if r['accepted']]
    ordered = sorted(random.Random(42).sample(range(len(ids)), 128))
    freeze(FOLDER / 'spotcheck_roster.json', {'schema': 'tgrd_teacher_quality_spotcheck_roster_v2',
        'protocol': PROTOCOL, 'quality_mode': MODE, 'seed': 42, 'root_count': 128,
        'source_ids': [ids[i] for i in ordered],
        'eligible_roster_sha256': digest(json.dumps(ids)),
        'source_file_sha256': baseline.sha(DATA / 'train7473.json'),
        'teacher_file_sha256': baseline.sha(OUT / 'materials/teacher_targets_train7473.jsonl'),
        'frozen_before_sample_specific_semantic_reviews': True,
        'some_roots_may_have_historical_nonrandom_reviews': True,
        'transfer128_roster': False, 'replacement_on_failure': False, 'public_test_used': False})
    freeze(FOLDER / 'decisions.json', decisions)
    paths = [Path(__file__), AMENDMENT, policy_path, FOLDER / 'decisions.json', FOLDER / 'spotcheck_roster.json',
        DATA / 'train7473.json', DATA / 'full_gsm8k_manifest.json',
        OUT / 'materials/teacher_targets_train7473.jsonl', OUT / 'signatures/teacher_full7473.json',
        OUT / 'transfer128_amendment_v1.json', HERE.parent / 'math_distill_v1/config.json']
    value = {'schema': SCHEMA, 'protocol': PROTOCOL, 'status': 'frozen', 'quality_mode': MODE,
        'candidate_root_count': 7473, 'decisions': decisions, 'accepted_root_ids': ids,
        'accepted_root_count': len(ids), 'all_candidate_full_text_reviewed': False,
        'review_policy': {'path': str(policy_path.relative_to(ROOT)), 'sha256': baseline.sha(policy_path)},
        'exclusions_by_check': {k: sum(not r['automatic_checks'][k] for r in decisions) for k in CHECK_NAMES},
        'all_candidate_decisions_semantic': False, 'formal_training_enabled': False,
        'public_test_used': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    validate(sources, teachers, value)
    freeze(GATE, value)
    print({'automatic_eligible': len(ids), 'excluded': 7473 - len(ids),
        'exclusions_by_check': value['exclusions_by_check'], 'full_text_review_claimed': False}, flush=True)
    return value


if __name__ == '__main__':
    prepare()
