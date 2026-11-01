"""Adopted uniform automatic eligibility and independent fixed128 actual sample.

Automatic eligibility is never a mathematical review. Historical signed
modules/ledgers remain untouched; sample judgments are explicit external input.
"""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from runtime_full import ROOT, HERE, OUT, read, freeze, digest, baseline
from auxiliary_explainer_recipe import FULL, RECIPE_ID
from explanation_pool import unique, mechanical_checks, insert_explanation, REVIEW_CHECKS
from review_full_explanations import complete_prefix, validate_decisions

FOLDER = OUT / 'explanation_quality_v3'
AUTHORITY = OUT / 'explanation_quality_workflow_amendment_v3.json'
ROSTER = OUT / 'explanation_quality_v3_proposed/spotcheck_roster_proposed_v3.json'
HISTORY = FOLDER / 'historical_actual2724_v1.jsonl'
SAMPLE = FOLDER / 'sample_decisions_v3.json'
REPORT = FOLDER / 'spotcheck_report_v3.json'
EXPORT = FOLDER / 'automatic_export_v3.json'


def verify_inputs(value):
    if not value.get('inputs'): raise ValueError('Missing concrete inputs')
    for name, sha in value['inputs'].items():
        if baseline.sha(ROOT / name) != sha: raise ValueError('Changed input: ' + name)


def authority():
    value = read(AUTHORITY); verify_inputs(value)
    if (value.get('status') != 'adopted' or value.get('full_semantic_review_required') is not False
            or value.get('sample_all_correct_required') is not False
            or value.get('user_authorization') != '自动检查＋固定128条抽检这个方案就行'):
        raise ValueError('Explicit adopted v3 authority required')
    return value


def sentence_count(text):
    # Decimal points, mathematical punctuation and abbreviations do not split
    # sentences. A trailing fragment counts conservatively as a sentence.
    s = re.sub(r'(?<=\d)\.(?=\d)', '<decimal>', text.strip())
    s = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|St|e\.g|i\.e)\.', '<abbr>', s)
    return len([x for x in re.split(r'[.!?]+(?:[\s\"\u201d]|$)', s) if x.strip()])


def load_context():
    authority()
    from full_explanations import load_requests, validate_generated, student_tokenizer
    context = load_requests(); requests = context[-1]
    tokenizer = student_tokenizer(); outputs = complete_prefix(FULL / 'generated.jsonl')
    validate_generated(outputs, requests, tokenizer, complete=True, recipe_id=RECIPE_ID)
    history = complete_prefix(HISTORY)
    validate_decisions(requests, outputs, history)
    if len(history) != 2724 or sum(not r['accepted'] for r in history) != 430:
        raise ValueError('Historical review baseline changed')
    roster = read(ROSTER)
    usable = sorted((r for r in outputs if not r['mechanical_checks']['hard_rejection_reasons']),
                    key=lambda r: r['candidate_id'])
    selected = random.Random(42).sample(usable, 128)
    if len(usable) != 10997 or [r['candidate_id'] for r in selected] != [r['candidate_id'] for r in roster['sample']]:
        raise ValueError('Fixed128 sample does not replay')
    out = unique(outputs, 'candidate_id')
    req = unique(requests, 'candidate_id')
    for entry in roster['sample']:
        row = out[entry['candidate_id']]; q = req[entry['candidate_id']]
        identity = {**q, **row, 'generated_record_sha256': row['record_sha256']}
        if any(identity.get(k) != v for k, v in entry.items()):
            raise ValueError('Fixed sample identity changed')
    return context, tokenizer, outputs, history


def commit_sample(path):
    context, _, outputs, history = load_context(); requests = context[-1]
    req = unique(requests, 'candidate_id'); out = unique(outputs, 'candidate_id')
    old = unique(history, 'candidate_id')
    expected = [x['candidate_id'] for x in read(ROSTER)['sample'] if x['candidate_id'] not in old]
    specs = read(path)
    if [s[0] for s in specs] != expected or len(specs) != 97:
        raise ValueError('Exactly the remaining fixed97 explicit judgments in frozen order required')
    reviews = []
    for cid, accepted, reason, failed in specs:
        if type(accepted) is not bool or not reason or not isinstance(failed, list):
            raise ValueError('Explicit outcome and individual reason required')
        q = req[cid]; r = out[cid]
        value = dict(candidate_id=cid, accepted=accepted, reason=reason,
            kind='assistant_mathematical_review', all_text_read=True, independent_human_review=False,
            review_scope='fixed128_v3_actual_visible_context_and_exact_output_read',
            recipe_id=RECIPE_ID, output_sha256=r['output_sha256'],
            generated_record_sha256=r['record_sha256'], request_sha256=r['request_sha256'],
            question_sha256=q['question_sha256'], teacher_output_sha256=q['teacher_output_sha256'],
            mechanical_hard_rejection_reasons=r['mechanical_checks']['hard_rejection_reasons'])
        value.update({k: k not in failed for k in REVIEW_CHECKS})
        if q['explanation_type'] == 'contrastive':
            value['contrastive_alternative_valid_and_plausible'] = 'contrastive_alternative_valid_and_plausible' not in failed
        reviews.append(value)
    validate_decisions(requests, outputs, reviews)
    freeze(SAMPLE, reviews)
    return sample_report(requests, outputs, history)


def sample_report(requests, outputs, history):
    new = read(SAMPLE); validate_decisions(requests, outputs, new)
    old = unique(history, 'candidate_id'); fresh = unique(new, 'candidate_id')
    ids = [r['candidate_id'] for r in read(ROSTER)['sample']]
    if set(fresh) != set(ids) - set(old) or len(fresh) != 97:
        raise ValueError('Fixed sample judgments incomplete or replaced')
    entries = []
    for cid in ids:
        r = old[cid] if cid in old else fresh[cid]
        entries.append({'candidate_id': cid, 'accepted': r['accepted'], 'reason': r['reason'],
            'output_sha256': r['output_sha256'], 'request_sha256': r['request_sha256'],
            'evidence': 'exact_historical_reuse' if cid in old else 'new_actual_read_v3',
            'review_sha256': r.get('record_sha256', digest(json.dumps(r, sort_keys=True)))})
    paths = [Path(__file__), AUTHORITY, ROSTER, HISTORY, SAMPLE, FULL/'requests.json', FULL/'generated.jsonl']
    result = dict(schema='tgrd_explanation_fixed128_actual_spotcheck_v3', complete=True,
        sample_size=128, existing_exact_reused=31, new_actual_reads=97,
        accepted=sum(r['accepted'] for r in entries), rejected=sum(not r['accepted'] for r in entries),
        entries=entries, population='original10997_mechanical_passes', descriptive_only=True,
        independent_human_review=False, all_correct_required=False, replacement_after_results=False,
        original_teacher_full_text_review_count=1433, public_test_used=False,
        inputs={str(p.relative_to(ROOT)): baseline.sha(p) for p in paths})
    freeze(REPORT, result); return result


def automatic_decision(request, output, source, teacher, tokenizer, review=None):
    checks = mechanical_checks(request, output, tokenizer)
    reasons = list(checks['hard_rejection_reasons'])
    if digest(output['output']) != output['output_sha256']:
        raise ValueError('Exact output hash changed')
    if request['question_sha256'] != source['question_sha256'] or request['teacher_output_sha256'] != teacher['output_sha256']:
        raise ValueError('Wrong original root/teacher')
    sentences = sentence_count(output['output'])
    if not 1 <= sentences <= 2: reasons.append('not_one_or_two_sentences_v3')
    completion = insert_explanation(teacher['output'], request['step'], output['output'])
    from learning import tokenize
    count = None
    try:
        sample = tokenize(tokenizer, source['question'], completion)
        count = len(sample['prompt_ids']) + len(sample['target_ids'])
    except ValueError: reasons.append('inserted_context_exceeds4096_including_EOS_v3')
    if review is not None and review['accepted'] is False: reasons.append('actual_semantic_rejection')
    return dict(candidate_id=request['candidate_id'], source_id=request['source_id'],
        output_sha256=output['output_sha256'], request_sha256=output['request_sha256'],
        generated_record_sha256=output['record_sha256'], eligible=not reasons, reasons=reasons,
        quality_mode='uniform_automatic_plus_fixed128_v3', sentence_count=sentences,
        full_context_tokens_including_EOS=count, student_tokens=checks['student_tokens'],
        all_text_read=review is not None, semantic_correctness_verified=bool(review and review['accepted']),
        actual_review_accepted=None if review is None else review['accepted'],
        actual_review_reason=None if review is None else review['reason'])


def export():
    context, tokenizer, outputs, history = load_context()
    _, cases, sources, teachers, diagnostics, views, requests = context
    report = sample_report(requests, outputs, history)
    reviews = unique(history + read(SAMPLE), 'candidate_id')
    decisions = [automatic_decision(q, r, sources[q['source_id']], teachers[q['source_id']], tokenizer,
        reviews.get(q['candidate_id'])) for q, r in zip(requests, outputs)]
    freeze(FOLDER/'automatic_decisions_v3.json', decisions)
    paths = [Path(__file__), AUTHORITY, REPORT, HISTORY, SAMPLE, FOLDER/'automatic_decisions_v3.json',
        FULL/'requests_manifest.json', FULL/'requests.json', FULL/'generated.jsonl']
    result = dict(schema='tgrd_explanation_automatic_export_v3', complete=True,
        generated=len(outputs), automatically_eligible=sum(r['eligible'] for r in decisions),
        excluded=sum(not r['eligible'] for r in decisions), actual_reviewed=len(reviews),
        actual_rejected=sum(not r['accepted'] for r in reviews.values()),
        mechanically_usable_unread=sum(not r['mechanical_checks']['hard_rejection_reasons'] and r['candidate_id'] not in reviews for r in outputs), all_text_read=False,
        semantic_correctness_verified=False, sample_accepted=report['accepted'],sample_rejected=report['rejected'],
        formal_training_enabled=False, public_test_used=False,
        inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in paths})
    freeze(EXPORT,result); print(result | {'inputs':len(paths)},flush=True); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('commit-sample','export'))
    p.add_argument('--decisions');a=p.parse_args()
    if a.command=='commit-sample':
        r=commit_sample(a.decisions);print({k:v for k,v in r.items() if k not in ('entries','inputs')})
    else:export()
