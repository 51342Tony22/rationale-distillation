"""Stream actual local-explanation reviews without altering active generation.

Review decisions are written by the assistant after reading the exact prompt
and text. This module validates and compiles them; it never infers acceptance.
The final review file appears only when every mechanically usable output has
an actual decision. Mechanically rejected attempts remain in the raw ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_full import ROOT, read, freeze, save, append, digest, baseline, now
from auxiliary_explainer_recipe import FULL, DIAGNOSTIC, RECIPE_ID
from explanation_pool import REVIEW_CHECKS, unique, reusable_output

LEDGER = FULL / 'actual_reviews_v1.jsonl'
BATCHES = FULL / 'actual_review_batches_v1'


def complete_prefix(path):
    """Read only newline-committed rows while another process appends."""
    path = Path(path)
    data = path.read_bytes() if path.exists() else b''
    lines = data.split(b'\n')[:-1]
    rows = []
    for line in lines:
        row = json.loads(line); value = dict(row); checksum = value.pop('record_sha256')
        if digest(json.dumps(value, sort_keys=True)) != checksum:
            raise ValueError('Corrupt committed record: ' + str(path))
        rows.append(row)
    return rows


def context():
    from full_explanations import validate_generated, student_tokenizer
    from auxiliary_explainer_recipe import load_recipe, verify_inputs
    load_recipe()
    manifest = read(FULL / 'requests_manifest.json'); verify_inputs(manifest)
    requests = read(FULL / 'requests.json'); outputs = complete_prefix(FULL / 'generated.jsonl')
    validate_generated(outputs, requests, student_tokenizer(), recipe_id=RECIPE_ID)
    return manifest, requests, outputs


def validate_decisions(requests, outputs, reviews):
    req = unique(requests, 'candidate_id'); out = unique(outputs, 'candidate_id')
    rev = unique(reviews, 'candidate_id')
    for cid, review in rev.items():
        if cid not in out or cid not in req:
            raise ValueError('Cannot review an ungenerated or undeclared output')
        request = req[cid]; row = out[cid]
        if (review.get('kind') != 'assistant_mathematical_review'
                or review.get('all_text_read') is not True
                or type(review.get('accepted')) is not bool or not review.get('reason')
                or review.get('output_sha256') != row['output_sha256']
                or review.get('request_sha256') != digest(request['system'] + '\n' + request['user'])
                or review.get('question_sha256') != request['question_sha256']
                or review.get('teacher_output_sha256') != request['teacher_output_sha256']
                or review.get('recipe_id') != RECIPE_ID
                or ('generated_record_sha256' in review
                    and review['generated_record_sha256'] != row.get('record_sha256'))
                or review.get('mechanical_hard_rejection_reasons') != row['mechanical_checks']['hard_rejection_reasons']):
            raise ValueError('Actual full-text review differs from the exact prompt/output: ' + cid)
        if review['accepted'] and (row['mechanical_checks']['hard_rejection_reasons']
                or any(review.get(k) is not True for k in REVIEW_CHECKS)
                or (request['explanation_type'] == 'contrastive'
                    and review.get('contrastive_alternative_valid_and_plausible') is not True)):
            raise ValueError('Accepted explanation lacks semantic/contrastive evidence: ' + cid)
    return req, out, rev


def commit_batch(name, reviews, *, provenance=None):
    """Call with explicit per-output judgments, never a default acceptance loop."""
    _, requests, outputs = context()
    existing = reconciled_reviews(requests, outputs)
    if Path(name).name != name or not name:
        raise ValueError('Review batch name must be a plain filename stem')
    batch = BATCHES / (name + '.json')
    # Idempotent replay repairs a crash between freezing a batch and appending
    # its last ledger rows. Conflicting decisions remain immutable errors.
    value = {'schema': 'tgrd_actual_explanation_review_batch_v1',
        'recipe_id': RECIPE_ID, 'reviews': reviews, 'provenance': provenance or {}}
    if not batch.exists() and any(r['candidate_id'] in {v['candidate_id'] for v in existing} for r in reviews):
        raise ValueError('An already reviewed output must retain its original batch')
    known = unique(existing, 'candidate_id')
    for review in reviews:
        old = known.get(review['candidate_id'])
        if old is not None:
            if {k: v for k, v in old.items() if k != 'record_sha256'} != review:
                raise ValueError('Existing actual review decision changed')
        else:
            known[review['candidate_id']] = review
    validate_decisions(requests, outputs, list(known.values()))
    freeze(batch, value)
    return progress()


def reconciled_reviews(requests, outputs):
    """Repair an interrupted ledger from immutable, explicitly judged batches."""
    existing = complete_prefix(LEDGER); known = unique(existing, 'candidate_id')
    batches = [r for p in sorted(BATCHES.glob('*.json')) for r in read(p)['reviews']]
    frozen = unique(batches, 'candidate_id')
    validate_decisions(requests, outputs, batches)
    for cid, review in known.items():
        if frozen.get(cid) != {k: v for k, v in review.items() if k != 'record_sha256'}:
            raise ValueError('Committed actual review is absent from or differs from its frozen batch')
    if LEDGER.exists():
        data = LEDGER.read_bytes(); boundary = data.rfind(b'\n') + 1
        if boundary < len(data):
            fragment = data[boundary:]
            target = FULL / 'interrupted_review_fragments_v1' / (digest(fragment.hex()) + '.bin')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes(fragment)
            with LEDGER.open('r+b') as stream: stream.truncate(boundary)
    for review in batches:
        if review['candidate_id'] not in known:
            known[review['candidate_id']] = append(LEDGER, review)
    if not LEDGER.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.touch()
    return complete_prefix(LEDGER)


def reuse_diagnostic():
    manifest, requests, outputs = context()
    old_requests = unique(read(DIAGNOSTIC / 'requests.json'), 'candidate_id')
    old_outputs = unique(complete_prefix(DIAGNOSTIC / 'generated.jsonl'), 'candidate_id')
    old_reviews = unique(complete_prefix(DIAGNOSTIC / 'full_text_reviews_v1.jsonl'), 'candidate_id')
    req = unique(requests, 'candidate_id'); existing = unique(reconciled_reviews(requests, outputs), 'candidate_id')
    reviews = []
    for row in outputs:
        cid = row['candidate_id']
        if cid in existing or cid not in manifest['previous_generation_registry']:
            continue
        request = req[cid]
        previous = reusable_output(request, old_requests, old_outputs)
        review = old_reviews[cid]
        if (previous is None or previous['output_sha256'] != row['output_sha256']
                or review['output_sha256'] != row['output_sha256']
                or review.get('all_text_read') is not True):
            raise ValueError('Earlier actual review does not match this exact reused output')
        value = {k: v for k, v in review.items() if k != 'record_sha256'}
        value.update(request_sha256=row['request_sha256'], question_sha256=request['question_sha256'],
            teacher_output_sha256=request['teacher_output_sha256'], recipe_id=RECIPE_ID,
            generated_record_sha256=row['record_sha256'],
            reused_actual_review_record_sha256=review['record_sha256'])
        if review['accepted'] and request['explanation_type'] == 'contrastive':
            # The earlier complete mathematical review already accepted this
            # exact contrastive text. Normalize its implicit type judgment;
            # preserve the original reason and signed record as evidence.
            value.setdefault('contrastive_alternative_valid_and_plausible', True)
            value['contrastive_evidence_source'] = 'original_accepted_contrastive_full_text_review'
        reviews.append(value)
    if reviews:
        name = 'diagnostic_reuse_' + digest(json.dumps(reviews, sort_keys=True))[:20]
        commit_batch(name, reviews, provenance={'old_actual_review_file':
            str((DIAGNOSTIC / 'full_text_reviews_v1.jsonl').relative_to(ROOT))})
    return len(reviews)


def coverage(requests, outputs, reviews):
    _, out, rev = validate_decisions(requests, outputs, reviews)
    pending = [r['candidate_id'] for r in outputs
        if not r['mechanical_checks']['hard_rejection_reasons'] and r['candidate_id'] not in rev]
    return {'generated_snapshot_count': len(outputs), 'total_requests': len(requests),
        'mechanical_rejected': sum(bool(r['mechanical_checks']['hard_rejection_reasons']) for r in outputs),
        'actual_reviewed': len(rev), 'actual_accepted': sum(r['accepted'] for r in reviews),
        'actual_rejected': sum(not r['accepted'] for r in reviews),
        'reused_actual_reviews': sum('reused_actual_review_record_sha256' in r for r in reviews),
        'mechanically_usable_pending_actual_review': len(pending), 'pending_candidate_ids': pending,
        'ready_for_final_export': len(outputs) == len(requests) and not pending,
        'formal_training_enabled': False, 'public_test_used': False}


def progress():
    _, requests, outputs = context(); reviews = reconciled_reviews(requests, outputs)
    report = coverage(requests, outputs, reviews)
    report.update(at_utc=now(), recipe_id=RECIPE_ID)
    save(FULL / 'review_progress_v1.json', report)
    if report['ready_for_final_export']:
        files = [FULL / 'requests_manifest.json', FULL / 'generated.jsonl', LEDGER,
                 *sorted(BATCHES.glob('*.json'))]
        freeze(FULL / 'reviews_ready_v1.json', {'ready_for_final_export': True,
            'recipe_id': RECIPE_ID, 'generated_count': len(outputs), 'actual_reviewed': len(reviews),
            'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}})
    return report


def export():
    _, requests, outputs = context(); reviews = reconciled_reviews(requests, outputs)
    report = coverage(requests, outputs, reviews)
    if not report['ready_for_final_export']:
        raise ValueError('Final review export waits for all attempts and actual usable-output decisions')
    batches = [r for p in sorted(BATCHES.glob('*.json')) for r in read(p)['reviews']]
    normalized = [{k: v for k, v in r.items() if k != 'record_sha256'} for r in reviews]
    if unique(batches, 'candidate_id') != unique(normalized, 'candidate_id'):
        raise ValueError('Frozen review batches and committed actual decisions disagree')
    ordered = {r['candidate_id']: r for r in normalized}
    freeze(FULL / 'full_text_reviews.json', [ordered[r['candidate_id']] for r in requests
                                            if r['candidate_id'] in ordered])
    files = [Path(__file__), FULL / 'requests_manifest.json', FULL / 'generated.jsonl', LEDGER,
             FULL / 'full_text_reviews.json', *sorted(BATCHES.glob('*.json'))]
    freeze(FULL / 'actual_review_export_v1.json', {**report,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('progress', 'reuse', 'export'))
    args = parser.parse_args()
    if args.command == 'reuse':
        print({'actual_reviews_reused': reuse_diagnostic()}, flush=True)
    else:
        result = progress() if args.command == 'progress' else export()
        print({k: v for k, v in result.items() if k != 'pending_candidate_ids'}, flush=True)
