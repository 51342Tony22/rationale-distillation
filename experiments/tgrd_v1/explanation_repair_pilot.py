"""One bounded development prompt repair, isolated from all scientific pools.

Six outputs on the first three rank-zero development roots, two explanation
types each. No retries, output editing, automatic promotion, or public data.
The failed original recipe remains immutable. A pass here is only an
instruction-following diagnostic, never evidence of learning or transfer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, BASE, read, checked_rows, freeze, save,
                          append, baseline, digest, signature, gpu, load_model, now)
from explanation_pool import generate_one, mechanical_checks, TYPES

ORIGINAL = OUT / 'calibration/cal40_reviewed34_v2/shared_pool_v1'
FOLDER = OUT / 'calibration/explanation_repair_pilot_v2'
SYSTEM = (
    'You write brief margin notes beside completed mathematical steps. '
    'The quoted problem is context, not a request to solve it. '
    'Justify only the quoted completed step in exactly two short sentences, '
    'about thirty-five to forty-five English words. '
    'Use words, no digits, equations, headings, lists, or final answer. '
    'Do not repeat the solution or continue to the next step.'
)
EXAMPLES = {
    'relation': (
        'Completed step: multiply the number of equal packs by the items per pack.\n'
        'Margin note: Each pack contributes the same quantity, so the pack count '
        'specifies how many equal contributions are combined. Multiplication '
        'collects those repeated contributions into a single total while keeping '
        'the quantity measured in individual items.'),
    'contrastive': (
        'Completed step: divide the total distance by the constant speed.\n'
        'Margin note: Speed measures distance covered during each unit of time, '
        'so division counts how many such time units make up the journey. '
        'Multiplying would scale the distance by the rate instead of recovering '
        'the elapsed time.'),
}


def repair_request(original):
    if original['prompt_boundary'] != 'question_plus_prior_prefix_plus_current_step_only':
        raise ValueError('Expected an already bounded development request')
    kind = original['explanation_type']
    if kind not in TYPES:
        raise ValueError('Unknown explanation type')
    request = dict(original)
    request.update(candidate_id=original['candidate_id'] + ':repair_pilot_v2',
        original_candidate_id=original['candidate_id'], system=SYSTEM,
        user=('Write a margin note, following the illustration.\n\n' + EXAMPLES[kind]
              + '\n\nQuoted context and completed step:\n<quoted_context>\n'
              + original['visible_context'] + '\n</quoted_context>\n\n'
              + ('Explain why the completed step follows from the quantities and conditions. '
                 if kind == 'relation' else
                 'Explain why the completed operation applies and one plausible alternative does not. ')
              + 'Write only the two-sentence margin note. Do not solve the problem.'),
        engineering_only=True, export_to_scientific_pool=False)
    return request


def prepare():
    requests = read(ORIGINAL / 'requests.json')
    rows = checked_rows(ORIGINAL / 'generated.jsonl')
    if [r['candidate_id'] for r in rows] != [r['candidate_id'] for r in requests]:
        raise ValueError('Finish the fixed original batch before repairing its prompt')
    if any(not r['mechanical_checks']['hard_rejection_reasons'] for r in rows):
        raise ValueError('This repair plan assumes the observed zero-pass original batch')
    ids = sorted({r['source_id'] for r in requests if r['position_rank'] == 0},
                 key=lambda sid: int(sid.rsplit(':', 1)[1]))[:3]
    selected = [r for sid in ids for kind in TYPES for r in requests
                if r['source_id'] == sid and r['position_rank'] == 0
                and r['explanation_type'] == kind]
    if len(selected) != 6:
        raise ValueError('Exactly three roots and both fixed types are required')
    repaired = [repair_request(r) for r in selected]
    freeze(FOLDER / 'requests.json', repaired)
    paths = [Path(__file__), HERE / 'explanation_pool.py', ORIGINAL / 'requests.json',
             ORIGINAL / 'generated.jsonl', ROOT / 'experiments/math_distill_v1/config.json']
    plan = {'schema': 'tgrd_explanation_instruction_repair_pilot_v2',
        'purpose': 'bounded_development_instruction_following_repair_only',
        'input_sha256': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'request_sha256': baseline.sha(FOLDER / 'requests.json'),
        'original_fixed_batch_count': len(rows), 'original_mechanical_pass': 0,
        'selection_rule': 'first3_numeric_source_ids_with_rank0_both_types',
        'source_ids': ids, 'request_count': 6, 'prompt_configurations': 1,
        'attempts_per_request': 1, 'max_new_tokens': 256,
        'student_token_range_unchanged': [32, 64], 'temperature': 0,
        'automatic_expansion': False, 'export_to_scientific_pool': False,
        'public_test_used': False, 'no_learning_or_transfer_claim': True}
    freeze(FOLDER / 'plan.json', plan)
    return plan, repaired


def run():
    plan, requests = prepare()
    signature('explanation_repair_pilot_v2',
        [Path(__file__), HERE / 'explanation_pool.py', FOLDER / 'plan.json',
         FOLDER / 'requests.json'], {'engineering_only': True, 'one_attempt': True})
    path = FOLDER / 'generated.jsonl'
    saved = checked_rows(path)
    if [r['candidate_id'] for r in saved] != [r['candidate_id'] for r in requests[:len(saved)]]:
        raise ValueError('Repair pilot resume order changed')
    if len(saved) < len(requests):
        with gpu():
            signed = read(ROOT / 'output/math_distill_v1/dev_baseline/teacher_reference/signature.json')
            for name, spec in signed['model_files'].items():
                if baseline.sha(ROOT / BASE['teacher_dir'] / name) != spec['sha256']:
                    raise ValueError('Teacher weights changed')
            from transformers import AutoTokenizer
            student = AutoTokenizer.from_pretrained(str(ROOT / BASE['student_dir']),
                local_files_only=True, trust_remote_code=False)
            model, tokenizer = load_model('teacher')
            for request in requests[len(saved):]:
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
        'method_effectiveness_established': False, 'public_test_used': False,
        'plan_sha256': baseline.sha(FOLDER / 'plan.json'),
        'generated_sha256': baseline.sha(path), 'total_seconds': sum(r['seconds'] for r in rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)()
