"""Freeze all 7,473 official training roots after explicit full-data authorization."""
from __future__ import annotations

import json
import re
from pathlib import Path

from runtime import ROOT, DATA, OUT, BASE, baseline, read, rows, freeze, digest, now, append, save


def source(i, row):
    rationale, gold = row['answer'].rsplit('####', 1)
    return {'source_id': f'gsm8k:train:{i}', 'source_index': i,
            'question': row['question'], 'gold': gold.strip(), 'source_answer': row['answer'],
            'reference_rationale': re.sub(r'<<[^>]*>>', '', rationale).strip(),
            'question_sha256': digest(row['question'])}


def main():
    train_path = ROOT / 'data/math_distill_v1/source/train.jsonl'
    test_path = ROOT / 'data/math_distill_v1/source/test.jsonl'
    original = rows(train_path)
    assert len(original) == 7473 and len(rows(test_path)) == 1319
    priority500 = read(DATA / 'train500.json')
    cal40 = read(DATA / 'cal40.json')
    priority_ids = [r['source_index'] for r in priority500 + cal40]
    assert len(priority_ids) == len(set(priority_ids)) == 540
    order = priority_ids + [i for i in range(7473) if i not in set(priority_ids)]
    all_rows = [source(i, original[i]) for i in order]
    assert len(all_rows) == len({r['source_id'] for r in all_rows}) == 7473
    freeze(DATA / 'train7473.json', all_rows)
    freeze(DATA / 'full_gsm8k_manifest.json', {
        'official_train_roots': 7473, 'official_test_roots': 1319,
        'train_source_sha256': baseline.sha(train_path),
        'test_source_sha256': baseline.sha(test_path),
        'train7473_sha256': baseline.sha(DATA / 'train7473.json'),
        'training_order': 'Existing frozen 500 candidates, then 40 calibration roots, then remaining official train order',
        'all_official_train_ids_covered_exactly_once': True,
        'official_test_in_training': False,
        'calibration_roots': [r['source_id'] for r in cal40],
        'former_internal_holdouts': 'Development only if incorporated into final7473; never presented as unseen confirmation',
        'eligibility': '7473 is the complete candidate root inventory. Teacher correctness/quality rejection and per-method fallback must be reported; accepted teacher-material N is not automatically7473.',
        'no_replacement_or_teacher_retry': True,
        'source_revision': BASE['dataset_revision']})
    authorization = {
        'recorded_utc': now(), 'scope': 'GSM8K full official train7473 and test1319',
        'user_authorization': '训练集也全量：最终以官方7,473道训练题为目标，允许完整多组实验超过当前20小时窗口。',
        'hard_deadline_utc': None, 'supersedes_original_20_hour_cutoff': True,
        'stop_conditions': ['User cancellation', 'Task complete', 'Quality gate failure requiring redesign', 'Nonrecoverable infrastructure failure'],
        'priority': ['GSM8K','SVAMP only after GSM8K if appropriate','GSM-Plus only after GSM8K if appropriate'],
        'training_seeds':[7,17,27], 'seeds_are_official_recommendation':False}
    freeze(OUT / 'full_run_authorization.json', authorization)
    priority=read(OUT/'execution_priority.json')
    priority.update(version=2,training_scope='all7473_official_training_roots',official_train_count=7473,
                    deadline_policy='user_explicitly_allows_exceeding_original20hours',
                    working_interpretation='Confirmed: full training7473 plus full test1319; old development sets become calibration only if reused for finaltraining',
                    source_manifest='data/tgrd_v1/full_gsm8k_manifest.json')
    save(OUT/'execution_priority.json', priority)
    append(OUT/'protocol_amendments.jsonl', {'at_utc':now(),'change':'Full7473 train confirmed;20-hour hard cutoff revoked;full official1319 test remains frozen; internal train-derived holdouts cannot remain independent after recycling','authorization':authorization['user_authorization']})
    print(json.dumps({'frozen_train':len(all_rows),'test_count':1319,'hard_deadline':None},ensure_ascii=False))


if __name__=='__main__':
    main()
