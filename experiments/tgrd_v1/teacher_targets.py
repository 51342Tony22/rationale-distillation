"""Generate the pinned training-source CoTs; acceptance requires full-text review."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from runtime import *
from pilot_materials import TEACHER_EXTRA


def main(split='train500'):
    items = read(DATA / f'{split}.json')
    prior_path = ROOT / 'output/math_distill_v1/pilot100/teacher_targets.jsonl'
    prior_review_path = prior_path.with_name('teacher_reviews.jsonl')
    prior = {r['source_id']: r for r in rows(prior_path)}
    prior_reviews = {r['source_id']: r for r in rows(prior_review_path)}
    signature(f'teacher_targets_{split}', [Path(__file__), HERE / 'runtime.py', DATA / f'{split}.json',
              prior_path, prior_review_path, ROOT / 'experiments/math_distill_v1/pilot_materials.py'],
              {'prompt_suffix': TEACHER_EXTRA, 'max_new_tokens': 1024, 'temperature': 0,
               'review_policy': 'Numerical correctness never automatically implies full reasoning acceptance'})
    path = OUT / 'materials' / f'teacher_targets_{split}.jsonl'
    saved = checked_rows(path)
    assert [r['source_id'] for r in saved] == [r['source_id'] for r in items[:len(saved)]]
    with gpu():
        # Verify the exact existing teacher weights before loading.
        signed = read(ROOT / 'output/math_distill_v1/dev_baseline/teacher_reference/signature.json')
        for name, spec in signed['model_files'].items():
            assert baseline.sha(ROOT / BASE['teacher_dir'] / name) == spec['sha256']
        model = tok = None
        for item in items[len(saved):]:
            check_time(reserve=60)
            sid = item['source_id']
            if sid in prior:
                source = prior[sid]
                assert source['question_sha256'] == item['question_sha256']
                assert digest(source['output']) == source['output_sha256']
                result = {k: v for k,v in source.items() if k != 'record_sha256'}
                result.update(reused_from=str(prior_path.relative_to(ROOT)), current_run_generation_seconds=0,
                              full_text_review=prior_reviews[sid], requires_new_full_text_review=False)
            else:
                if model is None:
                    model, tok = load_model('teacher')
                result = {'source_id': sid, 'question_sha256': item['question_sha256'], 'gold': item['gold'],
                          **generate(model, tok, item['question'], teacher=True,
                                     user_text=item['question'] + TEACHER_EXTRA),
                          'requires_new_full_text_review': True}
                result.update(baseline.score(result['output'], item['gold']))
            append(path, result)
            saved.append(result)
            status('teacher_targets', split=split, completed=len(saved), total=len(items), correct_numeric=sum(r['correct'] for r in saved),
                   full_text_review_pending=sum(r['requires_new_full_text_review'] for r in saved))
            print(f"teacher {len(saved)}/{len(items)} {sid} correct_numeric={result['correct']}", flush=True)
        status('teacher_targets_complete_review_pending', split=split, completed=len(saved), total=len(items),
               correct_numeric=sum(r['correct'] for r in saved),
               full_text_review_pending=sum(r['requires_new_full_text_review'] for r in saved),
               training_started=False, official_test_evaluated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train500', choices=['train500','cal40','probe60_candidates'])
    args = parser.parse_args()
    try:
        main(args.split)
    except TimeoutError as e:
        status('deadline_checkpointed', reason=str(e))
        print(str(e), flush=True)
