"""CPU-only full-text review packets; never auto-accept mathematical reasoning."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from runtime_full import ROOT, DATA, OUT, baseline, read, rows, checked_rows, freeze, save, digest, now


def main():
    sources={r['source_id']:r for r in read(DATA/'train7473.json')}
    packet_dir=OUT/'material_audits/teacher_packets'
    packet_dir.mkdir(parents=True,exist_ok=True)
    targets={}
    origins={}
    # A growing source may end in an in-progress write. Only consume complete lines.
    for name in ['teacher_targets_cal40','teacher_targets_train500','teacher_targets_train7473']:
        path=OUT/'materials'/f'{name}.jsonl'
        if not path.exists():continue
        raw=path.read_bytes()
        if not raw.endswith(b'\n'):
            raw=raw[:raw.rfind(b'\n')+1]
        records=[json.loads(x) for x in raw.decode().splitlines()]
        for record in records:
            value=dict(record)
            signature=value.pop('record_sha256')
            assert digest(json.dumps(value,sort_keys=True))==signature
            assert digest(record['output'])==record['output_sha256']
            sid=record['source_id']
            assert record['question_sha256']==sources[sid]['question_sha256']
            if sid in targets:
                assert targets[sid]['output_sha256']==record['output_sha256'], f'Conflicting reused targets: {sid}'
            else:
                targets[sid]=record
                origins[sid]=str(path.relative_to(ROOT))
    known=rows(ROOT/'output/math_distill_v1/pilot100/teacher_reviews.jsonl')
    review_path=OUT/'material_audits/cal40_teacher_full_text_reviews_v1.json'
    reviewed={r['source_id']:r for r in read(review_path)} if review_path.exists() else {}
    additional_path=OUT/'material_audits/teacher_additional_reviews_v1.jsonl'
    if additional_path.exists():
        for review in checked_rows(additional_path):
            sid=review['source_id']
            if sid in reviewed:
                raise ValueError('Duplicate full-text review: '+sid)
            reviewed[sid]=review
    imported=[]
    current=[]
    pending=[]
    mismatch=[]
    for sid,record in targets.items():
        legacy=next((r for r in known if r['source_id']==sid and r['output_sha256']==record['output_sha256']),None)
        if legacy:
            imported.append(legacy)
        review=reviewed.get(sid)
        if review:
            assert review['output_sha256']==record['output_sha256'], f'Stale review: {sid}'
            assert review['question_sha256']==sources[sid]['question_sha256']
            assert type(review['accepted']) is bool and review['reason'] and review['all_text_read']
            assert review['review_kind']=='assistant_full_text_mathematical_review'
            current.append(review)
            continue
        # Historical imports never satisfy the modern full-text gate. Keep
        # their roots in the same pending queue until actually rereviewed.
        src=sources[sid]
        packet={'source_id':sid,'question':src['question'],'gold':src['gold'],
                'reference_rationale':src['reference_rationale'], 'teacher_output':record['output'],
                'output_sha256':record['output_sha256'], 'origin':origins[sid],
                'numeric_answer_matches_gold':record['correct'], 'finish_reason':record['finish_reason'],
                'review_state':'pending_full_text_review',
                'required_checks':['all_steps_mathematically_valid','correct_entities_and_conditions',
                                   'units_match_gold_or_explicit_issue_flag','no_missing_assumptions',
                                   'complete_and_not_truncated'],
                'acceptance_not_inferred_from_numeric_match':True}
        freeze(packet_dir/f'{sid.replace(":","_")}.json',packet)
        pending.append({'source_id':sid,'output_sha256':record['output_sha256'],
                        'packet':str((packet_dir/f'{sid.replace(":","_")}.json').relative_to(ROOT)),
                        'numeric_answer_matches_gold':record['correct']})
        if not record['correct']:mismatch.append(sid)
    if {r['source_id'] for r in pending} != set(targets) - set(reviewed):
        raise ValueError('Pending queue must cover every missing actual full-text decision')
    freeze(OUT/'material_audits/imported_legacy_teacher_reviews.json',imported)
    save(OUT/'material_audits/pending_teacher_review_index.json',{'updated_utc':now(),'count':len(pending),'items':pending})
    summary={'updated_utc':now(),'unique_teacher_outputs':len(targets),'target_roots':7473,
             'imported_legacy_reviews':len(imported),'legacy_accepted':sum(r['accepted'] for r in imported),
             'legacy_only_not_full_text_reviewed':sum(r['source_id'] not in reviewed for r in imported),
             'legacy_rereviewed_with_actual_full_text':sum(r['source_id'] in reviewed for r in imported),
             'new_full_text_reviewed':len(current),'new_review_accepted':sum(r['accepted'] for r in current),
             'new_review_rejected_or_quarantined':sum(not r['accepted'] for r in current),
             'new_full_text_review_pending':len(pending),'pending_numeric_mismatch_ids':mismatch,
             'pending_includes_legacy_only_roots':True,
             'new_reviews_auto_accepted':0,'official_test_predictions_used':False}
    save(OUT/'material_audits/review_progress.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
