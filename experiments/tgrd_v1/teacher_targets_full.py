"""Full7473 teacher material generation; reuse exact signed earlier records."""
from __future__ import annotations

from pathlib import Path
from runtime_full import *
from pilot_materials import TEACHER_EXTRA


def main():
    items = read(DATA/'train7473.json')
    assert len(items)==7473
    candidates = [OUT/'materials/teacher_targets_train500.jsonl', OUT/'materials/teacher_targets_cal40.jsonl']
    # The handoff waits for both source jobs to finish before starting.
    state=read(OUT/'queue_state.json')
    assert {'teacher_cal40','teacher_train500'} <= set(state['completed'])
    prior={}
    for p in candidates:
        records=checked_rows(p)
        for r in records:
            assert digest(r['output'])==r['output_sha256']
            assert r['source_id'] not in prior
            prior[r['source_id']]={**r,'reuse_source':str(p.relative_to(ROOT)),
                                   'reuse_record_sha256':r['record_sha256']}
    assert len(prior)==540
    signature('teacher_full7473',[Path(__file__),HERE/'runtime_full.py',HERE/'runtime.py',DATA/'train7473.json',
              DATA/'full_gsm8k_manifest.json',OUT/'full_run_authorization.json',*candidates],
              {'prompt_suffix':TEACHER_EXTRA,'max_new_tokens':1024,'temperature':0,'review_required':True})
    path=OUT/'materials/teacher_targets_train7473.jsonl'
    saved=checked_rows(path)
    assert [r['source_id'] for r in saved]==[r['source_id'] for r in items[:len(saved)]]
    with gpu():
        signed=read(ROOT/'output/math_distill_v1/dev_baseline/teacher_reference/signature.json')
        for name,spec in signed['model_files'].items():
            assert baseline.sha(ROOT/BASE['teacher_dir']/name)==spec['sha256']
        model=tok=None
        for item in items[len(saved):]:
            check_time()
            sid=item['source_id']
            if sid in prior:
                value={k:v for k,v in prior[sid].items() if k!='record_sha256'}
                assert item['question_sha256']==value['question_sha256']
                value['current_full_run_generation_seconds']=0
            else:
                if model is None:model,tok=load_model('teacher')
                value={'source_id':sid,'question_sha256':item['question_sha256'],'gold':item['gold'],
                       **generate(model,tok,item['question'],teacher=True,user_text=item['question']+TEACHER_EXTRA),
                       'requires_new_full_text_review':True}
                value.update(baseline.score(value['output'],item['gold']))
            append(path,value)
            saved.append(value)
            status('teacher_targets_full7473',completed=len(saved),total=7473,
                   new_generations=sum('reuse_source' not in x for x in saved),
                   numeric_correct=sum(x['correct'] for x in saved),
                   quality_gate_pending=True,official_test_evaluated=False)
            if len(saved)%10==0 or 'reuse_source' not in value:
                print(f'full7473 {len(saved)}/7473 {sid}: numeric={value["correct"]}',flush=True)
        status('teacher_full7473_complete_needs_review',completed=7473,total=7473,
               numeric_correct=sum(x['correct'] for x in saved),training_started=False)


if __name__=='__main__':
    main()
