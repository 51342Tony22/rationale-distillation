"""GSM8K-first complete-test evaluator, enabled only after explicit protocol freeze."""
from __future__ import annotations

import argparse
from pathlib import Path

from runtime_full import *
import score_sensitivity_v2 as sensitivity
from evaluation_inputs import validate_request


def test_items():
    source = ROOT / 'data/math_distill_v1/source/test.jsonl'
    original = rows(source)
    assert len(original) == 1319
    result = []
    for i, row in enumerate(original):
        result.append({'source_id': f'gsm8k:test:{i}', 'question':row['question'],
                       'gold':row['answer'].rsplit('####',1)[1].strip(),
                       'question_sha256':digest(row['question'])})
    return result


def main(condition, seed, adapter):
    verified=validate_request(condition,seed,adapter)
    adapter_directory=verified['adapter_directory']
    items = test_items()
    folder = OUT / 'evaluation/GSM8K' / condition / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    extra = {'condition':condition,'seed':seed,'ordered_ids':[r['source_id'] for r in items],
             'prompt':BASE['student_prompt'],'greedy':True,'max_new_tokens':1024,'dataset_count':1319,
             'training_artifact':verified['training_artifact']}
    paths = [Path(__file__), HERE/'runtime.py',HERE/'runtime_full.py',HERE/'evaluation_inputs.py',
             HERE/'training_plan.py',HERE/'training_checkpoint.py',
             ROOT/'experiments/math_distill_v1/config.json',
             ROOT/'data/math_distill_v1/source/test.jsonl', Path(sensitivity.__file__), Path(baseline.__file__),
             *verified['signature_paths']]
    signature(f'eval_GSM8K_{condition}_{seed}', paths, extra)
    signature_sha=baseline.sha(OUT/'signatures'/f'eval_GSM8K_{condition}_{seed}.json')
    saved = checked_rows(folder/'predictions.jsonl')
    assert [r['source_id'] for r in saved] == [r['source_id'] for r in items[:len(saved)]]
    for record,item in zip(saved,items):
        if (record['condition']!=condition or record['seed']!=seed
                or record.get('evaluation_signature_file_sha256')!=signature_sha
                or record['question_sha256']!=item['question_sha256'] or record['gold']!=item['gold']
                or digest(record['output'])!=record['output_sha256']):
            raise ValueError('Saved prediction belongs to another input/model/evaluation signature')
    with gpu():
        check_time(reserve=60)
        import mlx.core as mx
        mx.set_memory_limit(14*1024**3)
        mx.set_cache_limit(512*1024**2)
        mx.random.seed(seed)
        if adapter_directory is not None:
            baseline.verify_student(BASE)
            from mlx_lm import load
            model, tokenizer = load(str(ROOT / BASE['student_dir']), adapter_path=str(adapter_directory),
                                    tokenizer_config={'local_files_only':True,'trust_remote_code':False})
        else:
            if condition=='student_base':baseline.verify_student(BASE)
            else:
                signed=read(ROOT/'output/math_distill_v1/dev_baseline/teacher_reference/signature.json')
                for name,spec in signed['model_files'].items():
                    if baseline.sha(ROOT/BASE['teacher_dir']/name)!=spec['sha256']:
                        raise ValueError('Teacher reference model changed')
            model, tokenizer = load_model('teacher' if condition == 'teacher_reference' else 'student')
        for item in items[len(saved):]:
            check_time(reserve=30)
            output = generate(model, tokenizer, item['question'],teacher=condition=='teacher_reference')
            v2, _ = sensitivity.parse(output['scoring_text'])
            value = {**item,**output,**baseline.score(output['scoring_text'],item['gold']),
                     'v2_correct':v2 is not None and v2==baseline.number(item['gold']),
                     'condition':condition,'seed':seed,'evaluation_signature_file_sha256':signature_sha}
            append(folder/'predictions.jsonl',value)
            saved.append(value)
            save(folder/'summary.json',{'completed':len(saved),'total':1319,'correct':sum(x['correct'] for x in saved),
                 'v2_correct':sum(x['v2_correct'] for x in saved),'generation_seconds':sum(x['seconds'] for x in saved),
                 'complete':len(saved)==1319,'accuracy':sum(x['correct'] for x in saved)/len(saved),
                 'condition':condition,'seed':seed,'training_artifact':verified['training_artifact'],
                 'evaluation_signature_file_sha256':signature_sha,
                 'partial_score_is_not_full_test_result':len(saved)!=1319})
            status('GSM8K_full_test',condition=condition,seed=seed,completed=len(saved),total=1319)
            if len(saved)%10==0:
                print(f'{condition}/{seed}: {len(saved)}/1319',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--adapter')
    a=parser.parse_args()
    main(a.condition,a.seed,a.adapter)
