"""Development-only candidate filtering plus the existing audited diagnostic core.

This is a stricter equation heuristic, not a general semantic reasoning parser.
The v1 token-deletion diagnostic remains explicitly labelled as such.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from runtime_full import *
from selector_masking_probe import prepare_case, MLXBackend
from selector_masking_core import diagnose_step, percentile_ranks, stable_seed
from calibration_cohort import COHORTS, cohort_paths


def genuine_transition(text):
    text=re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',text).strip()
    if '=' not in text or '\\boxed' in text:
        return False
    # A single assignment such as "oranges =70" only repeats a quantity.
    return bool(re.search(r'\\(?:times|div|frac|dfrac)(?=[^A-Za-z]|$)|[+*/−]|(?<=[\w}])\s*-\s*(?=[\w{])',text))


def usable_evidence(text,kind):
    if kind=='question':return True
    stripped=re.sub(r'[*#]','',text).strip()
    if not stripped or re.match(r'^(?:Step\s*\d+|\d+[.)])\s*:',stripped,re.I):return False
    if stripped.endswith(':') and '=' not in stripped:return False
    return bool(re.search(r'\d|=|\\frac|\b(?:half|twice|remaining|each|per|total|more|less)\b',stripped,re.I))


def prepare_bootstrap(cohort='bootstrap10'):
    paths=cohort_paths(cohort)
    from transformers import AutoTokenizer
    sources={r['source_id']:r for r in read(DATA/'cal40.json')}
    targets={r['source_id']:r for r in checked_rows(OUT/'materials/teacher_targets_cal40.jsonl')}
    reviews={r['source_id']:r for r in read(paths.reviews)}
    selected=read(paths.relations)
    if len(selected)!=paths.expected_roots or len({r['source_id'] for r in selected})!=len(selected):
        raise ValueError('Unexpected calibration relation coverage')
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']),local_files_only=True,trust_remote_code=False)
    cases=[]
    for relation in selected:
        sid=relation['source_id'];teacher=targets[sid];source=sources[sid]
        assert reviews[sid]['accepted'] and reviews[sid]['output_sha256']==teacher['output_sha256']
        prompt=BASE['student_prompt'].format(question=source['question'])
        completion=teacher['output']
        row={'source_id':sid,'prompt':prompt,'completion':completion,
             'prompt_ids':tokenizer.encode(prompt,add_special_tokens=False),
             'target_ids':tokenizer.encode(completion,add_special_tokens=False),
             'completion_sha256':digest(completion)}
        case=prepare_case(row,source,tokenizer)
        retained=[]
        for item in case['candidates']:
            if genuine_transition(item['step']['text']):
                item['units']=[u for u in item['units'] if usable_evidence(u['text'],u['kind'])]
                retained.append(item)
            else:
                case['omitted_candidates'].append({**item['step'],'reason':'v2_pure_assignment_or_nontransition'})
        case['candidates']=retained
        case['candidate_rule_version']='v2_stricter_equation_heuristic'
        cases.append(case)
    freeze(paths.cases,cases)
    save(paths.tokenization_report,{
        'roots':len(cases),'candidate_steps':sum(len(c['candidates']) for c in cases),
        'with_at_least_two_candidates':sum(len(c['candidates'])>=2 for c in cases),
        'cases':[{'source_id':c['source_id'],'candidate_count':len(c['candidates'])} for c in cases],
        'model_loaded':False,'official_test_used':False})
    return cases


def propose(steps):
    valid=[r for r in steps if r['status']=='diagnostic_complete']
    if not valid:
        return {'positions':[],'status':'no_valid_mask_controls','all_steps_kept_for_reporting':True}
    n=percentile_ranks([r['mean_nll'] for r in valid])
    m=percentile_ranks([r['mask_excess'] for r in valid])
    scores=[a+b for a,b in zip(n,m)]
    order=sorted(range(len(valid)),key=lambda i:(-scores[i],valid[i]['step_index']))
    return {'positions':[valid[i]['step_index'] for i in order[:2]],
            'scores':[{'step_index':valid[i]['step_index'],'score':scores[i]} for i in order],
            'status':'within_valid_candidate_rank_sum',
            'excluded_for_missing_controls':[r['step_index'] for r in steps if r not in valid],
            'deviation_from_v1':'v1_abstained_entire_question_if_any_candidate_invalid;v2_reports_invalid_and_ranks_valid_candidates',
            'masking_method':'v1_token_deletion_partial_control_not_semantic_causal_proof'}


def main(cohort='bootstrap10'):
    paths=cohort_paths(cohort)
    cases=prepare_bootstrap(cohort)
    with gpu():
        signature('positions_v2_'+paths.signature_prefix,[Path(__file__),HERE/'variants.py',
                  HERE/'calibration_cohort.py',
                  ROOT/'experiments/math_distill_v1/selector_masking_probe.py',
                  ROOT/'experiments/math_distill_v1/selector_masking_core.py',
                  paths.cases,paths.relations,paths.reviews],
                  {'phase':'development_only','cohort':cohort,'layers':4,'queries':8,'random_controls':3})
        path=paths.diagnostics
        saved=checked_rows(path)
        assert [r['source_id'] for r in saved]==[c['source_id'] for c in cases[:len(saved)]]
        if len(saved)==len(cases):return
        model,_=load_model('student');backend=MLXBackend(model,4)
        import mlx.core as mx
        for case in cases[len(saved):]:
            check_time()
            start=time.perf_counter();steps=[]
            for candidate in case['candidates']:
                steps.append(diagnose_step(case['input_ids'],candidate['step'],candidate['units'],backend,
                    seed=7,key=f'{case["source_id"]}:{candidate["step"]["step_index"]}',controls=3,queries=8))
                mx.clear_cache()
            row={'source_id':case['source_id'],'teacher_output_sha256':case['teacher_output_sha256'],
                 'steps':steps,'proposal':propose(steps),'seconds':time.perf_counter()-start,
                 'development_only':True}
            append(path,row);saved.append(row)
            print(f'{cohort} diagnostics {len(saved)}/{len(cases)} {row["proposal"]}',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cohort',choices=COHORTS,default='bootstrap10');args=parser.parse_args()
    if args.prepare_only:prepare_bootstrap(args.cohort)
    else:main(args.cohort)
