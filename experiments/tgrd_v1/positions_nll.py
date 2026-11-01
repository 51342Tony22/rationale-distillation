"""Independent teacher-forced NLL ranking of ALL eligible equation steps.

No attention observation, deletion, or mask-control eligibility enters this
runner. The candidate parser is shared with the joint selector for comparability.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path
from runtime_full import HERE,BASE,baseline,read,checked_rows,append,signature,gpu,load_model,check_time
from calibration_cohort import COHORTS,cohort_paths
from selector_masking_probe import MLXBackend


def diagnostic_path(cohort):
    return cohort_paths(cohort).pool_folder.parent/'positions_nll_v1.jsonl'


def propose_nll(scores):
    ids=[s['step_index'] for s in scores]
    if len(ids)!=len(set(ids)) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('NLL scores require unique nonnegative step indices')
    if any(not math.isfinite(s['mean_nll']) or s['mean_nll']<0 for s in scores):
        raise ValueError('Every eligible step needs finite nonnegative NLL')
    ranked=sorted(scores,key=lambda s:(-s['mean_nll'],s['step_index']))
    return {'positions':[s['step_index'] for s in ranked[:2]],
        'scores':[{'step_index':s['step_index'],'mean_nll':s['mean_nll']} for s in ranked],
        'status':'all_eligible_steps_NLL_descending' if scores else 'no_eligible_steps',
        'attention_used':False,'masking_used':False,'mask_control_filter_used':False}


def score_case(case,backend):
    scores=[]
    for candidate in case['candidates']:
        step=candidate['step'];targets=step['targets']
        if not targets or any(type(p) is not int or p<1 for p in targets):
            raise ValueError('Eligible NLL step has invalid target indices')
        # Exclude all later solution tokens from the forward pass.
        prefix=case['input_ids'][:max(targets)+1]
        losses=backend.score(prefix,targets)
        if len(losses)!=len(targets) or any(not math.isfinite(x) or x<0 for x in losses):
            raise ValueError('Invalid original-step NLL observations')
        scores.append({'step_index':step['step_index'],'mean_nll':statistics.mean(losses),
            'targets':targets,'target_ids':[case['input_ids'][p] for p in targets],
            'prefix_tokens':len(prefix)})
    return {'source_id':case['source_id'],'teacher_output_sha256':case['teacher_output_sha256'],
        'steps':scores,'proposal':propose_nll(scores),'development_only':True}


def main(cohort):
    paths=cohort_paths(cohort);cases=read(paths.cases);path=diagnostic_path(cohort)
    signature('positions_nll_'+paths.signature_prefix+'_v1',[
        Path(__file__),HERE/'calibration_cohort.py',HERE/'runtime.py',HERE/'runtime_full.py',
        HERE.parent/'math_distill_v1/selector_masking_probe.py',
        paths.cases,paths.reviews],{'top_k':2,'all_eligible_steps':True,
        'attention_used':False,'masking_used':False,'tie_break':'ascending_step_index'})
    saved=checked_rows(path)
    if len(cases)!=paths.expected_roots or len(saved)>len(cases):raise ValueError('NLL cohort coverage changed')
    for row,case in zip(saved,cases):
        if (row['source_id']!=case['source_id'] or row['teacher_output_sha256']!=case['teacher_output_sha256']
                or {s['step_index'] for s in row['steps']}!={c['step']['step_index'] for c in case['candidates']}
                or row['proposal']!=propose_nll(row['steps'])):
            raise ValueError('NLL resume provenance/ranking mismatch')
    if len(saved)==len(cases):return
    with gpu():
        baseline.verify_student(BASE)
        model,_=load_model('student');backend=MLXBackend(model,4)
        import mlx.core as mx
        for case in cases[len(saved):]:
            check_time();start=time.perf_counter();row=score_case(case,backend)
            row['seconds']=time.perf_counter()-start;append(path,row);mx.clear_cache()
            print(cohort,case['source_id'],row['proposal']['positions'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,required=True)
    main(parser.parse_args().cohort)
