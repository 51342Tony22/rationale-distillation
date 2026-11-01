"""Score the exact same reviewed completions before any student update."""
from __future__ import annotations

import time
from pathlib import Path
from runtime_full import *
from explanation_pool import unique
from calibration_cohort import COHORTS,cohort_paths
from static_selection import MLXScorer,scoring_inputs,select_static


def main(cohort='bootstrap10'):
    paths=cohort_paths(cohort);folder=paths.pool_folder
    gate=read(OUT/'preflight/static_scores_v1.json')
    if gate['status']!='passed':
        raise ValueError('Real-model static scoring preflight must pass')
    pools=read(folder/'reviewed_pools.json')
    if len(pools)!=paths.expected_roots or len({p['source_id'] for p in pools})!=len(pools):
        raise ValueError('Unexpected reviewed-pool root coverage')
    sources=unique(read(DATA/'cal40.json'),'source_id')
    path=folder/'static_scores.jsonl'
    signature(paths.signature_prefix+'_static_scores_v1',[Path(__file__),HERE/'static_selection.py',
        HERE/'calibration_cohort.py',
        HERE/'runtime.py',HERE/'runtime_full.py',folder/'reviewed_pools.json',DATA/'cal40.json',
        OUT/'references/static_selectors/source_manifest.json',OUT/'preflight/static_scores_v1.json'],
        {'RSR_rank_clip':100,'RSR_direction':'min','LALP_direction':'max','LALP_window_steps':2,
         'LALP_segmentation':'paragraphs','eos_scored':False,'student':'S0_no_updates',
         'nonempty_selection_except_no_qualified_candidates':True,'development_only':True})
    saved=checked_rows(path)
    if [row['source_id'] for row in saved] != [pool['source_id'] for pool in pools[:len(saved)]]:
        raise ValueError('Static scoring resume order changed')
    if len(saved)==len(pools):return
    with gpu():
        baseline.verify_student(BASE)
        model,tokenizer=load_model('student');scorer=MLXScorer(model)
        for pool in pools[len(saved):]:
            check_time()
            sid=pool['source_id'];source=sources[sid];start=time.perf_counter()
            if source['question_sha256']!=pool['question_sha256']:
                raise ValueError('Pool question changed')
            scores=[]
            for candidate in pool['candidates']:
                if digest(candidate['completion'])!=candidate['completion_sha256']:
                    raise ValueError('Pool completion changed')
                prepared=scoring_inputs(tokenizer,BASE['student_prompt'].format(question=source['question']),
                                        candidate['completion'],window=2)
                scores.append({'candidate_id':candidate['candidate_id'],
                    'completion_sha256':candidate['completion_sha256'],
                    'student_tokens':candidate['student_tokens'],**scorer.score(prepared)})
            append(path,{'source_id':sid,'candidates':scores,
                'RSR_selected_id':select_static(scores,'RSR'),'LALP_selected_id':select_static(scores,'LALP'),
                'seconds':time.perf_counter()-start,'public_test_used':False,'student_updated':False})
            print(sid,'static selection scored',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cohort',choices=COHORTS,default='bootstrap10')
    main(parser.parse_args().cohort)
