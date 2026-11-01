"""GPU preflight against independent full-model logits; synthetic math only."""
from __future__ import annotations

from pathlib import Path
from runtime_full import *
from static_selection import MLXScorer, scoring_inputs


def reference(model, context, target):
    import numpy as np
    import mlx.core as mx
    full = context + target
    logits = model(mx.array([full[:-1]]))[0,len(context)-1:].astype(mx.float32)
    mx.eval(logits)
    array = np.asarray(logits)
    values = array[np.arange(len(target)),target]
    maximum = array.max(axis=-1)
    nll = maximum + np.log(np.exp(array-maximum[:,None]).sum(axis=-1))-values
    rank = np.minimum(1+np.sum(array>values[:,None],axis=-1),100)
    return {'nll':nll.tolist(), 'rank':rank.tolist()}


def main():
    with gpu():
        signature('static_scorer_preflight_v1',[Path(__file__),HERE/'static_selection.py',
            HERE/'runtime.py',HERE/'runtime_full.py',OUT/'references/static_selectors/source_manifest.json'],
            {'synthetic_question_only':True,'response_projection_chunk_tokens':2,'local_window':2})
        baseline.verify_student(BASE)
        model,tokenizer=load_model('student')
        question='A box contains 3 red pencils and 4 blue pencils. How many pencils are there?'
        completion=('There are 3 red pencils.\n\nThere are 4 blue pencils.\n\n'
                    'Each pencil belongs to exactly one color group.\n\nThere are 3 + 4 = 7 pencils in total.')
        prepared=scoring_inputs(tokenizer,BASE['student_prompt'].format(question=question),completion,window=2)
        scored=MLXScorer(model,chunk_tokens=2).score(prepared)
        independent=reference(model,prepared['context_ids'],prepared['target_ids'])
        actual=scored['global_token_statistics']
        max_error=max(abs(a-b) for a,b in zip(actual['surprisals'],independent['nll']))
        assert max_error<1e-4,max_error
        assert actual['ranks']==independent['rank'],'Clipped ranks do not match full model logits'
        expected_rsr=sum(independent['rank'])/max(sum(independent['nll']),1e-12)
        assert abs(scored['RSR']-expected_rsr)<1e-4
        local_reference=[reference(model,step['context_ids'],step['target_ids']) for step in prepared['steps']]
        expected_lalp=-sum(sum(row['nll'])/len(row['nll']) for row in local_reference)/len(local_reference)
        assert abs(scored['LALP']-expected_lalp)<1e-4
        last=prepared['steps'][-1]
        assert last['preceding_step_span'][0]>0,'Preflight must actually remove an early paragraph'
        assert sum(len(step['target_ids']) for step in prepared['steps'])==len(prepared['target_ids'])
        save(OUT/'preflight/static_scores_v1.json',{
            'status':'passed','at_utc':now(),'scored_tokens':len(prepared['target_ids']),
            'step_count':len(prepared['steps']),'max_global_nll_error':max_error,
            'RSR':scored['RSR'],'independent_RSR':expected_rsr,
            'LALP':scored['LALP'],'independent_LALP':expected_lalp,
            'checks':['next_token_alignment','independent_full_model_logits','exact_clipped_rank_ties',
                      'chunked_output_projection','prompt_always_retained','older_steps_removed',
                      'equal_step_weight','complete_response_token_partition'],
            'synthetic_data_only':True,'public_test_used':False})


if __name__=='__main__':main()
