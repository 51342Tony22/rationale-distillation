"""Bounded joint-only extension; original outputs and v3 sample stay immutable.

One attempt per frozen request. An interrupted attempt is conservatively
excluded on resume, never regenerated. Automatic eligibility is not review.
"""
from pathlib import Path
import argparse
import json
from runtime_full import (ROOT, HERE, OUT, read, freeze, append, checked_rows,
                          digest, baseline, signature, gpu, check_time)
from budgeted_materials import PROTOCOL
from auxiliary_explainer_recipe import FULL, RECIPE, RECIPE_ID, MODEL, load_recipe
from explanation_pool import unique, generate_one, mechanical_checks, insert_explanation
from explanation_quality_v3 import automatic_decision, authority, EXPORT, FOLDER as QUALITY
from train_selector_quality_v3 import verify_inputs
from transfer128_prepare_v1 import FOLDER as PREPARATION, request_plan

FOLDER = OUT/'transfer128/explanations_v1'
PLAN = FOLDER/'generation_plan_v1.json'


def validate_prefix(saved, pending, tokenizer):
    unique(saved, 'candidate_id')
    if len(saved)>len(pending): raise ValueError('Too many extension outputs')
    for row, request in zip(saved,pending):
        expected = {'candidate_id':request['candidate_id'], 'source_id':request['source_id'],
            'recipe_id':RECIPE_ID, 'protocol':PROTOCOL,
            'request_sha256':digest(request['system']+'\n'+request['user']),
            'question_sha256':request['question_sha256'],
            'teacher_output_sha256':request['teacher_output_sha256']}
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('Extension output order, recipe or prompt changed')
        if row['mechanical_checks']!=mechanical_checks(request,row,tokenizer):
            raise ValueError('Mechanical checks must replay exactly')
        if row.get('all_text_read') is not False or row.get('semantic_correctness_verified') is not False:
            raise ValueError('Generation cannot create a semantic judgment')


def output_row(request, output, tokenizer):
    row={**output,'candidate_id':request['candidate_id'],'source_id':request['source_id'],
        'protocol':PROTOCOL,'recipe_id':RECIPE_ID,
        'question_sha256':request['question_sha256'],
        'teacher_output_sha256':request['teacher_output_sha256'],
        'request_sha256':digest(request['system']+'\n'+request['user']),
        'all_text_read':False,'semantic_correctness_verified':False,
        'attempt_policy':'one_attempt_interruption_excluded_no_retry'}
    row['mechanical_checks']=mechanical_checks(request,row,tokenizer)
    return row


def generate_pending(pending, folder, tokenizer, generate_callback):
    """Caller owns GPU lock; per-request intent survives an interrupted call."""
    folder=Path(folder);path=folder/'generated.jsonl'
    saved=checked_rows(path);validate_prefix(saved,pending,tokenizer)
    for request in pending[len(saved):]:
        check_time()
        intent=folder/'attempts'/(digest(request['candidate_id'])+'.json')
        identity={'candidate_id':request['candidate_id'],
            'request_sha256':digest(request['system']+'\n'+request['user']),
            'recipe_id':RECIPE_ID,'attempt':1}
        if intent.exists():
            if read(intent)!=identity: raise ValueError('Attempt identity changed')
            output={'output':'','output_sha256':digest(''),'generation_tokens':0,
                'seconds':0,'finish_reason':'interrupted_attempt_excluded_no_retry',
                'generation_tokens_before_interruption_unknown':True}
        else:
            freeze(intent,identity)
            output=generate_callback(request)
        row=append(path,output_row(request,output,tokenizer));saved.append(row)
        print({'extension_outputs_committed':len(saved),'total':len(pending)},flush=True)
    validate_prefix(saved,pending,tokenizer)
    return saved


def prepare():
    authority();load_recipe();verify_inputs(read(EXPORT))
    parent=read(PREPARATION/'manifest.json');verify_inputs(parent)
    from full_explanations import load_requests
    from transfer128_sampling import ROSTER
    _,cases,sources,teachers,diagnostics,_,old_requests=load_requests()
    old_outputs=checked_rows(FULL/'generated.jsonl')
    old_decisions=read(QUALITY/'automatic_decisions_v3.json')
    actual=request_plan(sources,teachers,cases,diagnostics['joint'],read(ROSTER),
        [c['source_id'] for c in cases],old_requests,old_outputs,old_decisions)
    names=['requests_joint128_v1.json','pending_generation_requests_v1.json',
           'exact_reuse_registry_v1.json','root_preparation_v1.json']
    for name,value in zip(names,actual):
        if value!=read(PREPARATION/name):raise ValueError('Frozen preparation does not reconstruct')
    requests,pending,reused,roots=actual
    if (len(requests),len(pending),len(reused),len(roots))!=(412,142,270,128):
        raise ValueError('Expected the exact412/142/270/128 frozen plan')
    paths=[Path(__file__),HERE/'transfer128_prepare_v1.py',HERE/'runtime.py',HERE/'runtime_full.py',
        HERE/'explanation_pool.py',HERE/'explanation_quality_v3.py',HERE/'learning.py',
        HERE/'auxiliary_explainer_recipe.py',PREPARATION/'manifest.json',
        EXPORT,RECIPE,OUT/'explanation_quality_workflow_amendment_v3.json',
        QUALITY/'spotcheck_report_v3.json',*[PREPARATION/n for n in names]]
    paths += [ROOT/n for n in parent['inputs']]
    value={'schema':'tgrd_transfer128_explanation_plan_v1','protocol':PROTOCOL,
        'recipe_id':RECIPE_ID,'pending_generations':142,'exact_output_reuses':270,
        'total_joint_requests':412,'frozen_roots':128,'temperature':0,'max_new_tokens':256,
        'attempts_per_request':1,'interrupted_attempt_policy':'exclude_without_retry',
        'quality_policy':'same_v3_automatic_checks_with_original_fixed128_report',
        'original_fixed128_sample_extended_or_repeated':False,
        'new142_covered_by_original_descriptive_sample':False,
        'all_text_read':False,'semantic_correctness_verified':False,
        'NLL_only_positions_enter_pool':False,'scientific_trial_enabled':False,
        'public_test_used':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}}
    freeze(PLAN,value);return value


def load_plan():
    plan=read(PLAN);verify_inputs(plan)
    if (plan.get('pending_generations')!=142 or plan.get('total_joint_requests')!=412
            or plan.get('recipe_id')!=RECIPE_ID or plan.get('attempts_per_request')!=1
            or plan.get('interrupted_attempt_policy')!='exclude_without_retry'):
        raise ValueError('The frozen bounded142 extension is required')
    return plan


def generate():
    plan=load_plan()
    from full_explanations import student_tokenizer
    tokenizer=student_tokenizer();pending=read(PREPARATION/'pending_generation_requests_v1.json')
    with gpu():
        signature('transfer128_joint_explanations_v1',
            [PLAN,*[ROOT/n for n in plan['inputs']]],
            {'recipe_id':RECIPE_ID,'attempts_per_request':1,'public_test_used':False})
        load_recipe(verify_model=True)
        model=teacher_tokenizer=None
        def run(request):
            nonlocal model,teacher_tokenizer
            if model is None:
                import mlx.core as mx
                from mlx_lm import load
                mx.set_memory_limit(14*1024**3);mx.set_cache_limit(512*1024**2)
                model,teacher_tokenizer=load(str(MODEL),tokenizer_config={
                    'local_files_only':True,'trust_remote_code':False})
            return generate_one(model,teacher_tokenizer,request)
        generate_pending(pending,FOLDER,tokenizer,run)
    paths=[PLAN,FOLDER/'generated.jsonl',OUT/'signatures/transfer128_joint_explanations_v1.json']
    value={'schema':'tgrd_transfer128_explanation_generation_complete_v1','complete':True,
        'outputs':len(pending),'all_text_read':False,'semantic_correctness_verified':False,
        'public_test_used':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    freeze(FOLDER/'generation_complete_v1.json',value);return value


def combine(requests,new_outputs,old_outputs,old_decisions,registry,sources,teachers,tokenizer):
    new=unique(new_outputs,'candidate_id');old=unique(old_outputs,'candidate_id')
    decisions=unique(old_decisions,'candidate_id');reuses=unique(registry,'candidate_id')
    if set(new)&set(reuses) or set(new)|set(reuses)!={r['candidate_id'] for r in requests}:
        raise ValueError('Exact one-output coverage of the joint-only plan required')
    result=[];outputs={}
    for q in requests:
        cid=q['candidate_id']
        if cid in reuses:
            row=old[cid];d=decisions[cid];r=reuses[cid]
            if (r['generated_record_sha256']!=row['record_sha256']
                    or r['quality_decision_sha256']!=digest(json.dumps(d,sort_keys=True))
                    or r['output_sha256']!=row['output_sha256']
                    or d['output_sha256']!=row['output_sha256']
                    or d['request_sha256']!=digest(q['system']+'\n'+q['user'])
                    or digest(row['output'])!=row['output_sha256']
                    or r['automatic_eligible']!=d['eligible']):
                raise ValueError('Existing exact output/quality must remain unchanged')
        else:
            row=new[cid]
            d=automatic_decision(q,row,sources[q['source_id']],teachers[q['source_id']],tokenizer)
        result.append(d);outputs[cid]=row
    return result,outputs


def export():
    load_plan();verify_inputs(read(FOLDER/'generation_complete_v1.json'))
    from full_explanations import student_tokenizer
    from full_selector import load_quality
    source_rows,teacher_rows,_,_=load_quality()
    sources=unique(source_rows,'source_id');teachers=unique(teacher_rows,'source_id')
    tokenizer=student_tokenizer()
    pending=read(PREPARATION/'pending_generation_requests_v1.json')
    saved=checked_rows(FOLDER/'generated.jsonl');validate_prefix(saved,pending,tokenizer)
    if len(saved)!=len(pending):raise ValueError('All142 terminal attempts must be committed')
    requests=read(PREPARATION/'requests_joint128_v1.json')
    decisions,outputs=combine(requests,saved,checked_rows(FULL/'generated.jsonl'),
        read(QUALITY/'automatic_decisions_v3.json'),read(PREPARATION/'exact_reuse_registry_v1.json'),
        sources,teachers,tokenizer)
    pools=joint_pools(read(PREPARATION/'root_preparation_v1.json'),requests,outputs,decisions,teachers)
    freeze(FOLDER/'automatic_decisions_v1.json',decisions)
    freeze(FOLDER/'joint_pools_v1.json',pools)
    paths=[PLAN,FOLDER/'generation_complete_v1.json',FOLDER/'generated.jsonl',
        FOLDER/'automatic_decisions_v1.json',FOLDER/'joint_pools_v1.json',EXPORT,
        QUALITY/'spotcheck_report_v3.json',PREPARATION/'manifest.json']
    result={'schema':'tgrd_transfer128_explanation_automatic_export_v1','complete':True,
        'requests':412,'new_outputs':142,'exact_output_reuses':270,
        'automatically_eligible':sum(d['eligible'] for d in decisions),
        'excluded':sum(not d['eligible'] for d in decisions),
        'new_automatically_eligible':sum(d['eligible'] for d in decisions if d['candidate_id'] in {r['candidate_id'] for r in saved}),
        'all_text_read':False,'semantic_correctness_verified':False,
        'new142_covered_by_original_descriptive_sample':False,
        'scientific_trial_enabled':False,'public_test_used':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    freeze(FOLDER/'automatic_export_v1.json',result);return result


def joint_pools(roots,requests,outputs,decisions,teachers):
    req=unique(requests,'candidate_id');checked=unique(decisions,'candidate_id');pools=[]
    for root in roots:
        sid=root['source_id'];teacher=teachers[sid]
        pool={'source_id':sid,'selector':'joint','G4_full_trial_pool':True,
            'question_sha256':root['question_sha256'],'teacher_output_sha256':teacher['output_sha256'],
            'joint_positions':root['joint_positions'],'candidates':[{'candidate_id':'e0',
                'completion':teacher['output'],'completion_sha256':teacher['output_sha256'],
                'explanation_type':'empty','student_tokens':0}],'rejected':[]}
        for cid in root['candidate_ids']:
            q=req[cid];d=checked[cid];row=outputs[cid]
            if (q['source_id']!=sid or q['step']['step_index'] not in root['joint_positions'][:2]
                    or q['position_rank']!=root['joint_positions'].index(q['step']['step_index'])
                    or d['output_sha256']!=row['output_sha256']):
                raise ValueError('Only exact joint positions/output may enter G4')
            if not d['eligible']:
                pool['rejected'].append({'candidate_id':cid,'reasons':d['reasons']});continue
            completion=insert_explanation(teacher['output'],q['step'],row['output'])
            pool['candidates'].append({'candidate_id':cid,'source_id':sid,
                'text':row['output'],'text_sha256':row['output_sha256'],
                'completion':completion,'completion_sha256':digest(completion),
                'position_rank':q['position_rank'],'step_index':q['step']['step_index'],
                'explanation_type':q['explanation_type'],'student_tokens':d['student_tokens'],
                'quality_provenance':d})
        if len(pool['candidates'])>5:raise ValueError('At most4 explanations plus e0')
        pools.append(pool)
    return pools


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','generate','export'])
    args=parser.parse_args();value=globals()[args.action]()
    print({k:v for k,v in value.items() if k!='inputs'},flush=True)
