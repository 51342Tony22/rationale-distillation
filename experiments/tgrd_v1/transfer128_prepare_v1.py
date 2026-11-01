"""CPU preparation of the frozen128 joint two-position explanation extension.

Never samples new roots, loads a model, or creates a review/training gate.
Exact old prompts retain their one existing output and quality decision even
when the old request came from NLL; membership always comes from JOINT positions.
"""
from pathlib import Path
import argparse
from runtime_full import ROOT, HERE, OUT, DATA, read, checked_rows, freeze, digest, baseline
from budgeted_materials import PROTOCOL, validate_roster
from explanation_pool import TYPES, make_request, reusable_output, unique
from auxiliary_explainer_recipe import FULL, RECIPE_ID, RECIPE, load_recipe
from transfer128_sampling import ROSTER, verify_roster

FOLDER = OUT/'transfer128/preparation_v1'


def request_plan(sources, teachers, cases, joint, roster, common_ids,
                 old_requests, old_outputs, old_decisions):
    selected = validate_roster(roster, common_ids)
    if ([c['source_id'] for c in cases] != common_ids
            or [d['source_id'] for d in joint] != common_ids):
        raise ValueError('Complete frozen common-root order required')
    bycase=unique(cases,'source_id');diagnostics=unique(joint,'source_id')
    previous=unique(old_requests,'candidate_id');generated=unique(old_outputs,'candidate_id')
    decisions=unique(old_decisions,'candidate_id')
    requests=[];pending=[];registry=[];roots=[]
    for sid in roster['source_ids']:
        source=sources[sid];teacher=teachers[sid];case=bycase[sid];diagnostic=diagnostics[sid]
        if (sid not in selected or source['question_sha256'] != digest(source['question'])
                or teacher['source_id']!=sid or teacher['question_sha256']!=source['question_sha256']
                or teacher['output_sha256']!=digest(teacher['output'])
                or case['teacher_output_sha256']!=teacher['output_sha256']
                or diagnostic['teacher_output_sha256']!=teacher['output_sha256']
                or diagnostic.get('selector')!='joint'):
            raise ValueError('Frozen joint/root/teacher identity mismatch')
        steps={c['step']['step_index']:c['step'] for c in case['candidates']}
        positions=diagnostic['proposal']['positions'][:2]
        if len(set(positions))!=len(positions) or any(i not in steps for i in positions):
            raise ValueError('Joint proposal has invalid positions')
        root_requests=[]
        for rank,index in enumerate(positions):
            for kind in TYPES:
                request=make_request(source,teacher,steps[index],kind,rank)
                request.update(protocol=PROTOCOL,development_only=False,
                    phase='fixed128_joint_extension_v1')
                cid=request['candidate_id'];requests.append(request);root_requests.append(cid)
                if cid not in previous:
                    if cid in generated or cid in decisions:
                        raise ValueError('Historical output or quality record lacks its original request')
                    pending.append(request);continue
                if cid not in generated or cid not in decisions:
                    raise ValueError('Historical request lacks its complete immutable output/quality decision')
                row=generated[cid];decision=decisions[cid]
                reused=reusable_output(request,previous,generated)
                if (reused is None or row.get('recipe_id')!=RECIPE_ID
                        or digest(row['output'])!=row['output_sha256']
                        or row['request_sha256']!=digest(request['system']+'\n'+request['user'])
                        or decision['candidate_id']!=cid
                        or decision['output_sha256']!=row['output_sha256']
                        or decision['request_sha256']!=row['request_sha256']
                        or type(decision['eligible']) is not bool or not row.get('record_sha256')):
                    raise ValueError('Exact general7b output/quality provenance required for reuse')
                registry.append({'candidate_id':cid,'recipe_id':RECIPE_ID,
                    'request_sha256':row['request_sha256'],'output_sha256':row['output_sha256'],
                    'generated_record_sha256':row['record_sha256'],
                    'previous_request_sha256':digest(__import__('json').dumps(previous[cid],sort_keys=True)),
                    'quality_decision_sha256':digest(__import__('json').dumps(decision,sort_keys=True)),
                    'automatic_eligible':decision['eligible'],
                    'all_text_read':decision.get('all_text_read',False),
                    'semantic_correctness_verified':False})
        roots.append({'source_id':sid,'question':source['question'],
            'question_sha256':source['question_sha256'],'gold':source['gold'],
            'teacher_output_sha256':teacher['output_sha256'],
            'joint_positions':positions,'candidate_ids':root_requests,
            'probe_preparation_status':'awaiting_actual_template_review',
            'probes_qualified':False,'trial_executed':False,'replace_on_failure':False})
    return requests,pending,registry,roots


def prepare():
    load_recipe()
    from full_explanations import load_requests
    from full_selector_materials_quality_v3 import FOLDER as MATERIALS
    from explanation_quality_v3 import EXPORT, FOLDER as QUALITY, authority
    from train_selector_quality_v3 import verify_inputs
    authority();export=read(EXPORT);verify_inputs(export)
    parent,cases,sources,teachers,diagnostics,_,old_requests=load_requests()
    common_ids=[c['source_id'] for c in cases];roster=read(ROSTER)
    evidence=verify_roster(roster,common_ids)
    # checked_rows authenticates every original record before using record SHA.
    old_outputs=checked_rows(FULL/'generated.jsonl')
    decisions=read(QUALITY/'automatic_decisions_v3.json')
    requests,pending,registry,roots=request_plan(sources,teachers,cases,diagnostics['joint'],
        roster,common_ids,old_requests,old_outputs,decisions)
    files={'requests_joint128_v1.json':requests,'pending_generation_requests_v1.json':pending,
        'exact_reuse_registry_v1.json':registry,'root_preparation_v1.json':roots}
    for name,value in files.items():freeze(FOLDER/name,value)
    paths=[Path(__file__),HERE/'fixed_trial_engine_v1.py',ROSTER,RECIPE,
        FULL/'requests.json',FULL/'requests_manifest.json',FULL/'generated.jsonl',
        QUALITY/'automatic_decisions_v3.json',EXPORT,
        MATERIALS/'materials_manifest.json',OUT/'full_selector_v1/positions_joint_v1.jsonl',
        OUT/'full_selector_v1/cases_manifest.json',DATA/'train7473.json',
        *evidence,*[FOLDER/n for n in files]]
    manifest={'schema':'tgrd_transfer128_joint_preparation_v1','protocol':PROTOCOL,
        'recipe_id':RECIPE_ID,'source_ids':roster['source_ids'],'roots':128,
        'request_count':len(requests),'existing_exact_outputs':len(registry),
        'pending_single_attempt_generations':len(pending),'position_ranks':[0,1],
        'max_nonempty_candidates_per_root':4,'NLL_only_positions_enter_joint_pool':False,
        'exact_output_from_NLL_request_reusable_only_at_joint_position':True,
        'new_generation_executed':False,'qualified_probe_bundles_created':False,
        'formal_training_gate_created':False,'public_test_used':False,
        'quality_of_pending_outputs':'not_generated_not_automatically_or_semantically_accepted',
        'existing_quality_evidence':'exact_v3_decision_preserved_including_rejections',
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}}
    freeze(FOLDER/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare'])
    parser.parse_args();value=prepare()
    print({k:v for k,v in value.items() if k not in ('inputs','source_ids')},flush=True)
