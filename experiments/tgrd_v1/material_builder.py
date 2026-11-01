"""Five-group assembly with independent NLL and joint explanation pools.

The CLI builds development previews only. It never enables a training or test
gate. Missing results are errors; scientific abstention is an explicit e0.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from runtime_full import ROOT,HERE,OUT,DATA,BASE,read,checked_rows,freeze,digest,baseline
from calibration_cohort import COHORTS,cohort_paths
from explanation_pool import REVIEW_CHECKS,TYPES,assemble,insert_explanation,unique,selector_paths
from trial_selection import choose,validate_probes
from variants import build
from variant_rationales import root_rationales
from training_plan import CONDITIONS,PROTOCOL,validate_materials


def validate_pool(source,teacher,pool,steps):
    sid=source['source_id'];candidates=pool['candidates'];ids=[c['candidate_id'] for c in candidates]
    if (pool['source_id']!=sid or pool['question_sha256']!=source['question_sha256']
            or pool['teacher_output_sha256']!=teacher['output_sha256']
            or not 1<=len(candidates)<=5 or ids[0]!='e0' or len(set(ids))!=len(ids)):
        raise ValueError('Invalid common pool identity/coverage')
    if (candidates[0]['completion']!=teacher['output'] or candidates[0]['student_tokens']!=0
            or candidates[0]['completion_sha256']!=teacher['output_sha256']):
        raise ValueError('e0 is not the reviewed ordinary CoT')
    for c in candidates:
        if digest(c['completion'])!=c['completion_sha256']:raise ValueError('Changed pool completion')
        if c['candidate_id']=='e0':continue
        review=c['review'];kind=c['explanation_type'];index=c['step_index']
        if (kind not in TYPES or type(c['position_rank']) is not int or c['position_rank'] not in (0,1)
                or c['candidate_id']!=f'{sid}:step{index}:{kind}' or index not in steps
                or c['source_id']!=sid or c['teacher_output_sha256']!=teacher['output_sha256']
                or c['text_sha256']!=digest(c['text']) or not 32<=c['student_tokens']<=64
                or review['candidate_id']!=c['candidate_id'] or review['accepted'] is not True
                or review.get('kind')!='assistant_mathematical_review' or not review.get('reason')
                or any(review.get(key) is not True for key in REVIEW_CHECKS)):
            raise ValueError('Candidate lacks required common-pool semantic provenance')
        if insert_explanation(teacher['output'],steps[index],c['text'])!=c['completion']:
            raise ValueError('Candidate insertion differs from the reviewed step')
        if 'position_proposal' in pool:
            positions=pool['position_proposal']['positions']
            if len(positions)<=c['position_rank'] or positions[c['position_rank']]!=index:
                raise ValueError('Candidate differs from its selector position/rank')
    return candidates


def transfer_choice(source,pool,probes,inputs,decisions,*,seed,updates,threshold):
    expected={'source_id':source['source_id'],'question_sha256':source['question_sha256'],
        'reviewed_pool_sha256':digest(json.dumps(pool,sort_keys=True)),
        'probe_bundle_sha256':digest(json.dumps(probes,sort_keys=True)),'seed':seed,'updates':updates,
        'thresholds':[0,.01,.03,.10]}
    if any(inputs.get(key)!=value for key,value in expected.items()):
        raise ValueError('Trial inputs do not match the common pool and exact probes')
    if (set(decisions)!={'source_id','decisions','Vsel_scores','untrained_Vsel','audit_can_change_selection'}
            or decisions['source_id']!=source['source_id'] or decisions['audit_can_change_selection'] is not False):
        raise ValueError('Only frozen before-audit decisions may enter material assembly')
    expected_decisions=[choose(decisions['Vsel_scores'],pool['candidates'],t) for t in (0,.01,.03,.10)]
    if decisions['decisions']!=expected_decisions:raise ValueError('Frozen decision does not reproduce Vsel-only choice')
    return next(d for d in expected_decisions if d['threshold']==threshold)['selected_id']


def fixed_choice(candidates):
    first=[c for c in candidates if c['candidate_id']!='e0' and c['position_rank']==0]
    return min(first,key=lambda c:(TYPES.index(c['explanation_type']),c['candidate_id']))['candidate_id'] if first else 'e0'


def validate_nll_pool(source,teacher,pool,steps):
    from positions_nll import propose_nll
    if pool is None or pool.get('selector')!='nll_only':
        raise ValueError('Independent all-step NLL pool is required, even when it has no accepted explanation')
    proposal=pool['position_proposal']
    if ({s['step_index'] for s in proposal['scores']}!=set(steps)
            or proposal!=propose_nll(proposal['scores'])):
        raise ValueError('NLL pool must rank ALL eligible steps without joint candidate filtering')
    candidates=validate_pool(source,teacher,pool,steps)
    for candidate in candidates[1:]:
        if (len(proposal['positions'])<=candidate['position_rank']
                or candidate['step_index']!=proposal['positions'][candidate['position_rank']]):
            raise ValueError('NLL candidate position/rank differs from independent proposal')
    return candidates


def root_materials(source,teacher,review,pool,steps,*,nll_pool=None,variant_bundle=None,
                   trial_inputs=None,trial_decisions=None,trial_seed=701,updates=3,threshold=.03):
    sid=source['source_id']
    from automatic_teacher_quality import MODE as AUTO_MODE, validate_decision
    automatic = review.get('review_kind') == AUTO_MODE
    if automatic:
        validate_decision(source, teacher, review)
    if updates not in (1,3,5) or threshold not in (0,.01,.03,.10) or trial_seed not in (701,1701):
        raise ValueError('Undeclared development trial configuration')
    if (not sid.startswith('gsm8k:train:') or digest(source['question'])!=source['question_sha256']
            or teacher['source_id']!=sid or teacher['question_sha256']!=source['question_sha256']
            or teacher.get('correct') is not True or teacher.get('gold')!=source['gold']
            or teacher.get('finish_reason')!='stop'
            or digest(teacher['output'])!=teacher['output_sha256'] or review['source_id']!=sid
            or review['accepted'] is not True
            or (not automatic and (review['all_text_read'] is not True
                or review.get('review_kind')!='assistant_full_text_mathematical_review'))
            or not review.get('reason')
            or review['question_sha256']!=source['question_sha256']
            or review['output_sha256']!=teacher['output_sha256']):
        raise ValueError('Common root must have exact full-text teacher acceptance')
    candidates=validate_pool(source,teacher,pool,steps);byid=unique(candidates,'candidate_id')
    nll_candidates=validate_nll_pool(source,teacher,nll_pool,steps)
    nll_byid=unique(nll_candidates,'candidate_id')
    for cid in set(byid)&set(nll_byid):
        if byid[cid]['completion_sha256']!=nll_byid[cid]['completion_sha256']:
            raise ValueError('Overlapping positions/types must reuse identical teacher explanation')
    selected={'M1':'e0','NLL':fixed_choice(nll_candidates),'M3':fixed_choice(candidates),'M6':'e0'};reasons={}
    for group in ('NLL','M3'):
        if selected[group]=='e0':reasons[group]='top_proposed_position_has_no_qualified_explanation'
    nonempty=len(candidates)>1
    probes=validate_probes(variant_bundle,source,build) if variant_bundle is not None else None
    if nonempty and probes is not None:
        if trial_inputs is None or trial_decisions is None:
            raise ValueError('Missing trial result is not scientific abstention')
        selected['M7']=transfer_choice(source,pool,probes,trial_inputs,trial_decisions,
            seed=trial_seed,updates=updates,threshold=threshold)
        if selected['M7']=='e0':reasons['M7']='no_explanation_passed_frozen_Vsel_gain_rule'
    else:
        selected['M7']='e0';reasons['M7']='no_qualified_joint_pool_explanation' if not nonempty else 'no_qualified_complete_6plus6_probes'
    verified=root_rationales(variant_bundle,source) if probes is not None else None
    if verified is None:reasons['M6']='no_qualified_complete_6plus6_probes'
    def material(group,cid):
        c=(nll_byid if group=='NLL' else byid)[cid]
        return {'root_source_id':sid,'question':source['question'],'question_sha256':source['question_sha256'],
            'completion':c['completion'],'completion_sha256':c['completion_sha256'],
            'kind':'ordinary_CoT' if cid=='e0' else 'explained_CoT','selected_candidate_id':cid}
    rows={group:{'source_id':sid,'question_sha256':source['question_sha256'],
        'epoch_materials':[copy.deepcopy(material(group,selected[group])) for _ in range(4)]}
        for group in CONDITIONS}
    if verified is not None:
        for epoch,variant in zip((1,3),verified['selection_rationales'][:2]):
            rows['M6']['epoch_materials'][epoch]=copy.deepcopy(variant)
    return rows,{'source_id':sid,'protocol':PROTOCOL,'selected_candidate_ids':selected,'fallback_reasons':reasons,
        'NLL_pool_sha256':digest(json.dumps(nll_pool,sort_keys=True)),
        'joint_pool_sha256':digest(json.dumps(pool,sort_keys=True)),
        'trial_seed':trial_seed,'trial_updates':updates,'threshold':threshold,
        'audit_predictions_read':False,'variant_labels_in_formal_gradients_only_for':['M6'],
        'M6_variant_ids':verified['M6_fixed_variant_ids'] if verified else []}


def development_paths(cohort,seed,updates):
    paths=cohort_paths(cohort);folder=paths.trials_folder/f'seed{seed}_updates{updates}'
    required=[paths.pool_folder/'reviewed_pools.json',selector_paths(cohort,'nll').pool_folder/'reviewed_pools.json']
    for case in read(paths.cases):
        root_folder=folder/case['source_id'].replace(':','_')
        required.extend([root_folder/'inputs.json',root_folder/'decisions_before_audit.json'])
    return paths,folder,required


def prepare_development(cohort,seed,updates,threshold,check_only=False):
    paths,trial_folder,required=development_paths(cohort,seed,updates)
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        if check_only:
            print({'ready':False,'missing_count':len(missing),'first_missing':missing[:3],
                'formal_training_enabled':False},flush=True);return
        raise ValueError('Development materials not ready: '+', '.join(missing[:3]))
    if check_only:
        print({'ready_to_validate':True,'formal_training_enabled':False},flush=True);return
    # Reconstruct immutable pools from exact generated outputs, requests and
    # semantic reviews, rather than trusting a manually supplied pool JSON.
    assemble(cohort)
    assemble(cohort,'nll')
    from transformers import AutoTokenizer
    from learning import tokenize
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']),local_files_only=True,trust_remote_code=False)
    sources=unique(read(DATA/'cal40.json'),'source_id')
    teacher_path=OUT/'materials/teacher_targets_cal40.jsonl'
    teachers=unique(checked_rows(teacher_path),'source_id')
    reviews=unique(read(paths.reviews),'source_id');pools=unique(read(paths.pool_folder/'reviewed_pools.json'),'source_id')
    nll_pools=unique(read(selector_paths(cohort,'nll').pool_folder/'reviewed_pools.json'),'source_id')
    variants=unique(read(paths.variants),'source_id');cases=read(paths.cases)
    ids=[c['source_id'] for c in cases]
    if set(pools)!=set(ids) or set(nll_pools)!=set(ids) or set(variants)!=set(ids):
        raise ValueError('All five groups require the exact same cohort coverage')
    bundles={g:{'schema':'tgrd_training_materials_v1','condition':g,'candidate_root_count':7473,
        'development_preview_only':True,'protocol':PROTOCOL,'rows':[]} for g in CONDITIONS};choices=[]
    for case in cases:
        sid=case['source_id'];folder=trial_folder/sid.replace(':','_')
        rows,decision=root_materials(sources[sid],teachers[sid],reviews[sid],pools[sid],
            {c['step']['step_index']:c['step'] for c in case['candidates']},nll_pool=nll_pools[sid],
            variant_bundle=variants[sid],trial_inputs=read(folder/'inputs.json'),
            trial_decisions=read(folder/'decisions_before_audit.json'),
            trial_seed=seed,updates=updates,threshold=threshold)
        for group,row in rows.items():
            for material in row['epoch_materials']:
                sample=tokenize(tokenizer,material['question'],material['completion'])
                material['completion_tokens_including_eos']=len(sample['target_ids'])
            bundles[group]['rows'].append(row)
        choices.append(decision)
    for bundle in bundles.values():validate_materials(bundle,ids)
    folder=paths.pool_folder.parent/'material_previews'/f'{PROTOCOL}_seed{seed}_updates{updates}_threshold{threshold}'
    inputs=[Path(__file__),HERE/'positions_nll.py',HERE/'trial_selection.py',
        HERE/'variants.py',HERE/'variant_rationales.py',HERE/'explanation_pool.py',HERE/'training_plan.py',HERE/'learning.py',
        HERE/'runtime.py',HERE/'runtime_full.py',HERE/'calibration_cohort.py',
        paths.reviews,paths.cases,paths.variants,DATA/'cal40.json',teacher_path,
        ROOT/'experiments/math_distill_v1/config.json',*required]
    for selector in ('joint','nll'):
        selected_paths=selector_paths(cohort,selector)
        inputs.extend(p for p in (selected_paths.diagnostics,selected_paths.pool_folder/'requests.json',
            selected_paths.pool_folder/'generated.jsonl',selected_paths.pool_folder/'full_text_reviews.json') if p.exists())
    freeze(folder/'inputs.json',{'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in inputs},
        'cohort':cohort,'development_preview_only':True,'public_test_used':False,
        'protocol':PROTOCOL,'M6_policy':'first_two_frozen_Vsel_instances_epochs2and4',
        'NLL_and_M3_policy':'rank0_only_relation_then_contrastive',
        'NLL_eligibility':'all_filtered_equation_steps_without_attention_or_mask_control_filter','trial_seed':seed,'updates':updates,'threshold':threshold})
    for group,bundle in bundles.items():freeze(folder/f'{group}.json',bundle)
    freeze(folder/'choices.json',choices)
    freeze(folder/'coverage.json',{'development_preview_only':True,'candidate_root_count':7473,
        'preview_roots_per_group':len(ids),'formal_training_enabled':False,
        'groups':{group:{
            'root_exposures':4*len(ids),
            'roots_with_explanation':sum(any(m['kind']=='explained_CoT' for m in row['epoch_materials']) for row in bundle['rows']),
            'roots_with_verified_variant':sum(any(m['kind']=='verified_Vsel_variant' for m in row['epoch_materials']) for row in bundle['rows']),
            'ordinary_only_roots':sum(all(m['kind']=='ordinary_CoT' for m in row['epoch_materials']) for row in bundle['rows']),
            'supervised_tokens_including_eos':sum(m['completion_tokens_including_eos'] for row in bundle['rows'] for m in row['epoch_materials'])
        } for group,bundle in bundles.items()}})
    print({'groups':len(bundles),'roots_per_group':len(ids),'development_preview_only':True,
        'formal_training_enabled':False,'audit_predictions_read':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,default='cal40_reviewed34_v2')
    parser.add_argument('--trial-seed',type=int,choices=(701,1701),default=701)
    parser.add_argument('--updates',type=int,choices=(1,3,5),default=3)
    parser.add_argument('--threshold',type=float,choices=(0,.01,.03,.10),default=.03)
    parser.add_argument('--check-prerequisites',action='store_true')
    a=parser.parse_args();prepare_development(a.cohort,a.trial_seed,a.updates,a.threshold,a.check_prerequisites)
