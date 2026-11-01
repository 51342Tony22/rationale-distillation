"""Complete-grid development analysis, never an independent audit or test gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from runtime_full import ROOT,HERE,OUT,DATA,read,freeze,checked_rows,digest,baseline,now
from calibration_cohort import COHORTS,cohort_paths
from trial_engine import THRESHOLDS
from trial_selection import summarize,choose,validate_probes
from variants import build

SEEDS=(701,1701)
UPDATES=(1,3,5)
RULE={'schema':'tgrd_calibration_summary_rule_v1','primary_selection_seed':701,
      'repeatability_seed':1701,'updates':list(UPDATES),'thresholds':list(THRESHOLDS),
      'criterion':'mean_root_paired_Vaudit_accuracy_gain_vs_equally_trained_e0',
      'tie_order':['fewer_updates','higher_threshold'],
      'nonpositive_best':'report_no_support_and_do_not_auto_scale',
      'repeat_seed_used_to_choose_hyperparameters':False,
      'Vaudit_role':'hyperparameter_development_not_independent_confirmation',
      'all_roots_including_fallbacks_required':True,'formal_training_enabled':False,
      'public_test_used':False}


def aggregate(cells,root_ids):
    if not root_ids or len(set(root_ids))!=len(root_ids):raise ValueError('Unique complete root roster required')
    expected={(s,u,t) for s in SEEDS for u in UPDATES for t in THRESHOLDS}
    keyed={}
    for cell in cells:
        key=(cell['seed'],cell['updates'],cell['threshold'])
        if key in keyed or key not in expected:raise ValueError('Duplicate or undeclared calibration cell')
        roots=cell['roots']
        if [r['source_id'] for r in roots]!=root_ids:raise ValueError('Every configuration requires the same complete ordered roots')
        for row in roots:
            if (any(type(row[k]) is not int or not 0<=row[k]<=6 for k in
                    ('baseline_correct','selected_correct','untrained_correct'))
                    or not math.isfinite(row['Vaudit_nll_gain']) or not row['selected_id']):
                raise ValueError('Invalid six-probe counts or NLL')
            if row['selected_id']=='e0' and (row['baseline_correct']!=row['selected_correct']
                    or row['Vaudit_nll_gain']!=0):raise ValueError('An e0 fallback must reuse exactly its baseline audit')
        keyed[key]=cell
    if set(keyed)!=expected:raise ValueError('All12 configurations for both seeds required; no partial-grid choice')
    summaries=[]
    for seed,updates,threshold in sorted(keyed):
        roots=keyed[seed,updates,threshold]['roots']
        differences=[r['selected_correct']-r['baseline_correct'] for r in roots]
        summaries.append({'seed':seed,'updates':updates,'threshold':threshold,'roots':len(roots),
            'paired_correct_count_gain':sum(differences),
            'mean_paired_Vaudit_accuracy_gain':sum(differences)/(6*len(roots)),
            'mean_Vaudit_nll_gain':sum(r['Vaudit_nll_gain'] for r in roots)/len(roots),
            'selected_nonempty_roots':sum(r['selected_id']!='e0' for r in roots),
            'audit_positive_roots':sum(d>0 for d in differences),
            'audit_negative_roots':sum(d<0 for d in differences),
            'audit_zero_roots':sum(d==0 for d in differences),
            'untrained_Vsel_ceiling_roots':sum(r['untrained_correct']==6 for r in roots),
            'selected_ids':[r['selected_id'] for r in roots]})
    primary=[c for c in summaries if c['seed']==701]
    best=min(primary,key=lambda c:(-c['paired_correct_count_gain'],c['updates'],-c['threshold']))
    supported=best['paired_correct_count_gain']>0
    repeat=next(c for c in summaries if c['seed']==1701 and
                (c['updates'],c['threshold'])==(best['updates'],best['threshold']))
    stability={'configuration':{'updates':best['updates'],'threshold':best['threshold']},
        'configuration_role':'primary_selected' if supported else 'best_nonpositive_descriptive_only',
        'candidate_agreement_roots':sum(a==b for a,b in zip(best['selected_ids'],repeat['selected_ids'])),
        'candidate_agreement_rate':sum(a==b for a,b in zip(best['selected_ids'],repeat['selected_ids']))/len(root_ids),
        'primary_mean_gain':best['mean_paired_Vaudit_accuracy_gain'],
        'repeat_mean_gain':repeat['mean_paired_Vaudit_accuracy_gain'],
        'repeat_gain_positive':repeat['paired_correct_count_gain']>0,
        'repeat_seed_did_not_reselect_configuration':True}
    return {'rule':RULE,'root_ids':root_ids,'all24_seed_configuration_cells_complete':True,
        'configuration_summaries':summaries,
        'status':'positive_development_signal_requires_confirmation' if supported else 'no_positive_development_support',
        'selected_configuration':{'updates':best['updates'],'threshold':best['threshold'],'trial_seed':701} if supported else None,
        'best_primary_development_mean_gain':best['mean_paired_Vaudit_accuracy_gain'],
        'stability':stability,'independent_confirmation_passed':False,
        'formal_training_enabled':False,'public_evaluation_enabled':False,'public_test_used':False}


def prerequisites(cohort, paths=None):
    paths=cohort_paths(cohort) if paths is None else paths
    files=[paths.pool_folder/'reviewed_pools.json',OUT/'preflight/trial_driver_gpu_v1.json']
    files.extend(paths.trials_folder/f'seed{s}_updates{u}/status.json' for s in SEEDS for u in UPDATES)
    return files


def main(cohort,check_only=False,prepare_only=False,*,context=None):
    paths=cohort_paths(cohort) if context is None else context.paths
    plan=paths.trials_folder/'summary_rule_v1.json'
    if not plan.exists() and any(paths.trials_folder.glob('seed*/gsm8k_train_*/inputs.json')):
        raise ValueError('Cannot retroactively declare the summary rule before existing grid trials')
    freeze(plan,{**RULE,'cohort':cohort,'expected_roots':paths.expected_roots,
                 'predeclared_before_any_actual_grid_result':True})
    if prepare_only:
        print({'rule_prepared':str(plan.relative_to(ROOT)),'actual_results_analyzed':False},flush=True);return
    missing=[str(p.relative_to(ROOT)) for p in prerequisites(cohort,paths) if not p.exists()]
    if check_only:
        print({'ready':not missing,'missing':missing,'actual_results_analyzed':False},flush=True);return
    if missing:raise ValueError('Full calibration grid is not ready: '+', '.join(missing[:3]))
    gate_path=OUT/'preflight/trial_driver_gpu_v1.json';gate=read(gate_path)
    gate_signature=OUT/'signatures/trial_driver_gpu_v1.json'
    if gate['status']!='passed' or baseline.sha(gate_signature)!=gate['signature_file_sha256']:
        raise ValueError('Actual trial driver preflight is required')
    for name,checksum in read(gate_signature)['inputs'].items():
        if baseline.sha(ROOT/name)!=checksum:raise ValueError('Actual trial driver preflight is stale: '+name)
    # Reconstruct the shared pool from its generated texts and semantic reviews.
    from material_builder import unique,validate_pool,transfer_choice
    if context is None:
        from explanation_pool import assemble
        assemble(cohort)
    else:
        context.prepare()
    pools=read(paths.pool_folder/'reviewed_pools.json');sources=unique(read(DATA/'cal40.json'),'source_id')
    teachers=unique(checked_rows(OUT/'materials/teacher_targets_cal40.jsonl'),'source_id')
    reviews=unique(read(paths.reviews),'source_id');cases=unique(read(paths.cases),'source_id')
    bundles=unique(read(paths.variants),'source_id');ids=[p['source_id'] for p in pools]
    if len(ids)!=paths.expected_roots or set(ids)!=set(bundles) or set(ids)!=set(cases):
        raise ValueError('Calibration report cannot silently drop roots')
    used={Path(__file__),HERE/'trial_selection.py',HERE/'trial_engine.py',HERE/'trial_calibration.py',
        HERE/'variants.py',HERE/'material_builder.py',HERE/'explanation_pool.py',HERE/'runtime.py',HERE/'runtime_full.py',
        HERE/'calibration_cohort.py',Path(baseline.__file__),plan,gate_path,gate_signature,paths.cases,paths.variants,
        paths.reviews,DATA/'cal40.json',OUT/'materials/teacher_targets_cal40.jsonl',
        paths.pool_folder/'reviewed_pools.json'}
    if context is not None:
        used.update(context.evidence_paths())
    optional_pool_inputs=[paths.pool_folder/name for name in
                          ('requests.json','generated.jsonl','full_text_reviews.json')]
    used.update(p for p in optional_pool_inputs if p.exists())
    cells=[]
    def metrics(path,probes):
        used.add(path);rows=checked_rows(path)
        if len(rows)!=6:raise ValueError('All six predictions are required')
        for row,probe in zip(rows,probes):
            if (row['id']!=probe['id'] or row['question_sha256']!=probe['question_sha256']
                    or row['question']!=probe['question'] or digest(row['question'])!=row['question_sha256']
                    or row['gold']!=probe['gold'] or digest(row['output'])!=row['output_sha256']
                    or not math.isfinite(row['answer_nll']) or row['answer_nll']<0
                    or row['correct']!=baseline.score(row['scoring_text'],probe['gold'])['correct']):
                raise ValueError('Prediction provenance or numeric score changed')
        value=summarize(rows,[p['id'] for p in probes])
        return value,sum(r['correct'] for r in rows)
    for seed in SEEDS:
        for updates in UPDATES:
            folder=paths.trials_folder/f'seed{seed}_updates{updates}';status_path=folder/'status.json'
            status=read(status_path);used.add(status_path)
            if (status.get('complete_for_declared_cohort') is not True or status['roots']!=len(ids)
                    or status['seed']!=seed or status['updates']!=updates or status['cohort']!=cohort
                    or status.get('public_test_used') is not False):raise ValueError('Grid job is incomplete or mislabelled')
            sig=OUT/'signatures'/f'trial_{paths.signature_prefix}_s{seed}_u{updates}_v1.json';used.add(sig)
            for name,checksum in read(sig)['inputs'].items():
                if baseline.sha(ROOT/name)!=checksum:raise ValueError('Grid input signature is stale: '+name)
            if context is not None:
                context.verify_job(status,read(sig),seed,updates)
            group={threshold:[] for threshold in THRESHOLDS}
            for pool in pools:
                sid=pool['source_id'];source=sources[sid];root=folder/sid.replace(':','_')
                teacher=teachers[sid];review=reviews[sid]
                if (not review['accepted'] or not review['all_text_read']
                        or review['question_sha256']!=source['question_sha256']
                        or review['output_sha256']!=teacher['output_sha256']):raise ValueError('Root review changed')
                candidates=validate_pool(source,teacher,pool,{c['step']['step_index']:c['step'] for c in cases[sid]['candidates']})
                probes=validate_probes(bundles[sid],source,build)
                inputs_path=root/'inputs.json';before_path=root/'decisions_before_audit.json';result_path=root/'development_results.json'
                inputs=read(inputs_path);before=read(before_path);result=read(result_path)
                used.update((inputs_path,before_path,result_path))
                untrained,count_untrained=metrics(root/'untrained_Vsel.jsonl',probes['select'])
                if before['untrained_Vsel']!=untrained:raise ValueError('Untrained selection metrics changed')
                scores={};branches={}
                for index,candidate in enumerate(candidates):
                    cid=candidate['candidate_id'];branch=root/f'branch{index}'
                    trained_path=branch/'trained.json';summary_path=branch/'summary.json';adapter=branch/'adapter.safetensors'
                    trained=read(trained_path);summary=read(summary_path);used.update((trained_path,summary_path,adapter))
                    expected={'source_id':sid,'candidate_id':cid,'seed':seed,'updates':updates,
                        'completion_sha256':candidate['completion_sha256'],
                        'root_input_file_sha256':baseline.sha(inputs_path),
                        'initial_adapter_sha256':inputs['initial_adapter_sha256'],
                        'training_data':'original_root_completion_only'}
                    if (any(trained.get(k)!=v for k,v in expected.items())
                            or baseline.sha(adapter)!=trained['adapter_file_sha256']
                            or [log['updates'] for log in trained['updates_log']]!=list(range(1,updates+1))):
                        raise ValueError('Committed trial branch changed')
                    value,_=metrics(branch/'Vsel.jsonl',probes['select'])
                    if summary!={**trained,'Vsel':value}:raise ValueError('Branch Vsel summary differs from saved predictions')
                    scores[cid]=value;branches[cid]=branch
                if scores!=before['Vsel_scores']:raise ValueError('Before-audit Vsel metrics changed')
                decisions=[choose(scores,candidates,t) for t in THRESHOLDS]
                selected_ids=['e0']+sorted({d['selected_id'] for d in decisions}-{'e0'})
                audits={cid:metrics(branches[cid]/'Vaudit.jsonl',probes['audit']) for cid in selected_ids}
                if (result['source_id']!=sid or result['development_data_not_independent_confirmation'] is not True
                        or result['audit_scores']!={cid:value[0] for cid,value in audits.items()}
                        or len(result['configurations'])!=4):raise ValueError('Development result identity/audits changed')
                for decision,stored in zip(decisions,result['configurations']):
                    t=decision['threshold'];cid=transfer_choice(source,pool,probes,inputs,before,
                        seed=seed,updates=updates,threshold=t)
                    acc=audits[cid][0]['accuracy']-audits['e0'][0]['accuracy']
                    nll=audits['e0'][0]['answer_nll']-audits[cid][0]['answer_nll']
                    if stored!={**decision,'Vaudit_accuracy_gain':acc,'Vaudit_nll_gain':nll}:
                        raise ValueError('Stored development gain differs from frozen decision and actual audit')
                    group[t].append({'source_id':sid,'selected_id':cid,'baseline_correct':audits['e0'][1],
                        'selected_correct':audits[cid][1],'Vaudit_nll_gain':nll,'untrained_correct':count_untrained})
            cells.extend({'seed':seed,'updates':updates,'threshold':t,'roots':group[t]} for t in THRESHOLDS)
    report=aggregate(cells,ids)
    report.update(cohort=cohort,actual_model_results_analyzed=True,
                  absent_optional_pool_inputs=[str(p.relative_to(ROOT)) for p in optional_pool_inputs if not p.exists()],
                  inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in sorted(used)})
    if context is not None:
        report.update(context.report_identity())
    freeze(paths.trials_folder/'development_grid_summary_v1.json',report)
    print({k:report[k] for k in ('status','selected_configuration','best_primary_development_mean_gain','stability')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-prerequisites',action='store_true')
    mode.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    main(args.cohort,args.check_prerequisites,args.prepare_only)
