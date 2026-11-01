"""Actual-grid-bound finite128 production trials; no public dataset access.

Negative development is a terminal unsupported outcome for automatic expansion.
Neither an arbitrary config hash nor CPU tests can authorize these trials.
"""
from pathlib import Path
import argparse
import json
import math
import re
from runtime_full import (ROOT,HERE,OUT,BASE,read,freeze,save,checked_rows,digest,
                          baseline,gpu,signature,check_time)
from budgeted_materials import PROTOCOL
from train_selector_quality_v3 import verify_inputs
from transfer128_sampling import ROSTER,verify_roster
from transfer128_explanations_v1 import FOLDER as EXPLANATIONS
from transfer128_variant_manifest_v2 import FOLDER as VARIANTS
from transfer128_context_v1 import FOLDER as CONTEXT
from transfer128_variants_v2 import normalized_source
from fixed_trial_engine_v1 import run_fixed_root
from trial_calibration import adapter_hash,restore_adapter,evaluate_probes
from trial_selection import choose,summarize,validate_probes
from variants import build

FOLDER=OUT/'transfer128/production_v1'
CONFIGURATION=FOLDER/'configuration_frozen_v1.json'
CPU=OUT/'preflight/transfer128_production_materials_cpu_v1.json'


def configuration_fields(summary):
    if (summary.get('protocol')!=PROTOCOL or summary.get('cohort')!='cal34_general7b_v2'
            or summary.get('recipe_id')!='local_explanation_general7b_v2'
            or summary.get('actual_model_results_analyzed') is not True
            or summary.get('all24_seed_configuration_cells_complete') is not True
            or summary.get('public_test_used') is not False):
        raise ValueError('Complete actual adopted-general34 development evidence required')
    config=summary.get('selected_configuration')
    if summary.get('status')=='no_positive_development_support':
        if config is not None:raise ValueError('Negative development cannot choose a configuration')
        return {'development_status':summary['status'],'selected_configuration':None,
            'scientific_trial_enabled':False,'automatic_expansion':'unsupported_no_expansion'}
    if (summary.get('status')!='positive_development_signal_requires_confirmation'
            or not isinstance(config,dict) or set(config)!={'updates','threshold','trial_seed'}
            or type(config['updates']) is not int or config['updates'] not in (1,3,5)
            or type(config['threshold']) not in (int,float) or config['threshold'] not in (0,.01,.03,.1)
            or type(config['trial_seed']) is not int or config['trial_seed']!=701):
        raise ValueError('One genuine declared development-selected configuration required')
    return {'development_status':summary['status'],'selected_configuration':config,
        'scientific_trial_enabled':True,'automatic_expansion':'fixed128_confirmation_only'}


def driver_evidence():
    path=OUT/'preflight/fixed_trial_driver_gpu_v1.json';report=read(path)
    sig=OUT/'signatures/fixed_trial_driver_gpu_v1.json'
    required=('cold_learner_resume','no_training_repeated_on_resume','cached_replay_no_generation',
              'adapters_unchanged_on_resume','choice_frozen_before_audit','reset_to_S0_after_trial',
              'single_threshold_frozen')
    result=OUT/'preflight/fixed_trial_driver_gpu_v1/root/trial_result.json'
    if (report.get('status')!='passed' or report.get('device')!='actual_student_model_GPU'
            or any(report.get(k) is not True for k in required)
            or report.get('generation_limit_engineering_only')!=32
            or report.get('formal_generation_limit_unchanged')!=1024
            or report.get('official_test_accessed') is not False
            or report.get('signature_file_sha256')!=baseline.sha(sig)
            or report.get('result_file_sha256')!=baseline.sha(result)):
        raise ValueError('Actual fixed-driver GPU engineering evidence required')
    paths=verify_inputs(read(sig))
    # Bind all real engineering predictions and committed adapters as well.
    paths += [p for p in result.parent.rglob('*') if p.is_file()]
    return [path,sig,*paths]


def actual_development():
    from calibration_general_v2 import GeneralContext,COHORT
    from calibration_summary import main
    context=GeneralContext();main(COHORT,context=context)
    path=context.paths.trials_folder/'development_grid_summary_v1.json'
    summary=read(path);paths=verify_inputs(summary)
    fields=configuration_fields(summary)
    return summary,fields,[path,*paths]


def material_evidence():
    # These replay programs never supply mathematical judgments or redraw roots.
    from transfer128_variant_manifest_v2 import prepare as replay_variants
    from transfer128_context_v1 import prepare as replay_context
    from transfer128_explanations_v1 import export as replay_explanations
    replay_variants();replay_context();replay_explanations()
    from full_selector import load_quality
    sources,teachers,_,ids=load_quality()
    roster=read(ROSTER);paths=verify_roster(roster,ids)
    manifests=[VARIANTS/'manifest_v2.json',CONTEXT/'context_preflight_v1.json',
        EXPLANATIONS/'automatic_export_v1.json',EXPLANATIONS/'generation_plan_v1.json']
    for path in manifests:paths += [path,*verify_inputs(read(path))]
    vm=read(manifests[0])
    for name,checksum in vm['outputs'].items():
        if baseline.sha(ROOT/name)!=checksum:raise ValueError('Variant export changed')
        paths.append(ROOT/name)
    bysource={s['source_id']:s for s in sources};byteacher={t['source_id']:t for t in teachers}
    pools=read(EXPLANATIONS/'joint_pools_v1.json');paths.append(EXPLANATIONS/'joint_pools_v1.json')
    if [p['source_id'] for p in pools]!=roster['source_ids']:raise ValueError('Exact original128 pool order required')
    bundles={b['source_id']:b for b in read(VARIANTS/'qualified_bundles_v2.json')['roots']}
    report=read(VARIANTS/'actual_review_report_v2.json')
    if ([r['source_id'] for r in report['roots']]!=roster['source_ids']
            or len(bundles)!=124 or sum(not r['qualified'] for r in report['roots'])!=4):
        raise ValueError('All128 explicit template decisions must be preserved')
    for sid,bundle in bundles.items():
        source,_=normalized_source(bysource[sid]);validate_probes(bundle,source,build)
    return roster,bysource,byteacher,pools,bundles,list(dict.fromkeys([ROSTER,*paths]))


def config_content(fields,roster,summary):
    if set(roster['source_ids']) & set(summary['root_ids']):raise ValueError('Calibration/confirmation roots overlap')
    return {'schema':'tgrd_transfer128_actual_configuration_v1','protocol':PROTOCOL,
        **fields,'source_ids':roster['source_ids'],'root_budget':128,'replacement_after_results':False,
        'max_nonempty_candidates':4,'select_probes':6,'audit_probes':6,
        'trial_initialization':'fresh_Learner701_common_across_all_roots_and_branches',
        'formal_initialization':'separate_common_Learner42_for_all_five_formal_conditions',
        'original_root_only_gradients':True,'empty_optimizer_per_branch':True,
        'audit_can_change_selection':False,'generation_limit':1024,
        'frozen_before_current_public_test':True,'public_test_used':False,
        'explanation_length_target_tokens':8,'length_target_enforced_as_success_filter':False,
        'training_root_derived_probes_are_external_generalization':False}


def configuration_evidence():
    summary,fields,paths=actual_development()
    # A negative outcome is frozen without constructing or running128 trials.
    # Exact material preparation still stays available for later reporting.
    from full_selector import load_quality
    _,_,_,ids=load_quality();roster=read(ROSTER)
    paths += [ROSTER,*verify_roster(roster,ids)]
    from evaluation_selector_quality_v3 import collect_training
    _,_,training_paths=collect_training();paths+=training_paths
    cpu=read(CPU);paths += [CPU,*verify_inputs(cpu)]
    if (cpu.get('status')!='passed' or cpu.get('device')!='MLX_CPU'
            or str(Path(__file__).relative_to(ROOT)) not in cpu['inputs']):
        raise ValueError('Production-wrapper CPU integration evidence required')
    if fields['scientific_trial_enabled']:
        _,_,_,_,_,materials=material_evidence();paths+=materials+driver_evidence()
    from evaluation_selector_inputs import TRANSFER_RULES
    paths += [Path(__file__),HERE/'fixed_trial_engine_v1.py',HERE/'trial_selection.py',
        HERE/'trial_calibration.py',HERE/'learning.py',HERE/'runtime.py',HERE/'runtime_full.py',
        HERE/'evaluation_selector_inputs.py',ROOT/'experiments/math_distill_v1/config.json']
    value=config_content(fields,roster,summary)
    value['rules']=TRANSFER_RULES
    value['inputs']={str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}
    return value


def freeze_configuration():
    # Config cannot be backdated after this study's public predictions.
    if not CONFIGURATION.exists() and any((OUT/'evaluation'/PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot freeze transfer configuration after current public predictions')
    value=configuration_evidence();freeze(CONFIGURATION,value);return value


def load_configuration():
    value=read(CONFIGURATION);verify_inputs(value)
    if value!=configuration_evidence():raise ValueError('Frozen configuration differs from reconstructed actual evidence')
    return value


def fallback(source,pool,configuration_sha256,reason):
    return {'schema':'tgrd_fixed128_explicit_fallback_v1','source_id':source['source_id'],
        'question_sha256':source['question_sha256'],'configuration_sha256':configuration_sha256,
        'pool_sha256':digest(json.dumps(pool,sort_keys=True)),
        'selected_candidate_id':'e0','reason':reason,'trial_executed':False,
        'frozen_roster_retained':True,'audit_can_change_selection':False,'public_test_used':False}


def run():
    value=load_configuration()
    if not value['scientific_trial_enabled']:
        outcome={'schema':'tgrd_fixed128_unsupported_outcome_v1','configuration_sha256':baseline.sha(CONFIGURATION),
            'status':'no_positive_development_support','trial_roots_executed':0,
            'fixed128_trials_complete':False,'G4_G5_automatic_expansion_enabled':False,
            'public_test_used':False,'interpretation':'unsupported_not128_scientific_fallback_results'}
        freeze(FOLDER/'unsupported_outcome_v1.json',outcome);return outcome
    roster,sources,_,pools,bundles,paths=material_evidence()
    params=value['selected_configuration'];checksum=baseline.sha(CONFIGURATION)
    with gpu():
        signature('transfer128_fixed_production_v1',[CONFIGURATION,*[ROOT/n for n in value['inputs']],*paths],
            {'configuration_sha256':checksum,'generation_limit':1024,'public_test_used':False})
        from learning import Learner
        baseline.verify_student(BASE);learner=Learner(701);initial=adapter_hash(learner)
        engineering_initial=read(OUT/'preflight/fixed_trial_driver_gpu_v1/root/inputs.json')['initial_adapter_sha256']
        if initial!=engineering_initial:raise ValueError('Actual fresh701 initialization differs from engineering S0')
        freeze(FOLDER/'initialization_seed701.json',{'seed':701,'initial_adapter_sha256':initial,
            'purpose':'common_S0_for_original_only_finite128_trials','formal_seed42_S0_reused':False})
        rows=[]
        for pool in pools:
            check_time();sid=pool['source_id'];source,_=normalized_source(sources[sid])
            folder=FOLDER/'roots'/sid.replace(':','_')
            reason=('actual_template_rejected' if sid not in bundles else
                    'no_automatically_eligible_joint_explanation' if len(pool['candidates'])==1 else None)
            if reason:
                row=fallback(source,pool,checksum,reason);freeze(folder/'fallback.json',row)
            else:
                probes=validate_probes(bundles[sid],source,build)
                result=run_fixed_root(learner,source,pool,probes,folder,
                    seed=701,updates=params['updates'],threshold=params['threshold'],
                    initial_hash=initial,weight_hash=adapter_hash,restore=restore_adapter,
                    evaluate=evaluate_probes,configuration_sha256=checksum)
                row={'source_id':sid,'selected_candidate_id':result['decision']['selected_id'],
                    'trial_executed':True,'result_sha256':baseline.sha(folder/'trial_result.json')}
            rows.append(row);print({'finite_roots_terminal':len(rows),'total':128,'source_id':sid},flush=True)
        result={'schema':'tgrd_fixed128_production_complete_v1','complete':True,
            'configuration_sha256':checksum,'source_ids':roster['source_ids'],'roots':rows,
            'actual_trial_roots':sum(r['trial_executed'] for r in rows),
            'explicit_fallback_roots':sum(not r['trial_executed'] for r in rows),
            'public_test_used':False,'audit_can_change_selection':False}
        freeze(FOLDER/'run_complete_v1.json',result)
    return result


def prediction_metrics(path,probes):
    rows=checked_rows(path)
    if len(rows)!=6:raise ValueError('All six actual predictions are required')
    for row,p in zip(rows,probes):
        if (any(row.get(k)!=v for k,v in p.items()) or digest(row['output'])!=row['output_sha256']
                or type(row['answer_nll']) not in (int,float) or not math.isfinite(row['answer_nll'])
                or row['answer_nll']<0 or type(row.get('generation_tokens')) is not int
                or not 0<row['generation_tokens']<=1024
                or row['prompt_sha256']!=digest(BASE['student_prompt'].format(question=p['question']))):
            raise ValueError('Exact probe identity, generation or NLL changed')
        boundary=re.search(r'(?:^|\n)\s*Question:',row['output'])
        scoring=row['output'][:boundary.start()] if boundary else row['output']
        score=baseline.score(scoring,p['gold'])
        if row['scoring_text']!=scoring or any(row.get(k)!=v for k,v in score.items()):
            raise ValueError('Raw scientific prediction does not reproduce its score')
    return summarize(rows,[p['id'] for p in probes])


def replay_root(source,pool,probes,folder,configuration_sha256,params,initial_hash):
    """Read all actual branch commits/predictions; winner determined by Vsel only."""
    folder=Path(folder);inputs=read(folder/'inputs.json')
    expected={'source_id':source['source_id'],'question_sha256':source['question_sha256'],
        'pool_sha256':digest(json.dumps(pool,sort_keys=True)),
        'probe_bundle_sha256':digest(json.dumps(probes,sort_keys=True)),
        'seed':701,'updates':params['updates'],'initial_adapter_sha256':initial_hash,
        'threshold':params['threshold'],'configuration_sha256':configuration_sha256,
        'schema':'tgrd_fixed128_trial_inputs_v1'}
    if inputs!=expected:raise ValueError('Actual fixed-trial input differs from frozen configuration/material')
    files=[folder/'inputs.json',folder/'untrained_Vsel.jsonl']
    untrained=prediction_metrics(files[-1],probes['select']);scores={};branches={}
    for index,c in enumerate(pool['candidates']):
        b=folder/f'branch{index}';trained=read(b/'trained.json');summary=read(b/'summary.json')
        identity={'candidate_id':c['candidate_id'],'source_id':source['source_id'],
            'updates':params['updates'],'seed':701,'completion_sha256':c['completion_sha256'],
            'initial_adapter_sha256':initial_hash,'root_input_file_sha256':baseline.sha(folder/'inputs.json'),
            'training_data':'original_root_completion_only'}
        if (any(trained.get(k)!=v for k,v in identity.items())
                or trained['adapter_file_sha256']!=baseline.sha(b/'adapter.safetensors')
                or trained['trained_adapter_sha256']==initial_hash
                or [u['updates'] for u in trained['updates_log']]!=list(range(1,params['updates']+1))):
            raise ValueError('Actual equal-budget trained branch provenance changed')
        scores[c['candidate_id']]=prediction_metrics(b/'Vsel.jsonl',probes['select'])
        if summary!={**trained,'Vsel':scores[c['candidate_id']]}:raise ValueError('Branch summary differs from raw predictions')
        branches[c['candidate_id']]=b
        files += [b/name for name in ['trained.json','summary.json','adapter.safetensors','Vsel.jsonl']]
    before=read(folder/'decision_before_audit.json');decision=choose(scores,pool['candidates'],params['threshold'])
    if before!={'source_id':source['source_id'],'decision':decision,
            'configuration_sha256':configuration_sha256,'root_input_file_sha256':baseline.sha(folder/'inputs.json'),
            'Vsel_scores':scores,'untrained_Vsel':untrained,'audit_can_change_selection':False}:
        raise ValueError('Frozen before-audit choice does not reproduce Vsel-only selection')
    needed=['e0']+([] if decision['selected_id']=='e0' else [decision['selected_id']])
    audit={cid:prediction_metrics(branches[cid]/'Vaudit.jsonl',probes['audit']) for cid in needed}
    if any((b/'Vaudit.jsonl').exists() for cid,b in branches.items() if cid not in needed):
        raise ValueError('Unselected branches must not have confirmation audit predictions')
    result={'schema':'tgrd_fixed128_trial_result_v1','source_id':source['source_id'],
        'configuration_sha256':configuration_sha256,'root_input_file_sha256':baseline.sha(folder/'inputs.json'),
        'decision':decision,'Vaudit_accuracy_gain':audit[decision['selected_id']]['accuracy']-audit['e0']['accuracy'],
        'Vaudit_nll_gain':audit['e0']['answer_nll']-audit[decision['selected_id']]['answer_nll'],
        'audit_scores':audit,'audit_can_change_selection':False,
        'original_root_only_gradients':True,'other_thresholds_evaluated':False}
    if read(folder/'trial_result.json')!=result:raise ValueError('Audit result changed or attempted reselection')
    files += [folder/'decision_before_audit.json',folder/'trial_result.json',*[branches[c]/'Vaudit.jsonl' for c in needed]]
    return result,files


def replay():
    value=load_configuration()
    if not value['scientific_trial_enabled']:raise ValueError('Unsupported development did not execute128 trials')
    roster,sources,_,pools,bundles,paths=material_evidence();checksum=baseline.sha(CONFIGURATION)
    sig=OUT/'signatures/transfer128_fixed_production_v1.json';signed=read(sig)
    if (signed.get('extra')!={'configuration_sha256':checksum,'generation_limit':1024,'public_test_used':False}
            or signed.get('inputs',{}).get(str(CONFIGURATION.relative_to(ROOT)))!=checksum):
        raise ValueError('Actual scientific production signature must bind this configuration')
    paths += [sig,*verify_inputs(signed)]
    complete=read(FOLDER/'run_complete_v1.json');initial=read(FOLDER/'initialization_seed701.json')
    paths += [FOLDER/'run_complete_v1.json',FOLDER/'initialization_seed701.json',CONFIGURATION]
    expected_initial=read(OUT/'preflight/fixed_trial_driver_gpu_v1/root/inputs.json')['initial_adapter_sha256']
    if initial!={'seed':701,'initial_adapter_sha256':expected_initial,
            'purpose':'common_S0_for_original_only_finite128_trials','formal_seed42_S0_reused':False}:
        raise ValueError('Actual trial initialization differs from the bound fresh701 S0')
    if (complete.get('complete') is not True or complete.get('source_ids')!=roster['source_ids']
            or complete.get('configuration_sha256')!=checksum
            or [r['source_id'] for r in complete['roots']]!=roster['source_ids']):
        raise ValueError('Complete original128 production outcomes required')
    results=[]
    for pool,stored in zip(pools,complete['roots']):
        sid=pool['source_id'];source,_=normalized_source(sources[sid]);folder=FOLDER/'roots'/sid.replace(':','_')
        reason=('actual_template_rejected' if sid not in bundles else
                'no_automatically_eligible_joint_explanation' if len(pool['candidates'])==1 else None)
        if reason:
            expected=fallback(source,pool,checksum,reason);path=folder/'fallback.json'
            if read(path)!=expected or stored!=expected or (folder/'trial_result.json').exists():
                raise ValueError('Scientific fallback must have its true qualified-material reason')
            results.append(expected);paths.append(path)
        else:
            probes=validate_probes(bundles[sid],source,build)
            r,files=replay_root(source,pool,probes,folder,checksum,value['selected_configuration'],initial['initial_adapter_sha256'])
            expected={'source_id':sid,'selected_candidate_id':r['decision']['selected_id'],
                'trial_executed':True,'result_sha256':baseline.sha(folder/'trial_result.json')}
            if stored!=expected:raise ValueError('Completion index differs from actual frozen result')
            results.append({**expected,'Vaudit_accuracy_gain':r['Vaudit_accuracy_gain'],
                'Vaudit_nll_gain':r['Vaudit_nll_gain']});paths+=files
    if (complete.get('actual_trial_roots')!=sum(r['trial_executed'] for r in results)
            or complete.get('explicit_fallback_roots')!=sum(not r['trial_executed'] for r in results)):
        raise ValueError('Actual trial/fallback counts changed')
    report={'schema':'tgrd_fixed128_production_replay_v1','complete':True,'roots':results,
        'source_ids':roster['source_ids'],'configuration_sha256':checksum,
        'actual_trial_roots':sum(r['trial_executed'] for r in results),
        'public_test_used':False,'audit_used_to_reselect':False,
        'mean_Vaudit_accuracy_gain_all128':sum(r.get('Vaudit_accuracy_gain',0) for r in results)/128,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}}
    freeze(FOLDER/'actual_replay_v1.json',report);return report


def finalize():
    value=load_configuration()
    return replay() if value['scientific_trial_enabled'] else run()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze-configuration','run','replay','finalize'])
    a=p.parse_args();v={'freeze-configuration':freeze_configuration,'run':run,'replay':replay,'finalize':finalize}[a.action]()
    print({k:v[k] for k in v if k not in ('inputs','roots','source_ids','rules')},flush=True)
