"""Actual-result-bound public handoff and complete-result reporting.

No test question loader runs before the current actual public gate validates.
Negative development admits the three completed selector controls only, never
fabricated128 outcomes or automatic G4/G5 expansion. Existing signed sources
are retained. Historical partial official-test exposure is explicitly unresolved.
"""
import argparse
import json
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,read,freeze,save,baseline,checked_rows,signature
from budgeted_materials import PROTOCOL
from evaluation_selector_inputs import TRANSFER_RULES,EVALUATION_ORDER,verify_inputs

FOLDER=OUT/'public_preparation_v1'
CPU=OUT/'preflight/public_evaluation_workflow_cpu_v1.json'
SCOPE=FOLDER/'actual_evaluation_scope_v1.json'
EXPOSURE=FOLDER/'historical_public_exposure_v1.json'
RESULTS=OUT/'results'/PROTOCOL/'final_results_v1.json'
MASTER=ROOT/'FYP_TGRD_MASTER_EXPERIMENT_PLAN_ZH.md'


def exposure_contents(master,history):
    matches=[line for line in master.splitlines() if '历史项目曾查看部分官方测试子集' in line]
    if len(matches)!=1 or not history.get('limits'):
        raise ValueError('Traceable historical partial public-test exposure evidence required')
    ids=history['source_ids']
    if not ids or len(ids)!=len(set(ids)) or any(not sid.startswith('gsm8k:train:') for sid in ids):
        raise ValueError('Historical train exposure list must not be relabelled as public-test IDs')
    return dict(schema='tgrd_historical_public_exposure_disclosure_v1',
        known_partial_official_test_exposure=True,source_excerpt=matches[0],
        exposed_official_test_ids=None,recoverable_official_test_exposure_count=None,
        exact_official_test_ID_recovery_complete=False,
        historical_training_ID_count=len(ids),historical_training_inventory_limit=history['limits'],
        current_test_is_first_ever_blind_evaluation=False,
        exposed_unexposed_sensitivity_possible_from_recovered_IDs=False,
        full1319_evaluation_retained=True,official_test_questions_read_for_this_snapshot=False)


def freeze_exposure():
    history=ROOT/'data/tgrd_v1/historical_exposure.json'
    value=exposure_contents(MASTER.read_text(),read(history))
    value['inputs']={str(p.relative_to(ROOT)):baseline.sha(p) for p in (MASTER,history,Path(__file__))}
    freeze(EXPOSURE,value);return value


def choose_scope(config):
    if config.get('scientific_trial_enabled') is True:
        if (config.get('development_status')!='positive_development_signal_requires_confirmation'
                or not isinstance(config.get('selected_configuration'),dict)
                or set(config['selected_configuration'])!={'updates','threshold','trial_seed'}
                or config['selected_configuration']['updates'] not in (1,3,5)
                or config['selected_configuration']['threshold'] not in (0,.01,.03,.1)
                or config['selected_configuration']['trial_seed']!=701):
            raise ValueError('Actual positive development-selected configuration required')
        return list(EVALUATION_ORDER)
    if (config.get('scientific_trial_enabled') is False
            and config.get('development_status')=='no_positive_development_support'
            and config.get('selected_configuration') is None):
        return list(EVALUATION_ORDER[:3])
    raise ValueError('No partial grid, loose boolean, or public score may choose evaluation scope')


def source_observations(ids,roster,history,calibration,teacher_reviews):
    if len(ids)!=7179 or len(set(ids))!=7179 or len(teacher_reviews)!=1433:
        raise ValueError('Frozen common inventory and actual1433 teacher reviews required')
    review_ids=[r['source_id'] for r in teacher_reviews]
    if len(set(review_ids))!=1433 or any(r.get('all_text_read') is not True for r in teacher_reviews):
        raise ValueError('Actual full-text teacher ledger cannot be inferred from automatic eligibility')
    accepted=set(ids);historical=set(history['source_ids']);cal={r['source_id'] for r in calibration}
    engineering={'gsm8k:train:0','gsm8k:train:2'}
    if (len(roster.get('source_ids',[]))!=128 or len(set(roster['source_ids']))!=128
            or set(roster['source_ids'])&(historical|cal|engineering)
            or not (historical|cal|engineering)<=set(roster.get('excluded_development_ids',[]))):
        raise ValueError('Fixed128 exclusions must preserve known historical/development roots')
    return dict(actual_teacher_full_text_reads=1433,teacher_unread=7473-1433,
        historical_train_roots_recycled_into_formal=sorted(accepted&historical),
        calibration40_roots_recycled_into_formal=sorted(accepted&cal),
        engineering_roots_recycled_into_formal=sorted(accepted&engineering),
        original7179_roster_modified=False)


def disclosure_contents(exposure,observations,teacher_report,explanation_export,explanation_report,
                        extension,token_budgets,scope):
    if (teacher_report.get('complete') is not True or teacher_report.get('sample_count')!=128
            or teacher_report.get('semantic_accepted')!=115 or teacher_report.get('semantic_flagged')!=13
            or explanation_export.get('complete') is not True or explanation_export.get('actual_reviewed')!=2821
            or explanation_export.get('actual_rejected')!=444 or explanation_export.get('mechanically_usable_unread')!=8192
            or explanation_export.get('all_text_read') is not False
            or explanation_export.get('semantic_correctness_verified') is not False
            or explanation_report.get('accepted')!=109 or explanation_report.get('rejected')!=19
            or explanation_report.get('existing_exact_reused')!=31 or explanation_report.get('new_actual_reads')!=97
            or extension.get('complete') is not True or extension.get('requests')!=412
            or extension.get('new_outputs')!=142 or extension.get('exact_output_reuses')!=270
            or extension.get('all_text_read') is not False or extension.get('semantic_correctness_verified') is not False
            or extension.get('new142_covered_by_original_descriptive_sample') is not False
            or set(token_budgets)!=set(scope) or any(type(v) is not int or v<=0 for v in token_budgets.values())
            or exposure.get('known_partial_official_test_exposure') is not True
            or exposure.get('exposed_official_test_ids') is not None):
        raise ValueError('Actual quality reports, separate142 population and actual final token budgets required')
    counts={k:extension[k] for k in ('requests','new_outputs','exact_output_reuses','automatically_eligible','excluded','new_automatically_eligible')}
    if (counts['automatically_eligible']+counts['excluded']!=412
            or not 0<=counts['new_automatically_eligible']<=142):
        raise ValueError('Actual extension automatic counts do not reconcile')
    return dict(protocol=PROTOCOL,
        historical_public_test_exposure=(
            '历史项目已查看部分官方GSM8K test子集（主计划有明确记录）；当前工作区未恢复其完整ID和题数。'
            f"历史{exposure['historical_training_ID_count']}训练侧ID清单不是公开test曝光清单。不能声称本项目从未看过test或本轮为首次盲测；"
            '保留完整1319分数，曝光/未曝光分层敏感性因ID未恢复而无法可靠计算，不伪造未曝光子集。'),
        development_root_reuse=(
            f"历史训练侧曝光根题中{len(observations['historical_train_roots_recycled_into_formal'])}条、"
            f"cal40中{len(observations['calibration40_roots_recycled_into_formal'])}条、工程根题中"
            f"{len(observations['engineering_roots_recycled_into_formal'])}条已按统一7179自动名单回收正式训练。"
            '这些数据不再是独立留出；general34/Vaudit用于超参开发，1701仅重复稳定性，不是最终训练seed重复。'
            '迁移128保留原固定名单，排除历史开发/cal40/工程根题，不按试学成功替换。'),
        automatic_teacher_quality_noise=(
            '7473原教师输出、7186数字匹配；统一自动准入7179、排除294。原1433条全文判断、6040未读不构成全量语义证明。'
            '修订前1329审核历史快照中91条严格标记仍在自动集合，未被选择性净化；包含措辞、范围、单位和过程问题，并非全为确定算错。'
            '固定质量128为115接受/13严格标记，仅描述，不要求抽检全对、不换题、不改变7179。'),
        automatic_explanation_quality_noise=(
            '原17932解释中10521自动准入、7411排除；2821实际判断（2377接受/444拒绝），8192原机械可用解释未读。'
            '固定质量128为31精确历史复用+97实际阅读，109接受/19严格标记，并非全是确定算错，非独立盲审。'
            '自动可用不等于全文语义正确，all_text_read=false、semantic_correctness_verified=false。'),
        single_seed_limitation='最终正式训练仅seed42一次；seed701有限试学和1701开发重复不能充当多次正式训练。组间差异不证明跨seed可复现或普遍优越。',
        explanation_length_difference=(
            '新增局部解释按32–64学生tokens自动准入；原8token目标未作为成功筛选。'
            '同7179根题/4epoch/5744更新不代表同监督tokens或同计算成本。实际含EOS监督tokens为'
            +json.dumps(token_budgets,sort_keys=True)+'；训练完整上下文≤4096且不截断，公开greedy最大1024。'
            'G5仅固定前两条Vsel交替入梯度，其内容与长度改变必须披露。'),
        finite128_confirmation_scope=(
            '固定128训练根题派生的有限候选确认，124套模板合格、4拒绝且保留预算；最多4解释+e0。'
            '仅原题梯度、相同S0/空优化器/唯一更新数，6Vsel真实准确率与答案NLL选择冻结后，只审核e0和赢家6Vaudit；负审核不改选。'
            '派生变式不是外部抽样泛化；G4名单外逐字G3；G5训练过Vsel不能充当泛化证据。'
            +('全开发非正，保留无支持，未执行128科学试学/G4G5自动扩量；只报告三个已完成正式对照。' if len(scope)==3 else
              '实际128试学重放和两后续最终训练是五组gate前置；工程通过不等于正收益。')),
        new142_explanation_population=(
            f"412联合请求中270条完全相同旧提示/输出精确复用并保留旧拒绝，142条新终止尝试；"
            f"其中新自动准入{counts['new_automatically_eligible']}、排除{142-counts['new_automatically_eligible']}。"
            '终止尝试不保证有效文本；新142属于不同总体，原固定质量128未覆盖，未追加重抽或假称全部实际阅读。'),
        new142_covered_by_original_descriptive_sample=False,
        transfer128_explanation_counts=counts,actual_supervised_tokens_including_EOS=token_budgets,
        exposure_ID_uncertainty=exposure,training_side_reuse_observations=observations,
        declared_evaluation_conditions=scope,public_test_used_to_choose_scope=False,
        primary_checkpoint='last_epoch4',same_root_updates_is_equal_token_compute_budget=False)


def materialize_disclosures(config,scope,training,artifacts,paths):
    from transfer128_sampling import ROSTER,verify_roster
    from teacher_spotcheck_v2 import REPORT as TEACHER_REPORT,verify as verify_teacher
    import explanation_quality_v3 as explanation
    from evaluation_selector_quality_v3 import quality_evidence
    from transfer128_explanations_v1 import export,FOLDER as EXTENSION
    from runtime_full import checked_rows
    exposure=freeze_exposure();verify_inputs(exposure)
    ids=training['common_root_ids'];roster=read(ROSTER);paths += verify_roster(roster,ids)
    history_path=ROOT/'data/tgrd_v1/historical_exposure.json';cal_path=ROOT/'data/tgrd_v1/cal40.json'
    review_paths=[OUT/'material_audits/cal40_teacher_full_text_reviews_v1.json',
        OUT/'material_audits/teacher_additional_reviews_v1.jsonl']
    reviews=read(review_paths[0])+checked_rows(review_paths[1])
    observations=source_observations(ids,roster,read(history_path),read(cal_path),reviews)
    paths += verify_teacher()+quality_evidence()
    extension=export();extension_path=EXTENSION/'automatic_export_v1.json'
    paths += [extension_path,*verify_inputs(extension)]
    token_budgets={}
    for group,entry in artifacts.items():
        metadata=ROOT/entry['adapter_directory']/'metadata.json';meta=read(metadata)
        token_budgets[group]=meta['supervised_tokens_including_eos'];paths.append(metadata)
    value=disclosure_contents(exposure,observations,read(TEACHER_REPORT),read(explanation.EXPORT),
        read(explanation.REPORT),extension,token_budgets,scope)
    # The two adopted historical statements are source-bound, not reconstructed
    # by relabelling current strict flags as the old snapshot.
    teacher_doc=HERE/'AUTOMATIC_TEACHER_QUALITY_V2_ZH.md'
    if '91条历史严格审核标记过的根题仍通过自动准入' not in teacher_doc.read_text():
        raise ValueError('Original historical91 noise disclosure evidence changed')
    value['explanation_quality_evidence_sha256']={str(p.relative_to(ROOT)):baseline.sha(p) for p in (explanation.REPORT,explanation.EXPORT)}
    value['transfer128_explanation_export_sha256']=baseline.sha(extension_path)
    paths += [EXPOSURE,history_path,cal_path,*review_paths,teacher_doc,CPU,Path(__file__)]
    paths += [ROOT/n for n in exposure['inputs']]
    value['inputs']={str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}
    destination=FOLDER/'concrete_disclosures_v1.json';freeze(destination,value)
    rules=FOLDER/'transfer_rules_v1.json';freeze(rules,TRANSFER_RULES)
    return ROSTER,rules,destination


def prepare():
    from transfer128_production_v1 import load_configuration,CONFIGURATION,FOLDER as PRODUCTION
    import evaluation_selector_quality_v3 as selector
    import evaluation_transfer128_v1 as transfer
    config=load_configuration()  # reconstructs all six actual grid jobs, never partial observations
    scope=choose_scope(config)
    cpu=read(CPU)
    if cpu.get('status')!='passed' or cpu.get('device')!='explicit_MLX_CPU' or cpu.get('scientific_results') is not False:
        raise ValueError('New workflow CPU integration evidence required')
    cpu_paths=verify_inputs(cpu)
    if not SCOPE.exists() and any((OUT/'evaluation'/PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot choose evaluation scope or create disclosures after public predictions')
    training,artifacts,paths=selector.collect_training()
    if len(scope)==5:
        more,files=transfer.collect_transfer_training(training['common_root_ids']);artifacts.update(more);paths += files
    else:
        outcome=read(PRODUCTION/'unsupported_outcome_v1.json')
        if (outcome.get('status')!='no_positive_development_support' or outcome.get('trial_roots_executed')!=0
                or outcome.get('fixed128_trials_complete') is not False
                or outcome.get('configuration_sha256')!=baseline.sha(CONFIGURATION)):
            raise ValueError('Genuine negative development outcome required; not fake128 fallbacks')
        paths.append(PRODUCTION/'unsupported_outcome_v1.json')
    paths += [CONFIGURATION,*[ROOT/n for n in config['inputs']],CPU,*cpu_paths]
    roster,rules,disclosures=materialize_disclosures(config,scope,training,artifacts,paths)
    selector.freeze_pretest(roster,rules,disclosures)
    gate=transfer.freeze_gate() if len(scope)==5 else selector.freeze_gate()
    gate_path=OUT/'gates'/(transfer.GATE_NAME if len(scope)==5 else selector.GATE_NAME)
    inputs=[Path(__file__),CPU,CONFIGURATION,roster,rules,disclosures,gate_path]
    inputs += [ROOT/n for n in read(disclosures)['inputs']]
    inputs += [ROOT/n for n in gate['inputs']]
    value=dict(schema='tgrd_actual_public_evaluation_scope_v1',protocol=PROTOCOL,conditions=scope,
        scope_selected_from_complete_development_only=True,public_test_used_to_choose_scope=False,
        selected_configuration=config['selected_configuration'],scientific_fixed128_trials_complete=len(scope)==5,
        G4_G5_status='actual_final_training_completed' if len(scope)==5 else 'unsupported_not_run',
        artifacts=gate['artifacts'],public_gate=str(gate_path.relative_to(ROOT)),
        inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(inputs)})
    freeze(SCOPE,value);signature('actual_public_evaluation_workflow_v1',inputs,
        {'protocol':PROTOCOL,'conditions':scope,'no_public_score_scope_selection':True})
    return value


def run(condition):
    workflow=prepare();verify_inputs(workflow)
    if condition not in workflow['conditions']:
        if condition not in ('M7','M6') or len(workflow['conditions'])!=3:
            raise ValueError('Condition outside the declared study')
        return dict(condition=condition,status='unsupported_not_run',official_predictions_generated=False)
    # This order check reads only already-frozen prior predictions and signatures.
    from evaluation_transfer128_v1 import validate_order
    validate_order(condition,workflow['artifacts'])
    if len(workflow['conditions'])==5:
        from evaluate_transfer128_v1 import main
    else:
        from evaluate_selector_quality_v3 import main
    main(condition,42)
    return dict(condition=condition,status='full_evaluation_returned')


def complete_results(scope,rows_by_condition,metadata):
    if set(rows_by_condition)!=set(scope) or set(metadata)!=set(scope):
        raise ValueError('All declared groups, never automatic e0/missing-result substitutes, required')
    ids=[f'gsm8k:test:{i}' for i in range(1319)]
    groups={};paired={};base=rows_by_condition['M1']
    for group in scope:
        rows=rows_by_condition[group]
        if (len(rows)!=1319 or [r['source_id'] for r in rows]!=ids
                or any(type(r.get('correct')) is not bool for r in rows)
                or any(rows[i]['question_sha256']!=base[i]['question_sha256'] or rows[i]['gold']!=base[i]['gold'] for i in range(1319))):
            raise ValueError('Exact full1319 common ordered inventory and recomputed scores required')
        correct=sum(r['correct'] for r in rows)
        groups[group]=dict(correct=correct,total=1319,accuracy=correct/1319,
            supervised_tokens_including_EOS=metadata[group]['supervised_tokens_including_eos'],
            generation_seconds=sum(r['seconds'] for r in rows))
        paired[group]=dict(wrong_to_right=sum(not b['correct'] and r['correct'] for b,r in zip(base,rows)),
            right_to_wrong=sum(b['correct'] and not r['correct'] for b,r in zip(base,rows)),
            net_correct_difference=correct-sum(r['correct'] for r in base),
            accuracy_difference=(correct-sum(r['correct'] for r in base))/1319)
    return dict(protocol=PROTOCOL,complete=True,conditions=scope,official_test_count=1319,
        groups=groups,paired_vs_M1=paired,formal_seed=42,training_epochs=4,primary_checkpoint='last_epoch4',
        historical_public_test_exposure='known_partial_IDs_not_fully_recovered',
        historical_exposed_unexposed_sensitivity='not_identifiable_no_fabricated_clean_subset',
        multi_seed_effectiveness_established=False,public_test_used_to_select_configuration_or_checkpoint=False,
        unsupported_groups=[] if len(scope)==5 else ['M7','M6'])


def finalize():
    workflow=prepare();verify_inputs(workflow)
    if len(workflow['conditions'])==5:
        from evaluation_transfer128_v1 import validate_request
    else:
        from evaluation_selector_quality_v3 import validate_request
    # Admission BEFORE opening the actual official question inventory, including
    # the final aggregation process. Never infer completion from summary booleans.
    validate_request(workflow['conditions'][-1],42)
    from evaluate_selector_quality_v3 import validate_saved
    from evaluate import test_items
    items=test_items();rows_by_condition={};metadata={};paths=[SCOPE]
    for group in workflow['conditions']:
        folder=OUT/'evaluation'/PROTOCOL/group/'seed42';predictions=folder/'predictions.jsonl'
        sig=OUT/'signatures'/f'{PROTOCOL}_eval_GSM8K_{group}_s42.json'
        rows=checked_rows(predictions);signed=read(sig);paths += [sig,*verify_inputs(signed)]
        validate_saved(rows,items,group,42,baseline.sha(sig))
        if len(rows)!=1319:raise ValueError('Partial official predictions are not final results')
        rows_by_condition[group]=rows
        meta=ROOT/workflow['artifacts'][group]['adapter_directory']/'metadata.json'
        metadata[group]=read(meta);paths += [meta,predictions]
    result=complete_results(workflow['conditions'],rows_by_condition,metadata)
    paths += [Path(__file__),CPU,*[ROOT/n for n in workflow['inputs']]]
    result['inputs']={str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}
    freeze(RESULTS,result)
    lines=['# GSM8K 当前冻结协议最终结果','',
        '完整官方1319题；最终训练seed42、4epoch/最后epoch4。历史项目已看过部分官方测试，完整曝光ID未恢复，不能称首次盲测。','',
        '| 组别 | 正确/1319 | 准确率 | 相对M1净正确数 | 含EOS监督tokens |',
        '|---|---:|---:|---:|---:|']
    for group in workflow['conditions']:
        g=result['groups'][group];delta=result['paired_vs_M1'][group]['net_correct_difference']
        lines.append(f"| {group} | {g['correct']}/1319 | {100*g['accuracy']:.2f}% | {delta:+d} | {g['supervised_tokens_including_EOS']} |")
    lines += ['', 'G4/G5未执行：全部开发无正支持，保留负结果，不自动扩量。' if len(workflow['conditions'])==3 else
        '五组均来自实际最终产物；有限128确认采用训练根题派生变式，不能当外部抽样泛化证据。', '',
        '单seed差异不证明跨seed可复现；同根题/更新数不等于同token或算力预算。原教师91历史严格标记及未读材料、解释8192未读和新142未被原质量抽检覆盖等具体限制，见冻结披露。',
        '', '完整逐题预测、签名、原始配对变化和输入哈希保留在结果JSON及原评测目录。未按test改阈值、选checkpoint、换题或扩量。']
    report=RESULTS.with_suffix('.md');text='\n'.join(lines)+'\n'
    if report.exists() and report.read_text()!=text:raise ValueError('Final report changed')
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(text)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('exposure','prepare','run','finalize'))
    p.add_argument('--condition',choices=EVALUATION_ORDER);a=p.parse_args()
    if a.action=='exposure':v=freeze_exposure()
    elif a.action=='prepare':v=prepare()
    elif a.action=='finalize':v=finalize()
    else:
        if a.condition is None:p.error('--condition required')
        v=run(a.condition)
    print({k:v[k] for k in ('schema','conditions','complete','status') if k in v},flush=True)
