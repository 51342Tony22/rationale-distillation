"""Report the separately trained 100-candidate pilot; never blend adapters' outputs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import baseline
import pilot_common as common
import score_sensitivity_v2 as sensitive
from pilot100 import C, CONFIG, DATA, OLD, OUT, ROOT
from pilot_report import paired

LABEL={"student_base":"未训练学生","ordinary_cot":"普通CoT","difficulty_only":"仅难度补充","difficulty_attention":"难度＋attention"}


def reviews(name):
    audit=common.rows(OUT / f"{name}_evaluation_reviews.jsonl")
    if name=="student_base":
        audit+=json.loads((ROOT / "output/math_distill_v1/dev_preview_audit.json").read_text())["records"]
        audit+=common.rows(OLD / "student_base_evaluation_reviews.jsonl")
    lookup={r["source_id"]:r for r in audit}
    assert len(lookup)==len(audit), "Duplicate audit"
    return lookup


def make_report(final=False):
    manifest=json.loads((DATA / "manifest.json").read_text())
    assert baseline.sha(CONFIG)==manifest["config_sha256"]
    assert baseline.sha(Path(sensitive.__file__))==manifest["sensitivity_parser_sha256"]
    items=json.loads((DATA / "dev.json").read_text())
    protocol=json.loads((OUT / "training_protocol.json").read_text())
    report={"experiment_id":C["experiment_id"],"train_candidates":100,"eligible_training":protocol["eligible"],
            "evaluation_count":250,"evaluation":"exploratory development extension after inspecting pilot50; official test untouched",
            "actual_independent_training":3,"seed":7,"paid_api_cost_cny":0,
            "training_protocol":protocol,"conditions":{},"attention_paired":{},"treatments_vs_base":{},"old_pilot_common100":{},
            "limitations":["One seed; confidence intervals resample questions, not independent training runs.",
              "Exploratory paired comparisons are not adjusted for multiple testing.",
              "The development expansion was decided after inspecting the earlier pilot; not an untouched final test.",
              "Attention only highlights prior context for explanations at the same NLL-selected position.",
              "No hard-answer, soft-distribution, random-evidence or strict prompt/token-budget controls in this run.",
              "Same question exposure; unequal supervised token budgets and explanation lengths.",
              "No tool execution; any generated code/output is model text.",
              "Full-text assistant audit is auxiliary, not independent blind review or a mathematical proof.",
              "Source gold and wording issues retained without changing the frozen primary scores.",
              "Sensitivity parser v2 was developed after inspecting the first 20 development outputs; v1 remains primary and both are reported unchanged.",
              "Teacher is pinned community MLX4bit, original upstream conversion commit unspecified; student original BF16.",
              "Exact and lexical overlap checks do not exclude semantic templates or pretraining contamination."]}
    scores={}
    base_settings=json.loads(CONFIG.with_name("config.json").read_text())
    report["frozen_sources"]={k:base_settings[k] for k in [
        "teacher_repo","teacher_revision","teacher_precision",
        "student_repo","student_revision","student_precision",
        "dataset_repo","dataset_revision","student_prompt"]}
    report["material_audit"]={kind:{"generated":len(common.rows(OUT / generated)),
        "reviewed":len(common.rows(OUT / audit)),
        "accepted":sum(r["accepted"] for r in common.rows(OUT / audit)),
        "rejected_ids":[r["source_id"] for r in common.rows(OUT / audit) if not r["accepted"]]}
        for kind,generated,audit in [("teacher","teacher_targets.jsonl","teacher_reviews.jsonl"),
             ("location","locations.jsonl","location_reviews.jsonl"),
             *[(n,f"{n}_edits_v4.jsonl",f"{n}_edit_v4_reviews.jsonl") for n in ["difficulty_only","difficulty_attention"]]]}
    report["eligible_attention_evidence_relevance"]=dict(Counter(r.get("evidence_relevance","unspecified")
        for r in common.rows(OUT / "location_reviews.jsonl") if r["source_id"] in protocol["source_ids"]))
    report["new_material_generation"]={}
    for filename in ["teacher_targets.jsonl","difficulty_only_edits_v4.jsonl","difficulty_attention_edits_v4.jsonl"]:
        generated=common.rows(OUT / filename)
        reused=len(common.rows(OLD / filename))
        fresh=generated[reused:]
        report["new_material_generation"][filename]={"reused_prefix":reused,"new_generations":len(fresh),
            "generation_seconds":sum(r["seconds"] for r in fresh),
            "generated_tokens":sum(r["generation_tokens"] for r in fresh)}
    report["source_gold_and_wording_issues"]=[
        {"source_id":"gsm8k:train:2867","issue":"Speed relation implies40.5, original gold23 retained."},
        {"source_id":"gsm8k:train:2882","issue":"One shared gallon gives0 remaining, original gold4 retained."},
        {"source_id":"gsm8k:train:5134","issue":"Inverse percentage: Ann300/0.85, not300*1.15; original gold983 retained despite inconsistent integer counts."},
        {"source_id":"gsm8k:train:7274","issue":"Joint probability6% requires unstated independence of the two marginals."},
        {"source_id":"gsm8k:train:1138","issue":"Saved in a month can denote cumulative140 versus reference weekly amount60."},
        {"source_id":"gsm8k:train:3155","issue":"Returned item/comparison wording ambiguity; baseline reasoning also has cancelling errors."},
        {"source_id":"gsm8k:train:2069","issue":"Question asks shoes but reference62 counts pairs;124 individual shoes."},
        {"source_id":"gsm8k:train:6832","issue":"Last4hours can be interpreted one-way or round-trip;250 reference versus500 under round-trip reading."},
        {"source_id":"gsm8k:train:1868","issue":"Reference sums lead-distance statements as sequential route segments; distances alone do not specify catch-up ability."},
        {"source_id":"gsm8k:train:675","issue":"Difference P(lose both)-P(win both)=20 percentage points follows from marginals; individual35%/15% require independence, not specified."},
        {"source_id":"gsm8k:train:337","issue":"Union31 counts wine-or-soda drinkers; equating this with all attendees assumes no neither group."},
        {"source_id":"gsm8k:train:2289","issue":"Discounted basket41; reaching50 needs9 but strictly over50 needs more than9."},
        {"source_id":"gsm8k:train:6795","issue":"Wording only buy1whole bottle conflicts with the apparent intended whole-bottle requirement3."},
        {"source_id":"gsm8k:train:6480","issue":"Three times more is interpreted as three times as many by reference; wording ambiguous."},
        {"source_id":"gsm8k:train:4823","issue":"Times more wording has multiplicative ambiguity; retain frozen reference interpretation."}]
    report["source_gold_and_wording_issues"] += [
        {"source_id":"gsm8k:train:4130","issue":"Which quantity is three times the old lifting capacity is ambiguous; adding old and new capacity also changes the requested target."},
        {"source_id":"gsm8k:train:6271","issue":"Tip can be based on pre-tax100 or post-tax104, giving110 versus110.24; reference uses pre-tax."}]
    recovery_path=OUT / "evaluation_recovery_preflight.json"
    if recovery_path.exists():
        report["evaluation_recovery"]=json.loads(recovery_path.read_text())
    for name in ["student_base",*C["conditions"]]:
        path=OUT / "evaluation" / name / "predictions.jsonl"
        predictions=common.rows(path)
        assert len(predictions)==len(items)==250, f"Incomplete evaluation: {name}"
        audit=reviews(name)
        assert set(audit)<={r["source_id"] for r in predictions}
        v1,v2=[],[]
        for row,item in zip(predictions,items):
            assert row["source_id"]==item["source_id"] and row["gold"]==item["gold"] and row["question_sha256"]==item["question_sha256"]
            assert common.digest_text(row["output"])==row["output_sha256"]
            assert all(row[k]==v for k,v in baseline.score(row["scoring_text"],item["gold"]).items())
            value,_=sensitive.parse(row["scoring_text"])
            v1.append(row["correct"])
            v2.append(value is not None and value==baseline.number(item["gold"]))
            if row["source_id"] in audit:
                assert audit[row["source_id"]]["output_sha256"]==row["output_sha256"] and audit[row["source_id"]].get("mathematics")
        scores[name]={"v1":v1,"v2":v2}
        summary={"n":250,"v1_correct":sum(v1),"v2_correct":sum(v2),"v1_invalid":sum(r["invalid"] for r in predictions),
                 "v2_invalid":sum(sensitive.parse(r["scoring_text"])[0] is None for r in predictions),
                 "length_stops":sum(r["finish_reason"]=="length" for r in predictions),
                 "next_question_stops":sum(r["finish_reason"]=="next_question" for r in predictions),
                 "mean_generated_tokens":sum(r["generation_tokens"] for r in predictions)/250,
                 "generation_seconds_total_including_reused_base":sum(r["seconds"] for r in predictions),
                 "new_generation_seconds":sum(r["seconds"] for r in (predictions[200:] if name=="student_base" else predictions)),
                 "reused_predictions":200 if name=="student_base" else 0,
                 "shared_bridge_outputs":sum("This step is valid because " in r["output"] for r in predictions),
                 "multiple_bridge_outputs":sum(r["output"].count("This step is valid because ")>1 for r in predictions),
                 "generated_code_style_outputs":sum("```python" in r["output"] or "```output" in r["output"] for r in predictions),
                 "actual_tool_executions":0,"predictions_sha256":baseline.sha(path),
                 "full_text_reviewed":len(audit),"audit_complete":len(audit)==250,
                 "mathematics_audit_categories":dict(Counter(r["mathematics"] for r in audit.values()))}
        if name!="student_base":
            summary["training"]=json.loads((OUT / "training" / name / "metadata.json").read_text())
        report["conditions"][name]=summary
        summary["auxiliary_reasoning_error_examples"]=[{"source_id":r["source_id"],"v1":r["correct"],
             "v2":scores[name]["v2"][i],"audit":audit[r["source_id"]]}
             for i,r in enumerate(predictions) if r["source_id"] in audit
             and audit[r["source_id"]]["mathematics"].startswith(("incorrect","correct_final_incorrect","correct_final_inconsistent"))][:5]
        old=json.loads((OLD / "results.json").read_text())["conditions"][name]
        report["old_pilot_common100"][name]={"old_training_eligible":48,"new_training_eligible":protocol["eligible"],
             "old_v1_correct":old["v1_correct"],"new_v1_correct":sum(v1[:100]),"old_v2_correct":old["v2_correct"],"new_v2_correct":sum(v2[:100]),
             "note":"Same first 100 questions; old and new adapters separately evaluated, no output prefix reuse for trained groups."}
    for name in ["student_base","ordinary_cot","difficulty_only"]:
        report["attention_paired"][name]={}
        for metric in ["v1","v2"]:
            a,b=scores["difficulty_attention"][metric],scores[name][metric]
            p=paired(a,b)
            p["improved_source_ids"]=[r["source_id"] for r,x,y in zip(items,a,b) if x and not y]
            p["regressed_source_ids"]=[r["source_id"] for r,x,y in zip(items,a,b) if y and not x]
            report["attention_paired"][name][metric]=p
    complete=all(r["audit_complete"] for r in report["conditions"].values())
    for name in C["conditions"]:
        report["treatments_vs_base"][name]={metric:paired(scores[name][metric],scores["student_base"][metric])
                                              for metric in ["v1","v2"]}
    if final:
        assert complete,"Full-text output audits incomplete"
    report["stage"]="pilot100_complete_full_text_audited" if complete else "results_pending_full_text_audit"
    baseline.save(OUT / "results.json",report)
    lines=["# 100候选训练／250开发题扩展结果","",f'状态：{report["stage"]}。共同有效训练{protocol["eligible"]}/100；三个新adapter均独立从同一原始1.5B学生初始化。',
           "", "本报告为开发探索结果。原48题训练小试和所有原输出保留，正式1319题test未运行。未训练200输出复用，只新增50；新训练模型各自重新生成全部250题。", "",
           f'教师：{base_settings["teacher_repo"]}（本地4bit）；学生：{base_settings["student_repo"]}（原始BF16）。数据：GSM8K，版本{base_settings["dataset_revision"]}，训练与开发均取官方train的冻结互斥子集。模型固定revision与prompt见results.json。', "",
           "| 条件 | v1正确 | 固定v2正确 | v1/v2 invalid | 长度停止 | 全文审核 |","|---|---:|---:|---:|---:|---:|"]
    for name,r in report["conditions"].items():
        lines.append(f'| {LABEL[name]} | {r["v1_correct"]}/250 | {r["v2_correct"]}/250 | {r["v1_invalid"]}/{r["v2_invalid"]} | {r["length_stops"]} | {r["full_text_reviewed"]}/250 |')
    lines += ["", "v1/v2为同一输出的两种固定解析，解析差异不是训练提升。v2在查看最初20道开发输出后制定，不能称盲测主指标。数字命中不保证解释正确，原gold和单位/措辞问题按辅助审核披露。", "", "## attention相对其他组的同题配对比较", "",
              "| 参照 | v1净变化／改对／改错 | v2净变化／改对／改错 | v2按题bootstrap95%区间 | v2精确McNemar p |","|---|---:|---:|---:|---:|"]
    for name,value in report["attention_paired"].items():
        a,b=value["v1"],value["v2"]
        lines.append(f'| {LABEL[name]} | {a["difference"]*100:+.1f}pp / {a["wins"]} / {a["losses"]} | {b["difference"]*100:+.1f}pp / {b["wins"]} / {b["losses"]} | [{b["bootstrap95"][0]*100:.1f}, {b["bootstrap95"][1]*100:.1f}]pp | {b["mcnemar_exact_two_sided"]:.4g} |')
    lines += ["", "区间使用10000次按题重采样、seed7；不衡量训练seed方差。原始逐题改善/退步ID和v1区间见results.json。", "", "## 训练量扩充：共同前100评测题", "",
              "| 条件 | 原48题训练v1/v2 | 新训练v1/v2 |","|---|---:|---:|"]
    for name,r in report["old_pilot_common100"].items():
        lines.append(f'| {LABEL[name]} | {r["old_v1_correct"]}/{r["old_v2_correct"]} | {r["new_v1_correct"]}/{r["new_v2_correct"]} |')
    evidence=report["attention_paired"]["difficulty_only"]["v2"]
    ci=evidence["bootstrap95"]
    interpretation=("区间跨过或接触0，本轮未建立attention相对仅难度的额外优势。" if ci[0]<=0<=ci[1] else
        "按题区间未跨0，显示本次固定模型与开发题上的差异；仍不能外推训练seed稳定性或最终测试表现。")
    lines += ["", "此表各数字分母均为100，仅作同题训练规模探索；不能把旧100题总分与新250题总分直接比较。", "", "## 如何解释本轮结果", "",
              f'主要机制对照是attention对仅难度：固定v2净差{evidence["difference"]*100:+.1f}个百分点。{interpretation}',
              "普通CoT对未训练学生检验本轮监督训练的整体变化；attention对普通CoT检验完整补充方法的净收益。它们与attention自身的增量不是同一结论，全部配对数据见results.json。",
              "", f'97个共同有效样本中，{protocol["paired_edit_lengths"]["identical_explanations"]}对解释完全相同，{protocol["paired_edit_lengths"]["pairs_over_16_tokens"]}对长度差超过16tokens、最大差{protocol["paired_edit_lengths"]["max_difference"]}。前文线索相关性分类（辅助主观判断、非盲审）为{report["eligible_attention_evidence_relevance"]}。不能将本轮称为严格长度控制、因果attention定位或通用数学步骤发现。',
              "", "## 训练、生成与行为", "",
              "| 条件 | 一遍监督tokens | 训练秒 | 本轮新生成秒 | 平均生成tokens | 多桥句输出数 |","|---|---:|---:|---:|---:|---:|"]
    for name,r in report["conditions"].items():
        token=protocol["token_audit"].get(name,{})
        lines.append(f'| {LABEL[name]} | {token.get("supervised_tokens_per_epoch","—")} | {r.get("training",{}).get("elapsed_seconds",0):.2f} | {r["new_generation_seconds"]:.2f} | {r["mean_generated_tokens"]:.2f} | {r["multiple_bridge_outputs"]} |')
    lines += ["", "4轮、batch1、梯度累积5、seed7；仅completion及EOS监督，无prompt/padding目标、无训练截断。实际微步、更新、adapter哈希和费用见results.json及metadata。付费API支出0，未核算电力和人工成本。", "",
              "新增材料生成50条教师目标及两组各49条补充，均保留原始输出并逐条审核；旧50条前缀原字节复用。各阶段生成秒数见results.json，不含加载、定位、审核与未保存工作，不能等同整批墙钟时间。", "",
              "训练全部完成后，原入口在新增基线生成前因旧signature的SHA字符串格式读取错误停止；单独的恢复入口修正校验格式，验证已签名材料与adapter后接续评测。未修改已签名代码、重训或覆盖旧输出，失败日志保留。", "",
              "多桥句指标仅计算共同句式出现次数，不将解释不同步骤一概视为重复错误。全文记录另审计算、条件、任务目标、前后矛盾、循环、单位与截断。", "",
              "## 全文审核的具体问题示例", ""]
    for name,summary in report["conditions"].items():
        for example in summary["auxiliary_reasoning_error_examples"][:3]:
            note=example["audit"].get("reason", "既有全文审核分类："+example["audit"]["mathematics"]+"；原审核保留，未补写旧记录。")
            lines.append(f'- {LABEL[name]} / {example["source_id"]}（v1={example["v1"]}、v2={example["v2"]}）：{note}')
    lines += ["", "所有逐题审核见各evaluation_reviews.jsonl；原200未训练预测的前100审核分别复用dev_preview_audit.json与pilot50原80条，其余150条保存在本目录。这些辅助判断不重写主分，也不是独立人工盲审。", "",
              "源题问题保留：2867速度关系应得40.5而gold23；2882按共享一加仑应剩0而gold4；5134逆百分比应为300/0.85，原gold983保留；7274的6%联合概率需额外独立性假设；1138的月累计与当周金额、3155的返还/比较措辞均有歧义。单位等价、错误抵消与正确最终数值但推理矛盾另记，不修改主分。", "", "## 适用边界", ""]
    lines += ["- "+r for r in report["limitations"]]
    (OUT / "RESULTS_ZH.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"stage":report["stage"],"eligible":protocol["eligible"],"conditions":{k:{q:v[q] for q in ["v1_correct","v2_correct","full_text_reviewed"]} for k,v in report["conditions"].items()}},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final",action="store_true")
    make_report(parser.parse_args().final)
