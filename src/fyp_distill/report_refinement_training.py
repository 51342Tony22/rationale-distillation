"""Require complete seven-subject pilot training/evaluation, then report counts."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from .common import artifact_dir, load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import GROUPS, unique_index
from .summarize import bootstrap_difference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    root=resolve_path(config["refinement"]["root"])
    report=artifact_dir(config,"mlx_report_dir","reports_mlx")
    questions=read_jsonl(config["data"]["dev_file"])
    expected=[r["id"] for r in questions]
    if len(expected)!=70 or len(set(expected))!=70:
        raise ValueError("Pilot evaluation must retain the frozen70 questions")
    subjects=sorted({r["subject"] for r in questions})
    audits=read_jsonl(root/"selection_audit.jsonl")
    bases=read_jsonl(root/"bases.jsonl")
    if not bases or {r["id"] for r in audits}!={r["id"] for r in bases}:
        raise ValueError("Unbuilt/incomplete training dataset")
    selected=unique_index(read_jsonl(root/"selected_questions.jsonl"))
    token_audit=json.loads(resolve_path(config["artifacts"]["token_audit"]).read_text())
    outputs={}
    metrics={}
    training={}
    for variant in ("base_student",*GROUPS):
        path=report/"dev"/variant/"seed_7"
        metric=json.loads((path/"metrics.json").read_text())
        predictions=read_jsonl(path/"predictions.jsonl")
        if [r["id"] for r in predictions]!=expected or metric["n"]!=70:
            raise ValueError("Incomplete independent evaluation")
        outputs[variant]={r["id"]:r for r in predictions}
        metrics[variant]={"correct":sum(r["correct"] for r in predictions),"n":70,
                          "invalid_answers":sum(r["primary_answer"] is None for r in predictions),
                          "generation_limit_hits":sum(r.get("generation_tokens",0)>=config["evaluation"]["max_new_tokens"] for r in predictions),
                          "generation_seconds":sum(r.get("elapsed_seconds",0) for r in predictions),
                          "by_subject":{s:{"correct":sum(outputs[variant][q["id"]]["correct"] for q in questions if q["subject"]==s),
                                            "n":sum(q["subject"]==s for q in questions)} for s in subjects}}
        if variant!="base_student":
            run=artifact_dir(config,"mlx_output_dir","outputs_mlx")/variant/"seed_7"
            metadata=json.loads((run/"run_metadata.json").read_text())
            if not (run/"adapter/adapters.safetensors").exists() or metadata["source_rows"]!=len(bases):
                raise ValueError("Every condition needs a completed compatible training artifact")
            training[variant]=metadata
    comparisons={}
    for left,right in config["evaluation"]["comparison_pairs"]:
        x={k:float(r["correct"]) for k,r in outputs[left].items()}
        y={k:float(r["correct"]) for k,r in outputs[right].items()}
        comparisons[left+"_minus_"+right]={**bootstrap_difference(x,y,config["evaluation"]["bootstrap_samples"],config["seed"]),
                                            "correct_count_difference":metrics[left]["correct"]-metrics[right]["correct"],
                                            "paired_wrong_to_right":sum(not outputs[right][k]["correct"] and outputs[left][k]["correct"] for k in expected),
                                            "paired_right_to_wrong":sum(outputs[right][k]["correct"] and not outputs[left][k]["correct"] for k in expected)}
    ledger=read_jsonl(config["refinement"]["budget_ledger"])
    local_seconds=sum(t["elapsed_seconds"] for r in read_jsonl(root/"screening.jsonl") for values in r["boundaries"].values() for t in values)
    for row in read_jsonl(root/"verification.jsonl"):
        local_seconds+=sum(t["elapsed_seconds"] for ts in [row["baseline"],*row["hints"].values()] for t in ts)
    summary={"created_at":datetime.now(UTC).isoformat(),"scope":"small pilot, single training seed, same question exposure not matched supervision tokens",
             "train_questions":len(bases),"train_by_subject":{s:sum(selected[b["id"]]["subject"]==s for b in bases) for s in subjects},
             "train_subject_levels":{s:sorted({selected[b["id"]]["level"] for b in bases if selected[b["id"]]["subject"]==s}) for s in subjects},
             "selected_but_not_trained":[k for k in selected if k not in {b["id"] for b in bases}],
             "source_quality_exclusions":read_jsonl(root/"data_quality_exclusions.jsonl"),
             "original_response_rejections":read_jsonl(root/"rejected_original_response_reviews.jsonl"),
             "original_processing_exclusions":read_jsonl(root/"original_processing_exclusions.jsonl"),
             "manual_math_corrected_originals":[{"source_id":b["id"],"candidate_id":b["candidate_id"],
                                                "derived_from_candidate_id":b["derived_from_candidate_id"]}
                                               for b in bases if b.get("manual_math_correction")],
             "normalized_originals":[b["id"] for b in bases if b.get("original_text_normalization")],
             "evaluation_questions":70,"accepted_questions":sum(a["accepted"] for a in audits),
             "token_audit":token_audit,"metrics":metrics,"comparisons":comparisons,
             "actual_independent_training_runs":sum(m.get("actual_new_training",True) for m in training.values()),
             "training_reuse":{v:m.get("reused_from") for v,m in training.items()},
             "actual_training_seconds":sum(m["elapsed_seconds"] for m in training.values() if m.get("actual_new_training",True)),
             "student_diagnostic_seconds":local_seconds,
             "evaluation_seconds_actual":sum(m["generation_seconds"] for v,m in metrics.items() if v=="base_student" or not training[v].get("reused_from")),
             "shared_new_charged_or_reserved_cny":sum(r["charged_cny"] for r in ledger),
             "shared_unknown_usage_reserve_cny":sum(r["charged_cny"] for r in ledger if not r.get("usage")),
             "historical_cny":2.549316,"combined_charged_or_reserved_cny":2.549316+sum(r["charged_cny"] for r in ledger),
             "human_output_audit":"separate supplementary artifact; does not replace frozen protocol scores"}
    assert summary["combined_charged_or_reserved_cny"]<=42
    report.mkdir(parents=True,exist_ok=True)
    write_jsonl(root/"training_pilot_summary.jsonl",[summary])
    labels=dict(zip(GROUPS,("A 原解答","B 普通扩写","C 随机位置","D 解释在开头","E 筛选局部解释","F 未筛选局部解释")))
    labels["base_student"]="未训练base"
    lines=["# 七学科小批次训练实验结果","",f"完成{len(bases)}道有效训练题、70道独立评测题；E接受{summary['accepted_questions']}题。一个训练种子，不能估计训练方差。", "",
           "| 条件 | 答对/70 | 无答案 | 达到输出上限 | 训练复用 |","|---|---:|---:|---:|---|"]
    for v,m in metrics.items():
        lines.append(f"| {labels[v]} | {m['correct']}/70 | {m['invalid_answers']} | {m['generation_limit_hits']} | {training.get(v,{}).get('reused_from') or '无'} |")
    lines += ["","## 按学科计数","","| 学科 | base | A | B | C | D | E | F |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in subjects:
        lines.append("| "+s+" | "+" | ".join(str(metrics[v]["by_subject"][s]["correct"])+"/10" for v in ("base_student",*GROUPS))+" |")
    lines += ["","## 完整性、成本与解释","",
              f"实际独立训练{summary['actual_independent_training_runs']}次；相同监督条件复用关系已公开，不能当作独立重复。相同题目曝光，监督token数量详见token_audit.json，不能声称token预算匹配。",
              f"共享旧+新费用估算/未知预留¥{summary['combined_charged_or_reserved_cny']:.6f}，其中未知用量预留¥{summary['shared_unknown_usage_reserve_cny']:.6f}仍保留。实际平台账单可能不同。",
              f"原解答中{len(summary['manual_math_corrected_originals'])}条包含显式人工辅助数学修正、{len(summary['normalized_originals'])}条规范化重复think文本；原响应与旧候选保留，详见ORIGINAL_MATH_CORRECTIONS_ZH.md与各base来源字段，不能称全自动R1原文。",
              f"实际训练学科/难度覆盖：{json.dumps(summary['train_subject_levels'],ensure_ascii=False)}；固定选择中未训练ID及源题/响应/处理排除理由保存在机器汇总，不能把配置名21当成全部完成。",
              "评测为冻结70题、greedy无hint、first_answer_v3固定；数据选择与阈值不按评测调参。评测覆盖七学科、难度1–4，未覆盖难度5。",
              "格式不接受与真实数学错误需在辅助人工审核中区分。自动分数只评分首个明确答案，可能漏掉后续矛盾；不能把全部分数差归为推理改善。",
              "21题目标、70题评测、单seed只作是否值得扩大的一次可行性信号，不证明稳定主实验效果；没有提升也不能直接否定整个研究方向。",
              "", "逐题配对差及bootstrap区间（仅题目抽样不估训练seed方差）：", "", "```json", json.dumps(comparisons,ensure_ascii=False,indent=2),"```",""]
    if summary["accepted_questions"]==0:
        lines += ["E未接受任何解释，A–E训练文本相同；本批没有形成五种不同的实际干预，不能从相同结果推断局部位置与普通扩写等效。", ""]
    (report/"TRAINING_PILOT_REPORT_ZH.md").write_text("\n".join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
