"""Paired exploratory pilot report, keeping frozen and sensitivity scores apart."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from collections import Counter

import baseline
import score_sensitivity_v2 as sensitive
from pilot_common import BASE, C, DATA, OUT, ROOT, rows, sources

LABEL = {"student_base": "未训练学生", "ordinary_cot": "普通CoT", "difficulty_only": "仅难度补充", "difficulty_attention": "难度＋attention补充"}


def paired(a, b):
    import numpy as np
    # Difference is attention minus reference, same questions in the same order.
    delta = np.array(a, dtype=np.int32) - np.array(b, dtype=np.int32)
    rng = np.random.default_rng(C["seed"])
    samples = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    trials = wins + losses
    p = min(1.0, 2 * sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / 2**trials) if trials else 1.0
    return {"difference": float(delta.mean()), "wins": wins, "losses": losses,
            "bootstrap95": [float(x) for x in np.quantile(samples, [.025, .975])], "mcnemar_exact_two_sided": p}


def main():
    items = sources("dev")
    manifest = json.loads((DATA / "manifest.json").read_text())
    baseline_source = ROOT / BASE["output_dir"] / "dev_baseline/student_base/predictions.jsonl"
    assert baseline.sha(baseline_source) == manifest["baseline_predictions_sha256"]
    assert baseline.sha(Path(sensitive.__file__)) == manifest["sensitivity_parser_sha256"]
    base_lookup = {r["source_id"]: r for r in rows(baseline_source)}
    predictions = {"student_base": [base_lookup[r["source_id"]] for r in items]}
    for condition in C["conditions"]:
        predictions[condition] = rows(OUT / "evaluation" / condition / "predictions.jsonl")
    report = {"stage": "generated_results_needs_full_text_audit", "evaluation": "100 previously inspected development questions, not formal test",
              "conditions": {}, "attention_paired": {}, "training_protocol": json.loads((OUT / "training_protocol.json").read_text()),
              "source_gold_issues": [{"source_id": "gsm8k:train:2867", "original_gold": "23", "audit_math_answer": "40.5", "primary_gold_changed": False},
                                     {"source_id": "gsm8k:train:2289", "issue": "over $50 versus reference using reaching $50", "primary_gold_changed": False}],
              "limitations": ["Single seed; intervals resample questions, not training runs.",
                              "Question exposure matched; supervised tokens and explanation lengths differ.",
                              "Development outputs were previously inspected and the sensitivity parser was designed on development data.",
                              "Attention is a relevance heuristic; random and complete budget controls deferred.",
                              "Teacher is a pinned community 4bit conversion; exact upstream original-weight commit unspecified. Student is original BF16; no paid API.",
                              "No exact question overlap was found, but semantic duplication and benchmark pretraining contamination cannot be excluded.",
                              "Numerical exact match does not validate the whole reasoning; full-text audit coverage must be read separately."]}
    report["source_gold_issues"] += [
        {"source_id": "gsm8k:train:2882", "original_gold": "4", "audit_math_answer": "0 under one shared gallon wording", "primary_gold_changed": False},
        {"source_id": "gsm8k:train:3360", "issue": "each animal type 6 minutes versus all types 54 minutes", "primary_gold_changed": False},
        {"source_id": "gsm8k:train:4130", "issue": "which is three times: increase versus total amount", "primary_gold_changed": False},
        {"source_id": "gsm8k:train:6480", "issue": "three times more versus three times as many", "primary_gold_changed": False}]
    preparation_files = ["teacher_targets.jsonl", "difficulty_only_edits.jsonl", "edit_probe.jsonl", "edit_probe_v2.jsonl", "edit_probe_v3.jsonl",
                         "difficulty_only_edits_v3.jsonl", "difficulty_attention_edits_v3.jsonl", "difficulty_only_edits_v4.jsonl", "difficulty_attention_edits_v4.jsonl"]
    report["saved_material_generation"] = {}
    for file in preparation_files:
        records = rows(OUT / file)
        fresh = [r for r in records if not r.get("reused_from")]
        report["saved_material_generation"][file] = {"saved_records": len(records), "new_generations": len(fresh),
                                                     "generation_seconds": sum(r["seconds"] for r in fresh),
                                                     "generated_tokens": sum(r["generation_tokens"] for r in fresh),
                                                     "sha256": baseline.sha(OUT / file)}
    report["saved_material_generation_total"] = {key: sum(r[key] for r in report["saved_material_generation"].values())
                                                 for key in ["new_generations", "generation_seconds", "generated_tokens"]}
    report["saved_material_generation_total"]["limitation"] = "Saved completed records only; excludes model loads, unsaved interrupted work, localization and human review. Engineering attempts retained, not best-of target selection."
    score_arrays = {}
    for name, predicted in predictions.items():
        assert len(predicted) == len(items)
        for row, item in zip(predicted, items):
            assert row["source_id"] == item["source_id"] and row["gold"] == item["gold"]
            assert hashlib.sha256(row["output"].encode()).hexdigest() == row["output_sha256"]
        sensitivity = [sensitive.parse(r["scoring_text"]) for r in predicted]
        v1 = [r["correct"] for r in predicted]
        v2 = [v is not None and v == baseline.number(item["gold"]) for (v, _), item in zip(sensitivity, items)]
        score_arrays[name] = {"v1": v1, "v2": v2}
        audits = rows(OUT / f"{name}_evaluation_reviews.jsonl")
        if name == "student_base":
            audits = json.loads((ROOT / BASE["output_dir"] / "dev_preview_audit.json").read_text())["records"] + audits
        lookup = {r["source_id"]: r for r in audits}
        assert len(lookup) == len(audits), "Duplicate full-text review"
        assert set(lookup) <= {r["source_id"] for r in predicted}
        for r in predicted:
            if r["source_id"] in lookup:
                assert r["output_sha256"] == lookup[r["source_id"]]["output_sha256"]
        audit_status = {"full_text_reviewed": len(lookup), "expected": len(items), "complete": len(lookup) == len(items)}
        audit_status["mathematics_categories"] = dict(Counter(r.get("mathematics", "unspecified") for r in lookup.values()))
        audit_file = OUT / f"{name}_evaluation_reviews.jsonl"
        audit_status["review_file_sha256"] = baseline.sha(audit_file) if audit_file.exists() else None
        if name == "student_base":
            audit_status["existing_first20_review_sha256"] = baseline.sha(ROOT / BASE["output_dir"] / "dev_preview_audit.json")
        summary = {"n": len(items), "v1_correct": sum(v1), "v1_accuracy": sum(v1) / len(items),
                   "v2_correct": sum(v2), "v2_accuracy": sum(v2) / len(items),
                   "v1_invalid": sum(r["invalid"] for r in predicted), "v2_invalid": sum(v is None for v, _ in sensitivity),
                   "strict_format_correct": sum(r["strict_format_correct"] for r in predicted),
                   "length_stops": sum(r["finish_reason"] == "length" for r in predicted),
                   "next_question_stops": sum(r["finish_reason"] == "next_question" for r in predicted),
                   "generation_seconds": sum(r["seconds"] for r in predicted), "generated_tokens": sum(r["generation_tokens"] for r in predicted),
                   "mean_generated_tokens": sum(r["generation_tokens"] for r in predicted) / len(predicted),
                   "shared_explanation_phrase_outputs": sum("This step is valid because " in r["output"] for r in predicted),
                   "repeated_shared_phrase_outputs": sum(r["output"].count("This step is valid because ") > 1 for r in predicted),
                   "generated_code_style_outputs": sum("```python" in r["output"] or "```output" in r["output"] for r in predicted),
                   "actual_tool_executions": 0,
                   "math_audit": audit_status}
        if name != "student_base":
            summary["training"] = json.loads((OUT / "training" / name / "metadata.json").read_text())
        report["conditions"][name] = summary
    attention = score_arrays["difficulty_attention"]
    for reference in ["student_base", "ordinary_cot", "difficulty_only"]:
        report["attention_paired"][reference] = {score: paired(attention[score], score_arrays[reference][score]) for score in ["v1", "v2"]}
        for score in ["v1", "v2"]:
            report["attention_paired"][reference][score]["improved_source_ids"] = [item["source_id"] for item, a, b in zip(items, attention[score], score_arrays[reference][score]) if a and not b]
            report["attention_paired"][reference][score]["regressed_source_ids"] = [item["source_id"] for item, a, b in zip(items, attention[score], score_arrays[reference][score]) if b and not a]
    complete = all(r["math_audit"]["complete"] for r in report["conditions"].values())
    report["stage"] = "pilot_complete_full_text_audited" if complete else "generated_results_needs_full_text_audit"
    report["paid_api_cost_cny"] = 0
    report["unpriced_costs"] = ["local electricity", "human review and implementation time"]
    condition_data = report["conditions"]
    attention_only = report["attention_paired"]["difficulty_only"]["v2"]
    attention_cot = report["attention_paired"]["ordinary_cot"]["v2"]
    report["interpretation_zh"] = [
        f'普通CoT在同100开发题的v2数字分由未训练的{condition_data["student_base"]["v2_correct"]}提升到{condition_data["ordinary_cot"]["v2_correct"]}；这支持该小样本训练有方向性收益，不能全归因于数学推理改善，格式风格也改变。',
        f'attention组相对仅难度组净{attention_only["difference"]*100:+.1f}个百分点，相对普通CoT净{attention_cot["difference"]*100:+.1f}个百分点；这些是固定开发题上的结果。',
        '尚不能证明attention带来可靠额外收益：只有一个训练seed，开发集已被查看，且没有随机线索与完整提示/长度预算对照。本版attention提供前文线索，不参与改变NLL所选位置。',
        '保留解释重复、任务目标误读、漏条件、算术错误、单位等价和源gold问题；冻结数字主分不因辅助审核改动。']
    baseline.save(OUT / "results.json", report)
    lines = ["# 50题探索小试结果", "", "状态：" + ("四条件共400条输出已全文辅助审核。" if complete else "生成结果，全文辅助审核仍未完成。") + "100题为开发集，尚未进行1319题正式测试。", ""]
    lines += report["interpretation_zh"] + ["", "教师：Qwen2.5-Math-7B-Instruct的固定社区MLX4bit版本；学生：Qwen2.5-Math-1.5B Base原BF16。数据：官方GSM8K冻结划分，前50训练候选共同审核后48题，固定前100开发题；全部本地调用。", "",
             "| 条件 | 原v1正确 | v2敏感性正确 | v1/v2 invalid | 长度停止 |", "|---|---:|---:|---:|---:|"]
    for name, data in report["conditions"].items():
        lines.append(f'| {LABEL[name]} | {data["v1_correct"]}/100 | {data["v2_correct"]}/100 | {data["v1_invalid"]}/{data["v2_invalid"]} | {data["length_stops"]} |')
    lines += ["", "v1与v2是同一输出的两种解析，二者差值不是训练收益。源gold速度题2867保留原23，数学审核认为应40.5，单列披露。", "",
              "## attention组相对其他组的配对变化", "", "| 参照 | v1净变化/改善/退步 | v2净变化/改善/退步 | v2 bootstrap95%差值区间 |", "|---|---:|---:|---:|"]
    for name, data in report["attention_paired"].items():
        a, b = data["v1"], data["v2"]
        lines.append(f'| {LABEL[name]} | {a["difference"]*100:+.1f}pp / {a["wins"]} / {a["losses"]} | {b["difference"]*100:+.1f}pp / {b["wins"]} / {b["losses"]} | [{b["bootstrap95"][0]*100:.1f}, {b["bootstrap95"][1]*100:.1f}]pp |')
    lines += ["", "## 数据、训练与边界", "", f'共同有效训练题{report["training_protocol"]["eligible"]}/50，排除原因和每组监督token数见results.json。', ""]
    lines += ["| 条件 | 全文审核 | 一遍监督tokens | 训练秒 | 生成秒 |", "|---|---:|---:|---:|---:|"]
    for name, data in report["conditions"].items():
        trained = data.get("training", {})
        token_audit = report["training_protocol"]["token_audit"].get(name, {})
        lines.append(f'| {LABEL[name]} | {data["math_audit"]["full_text_reviewed"]}/100 | {token_audit.get("supervised_tokens_per_epoch", "—")} | {trained.get("elapsed_seconds", 0):.2f} | {data["generation_seconds"]:.2f} |')
    lines += ["", "3组均独立从同一原始学生训练，4轮、192微步、40更新；LoRA参数1,245,184。无旧adapter初始化、无padding监督、无截断。训练同题曝光，不是同监督token预算。",
              "", "位置均由学生NLL启发式选择；attention在这个版本中为局部解释提供前文线索，尚未比较不同位置选择器。48对补充中15对解释相同，4对长度差超过16tokens（最大40），相关性和长度效应尚未完全分离。",
              "", "材料准备保存的新生成共261次，约819.79秒，包括失败提示尝试及修正流式小数截断后的两组重生成；这些是本地调用，付费API支出0元，未核算电力和人工成本。该秒数不含模型加载、定位和人工审核，不能视为全部墙钟耗时。",
              "", "源gold/措辞问题见results.json与全文审核；2867、2882等不改冻结主分。生成的Python/output均为文本，未执行工具；数值正确仍可能含单位错误、自我纠错或前后矛盾，辅助分类单列。", ""]
    lines += ["## 输出行为与全文审核", "", "| 条件 | 平均生成tokens | 含补充桥句的输出 | 桥句多次出现的输出 | 代码风格输出 |", "|---|---:|---:|---:|---:|"]
    for name, data in report["conditions"].items():
        lines.append(f'| {LABEL[name]} | {data["mean_generated_tokens"]:.2f} | {data["shared_explanation_phrase_outputs"]} | {data["repeated_shared_phrase_outputs"]} | {data["generated_code_style_outputs"]} |')
    lines += ["", "桥句计数是文本行为指标，多次出现并不一概代表错误；具体重复与推理问题已逐条记录。例：仅难度组4130循环至1024 tokens而未完成；补充组2005多次复述题设但最终数值仍正确；5182把卖给两位朋友误算三根；700输出300美分等价3美元但数字主分拒绝。6481中普通CoT/仅难度组有真实算术错误，attention组则正确输出等价的2/3立方英尺，冻结数字分同样拒绝，不能将三者一律归为格式问题。", "",
              "attention组还出现5032将平板充电时间误套手机、230把速度与时间反比写反；4441最终22正确但桥句把150+16说成168；4857最终196正确但桥句把240只幼鸽称为总鸽数，与后续280矛盾。1400最终22正确，但桥句解释加10却放在2×6之前，位置不对齐。以上均保持原分，保留具体文本问题。", "",
              "辅助审核是本次助手对完整文本的复核，非独立盲评者或形式证明；各组记录绑定原输出SHA256，详见同目录*_evaluation_reviews.jsonl。", "", "## 适用边界", ""]
    lines += ["- " + item for item in report["limitations"]]
    (OUT / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["conditions"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
