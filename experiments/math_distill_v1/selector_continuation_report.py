"""Report the six-case engineering check; never assert selector efficacy."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import baseline
from pilot_common import digest_text, rows
import selector_continuation as run
import selector_masking_probe as probe


def position_results(predictions, replicates):
    grouped = {}
    for row in predictions:
        key = row["position_id"]
        group = grouped.setdefault(key, {})
        pair = (row["replicate"], row["condition"])
        assert pair not in group, "Duplicate continuation"
        group[pair] = row
    result = {}
    expected = {(i, c) for i in range(replicates) for c in ["explanation", "no_explanation"]}
    for key, group in grouped.items():
        assert set(group) == expected, "Incomplete paired continuation"
        for i in range(replicates):
            assert group[i, "explanation"]["seed"] == group[i, "no_explanation"]["seed"]
        counts = {c: sum(group[i, c]["correct"] for i in range(replicates)) for c in ["explanation", "no_explanation"]}
        counts_v2 = {c: sum(group[i, c]["v2_correct"] for i in range(replicates)) for c in ["explanation", "no_explanation"]}
        result[key] = {"source_id": next(iter(group.values()))["source_id"],
                       "step_index": next(iter(group.values()))["step_index"],
                       "replicates": replicates, "counts": counts, "counts_v2": counts_v2,
                       "delta": (counts["explanation"] - counts["no_explanation"]) / replicates,
                       "delta_v2": (counts_v2["explanation"] - counts_v2["no_explanation"]) / replicates}
    return result


def main():
    directory = run.OUT
    # Verify every signed implementation and input, not just the outputs.
    for stage in ["edits", "evaluation"]:
        signed = json.loads((directory / f"{stage}_signature.json").read_text())
        for name, expected in signed["inputs"].items():
            assert baseline.sha(run.ROOT / name) == expected, f"Signed source changed: {name}"
    accepted = run.reviewed_edits()
    items = [m for m in run.material() if m["id"] in accepted]
    predictions = rows(directory / "predictions.jsonl")
    expected = [f'{m["id"]}:rep:{rep}:{c}' for m in items for rep in range(run.C["replicates"])
                for c in ["no_explanation", "explanation"]]
    run.verify_records(predictions, expected)
    assert len(predictions) == len(expected)
    for row in predictions:
        assert digest_text(row["output"]) == row["output_sha256"]
        assert all(row[k] == v for k, v in baseline.score(row["scoring_text"], row["gold"]).items())
        parsed, _ = run.sensitivity.parse(row["scoring_text"])
        assert row["v2_correct"] == (parsed is not None and parsed == baseline.number(row["gold"]))
    reviews = rows(directory / "continuation_reviews.jsonl")
    assert len(reviews) == len(predictions), "All outputs require individual full-text review"
    lookup = {r["id"]: r for r in reviews}
    assert len(lookup) == len(reviews)
    for row in predictions:
        review = lookup[row["id"]]
        assert review["record_sha256"] == row["record_sha256"] and review["reason"]
        assert review["full_text_reviewed"] is True and isinstance(review["math_error"], bool)
    positions = position_results(predictions, run.C["replicates"])
    diagnostics = rows(probe.OUT / "diagnostics.jsonl")
    questions = []
    for diagnostic in diagnostics:
        sid = diagnostic["source_id"]
        choices = diagnostic["choices"]
        available = [v for v in positions.values() if v["source_id"] == sid]
        lengths = [r["student_tokens"] for r in accepted.values() if r["source_id"] == sid]
        selected = {method: positions.get(f'{sid}:step:{choices[method]}') if choices[method] is not None else None
                    for method in ["nll", "random", "combined"]}
        oracle = max([0] + [v["delta"] for v in available])
        questions.append({"source_id": sid, "candidate_steps": len(diagnostic["steps"]),
                          "accepted_explanations": len(available), "choices": choices,
                          "selected": selected, "common_coverage": all(selected.values()),
                          "explanation_tokens_range": [min(lengths), max(lengths)] if lengths else None,
                          "length_spread_le_16": bool(lengths) and max(lengths) - min(lengths) <= 16,
                          "observed_oracle_delta_including_no_supplement": oracle,
                          "regret": {k: oracle - v["delta"] if v else None for k, v in selected.items()}})

    cohorts = {}
    for name, cohort in [("common", [q for q in questions if q["common_coverage"]]),
                         ("common_length_spread_le_16", [q for q in questions if q["common_coverage"] and q["length_spread_le_16"]]),
                         ("common_multiple_candidates", [q for q in questions if q["common_coverage"] and q["candidate_steps"] > 1])]:
        cohorts[name] = {"questions": len(cohort), "source_ids": [q["source_id"] for q in cohort],
                         "mean_selected_delta": {k: statistics.mean(q["selected"][k]["delta"] for q in cohort) if cohort else None
                                                 for k in ["nll", "random", "combined"]},
                         "mean_selected_delta_v2": {k: statistics.mean(q["selected"][k]["delta_v2"] for q in cohort) if cohort else None
                                                    for k in ["nll", "random", "combined"]}}
    # Independently recheck the old experiment remains byte-for-byte unchanged.
    old_audit_path = probe.OLD / "final_integrity_audit.json"
    old_audit = json.loads(old_audit_path.read_text())
    for name, expected_hash in old_audit["files_sha256"].items():
        assert baseline.sha(run.ROOT / name) == expected_hash, f"Old frozen result changed: {name}"
    conditions = {}
    for condition in ["no_explanation", "explanation"]:
        group = [r for r in predictions if r["condition"] == condition]
        conditions[condition] = {"outputs": len(group), "correct_v1": sum(r["correct"] for r in group),
                                 "correct_v2": sum(r["v2_correct"] for r in group),
                                 "invalid": sum(r["invalid"] for r in group),
                                 "length_stops": sum(r["finish_reason"] == "length" for r in group)}
    result = {"status": "engineering_smoke_complete_not_selector_efficacy_validation",
              "questions": questions, "cohorts": cohorts, "positions": positions, "conditions": conditions,
              "diagnostic_candidate_steps": sum(len(r["steps"]) for r in diagnostics),
              "complete_mask_diagnostics": sum(s["status"] == "diagnostic_complete" for r in diagnostics for s in r["steps"]),
              "changed_choice_vs_nll": sum(r["choices"]["combined"] is not None and r["choices"]["combined"] != r["choices"]["nll"] for r in diagnostics),
              "raw_total_is_not_selector_accuracy": True,
              "teacher_edits": len(rows(directory / "edits.jsonl")), "accepted_edits": len(accepted),
              "predictions": len(predictions), "full_text_reviewed": len(reviews),
              "auxiliary_math_errors": sum(r["math_error"] for r in reviews),
              "auxiliary_format_issues": sum(r["format_issue"] for r in reviews),
              "simulated_tool_outputs_not_executed": sum(r["simulated_tool_output"] for r in reviews),
              "new_adapters_trained": 0, "paid_api_cost_yuan": 0,
              "old_pilot100_audit_sha256": baseline.sha(old_audit_path),
              "teacher_generation_seconds": sum(r["seconds"] for r in rows(directory / "edits.jsonl")),
              "student_generation_seconds": sum(r["seconds"] for r in predictions),
              "diagnostic_seconds": sum(r["seconds"] for r in diagnostics),
              "limits": ["Six previously used training-source questions, not independent evaluation",
                         "Two repeated continuations per position and condition; extremely coarse estimates",
                         "No selector weights fitted to outcomes; combined score is an uncalibrated heuristic",
                         "Unequal explanation lengths, weak explanations and rejected new-result disclosure",
                         "Oracle and regret calculated from the same tiny sample are noisy and optimistically selected",
                         "Teacher prefixes can make continuation easy; this does not measure standalone student accuracy",
                         "Deletion/random control semantic and position confounds; no calibrated abstention threshold",
                         "No evidence yet that attention/masking improves independent selection or distillation"]}
    baseline.save(directory / "results.json", result)
    table = ["| 题目 | 候选/合格解释 | NLL选择 | 联合选择 | NLL位置收益 | 联合位置收益 |",
             "|---|---:|---:|---:|---:|---:|"]
    def gain(value):
        return f'{value["delta"]:+.0%}' if value else "不可用"
    for q in questions:
        table.append(f'| {q["source_id"]} | {q["candidate_steps"]}/{q["accepted_explanations"]} | {q["choices"]["nll"]} | {q["choices"]["combined"]} | {gain(q["selected"]["nll"])} | {gain(q["selected"]["combined"])} |')
    document = "\n".join([
        "# attention＋masking选择器：6题端到端工程检查", "",
        "代码、真实学生诊断、教师解释审核和配对续写链路已跑通。本批不是独立位置选择效果验证，不证明比NLL选得更好，也不进入蒸馏训练。", "",
        f'关键观察：全部合格位置的不补充输出v1/v2均为{conditions["no_explanation"]["correct_v1"]}/{conditions["no_explanation"]["outputs"]}；补充后v1为{conditions["explanation"]["correct_v1"]}/{conditions["explanation"]["outputs"]}、v2为{conditions["explanation"]["correct_v2"]}/{conditions["explanation"]["outputs"]}。不补充已接近天花板；补充还出现算术/漏条件错误和格式改变，不能默认解释有益。这些是同题多个教师前缀的重复续写，不能当38道独立题的准确率或选择器排名。', "",
        f'6题共21候选，教师解释{result["teacher_edits"]}条，合格{result["accepted_edits"]}条；保留拒绝原文、不重试、不换题。共{len(predictions)}条续写均逐条全文辅助审核，冻结v1和固定v2重算核验，主分不因审核改动。', "",
        "每个候选两次续写/条件；位置收益＝同一教师前缀下解释正确率−无解释正确率。0%只表示这两次样本相同，不能认定真收益为0。表中步骤编号从0开始。", "",
        *table, "",
        "## 同题共同覆盖比较", "", "```json", json.dumps(cohorts, ensure_ascii=False, indent=2), "```", "",
        "## 全部合格位置的输出核对（不是选择器准确率）", "", "```json", json.dumps(conditions, ensure_ascii=False, indent=2), "```", "",
        "## 局限与下一步", "", *[f'- {v}' for v in result["limits"]], "",
        "候选与线索也需改进：如4543的联合规则从复述总人数450改选复述橙子人数70，两者可能都不需要解释；4868部分attention最高前文是章节标题。当前等号筛选和平均attention不能保证语义上有价值的推理缺口。后续改进需另记版本并在新留出题验证，不能回改本批候选或根据这6题调权重后声称泛化。", "",
        f'全文辅助审核发现{result["auxiliary_math_errors"]}条真实数学/条件错误、{result["auxiliary_format_issues"]}条数学正确但v1格式拒绝，以及{result["simulated_tool_outputs_not_executed"]}条附模拟工具output的生成文本。全部未执行生成代码，未改冻结主分；2417位置0两条解释输出的格式差异须结合v2比较。', "",
        "下一步须先冻结候选/解释质量与长度控制、改进单元匹配遮蔽及不补充规则，再在独立留出题做更多重复的选位验证。位置验证通过后才做固定预算训练对照，最终指标仍为无教师/无提示学生的数字答案准确率。", "",
        "本轮无新训练、付费API费用0元（未计本地电力/人工），正式test未使用；旧pilot100完整性审计覆盖文件全部重新核验通过。详细删除索引、NLL、attention、原始解释、续写和审核hash见同目录JSON。", ""])
    # Ordinary report generation is deterministic and does not alter experiment inputs.
    (directory / "RESULTS_ZH.md").write_text(document)
    receipt = {"status": "passed", "old_pilot100_unchanged": True, "full_text_reviews": len(reviews),
               "files_sha256": {p.name: baseline.sha(p) for p in [directory / "results.json", directory / "RESULTS_ZH.md",
                                  directory / "predictions.jsonl", directory / "continuation_reviews.jsonl",
                                  directory / "edits.jsonl", directory / "edit_reviews.jsonl", Path(__file__)]}}
    baseline.save(directory / "integrity_audit.json", receipt)
    baseline.save(directory / "status.json", {"stage": result["status"], "full_text_reviewed": len(reviews),
                                              "training_started": False, "independent_efficacy_validation_complete": False})
    print(json.dumps({"status": result["status"], "cohorts": cohorts, "conditions": conditions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
