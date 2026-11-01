"""Read existing predictions to report the identical 200-question teacher comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import baseline
import score_sensitivity_v2 as sensitive

ROOT = baseline.ROOT
OUT = ROOT / "output/math_distill_v1/teacher_reference_common200"
LABELS = {
    "teacher_reference": "教师7B Instruct（4bit）",
    "student_base": "未训练学生1.5B Base",
    "ordinary_cot": "普通CoT蒸馏学生",
    "difficulty_only": "NLL定位补充学生",
    "difficulty_attention": "NLL定位＋attention前文线索学生",
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    sensitive.selfcheck()
    data_path = ROOT / "data/math_distill_v1/pilot100/dev.json"
    items = json.loads(data_path.read_text())[:200]
    assert len(items) == len({r["source_id"] for r in items}) == 200
    teacher_dir = ROOT / "output/math_distill_v1/dev_baseline/teacher_reference"
    summary = json.loads((teacher_dir / "summary.json").read_text())
    signature = json.loads((teacher_dir / "signature.json").read_text())
    assert signature["source_ids"] == [r["source_id"] for r in items]
    assert baseline.sha(teacher_dir / "signature.json") == summary["signature_sha256"]
    report = {
        "scope": "Same ordered first 200 development questions; existing outputs only",
        "new_generations": 0, "primary_scores_changed": False,
        "data_sha256": baseline.sha(data_path),
        "script_sha256": baseline.sha(Path(__file__)),
        "v1_parser_sha256": baseline.sha(Path(baseline.__file__)),
        "v2_parser_sha256": baseline.sha(Path(sensitive.__file__)),
        "teacher_full_math_review": summary["full_math_review"],
        "conditions": {}, "rows": {},
    }
    for name in LABELS:
        path = (teacher_dir if name == "teacher_reference" else
                ROOT / "output/math_distill_v1/pilot100/evaluation" / name) / "predictions.jsonl"
        predictions = rows(path)
        assert len(predictions) == (200 if name == "teacher_reference" else 250)
        if name == "teacher_reference":
            assert baseline.sha(path) == summary["predictions_sha256"]
        results = []
        for row, item in zip(predictions[:200], items):
            assert all(row[k] == item[k] for k in ("source_id", "question_sha256", "gold"))
            assert hashlib.sha256(row["output"].encode()).hexdigest() == row["output_sha256"]
            assert all(row[k] == v for k, v in baseline.score(row["scoring_text"], item["gold"]).items())
            value, rule = sensitive.parse(row["scoring_text"])
            results.append({"source_id": row["source_id"], "output_sha256": row["output_sha256"],
                            "v1_correct": row["correct"], "v1_invalid": row["invalid"],
                            "v2_correct": value is not None and value == baseline.number(item["gold"]),
                            "v2_invalid": value is None, "v2_parse_rule": rule})
        report["conditions"][name] = {
            "n": 200, "available_predictions": len(predictions),
            "source_predictions_sha256": baseline.sha(path),
            **{key: sum(r[key] for r in results) for key in
               ("v1_correct", "v2_correct", "v1_invalid", "v2_invalid")},
        }
        report["rows"][name] = results
    assert report["conditions"]["teacher_reference"]["v1_correct"] == summary["correct"]
    lines = ["# 教师与学生：同200道开发题补充比较", "",
             "复用既有输出，逐条核对有序ID、题干hash、gold、输出hash并重新核验固定评分；无新生成、无重训，不修改旧pilot100结果。", "",
             "| 条件 | v1主数字EM | 固定v2敏感性EM | v1/v2无效输出 |",
             "|---|---:|---:|---:|"]
    for name, label in LABELS.items():
        r = report["conditions"][name]
        lines.append(f"| {label} | {r['v1_correct']}/200（{r['v1_correct']/2:.1f}%） | "
                     f"{r['v2_correct']}/200（{r['v2_correct']/2:.1f}%） | {r['v1_invalid']}/{r['v2_invalid']} |")
    lines += ["", "教师仅有原200题，学生各有250题；本表严格使用共同200题。完整250题学生结果仍见旧pilot100报告，不能拿教师183/200与学生224/250直接作配对结论。", "",
              "v1为冻结主数字评分，v2为查看最初20道开发输出后制定的固定解析敏感性分析；两者用同一输出，不是模型能力变化。数字EM不保证推理正确，单位等价/源gold及措辞问题均不修改原分。学生250条输出已有全文辅助审核，教师200条完整数学推理审核仍待完成。", "",
              "教师是能力参照，不是学生不可超过的理论上限；它也用于生成训练目标。不同Base/Instruct模型采用各自固定的合适提示格式；题目、gold、解析器及greedy解码口径一致。", "",
              "本表中的attention只在NLL已选位置提供前文线索，未使用新attention＋masking选择器。新选择器仅完成6题工程检查，尚无独立选位优势或训练提升结论。", "",
              "本批为单seed、已查看的开发探索；教师分高不能证明所提方法有效。方法的额外收益须看同题同预算下与普通CoT及NLL定位组的差异。hard-answer、soft-distribution、官方test和多seed仍未完成。", ""]
    baseline.save(OUT / "results.json", report)
    (OUT / "RESULTS_ZH.md").write_text("\n".join(lines))
    print(json.dumps(report["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
