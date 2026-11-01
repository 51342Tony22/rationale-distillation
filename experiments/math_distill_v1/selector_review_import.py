"""Bind explicit, individually authored full-text notes to immutable output hashes.

Never infers review completion/correctness from scores or creates missing notes.
"""
import json

import baseline
from pilot_common import append, rows
from selector_continuation import OUT, verify_records


NOTES = {
    "time20": (False, False, False, "全文审核：30×25=750，450+750=1200分钟，除60得20小时；当前续写及最终答案正确，无截断或任务漂移。"),
    "equation8": (False, False, False, "全文审核：由2x−5=11，两边加5得16、除2得x=8；变形、运算和最终人数正确，无截断，重复解释无数学矛盾。"),
    "equation8_code": (False, False, True, "全文审核：等式求解并代入检验正确，最终8；附Python及模拟output8.0均为生成文本，未执行工具，不算真实验证。"),
    "biscuits40": (False, False, False, "全文审核：32+13+15=60，再减20得40，算术与最终答案正确，无截断。"),
    "biscuits_arithmetic_error": (True, False, False, "全文审核：错误写32+13+15=50，继而50−20=30，最终30错误；应60−20=40，属于真实算术错误，主分不改。"),
    "fruit113": (False, False, False, "全文审核：70+120+147=337，450−337=113；分类与最终草莓人数正确，无截断或矛盾答案。"),
    "fruit113_sequential": (False, False, False, "全文审核：顺次450−70=380，380−120=260，260−147=113，最终正确，无截断。"),
    "whale178": (False, False, False, "全文审核：首趟28+56=84，第二趟幼鲸及双亲24，第三趟14+56=70，合178；沿用源题每只幼鲸随双亲的常规假设，无截断。"),
    "whale_omitted_parents": (True, False, False, "全文审核：第二趟只计8幼鲸，漏16双亲，最终84+8+70=162错；题意下应178，真实漏条件。"),
    "whale178_code_wording": (False, False, True, "全文审核：第二趟先写16 whales from baby whales措辞欠清，但随后明确8幼鲸+16父母=24，总178正确；Python/模拟output为生成文本未执行。"),
    "whale_one_parent_code": (True, False, True, "全文审核：错误将每幼鲸配一名家长，第二趟16，总170而非178；Python/模拟output重复错误且未执行，主分保留。"),
    "whale_one_parent": (True, False, False, "全文审核：自设每幼鲸一名家长，第二趟16，总170错误；与题面parents及源解双亲解释不符，非评分格式问题。"),
    "whale_prefix_contradiction": (True, False, False, "全文审核：教师前缀已明确8幼鲸+16家长，续写仍称第二趟8，总162；漏家长且与前缀矛盾，主分不改。"),
    "whale178_code": (False, False, True, "全文审核：84+24+70=178及代码按幼鲸/双亲三组求和均正确；代码和output178为生成文本，未实际运行。"),
    "water4_code": (False, False, True, "全文审核：64÷8=8份、64÷16=4份、减少4份，最终boxed4正确；代码及模拟output只作生成内容，未执行。"),
    "water4_format": (False, True, False, "全文审核：8份变4份，少4份的数学与最终自然语言结论正确；无boxed且末句数字不在尾端，冻结v1格式拒绝、v2恢复正确，不改主分。"),
    "water4_code_only": (False, False, False, "全文审核：8份变4份、差4，boxed4正确；随后给Python代码但无实际工具运行或output，代码块正常闭合，无截断。")
}


def main():
    path = OUT / "predictions.jsonl"
    notes = json.loads((OUT / "continuation_review_notes.json").read_text())
    assert baseline.sha(path) == notes["predictions_sha256"]
    generated = rows(path)
    assert len(generated) == len(notes["rows"]) == 76
    verify_records(generated, [r["id"] for r in generated])
    completed = []
    for index, (ordinal, sid, step, rep, condition, label) in enumerate(notes["rows"]):
        row = generated[index]
        assert ordinal == index + 1
        assert row["id"] == f"gsm8k:train:{sid}:step:{step}:rep:{rep}:{condition}"
        math_error, format_issue, simulated, reason = NOTES[label]
        completed.append({"id": row["id"], "record_sha256": row["record_sha256"],
                          "full_text_reviewed": True, "math_error": math_error, "format_issue": format_issue,
                          "simulated_tool_output": simulated, "reason": reason,
                          "review_notes_sha256": baseline.sha(OUT / "continuation_review_notes.json")})
    target = OUT / "continuation_reviews.jsonl"
    old = rows(target)
    assert old == completed[:len(old)], "Existing reviews differ; do not overwrite"
    for row in completed[len(old):]:
        append(target, row)
    print(f"Bound {len(completed)} explicit full-text reviews; no model scores changed.")


if __name__ == "__main__":
    main()
