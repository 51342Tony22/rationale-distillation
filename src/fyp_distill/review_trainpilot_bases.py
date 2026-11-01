"""Materialize manually audited drafts; never freeze incomplete selection as bases."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from .common import read_jsonl, resolve_path, write_jsonl
from .refinement import digest, stable_seed, unique_index


ANCHORS = {
    "math_train_algebra_2425.json": [
        "\\[8x^2 - 65x + 8 = 0\\]  \n", "and \\(c = 8\\).  \n", "= 4225 - 256 = 3969\\]  \n"],
    "math_train_algebra_325.json": [
        "for some constant \\(b\\).\n\n", "= 9x^2 + 6b x + b^2.\n\\]\n\n", "\\(6b = 24\\).\n"],
    "math_train_algebra_1961.json": [
        "\\(c = -8\\).\n\n", "= 49 + 96 = 145 \\]\n\n", "\\[ \\frac{\\sqrt{145}}{3} \\]\n\n"],
    "math_train_counting_and_probability_360.json": [
        "and its respective winnings.\n\n", "\\frac{1}{2} \\times 1 = \\frac{1}{2}\n\\]\n\n", "\\frac{1}{3} \\times 3 = 1\n\\]\n\n"],
    "math_train_counting_and_probability_393.json": [
        "4 red, 5 white, and 6 blue.\n\n", "= \\frac{2730}{6} = 455.\n\\]\n\n", "all red, all white, or all blue.\n\n"],
    "math_train_counting_and_probability_911.json": [
        "the remaining boxes empty and not distinguished).\n\n",
        "for \\(k = 1, 2, 3\\).\n\n",
        "which corresponds to all balls in one box.\n"],
    "math_train_geometry_98.json": [
        "which is the side length of the equilateral triangle.  \n",
        "\\[ (x - 8)^2 + (y - 5)^2 = 64 \\]  \n",
        "\\[ x = 4 \\]  \n"],
    "math_train_geometry_85.json": [
        "\\(s = \\frac{18}{2} = 9\\).  \n\n",
        "\\(A = \\sqrt{s(s - a)(s - b)(s - c)}\\).  \n",
        "= 6\\sqrt{6}\\).  \n\n"],
    "math_train_intermediate_algebra_1229.json": [
        "so \\(a_n = \\log_{2002} n\\).\n\n",
        "\\log_{2002} 13 + \\log_{2002} 14.\\]\n\n",
        "\\log_{2002} 240240.\\]\n\n"],
    "math_train_intermediate_algebra_1971.json": [
        "\\frac{n}{x} = \\frac{9}{10}\n\\]\n",
        "x = \\frac{10n}{9}\n\\]\n",
        "n \\leq \\frac{10n}{9} < n+1\n\\]\n"],
    "math_train_number_theory_1030.json": [
        "\\(A, B \\in \\{0, 1, 2, 3, 4, 5\\}\\).\n\n",
        "Thus, \\(A = (B + 1) \\mod 6\\).\n\n",
        "\\(C_2 = \\lfloor (A + 4 + C_1) / 6 \\rfloor\\).\n\n"],
    "math_train_number_theory_715.json": [
        "\\( 4 \\times 8^2 + 2 \\times 8^1 + 7 \\times 8^0 \\)  \n",
        "Thus, \\( 427_8 = 279_{10} \\).\n\n",
        "So, start with \\( 5^3 = 125 \\).\n\n"],
    "math_train_number_theory_336.json": [
        "\\(2 + 41 + 61 = 104\\).\n\n",
        "sum to \\(104 - 2 = 102\\).\n\n",
        "compute \\(q = 102 - p\\) and check if \\(q\\) is prime:\n"],
    "math_train_prealgebra_236.json": [
        "There are 8 possible choices for the president.  \n\n",
        "the committee is chosen from these 7 people.  \n\n",
        "This is given by \\(\\binom{7}{2}\\).  \n\n"],
    "math_train_prealgebra_392.json": [
        "opposite to A and adjacent to B.\n\n",
        "the internal angle of the hexagon at B.  \n",
        "Thus, angle ABC is \\(120^\\circ\\).\n\n"],
    "math_train_prealgebra_1164.json": [
        "total capacity of the tank in gallons.\n\n",
        "\\[ \\frac{1}{6}x + 130 = \\frac{3}{5}x \\]  \n\n",
        "\\[ 130 = \\frac{3}{5}x - \\frac{1}{6}x \\]  \n\n"],
    "math_train_precalculus_359.json": [
        "8 \\sin x \\cos x (\\cos^4 x - \\sin^4 x) = 1\n\\]\n\n",
        "8 \\sin x \\cos x \\cdot \\cos 2x = 1\n\\]\n\n",
        "4 \\sin 2x \\cos 2x = 1\n\\]\n\n"],
}

NOTES = {
    "math_train_algebra_2425.json":"分母(8x−1)(x−8)，两根8和1/8；3边界在根之前。",
    "math_train_algebra_325.json":"展开(3x+b)^2的交叉项6bx，6b=24得b4、a16；在b值之前固定边界。",
    "math_train_algebra_1961.json":"D49+96=145=5×29平方自由，根差sqrt145/3，m+n148；第三前缀给根差未写最终和，仍有简单加法天花板风险。",
    "math_train_counting_and_probability_360.json":"E=1/2+1−5/6=2/3，按题要求四舍五入到分为0.67。原gold混合精确/近似格式，规范为0.67，原gold保留。",
    "math_train_counting_and_probability_393.json":"总C15,3=455，同色C4,3+C5,3+C6,3=34，概率34/455；三个前缀均未给有利总数。",
    "math_train_counting_and_probability_911.json":"可区分5球放不可区分3箱，允许空箱，因此数集合分拆为1/2/3个非空块，S51=1，S52=(2^5−2)/2=15，S53=(3^5−3×2^5+3)/6=25，合计41。teacher各步与空箱解释正确；三个前缀在S52/S53数值和总数前，未泄露最终答案。",
    "math_train_geometry_98.json":"等边边8、两距离平方式相减得x4，y=5±4sqrt3；负支y<0不在第一象限，正支唯一。Asymptote与文字给相同两点，不靠实际看图推测；三边界在y两支之前。",
    "math_train_geometry_85.json":"三角形5,6,7满足三角不等式，半周长9，Heron面积sqrt216=6sqrt6，内切半径面积/半周长=2sqrt6/3；三前缀未写内切半径。",
    "math_train_intermediate_algebra_1229.json":"a_n=log2002(n)，b=log2002(120)，c=log2002(240240)，后者=log2002(120)+1；三个边界在比值1/2002与−1之前。",
    "math_train_intermediate_algebra_1971.json":"原teacher两个正性推断理由不足；新人工修正候选显式排除负x比值>=1，结合方程得n>=1。随后x10n/9及n<=x<n+1得n<9，最大n8、x80/9代回正确；原错误候选完整保留，修正来源单列，边界在n<9和最终值前。",
    "math_train_number_theory_1030.json":"枚举0<=A,B<6满足4AB6+416=53A6唯一A5B4；carry分支完整正确，三个前缀尚未给A/B值。",
    "math_train_number_theory_715.json":"八进制4×64+2×8+7=279；五进制最高幂125，各位整除商2、1、0、4，余数29、4、4、0，逆算2×125+25+4=279。文中division的整数商按带余除法理解；三完整步骤前缀均在五进制各位计算之前，不给最终数字。",
    "math_train_number_theory_336.json":"5002=2×41×61，和104；三不同素数的偶数和必须含2，另两者和102。独立试除枚举p<51验证8对，排除41/61剩7、各乘积正确且唯一分解保不同；两个think段均核查，草稿保留后段、原响应不改；三边界在枚举结果、对数和总数前。gold7去计数单位文字并保留原gold，首次screen前完成。",
    "math_train_prealgebra_236.json":"先总统8选一，再7选2=21，积168；三个边界在组合数21与最终乘积前。",
    "math_train_prealgebra_392.json":"Asymptote的A=10dir240、B=10dir300、C=10dir0，BA与BC为六边形相邻边；AB水平长10且BD竖直下10，ABD为正方形内角90；ABC120、总210。付费原响应有孤立</think>及前后重复正确解答，训练草稿仅保留标记后的完整解答，原响应不变；三边界均在方形角90之前，不给最终和。",
    "math_train_prealgebra_1164.json":"容量x，x/6+130=3x/5，差3/5−1/6=13/30，故x=130×30/13=300；逐步计算与单位正确，三边界在通分、求解容量之前。",
    "math_train_precalculus_359.json":"提公因式及平方差恒等式得8sinxcosxcos2x=1，两次倍角得2sin4x=1；两族解7.5+90k与37.5+90k，k<=−1均负，最小正解7.5，代回满足。付费响应有孤立</think>与提前完整答案，草稿只保留标记后完整推导；三边界在sin4x与角度求解前，原响应保留。",
}


def main():
    root=resolve_path("data/refinement_math_trainpilot21_v3")
    selected=unique_index(read_jsonl(root/"selected_questions.jsonl"))
    teacher=unique_index(read_jsonl(root/"teacher_candidates.jsonl"))
    corrected=unique_index(read_jsonl(root/"manual_corrected_candidates.jsonl"))
    prior=unique_index(read_jsonl(root/"draft_bases_reviewed.jsonl"))
    for source_id,anchors in ANCHORS.items():
        if source_id not in teacher:
            continue
        row=teacher[source_id]
        candidate=corrected.get(source_id,row)["candidates"][0]
        text=candidate["text"]
        normalization = None
        if source_id in {"math_train_prealgebra_392.json", "math_train_precalculus_359.json", "math_train_number_theory_336.json"}:
            if text.count("</think>") != 1:
                raise ValueError("Manually audited response shape changed")
            normalization = {"reason": "retain complete post-think solution; remove duplicated reasoning and orphan delimiter",
                             "raw_candidate_text_hash": digest(text)}
            text = text.split("</think>", 1)[1].strip()
        bounds=sorted(text.index(a)+len(a) for a in anchors)
        source=selected[source_id]
        canonical_gold={"math_train_counting_and_probability_360.json":"0.67",
                        "math_train_number_theory_336.json":"7"}
        gold=canonical_gold.get(source_id,source["gold_answer"])
        base={"id":source_id,"split":"train","dataset":"math","question":row["question"],
              "gold_answer":gold,"teacher_model":row["teacher_model"],"candidate_id":candidate["candidate_id"],
              "original":text,"boundaries":bounds,"random_boundary":bounds[stable_seed(7,source_id,"trainpilot_random")%3],
              "review":{"correct_steps":True,"prefixes_do_not_reveal_answer":True,
                        "manual_math_review":NOTES[source_id]}}
        if normalization is not None:
            base["original_text_normalization"] = normalization
        if source_id in corrected:
            base["manual_math_correction"] = candidate["provenance"]
            base["derived_from_candidate_id"] = candidate["derived_from_candidate_id"]
        if gold!=source["gold_answer"]:
            base["original_gold_answer"]=source["gold_answer"]
            base["gold_canonicalization_reason"]=("problem explicitly requests dollars rounded to nearest cent; frozen before student screening"
                                                  if source_id=="math_train_counting_and_probability_360.json" else
                                                  "exact integer count; remove unit prose from source gold after independent prime enumeration, before student screening")
        base["source_hash"]=digest({k:v for k,v in base.items() if k!="review"})
        if source_id in prior and prior[source_id]!=base:
            raise ValueError("Prior audited draft differs; retain it and explicitly revise before screening")
        prior[source_id]=base
    write_jsonl(root/"draft_bases_reviewed.jsonl",[prior[r["id"]] for r in selected.values() if r["id"] in prior])
    write_jsonl(root/"original_review_progress.jsonl",[{"reviewed":len(prior),"selected":len(selected),
                                                         "updated_at":datetime.now(UTC).isoformat(),
                                                         "draft_only":True,"screen_started":False}])
    print(json.dumps({"reviewed_drafts":len(prior),"unreviewed":len(selected)-len(prior)}))


if __name__=="__main__":
    main()
