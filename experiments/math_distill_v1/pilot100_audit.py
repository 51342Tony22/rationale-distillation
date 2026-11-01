"""Print complete pending material or evaluation rows; never auto-approve outputs."""
import argparse
import json

import baseline
import pilot_common as common
from pilot100 import DATA, OLD, OUT, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["teacher", "location", "difficulty_only_edit_v4", "difficulty_attention_edit_v4",
                                       "student_base", "ordinary_cot", "difficulty_only", "difficulty_attention"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--context-reviewed", action="store_true", help="For edits only: omit previously reviewed teacher full text; retain chosen step, evidence and complete new generation.")
    args = parser.parse_args()
    kind = args.kind
    materials = kind in ["teacher", "location"] or kind.endswith("_edit_v4")
    if materials:
        split="train"
        generated_file = {"teacher":"teacher_targets.jsonl", "location":"locations.jsonl"}.get(kind,kind.replace("_edit_v4","_edits_v4")+".jsonl")
        path = OUT / generated_file
        audit_path = OUT / (kind+"_reviews.jsonl")
    else:
        split="dev"
        path=OUT / "evaluation" / kind / "predictions.jsonl"
        audit_path=OUT / (kind+"_evaluation_reviews.jsonl")
    audits=common.rows(audit_path)
    if kind == "student_base":
        audits += json.loads((ROOT / "output/math_distill_v1/dev_preview_audit.json").read_text())["records"]
        audits += common.rows(OLD / "student_base_evaluation_reviews.jsonl")
    done={r["source_id"]:r for r in audits}
    assert len(done)==len(audits)
    source={r["source_id"]:r for r in json.loads((DATA / f"{split}.json").read_text())}
    generated=common.rows(path)
    own=common.rows(audit_path)
    print(json.dumps({"generated":len(generated),"reviewed":len(done),"kind":kind,"anchor":own[-1] if own else None},ensure_ascii=False))
    targets={r["source_id"]:r for r in common.rows(OUT / "teacher_targets.jsonl")}
    locations={r["source_id"]:r for r in common.rows(OUT / "locations.jsonl")}
    count=0
    for n,row in enumerate(generated,1):
        sid=row["source_id"]
        if sid in done:
            assert row["output_sha256"]==done[sid]["output_sha256"]
            continue
        result={"n":n,"source_id":sid,"question":source[sid]["question"],"gold":source[sid]["gold"],"output_sha256":row["output_sha256"]}
        if kind=="location":
            result.update({"teacher_output":targets[sid]["output"],"chosen":row["chosen"],"evidence":row["evidence"],
                           "max_logits_difference":row["max_logits_difference"],"future_attention_mass":row["future_attention_mass"],
                           "attention_sum_error":row["attention_sum_error"],"input_tokens":row["input_tokens"]})
        elif kind.endswith("_edit_v4"):
            result.update({"teacher_output":targets[sid]["output"],"chosen":locations[sid]["chosen"],"evidence":locations[sid]["evidence"],
                           "raw_output":row["output"],"explanation":row["explanation"],"student_tokens":row["student_tokens"],"stop":row["finish_reason"]})
            if args.context_reviewed:
                teacher_audits={r["source_id"]:r for r in common.rows(OUT / "teacher_reviews.jsonl")}
                assert teacher_audits[sid]["accepted"] and teacher_audits[sid]["output_sha256"]==targets[sid]["output_sha256"]
                result.pop("teacher_output")
                result["teacher_reviewed_sha256"]=targets[sid]["output_sha256"]
        else:
            result.update({"output":row["output"],"v1":row["correct"],"stop":row["finish_reason"]})
            if not materials:
                import score_sensitivity_v2 as sensitive
                value,_=sensitive.parse(row["scoring_text"])
                result["v2"]=value is not None and value==baseline.number(row["gold"])
        print(json.dumps(result,ensure_ascii=False))
        count+=1
        if count>=args.limit:
            break


if __name__=="__main__":
    main()
