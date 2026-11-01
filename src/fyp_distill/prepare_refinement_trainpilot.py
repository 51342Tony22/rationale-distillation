"""Freeze the small seven-subject train/evaluation pilot before new outputs."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import digest, question_key, stable_seed, unique_index, write_frozen


def main():
    root = "data/refinement_math_trainpilot21_v3"
    config_path = resolve_path("configs/refinement_math_trainpilot21_v3.yaml")
    config = load_config("configs/refinement_math_calibration7_v3.yaml")
    config["refinement"].update(root=root, question_limit=21,
                              request_namespace="trainpilot21_v3")
    config["data"].update(teacher_file=root+"/teacher_candidates.jsonl",
                          variants_dir=root+"/variants", dev_file=root+"/eval_questions.jsonl")
    config["mlx_training"].update(num_train_epochs=10, logging_steps=8,
                                 save_steps=208, budget_mode="question_exposure_pilot")
    config["evaluation"].update(stochastic_samples=0, behavioural_limit=0)
    for key, value in config["artifacts"].items():
        config["artifacts"][key] = value.replace("calibration7", "trainpilot21")
    config["pilot"] = {"train_target": 21, "evaluation_target": 70,
                       "training_seeds": [7], "no_test_tuning": True,
                       "deduplicate_identical_supervision": True,
                       "old_results": "quick2 retained; previous originals reused, new diagnostics",
                       "stop_after_complete_report": True}
    if config_path.exists():
        if load_config(config_path) != config:
            raise ValueError("Existing frozen pilot config differs")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n")
    train = read_jsonl(config["data"]["train_file"])
    dev = read_jsonl("data/refinement_math/processed/math_dev.jsonl")
    previous = read_jsonl("data/refinement_math_calibration7_v3/selected_questions.jsonl")
    selected = list(previous)
    subjects = sorted({r["subject"] for r in train})
    assert len(subjects) == 7 and len(previous) == 7
    for subject in subjects:
        for level in (3,4,5):
            if any(r["subject"]==subject and r["level"]==level for r in selected):
                continue
            eligible = [r for r in train if r["subject"]==subject and r["level"]==level]
            selected.append(min(eligible,key=lambda r:stable_seed(7,"trainpilot21",r["id"])))
    selected.sort(key=lambda r:(r["subject"],r["level"],r["id"]))
    assert len(selected) == 21
    # Fixed 10 per subject: two each at levels 1 and 2, three each at 3 and 4.
    # Sparse level-1 subjects use one level-1 and three level-2 instead.
    evaluation = []
    for subject in subjects:
        counts = {1:2,2:2,3:3,4:3}
        count1 = sum(r["subject"]==subject and r["level"]==1 for r in dev)
        if count1 < 2:
            counts[1],counts[2] = count1,4-count1
        for level, count in counts.items():
            eligible = sorted([r for r in dev if r["subject"]==subject and r["level"]==level],
                              key=lambda r:stable_seed(7,"trainpilot_eval",r["id"]))
            if len(eligible) < count:
                raise ValueError("Insufficient frozen eval stratum")
            evaluation.extend(eligible[:count])
    assert len(evaluation)==70 and not {question_key(r["question"]) for r in selected}&{question_key(r["question"]) for r in evaluation}
    write_frozen(resolve_path(root+"/selected_questions.jsonl"),selected)
    write_frozen(resolve_path(root+"/eval_questions.jsonl"),evaluation)
    ids={r["id"] for r in selected}
    originals=[r for r in read_jsonl("data/refinement_math_calibration7_v3/teacher_candidates.jsonl") if r["id"] in ids]
    originals_path=resolve_path(config["data"]["teacher_file"])
    if not originals_path.exists():
        write_frozen(originals_path,originals)
    ledger=unique_index(read_jsonl(config["refinement"]["budget_ledger"]))
    old_requests=unique_index(read_jsonl("data/refinement_math_calibration7_v3/original_requests.jsonl"))
    known={r["id"] for r in originals}
    requests=[]
    for row in selected:
        if row["id"] in known:
            continue
        old=old_requests.get(row["id"]+":original:0")
        if old:
            entry=ledger[old["id"]]
            if entry.get("response_content") or entry["status"]!="failed_or_uncertain":
                raise ValueError("Recover saved old response; never issue a duplicate original")
            request={**old,"id":old["id"]+":repair:trainpilot21:1",
                     "replaces_request_id":old["id"],"repair_reason":entry.get("error_type")}
        else:
            prompt=("Solve this mathematics problem accurately. Give a concise, self-contained public "
                    "solution. Put each meaningful reasoning step on a separate line, keeping an equation "
                    "together with its introductory sentence. Use at most roughly 1800 output tokens. "
                    "Do not include a boxed final answer earlier in the derivation. "
                    "End with one line `Final answer: <answer>`.\n\nProblem:\n"+row["question"])
            request={"id":"trainpilot21_v3:"+row["id"]+":original:0","source_id":row["id"],
                     "source_hash":digest(row),"kind":"original","dataset":"math",
                     "question":row["question"],"gold_answer":row["gold_answer"],
                     "model":"deepseek-ai/DeepSeek-R1","prompt":prompt,
                     "request_hash":digest(prompt),"max_output_tokens":2048}
        request["answer_protocol"]="first_answer_v3"
        requests.append(request)
    write_frozen(resolve_path(root+"/original_requests.jsonl"),requests)
    manifest={"frozen_at":datetime.now(UTC).isoformat(),"train": [{k:r[k] for k in ("id","subject","level")} for r in selected],
              "eval":[{k:r[k] for k in ("id","subject","level")} for r in evaluation],
              "selection":"retain previous preregistered seven; fill subject levels3/4/5 with fixed hash before new outputs",
              "eval_selection":"70 frozen dev questions; 10 per subject, levels1-4; no test tuning",
              "teacher_originals_reused":len(originals),"new_original_requests":len(requests),
              "explicit_prior_failed_repairs":sum("replaces_request_id" in r for r in requests),
              "training":"six groups, one seed7, 208 microsteps/26 optimizer updates at21 valid rows; same exposure, not token-matched"}
    path=resolve_path(root+"/selection_manifest.jsonl")
    if not path.exists():
        write_frozen(path,[manifest])
    if not resolve_path(root+"/run_status.jsonl").exists():
        write_jsonl(root+"/run_status.jsonl",[{"state":"prepared","stage":"frozen_selection",
                                              "started_at":manifest["frozen_at"],"training_started":False}])
    print(json.dumps({"config":str(config_path),"train":len(selected),"eval":len(evaluation),
                      "new_original_requests":len(requests),"originals_reused":len(originals)}))


if __name__ == "__main__":
    main()
