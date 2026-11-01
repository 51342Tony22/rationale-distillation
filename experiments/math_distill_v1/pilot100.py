"""Expand frozen material prefixes; reuse signed pilot50 implementations intact."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import baseline
import pilot_common as common

HERE = Path(__file__).absolute().parent
ROOT = baseline.ROOT
CONFIG = HERE / "pilot100_config.json"
C = json.loads(CONFIG.read_text())
DATA = ROOT / C["data_dir"]
OUT = ROOT / C["output_dir"]
OLD = ROOT / "output/math_distill_v1/pilot50"
PYTHON = ROOT.parent.parent / "FYP Mini Agent/.venv-mlx/bin/python"
REUSE = ["teacher_targets.jsonl", "teacher_reviews.jsonl", "locations.jsonl", "location_reviews.jsonl",
         "difficulty_only_edits_v4.jsonl", "difficulty_attention_edits_v4.jsonl",
         "difficulty_only_edit_v4_reviews.jsonl", "difficulty_attention_edit_v4_reviews.jsonl"]


def freeze_bytes(path, content):
    if path.exists():
        assert path.read_bytes() == content, f"Existing frozen prefix differs: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def prepare():
    parent = ROOT / "data/math_distill_v1"
    provenance = json.loads((parent / "source_manifest.json").read_text())
    for split in ["train", "dev"]:
        assert baseline.sha(parent / f"{split}.jsonl") == provenance["splits"][split]["sha256"]
    source_path = parent / "source/train.jsonl"
    assert baseline.sha(source_path) == provenance["sources"]["train"]["sha256"]
    originals = common.rows(source_path)
    order = list(range(len(originals)))
    random.Random(C["seed"]).shuffle(order)
    train = common.rows(parent / "train.jsonl")[:100]
    dev = common.rows(parent / "dev.jsonl")
    assert [r["source_index"] for r in dev] == order[:200]
    assert [r["source_index"] for r in common.rows(parent / "train.jsonl")] == order[200:700]
    for index in order[700:750]:
        row = originals[index]
        rationale, final = row["answer"].split("####")
        dev.append({"source_id": f"gsm8k:train:{index}", "source_index": index,
                    "question": row["question"], "source_answer": row["answer"],
                    "reference_rationale": re.sub(r"<<[^>]*>>", "", rationale).strip(),
                    "gold": final.strip(), "question_sha256": common.digest_text(row["question"])})
    assert len(train) == 100 and len(dev) == 250
    assert len({r["source_id"] for r in dev}) == 250
    candidates = common.rows(parent / "train.jsonl")
    assert not {r["source_id"] for r in candidates} & {r["source_id"] for r in dev}
    normal = lambda q: " ".join(re.findall(r"[a-z0-9]+", q.lower()))
    tests = common.rows(parent / "source/test.jsonl")
    train_q = {normal(r["question"]) for r in candidates}
    dev_q = {normal(r["question"]) for r in dev}
    test_q = {normal(r["question"]) for r in tests}
    overlaps = {"all500_candidates_dev250": len(train_q & dev_q), "dev250_official_test": len(dev_q & test_q)}
    assert not any(overlaps.values()), f"Exact source overlap; preserve and inspect: {overlaps}"
    for name, values in [("train", train), ("dev", dev)]:
        baseline.freeze(DATA / f"{name}.json", values)
    manifest = {"scope": "100 training candidates, 250 development questions; no deadline",
                "selection": C["eval"], "train_selection": C["selection"], "seed": C["seed"],
                "source_sha256": baseline.sha(source_path), "config_sha256": baseline.sha(CONFIG),
                "baseline_predictions_sha256": baseline.sha(ROOT / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl"),
                "sensitivity_parser_sha256": baseline.sha(HERE / "score_sensitivity_v2.py"),
                "normalized_exact_overlap": overlaps, "splits": {}}
    for name, values in [("train", train), ("dev", dev)]:
        manifest["splits"][name] = {"count": len(values), "source_ids": [r["source_id"] for r in values],
                                   "frozen_rows_sha256": baseline.sha(DATA / f"{name}.json")}
    baseline.freeze(DATA / "manifest.json", manifest)
    reuse = {}
    for name in REUSE:
        payload = (OLD / name).read_bytes()
        destination = OUT / name
        if destination.exists():
            assert destination.read_bytes().startswith(payload)
        else:
            freeze_bytes(destination, payload)
        reuse[name] = {"source": str((OLD / name).relative_to(ROOT)), "bytes": len(payload),
                       "sha256": baseline.sha(OLD / name), "records": len(common.rows(OLD / name))}
    baseline.freeze(OUT / "reused_material_prefixes.json", reuse)
    original_base = ROOT / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl"
    destination = OUT / "evaluation/student_base/predictions.jsonl"
    if destination.exists():
        assert destination.read_bytes().startswith(original_base.read_bytes())
    else:
        freeze_bytes(destination, original_base.read_bytes())
    baseline.freeze(OUT / "initialization.json", {"actual_new_training": "pending",
                    "old_adapters_used_for_initialization": False, "baseline_predictions_reused": 200,
                    "old_model_evaluation_outputs_reused_for_new_adapters": False,
                    "pilot50_integrity_audit_sha256": baseline.sha(OLD / "final_integrity_audit.json")})
    # Question-only lexical screening of the 50 added dev items; never test generation.
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sections = [dev[200:], candidates, tests]
    all_q = [normal(r["question"]) for section in sections for r in section]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(all_q)
    screen = {}
    for name, lo, hi in [("all500_candidates", 50, 550), ("official_test_question_only", 550, len(all_q))]:
        sim = cosine_similarity(matrix[:50], matrix[lo:hi])
        matches = [{"added_dev_id": dev[200+int(i)]["source_id"], "other_index": int(j), "cosine": float(sim[i,j])}
                   for i,j in zip(*np.where(sim >= .85))]
        screen[name] = {"maximum_cosine": float(sim.max()), "pairs_ge_0_85": matches}
    baseline.freeze(OUT / "added_dev_lexical_screen.json", {"comparisons": screen,
                    "limitation": "Lexical screening cannot rule out semantic or pretraining contamination; fixed IDs not replaced."})
    print(json.dumps({"prepared": True, "train_candidates": 100, "development": 250,
                      "reused_materials": {k:v["records"] for k,v in reuse.items()}, "lexical": screen}, ensure_ascii=False))


def bind():
    common.C, common.CONFIG, common.OUT, common.DATA = C, CONFIG, OUT, DATA
    original_sign = common.sign
    def sign(stage, files, extra=None):
        original_sign(stage, [*files, Path(__file__), OUT / "reused_material_prefixes.json"], extra)
    common.sign = sign
    reuse = json.loads((OUT / "reused_material_prefixes.json").read_text())
    for name, spec in reuse.items():
        source = ROOT / spec["source"]
        assert baseline.sha(source) == spec["sha256"]
        assert (OUT / name).read_bytes().startswith(source.read_bytes())


def wait_reviews(file, review_file):
    while True:
        generated = common.rows(OUT / file)
        audits = common.rows(OUT / review_file)
        lookup = {r["source_id"]: r for r in audits}
        assert len(lookup) == len(audits)
        for r in generated:
            if r["source_id"] in lookup:
                a = lookup[r["source_id"]]
                assert a["output_sha256"] == r["output_sha256"] and isinstance(a["accepted"], bool) and a.get("reason")
        if all(r["source_id"] in lookup for r in generated):
            return
        baseline.save(OUT / "pipeline_status.json", {"stage": "awaiting_full_text_material_audit", "file": file,
                     "generated": len(generated), "reviewed": len(audits), "pid": os.getpid()})
        time.sleep(3)


def evaluate_base():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import score_sensitivity_v2 as sensitive
    items = common.sources("dev")
    directory = OUT / "evaluation/student_base"
    path = directory / "predictions.jsonl"
    with common.gpu("evaluate_base_added50"):
        base_sig_path = ROOT / "output/math_distill_v1/dev_baseline/student_base/signature.json"
        signed = json.loads(base_sig_path.read_text())
        assert baseline.sha(Path(baseline.__file__)) == signed["runner_sha256"]
        for name, spec in signed["model_files"].items():
            assert baseline.sha(ROOT / common.BASE["student_dir"] / name) == spec["sha256"]
        common.sign("evaluation_student_base", [Path(sensitive.__file__), DATA / "dev.json", base_sig_path],
                    {"prompt": common.BASE["student_prompt"], "max_new_tokens": C["eval_max_new_tokens"], "greedy": True,
                     "reused_prefix": 200, "actual_tool_executions": 0})
        old = common.rows(path)
        assert 200 <= len(old) <= 250
        for row,item in zip(old,items):
            assert row["source_id"] == item["source_id"] and row["question_sha256"] == item["question_sha256"]
            assert row["output_sha256"] == common.digest_text(row["output"])
            assert all(row[k] == v for k,v in baseline.score(row["scoring_text"],item["gold"]).items())
        if len(old) < 250:
            mx.set_memory_limit(14*1024**3)
            mx.set_cache_limit(512*1024**2)
            mx.random.seed(C["seed"])
            model,tok = load(str(ROOT / common.BASE["student_dir"]),tokenizer_config={"local_files_only":True})
            model.eval()
            for item in items[len(old):]:
                ids = tok.encode(common.BASE["student_prompt"].format(question=item["question"]),add_special_tokens=False)
                assert len(ids)+C["eval_max_new_tokens"] <= C["context_tokens"]
                text,boundary,last = "",None,None
                tick=time.perf_counter()
                for part in stream_generate(model,tok,ids,max_tokens=C["eval_max_new_tokens"],sampler=make_sampler(temp=0.0)):
                    text += part.text
                    last=part
                    boundary=re.search(r"(?:^|\n)\s*(?:###\s*)?Question:",text)
                    if boundary:
                        break
                assert last is not None
                scored=text[:boundary.start()] if boundary else text
                value,rule=sensitive.parse(scored)
                row={"source_id":item["source_id"],"question_sha256":item["question_sha256"],"gold":item["gold"],
                     "output":text,"scoring_text":scored,"output_sha256":common.digest_text(text),
                     "generation_tokens":last.generation_tokens,"seconds":time.perf_counter()-tick,
                     "finish_reason":"next_question" if boundary else last.finish_reason,**baseline.score(scored,item["gold"]),
                     "v2_correct":value is not None and value==baseline.number(item["gold"]),"v2_invalid":value is None,"v2_rule":rule}
                common.append(path,row)
                old.append(row)
                baseline.save(directory / "progress.json",{"completed":len(old),"total":250,"reused_prefix":200})
        baseline.save(directory / "summary.json",{"completed":len(old),"reused_prefix":200,"new_generations":len(old)-200,
                                                  "predictions_sha256":baseline.sha(path),"full_math_review":"pending"})


def step(stage,condition=None):
    bind()
    if stage == "teacher":
        import pilot_materials
        pilot_materials.teacher(False)
    elif stage == "locate":
        import pilot_localize_v2
        pilot_localize_v2.main()
    elif stage == "edits":
        import pilot_edits_v4
        pilot_edits_v4.main()
    elif stage == "preflight":
        import pilot_preflight
        pilot_preflight.main()
    elif stage in ["build","train","evaluate"]:
        import pilot_train
        pilot_train.build() if stage == "build" else getattr(pilot_train,stage)(condition)
    elif stage == "evaluate-base":
        evaluate_base()


def pipeline():
    def run(stage,condition=None):
        baseline.save(OUT / "pipeline_status.json",{"stage":stage,"condition":condition,"pid":os.getpid()})
        common.append(OUT / "stage_events.jsonl",{"stage":stage,"condition":condition,"time":time.time()})
        cmd=[str(PYTHON),"-u",str(Path(__file__).absolute()),stage]
        if condition:
            cmd += ["--condition",condition]
        subprocess.run(cmd,check=True)
    run("teacher")
    wait_reviews("teacher_targets.jsonl","teacher_reviews.jsonl")
    run("locate")
    wait_reviews("locations.jsonl","location_reviews.jsonl")
    run("edits")
    for name in ["difficulty_only","difficulty_attention"]:
        wait_reviews(f"{name}_edits_v4.jsonl",f"{name}_edit_v4_reviews.jsonl")
    run("build")
    run("preflight")
    for name in C["conditions"]:
        run("train",name)
    run("evaluate-base")
    for name in C["conditions"]:
        run("evaluate",name)
    baseline.save(OUT / "pipeline_status.json",{"stage":"needs_final_output_audit_and_report","trained_conditions":3,
                  "evaluated_per_condition":250,"formal_test_evaluated":False,"pid":os.getpid()})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","teacher","locate","edits","build","preflight","train","evaluate","evaluate-base","pipeline"])
    parser.add_argument("--condition",choices=C["conditions"])
    parser.add_argument("--background",action="store_true")
    parser.add_argument("--lock-fd",type=int)
    args=parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage != "pipeline":
        step(args.stage,args.condition)
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        assert PYTHON.is_file(), f"Keep configured venv path: {PYTHON}"
        if args.background:
            with (OUT / "pipeline.lock").open("a+") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with (OUT / "pipeline.log").open("a") as log:
                    proc=subprocess.Popen([str(PYTHON),"-u",str(Path(__file__).absolute()),"pipeline","--lock-fd",str(lock.fileno())],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
                baseline.save(OUT / "launch.json",{"pid":proc.pid,"time":time.time(),"log":str(OUT / "pipeline.log")})
                print(json.dumps({"pid":proc.pid,"log":str(OUT / "pipeline.log")}))
        else:
            handle=os.fdopen(args.lock_fd,"a+") if args.lock_fd is not None else (OUT / "pipeline.lock").open("a+")
            with handle as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    pipeline()
                except BaseException as exc:
                    baseline.save(OUT / "pipeline_status.json",{"stage":"failed","error_type":type(exc).__name__,"message":str(exc)})
                    raise
