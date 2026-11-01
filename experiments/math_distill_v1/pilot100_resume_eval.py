"""Resume evaluation only, preserving the signed training runner and artifacts.

The original entry expected nested model-file SHA records; the frozen baseline
signature actually stores SHA strings. This entry checks that exact schema and
keeps prompt, generation, score and stop handling unchanged.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import baseline
import pilot_common as common
import pilot100 as pilot
import score_sensitivity_v2 as sensitive

OUT, ROOT, C, DATA = pilot.OUT, pilot.ROOT, pilot.C, pilot.DATA
HERE = Path(__file__).absolute()


def verify():
    evidence = {}
    for stage in ["teacher_targets", "locations", "edits_v4", *["training_"+n for n in C["conditions"]]]:
        path = OUT / f"{stage}_signature.json"
        signed = json.loads(path.read_text())
        for relative, expected in signed["inputs"].items():
            assert baseline.sha(ROOT / relative) == expected, relative
        evidence[str(path.relative_to(ROOT))] = baseline.sha(path)
    for name in C["conditions"]:
        folder = OUT / "training" / name
        metadata = json.loads((folder / "metadata.json").read_text())
        assert metadata["completed"] and metadata["actual_new_training"]
        assert baseline.sha(folder / "adapters.safetensors") == metadata["adapter_sha256"]
        evidence[str((folder / "metadata.json").relative_to(ROOT))] = baseline.sha(folder / "metadata.json")
    original = ROOT / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl"
    assert (OUT / "evaluation/student_base/predictions.jsonl").read_bytes().startswith(original.read_bytes())
    baseline.freeze(OUT / "evaluation_recovery_preflight.json", {
        "reason": "Original evaluate_base treated SHA string as nested object; failed before new generation.",
        "runner_sha256": baseline.sha(HERE), "signed_artifacts_verified": evidence,
        "training_repeated": False, "old_outputs_overwritten": False,
        "original_failure_status": json.loads((OUT / "evaluation_failure_status.json").read_text())})


def evaluate_base():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    items = common.sources("dev")
    directory = OUT / "evaluation/student_base"
    path = directory / "predictions.jsonl"
    with common.gpu("evaluate_base_added50_recovery"):
        base_sig_path = ROOT / "output/math_distill_v1/dev_baseline/student_base/signature.json"
        signed = json.loads(base_sig_path.read_text())
        assert baseline.sha(Path(baseline.__file__)) == signed["runner_sha256"]
        for name, expected_sha in signed["model_files"].items():
            assert isinstance(expected_sha, str) and len(expected_sha) == 64
            assert baseline.sha(ROOT / common.BASE["student_dir"] / name) == expected_sha
        common.sign("evaluation_student_base", [Path(sensitive.__file__), DATA / "dev.json", base_sig_path, HERE],
                    {"prompt": common.BASE["student_prompt"], "max_new_tokens": C["eval_max_new_tokens"], "greedy": True,
                     "reused_prefix": 200, "actual_tool_executions": 0,
                     "recovery": "Read baseline model_files SHA strings; unchanged inference and scores"})
        old = common.rows(path)
        assert 200 <= len(old) <= 250
        for row, item in zip(old, items):
            assert row["source_id"] == item["source_id"] and row["question_sha256"] == item["question_sha256"]
            assert row["output_sha256"] == common.digest_text(row["output"])
            assert all(row[k] == v for k,v in baseline.score(row["scoring_text"],item["gold"]).items())
        if len(old) < 250:
            mx.set_memory_limit(14*1024**3)
            mx.set_cache_limit(512*1024**2)
            mx.random.seed(C["seed"])
            model, tok = load(str(ROOT / common.BASE["student_dir"]), tokenizer_config={"local_files_only":True})
            model.eval()
            for item in items[len(old):]:
                ids = tok.encode(common.BASE["student_prompt"].format(question=item["question"]), add_special_tokens=False)
                assert len(ids)+C["eval_max_new_tokens"] <= C["context_tokens"]
                text, boundary, last = "", None, None
                tick = time.perf_counter()
                for part in stream_generate(model,tok,ids,max_tokens=C["eval_max_new_tokens"],sampler=make_sampler(temp=0.0)):
                    text += part.text
                    last = part
                    boundary = re.search(r"(?:^|\n)\s*(?:###\s*)?Question:",text)
                    if boundary:
                        break
                assert last is not None
                scored = text[:boundary.start()] if boundary else text
                value, rule = sensitive.parse(scored)
                row = {"source_id":item["source_id"],"question_sha256":item["question_sha256"],"gold":item["gold"],
                       "output":text,"scoring_text":scored,"output_sha256":common.digest_text(text),
                       "generation_tokens":last.generation_tokens,"seconds":time.perf_counter()-tick,
                       "finish_reason":"next_question" if boundary else last.finish_reason,**baseline.score(scored,item["gold"]),
                       "v2_correct":value is not None and value==baseline.number(item["gold"]),"v2_invalid":value is None,"v2_rule":rule}
                common.append(path,row)
                old.append(row)
                baseline.save(directory / "progress.json",{"completed":len(old),"total":250,"reused_prefix":200})
        baseline.save(directory / "summary.json",{"completed":len(old),"reused_prefix":200,"new_generations":len(old)-200,
                                                 "predictions_sha256":baseline.sha(path),"full_math_review":"pending"})


def worker():
    pilot.bind()
    verify()
    baseline.save(OUT / "pipeline_status.json",{"stage":"evaluate-base","recovery":True,"pid":os.getpid()})
    common.append(OUT / "stage_events.jsonl",{"stage":"evaluate-base-recovery","time":time.time()})
    evaluate_base()
    for name in C["conditions"]:
        baseline.save(OUT / "pipeline_status.json",{"stage":"evaluate","condition":name,"recovery":True,"pid":os.getpid()})
        common.append(OUT / "stage_events.jsonl",{"stage":"evaluate","condition":name,"time":time.time()})
        subprocess.run([str(pilot.PYTHON),"-u",str(Path(pilot.__file__).absolute()),"evaluate","--condition",name],check=True)
    baseline.save(OUT / "pipeline_status.json",{"stage":"needs_final_output_audit_and_report","trained_conditions":3,
                  "evaluated_per_condition":250,"formal_test_evaluated":False,"recovery":True,"pid":os.getpid()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background",action="store_true")
    parser.add_argument("--lock-fd",type=int)
    args = parser.parse_args()
    if args.background:
        with (OUT / "pipeline.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            status = json.loads((OUT / "pipeline_status.json").read_text())
            assert status["stage"] == "failed"
            baseline.freeze(OUT / "evaluation_failure_status.json",status)
            with (OUT / "evaluation_resume.log").open("a") as log:
                proc = subprocess.Popen([str(pilot.PYTHON),"-u",str(HERE),"--lock-fd",str(lock.fileno())],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
            baseline.save(OUT / "evaluation_resume_launch.json",{"pid":proc.pid,"time":time.time(),"runner_sha256":baseline.sha(HERE)})
            print(json.dumps({"pid":proc.pid,"training_repeated":False}))
    else:
        with os.fdopen(args.lock_fd,"a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                worker()
            except BaseException as exc:
                baseline.save(OUT / "pipeline_status.json",{"stage":"failed","recovery":True,"error_type":type(exc).__name__,"message":str(exc)})
                raise
