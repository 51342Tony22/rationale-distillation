"""Durable staged pilot runner. Semantic reviews remain explicit artifacts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime

from .common import ROOT, load_config, read_jsonl, resolve_path, write_jsonl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("bootstrap","repair-originals","screen","locals","verify","train-evaluate"))
    parser.add_argument("--config",required=True)
    parser.add_argument("--background",action="store_true")
    parser.add_argument("--requests",help="Explicit saved original repair queue; only for repair-originals")
    args=parser.parse_args()
    if args.requests and args.stage!="repair-originals":
        parser.error("--requests only applies to repair-originals")
    config_path=resolve_path(args.config)
    config=load_config(config_path)
    root=resolve_path(config["refinement"]["root"])
    status_path=root/"run_status.jsonl"
    log_path=root/(args.stage+"_pipeline.log")
    if args.background:
        prior=read_jsonl(status_path)
        if prior and prior[0].get("state")=="running" and prior[0].get("pid"):
            try:
                os.kill(prior[0]["pid"],0)
            except ProcessLookupError:
                pass
            else:
                print(json.dumps({"state":"already_running","pid":prior[0]["pid"]}))
                return
        with log_path.open("a") as log:
            command=[sys.executable,"-u","-m","fyp_distill.run_refinement_trainpilot",args.stage,"--config",str(config_path)]
            if args.requests:
                command += ["--requests",str(resolve_path(args.requests))]
            process=subprocess.Popen(command,cwd=ROOT,
                                     env=dict(os.environ,PYTHONPATH=str(ROOT/"src")),
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_jsonl(status_path,[{"state":"running","stage":args.stage,"pid":process.pid,
                                  "updated_at":datetime.now(UTC).isoformat(),"log":str(log_path)}])
        subprocess.Popen(["/usr/bin/caffeinate","-i","-w",str(process.pid)],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({"state":"started","stage":args.stage,"pid":process.pid,"log":str(log_path)}))
        return
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),PYTHONUNBUFFERED="1")
    py=str(ROOT/".venv/bin/python")
    mlx=str(ROOT/".venv-mlx/bin/python")
    started=datetime.now(UTC).isoformat()

    def status(state,stage,**extra):
        write_jsonl(status_path,[{"state":state,"stage":stage,"pid":os.getpid(),
                                  "config":str(config_path),"started_at":started,
                                  "updated_at":datetime.now(UTC).isoformat(),"log":str(log_path),**extra}])

    def run(stage,command):
        status("running",stage)
        print(json.dumps({"stage":stage,"command":command}),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)

    def diagnostic(stage):
        return [mlx,"-u","-m","fyp_distill.diagnose_refinement_mlx",stage,"--config",str(config_path)]

    with (root/"pilot.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.stage=="bootstrap":
                requests=read_jsonl(root/"original_requests.jsonl")
                status("running","originals_and_base_evaluation",planned_original_calls=len(requests),evaluation_questions=70)
                # API and the one GPU evaluation are independent; no competing MLX jobs.
                with (root/"original_pipeline.log").open("a") as original_log:
                    teacher=subprocess.Popen([py,"-u","-m","fyp_distill.request_refinement_hints",
                                               "--config",str(config_path),"--requests",str(root/"original_requests.jsonl"),
                                               "--limit",str(len(requests))],cwd=ROOT,env=env,stdout=original_log,stderr=subprocess.STDOUT)
                    evaluation=subprocess.Popen([mlx,"-u","-m","fyp_distill.evaluate_mlx",
                                                  "--config",str(config_path),"--variant","base_student",
                                                  "--seed","7","--split","dev","--skip-behavioural"],cwd=ROOT,env=env)
                    status("running","originals_and_base_evaluation",teacher_pid=teacher.pid,evaluation_pid=evaluation.pid,
                           planned_original_calls=len(requests),evaluation_questions=70)
                    teacher_code=teacher.wait()
                    eval_code=evaluation.wait()
                if teacher_code or eval_code:
                    raise RuntimeError(f"bootstrap returncodes teacher={teacher_code}, evaluation={eval_code}")
                status("needs_original_review","bootstrap_finished",originals=len(read_jsonl(config["data"]["teacher_file"])),training_started=False)
            elif args.stage=="repair-originals":
                queue_path=resolve_path(args.requests) if args.requests else root/"original_repair_queue_1.jsonl"
                requests=read_jsonl(queue_path)
                if not requests or any(r.get("kind")!="original" or not r.get("replaces_request_id") for r in requests):
                    raise ValueError("Require explicitly audited, saved original repair requests")
                run("teacher_original_repairs",[py,"-u","-m","fyp_distill.request_refinement_hints",
                                                "--config",str(config_path),"--requests",str(queue_path),
                                                "--limit",str(len(requests))])
                status("needs_original_review","original_repair_queue_finished",
                       originals=len(read_jsonl(config["data"]["teacher_file"])),training_started=False)
            elif args.stage=="screen":
                bases=read_jsonl(root/"bases.jsonl")
                if not bases or not all(all(b.get("review",{}).get(k) is True for k in
                                           ("correct_steps","prefixes_do_not_reveal_answer")) for b in bases):
                    raise ValueError("All originals and boundaries require actual review")
                run("student_screening",diagnostic("screen"))
                status("needs_screen_review","screen_finished",training_started=False)
            elif args.stage=="locals":
                run("prepare_hint_requests",[py,"-m","fyp_distill.refinement","requests","--config",str(config_path)])
                from .refinement import write_frozen
                requests=[r for r in read_jsonl(root/"teacher_requests.jsonl") if r["kind"]=="local"]
                write_frozen(root/"local_requests.jsonl",requests)
                run("teacher_local_generation",[py,"-u","-m","fyp_distill.request_refinement_hints","--config",str(config_path),
                                                "--requests",str(root/"local_requests.jsonl"),"--limit",str(len(requests))])
                status("needs_hint_review","local_queue_finished",training_started=False)
            elif args.stage=="verify":
                run("independent_verification",diagnostic("verify")+["--hints",str(root/"reviewed_hints.jsonl")])
                status("needs_controls_and_build","verification_finished",training_started=False)
            else:
                from .refinement import GROUPS
                # This stage is invoked only after six groups have passed the builder.
                variants=root/"variants"
                if not all(read_jsonl(variants/(g+".jsonl")) for g in GROUPS):
                    raise ValueError("Build all six audited groups before training")
                run("audit_supervised_tokens",[mlx,"-m","fyp_distill.audit_refinement_training_tokens","--config",str(config_path)])
                run("train_six_groups",[mlx,"-u","-m","fyp_distill.run_mlx_matrix","--stage","train","--config",str(config_path)])
                run("evaluate_six_groups",[mlx,"-u","-m","fyp_distill.run_mlx_matrix","--stage","evaluate",
                                          "--split","dev","--skip-behavioural","--config",str(config_path)])
                run("summarize",[py,"-m","fyp_distill.summarize_mlx","--split","dev","--config",str(config_path)])
                run("report_training_pilot",[py,"-m","fyp_distill.report_refinement_training","--config",str(config_path)])
                status("needs_final_audit","training_and_evaluation_finished",training_started=True)
        except Exception as error:
            status("failed",args.stage,error_type=type(error).__name__,error=str(error))
            raise


if __name__=="__main__":
    main()
