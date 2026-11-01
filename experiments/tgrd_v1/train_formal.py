"""Five-group formal trainer, gated until development and material review finish.

This file's existence is not a completed experiment or an enabled training gate.
No public-test file is loaded by training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,DATA,BASE,read,freeze,baseline,signature,gpu,save,now
from training_plan import CONDITIONS,SEEDS,PROTOCOL,validate_materials
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from training_engine import run_training
from trial_calibration import adapter_hash


def validate_gate(condition,seed):
    gate_path=OUT/'gates/final_training_frozen.json';gate=read(gate_path)
    if (gate.get('enabled') is not True or gate.get('primary_dataset')!='GSM8K'
            or gate.get('protocol')!=PROTOCOL
            or gate.get('independent_NLL_all_steps_validated') is not True
            or gate.get('five_group_amendment_sha256')!=baseline.sha(OUT/'five_group_amendment_v1.json')
            or set(gate['conditions'])!=set(CONDITIONS) or gate['seeds']!=list(SEEDS)
            or gate.get('development_accepted') is not True
            or gate.get('full_teacher_quality_review_complete') is not True
            or gate['candidate_root_count']!=7473):
        raise ValueError('Final development, quality and complete five-group protocol must be frozen')
    if condition not in CONDITIONS or seed not in SEEDS:raise ValueError('Undeclared formal group/seed')
    preflight=read(OUT/'preflight/formal_training_gpu_v1.json')
    if preflight['status']!='passed':raise ValueError('Actual formal-training continuation preflight required')
    signed_path=OUT/'signatures/formal_training_gpu_v1.json'
    if baseline.sha(signed_path)!=preflight['signature_file_sha256']:
        raise ValueError('Training preflight signature changed')
    for name,checksum in read(signed_path)['inputs'].items():
        if baseline.sha(ROOT/name)!=checksum:
            raise ValueError('Training code/input changed after actual-model preflight: '+name)
    material_paths=[];selected=None
    if set(gate['materials'])!=set(CONDITIONS):raise ValueError('All five material sets required')
    for group in CONDITIONS:
        entry=gate['materials'][group];path=ROOT/entry['path']
        if baseline.sha(path)!=entry['sha256']:raise ValueError('Frozen material file changed')
        bundle=read(path)
        if bundle['condition']!=group:raise ValueError('Material condition label mismatch')
        if bundle.get('protocol')!=PROTOCOL:raise ValueError('Material protocol is not the current five-group study')
        if bundle.get('development_preview_only') is True:
            raise ValueError('Development material previews cannot be used as formal experiment materials')
        validate_materials(bundle,gate['common_root_ids'])
        material_paths.append(path)
        if group==condition:selected=bundle
    sources={r['source_id']:r for r in read(DATA/'train7473.json')}
    for row in selected['rows']:
        if row['question_sha256']!=sources[row['source_id']]['question_sha256']:
            raise ValueError('Formal material root/source mismatch')
    return gate_path,material_paths,selected


def main(condition,seed):
    gate_path,material_paths,bundle=validate_gate(condition,seed)
    folder=OUT/'training'/condition/f'seed{seed}'
    signature(f'formal_training_{condition}_s{seed}_v1',[Path(__file__),HERE/'training_plan.py',
        HERE/'training_checkpoint.py',HERE/'training_engine.py',HERE/'learning.py',
        HERE/'trial_calibration.py',HERE/'runtime.py',HERE/'runtime_full.py',gate_path,
        OUT/'full_run_authorization.json',OUT/'formal_seed_amendment_v1.json',OUT/'four_group_amendment_v1.json',OUT/'five_group_amendment_v1.json',
        OUT/'preflight/formal_training_gpu_v1.json',
        ROOT/'experiments/math_distill_v1/config.json',*material_paths],
        {'condition':condition,'seed':seed,'public_test_used':False,
         'primary_checkpoint':'last_epoch4','accepted_roots':len(bundle['rows'])})
    with gpu():
        from learning import Learner,LORA,tokenize
        baseline.verify_student(BASE);learner=Learner(seed)
        freeze(OUT/'training/initializations'/f'seed{seed}.json',{
            'seed':seed,'initial_adapter_sha256':adapter_hash(learner),
            'student_revision':BASE['student_revision'],'LoRA':LORA,'layers':16})
        freeze(folder/'adapter_config.json',{'fine_tune_type':'lora','num_layers':16,'lora_parameters':LORA})
        metadata=run_training(learner,bundle,folder,seed=seed,tokenize=tokenize,weight_hash=adapter_hash,
            save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,
            latest_checkpoint=latest_checkpoint)
        save(folder/'completion_status.json',{'at_utc':now(),**metadata})
        print(metadata,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=CONDITIONS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=parser.parse_args();main(args.condition,args.seed)
