"""Real-student continuation check on two reviewed official TRAIN roots only.

Prepare on CPU while the teacher runs. GPU execution waits for the shared lock.
Engineering use of these train roots is development exposure, not confirmation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import subprocess
import sys
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,DATA,BASE,read,freeze,save,baseline,signature,gpu,checked_rows,digest,now
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from training_engine import run_training
from trial_calibration import adapter_hash

FIXTURE=OUT/'preflight/formal_training_fixture_v1.json'


def cpu_preflight():
    prepare()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),
        '-p','test_*.py','-v'],cwd=ROOT,capture_output=True,text=True)
    log=OUT/'preflight'/('formal_training_cpu_'+now().replace(':','-')+'.log')
    log.write_text(result.stdout+result.stderr)
    count=re.search(r'Ran (\d+) tests?',result.stdout+result.stderr)
    if result.returncode or count is None:
        raise RuntimeError('Formal-training CPU checks failed; retained log: '+str(log))
    paths=[HERE/name for name in ('training_plan.py','training_checkpoint.py','training_engine.py',
        'train_formal.py','preflight_formal_training.py','test_training.py','learning.py',
        'runtime.py','runtime_full.py','trial_calibration.py','variant_rationales.py','material_builder.py',
        'evaluate.py','evaluation_inputs.py','preflight_adapter_loading.py',
        'whole_span_masking.py','whole_span_probe.py','calibration_summary.py','positions_nll.py','explanation_pool.py',
        'budgeted_materials.py','full_selector.py','full_explanations.py','full_selector_materials.py',
        'train_selector_quality_v3.py','auxiliary_explainer_recipe.py','review_general_explainer.py',
        'automatic_teacher_quality.py','teacher_spotcheck_v2.py')]
    paths.extend(sorted(HERE.glob('test_*.py')))
    paths.extend([FIXTURE,OUT/'five_group_amendment_v1.json',OUT/'transfer128_amendment_v1.json',
        OUT/'execution_priority.json',OUT/'preflight/learning.json',ROOT/'experiments/math_distill_v1/config.json'])
    report={'status':'passed','at_utc':now(),'total_CPU_tests_passed':int(count.group(1)),
        'device':'tiny_actual_MLX_CPU_and_state_machine_only','real_student_GPU_validation':'queued_not_run',
        'full_optimizer_and_PRNG_continuation_checked':True,'five_group_materials_and_M6_epoch_switch_checked':True,
        'retained_log':str(log.relative_to(ROOT)),'log_sha256':baseline.sha(log),
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths},'official_test_used':False}
    report_path=OUT/'preflight/formal_training_cpu_v1.json'
    if report_path.exists():
        freeze(OUT/'preflight/history'/('formal_training_cpu_'+baseline.sha(report_path)+'.json'),read(report_path))
    save(report_path,report)
    print({'CPU_tests_passed':int(count.group(1)),'real_student_GPU_test':'not_run'},flush=True)


def prepare():
    sources={r['source_id']:r for r in read(DATA/'train7473.json')}
    reviews={r['source_id']:r for r in checked_rows(OUT/'material_audits/teacher_additional_reviews_v1.jsonl')}
    wanted={'gsm8k:train:0','gsm8k:train:2'};targets={}
    # Read only newline-terminated records from the active generation file.
    raw=(OUT/'materials/teacher_targets_train7473.jsonl').read_bytes()
    for line in raw.split(b'\n')[:-1]:
        row=json.loads(line)
        if row['source_id'] in wanted:
            value=dict(row);checksum=value.pop('record_sha256')
            if digest(json.dumps(value,sort_keys=True))!=checksum:raise ValueError('Bad teacher row signature')
            targets[row['source_id']]=row
    if set(targets)!=wanted:raise ValueError('Two reviewed teacher fixtures not yet available')
    rows=[]
    for sid in sorted(wanted):
        src=sources[sid];teacher=targets[sid];review=reviews[sid]
        if (not review['accepted'] or not review['all_text_read']
                or review['output_sha256']!=teacher['output_sha256']
                or review['question_sha256']!=src['question_sha256']):
            raise ValueError('Teacher fixture needs exact full-text acceptance')
        material={'root_source_id':sid,'question':src['question'],'question_sha256':src['question_sha256'],
            'completion':teacher['output'],'completion_sha256':teacher['output_sha256'],'kind':'ordinary_CoT'}
        rows.append({'source_id':sid,'question_sha256':src['question_sha256'],
                     'epoch_materials':[copy.deepcopy(material) for _ in range(4)]})
    value={'schema':'tgrd_training_materials_v1','condition':'M1','candidate_root_count':7473,'rows':rows,
        'purpose':'engineering_only_two_reviewed_train_roots_not_formal_experiment',
        'teacher_record_proofs':[targets[sid] for sid in sorted(wanted)],
        'full_text_reviews':[reviews[sid] for sid in sorted(wanted)],
        'internal_train_roots_are_development_exposure':True,'official_test_used':False}
    freeze(FIXTURE,value)
    print({'prepared_roots':2,'GPU_loaded':False,'fixture':str(FIXTURE.relative_to(ROOT))},flush=True)


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from learning import Learner,tokenize
    cpu=read(OUT/'preflight/formal_training_cpu_v1.json')
    if cpu['status']!='passed':raise ValueError('Formal-training CPU validation required')
    for name,checksum in cpu['inputs'].items():
        if baseline.sha(ROOT/name)!=checksum:
            raise ValueError('Formal-training CPU validation is stale: '+name)
    fixture=read(FIXTURE)
    with gpu():
        signature('formal_training_gpu_quality_v3',[Path(__file__),HERE/'training_plan.py',HERE/'training_checkpoint.py',
            HERE/'training_engine.py',HERE/'learning.py',HERE/'train_formal.py',
            HERE/'train_selector_quality_v3.py',HERE/'trial_calibration.py',
            HERE/'runtime.py',HERE/'runtime_full.py',FIXTURE,OUT/'preflight/learning.json',
            ROOT/'experiments/math_distill_v1/config.json'],
            {'roots':2,'data':'reviewed_train0_and_train2_development_only','updates':4,
             'two_microbatches_per_update':True,'seed':42,'purpose':'engineering_not_method_effectiveness'})
        if read(OUT/'preflight/learning.json')['status']!='passed':
            raise ValueError('Learning preflight not passed')
        baseline.verify_student(BASE)
        folder=OUT/'preflight/formal_training_gpu_quality_v3'
        args=dict(seed=42,tokenize=tokenize,weight_hash=adapter_hash,save_checkpoint=save_checkpoint,
                  restore_checkpoint=restore_checkpoint,latest_checkpoint=latest_checkpoint,checkpoint_interval=1)
        learner=Learner(42)
        reference=run_training(learner,fixture,folder/'uninterrupted',**args)
        # A prior completed reference may return before restoring final weights.
        restore_checkpoint(learner,latest_checkpoint(folder/'uninterrupted/checkpoints'))
        reference_hash=adapter_hash(learner)
        reference_optimizer={k:mx.array(v) for k,v in tree_flatten(learner.optimizer.state)}
        reference_random=[mx.array(v) for v in mx.random.state]
        mx.eval(reference_optimizer,reference_random)
        del learner;gc.collect();mx.clear_cache()
        learner=Learner(42)
        marker=folder/'expected_interruption.json'
        if not marker.exists():
            def stop(step):
                if step['update']==2:raise InterruptedError('Intentional checkpoint2 interruption')
            try:run_training(learner,fixture,folder/'interrupted',after_update=stop,**args)
            except InterruptedError:
                freeze(marker,{'intentional_interruption_after_update':2,'checkpoint_committed':True})
            else:raise ValueError('Controlled formal-training interruption was not reached')
            del learner;gc.collect();mx.clear_cache();learner=Learner(42)
        resumed=run_training(learner,fixture,folder/'interrupted',**args)
        restore_checkpoint(learner,latest_checkpoint(folder/'interrupted/checkpoints'))
        if adapter_hash(learner)!=reference_hash:
            raise ValueError('Cold resume final adapter differs from uninterrupted training')
        current=dict(tree_flatten(learner.optimizer.state))
        if set(current)!=set(reference_optimizer) or any(
                not bool(mx.all(current[k]==v).item()) for k,v in reference_optimizer.items()):
            raise ValueError('Cold resume optimizer moments/step differ from uninterrupted training')
        if any(not bool(mx.all(a==b).item()) for a,b in zip(mx.random.state,reference_random)):
            raise ValueError('Cold resume global PRNG state differs')
        if (reference['updates']!=4 or resumed['updates']!=4 or resumed['root_exposures']!=8
                or reference['supervised_tokens_including_eos']!=resumed['supervised_tokens_including_eos']):
            raise ValueError('Root/update/token budget mismatch after resume')
        save(OUT/'preflight/formal_training_gpu_quality_v3.json',{'status':'passed','at_utc':now(),
            'device':'actual_student_GPU','reviewed_training_fixture_roots':2,'updates':4,
            'cold_resume_exact_final_adapter':True,'cold_resume_exact_optimizer':True,
            'cold_resume_exact_global_PRNG':True,'root_exposures':8,
            'train_roots_used_for_development_not_confirmation':True,
            'formal_five_group_training_started':False,'public_test_used':False,
            'signature_file_sha256':baseline.sha(OUT/'signatures/formal_training_gpu_quality_v3.json')})
        print(read(OUT/'preflight/formal_training_gpu_quality_v3.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:prepare()
    elif args.cpu_only:cpu_preflight()
    else:main()
