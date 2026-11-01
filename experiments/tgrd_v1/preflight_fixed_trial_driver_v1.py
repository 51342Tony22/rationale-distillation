"""Actual-model synthetic fixed-configuration trial-driver check, run only under the shared GPU lock.

Exercises two real updates, a controlled interruption after an adapter commit,
a newly loaded learner, immutable Vsel choices before Vaudit, exact restoration,
and a fully cached replay. Short 32-token generations are engineering fixtures,
not a scientific GSM8K accuracy measurement or formal generation setting.
"""
from __future__ import annotations

import gc
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,read,freeze,save,signature,gpu,baseline,BASE,digest,checked_rows,now
from trial_calibration import adapter_hash,restore_adapter,evaluate_probes
from fixed_trial_engine_v1 import run_fixed_root


def fixtures():
    question='A box contains 3 red pencils and 4 blue pencils. How many pencils are in the box?'
    completion='There are 3 + 4 = 7 pencils in total.\nThe answer is 7.'
    explanation=('The two counts describe separate groups in the same box. Combining these groups '
                 'requires adding their counts because neither group is already included in the '
                 'other; multiplication would describe repeated equal groups instead.')
    augmented=completion.replace('\nThe answer', '\nExplanation: '+explanation+'\nThe answer')
    source={'source_id':'synthetic:trial_driver:1','question':question,'question_sha256':digest(question)}
    candidates=[{'candidate_id':cid,'completion':text,'completion_sha256':digest(text),
                 'student_tokens':length} for cid,text,length in [('e0',completion,0),('explain',augmented,40)]]
    pool={'source_id':source['source_id'],'question_sha256':digest(question),
          'teacher_output_sha256':digest(completion),'candidates':candidates,
          'purpose':'assistant_authored_synthetic_engineering_fixture_not_formal_material'}
    probes={}
    for split,pairs in [('select',[(1,2),(2,4),(3,5),(4,5),(2,7),(5,6)]),
                        ('audit',[(1,4),(2,5),(3,6),(4,7),(5,8),(6,9)])]:
        probes[split]=[]
        for i,(red,blue) in enumerate(pairs):
            q=(f'A box contains {red} red pencils and {blue} blue pencils. How many pencils are in the box?'
               if split=='select' else
               f'There are {red} red pencils and {blue} blue pencils in one case. How many pencils are there altogether?')
            probes[split].append({'id':f'synthetic:trial_driver:{split}:{i}',
                'source_id':source['source_id'],'split':split,'question':q,
                'question_sha256':digest(q),'gold':str(red+blue)})
    return source,pool,probes


def main():
    import mlx.core as mx
    from learning import Learner
    from train_selector_quality_v3 import verify_inputs
    cpu=read(OUT/'preflight/transfer128_preparation_and_fixed_trial_cpu_v1.json')
    if cpu.get('status')!='passed' or cpu.get('new_CPU_tests_passed')!=13:
        raise ValueError('Actual fixed-trial CPU integration report required')
    verify_inputs(cpu)
    if read(OUT/'preflight/learning.json')['status']!='passed':
        raise ValueError('Actual learning/reset preflight must pass first')
    with gpu():
        signature('fixed_trial_driver_gpu_v1',[Path(__file__),HERE/'fixed_trial_engine_v1.py',HERE/'trial_engine.py',HERE/'trial_calibration.py',
            HERE/'trial_selection.py',HERE/'learning.py',HERE/'runtime.py',HERE/'runtime_full.py',
            ROOT/'experiments/math_distill_v1/config.json',OUT/'preflight/learning.json',
            OUT/'preflight/transfer128_preparation_and_fixed_trial_cpu_v1.json'],
            {'data':'synthetic_only','updates':1,'seed':701,'generation_limit':32,
             'purpose':'engineering_not_method_effectiveness','controlled_interruption':True})
        baseline.verify_student(BASE)
        source,pool,probes=fixtures()
        folder=OUT/'preflight/fixed_trial_driver_gpu_v1/root'
        marker=folder.parent/'intentional_interruption.json'
        learner=Learner(701)
        # Record the exact synthetic explanation's real tokenizer length.
        text=pool['candidates'][1]['completion'].split('Explanation: ',1)[1].split('\nThe answer',1)[0]
        pool['candidates'][1]['student_tokens']=len(learner.tokenizer.encode(text,add_special_tokens=False))
        initial_hash=adapter_hash(learner)

        def evaluate(current,items,path):
            if items[0]['split']=='audit' and not (folder/'decision_before_audit.json').exists():
                raise ValueError('Audit was reached before frozen Vsel decisions')
            return evaluate_probes(current,items,path,generation_limit=32)

        arguments=dict(seed=701,updates=1,initial_hash=initial_hash,
                       weight_hash=adapter_hash,restore=restore_adapter,threshold=.03,
                       configuration_sha256=digest('synthetic_engineering_configuration_only_not_development_selection'))
        if not marker.exists():
            def interrupt(current,items,path):
                if path.parent.name=='branch1' and items[0]['split']=='select':
                    raise InterruptedError('Intentional engineering interruption after committed training')
                return evaluate(current,items,path)
            try:
                run_fixed_root(learner,source,pool,probes,folder,evaluate=interrupt,**arguments)
            except InterruptedError:
                if not (folder/'branch1/trained.json').exists():raise
                if (folder/'decision_before_audit.json').exists():
                    raise ValueError('Unexpected early decision freeze')
                freeze(marker,{'interruption':'intentional_after_branch1_training_commit',
                    'both_real_training_branches_committed':True,'decisions_or_audit_reached':False})
            else:
                raise ValueError('Controlled interruption was not reached')
            # A fresh real learner models a process restart. No adapters or
            # optimizer objects from the interrupted learner are reused.
            del learner;gc.collect();mx.clear_cache()
            learner=Learner(701)
            if adapter_hash(learner)!=initial_hash:
                raise ValueError('Fresh learner does not reproduce the same S0 adapter')

        def must_not_train(*args,**kwargs):
            raise ValueError('Committed training was repeated during resume')
        learner.trial=must_not_train
        before_adapters={str(p):baseline.sha(p) for p in folder.glob('branch*/adapter.safetensors')}
        result=run_fixed_root(learner,source,pool,probes,folder,evaluate=evaluate,**arguments)
        if before_adapters!={str(p):baseline.sha(p) for p in folder.glob('branch*/adapter.safetensors')}:
            raise ValueError('Resume modified a committed adapter')
        all_hashes={str(p):baseline.sha(p) for p in folder.rglob('*') if p.is_file()}

        def cached_only(current,items,path):
            if len(checked_rows(path))!=6:
                raise ValueError('Completed replay would trigger new generation')
            return evaluate(current,items,path)
        replay=run_fixed_root(learner,source,pool,probes,folder,evaluate=cached_only,**arguments)
        if replay!=result or all_hashes!={str(p):baseline.sha(p) for p in folder.rglob('*') if p.is_file()}:
            raise ValueError('Completed replay changed immutable artifacts')
        if adapter_hash(learner)!=initial_hash or learner.updates!=0:
            raise ValueError('Trial driver did not discard trial weights after completion')
        save(OUT/'preflight/fixed_trial_driver_gpu_v1.json',{'status':'passed','at_utc':now(),
            'device':'actual_student_model_GPU','synthetic_roots':1,'real_training_branches':2,
            'updates_per_branch':1,'cold_learner_resume':True,'no_training_repeated_on_resume':True,
            'cached_replay_no_generation':True,'adapters_unchanged_on_resume':True,
            'choice_frozen_before_audit':True,'reset_to_S0_after_trial':True,
            'generation_limit_engineering_only':32,'formal_generation_limit_unchanged':1024,'single_threshold_frozen':True,
            'other_thresholds_evaluated':False,
            'official_test_accessed':False,'scientific_effectiveness_established':False,
            'result_file_sha256':baseline.sha(folder/'trial_result.json'),
            'signature_file_sha256':baseline.sha(OUT/'signatures/fixed_trial_driver_gpu_v1.json')})
        print(read(OUT/'preflight/fixed_trial_driver_gpu_v1.json'),flush=True)


if __name__=='__main__':main()
