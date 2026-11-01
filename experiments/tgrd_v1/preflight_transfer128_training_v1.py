"""Actual student engineering for new finite128 entry; synthetic, not science.

Only the two already exposed engineering roots are used. Same four-epoch
engine, both material routes, cold loading and inference directory API.
"""
import argparse
import copy
import gc
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,BASE,read,freeze,save,baseline,signature,gpu,now,digest
from train_transfer128_v1 import CPU,verify_inputs,verify_initialization
from training_engine import run_training
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from trial_calibration import adapter_hash

REPORT=OUT/'preflight/transfer128_training_gpu_v1.json'
SIGNED=OUT/'signatures/transfer128_training_gpu_v1.json'
FOLDER=OUT/'preflight/transfer128_training_gpu_v1'


def fixtures(source):
    bundles={}
    for group in ('M7','M6'):
        bundle=copy.deepcopy(source);bundle['condition']=group
        bundle.update(protocol='engineering_only',stage='finite_transfer128',
            purpose='synthetic_engineering_not_qualified_scientific_materials',official_test_used=False)
        if group=='M6':
            for row in bundle['rows']:
                for epoch,value in ((1,2),(3,4)):
                    m=row['epoch_materials'][epoch]
                    m.update(question=f'A box contains {value} pencils. How many pencils does it contain?',
                        completion=f'The box contains {value} pencils. The answer is {value}.',
                        kind='verified_Vsel_variant',split='select',variant_id=f'{row["source_id"]}:select:engineering:{epoch}')
                    m['question_sha256']=digest(m['question']);m['completion_sha256']=digest(m['completion'])
        bundles[group]=bundle
    return bundles


def main():
    if REPORT.exists():
        report=read(REPORT)
        if report.get('signature_file_sha256')!=baseline.sha(SIGNED):raise ValueError('Engineering signature changed')
        verify_inputs(read(SIGNED));verify_inputs({'inputs':report['artifact_inputs']})
        return report
    cpu=read(CPU)
    if cpu.get('status')!='passed' or cpu.get('device')!='explicit_MLX_CPU':
        raise ValueError('New versioned CPU integration checks required before GPU')
    paths=verify_inputs(cpu)
    source=OUT/'preflight/formal_training_fixture_v1.json'
    paths += [source,CPU,Path(__file__),HERE/'train_transfer128_v1.py',HERE/'learning.py',
        HERE/'training_engine.py',HERE/'training_checkpoint.py',HERE/'training_plan.py',
        HERE/'runtime.py',HERE/'runtime_full.py',HERE/'trial_calibration.py',
        OUT/'training/selector_first_transfer128_seed42_v1/initializations/seed42.json']
    with gpu():
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from learning import Learner,LORA,tokenize,loss
        baseline.verify_student(BASE)
        signature('transfer128_training_gpu_v1',paths,dict(purpose='synthetic_engineering_not_science',
            public_test_used=False,conditions=['M7','M6'],seed=42,updates_per_run=4))
        bundles=fixtures(read(source));results={}
        args=dict(seed=42,tokenize=tokenize,weight_hash=adapter_hash,save_checkpoint=save_checkpoint,
            restore_checkpoint=restore_checkpoint,latest_checkpoint=latest_checkpoint,checkpoint_interval=1)
        for group,bundle in bundles.items():
            folder=FOLDER/group
            freeze(folder/'fixture.json',bundle)
            learner=Learner(42)
            verify_initialization(learner,read(paths[-1]))
            reference=run_training(learner,bundle,folder/'reference',**args)
            restore_checkpoint(learner,latest_checkpoint(folder/'reference/checkpoints'))
            weight=adapter_hash(learner)
            optimizer={k:mx.array(v) for k,v in tree_flatten(learner.optimizer.state)}
            random=[mx.array(v) for v in mx.random.state];mx.eval(optimizer,random)
            del learner;gc.collect();mx.clear_cache()
            learner=Learner(42);verify_initialization(learner,read(paths[-1]))
            marker=folder/'interruption.json'
            if not marker.exists():
                def stop(step):
                    if step['update']==2:raise InterruptedError('engineering interruption after committed2')
                try:run_training(learner,bundle,folder/'resumed',after_update=stop,**args)
                except InterruptedError:freeze(marker,{'committed_interruption':2})
                else:raise ValueError('Expected controlled interruption')
                del learner;gc.collect();mx.clear_cache();learner=Learner(42)
            resumed=run_training(learner,bundle,folder/'resumed',**args)
            restore_checkpoint(learner,latest_checkpoint(folder/'resumed/checkpoints'))
            current=dict(tree_flatten(learner.optimizer.state))
            if (adapter_hash(learner)!=weight or set(current)!=set(optimizer)
                    or any(not bool(mx.all(current[k]==v).item()) for k,v in optimizer.items())
                    or len(mx.random.state)!=len(random)
                    or any(not bool(mx.all(a==b).item()) for a,b in zip(mx.random.state,random))
                    or reference['updates']!=4 or resumed['updates']!=4
                    or reference['root_exposures']!=resumed['root_exposures']):
                raise ValueError('Actual student finite128 route cold resume differs')
            material=bundle['rows'][0]['epoch_materials'][3]
            def nll(model,tokenizer):
                sample=tokenize(tokenizer,material['question'],material['completion'])
                targets=sample['target_ids'][:16];ids=sample['prompt_ids']+targets
                v=loss(model,mx.array([ids[:-1]]),mx.array([targets]),len(sample['prompt_ids']))
                mx.eval(v);return float(v.item())
            expected=nll(learner.model,learner.tokenizer)
            del learner,optimizer,current,random;gc.collect();mx.clear_cache()
            adapter=folder/'resumed'
            freeze(adapter/'adapter_config.json',dict(fine_tune_type='lora',num_layers=16,lora_parameters=LORA))
            model,tokenizer=load(str(ROOT/BASE['student_dir']),adapter_path=str(adapter),
                tokenizer_config={'local_files_only':True,'trust_remote_code':False})
            model.eval();stored=mx.load(str(adapter/'adapters.safetensors'));loaded=dict(tree_flatten(model.parameters()))
            if not stored or any(k not in loaded or not bool(mx.all(v==loaded[k]).item()) for k,v in stored.items()):
                raise ValueError('Actual inference API failed to restore every LoRA tensor')
            actual=nll(model,tokenizer)
            if abs(actual-expected)>1e-6:raise ValueError('Inference completion NLL differs')
            results[group]=dict(updates=4,cold_resume_exact_weights_optimizer_PRNG=True,
                reference_NLL=expected,inference_NLL=actual,maximum_NLL_difference=abs(actual-expected),
                exact_saved_LoRA_tensors_restored=True,adapter_directory_API_validated=True)
            del model,tokenizer,stored,loaded;gc.collect();mx.clear_cache()
        files=[p for p in FOLDER.rglob('*') if p.is_file()]
        value=dict(status='passed',at_utc=now(),device='actual_student_GPU',
            synthetic_engineering_only=True,scientific_results=False,public_test_used=False,
            conditions=results,signature_file_sha256=baseline.sha(SIGNED),
            artifact_inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in files})
        save(REPORT,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--if-supported',action='store_true');a=p.parse_args()
    if a.if_supported:
        from transfer128_production_v1 import load_configuration
        if not load_configuration()['scientific_trial_enabled']:
            print({'status':'unsupported_no_expansion','GPU_engineering_executed':False},flush=True)
            raise SystemExit(0)
    v=main();print({k:v[k] for k in ('status','device','synthetic_engineering_only','scientific_results')},flush=True)
