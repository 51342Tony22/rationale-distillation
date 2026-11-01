"""Load an engineering adapter through the actual inference directory API.

Uses the two already-exposed reviewed TRAIN roots, never public test items.
Runs only after the real formal-training continuation preflight and GPU lock.
"""
from __future__ import annotations

import gc
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,BASE,read,freeze,save,signature,baseline,gpu,now
from training_checkpoint import latest_checkpoint,restore_checkpoint


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from learning import Learner,LORA,loss,tokenize
    previous=read(OUT/'preflight/formal_training_gpu_quality_v3.json')
    if previous['status']!='passed':raise ValueError('Actual training continuation preflight required')
    previous_signature=OUT/'signatures/formal_training_gpu_quality_v3.json'
    if baseline.sha(previous_signature)!=previous['signature_file_sha256']:
        raise ValueError('Training engineering preflight signature changed')
    for name,checksum in read(previous_signature)['inputs'].items():
        if baseline.sha(ROOT/name)!=checksum:raise ValueError('Training engineering preflight is stale: '+name)
    source=OUT/'preflight/formal_training_gpu_quality_v3/interrupted'
    fixture=read(OUT/'preflight/formal_training_fixture_v1.json')
    adapter=OUT/'preflight/adapter_loading_gpu_quality_v3/inference_adapter'
    adapter.mkdir(parents=True,exist_ok=True)
    weight=source/'adapters.safetensors';destination=adapter/'adapters.safetensors'
    if destination.exists():
        if baseline.sha(destination)!=baseline.sha(weight):raise ValueError('Engineering inference adapter changed')
    else:
        temporary=adapter/'adapters.safetensors.tmp';temporary.write_bytes(weight.read_bytes());temporary.rename(destination)
    freeze(adapter/'adapter_config.json',{'fine_tune_type':'lora','num_layers':16,'lora_parameters':LORA})
    checkpoint=latest_checkpoint(source/'checkpoints')
    if (checkpoint is None or baseline.sha(destination)!=baseline.sha(checkpoint/'adapter.safetensors')
            or read(source/'metadata.json')['adapter_sha256']!=baseline.sha(destination)):
        raise ValueError('Engineering exported adapter differs from the committed final checkpoint')
    with gpu():
        signature('adapter_loading_gpu_quality_v3',[Path(__file__),HERE/'evaluation_inputs.py',HERE/'evaluate.py',
            HERE/'train_selector_quality_v3.py',
            HERE/'learning.py',HERE/'runtime.py',HERE/'runtime_full.py',HERE/'training_checkpoint.py',
            OUT/'preflight/formal_training_gpu_quality_v3.json',previous_signature,
            OUT/'preflight/formal_training_fixture_v1.json',source/'metadata.json',
            checkpoint/'commit.json',destination,adapter/'adapter_config.json',
            ROOT/'experiments/math_distill_v1/config.json'],
            {'purpose':'actual_MLX_inference_adapter_directory_engineering_validation',
             'public_test_used':False,'roots':'already_exposed_reviewed_train0_and2','formal_evaluation_started':False})
        baseline.verify_student(BASE)
        learner=Learner(42);restore_checkpoint(learner,checkpoint)
        expected=[]
        for row in fixture['rows']:
            material=row['epoch_materials'][0]
            # Check actual completion prediction, not an injected answer. This
            # teacher-text prefix is engineering only, never public evaluation.
            sample=tokenize(learner.tokenizer,material['question'],material['completion'])
            targets=sample['target_ids'][:16];ids=sample['prompt_ids']+targets
            value=loss(learner.model,mx.array([ids[:-1]]),mx.array([targets]),len(sample['prompt_ids']))
            mx.eval(value);expected.append(float(value.item()))
        del learner;gc.collect();mx.clear_cache()
        model,tokenizer=load(str(ROOT/BASE['student_dir']),adapter_path=str(adapter),
            tokenizer_config={'local_files_only':True,'trust_remote_code':False})
        model.eval()
        loaded=dict(tree_flatten(model.parameters()));stored=mx.load(str(destination))
        if not stored or any(name not in loaded or not bool(mx.all(loaded[name]==value).item())
                for name,value in stored.items()):
            raise ValueError('Inference API did not restore every saved LoRA tensor exactly')
        actual=[]
        for row in fixture['rows']:
            material=row['epoch_materials'][0]
            sample=tokenize(tokenizer,material['question'],material['completion'])
            targets=sample['target_ids'][:16];ids=sample['prompt_ids']+targets
            value=loss(model,mx.array([ids[:-1]]),mx.array([targets]),len(sample['prompt_ids']))
            mx.eval(value);actual.append(float(value.item()))
        difference=max(abs(a-b) for a,b in zip(expected,actual))
        if difference>1e-6:raise ValueError('Inference loader changed completion NLL: '+str(difference))
        report={'status':'passed','at_utc':now(),'device':'actual_student_GPU',
            'exact_saved_LoRA_tensors_restored':True,'reference_NLL':expected,'inference_NLL':actual,
            'maximum_NLL_difference':difference,'adapter_directory_API_validated':True,
            'already_exposed_train_roots_only':True,'public_test_used':False,'formal_evaluation_started':False,
            'signature_file_sha256':baseline.sha(OUT/'signatures/adapter_loading_gpu_quality_v3.json')}
        save(OUT/'preflight/adapter_loading_gpu_quality_v3.json',report)
        print(report,flush=True)


if __name__=='__main__':main()
