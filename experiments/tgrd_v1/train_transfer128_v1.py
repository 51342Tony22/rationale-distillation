"""Finite128 formal training: actual production replay before every admission.

No public test loading; unsupported development creates no training or materials.
The existing training engine/checkpoint/actual GPU engineering sources stay intact.
"""
from pathlib import Path
import argparse
from runtime_full import ROOT,HERE,OUT,BASE,read,freeze,save,baseline,signature,gpu,now
from budgeted_materials import PROTOCOL
from train_selector_quality_v3 import verify_inputs,validate_authority,verify_engineering
from training_plan import validate_materials
from training_engine import run_training
from training_checkpoint import save_checkpoint,restore_checkpoint,latest_checkpoint
from trial_calibration import adapter_hash

CONDITIONS=('M7','M6')
GATE=OUT/'gates'/(PROTOCOL+'_transfer128_training_v1.json')
CPU=OUT/'preflight/transfer128_training_evaluation_cpu_v1.json'


def verify_transfer_engineering():
    report_path=OUT/'preflight/transfer128_training_gpu_v1.json'
    signed_path=OUT/'signatures/transfer128_training_gpu_v1.json'
    report=read(report_path)
    if (report.get('status')!='passed' or report.get('device')!='actual_student_GPU'
            or report.get('synthetic_engineering_only') is not True
            or report.get('scientific_results') is not False or report.get('public_test_used') is not False
            or report.get('signature_file_sha256')!=baseline.sha(signed_path)
            or set(report.get('conditions',{}))!=set(CONDITIONS)):
        raise ValueError('New finite128 actual student GPU route evidence required')
    for entry in report['conditions'].values():
        if (entry.get('updates')!=4 or entry.get('cold_resume_exact_weights_optimizer_PRNG') is not True
                or entry.get('exact_saved_LoRA_tensors_restored') is not True
                or entry.get('adapter_directory_API_validated') is not True
                or type(entry.get('maximum_NLL_difference')) not in (int,float)
                or not 0<=entry['maximum_NLL_difference']<=1e-6):
            raise ValueError('Actual cold resume and inference equality required for M7 and M6')
    signed=read(signed_path)
    if 'experiments/tgrd_v1/train_transfer128_v1.py' not in signed.get('inputs',{}):
        raise ValueError('GPU engineering must bind this new formal entry')
    return [report_path,signed_path,*verify_inputs(signed),
        *verify_inputs({'inputs':report['artifact_inputs']})]


def evidence():
    from transfer128_production_v1 import load_configuration,CONFIGURATION,FOLDER as PRODUCTION
    from transfer128_materials_v1 import prepare,FOLDER
    from evaluation_selector_quality_v3 import collect_training
    config=load_configuration()
    if config['scientific_trial_enabled'] is not True:
        raise ValueError('Unsupported development cannot admit G4/G5 training')
    selector,artifacts,paths=collect_training()
    manifest=prepare()  # reconstructs genuine128 results and all exact materials
    paths += verify_inputs(manifest)
    ids=selector['common_root_ids'];bundles={}
    if (len(ids)!=7179 or manifest.get('schema')!='tgrd_transfer128_materials_quality_v3_v1'
            or manifest.get('protocol')!=PROTOCOL or manifest.get('stage')!='finite_transfer128'
            or manifest.get('explanation_quality_version')!='v3'
            or set(manifest.get('materials',{}))!=set(CONDITIONS)
            or manifest.get('public_test_used') is not False
            or manifest.get('audit_used_to_choose_material') is not False
            or manifest.get('coverage',{}).get('G4_exact_G3_outside128') is not True):
        raise ValueError('Complete actual finite128 materials required')
    for group in CONDITIONS:
        entry=manifest['materials'][group];path=ROOT/entry['path'];bundle=read(path)
        if (baseline.sha(path)!=entry['sha256'] or bundle.get('condition')!=group
                or bundle.get('protocol')!=PROTOCOL or bundle.get('stage')!='finite_transfer128'
                or bundle.get('development_preview_only') is not False):
            raise ValueError('Frozen finite128 material entry changed')
        validate_materials(bundle,ids);bundles[group]=bundle;paths.append(path)
    validate_authority(read(OUT/'transfer128_amendment_v1.json'))
    paths += verify_engineering()
    paths += verify_transfer_engineering()
    cpu=read(CPU)
    if (cpu.get('status')!='passed' or cpu.get('device')!='explicit_MLX_CPU'
            or cpu.get('scientific_results') is not False):
        raise ValueError('Versioned CPU integration evidence required')
    paths += verify_inputs(cpu)
    paths += [CPU,CONFIGURATION,PRODUCTION/'actual_replay_v1.json',
        FOLDER/'materials_manifest_v1.json',Path(__file__),HERE/'transfer128_materials_v1.py',
        HERE/'transfer128_production_v1.py',OUT/'transfer128_amendment_v1.json']
    return bundles,ids,artifacts,list(dict.fromkeys(paths))


def gate_fields(ids):
    return dict(schema='tgrd_transfer128_training_gate_v1',protocol=PROTOCOL,
        stage='finite_transfer128',enabled=True,conditions=list(CONDITIONS),seed=42,
        epochs=4,common_root_ids=ids,primary_checkpoint='last_epoch4',
        public_test_used=False,public_evaluation_enabled=False)


def freeze_gate():
    bundles,ids,artifacts,paths=evidence()
    if not GATE.exists() and any((OUT/'evaluation'/PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot first freeze transfer training after public predictions')
    value={**gate_fields(ids),'selector_final_artifacts':artifacts,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    freeze(GATE,value);return value


def validate_gate(condition,seed=42):
    if condition not in CONDITIONS or type(seed) is not int or seed!=42:
        raise ValueError('Only declared finite128 M7/M6 seed42 training is authorized')
    gate=read(GATE);paths=verify_inputs(gate)
    bundles,ids,artifacts,actual=evidence()
    if (any(gate.get(k)!=v for k,v in gate_fields(ids).items())
            or gate.get('selector_final_artifacts')!=artifacts or not set(actual)<=set(paths)):
        raise ValueError('Training gate differs from actual production/material evidence')
    return bundles[condition],list(dict.fromkeys([GATE,*paths]))


def verify_initialization(learner,initialization):
    from learning import LORA
    if initialization!={'protocol':PROTOCOL,'seed':42,
            'initial_adapter_sha256':adapter_hash(learner),'student_revision':BASE['student_revision'],
            'LoRA':LORA,'layers':16} or learner.updates!=0:
        raise ValueError('Formal42 fresh S0 must exactly match G1-G3; trial701 S0 cannot substitute')


def train(condition,seed=42):
    bundle,paths=validate_gate(condition,seed)
    init=OUT/'training'/PROTOCOL/'initializations/seed42.json'
    paths += [init,Path(__file__),HERE/'training_plan.py',HERE/'training_engine.py',
        HERE/'training_checkpoint.py',HERE/'learning.py',HERE/'trial_calibration.py',
        HERE/'runtime.py',HERE/'runtime_full.py']
    signature(f'{PROTOCOL}_training_{condition}_s{seed}',paths,
        dict(protocol=PROTOCOL,stage='finite_transfer128',condition=condition,seed=seed,
             primary_checkpoint='last_epoch4',public_test_used=False))
    folder=OUT/'training'/PROTOCOL/condition/f'seed{seed}'
    with gpu():
        from learning import Learner,LORA,tokenize
        baseline.verify_student(BASE);learner=Learner(seed)
        verify_initialization(learner,read(init))
        freeze(folder/'adapter_config.json',dict(fine_tune_type='lora',num_layers=16,lora_parameters=LORA))
        meta=run_training(learner,bundle,folder,seed=seed,tokenize=tokenize,weight_hash=adapter_hash,
            save_checkpoint=save_checkpoint,restore_checkpoint=restore_checkpoint,latest_checkpoint=latest_checkpoint)
        save(folder/'completion_status.json',{'at_utc':now(),**meta});return meta


def advance(condition):
    """Queue adapter: a negative grid records unsupported, never fake training."""
    from transfer128_production_v1 import load_configuration,run
    config=load_configuration()
    if not config['scientific_trial_enabled']:
        run();return {'status':'unsupported_no_expansion','condition':condition,
            'formal_training_executed':False,'fixed128_trials_complete':False}
    freeze_gate();return train(condition)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze-gate','train','advance'))
    p.add_argument('--condition',choices=CONDITIONS);a=p.parse_args()
    if a.command=='freeze-gate':v=freeze_gate();print({'enabled':v['enabled'],'conditions':v['conditions']},flush=True)
    else:
        if a.condition is None:p.error('--condition required')
        print(train(a.condition) if a.command=='train' else advance(a.condition),flush=True)
