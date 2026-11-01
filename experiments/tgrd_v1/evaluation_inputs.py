"""Verify the full evaluation roster and final adapter directories without test data."""
from __future__ import annotations

from pathlib import Path
from runtime_full import ROOT,OUT,read,baseline
from training_plan import CONDITIONS,SEEDS,REFERENCE_SEED,PROTOCOL
from training_checkpoint import verify_checkpoint
from learning import LORA

REFERENCES=('student_base','teacher_reference')


def validate_request(condition,seed,adapter=None,*,root=ROOT,out=OUT):
    root=Path(root);out=Path(out);gate_path=out/'gates/public_evaluation_frozen.json';gate=read(gate_path)
    if (gate.get('enabled') is not True or gate.get('primary_dataset')!='GSM8K'
            or gate.get('protocol')!=PROTOCOL
            or any(gate.get(key) is not True for key in ('protocol_frozen_before_public_test',
                'development_accepted','full_teacher_quality_review_complete','formal_training_complete'))):
        raise ValueError('Development, quality and the complete evaluation protocol must be frozen')
    preflight_path=out/'preflight/adapter_loading_gpu_v1.json';preflight=read(preflight_path)
    preflight_signature=out/'signatures/adapter_loading_gpu_v1.json'
    if (preflight['status']!='passed' or baseline.sha(preflight_signature)!=preflight['signature_file_sha256']):
        raise ValueError('Actual inference-adapter loading preflight required')
    for name,checksum in read(preflight_signature)['inputs'].items():
        if baseline.sha(root/name)!=checksum:raise ValueError('Inference loading preflight is stale: '+name)
    groups=gate['conditions']
    if (len(set(groups))!=len(groups) or not set(CONDITIONS)<=set(groups)
            or set(groups)-set(CONDITIONS)-set(REFERENCES)
            or set(gate['seeds'])!=set(groups) or condition not in groups):
        raise ValueError('Public evaluation must retain all five formal groups')
    for group in groups:
        expected=list(SEEDS) if group in CONDITIONS else [REFERENCE_SEED]
        if gate['seeds'][group]!=expected:raise ValueError('Evaluation group/seed roster changed')
    if type(seed) is not int or seed not in gate['seeds'][condition]:raise ValueError('Undeclared evaluation seed')
    training_path=out/'gates/final_training_frozen.json'
    if baseline.sha(training_path)!=gate['training_gate_sha256']:
        raise ValueError('Final training protocol changed after evaluation freeze')
    training=read(training_path)
    if (training.get('enabled') is not True or training.get('full_teacher_quality_review_complete') is not True
            or training.get('protocol')!=PROTOCOL
            or training.get('development_accepted') is not True or set(training['conditions'])!=set(CONDITIONS)
            or training['seeds']!=list(SEEDS) or not training['common_root_ids']):
        raise ValueError('Final training protocol lacks complete quality/roster acceptance')
    artifacts=gate['artifacts']
    count=len(training['common_root_ids']);updates=4*((count+4)//5)
    if set(artifacts)!=set(CONDITIONS):raise ValueError('All declared formal groups and seeds must be ready before public evaluation')
    for group in CONDITIONS:
        if set(artifacts[group])!={str(s) for s in SEEDS}:raise ValueError('Missing formal training artifact seed')
        for s in SEEDS:
            entry=artifacts[group][str(s)];directory=(root/entry['adapter_directory']).resolve()
            canonical=(out/'training'/group/f'seed{s}').resolve()
            if directory!=canonical or not directory.is_dir() or not entry.get('files'):
                raise ValueError('Missing or mislabelled trained adapter directory')
            # Even a reference run waits for every declared formal run. Detailed
            # hashes and final-checkpoint identity are checked for each used run.
            if any(not (directory/name).is_file() for name in
                    ('adapters.safetensors','adapter_config.json','metadata.json','inputs.json','epochs/epoch4.json')):
                raise ValueError('A declared formal run is not complete')
            meta=read(directory/'metadata.json')
            if (meta.get('completed') is not True or meta['condition']!=group or meta['seed']!=s
                    or meta['epochs']!=4 or meta['accepted_root_count']!=count or meta['updates']!=updates):
                raise ValueError('A declared formal run is unfinished or belongs to another group/seed')
    if condition in REFERENCES:
        if adapter is not None:raise ValueError('Reference evaluation cannot silently load a trained adapter')
        return {'adapter_directory':None,'signature_paths':[gate_path,training_path,preflight_path,preflight_signature],
            'condition':condition,'seed':seed,'training_artifact':None}
    entry=artifacts[condition][str(seed)];directory=(root/entry['adapter_directory']).resolve()
    if adapter is not None and Path(adapter).resolve()!=directory:
        raise ValueError('Adapter argument differs from this frozen group/seed directory')
    epoch=read(directory/'epochs/epoch4.json');checkpoint=(directory/epoch['checkpoint_directory']).resolve()
    try:checkpoint.relative_to(directory/'checkpoints')
    except ValueError:raise ValueError('Final epoch checkpoint is outside this run')
    required={'adapters.safetensors','adapter_config.json','metadata.json','inputs.json','epochs/epoch4.json',
        str((checkpoint/'commit.json').relative_to(directory))}
    if set(entry['files'])!=required:raise ValueError('Frozen artifact needs exact weights/config/provenance/final-commit files')
    for name,checksum in entry['files'].items():
        if baseline.sha(directory/name)!=checksum:raise ValueError('Frozen training artifact changed: '+name)
    meta=read(directory/'metadata.json');inputs=read(directory/'inputs.json');commit=verify_checkpoint(checkpoint)
    config=read(directory/'adapter_config.json')
    if config!={'fine_tune_type':'lora','num_layers':16,'lora_parameters':LORA}:
        raise ValueError('Adapter configuration differs from the validated formal LoRA')
    weights_sha=baseline.sha(directory/'adapters.safetensors')
    if (meta.get('completed') is not True or meta['condition']!=condition or meta['seed']!=seed
            or meta['primary_checkpoint']!='last_epoch4' or meta['epochs']!=4 or meta['accepted_root_count']!=count
            or meta['candidate_root_count']!=7473 or meta['updates']!=updates or meta['root_exposures']!=4*count
            or meta['public_test_used_to_select_checkpoint'] is not False or meta['adapter_sha256']!=weights_sha
            or meta['input_file_sha256']!=baseline.sha(directory/'inputs.json')
            or inputs['condition']!=condition or inputs['seed']!=seed
            or inputs['ordered_root_ids']!=training['common_root_ids']
            or inputs['primary_checkpoint']!='last_epoch4' or inputs['updates']!=updates
            or epoch['epoch']!=4 or epoch['update']!=updates or epoch['primary'] is not True
            or epoch['adapter_sha256']!=weights_sha or baseline.sha(checkpoint/'adapter.safetensors')!=weights_sha
            or commit['learner_updates']!=updates or commit['progress']['update']!=updates
            or commit['progress']['epoch']!=4 or commit['progress']['epoch_end'] is not True
            or commit['progress']['input_file_sha256']!=meta['input_file_sha256']
            or commit['progress']['root_exposures']!=4*count):
        raise ValueError('Training artifact is not this group/seed complete common-root epoch4 result')
    return {'adapter_directory':directory,'signature_paths':[gate_path,training_path,preflight_path,preflight_signature,
        *[directory/name for name in sorted(required)]], 'condition':condition,'seed':seed,
        'training_artifact':{'condition':condition,'seed':seed,'adapter_sha256':weights_sha,
            'accepted_root_count':count,'primary_checkpoint':'last_epoch4'}}
