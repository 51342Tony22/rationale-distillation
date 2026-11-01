"""Root-balanced batch updates with committed full-state continuation."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from runtime_full import read,freeze,save,append,digest,baseline,now,check_time
from training_plan import validate_materials,schedule,CHECKPOINT_INTERVAL
from training_checkpoint import verify_checkpoint


def run_training(learner,bundle,folder,*,seed,tokenize,weight_hash,
                 save_checkpoint,restore_checkpoint,latest_checkpoint,
                 checkpoint_interval=CHECKPOINT_INTERVAL,after_update=None):
    rows=validate_materials(bundle);plan=schedule(len(rows),seed);folder=Path(folder)
    if checkpoint_interval<1:raise ValueError('Positive checkpoint interval required')
    if learner.updates!=0:raise ValueError('Formal training must start with a fresh learner before restore')
    initial_hash=weight_hash(learner)
    inputs={'schema':'tgrd_training_run_v1','condition':bundle['condition'],'seed':seed,
        'materials_sha256':digest(json.dumps(bundle,sort_keys=True)),
        'ordered_root_ids':[r['source_id'] for r in rows],
        'initial_adapter_sha256':initial_hash,'schedule_sha256':digest(json.dumps(plan,sort_keys=True)),
        'updates':len(plan),'epochs':4,'checkpoint_interval':checkpoint_interval,
        'equal_budget':'one_root_exposure_per_epoch;five_microbatches_per_full_update',
        'primary_checkpoint':'last_epoch4_not_selected_on_test'}
    freeze(folder/'inputs.json',inputs);input_sha=baseline.sha(folder/'inputs.json')
    if (folder/'metadata.json').exists():
        done=read(folder/'metadata.json')
        if (done['input_file_sha256']!=input_sha or done['completed'] is not True
                or done['updates']!=len(plan) or done['adapter_sha256']!=baseline.sha(folder/'adapters.safetensors')):
            raise ValueError('Completed training provenance mismatch')
        return done
    checkpoints=folder/'checkpoints'
    latest=latest_checkpoint(checkpoints)
    if latest is None:
        progress={'update':0,'input_file_sha256':input_sha,'tokens':0,'root_exposures':0,
                  'epoch_end':False,'epoch':0,'optimization_seconds_committed':0.}
        latest=checkpoints/'update_00000000'
        save_checkpoint(learner,latest,progress)
    else:
        progress=restore_checkpoint(learner,latest)
        if (progress['input_file_sha256']!=input_sha or progress['update']!=learner.updates
                or not 0<=progress['update']<=len(plan)):
            raise ValueError('Checkpoint cursor/materials mismatch')
    attempt=now()
    append(folder/'events.jsonl',{'event':'start_or_resume','at_utc':attempt,
        'from_update':progress['update'],'retains_uncommitted_prior_attempt_logs':True})

    def export_epoch(checkpoint,point):
        if point['epoch_end']:
            freeze(folder/'epochs'/f'epoch{point["epoch"]}.json',{
                'epoch':point['epoch'],'update':point['update'],
                'checkpoint_directory':str(Path(checkpoint).relative_to(folder)),
                'adapter_sha256':baseline.sha(Path(checkpoint)/'adapter.safetensors'),
                'primary':point['epoch']==4})
    # Repair an interrupted epoch-index export from already committed states.
    for checkpoint in sorted(checkpoints.glob('update_*')):
        if (checkpoint/'commit.json').exists():
            committed=verify_checkpoint(checkpoint)
            if committed['progress']['input_file_sha256']!=input_sha:
                raise ValueError('Historical checkpoint belongs to different training inputs')
            export_epoch(checkpoint,committed['progress'])
    for step in plan[progress['update']:]:
        check_time(reserve=15)
        materials=[rows[i]['epoch_materials'][step['epoch']-1] for i in step['root_indices']]
        samples=[tokenize(learner.tokenizer,m['question'],m['completion']) for m in materials]
        start=time.perf_counter();metrics=learner.update_batch(samples);seconds=time.perf_counter()-start
        if learner.updates!=step['update'] or metrics['updates']!=step['update']:
            raise ValueError('Optimizer update count differs from root schedule')
        progress={'update':step['update'],'epoch':step['epoch'],'epoch_end':step['epoch_end'],
            'input_file_sha256':input_sha,'tokens':progress['tokens']+metrics['tokens'],
            'root_exposures':progress['root_exposures']+len(samples),
            'optimization_seconds_committed':progress['optimization_seconds_committed']+seconds}
        append(folder/'events.jsonl',{'event':'optimizer_update','attempt':attempt,
            'update':step['update'],'epoch':step['epoch'],'root_ids':[rows[i]['source_id'] for i in step['root_indices']],
            'material_kinds':[m['kind'] for m in materials],
            'question_sha256':[m['question_sha256'] for m in materials],
            'loss':metrics['loss'],'tokens':metrics['tokens'],'seconds':seconds,
            'may_be_replayed_if_later_than_last_committed_checkpoint':True})
        if step['epoch_end'] or step['update']%checkpoint_interval==0:
            latest=checkpoints/f'update_{step["update"]:08d}'
            save_checkpoint(learner,latest,progress);export_epoch(latest,progress)
            save(folder/'progress.json',{**progress,'at_utc':now(),'checkpoint':str(latest.relative_to(folder))})
        if after_update is not None:after_update(step)
    if progress['update']!=len(plan) or progress['root_exposures']!=4*len(rows):
        raise ValueError('Final root exposures/update budget incomplete')
    final=folder/'adapters.safetensors';source=Path(latest)/'adapter.safetensors'
    if final.exists():
        if baseline.sha(final)!=baseline.sha(source):
            raise ValueError('Uncommitted final adapter differs from the final checkpoint')
    else:
        temporary=folder/'adapters.safetensors.tmp'
        shutil.copyfile(source,temporary);temporary.rename(final)
    done={'completed':True,'condition':bundle['condition'],'seed':seed,'epochs':4,
        'accepted_root_count':len(rows),'candidate_root_count':7473,'updates':len(plan),
        'root_exposures':progress['root_exposures'],'supervised_tokens_including_eos':progress['tokens'],
        'optimization_seconds_committed':progress['optimization_seconds_committed'],
        'extra_replayed_compute_retained_in_events':True,
        'input_file_sha256':input_sha,'initial_adapter_sha256':initial_hash,
        'adapter_sha256':baseline.sha(final),'primary_checkpoint':'last_epoch4',
        'public_test_used_to_select_checkpoint':False}
    freeze(folder/'metadata.json',done)
    return done
