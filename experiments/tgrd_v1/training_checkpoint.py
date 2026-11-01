"""Atomic adapter + optimizer + global MLX PRNG snapshots without pickle."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from runtime_full import read,freeze,baseline


def pack_tree(value,arrays):
    import mlx.core as mx
    if isinstance(value,mx.array):
        key=f'array{len(arrays)}';arrays[key]=value
        return {'type':'array','key':key}
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('Only string state keys supported')
        return {'type':'dict','items':[[k,pack_tree(v,arrays)] for k,v in value.items()]}
    if isinstance(value,(tuple,list)):
        return {'type':'tuple' if isinstance(value,tuple) else 'list',
                'items':[pack_tree(v,arrays) for v in value]}
    if value is None or type(value) in (str,bool,int,float):
        if type(value) is float and not math.isfinite(value):raise ValueError('Nonfinite state scalar')
        return {'type':'scalar','value':value}
    raise ValueError('Unsupported state leaf type: '+type(value).__name__)


def unpack_tree(schema,arrays):
    kind=schema['type']
    if kind=='array':return arrays[schema['key']]
    if kind=='dict':return {key:unpack_tree(v,arrays) for key,v in schema['items']}
    if kind in ('list','tuple'):
        items=[unpack_tree(v,arrays) for v in schema['items']]
        return tuple(items) if kind=='tuple' else items
    if kind=='scalar':return schema['value']
    raise ValueError('Unsupported state schema')


def verify_checkpoint(folder):
    folder=Path(folder);manifest=read(folder/'commit.json')
    if manifest['schema']!='tgrd_training_checkpoint_v1':
        raise ValueError('Unknown training checkpoint format')
    for name,checksum in manifest['files'].items():
        if name not in ('adapter.safetensors','state.safetensors','state_structure.json'):
            raise ValueError('Unexpected checkpoint file')
        if baseline.sha(folder/name)!=checksum:
            raise ValueError('Training checkpoint file changed: '+name)
    if set(manifest['files'])!={'adapter.safetensors','state.safetensors','state_structure.json'}:
        raise ValueError('Incomplete checkpoint commit')
    return manifest


def save_checkpoint(learner,folder,progress):
    import mlx.core as mx
    folder=Path(folder)
    if folder.exists():raise FileExistsError('Committed checkpoints are immutable: '+str(folder))
    folder.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.uncommitted_'+folder.name+'_',dir=folder.parent))
    # _RandomState is read-only as a container. Snapshot its array keys; restore
    # in-place below, never rebind mx.random.state or replace it with a list.
    state={'optimizer':learner.optimizer.state,'mlx_random':[mx.array(v) for v in mx.random.state]}
    arrays={};schema=pack_tree(state,arrays)
    mx.eval(arrays,learner.weights())
    learner.save(staging/'adapter.safetensors')
    mx.save_safetensors(str(staging/'state.safetensors'),arrays)
    freeze(staging/'state_structure.json',schema)
    files={p.name:baseline.sha(p) for p in staging.iterdir()}
    freeze(staging/'commit.json',{'schema':'tgrd_training_checkpoint_v1','progress':progress,
        'learner_updates':learner.updates,'files':files})
    for p in staging.iterdir():
        with p.open('rb') as stream:os.fsync(stream.fileno())
    # The commit becomes visible with the entire directory in one rename.
    staging.rename(folder)
    return verify_checkpoint(folder)


def restore_checkpoint(learner,folder):
    import mlx.core as mx
    folder=Path(folder);manifest=verify_checkpoint(folder)
    state=unpack_tree(read(folder/'state_structure.json'),mx.load(str(folder/'state.safetensors')))
    learner.model.load_weights(str(folder/'adapter.safetensors'),strict=False)
    learner.optimizer.state=state['optimizer']
    learner.updates=manifest['learner_updates']
    if len(state['mlx_random'])!=len(mx.random.state):
        raise ValueError('PRNG state shape differs from the saved MLX version')
    # Finish lazy parameter/optimizer loads before restoring the exact PRNG key.
    mx.eval(learner.model.parameters(),learner.optimizer.state)
    for destination,saved in zip(mx.random.state,state['mlx_random']):
        if destination.shape!=saved.shape or destination.dtype!=saved.dtype:
            raise ValueError('PRNG key shape/dtype mismatch')
        destination[:]=saved
    mx.eval(list(mx.random.state))
    learner.model.eval()
    return manifest['progress']


def latest_checkpoint(folder):
    folder=Path(folder)
    candidates=sorted(p for p in folder.glob('update_*') if p.is_dir() and (p/'commit.json').exists())
    if not candidates:return None
    latest=candidates[-1]
    # Never silently fall back from a corrupt newest committed checkpoint.
    verify_checkpoint(latest)
    return latest
