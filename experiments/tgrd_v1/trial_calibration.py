"""Real-model calibration runner, gated by reviewed bootstrap explanation pools.

One invocation evaluates one predeclared update count and all four thresholds.
It records choices before reading audit predictions and never changes a choice
in response to its audit. Development Vaudit is tuning data, not an independent
reported confirmation set. Ten bootstrap roots do not complete Dcal40.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, DATA, BASE, read, freeze, save, append,
                          checked_rows, signature, baseline, digest, gpu,
                          generate, check_time, now)
from explanation_pool import unique
from calibration_cohort import COHORTS, cohort_paths
from trial_selection import summarize, validate_probes
from trial_engine import run_root, THRESHOLDS
from variants import build


def adapter_hash(learner):
    import hashlib
    import numpy as np
    import mlx.core as mx
    values = learner.weights()
    mx.eval(values)
    checksum = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.asarray(value.astype(mx.float32))
        checksum.update(name.encode()); checksum.update(str(array.shape).encode()); checksum.update(array.tobytes())
    return checksum.hexdigest()


def restore_adapter(learner, path):
    import mlx.core as mx
    learner.model.load_weights(str(path),strict=False)
    mx.eval(learner.model.parameters());learner.model.eval()


def evaluate_probes(learner, probes, path, *, generation_limit=1024):
    predictions = checked_rows(path)
    expected = [row['id'] for row in probes]
    if [row['id'] for row in predictions] != expected[:len(predictions)]:
        raise ValueError('Probe resume order mismatch')
    for saved, probe in zip(predictions, probes):
        if (saved['question_sha256'] != probe['question_sha256'] or saved['gold'] != probe['gold']
                or digest(saved['output']) != saved['output_sha256']):
            raise ValueError('Probe output/label changed')
    for probe in probes[len(predictions):]:
        check_time(reserve=15)
        generated = generate(learner.model, learner.tokenizer, probe['question'],limit=generation_limit)
        row = {**probe, **generated, **baseline.score(generated['scoring_text'], probe['gold']),
               'answer_nll': learner.answer_nll(probe['question'], probe['gold'])}
        if not math.isfinite(row['answer_nll']):
            raise ValueError('Nonfinite probe answer NLL')
        predictions.append(append(path, row))
    return summarize(predictions, expected)


def main(updates, seed, cohort='bootstrap10'):
    paths = cohort_paths(cohort); pool_folder = paths.pool_folder
    if updates not in (1,3,5) or seed not in (701,1701):
        raise ValueError('Only declared update counts and trial/repeat seeds are supported')
    if read(OUT/'preflight/learning.json')['status'] != 'passed':
        raise ValueError('Real branch-reset preflight must pass first')
    driver_gate=read(OUT/'preflight/trial_driver_gpu_v1.json')
    if driver_gate['status'] != 'passed':
        raise ValueError('Real-model trial-driver preflight must pass first')
    driver_signature=OUT/'signatures/trial_driver_gpu_v1.json'
    if baseline.sha(driver_signature)!=driver_gate['signature_file_sha256']:
        raise ValueError('Trial-driver preflight signature changed')
    for name,checksum in read(driver_signature)['inputs'].items():
        if baseline.sha(ROOT/name)!=checksum:
            raise ValueError('Trial-driver code/input changed after real-model validation: '+name)
    pools = read(pool_folder/'reviewed_pools.json')
    sources = unique(read(DATA/'cal40.json'), 'source_id')
    bundles = unique(read(paths.variants), 'source_id')
    folder = paths.trials_folder/f'seed{seed}_updates{updates}'
    # All calibration bundles must be present, including fallback/empty roots.
    if (len(pools) != paths.expected_roots or len(bundles) != paths.expected_roots
            or {pool['source_id'] for pool in pools} != set(bundles)):
        raise ValueError('Reviewed pool and calibration root coverage differ')
    signature(f'trial_{paths.signature_prefix}_s{seed}_u{updates}_v1', [Path(__file__), HERE/'trial_selection.py',
        HERE/'trial_engine.py', HERE/'learning.py', HERE/'variants.py', HERE/'runtime_full.py', HERE/'runtime.py',
        HERE/'calibration_cohort.py', pool_folder/'reviewed_pools.json', paths.variants, DATA/'cal40.json',
        OUT/'preflight/learning.json',OUT/'preflight/trial_driver_gpu_v1.json',
        ROOT/'experiments/math_distill_v1/config.json'], {'updates':updates, 'seed':seed, 'thresholds':THRESHOLDS,
        'development_only':True, 'audit_role':'hyperparameter_development_not_independent_confirmation',
        'teacher_prefix_in_probe':False, 'variant_labels_in_gradients':False})
    with gpu():
        from learning import Learner
        baseline.verify_student(BASE)
        learner = Learner(seed)
        initial_hash = adapter_hash(learner)
        for pool in pools:
            sid = pool['source_id']; source = sources[sid]
            probes = validate_probes(bundles[sid], source, build)
            root_folder = folder/sid.replace(':','_')
            result=run_root(learner,source,pool,probes,root_folder,seed=seed,updates=updates,
                initial_hash=initial_hash,weight_hash=adapter_hash,restore=restore_adapter,
                evaluate=evaluate_probes)
            print(sid, [(r['threshold'],r['selected_id'],r['Vaudit_accuracy_gain'])
                        for r in result['configurations']], flush=True)
        save(folder/'status.json', {'at_utc':now(), 'roots':len(pools), 'updates':updates, 'seed':seed,
            'cohort':cohort, 'complete_for_declared_cohort':True,
            'Dcal40_complete':len(pools)==40,
            'public_test_used':False, 'formal_training_started':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--updates',type=int,choices=[1,3,5],required=True)
    parser.add_argument('--seed',type=int,choices=[701,1701],default=701)
    parser.add_argument('--cohort',choices=COHORTS,default='bootstrap10')
    args = parser.parse_args(); main(args.updates,args.seed,args.cohort)
