"""One frozen configuration for the finite128 confirmation trial.

This new state machine leaves the signed four-threshold development engine
unchanged. Callers must verify the real grid-derived configuration and exact
reviewed probe bundle before entering. This module alone grants no GPU or
training/test permission. Audit results never change the Vsel decision.
"""
from pathlib import Path
import json
import re
from runtime_full import read, freeze, digest, baseline, now
from trial_engine import THRESHOLDS, preserve_orphan
from trial_selection import choose


def run_fixed_root(learner, source, pool, probes, folder, *, seed, updates,
             initial_hash, weight_hash, restore, evaluate, threshold, configuration_sha256):
    if (type(seed) is not int or seed != 701 or type(updates) is not int or updates not in (1, 3, 5)
            or type(threshold) not in (int, float) or threshold not in THRESHOLDS
            or not isinstance(configuration_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', configuration_sha256)):
        raise ValueError('A single declared development-selected configuration hash is required')
    folder=Path(folder);sid=source['source_id'];candidates=pool['candidates']
    ids=[c['candidate_id'] for c in candidates]
    if (not 1<=len(candidates)<=5 or ids[0]!='e0' or len(set(ids))!=len(ids)
            or source['question_sha256']!=digest(source['question'])
            or pool['source_id']!=sid or pool['question_sha256']!=source['question_sha256']):
        raise ValueError('Invalid source/pool identity or candidate ordering')
    if (pool['teacher_output_sha256']!=candidates[0]['completion_sha256']
            or candidates[0]['student_tokens']!=0):
        raise ValueError('e0 must be the exact frozen ordinary teacher CoT')
    for c in candidates:
        if digest(c['completion'])!=c['completion_sha256']:
            raise ValueError('Frozen training completion changed')
    if set(probes)!={'select','audit'}:
        raise ValueError('Expected frozen selection and audit probes')
    for split in ('select','audit'):
        if len(probes[split])!=6 or len({p['id'] for p in probes[split]})!=6:
            raise ValueError('Six distinct frozen probes required per split')
    if set(p['id'] for p in probes['select']) & set(p['id'] for p in probes['audit']):
        raise ValueError('Selection and audit probes must be disjoint')
    # Includes immutable probe metadata, not any student audit predictions.
    inputs={'source_id':sid,'question_sha256':source['question_sha256'],
        'pool_sha256':digest(json.dumps(pool,sort_keys=True)),
        'probe_bundle_sha256':digest(json.dumps(probes,sort_keys=True)),
        'seed':seed,'updates':updates,'initial_adapter_sha256':initial_hash,
        'threshold':threshold,'configuration_sha256':configuration_sha256,
        'schema':'tgrd_fixed128_trial_inputs_v1'}
    freeze(folder/'inputs.json',inputs)
    input_hash=baseline.sha(folder/'inputs.json')

    def reset():
        learner.reset(seed)
        if learner.updates!=0 or weight_hash(learner)!=initial_hash:
            raise ValueError('Branch reset did not restore S0 and a zero update count')

    reset()
    base_probe=evaluate(learner,probes['select'],folder/'untrained_Vsel.jsonl')
    scores={};branches={}
    for index,candidate in enumerate(candidates):
        cid=candidate['candidate_id'];branch=folder/f'branch{index}'
        summary_path=branch/'summary.json';trained_path=branch/'trained.json'
        adapter_path=branch/'adapter.safetensors'
        expected={'candidate_id':cid,'source_id':sid,'updates':updates,'seed':seed,
            'completion_sha256':candidate['completion_sha256'],
            'initial_adapter_sha256':initial_hash,'root_input_file_sha256':input_hash,
            'training_data':'original_root_completion_only'}
        if trained_path.exists():
            trained=read(trained_path)
            if any(trained.get(k)!=v for k,v in expected.items()):
                raise ValueError('Committed training provenance mismatch')
            if baseline.sha(adapter_path)!=trained['adapter_file_sha256']:
                raise ValueError('Committed trained adapter changed')
            reset();restore(learner,adapter_path)
            if weight_hash(learner)!=trained['trained_adapter_sha256']:
                raise ValueError('Restored training branch differs from saved parameters')
        else:
            if summary_path.exists():
                raise ValueError('Branch summary exists without committed training metadata')
            branch.mkdir(parents=True,exist_ok=True)
            preserve_orphan(adapter_path)
            reset()
            logs=learner.trial(source['question'],candidate['completion'],updates=updates)
            after=weight_hash(learner)
            if (after==initial_hash or learner.updates!=updates or len(logs)!=updates
                    or [row['updates'] for row in logs]!=list(range(1,updates+1))):
                raise ValueError('Trial did not execute exactly the declared fresh updates')
            learner.save(adapter_path)
            trained={**expected,'updates_log':logs,'trained_adapter_sha256':after,
                'adapter_file_sha256':baseline.sha(adapter_path),'at_utc':now()}
            # Commit before lengthy generations. Resume restores this adapter
            # instead of repeating training or overwriting earlier predictions.
            freeze(trained_path,trained)
        metrics=evaluate(learner,probes['select'],branch/'Vsel.jsonl')
        result={**trained,'Vsel':metrics}
        freeze(summary_path,result)
        scores[cid]=metrics;branches[cid]=(branch,trained)

    decision=choose(scores,candidates,threshold)
    freeze(folder/'decision_before_audit.json',{'source_id':sid,'decision':decision,
        'configuration_sha256':configuration_sha256,'root_input_file_sha256':input_hash,
        'Vsel_scores':scores,'untrained_Vsel':base_probe,'audit_can_change_selection':False})
    needed=['e0']+([] if decision['selected_id']=='e0' else [decision['selected_id']])
    audits={}
    for cid in needed:
        branch,trained=branches[cid];adapter_path=branch/'adapter.safetensors'
        if baseline.sha(adapter_path)!=trained['adapter_file_sha256']:
            raise ValueError('Audit adapter file changed')
        reset();restore(learner,adapter_path)
        if weight_hash(learner)!=trained['trained_adapter_sha256']:
            raise ValueError('Audit did not restore the selected adapter')
        audits[cid]=evaluate(learner,probes['audit'],branch/'Vaudit.jsonl')
    result={'schema':'tgrd_fixed128_trial_result_v1','source_id':sid,
        'configuration_sha256':configuration_sha256,
        'root_input_file_sha256':input_hash,'decision':decision,
        'Vaudit_accuracy_gain':audits[decision['selected_id']]['accuracy']-audits['e0']['accuracy'],
        'Vaudit_nll_gain':audits['e0']['answer_nll']-audits[decision['selected_id']]['answer_nll'],
        'audit_scores':audits,'audit_can_change_selection':False,
        'original_root_only_gradients':True,'other_thresholds_evaluated':False}
    freeze(folder/'trial_result.json',result)
    reset()
    return result
