"""All-five public admission after actual positive finite128 confirmation.

Final epoch4, common formal S0, actual grid and production replay are rebuilt.
No public question loader is imported or called by this provenance module.
"""
import json
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,BASE,read,freeze,baseline,digest,checked_rows
from budgeted_materials import PROTOCOL
from training_plan import schedule,validate_materials
from training_checkpoint import verify_checkpoint
from learning import LORA
from train_transfer128_v1 import CONDITIONS
from evaluation_selector_inputs import verify_inputs,EVALUATION_ORDER

GATE_NAME=PROTOCOL+'_all_five_public_evaluation_transfer128_v1.json'


def training_artifacts(bundles, ids, initialization, *, root=ROOT, out=OUT):
    """Validate both G4/G5 final checkpoint, including runs not being scored."""
    root = Path(root); out = Path(out)
    if (set(bundles) != set(CONDITIONS) or not ids
            or initialization.get('protocol') != PROTOCOL
            or initialization.get('seed') != 42
            or initialization.get('student_revision') != BASE['student_revision']
            or initialization.get('LoRA') != LORA or initialization.get('layers') != 16
            or not initialization.get('initial_adapter_sha256')):
        raise ValueError('Complete staged roster and common actual initialization required')
    initial = initialization['initial_adapter_sha256']
    plan = schedule(len(ids), 42); updates = len(plan)
    artifacts = {}; paths = []
    for group in CONDITIONS:
        bundle = bundles[group]
        if (bundle.get('condition') != group or bundle.get('protocol') != PROTOCOL
                or bundle.get('stage') != 'finite_transfer128'
                or bundle.get('development_preview_only') is not False):
            raise ValueError('Current-protocol formal finite128 materials required')
        validate_materials(bundle, ids)
        folder = out / 'training' / PROTOCOL / group / 'seed42'
        inputs = read(folder / 'inputs.json'); meta = read(folder / 'metadata.json')
        epoch = read(folder / 'epochs/epoch4.json')
        checkpoint = folder / 'checkpoints' / f'update_{updates:08d}'
        if ((folder / epoch['checkpoint_directory']).resolve() != checkpoint.resolve()
                or read(folder / 'adapter_config.json') != {
                    'fine_tune_type': 'lora', 'num_layers': 16, 'lora_parameters': LORA}):
            raise ValueError('Final checkpoint or inference adapter configuration differs')
        commit = verify_checkpoint(checkpoint)
        weights = baseline.sha(folder / 'adapters.safetensors')
        input_sha = baseline.sha(folder / 'inputs.json')
        expected_inputs = {
            'schema': 'tgrd_training_run_v1', 'condition': group, 'seed': 42,
            'ordered_root_ids': ids, 'epochs': 4, 'updates': updates,
            'initial_adapter_sha256': initial,
            'materials_sha256': digest(json.dumps(bundle, sort_keys=True)),
            'schedule_sha256': digest(json.dumps(plan, sort_keys=True)),
            'primary_checkpoint': 'last_epoch4_not_selected_on_test',
            'equal_budget': 'one_root_exposure_per_epoch;five_microbatches_per_full_update',
        }
        expected_meta = {
            'completed': True, 'condition': group, 'seed': 42, 'epochs': 4,
            'accepted_root_count': len(ids), 'candidate_root_count': 7473,
            'updates': updates, 'root_exposures': 4 * len(ids),
            'initial_adapter_sha256': initial, 'input_file_sha256': input_sha,
            'adapter_sha256': weights, 'primary_checkpoint': 'last_epoch4',
            'public_test_used_to_select_checkpoint': False,
        }
        point = commit['progress']
        if (any(inputs.get(k) != v for k, v in expected_inputs.items())
                or any(meta.get(k) != v for k, v in expected_meta.items())
                or epoch != {'epoch': 4, 'update': updates,
                    'checkpoint_directory': str(checkpoint.relative_to(folder)),
                    'adapter_sha256': weights, 'primary': True}
                or baseline.sha(checkpoint / 'adapter.safetensors') != weights
                or commit['learner_updates'] != updates
                or any(point.get(k) != v for k, v in {
                    'epoch': 4, 'update': updates, 'epoch_end': True,
                    'input_file_sha256': input_sha, 'root_exposures': 4 * len(ids)}.items())):
            raise ValueError('Not the complete common-root seed42 final-epoch4 artifact: ' + group)
        names = ['inputs.json', 'metadata.json', 'epochs/epoch4.json',
                 'adapters.safetensors', 'adapter_config.json']
        names += [str((checkpoint / n).relative_to(folder)) for n in
                  ('commit.json', 'adapter.safetensors', 'state.safetensors', 'state_structure.json')]
        files = {name: baseline.sha(folder / name) for name in names}
        paths += [folder / name for name in names]
        artifacts[group] = {'adapter_directory': str(folder.relative_to(root)),
            'files': files, 'adapter_sha256': weights, 'accepted_root_count': len(ids),
            'primary_checkpoint': 'last_epoch4', 'seed': 42}
    return artifacts, paths


def collect_transfer_training(ids):
    from train_transfer128_v1 import validate_gate,GATE
    _,paths=validate_gate('M7',42)
    from transfer128_materials_v1 import FOLDER
    manifest_path=FOLDER/'materials_manifest_v1.json';manifest=read(manifest_path)
    paths += [GATE,manifest_path,*verify_inputs(manifest)]
    bundles={g:read(ROOT/manifest['materials'][g]['path']) for g in CONDITIONS}
    initialization=OUT/'training'/PROTOCOL/'initializations/seed42.json'
    artifacts,files=training_artifacts(bundles,ids,read(initialization),root=ROOT,out=OUT)
    paths += [initialization,*files]
    for group in CONDITIONS:
        signed_path=OUT/'signatures'/f'{PROTOCOL}_training_{group}_s42.json';signed=read(signed_path)
        if (signed.get('extra')!=dict(protocol=PROTOCOL,stage='finite_transfer128',condition=group,
                seed=42,primary_checkpoint='last_epoch4',public_test_used=False)
                or not {str(GATE.relative_to(ROOT)),manifest['materials'][group]['path'],
                    'experiments/tgrd_v1/train_transfer128_v1.py'}<=set(signed.get('inputs',{}))):
            raise ValueError('Actual current finite128 formal training signature required')
        paths += [signed_path,*verify_inputs(signed)]
    return artifacts,list(dict.fromkeys(paths))


def extension_disclosures(disclosures,export,checksum):
    if (not isinstance(disclosures.get('finite128_confirmation_scope'),str)
            or not disclosures['finite128_confirmation_scope'].strip()
            or not isinstance(disclosures.get('new142_explanation_population'),str)
            or not disclosures['new142_explanation_population'].strip()
            or disclosures.get('new142_covered_by_original_descriptive_sample') is not False
            or disclosures.get('transfer128_explanation_export_sha256')!=checksum
            or disclosures.get('transfer128_explanation_counts')!={k:export[k] for k in
                ('requests','new_outputs','exact_output_reuses','automatically_eligible','excluded','new_automatically_eligible')}
            or export.get('complete') is not True or export.get('requests')!=412
            or export.get('new_outputs')!=142 or export.get('exact_output_reuses')!=270
            or export.get('all_text_read') is not False
            or export.get('semantic_correctness_verified') is not False
            or export.get('new142_covered_by_original_descriptive_sample') is not False):
        raise ValueError('Concrete finite128 limits and separate unsampled142 population disclosure required')


def current_evidence():
    import evaluation_selector_quality_v3 as selector
    from transfer128_production_v1 import load_configuration,CONFIGURATION
    from transfer128_explanations_v1 import FOLDER as EXPLANATIONS
    training,artifacts,paths=selector.current_evidence()
    config=load_configuration()
    if config['scientific_trial_enabled'] is not True:
        raise ValueError('No-positive development cannot authorize automatic five-group expansion')
    pretest_path=OUT/'gates'/selector.PRETEST_NAME;pretest=read(pretest_path)
    if config['selected_configuration']!=pretest['selected_configuration']:
        raise ValueError('Actual trial configuration differs from pretest development freeze')
    more,files=collect_transfer_training(training['common_root_ids']);artifacts.update(more)
    export_path=EXPLANATIONS/'automatic_export_v1.json';export=read(export_path)
    disclosures_path=ROOT/pretest['disclosures']
    extension_disclosures(read(disclosures_path),export,baseline.sha(export_path))
    paths += [CONFIGURATION,pretest_path,disclosures_path,export_path,*verify_inputs(export),*files]
    return training,artifacts,list(dict.fromkeys(paths))


def gate_fields():
    return dict(schema='tgrd_all_five_public_gate_transfer128_v1',enabled=True,protocol=PROTOCOL,
        explanation_quality_version='v3',stage='selector_and_finite_transfer128',primary_dataset='GSM8K',
        conditions=EVALUATION_ORDER,seed=42,epochs=4,test_count=1319,max_new_tokens=1024,
        greedy=True,primary_checkpoint='last_epoch4',public_test_used_for_freeze=False,
        all_five_groups_completed=True)


def freeze_gate():
    training,artifacts,paths=current_evidence()
    paths += [Path(__file__),HERE/'evaluate_transfer128_v1.py']
    path=OUT/'gates'/GATE_NAME
    if not path.exists() and any((OUT/'evaluation'/PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('All-five gate must be frozen before any current public predictions')
    value={**gate_fields(),'artifacts':artifacts,'accepted_root_count':len(training['common_root_ids']),
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}}
    freeze(path,value);return value


def validate_order(condition,artifacts):
    for previous in EVALUATION_ORDER[:EVALUATION_ORDER.index(condition)]:
        folder=OUT/'evaluation'/PROTOCOL/previous/'seed42'
        summary=read(folder/'summary.json');rows=checked_rows(folder/'predictions.jsonl')
        sig=OUT/'signatures'/f'{PROTOCOL}_eval_GSM8K_{previous}_s42.json'
        signature_sha=baseline.sha(sig)
        if (summary.get('complete') is not True or summary.get('completed')!=1319
                or summary.get('total')!=1319 or summary.get('training_artifact')!=artifacts[previous]
                or summary.get('evaluation_signature_file_sha256')!=signature_sha
                or len(rows)!=1319 or len({r['source_id'] for r in rows})!=1319
                or any(r.get('condition')!=previous or r.get('seed')!=42
                    or r.get('evaluation_signature_file_sha256')!=signature_sha
                    or digest(r['output'])!=r.get('output_sha256') for r in rows)):
            raise ValueError('Full official evaluation must follow frozen M1/NLL/M3/M7/M6 order')
        verify_inputs(read(sig))


def validate_request(condition,seed=42,adapter=None):
    if condition not in EVALUATION_ORDER or type(seed) is not int or seed!=42:
        raise ValueError('Only declared five groups at seed42 supported')
    path=OUT/'gates'/GATE_NAME;gate=read(path)
    if any(gate.get(k)!=v for k,v in gate_fields().items()):
        raise ValueError('Genuine actual five-group public gate required')
    paths=verify_inputs(gate);training,artifacts,evidence=current_evidence()
    if (gate.get('artifacts')!=artifacts or gate.get('accepted_root_count')!=len(training['common_root_ids'])
            or not set(evidence)<=set(paths)):
        raise ValueError('Gate differs from final training/development/finite128 evidence')
    directory=ROOT/artifacts[condition]['adapter_directory']
    if adapter is not None and Path(adapter).resolve()!=directory.resolve():
        raise ValueError('Adapter override differs from frozen final epoch4')
    validate_order(condition,artifacts)
    return dict(adapter_directory=directory,training_artifact=artifacts[condition],signature_paths=[path,*paths])
