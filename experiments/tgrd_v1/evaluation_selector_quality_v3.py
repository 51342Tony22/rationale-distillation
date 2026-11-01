"""Public-evaluation provenance for adopted automatic explanation quality v3.

No test dataset is opened here. The legacy evaluator and all active training
inputs remain immutable. Concrete final epoch4 artifacts and an actual complete
general34 development grid are prerequisites; CPU success never unlocks test.
"""
from pathlib import Path
from runtime_full import ROOT, HERE, OUT, BASE, read, freeze, baseline
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from evaluation_selector_inputs import (training_artifacts, verify_inputs,
    pretest_evidence as legacy_pretest_evidence, TRANSFER_RULES, EVALUATION_ORDER)

GATE_NAME = PROTOCOL + '_selector_public_evaluation_quality_v3.json'
PRETEST_NAME = PROTOCOL + '_pretest_transfer_quality_v3.json'


def quality_evidence():
    import explanation_quality_v3 as quality
    quality.authority()
    paths = [quality.AUTHORITY, quality.EXPORT, quality.REPORT]
    for path in paths.copy():
        paths += verify_inputs(read(path))
    export = read(quality.EXPORT); report = read(quality.REPORT)
    if (export.get('schema') != 'tgrd_explanation_automatic_export_v3'
            or export.get('complete') is not True
            or export.get('all_text_read') is not False
            or export.get('semantic_correctness_verified') is not False
            or report.get('schema') != 'tgrd_explanation_fixed128_actual_spotcheck_v3'
            or report.get('complete') is not True or report.get('sample_size') != 128
            or report.get('existing_exact_reused') != 31 or report.get('new_actual_reads') != 97
            or len(report.get('entries', [])) != 128
            or report.get('all_correct_required') is not False
            or export.get('public_test_used') is not False):
        raise ValueError('Actual complete v3 automatic decisions and fixed128 evidence required')
    return list(dict.fromkeys(paths))


def collect_training():
    from train_selector_quality_v3 import validate_gate, GATE, validate_stage_bundle
    _, paths = validate_gate('M1', 42)
    training = read(GATE); manifest = read(ROOT / training['materials_manifest'])
    if manifest.get('explanation_quality_version') != 'v3':
        raise ValueError('Adopted v3 formal materials required')
    paths += quality_evidence()
    bundles = {g: read(ROOT / manifest['materials'][g]['path']) for g in SELECTOR_CONDITIONS}
    ids = training['common_root_ids']; validate_stage_bundle(manifest, bundles, ids)
    initialization = OUT / 'training' / PROTOCOL / 'initializations/seed42.json'
    artifacts, files = training_artifacts(bundles, ids, read(initialization), root=ROOT, out=OUT)
    paths += [GATE, initialization, *files]
    for group in SELECTOR_CONDITIONS:
        path = OUT / 'signatures' / f'{PROTOCOL}_training_{group}_s42.json'
        signed = read(path)
        if (signed.get('extra') != {'protocol': PROTOCOL, 'stage': 'selector_first',
                'condition': group, 'seed': 42, 'primary_checkpoint': 'last_epoch4',
                'public_test_used': False}
                or str(GATE.relative_to(ROOT)) not in signed.get('inputs', {})
                or manifest['materials'][group]['path'] not in signed.get('inputs', {})
                or 'experiments/tgrd_v1/train_selector_quality_v3.py' not in signed.get('inputs', {})):
            raise ValueError('Current v3 final training signature required for every group')
        paths += [path, *verify_inputs(signed)]
    return training, artifacts, list(dict.fromkeys(paths))


def pretest_evidence(roster_path, rules_path, disclosures_path, ids):
    # This reconstructs all six actual general34 jobs, all24 threshold cells,
    # signed predictions and sampling evidence; no loose booleans can replace it.
    result, paths = legacy_pretest_evidence(roster_path, rules_path, disclosures_path, ids)
    paths += quality_evidence()
    value = read(disclosures_path)
    text = value.get('automatic_explanation_quality_noise')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Concrete automatic explanation quality/noise disclosure required')
    from explanation_quality_v3 import REPORT, EXPORT
    expected = {str(p.relative_to(ROOT)): baseline.sha(p) for p in (REPORT, EXPORT)}
    if value.get('explanation_quality_evidence_sha256') != expected:
        raise ValueError('Disclosure must bind the actual explanation report and automatic export')
    paths += [Path(__file__), HERE/'evaluation_selector_inputs.py']
    return result, list(dict.fromkeys(paths))


def _no_existing_predictions():
    if any((OUT/'evaluation'/PROTOCOL).glob('*/seed42/predictions.jsonl')):
        raise ValueError('Cannot backdate a pretest freeze or public gate after predictions')


def freeze_pretest(roster_path, rules_path, disclosures_path):
    from full_selector import load_quality
    _, _, _, ids = load_quality()
    destination = OUT/'gates'/PRETEST_NAME
    if not destination.exists(): _no_existing_predictions()
    result, paths = pretest_evidence(roster_path, rules_path, disclosures_path, ids)
    value = {'schema': 'tgrd_transfer_pretest_quality_v3', 'protocol': PROTOCOL,
        'explanation_quality_version': 'v3', 'frozen_before_current_public_test': True,
        'public_test_used': False,
        **{k: str(Path(p).resolve().relative_to(ROOT)) for k,p in (
            ('roster',roster_path),('rules',rules_path),('disclosures',disclosures_path))},
        **result, 'inputs': {str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    freeze(destination,value);return value


def verify_pretest(ids):
    path=OUT/'gates'/PRETEST_NAME;value=read(path)
    if (value.get('schema')!='tgrd_transfer_pretest_quality_v3' or value.get('protocol')!=PROTOCOL
            or value.get('explanation_quality_version')!='v3'
            or value.get('frozen_before_current_public_test') is not True
            or value.get('public_test_used') is not False):
        raise ValueError('Actual v3 pretest transfer freeze required')
    paths=verify_inputs(value)
    result,evidence=pretest_evidence(*(ROOT/value[k] for k in ('roster','rules','disclosures')),ids)
    if any(value.get(k)!=v for k,v in result.items()) or not set(evidence)<=set(paths):
        raise ValueError('Pretest freeze omits or differs from actual development/quality evidence')
    return [path,*paths]


def current_evidence():
    training,artifacts,paths=collect_training()
    paths+=verify_pretest(training['common_root_ids'])
    return training,artifacts,list(dict.fromkeys(paths))


def gate_fields():
    return dict(schema='tgrd_selector_public_gate_quality_v3',enabled=True,
        protocol=PROTOCOL,explanation_quality_version='v3',stage='selector_first',primary_dataset='GSM8K',
        conditions=list(SELECTOR_CONDITIONS),seed=42,epochs=4,test_count=1319,
        max_new_tokens=1024,greedy=True,primary_checkpoint='last_epoch4',
        public_test_used_for_freeze=False,all_five_groups_completed=False)


def freeze_gate():
    training,artifacts,paths=current_evidence()
    paths += [Path(__file__), HERE/'evaluate_selector_quality_v3.py']
    value={**gate_fields(),'artifacts':artifacts,'accepted_root_count':len(training['common_root_ids']),
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    destination=OUT/'gates'/GATE_NAME
    if not destination.exists():_no_existing_predictions()
    freeze(destination,value);return value


def validate_request(condition,seed=42,adapter=None):
    if condition not in SELECTOR_CONDITIONS or type(seed) is not int or seed!=42:
        raise ValueError('Only current G1-G3 seed42 public evaluation is supported here')
    path=OUT/'gates'/GATE_NAME;gate=read(path)
    if any(gate.get(k)!=v for k,v in gate_fields().items()):
        raise ValueError('A genuine v3 public evaluation gate is required')
    paths=verify_inputs(gate)
    training,artifacts,evidence=current_evidence()
    if (gate.get('artifacts')!=artifacts
            or gate.get('accepted_root_count')!=len(training['common_root_ids'])
            or not set(evidence)<=set(paths)):
        raise ValueError('Public gate does not match complete training/development evidence')
    entry=artifacts[condition];directory=ROOT/entry['adapter_directory']
    if adapter is not None and Path(adapter).resolve()!=directory.resolve():
        raise ValueError('Adapter override differs from frozen final checkpoint')
    return dict(adapter_directory=directory,training_artifact=entry,signature_paths=[path,*paths])
