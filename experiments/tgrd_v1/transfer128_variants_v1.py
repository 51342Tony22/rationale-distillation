"""Commit explicit finite128 relation reviews and exactly replayed 6+6 probes.

Review decisions are external authored input; this program never infers a
mathematical acceptance. Partial batches remain partial and cannot enable trials.
"""
from pathlib import Path
import argparse
import json
from runtime_full import ROOT, HERE, OUT, DATA, read, freeze, save, digest, baseline
from budgeted_materials import PROTOCOL, validate_roster
from transfer128_sampling import ROSTER
from variants import build
from trial_selection import validate_probes
from variant_rationales import root_rationales

FOLDER=OUT/'transfer128/variants_v1'


def validate_entry(entry, source):
    if (entry.get('source_id')!=source['source_id']
            or entry.get('question_sha256')!=source['question_sha256']
            or source['question_sha256']!=digest(source['question'])
            or type(entry.get('accepted')) is not bool or not entry.get('reason')
            or entry.get('review_kind')!='assistant_actual_relation_and_template_review'
            or entry.get('original_question_text_read') is not True):
        raise ValueError('Explicit exact-root actual relation review required')
    if not entry['accepted']:
        if entry.get('spec') is not None:
            raise ValueError('Rejected relation must not carry a qualified template spec')
        return {'source_id':source['source_id'],'qualified':False,'reason':entry['reason']}
    spec=entry['spec']
    if (entry.get('all_six_templates_text_read') is not True
            or spec.get('sampling_parameter_order') is None
            or spec.get('review',{}).get('reason')!=entry['reason']):
        raise ValueError('Accepted explicit template review and deterministic parameter order required')
    bundle={'source_id':source['source_id'],'spec':spec,'variants':build(spec,source)}
    validate_probes(bundle,source,build)
    rationales=root_rationales(bundle,source)
    rationales.update(development_only=False,stage='finite_transfer128_preparation',
        scientific_trial_executed=False,trained_Vsel_is_generalization_evidence=False)
    return {'source_id':source['source_id'],'qualified':True,
        'bundle':bundle,'Vsel_rationales':rationales}


def commit(batch_path):
    from train_selector_quality_v3 import verify_inputs
    from full_selector import load_quality
    sources,_,_,ids=load_quality();sources={r['source_id']:r for r in sources}
    roster=read(ROSTER);selected=validate_roster(roster,ids)
    manifest=OUT/'transfer128/preparation_v1/manifest.json';verify_inputs(read(manifest))
    entries=read(batch_path)
    if not isinstance(entries,list) or not entries or len(entries)!=len({e['source_id'] for e in entries}):
        raise ValueError('A nonempty unique explicit review batch required')
    batch_path=Path(batch_path).resolve();batch_path.relative_to(ROOT)
    results=[]
    for entry in entries:
        if entry['source_id'] not in selected:raise ValueError('Only original frozen128 roots allowed')
        results.append(validate_entry(entry,sources[entry['source_id']]))
    other={}
    for path in sorted((FOLDER/'batches').glob('*.json')):
        report=read(path);verify_inputs(report)
        for entry in report['actual_review_entries']:
            sid=entry['source_id']
            if sid in other:raise ValueError('Duplicate committed root review')
            other[sid]=entry
    for entry in entries:
        if entry['source_id'] in other and other[entry['source_id']]!=entry:
            raise ValueError('Cannot silently change an existing actual review or replace a failed root')
    name='batch_'+baseline.sha(batch_path)[:20]+'.json'
    for report in (FOLDER/'batches').glob('*.json'):
        if any(e['source_id'] in {x['source_id'] for x in entries} for e in read(report)['actual_review_entries']):
            if report.name!=name:raise ValueError('Each root has one immutable actual batch')
    inputs=[Path(__file__),batch_path,ROSTER,manifest,DATA/'train7473.json',
        HERE/'variants.py',HERE/'trial_selection.py',HERE/'variant_rationales.py']
    value={'schema':'tgrd_transfer128_actual_variant_batch_v1','protocol':PROTOCOL,
        'actual_review_entries':entries,'validated_results':results,
        'reviewed_roots':len(entries),'qualified_roots':sum(r['qualified'] for r in results),
        'public_test_used':False,'model_loaded':False,'scientific_trial_executed':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in inputs}}
    freeze(FOLDER/'batches'/name,value)
    other.update({e['source_id']:e for e in entries})
    paths=sorted((FOLDER/'batches').glob('*.json'))
    progress={'schema':'tgrd_transfer128_actual_variant_review_progress_v1',
        'complete':len(other)==128,'actual_reviewed_roots':len(other),
        'qualified_roots':sum(e['accepted'] for e in other.values()),
        'rejected_roots':sum(not e['accepted'] for e in other.values()),
        'pending_roots':[sid for sid in roster['source_ids'] if sid not in other],
        'all_128_probe_bundles_qualified':len(other)==128 and all(e['accepted'] for e in other.values()),
        'trial_enabled':False,'scientific_effectiveness_established':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in paths}}
    save(FOLDER/'review_progress_v1.json',progress)
    return progress


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['commit'])
    parser.add_argument('--batch',type=Path,required=True);args=parser.parse_args()
    report=commit(args.batch)
    print({k:v for k,v in report.items() if k not in ('inputs','pending_roots')},flush=True)
