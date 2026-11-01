"""Versioned numeric-reference formatting adapter for actual finite128 reviews.

Preserves v1 and raw data; partial batches cannot enable scientific trials.
"""
from pathlib import Path
import argparse
import json
from runtime_full import ROOT, HERE, OUT, DATA, read, freeze, save, digest, baseline
from budgeted_materials import PROTOCOL, validate_roster
from transfer128_sampling import ROSTER
from transfer128_variants_v1 import validate_entry as validate_entry_v1
from trial_selection import validate_probes
from variant_rationales import root_rationales

FOLDER=OUT/'transfer128/variants_v1'


def normalized_source(source):
    """Normalize only valid thousands grouping in a private comparison copy.

    Raw question/gold/source files stay unchanged. This is numeric formatting,
    never permission to repair the reference or infer semantic acceptance.
    """
    from fractions import Fraction
    import re
    raw=source['gold']
    if not isinstance(raw,str):
        raise ValueError('Reference must retain its raw string provenance')
    comparison=raw
    if ',' in raw:
        if not re.fullmatch(r'[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?',raw):
            raise ValueError('Malformed thousands-grouped reference')
        comparison=raw.replace(',','')
    value=Fraction(comparison)
    canonical=str(value.numerator) if value.denominator==1 else str(value)
    normalized={**source,'gold':canonical}
    binding={'source_id':source['source_id'],
        'question_sha256':source['question_sha256'],
        'raw_gold':raw,'raw_gold_sha256':digest(raw),
        'comparison_gold':canonical,
        'normalization':'strict_thousands_grouping_then_exact_rational',
        'source_question_or_reference_modified':False}
    return normalized,binding


def validate_entry(entry,source):
    # The v1 explicit-read/semantic/observable-only checks still execute.
    normalized,binding=normalized_source(source)
    result=validate_entry_v1(entry,normalized)
    result['numeric_reference_binding']=binding
    return result


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
    inputs=[Path(__file__),HERE/'transfer128_variants_v1.py',batch_path,ROSTER,manifest,DATA/'train7473.json',
        HERE/'variants.py',HERE/'trial_selection.py',HERE/'variant_rationales.py']
    value={'schema':'tgrd_transfer128_actual_variant_batch_v2','protocol':PROTOCOL,
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
