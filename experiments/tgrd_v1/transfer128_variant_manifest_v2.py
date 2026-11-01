"""Replay the complete fixed128 actual template ledger; never unlock trials."""
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,DATA,read,freeze,baseline
from budgeted_materials import PROTOCOL,validate_roster
from transfer128_sampling import ROSTER
from train_selector_quality_v3 import verify_inputs
from transfer128_variants_v1 import validate_entry as validate_v1
from transfer128_variants_v2 import validate_entry as validate_v2,normalized_source

FOLDER=OUT/'transfer128/variants_v1'


def replay_batches(reports,sources,roster_ids):
    entries={};results={}
    for report in reports:
        schema=report['schema']
        validator={'tgrd_transfer128_actual_variant_batch_v1':validate_v1,
                   'tgrd_transfer128_actual_variant_batch_v2':validate_v2}.get(schema)
        if validator is None or report.get('scientific_trial_executed') is not False:
            raise ValueError('Unknown actual-review batch or invalid scientific claim')
        if len(report['actual_review_entries'])!=len(report['validated_results']):
            raise ValueError('Every actual entry needs an exact replay result')
        for entry,stored in zip(report['actual_review_entries'],report['validated_results']):
            sid=entry['source_id']
            if sid not in roster_ids or sid in entries:
                raise ValueError('Original roster only, exactly one actual decision per root')
            replay=validator(entry,sources[sid])
            if replay!=stored:
                raise ValueError('Stored probes, rationales or reference binding changed')
            entries[sid]=entry;results[sid]=replay
        if (report['reviewed_roots']!=len(report['actual_review_entries'])
                or report['qualified_roots']!=sum(r['qualified'] for r in report['validated_results'])):
            raise ValueError('Actual batch counts disagree with evidence')
    if set(entries)!=set(roster_ids) or len(roster_ids)!=128:
        raise ValueError('Complete original128 actual decisions required, without replacement')
    return [entries[sid] for sid in roster_ids],[results[sid] for sid in roster_ids]


def prepare():
    from full_selector import load_quality
    source_rows,_,_,ids=load_quality();sources={r['source_id']:r for r in source_rows}
    roster=read(ROSTER);selected=validate_roster(roster,ids)
    prep=OUT/'transfer128/preparation_v1/manifest.json';verify_inputs(read(prep))
    paths=sorted((FOLDER/'batches').glob('*.json'))
    reports=[read(p) for p in paths]
    for report in reports:verify_inputs(report)
    entries,results=replay_batches(reports,sources,selected)
    inputs=[Path(__file__),HERE/'transfer128_variants_v1.py',HERE/'transfer128_variants_v2.py',
        HERE/'variants.py',HERE/'trial_selection.py',HERE/'variant_rationales.py',ROSTER,prep,
        DATA/'train7473.json',*paths]
    checksums={str(p.relative_to(ROOT)):baseline.sha(p) for p in inputs}
    roots=[];bundles=[];rationales=[]
    for entry,result in zip(entries,results):
        sid=entry['source_id'];_,binding=normalized_source(sources[sid])
        roots.append({'source_id':sid,'qualified':result['qualified'],'reason':entry['reason'],
            'numeric_reference_binding':binding,
            'original_question_text_read':entry['original_question_text_read'],
            'all_six_templates_text_read':entry.get('all_six_templates_text_read',False),
            'ordinary_CoT_fallback_required':not result['qualified']})
        if result['qualified']:
            bundles.append(result['bundle']);rationales.append(result['Vsel_rationales'])
    qualified=len(bundles)
    common={'protocol':PROTOCOL,'scientific_trial_executed':False,'trial_enabled':False,
        'public_test_used':False,'model_loaded':False,'inputs':checksums}
    bundle_path=FOLDER/'qualified_bundles_v2.json'
    rationale_path=FOLDER/'qualified_Vsel_rationales_v2.json'
    report_path=FOLDER/'actual_review_report_v2.json'
    freeze(bundle_path,{**common,'schema':'tgrd_transfer128_qualified_bundles_v2','roots':bundles})
    freeze(rationale_path,{**common,'schema':'tgrd_transfer128_qualified_Vsel_v2','roots':rationales,
        'trained_Vsel_is_generalization_evidence':False,'Vaudit_rationales_generated':False})
    freeze(report_path,{**common,'schema':'tgrd_transfer128_actual_variant_report_v2','complete':True,
        'actual_reviewed_roots':128,'qualified_roots':qualified,'rejected_roots':128-qualified,
        'roots':roots,'Vsel_probes':6*qualified,'Vaudit_probes':6*qualified,
        'Vsel_observed_only_rationales':6*qualified,'roster_replaced':False,
        'original_teacher1433_count_unchanged':True,'explanation2821_count_unchanged':True,
        'review_kind':'assistant_actual_relation_and_template_review',
        'independent_human_review':False,
        'limitations':[
            'Qualified explicit template families do not prove original teacher reasoning correct.',
            'Original-reference assumptions and wording clarifications are individually retained in reasons.',
            'Halving families4384/5595 fix two/three complete intervals;5112 fixes three half-folds.',
            'Toy composition, travel and scoring models are not real-world scientific or sporting claims.',
            'Integer counts, exact installments and nonnegative-quantity restrictions bound the template domains.',
            '3124/4560 domains expanded before commit solely to supply12 distinct exact valid tuples, without model outputs.',
            'These are training-root-derived probes with different wording, not externally sampled generalization data.',
            'No model trial, context-length production qualification, actual-grid configuration or test gate is supplied.']})
    manifest={**common,'schema':'tgrd_transfer128_variant_manifest_v2','complete_actual_relation_review':True,
        'qualified_roots':qualified,'rejected_roots':128-qualified,'frozen_roots':128,
        'Vsel_probes':6*qualified,'Vaudit_probes':6*qualified,
        'outputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in (bundle_path,rationale_path,report_path)},
        'all128_probe_bundles_qualified':qualified==128,
        'rejected_roots_require_ordinary_CoT':True,
        'production_ready':False,'remaining_requirements':[
            '142 new explanation outputs and bound quality evidence',
            'production tokenization/context checks and exact joint-only pool binding',
            'actual complete general34 grid and frozen selected configuration or explicit unsupported outcome',
            'actual fixed128 production trials followed by G4/G5 material/training/evaluation adapters']}
    freeze(FOLDER/'manifest_v2.json',manifest)
    return manifest


if __name__=='__main__':
    report=prepare()
    print({k:report[k] for k in ['frozen_roots','qualified_roots','rejected_roots','Vsel_probes','Vaudit_probes','production_ready']})
