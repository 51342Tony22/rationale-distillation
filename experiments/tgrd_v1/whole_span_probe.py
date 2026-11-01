"""Sidecar diagnosis; preserves existing v2 candidate positions and outputs."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from runtime_full import ROOT, HERE, OUT, DATA, BASE, read, freeze, save, signature, checked_rows, append, baseline, gpu, load_model, check_time, now
from calibration_cohort import COHORTS, cohort_paths
from selector_masking_probe import MLXBackend
from whole_span_masking import whole_units, matched_controls, diagnose_whole_step


def prepare(cohort):
    from transformers import AutoTokenizer
    paths = cohort_paths(cohort)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']), local_files_only=True,
                                             trust_remote_code=False)
    sources = {r['source_id']:r for r in read(DATA/'cal40.json')}
    teachers = {r['source_id']:r for r in checked_rows(OUT/'materials/teacher_targets_cal40.jsonl')}
    reviews = {r['source_id']:r for r in read(paths.reviews)}
    cases = []
    for case in read(paths.cases):
        sid = case['source_id']; source = sources[sid]; teacher = teachers[sid]; review = reviews[sid]
        if (review['accepted'] is not True or review['output_sha256'] != teacher['output_sha256']
                or review['question_sha256'] != source['question_sha256']
                or case['teacher_output_sha256'] != teacher['output_sha256']):
            raise ValueError('Whole-span case lacks exact reviewed source provenance')
        spans = whole_units(BASE['student_prompt'].format(question=source['question']), source['question'],
                            teacher['output'], tokenizer)
        if spans['input_ids'] != case['input_ids']:
            raise ValueError('Whole-span input differs from existing v2 tokenization')
        steps = []
        for candidate in case['candidates']:
            step = candidate['step']
            units = [u for u in spans['units'] if max(u['positions']) < step['start_position']]
            supported = sum(matched_controls(u, units, seed=7, key=sid, count=3)[1] >= 3 for u in units)
            steps.append({'step': step, 'units': units,
                          'units_with_three_exact_controls': supported,
                          'any_attended_unit_will_have_controls': bool(units) and supported == len(units)})
        cases.append({'source_id':sid, 'question_sha256':source['question_sha256'],
                      'teacher_output_sha256':teacher['output_sha256'], 'input_ids':case['input_ids'],
                      'steps':steps, 'omitted_units':spans['omitted_units']})
    if len(cases) != paths.expected_roots:
        raise ValueError('Incomplete whole-span cohort')
    folder = paths.cases.parent/'whole_span_v1'
    freeze(folder/'cases.json', cases)
    inputs = [Path(__file__), HERE/'whole_span_masking.py', HERE/'calibration_cohort.py',
              HERE/'runtime.py', HERE/'runtime_full.py',
              ROOT/'experiments/math_distill_v1/selector_masking_core.py',
              ROOT/'experiments/math_distill_v1/selector_masking_probe.py', paths.cases, paths.reviews,
              DATA/'cal40.json', OUT/'materials/teacher_targets_cal40.jsonl',
              ROOT/'experiments/math_distill_v1/config.json', folder/'cases.json']
    possible = sum(step['units_with_three_exact_controls'] > 0 for c in cases for step in c['steps'])
    report = {'status':'prepared_CPU_only', 'at_utc':now(), 'roots':len(cases),
              'candidate_steps':sum(len(c['steps']) for c in cases),
              'steps_with_any_possible_exact_controlled_unit':possible,
              'steps_guaranteed_controls_for_any_attention_winner':sum(
                  step['any_attended_unit_will_have_controls'] for c in cases for step in c['steps']),
              'omitted_token_boundary_units':sum(len(c['omitted_units']) for c in cases),
              'same_kind_and_span_type':True, 'exact_removed_token_count':True,
              'required_control_count':3, 'GPU_execution_recommended':possible > 0,
              'distance_matched':False, 'semantic_relevance_matched':False,
              'model_loaded':False, 'v2_selector_changed':False, 'public_test_used':False,
              'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in inputs}}
    report_path = OUT/'preflight'/f'{cohort}_whole_span_cpu_v1.json'
    if report_path.exists():
        freeze(OUT/'preflight/history'/('whole_span_cpu_'+baseline.sha(report_path)+'.json'), read(report_path))
    save(report_path, report)
    return cases, folder, report


def main(cohort, prepare_only=False):
    cases, folder, report = prepare(cohort)
    if prepare_only:
        print({key:value for key,value in report.items() if key != 'inputs'}, flush=True)
        return
    if not report['GPU_execution_recommended']:
        raise ValueError('No step has any whole span with three exact controls; retained CPU coverage result, no GPU execution')
    with gpu():
        signature('whole_span_'+cohort, [ROOT/name for name in report['inputs']],
                  {'cohort':cohort, 'sidecar_only':True, 'layers':4, 'queries':8, 'controls':3,
                   'control_rule':'whole_span_same_kind_type_exact_token_count'})
        path = folder/'diagnostics.jsonl';saved = checked_rows(path)
        if [r['source_id'] for r in saved] != [c['source_id'] for c in cases[:len(saved)]]:
            raise ValueError('Whole-span resume order mismatch')
        model = None
        if len(saved) < len(cases):
            baseline.verify_student(BASE)
            model, _ = load_model('student');backend = MLXBackend(model, 4)
            import mlx.core as mx
            for case in cases[len(saved):]:
                check_time();start = time.perf_counter();results = []
                for entry in case['steps']:
                    results.append(diagnose_whole_step(case['input_ids'], entry['step'], entry['units'], backend,
                        seed=7, key=f'{case["source_id"]}:{entry["step"]["step_index"]}', controls=3, queries=8))
                    mx.clear_cache()
                record = {'source_id':case['source_id'], 'teacher_output_sha256':case['teacher_output_sha256'],
                          'steps':results, 'seconds':time.perf_counter()-start,
                          'sidecar_only':True, 'v2_selector_changed':False, 'public_test_used':False}
                append(path,record);saved.append(record)
                print(f'whole-span {len(saved)}/{len(cases)}', flush=True)
        save(folder/'summary.json', {'completed_roots':len(saved), 'expected_roots':len(cases),
             'completed_steps':sum(s['status']=='whole_span_diagnostic_complete' for r in saved for s in r['steps']),
             'all_candidate_steps':sum(len(r['steps']) for r in saved),
             'insufficient_exact_controls':sum(s['status']=='insufficient_exact_whole_span_controls' for r in saved for s in r['steps']),
             'diagnostics_sha256':baseline.sha(path), 'sidecar_only':True,
             'distance_matched':False, 'semantic_relevance_matched':False,
             'learning_gain_measured':False, 'v2_selector_changed':False, 'public_test_used':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,required=True)
    parser.add_argument('--prepare-only',action='store_true');args = parser.parse_args()
    main(args.cohort,args.prepare_only)
