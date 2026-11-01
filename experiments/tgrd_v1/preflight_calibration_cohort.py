"""CPU checks for the actual cohort, provenance and all prospective prompts.

This does not run attention diagnostics, generate explanations, accept their
semantics, or validate the real-model trial driver.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from calibration_cohort import COHORTS, cohort_paths
from explanation_pool import make_request, TYPES, unique
from runtime_full import ROOT, HERE, DATA, OUT, BASE, read, save, checked_rows, baseline, digest
from trial_selection import validate_probes
from variants import build


def main(cohort):
    from transformers import AutoTokenizer
    paths=cohort_paths(cohort)
    relations=unique(read(paths.relations),'source_id')
    bundles=unique(read(paths.variants),'source_id')
    sources=unique(read(DATA/'cal40.json'),'source_id')
    teachers=unique(checked_rows(OUT/'materials/teacher_targets_cal40.jsonl'),'source_id')
    reviews=unique(read(paths.reviews),'source_id')
    cases=unique(read(paths.cases),'source_id')
    if not (set(relations)==set(bundles)==set(cases) and len(cases)==paths.expected_roots):
        raise ValueError('Declared cohort coverage is inconsistent')
    if cohort=='cal40_reviewed34_v2':
        if set(cases)!={sid for sid,r in reviews.items() if r['accepted'] is True}:
            raise ValueError('Expanded cohort must include every accepted cal40 teacher')
        manifest=read(DATA/'cal40_reviewed_manifest_v2.json')
        for name,sha in manifest['inputs'].items():
            if baseline.sha(ROOT/name)!=sha:
                raise ValueError('Reviewed relation manifest input changed: '+name)
    student=AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']),local_files_only=True,
                                          trust_remote_code=False)
    teacher_tok=AutoTokenizer.from_pretrained(str(ROOT/BASE['teacher_dir']),local_files_only=True,
                                              trust_remote_code=False)
    requests=[];question_hashes=set();base_teacher_lengths=[];total_probes=0
    bootstrap=unique(read(DATA/'bootstrap_calibration_variants_v2.json'),'source_id')
    for sid,case in cases.items():
        source=sources[sid];teacher=teachers[sid];review=reviews[sid]
        if (review['accepted'] is not True or review.get('all_text_read') is not True
                or review['output_sha256']!=teacher['output_sha256']
                or review['question_sha256']!=source['question_sha256']
                or case['teacher_output_sha256']!=teacher['output_sha256']
                or digest(teacher['output'])!=teacher['output_sha256']):
            raise ValueError('Calibration source, teacher, case and full-text review mismatch')
        if bundles[sid]['spec']!=relations[sid]:
            raise ValueError('Frozen relation and variant bundle differ')
        probes=validate_probes(bundles[sid],source,build)
        if sid in bootstrap and probes!=bootstrap[sid]['variants']:
            raise ValueError('Original ten-root probe instances changed')
        for split in ('select','audit'):
            for probe in probes[split]:
                if probe['question_sha256'] in question_hashes:
                    raise ValueError('Duplicate question across calibration probes')
                question_hashes.add(probe['question_sha256']);total_probes+=1
        prompt=BASE['student_prompt'].format(question=source['question'])
        total=len(student.encode(prompt,add_special_tokens=False))+len(
            student.encode(teacher['output'],add_special_tokens=False))+1
        if total>4096:raise ValueError('Untruncated e0 training context overflow')
        base_teacher_lengths.append(total)
        for candidate in case['candidates']:
            step=candidate['step']
            for kind in TYPES:
                request=make_request(source,teacher,step,kind,0)
                # Compare the whole user prompt, not a suffix substring that can
                # legitimately repeat an earlier sentence or a given quantity.
                expected_context=('Problem:\n'+source['question']+
                    '\n\nSolution before this step:\n'+teacher['output'][:step['char_start']]+
                    '\n\nStep to explain:\n'+step['text'])
                if request['visible_context']!=expected_context or 'gold' in request:
                    raise ValueError('Explanation prompt crossed its visibility boundary')
                chat=teacher_tok.apply_chat_template([
                    {'role':'system','content':request['system']},
                    {'role':'user','content':request['user']}],tokenize=False,add_generation_prompt=True)
                length=len(teacher_tok.encode(chat,add_special_tokens=False))
                if length+256>4096:raise ValueError('Explanation prompt context overflow')
                requests.append(length)
    report={'status':'passed','device':'CPU_tokenizers_only_no_model_weights',
        'cohort':cohort,'roots':len(cases),'validated_probe_count':total_probes,
        'prospective_requests_all_filtered_positions':len(requests),
        'actual_selection_still_requires_GPU_diagnostics':True,
        'max_actual_requests_after_at_most_two_positions':sum(
            min(2,len(c['candidates']))*len(TYPES) for c in cases.values()),
        'max_teacher_prompt_tokens':max(requests,default=0),
        'max_e0_training_context_including_eos':max(base_teacher_lengths),
        'explanations_actually_generated':False,'explanations_accepted':0,
        'real_trial_driver_validated':False,'public_test_used':False,
        'source_hashes':{str(p.relative_to(ROOT)):baseline.sha(p) for p in [
            Path(__file__),HERE/'calibration_cohort.py',HERE/'explanation_pool.py',HERE/'positions_v2.py',
            HERE/'trial_calibration.py',HERE/'trial_engine.py',HERE/'trial_selection.py',HERE/'static_pool_scores.py',HERE/'variants.py',paths.cases,
            paths.relations,paths.variants,paths.reviews,DATA/'cal40.json',
            OUT/'materials/teacher_targets_cal40.jsonl']}}
    save(OUT/'preflight'/f'{cohort}_material_selection_cpu_v1.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,required=True)
    main(parser.parse_args().cohort)
