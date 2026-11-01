"""Build truthful G1-G3 materials from automatic explanation eligibility v3.

No synthetic semantic reviews are passed to the legacy strict review builder.
Each pool candidate links to its exact automatic decision and actual evidence,
if any. Same ID/text has the same decision in both selectors.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from runtime_full import ROOT, HERE, OUT, read, freeze, digest, baseline
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from full_explanations import load_requests, FOLDER as GENERATIONS, student_tokenizer
from full_selector import load_quality, quality_decisions
from explanation_pool import unique, insert_explanation
from material_builder import fixed_choice
from training_plan import validate_materials
from learning import tokenize
import explanation_quality_v3 as quality

FOLDER = OUT / 'formal_materials' / PROTOCOL / 'selector_first_quality_v3'


def build_pools(cases, sources, teachers, diagnostics, views, outputs, decisions):
    generated = unique(outputs, 'candidate_id'); checked = unique(decisions, 'candidate_id')
    result = {}
    for selector in ('joint', 'nll'):
        pools = {}
        for case, diagnostic in zip(cases, diagnostics[selector]):
            sid = case['source_id']; teacher = teachers[sid]
            pools[sid] = {'source_id': sid, 'question_sha256': sources[sid]['question_sha256'],
                'teacher_output_sha256': teacher['output_sha256'],
                'selector': 'nll_only' if selector == 'nll' else 'joint',
                'position_proposal': diagnostic['proposal'], 'diagnostic_steps': diagnostic['steps'],
                'position_ranks_generated': [0],
                'quality_mode': 'uniform_automatic_plus_fixed128_v3', 'G4_full_trial_pool': False,
                'candidates': [{'candidate_id':'e0','completion':teacher['output'],
                    'completion_sha256':teacher['output_sha256'],'student_tokens':0,'explanation_type':'empty'}],
                'rejected': []}
        for request in views[selector]:
            cid = request['candidate_id']; sid = request['source_id']; decision = checked[cid]
            output = generated[cid]
            if (decision['output_sha256'] != output['output_sha256']
                    or decision['request_sha256'] != output['request_sha256']):
                raise ValueError('Eligibility differs from exact output/prompt')
            if not decision['eligible']:
                pools[sid]['rejected'].append({'candidate_id':cid,'reasons':decision['reasons']});continue
            completion=insert_explanation(teachers[sid]['output'],request['step'],output['output'])
            pools[sid]['candidates'].append({'candidate_id':cid,'source_id':sid,
                'teacher_output_sha256':teachers[sid]['output_sha256'],
                'text':output['output'],'text_sha256':output['output_sha256'],
                'completion':completion,'completion_sha256':digest(completion),
                'step_index':request['step']['step_index'],'position_rank':0,
                'explanation_type':request['explanation_type'],'student_tokens':decision['student_tokens'],
                'quality_provenance':copy.deepcopy(decision)})
        result[selector]=pools
    return result


def build_bundles(cases,sources,teachers,root_reviews,pools,tokenizer):
    from automatic_teacher_quality import validate_decision
    from positions_nll import propose_nll
    from positions_v2 import propose
    ids=[c['source_id'] for c in cases]
    if any(set(x)!=set(ids) for x in (sources,teachers,root_reviews,pools['joint'],pools['nll'])):
        raise ValueError('Same full automatic teacher root coverage required')
    bundles={g:dict(schema='tgrd_training_materials_v1',condition=g,protocol=PROTOCOL,
        stage='selector_first',explanation_quality_version='v3',candidate_root_count=7473,
        development_preview_only=False,formal_training_gate_required=True,rows=[]) for g in SELECTOR_CONDITIONS}
    choices=[]
    for case in cases:
        sid=case['source_id'];source=sources[sid];teacher=teachers[sid]
        validate_decision(source,teacher,root_reviews[sid])
        if case['teacher_output_sha256']!=teacher['output_sha256']:raise ValueError('Wrong teacher')
        steps={c['step']['step_index']:c['step'] for c in case['candidates']}
        by={}
        for selector in ('joint','nll'):
            pool=pools[selector][sid];proposal=pool['position_proposal']
            builder=propose_nll if selector=='nll' else propose
            scores=pool['diagnostic_steps']
            if set(s['step_index'] for s in scores)!=set(steps) or builder(scores)!=proposal:
                raise ValueError('Full all-step selector proposal changed')
            by[selector]=unique(pool['candidates'],'candidate_id')
            for c in pool['candidates']:
                if digest(c['completion'])!=c['completion_sha256']:raise ValueError('Changed completion')
                if c['candidate_id']=='e0':
                    if c['completion']!=teacher['output']:raise ValueError('Changed ordinary fallback')
                elif (c['step_index'] not in proposal['positions'][:1]
                        or c['quality_provenance']['eligible'] is not True
                        or c['quality_provenance']['actual_review_accepted'] is False
                        or c['completion']!=insert_explanation(teacher['output'],steps[c['step_index']],c['text'])):
                    raise ValueError('Candidate outside selected position/quality rules')
        for cid in set(by['joint']) & set(by['nll']):
            if by['joint'][cid]!=by['nll'][cid]:raise ValueError('Shared candidate must be identical')
        selected={'M1':'e0','NLL':fixed_choice(list(by['nll'].values())),
                  'M3':fixed_choice(list(by['joint'].values()))}
        for g,cid in selected.items():
            candidate=by['nll' if g=='NLL' else 'joint'][cid]
            sample=tokenize(tokenizer,source['question'],candidate['completion'])
            m=dict(root_source_id=sid,question=source['question'],question_sha256=source['question_sha256'],
                completion=candidate['completion'],completion_sha256=candidate['completion_sha256'],
                kind='ordinary_CoT' if cid=='e0' else 'explained_CoT',selected_candidate_id=cid,
                completion_tokens_including_eos=len(sample['target_ids']))
            if cid!='e0':m['explanation_quality_provenance']=candidate['quality_provenance']
            bundles[g]['rows'].append(dict(source_id=sid,question_sha256=source['question_sha256'],
                epoch_materials=[copy.deepcopy(m) for _ in range(4)]))
        choices.append(dict(source_id=sid,selected_candidate_ids=selected,
            fallback_reasons={g:'no_qualified_rank0_explanation' for g in ('NLL','M3') if selected[g]=='e0'}))
    for bundle in bundles.values():validate_materials(bundle,ids)
    coverage=dict(accepted_root_count=len(ids),common_root_order=True,
        all_explanations_text_read=False,semantic_correctness_verified=False,public_test_used=False,
        budget_equivalence='same_roots_order_initialization_and_optimizer_updates;token_totals_reported',
        groups={g:dict(root_exposures=4*len(ids),
            roots_with_explanation=sum(r['epoch_materials'][0]['kind']=='explained_CoT' for r in b['rows']),
            ordinary_only_roots=sum(r['epoch_materials'][0]['kind']=='ordinary_CoT' for r in b['rows']),
            supervised_tokens_including_eos=sum(m['completion_tokens_including_eos'] for r in b['rows'] for m in r['epoch_materials'])) for g,b in bundles.items()})
    return bundles,choices,coverage


def prepare():
    quality.export()
    parent,cases,sources,teachers,diagnostics,views,requests=load_requests()
    from review_full_explanations import complete_prefix
    pools=build_pools(cases,sources,teachers,diagnostics,views,
        complete_prefix(GENERATIONS/'generated.jsonl'),read(quality.FOLDER/'automatic_decisions_v3.json'))
    _,_,root_quality,ids=load_quality()
    root_reviews=unique([x for x in quality_decisions(root_quality) if x['source_id'] in set(ids)],'source_id')
    bundles,choices,coverage=build_bundles(cases,sources,teachers,root_reviews,pools,student_tokenizer())
    for s,rows in pools.items():freeze(FOLDER/f'automatic_pools_{s}_v3.json',list(rows.values()))
    for g,b in bundles.items():freeze(FOLDER/f'{g}.json',b)
    freeze(FOLDER/'choices.json',choices);freeze(FOLDER/'coverage.json',coverage)
    paths=[Path(__file__),quality.EXPORT,quality.REPORT,quality.AUTHORITY,
        quality.FOLDER/'automatic_decisions_v3.json',HERE/'explanation_quality_v3.py',
        GENERATIONS/'requests_manifest.json',OUT/'transfer128_amendment_v1.json',
        FOLDER/'choices.json',FOLDER/'coverage.json']
    paths += [ROOT/p for p in read(quality.EXPORT)['inputs']]
    paths += [ROOT/p for p in parent['inputs']]
    paths += [FOLDER/f'automatic_pools_{s}_v3.json' for s in ('joint','nll')]
    manifest=dict(schema='tgrd_selector_stage_materials_v1',protocol=PROTOCOL,stage='selector_first',
        explanation_quality_version='v3',conditions=list(SELECTOR_CONDITIONS),formal_seeds=[42],epochs=4,
        candidate_root_count=7473,common_root_ids=ids,G4_or_G5_materials_required=False,
        formal_training_enabled=False,public_test_used=False,
        inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in paths},
        materials={g:dict(path=str((FOLDER/f'{g}.json').relative_to(ROOT)),sha256=baseline.sha(FOLDER/f'{g}.json')) for g in SELECTOR_CONDITIONS})
    freeze(FOLDER/'materials_manifest.json',manifest);print(coverage,flush=True);return manifest

if __name__=='__main__':prepare()
