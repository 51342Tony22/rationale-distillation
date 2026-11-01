"""G4 exact G3 outside128; G5 observed-only frozen Vsel in epochs2/4.

Production requires replayed actual trials. Missing work is never e0 fallback.
This module creates materials only; it grants no training or test gate.
"""
import argparse
import copy
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,read,freeze,digest,baseline
from budgeted_materials import PROTOCOL,validate_roster,validate_transfer_coverage
from training_plan import validate_materials
from explanation_pool import unique
from learning import tokenize
from transfer128_variants_v2 import normalized_source

FOLDER=OUT/'formal_materials'/PROTOCOL/'transfer128_quality_v3_v1'


def build_bundles(g3,sources,teachers,roster,pools,outcomes,rationales,tokenizer):
    ids=[r['source_id'] for r in validate_materials(g3)]
    selected=validate_roster(roster,ids)
    source=unique(sources,'source_id');teacher=unique(teachers,'source_id')
    bypool=unique(pools,'source_id');byoutcome=unique(outcomes,'source_id')
    byrat={r['selection_rationales'][0]['root_source_id']:r for r in rationales}
    if (len(byrat)!=len(rationales) or set(bypool)!=selected or set(byoutcome)!=selected
            or not set(byrat)<=selected or g3['condition']!='M3' or g3.get('protocol')!=PROTOCOL):
        raise ValueError('Exact128 pools/outcomes and qualified Vsel material only required')
    bundles={g:{**{k:copy.deepcopy(v) for k,v in g3.items() if k not in ('condition','rows','stage')},
        'condition':g,'stage':'finite_transfer128','rows':[]} for g in ('M7','M6')}
    choices=[];max_tokens=0
    def material(s,t,c):
        return {'root_source_id':s['source_id'],'question':s['question'],
            'question_sha256':s['question_sha256'],'completion':c['completion'],
            'completion_sha256':c['completion_sha256'],
            'kind':'ordinary_CoT' if c['candidate_id']=='e0' else 'explained_CoT',
            'selected_candidate_id':c['candidate_id'],
            **({'explanation_quality_provenance':copy.deepcopy(c['quality_provenance'])}
                if c['candidate_id']!='e0' else {})}
    for original in g3['rows']:
        sid=original['source_id'];s=source[sid];t=teacher[sid]
        if (s['question_sha256']!=digest(s['question']) or t['output_sha256']!=digest(t['output'])
                or t['question_sha256']!=s['question_sha256'] or original['question_sha256']!=s['question_sha256']):
            raise ValueError('Original question or teacher changed')
        empty={'candidate_id':'e0','completion':t['output'],'completion_sha256':t['output_sha256']}
        ordinary=material(s,t,empty)
        g5={'source_id':sid,'question_sha256':s['question_sha256'],
            'epoch_materials':[copy.deepcopy(ordinary) for _ in range(4)]}
        if sid not in selected:
            g4=copy.deepcopy(original);cid=original['epoch_materials'][0]['selected_candidate_id']
            policy='exact_G3_outside128'
        else:
            pool=bypool[sid];outcome=byoutcome[sid];cid=outcome['selected_candidate_id']
            candidates=unique(pool['candidates'],'candidate_id')
            if (pool['source_id']!=sid or pool['question_sha256']!=s['question_sha256']
                    or pool['teacher_output_sha256']!=t['output_sha256'] or cid not in candidates
                    or candidates['e0']['completion']!=t['output']):
                raise ValueError('Frozen actual outcome does not match exact joint pool')
            if not outcome['trial_executed']:
                reason=outcome.get('reason')
                if (cid!='e0' or reason not in ('actual_template_rejected','no_automatically_eligible_joint_explanation')
                        or reason=='actual_template_rejected' and sid in byrat
                        or reason=='no_automatically_eligible_joint_explanation' and len(candidates)!=1):
                    raise ValueError('Only genuine qualified-material absence can justify fallback')
            elif sid not in byrat:
                raise ValueError('An actual trial cannot exist without qualified probes')
            candidate=candidates[cid]
            if cid!='e0':
                q=candidate['quality_provenance']
                if q.get('eligible') is not True or q.get('actual_review_accepted') is False:
                    raise ValueError('Automatically rejected explanation cannot enter G4')
            g4={'source_id':sid,'question_sha256':s['question_sha256'],
                'epoch_materials':[copy.deepcopy(material(s,t,candidate)) for _ in range(4)]}
            policy='frozen_Vsel_decision_no_audit_reselection' if outcome['trial_executed'] else outcome['reason']
            if sid in byrat:
                r=byrat[sid];variants=r['selection_rationales']
                if (len(variants)!=6 or r['M6_fixed_variant_ids']!=[v['variant_id'] for v in variants[:2]]
                        or r['Vaudit_rationales_generated'] is not False):
                    raise ValueError('Exact frozen first2Vsel derivations required')
                for epoch,v in zip((1,3),variants[:2]):
                    if v['split']!='select' or v['root_source_id']!=sid:
                        raise ValueError('Audit or foreign-root gradient forbidden')
                    g5['epoch_materials'][epoch]=copy.deepcopy(v)
        for condition,row in [('M7',g4),('M6',g5)]:
            for m in row['epoch_materials']:
                tokens=tokenize(tokenizer,m['question'],m['completion'])
                max_tokens=max(max_tokens,len(tokens['prompt_ids'])+len(tokens['target_ids']))
            bundles[condition]['rows'].append(row)
        choices.append({'source_id':sid,'transfer_budget_member':sid in selected,
            'G4_selected_candidate_id':cid,'G4_policy':policy,
            'G5_variant_ids':[m['variant_id'] for m in g5['epoch_materials'] if m['kind']=='verified_Vsel_variant'],
            'audit_used_to_choose_material':False})
    for bundle in bundles.values():validate_materials(bundle,ids)
    validate_transfer_coverage(g3,bundles['M7'],roster)
    return bundles,choices,{'common_roots':len(ids),'transfer_roots':128,
        'max_context_tokens_including_EOS':max_tokens,'truncation_used':False,
        'G4_exact_G3_outside128':True,'G5_trained_Vsel_is_generalization_evidence':False}


def prepare():
    from transfer128_production_v1 import replay,load_configuration,material_evidence,CONFIGURATION
    value=load_configuration()
    if not value['scientific_trial_enabled']:
        raise ValueError('Unsupported development cannot auto-createG4/G5 formal materials')
    actual=replay();roster,sources,teachers,pools,_,paths=material_evidence()
    from full_selector_materials_quality_v3 import FOLDER as SELECTOR
    from transfer128_variant_manifest_v2 import FOLDER as VARIANTS
    from transfer128_production_v1 import FOLDER as PRODUCTION
    from full_explanations import student_tokenizer
    from train_selector_quality_v3 import verify_inputs
    g3_path=ROOT/read(SELECTOR/'materials_manifest.json')['materials']['M3']['path']
    g3=read(g3_path)
    rationale_path=VARIANTS/'qualified_Vsel_rationales_v2.json';rationale=read(rationale_path)
    verify_inputs(rationale)
    bundles,choices,coverage=build_bundles(g3,list(sources.values()),list(teachers.values()),roster,
        pools,actual['roots'],rationale['roots'],student_tokenizer())
    files={}
    for group,bundle in bundles.items():
        path=FOLDER/(group+'_materials_v1.json');freeze(path,bundle)
        files[group]={'path':str(path.relative_to(ROOT)),'sha256':baseline.sha(path)}
    freeze(FOLDER/'choices_v1.json',choices);freeze(FOLDER/'coverage_v1.json',coverage)
    paths += [Path(__file__),HERE/'training_plan.py',HERE/'learning.py',
        HERE/'budgeted_materials.py',g3_path,SELECTOR/'materials_manifest.json',rationale_path,
        CONFIGURATION,PRODUCTION/'actual_replay_v1.json',FOLDER/'choices_v1.json',FOLDER/'coverage_v1.json']
    paths += [ROOT/n for n in actual['inputs']]
    paths += [ROOT/e['path'] for e in files.values()]
    result={'schema':'tgrd_transfer128_materials_quality_v3_v1','protocol':PROTOCOL,
        'stage':'finite_transfer128','explanation_quality_version':'v3',
        'materials':files,'coverage':coverage,'formal_training_gate_enabled':False,
        'public_test_used':False,'audit_used_to_choose_material':False,
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in dict.fromkeys(paths)}}
    freeze(FOLDER/'materials_manifest_v1.json',result);return result


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();v=prepare()
    print({k:v[k] for k in ('stage','coverage','formal_training_gate_enabled','public_test_used')},flush=True)
