"""Tokenizer-only qualification of frozen128 probes/rationales; no MLX model."""
from pathlib import Path
from runtime_full import ROOT,HERE,OUT,read,freeze,digest,baseline,BASE
from learning import tokenize
from explanation_pool import unique
from train_selector_quality_v3 import verify_inputs
from transfer128_prepare_v1 import FOLDER as PREPARATION
from transfer128_variant_manifest_v2 import FOLDER as VARIANTS
from transfer128_variants_v2 import normalized_source

FOLDER=OUT/'transfer128/context_v1'


def check_context(roots,sources,teachers,bundles,rationales,tokenizer):
    source_map=unique(sources,'source_id');teacher_map=unique(teachers,'source_id')
    bybundle=unique(bundles,'source_id');byrationale={}
    for r in rationales:
        sid=r['selection_rationales'][0]['root_source_id']
        if sid in byrationale:raise ValueError('Duplicate rationale root')
        byrationale[sid]=r
    if set(bybundle)!=set(byrationale) or not set(bybundle)<=set(roots):
        raise ValueError('Complete qualified bundle/rationale coverage required')
    rows=[];max_training=max_inference_prompt=0;probe_count=rationale_count=0
    def training(question,completion):
        nonlocal max_training
        tokens=tokenize(tokenizer,question,completion)
        n=len(tokens['prompt_ids'])+len(tokens['target_ids']);max_training=max(max_training,n)
        return n
    def inference(question,gold):
        nonlocal max_inference_prompt
        n=len(tokenizer.encode(BASE['student_prompt'].format(question=question),add_special_tokens=False))
        if n+1024>4096:raise ValueError('Scientific1024-token decoding would exceed context')
        training(question,gold);max_inference_prompt=max(max_inference_prompt,n);return n
    for sid in roots:
        source=source_map[sid];teacher=teacher_map[sid]
        if (source['question_sha256']!=digest(source['question'])
                or teacher['question_sha256']!=source['question_sha256']
                or teacher['output_sha256']!=digest(teacher['output'])):
            raise ValueError('Original root/teacher changed')
        normalized,binding=normalized_source(source)
        row={'source_id':sid,'numeric_reference_binding':binding,
            'original_training_tokens_including_EOS':training(source['question'],teacher['output']),
            'original_inference_prompt_tokens':inference(source['question'],normalized['gold']),
            'template_qualified':sid in bybundle,'probe_count':0,'Vsel_rationale_count':0}
        if sid in bybundle:
            bundle=bybundle[sid];r=byrationale[sid]
            select=bundle['variants']['select'];audit=bundle['variants']['audit']
            if len(select)!=6 or len(audit)!=6:raise ValueError('Exactly6+6 frozen probes required')
            byprobe=unique(select+audit,'id')
            for split,probes in [('select',select),('audit',audit)]:
                for p in probes:
                    if (p['source_id']!=sid or p['split']!=split
                            or digest(p['question'])!=p['question_sha256']):
                        raise ValueError('Probe identity changed')
                    inference(p['question'],p['gold']);probe_count+=1;row['probe_count']+=1
            rationales=r['selection_rationales']
            if (len(rationales)!=6 or [x['variant_id'] for x in rationales]!=[p['id'] for p in select]
                    or r['M6_fixed_variant_ids']!=[p['id'] for p in select[:2]]
                    or r['Vaudit_rationales_generated'] is not False):
                raise ValueError('Only the exact frozen Vsel rationales can enter training')
            for x in rationales:
                p=byprobe[x['variant_id']]
                if (x['split']!='select' or x['question']!=p['question'] or x['gold']!=p['gold']
                        or x['root_source_id']!=sid or x['question_sha256']!=p['question_sha256']
                        or x['probe_sha256']!=digest(__import__('json').dumps(p,sort_keys=True))
                        or digest(x['completion'])!=x['completion_sha256']):
                    raise ValueError('Frozen rationale/probe binding changed')
                training(x['question'],x['completion']);rationale_count+=1;row['Vsel_rationale_count']+=1
        rows.append(row)
    return {'roots':rows,'frozen_root_count':len(roots),'qualified_root_count':len(bundles),
        'probes_checked':probe_count,'Vsel_rationales_checked':rationale_count,
        'max_training_tokens_including_EOS':max_training,'max_inference_prompt_tokens':max_inference_prompt,
        'generation_limit':1024,'context_limit':4096,'truncation_used':False,
        'scientific_trial_enabled':False,'public_test_used':False}


def prepare():
    from full_selector import load_quality
    from full_explanations import student_tokenizer
    from transfer128_sampling import ROSTER
    manifest=read(VARIANTS/'manifest_v2.json');verify_inputs(manifest)
    for name,sha in manifest['outputs'].items():
        if baseline.sha(ROOT/name)!=sha:raise ValueError('Frozen variant output changed')
    parent=read(PREPARATION/'manifest.json');verify_inputs(parent)
    bundles=read(VARIANTS/'qualified_bundles_v2.json');rationales=read(VARIANTS/'qualified_Vsel_rationales_v2.json')
    verify_inputs(bundles);verify_inputs(rationales)
    sources,teachers,_,_=load_quality();roster=read(ROSTER)
    value=check_context(roster['source_ids'],sources,teachers,bundles['roots'],rationales['roots'],student_tokenizer())
    if (value['frozen_root_count'],value['qualified_root_count'],value['probes_checked'],value['Vsel_rationales_checked'])!=(128,124,1488,744):
        raise ValueError('All original128 and124 qualified suites must be checked')
    paths=[Path(__file__),HERE/'learning.py',HERE/'transfer128_variants_v2.py',ROSTER,
        PREPARATION/'manifest.json',VARIANTS/'manifest_v2.json',
        VARIANTS/'qualified_bundles_v2.json',VARIANTS/'qualified_Vsel_rationales_v2.json']
    value.update(schema='tgrd_transfer128_context_preflight_v1',status='passed',
        explanation_contexts_checked=False,
        remaining_explanation_context_check='v3 automatic_decision on exact inserted outputs',
        inputs={str(p.relative_to(ROOT)):baseline.sha(p) for p in paths})
    freeze(FOLDER/'context_preflight_v1.json',value)
    return value


if __name__=='__main__':
    value=prepare();print({k:v for k,v in value.items() if k not in ('roots','inputs')},flush=True)
