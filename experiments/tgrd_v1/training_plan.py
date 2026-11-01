"""Root-balanced four-epoch schedules for the authorized five-group study."""
from __future__ import annotations

import math
import re
from learning import order_for_epoch
from runtime_full import digest

# Retain old identifiers where their meaning is unchanged. NLL is a NEW group;
# do not relabel historical M2 (random explanations) as the new NLL baseline.
CONDITIONS=('M1','NLL','M3','M7','M6')
CONDITION_LABELS={'M1':'G1 Ordinary-CoT','NLL':'G2 NLL-only',
                  'M3':'G3 Joint-selector','M7':'G4 Joint-selector-plus-transfer',
                  'M6':'G5 Direct-variant-augmentation'}
PROTOCOL='five_groups_seed42_v1'
SEEDS=(42,)
DEFERRED_SEEDS=(7,17)
REFERENCE_SEED=42
EPOCHS=4
ACCUMULATION=5
CHECKPOINT_INTERVAL=100


def validate_materials(bundle,expected_ids=None):
    if bundle.get('schema')!='tgrd_training_materials_v1' or bundle.get('condition') not in CONDITIONS:
        raise ValueError('Unknown formal training material schema or condition')
    rows=bundle['rows'];ids=[r['source_id'] for r in rows]
    if (not rows or len(ids)!=len(set(ids)) or len(ids)>7473
            or any(not re.fullmatch(r'gsm8k:train:(?:0|[1-9]\d*)',sid) for sid in ids)
            or any(int(sid.rsplit(':',1)[1])>=7473 for sid in ids)):
        raise ValueError('Expected unique official training roots only')
    if expected_ids is not None and ids!=expected_ids:
        raise ValueError('Five groups must use the same accepted roots in the same frozen order')
    if bundle.get('candidate_root_count')!=7473:
        raise ValueError('Full official candidate inventory must be disclosed')
    for row in rows:
        materials=row['epoch_materials']
        if len(materials)!=EPOCHS:
            raise ValueError('Exactly one material per root per epoch is required')
        for epoch,material in enumerate(materials):
            if material['kind'] not in ('ordinary_CoT','explained_CoT','verified_Vsel_variant'):
                raise ValueError('Unknown supervised material kind')
            if (bundle['condition']=='M1' or (bundle['condition']=='M6' and epoch in (0,2))) and material['kind']!='ordinary_CoT':
                raise ValueError('Ordinary-CoT group/epochs cannot contain an explanation or a variant')
            if (not material['question'].strip() or not material['completion'].strip()
                    or material['question_sha256']!=digest(material['question'])
                    or material['completion_sha256']!=digest(material['completion'])):
                raise ValueError('Training material text/hash mismatch')
            if material['root_source_id']!=row['source_id']:
                raise ValueError('Material assigned to the wrong training root')
            variant=material['kind']=='verified_Vsel_variant'
            if bundle['condition']!='M6' and variant:
                raise ValueError('Only M6 receives variant supervision')
            if variant and (epoch not in (1,3) or not material.get('variant_id')
                    or material.get('split')!='select' or ':select:' not in material['variant_id']):
                raise ValueError('M6 variant supervision is confined to epochs2/4')
            if not variant and (material.get('variant_id') or material['question_sha256']!=row['question_sha256']):
                raise ValueError('Original-root supervision question changed')
        if bundle['condition']=='M6':
            if materials[0]!=materials[2]:
                raise ValueError('M6 epochs1/3 must use the same original CoT')
            variants=[m for m in (materials[1],materials[3]) if m['kind']=='verified_Vsel_variant']
            if variants and (len(variants)!=2 or variants[0]['variant_id']==variants[1]['variant_id']
                    or variants[0]['question_sha256']==variants[1]['question_sha256']):
                raise ValueError('M6 needs two distinct fixed Vsel instances or ordinary fallback in both epochs')
            if not variants and any(m!=materials[0] for m in materials):
                raise ValueError('M6 without qualified variants must uniformly fall back to ordinary CoT')
        elif any(m!=materials[0] for m in materials[1:]):
            raise ValueError('Selected materials must stay fixed across the four formal epochs')
    return rows


def schedule(root_count,seed):
    if type(root_count) is not int or root_count<1 or root_count>7473 or seed not in SEEDS:
        raise ValueError('Invalid formal root count or training seed')
    result=[]
    for epoch in range(EPOCHS):
        order=order_for_epoch(root_count,seed,epoch)
        for start in range(0,root_count,ACCUMULATION):
            result.append({'update':len(result)+1,'epoch':epoch+1,
                'root_indices':order[start:start+ACCUMULATION],
                'epoch_end':start+ACCUMULATION>=root_count})
    if len(result)!=EPOCHS*math.ceil(root_count/ACCUMULATION):
        raise ValueError('Optimizer update budget does not match the frozen plan')
    return result
