"""Selection uses Vsel only. Vaudit values are never arguments to this module."""
from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
import hashlib
from variants import validate_spec, solve, number


def summarize(predictions, expected_ids):
    if [row['id'] for row in predictions] != expected_ids or len(set(expected_ids)) != 6:
        raise ValueError('Need all six probes in their frozen order')
    for row in predictions:
        if type(row['correct']) is not bool or not math.isfinite(row['answer_nll']):
            raise ValueError('Probe score is invalid')
    return {'accuracy': sum(row['correct'] for row in predictions)/6,
            'answer_nll': sum(row['answer_nll'] for row in predictions)/6}


def choose(scores, candidates, threshold):
    if threshold not in (0, .01, .03, .10):
        raise ValueError('Threshold must be in the predeclared development grid')
    identifiers = [row['candidate_id'] for row in candidates]
    if len(set(identifiers)) != len(identifiers) or set(scores) != set(identifiers) or 'e0' not in scores:
        raise ValueError('All candidates and the equally trained empty branch must be scored')
    for score in scores.values():
        if not (math.isfinite(score['accuracy']) and 0 <= score['accuracy'] <= 1
                and math.isfinite(score['answer_nll']) and score['answer_nll'] >= 0):
            raise ValueError('Nonfinite or invalid selection score')
    base = scores['e0']; eligible = []; gains = []
    for candidate in candidates:
        cid = candidate['candidate_id']
        if cid == 'e0':
            continue
        gain = {'candidate_id': cid, 'accuracy_gain': scores[cid]['accuracy']-base['accuracy'],
                'nll_gain': base['answer_nll']-scores[cid]['answer_nll'],
                'student_tokens': candidate['student_tokens']}
        gains.append(gain)
        if gain['accuracy_gain'] > 0 and gain['nll_gain'] > threshold:
            eligible.append(gain)
    eligible.sort(key=lambda g: (-g['accuracy_gain'], -g['nll_gain'], g['student_tokens'], g['candidate_id']))
    return {'selected_id': eligible[0]['candidate_id'] if eligible else 'e0',
            'threshold': threshold, 'gains': gains, 'selection_data': 'Vsel_only',
            'abstained': not eligible}


def validate_probes(bundle, root, rebuild):
    spec=bundle['spec']
    validate_spec(spec,root)
    if bundle['source_id']!=root['source_id']:
        raise ValueError('Bundle belongs to a different source')
    # Explicit-order cohorts reproduce the entire draw after JSON serialization.
    # Legacy files did not bind dictionary iteration order; preserve their
    # frozen instances and exactly replay each stored tuple instead of redrawing.
    if 'sampling_parameter_order' in spec and bundle['variants']!=rebuild(spec,root):
        raise ValueError('Probe bundle changed or does not reproduce its exact solver labels')
    probes = bundle['variants']
    if set(probes)!={'select','audit'}:
        raise ValueError('Only selection and audit probe splits are permitted')
    seen_parameters=set();seen_questions=set();seen_ids=set()
    for split in ('select','audit'):
        if len(probes[split]) != 6 or Counter(row['kind'] for row in probes[split]) != {
                'numeric':2, 'wording':2, 'structural':2}:
            raise ValueError('Need two probes of each declared type per split')
        kind_counts=Counter()
        for row in probes[split]:
            if row['source_id'] != root['source_id'] or row['split'] != split:
                raise ValueError('Probe is assigned to the wrong root or split')
            kind=row['kind'];template=spec['templates'][kind][0 if split=='select' else 1]
            if row['template_id']!=template['id'] or row['answer_expression']!=template['answer']:
                raise ValueError('Probe template or answer expression changed')
            base=row['base_parameters']
            if set(base)!=set(spec['domains']) or any(
                    base[name] not in spec['domains'][name] for name in base):
                raise ValueError('Probe base values are outside the reviewed domains')
            pair=tuple(sorted((name,Fraction(str(v))) for name,v in base.items()))
            original=tuple(sorted((name,Fraction(str(v))) for name,v in spec['original_parameters'].items()))
            if pair in seen_parameters or pair==original:
                raise ValueError('Probe reused another split instance or the original parameters')
            question,gold,parameters=solve(spec,template,base)
            if (row['question']!=question or row['gold']!=gold
                    or row['parameters']!={k:number(v) for k,v in parameters.items()}
                    or row['question_sha256']!=hashlib.sha256(question.encode()).hexdigest()):
                raise ValueError('Frozen probe does not match exact template/parameter replay')
            expected_id=f'{root["source_id"]}:{split}:{kind}:{kind_counts[kind]}'
            if row['id']!=expected_id or row['id'] in seen_ids or question in seen_questions or question==root['question']:
                raise ValueError('Probe identity, question uniqueness or ordering changed')
            kind_counts[kind]+=1
            seen_parameters.add(pair);seen_questions.add(question);seen_ids.add(row['id'])
    return probes
