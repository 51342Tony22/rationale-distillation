"""Exact, reviewed relation templates. No execution of teacher-supplied code.

Matching an original answer is an arithmetic check, not semantic verification.
Each template specification must have an explicit review record before use.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import string
from fractions import Fraction
from itertools import product


def exact(expression, parameters):
    if not isinstance(expression, str) or len(expression) > 1000:
        raise ValueError('Expression must be a bounded string')
    tree = ast.parse(expression, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 150:
        raise ValueError('Expression too complex')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.Name) and node.id in parameters:
            return Fraction(str(parameters[node.id]))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): result = a + b
            elif isinstance(node.op, ast.Sub): result = a - b
            elif isinstance(node.op, ast.Mult): result = a * b
            else: result = a / b
            if abs(result.numerator) > 10**15 or result.denominator > 10**12:
                raise ValueError('Unreasonable rational magnitude')
            return result
        raise ValueError(f'Unsupported expression node: {type(node).__name__}')
    return visit(tree)


def number(value):
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    divisor = value.denominator
    while divisor % 2 == 0: divisor //= 2
    while divisor % 5 == 0: divisor //= 5
    if divisor == 1:
        from decimal import Decimal, localcontext
        with localcontext() as context:
            context.prec = 60
            return format(Decimal(value.numerator)/Decimal(value.denominator), 'f').rstrip('0').rstrip('.')
    return f'{value.numerator}/{value.denominator}'


def render(template, parameters):
    if len(template) > 6000:
        raise ValueError('Question template too large')
    values={key:number(Fraction(str(value))) for key,value in parameters.items()}
    for _, field, formatting, conversion in string.Formatter().parse(template):
        if field is not None and (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',field) or field not in values or formatting or conversion):
            raise ValueError('Only plain, known parameter placeholders are permitted')
    return template.format(**values)


def parameter_values(spec, base):
    parameters={name:Fraction(str(value)) for name,value in base.items()}
    for name, expression in spec.get('derived',{}).items():
        if name in parameters:
            raise ValueError('A derived name may not overwrite a base parameter')
        parameters[name]=exact(expression,parameters)
    return parameters


def solve(spec, template, base):
    parameters=parameter_values(spec,base)
    answer=exact(template['answer'],parameters)
    if answer < 0:
        raise ValueError('Negative result violates this nonnegative quantity template suite')
    if template.get('integer_answer',True) and answer.denominator != 1:
        raise ValueError('A count answer must be an integer')
    for expression in spec.get('nonnegative_constraints',[]):
        if exact(expression,parameters) < 0:
            raise ValueError('Quantity constraint violated')
    for expression in spec.get('integer_constraints',[]):
        if exact(expression,parameters).denominator != 1:
            raise ValueError('A physical count must be an integer')
    question=render(template['question'],parameters)
    if not question.endswith('?'):
        raise ValueError('A variant must be a complete question')
    return question,number(answer),parameters


def validate_spec(spec, source):
    if spec['source_id'] != source['source_id']:
        raise ValueError('Root mismatch')
    review=spec.get('review',{})
    if review.get('accepted') is not True or not review.get('reason') or review.get('kind') != 'assistant_mathematical_review':
        raise ValueError('Explicit template semantic review required')
    original=exact(spec['original_answer'],parameter_values(spec,spec['original_parameters']))
    if original != Fraction(source['gold']):
        raise ValueError('Reviewed relation does not recover the original reference answer')
    if set(spec['domains']) != set(spec['original_parameters']):
        raise ValueError('All original base parameters require declared domains')
    if 'sampling_parameter_order' in spec:
        order=spec['sampling_parameter_order']
        if len(order)!=len(spec['domains']) or set(order)!=set(spec['domains']):
            raise ValueError('Sampling order must list every base parameter exactly once')
    for name,domain in spec['domains'].items():
        if len(domain) < 4 or len(domain) != len(set(domain)) or len(domain)>100:
            raise ValueError('Each domain needs4..100 distinct values')
    for kind in ['numeric','wording','structural']:
        templates=spec['templates'][kind]
        if len(templates)!=2 or templates[0]['id']==templates[1]['id']:
            raise ValueError('Separate select/audit rendering templates required')
        for template in templates:
            solve(spec,template,spec['original_parameters'])
    return True


def build(spec, source, seed=701):
    validate_spec(spec,source)
    key=int.from_bytes(hashlib.sha256(f'{seed}:{spec["source_id"]}'.encode()).digest()[:8],'big')
    rng=random.Random(key)
    # New cohorts bind this order explicitly. Historical frozen instances lack
    # the metadata and are validated by replaying their stored parameters.
    order=spec.get('sampling_parameter_order',list(spec['domains']))
    domains={name:spec['domains'][name] for name in order}
    seen_tuples=set()
    seen_questions=set()
    result={'select':[],'audit':[]}
    for split_index,split in enumerate(['select','audit']):
        for kind in ['numeric','wording','structural']:
            template=spec['templates'][kind][split_index]
            accepted=0
            for attempt in range(5000):
                base={name:rng.choice(values) for name,values in domains.items()}
                pair=tuple(sorted(base.items()))
                if pair in seen_tuples or base==spec['original_parameters']:
                    continue
                try:
                    question,gold,parameters=solve(spec,template,base)
                except (ValueError,ZeroDivisionError):
                    continue
                if question in seen_questions or question==source['question']:
                    continue
                seen_tuples.add(pair)
                seen_questions.add(question)
                value={'id':f'{spec["source_id"]}:{split}:{kind}:{accepted}',
                       'source_id':spec['source_id'],'split':split,'kind':kind,'template_id':template['id'],
                       'question':question,'gold':gold,'question_sha256':hashlib.sha256(question.encode()).hexdigest(),
                       'base_parameters':base,'parameters':{k:number(v) for k,v in parameters.items()},
                       'answer_expression':template['answer'],
                       'verified_by':'restricted_AST_exact_rational_solver_plus_assistant_reviewed_relation',
                       'review_kind':spec['review']['kind'],'independent_human_review':False}
                result[split].append(value)
                accepted+=1
                if accepted==2:break
            if accepted!=2:
                raise ValueError(f'Insufficient valid distinct instances for {split}/{kind}')
    assert len(result['select'])==len(result['audit'])==6
    return result
