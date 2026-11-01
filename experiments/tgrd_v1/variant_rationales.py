"""CPU-only M6 derivations from observable quantities of reviewed templates.

Generation parameters can contain the unknown answer. They MUST NOT be printed
as givens. Only placeholders actually rendered in the selected question are
available; inaccessible derived quantities are expanded symbolically first.
No audit rationale is generated and no public-test data is read.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import string
from fractions import Fraction
from pathlib import Path

from runtime_full import ROOT,HERE,OUT,DATA,BASE,read,freeze,digest,baseline
from calibration_cohort import COHORTS,cohort_paths
from trial_selection import validate_probes
from variants import build,parameter_values,render,number,exact


def observed_names(template):
    names=[]
    for _,name,formatting,conversion in string.Formatter().parse(template['question']):
        if name is not None:
            if formatting or conversion:raise ValueError('Only plain question placeholders allowed')
            if name not in names:names.append(name)
    return names


def observable_expression(expression,visible,derived,stack=()):
    """Expand inaccessible derived names, never an inaccessible base parameter."""
    if len(stack)>30:raise ValueError('Excessive derived dependency depth')
    root=ast.parse(expression,mode='eval').body
    def expand(node):
        if isinstance(node,ast.Name):
            if node.id in visible:return copy.deepcopy(node)
            if node.id not in derived:
                raise ValueError('Answer expression uses a hidden generation parameter: '+node.id)
            if node.id in stack:raise ValueError('Cyclic derived relation')
            return observable_expression(derived[node.id],visible,derived,stack+(node.id,))
        if isinstance(node,ast.Constant) and type(node.value) is int:return copy.deepcopy(node)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return ast.UnaryOp(op=copy.deepcopy(node.op),operand=expand(node.operand))
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            return ast.BinOp(left=expand(node.left),op=copy.deepcopy(node.op),right=expand(node.right))
        raise ValueError('Unsupported trace expression node')
    result=expand(root)
    if len(list(ast.walk(result)))>150:raise ValueError('Expanded relation too complex')
    return result


def symbolic(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Constant):return str(node.value)
    if isinstance(node,ast.UnaryOp):return ('+' if isinstance(node.op,ast.UAdd) else '-')+'('+symbolic(node.operand)+')'
    operations={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}
    return '('+symbolic(node.left)+' '+operations[type(node.op)]+' '+symbolic(node.right)+')'


def derivation(spec,template,probe):
    if (probe['split']!='select' or probe['template_id']!=template['id']
            or probe['answer_expression']!=template['answer']):
        raise ValueError('M6 accepts the exact frozen selection template only')
    parameters=parameter_values(spec,probe['base_parameters'])
    if render(template['question'],parameters)!=probe['question'] or digest(probe['question'])!=probe['question_sha256']:
        raise ValueError('Variant question changed')
    names=observed_names(template)
    given={name:parameters[name] for name in names}
    tree=observable_expression(template['answer'],set(given),spec.get('derived',{}))
    expression=symbolic(tree)
    # Re-evaluate independently through the restricted rational solver, using
    # only the observed quantities, not the generator's hidden base tuple.
    answer=exact(expression,given)
    if answer!=Fraction(probe['gold']):raise ValueError('Observed-only derivation disagrees with frozen label')
    steps=[];seen=set()
    def operand(value):
        text=number(value)
        return '('+text+')' if '/' in text or text.startswith('-') else text
    def visit(node):
        if isinstance(node,ast.Name):return given[node.id]
        if isinstance(node,ast.Constant):return Fraction(node.value)
        if isinstance(node,ast.UnaryOp):
            value=visit(node.operand);result=value if isinstance(node.op,ast.UAdd) else -value
            numeric=('+' if isinstance(node.op,ast.UAdd) else '-')+'('+number(value)+')'
        else:
            a,b=visit(node.left),visit(node.right)
            operations={ast.Add:('+',lambda:a+b),ast.Sub:('-',lambda:a-b),
                        ast.Mult:('*',lambda:a*b),ast.Div:('/',lambda:a/b)}
            op,calculate=operations[type(node.op)];result=calculate()
            numeric=operand(a)+' '+op+' '+operand(b)
        left=symbolic(node)
        if left not in seen:
            steps.append({'expression':left,'substitution':numeric,'result':number(result)})
            seen.add(left)
        return result
    if visit(tree)!=answer:raise ValueError('Trace and exact expression disagree')
    lines=['The quantities given in the problem are: '+', '.join(name+' = '+number(given[name]) for name in names)+'.',
           'Use the relation '+expression+'.']
    lines.extend(row['expression']+' = '+row['substitution']+' = '+row['result']+'.' for row in steps)
    lines.append('The answer is \\boxed{'+number(answer)+'}.')
    completion='\n'.join(lines)
    return {'variant_id':probe['id'],'root_source_id':probe['source_id'],'split':'select',
        'question':probe['question'],'question_sha256':probe['question_sha256'],'gold':probe['gold'],
        'completion':completion,'completion_sha256':digest(completion),
        'kind':'verified_Vsel_variant','observed_parameters':{name:number(v) for name,v in given.items()},
        'observable_expression':expression,'arithmetic_trace':steps,
        'verification':'reviewed_template_relation_plus_observed_only_exact_rational_trace',
        'program_generated_not_teacher_rationale':True,'independent_human_review':False,
        'probe_sha256':digest(json.dumps(probe,sort_keys=True)),
        'spec_sha256':digest(json.dumps(spec,sort_keys=True))}


def root_rationales(bundle,source):
    probes=validate_probes(bundle,source,build)
    spec=bundle['spec'];result=[]
    for probe in probes['select']:
        template=spec['templates'][probe['kind']][0]
        result.append(derivation(spec,template,probe))
    if len({row['question_sha256'] for row in result})!=6:
        raise ValueError('Six distinct selection instances required')
    return {'source_id':source['source_id'],'selection_rationales':result,
        'M6_fixed_variant_ids':[r['variant_id'] for r in result[:2]],
        'M6_order':'first_two_in_frozen_Vsel_order','Vaudit_rationales_generated':False,
        'development_only':True}


def prepare(cohort):
    from transformers import AutoTokenizer
    from learning import tokenize
    paths=cohort_paths(cohort)
    sources={r['source_id']:r for r in read(DATA/'cal40.json')}
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/BASE['student_dir']),local_files_only=True,trust_remote_code=False)
    bundles=read(paths.variants);rows=[];lengths=[]
    if len(bundles)!=paths.expected_roots:raise ValueError('Incomplete cohort')
    for bundle in bundles:
        row=root_rationales(bundle,sources[bundle['source_id']])
        for material in row['selection_rationales']:
            sample=tokenize(tokenizer,material['question'],material['completion'])
            material['completion_tokens_including_eos']=len(sample['target_ids'])
            lengths.append(len(sample['prompt_ids'])+len(sample['target_ids']))
        rows.append(row)
    inputs=[Path(__file__),HERE/'variants.py',HERE/'trial_selection.py',HERE/'learning.py',
        HERE/'runtime.py',HERE/'runtime_full.py',HERE/'calibration_cohort.py',paths.variants,DATA/'cal40.json',
        ROOT/'experiments/math_distill_v1/config.json']
    target=paths.pool_folder.parent/'verified_Vsel_rationales_v1.json'
    value={'schema':'tgrd_verified_Vsel_rationales_v1','cohort':cohort,'roots':rows,
        'status':'development_materials_prepared_not_formal_training_enabled',
        'inputs':{str(p.relative_to(ROOT)):baseline.sha(p) for p in inputs},
        'rationale_count':len(lengths),'max_training_tokens':max(lengths),
        'M6_order':'first_two_in_frozen_Vsel_order',
        'style_limitation':'program_algebraic_traces_differ_from_teacher_natural_language_CoT',
        'no_generation_hidden_parameter_as_given':True,'public_test_used':False}
    freeze(target,value)
    print({'prepared_roots':len(rows),'verified_Vsel_rationales':len(lengths),
        'max_training_tokens':max(lengths),'GPU_loaded':False,'formal_training_enabled':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=COHORTS,default='cal40_reviewed34_v2')
    prepare(parser.parse_args().cohort)
