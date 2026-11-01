"""Explicit v2 auxiliary teacher adoption and recipe-specific provenance.

Only the local explainer changes. The original mathematical CoT teacher and
all signed root solutions remain untouched. This module never grants training
or public evaluation permission, and never searches models/prompts.
"""
from __future__ import annotations

from pathlib import Path
from runtime_full import ROOT, HERE, OUT, read, checked_rows, freeze, baseline, now
from budgeted_materials import PROTOCOL
from explanation_teacher_pilot import MODEL, REPO, REVISION
from explanation_teacher_cal34 import FOLDER as DIAGNOSTIC, COHORT
from explanation_pool import (SYSTEM, INSTRUCTIONS, TYPES, unique, inputs,
    selector_paths, reviewed_candidate, reusable_output)
from calibration_cohort import cohort_paths

RECIPE_ID = 'local_explanation_general7b_v2'
RECIPE = OUT / 'local_explanation_general7b_recipe_v2.json'
READY = OUT / 'gates/local_explanation_recipe_ready_v1.json'
DEVELOPMENT = DIAGNOSTIC / 'adopted_development_pools_v2'
FULL = OUT / 'full_explanations_general7b_v2'


def verify_inputs(value):
    if not value.get('inputs'):
        raise ValueError('Recipe needs concrete immutable input evidence')
    for name, checksum in value['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Explanation recipe evidence changed: ' + name)


def validate_evidence(report):
    if (report.get('schema') != 'tgrd_general_explainer_cal34_semantic_report_v1'
            or report.get('completed_unique') != 134
            or report.get('all_outputs_actually_full_text_reviewed') is not True
            or report.get('semantic_accepted') != 73 or report.get('semantic_rejected') != 61
            or report.get('accepted_by_type') != {'relation': 47, 'contrastive': 26}
            or report.get('nll_only_contaminates_joint') is not False
            or report.get('fixed_material_disagreements') != 8
            or report.get('scientific_pool_exported') is not False
            or report.get('learning_or_transfer_evidence') is not False
            or report.get('public_test_used') is not False
            or report.get('selectors', {}).get('joint', {}).get('roots_with_nonempty_fixed_explanation') != 28
            or report.get('selectors', {}).get('nll', {}).get('roots_with_nonempty_fixed_explanation') != 32):
        raise ValueError('Expected the actual bounded134-output evidence, not a fabricated readiness result')


def adopt():
    """Deliberate documented implementation choice under authorized material work."""
    report_path = DIAGNOSTIC / 'semantic_review_v1.json'
    report = read(report_path); verify_inputs(report); validate_evidence(report)
    model_manifest = OUT / 'calibration/explanation_teacher_pilot_v1/model_manifest.json'
    manifest = read(model_manifest)
    if manifest['repo'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Only the fixed already-reviewed general teacher is adopted')
    files = [Path(__file__), HERE / 'explanation_pool.py', report_path, model_manifest,
             OUT / 'transfer128_amendment_v1.json']
    recipe = {'schema': 'tgrd_auxiliary_explainer_recipe_v2', 'protocol': PROTOCOL,
        'recipe_id': RECIPE_ID, 'status': 'adopted_for_reviewed_material_preparation',
        'teacher_role': 'local_explanations_only_for_NLL_joint_and_joint_transfer_pool',
        'teacher_repo': REPO, 'teacher_revision': REVISION,
        'teacher_dir': str(MODEL.relative_to(ROOT)),
        'model_manifest': str(model_manifest.relative_to(ROOT)),
        'root_CoT_teacher_or_outputs_changed': False,
        'system': SYSTEM, 'type_instructions': INSTRUCTIONS,
        'temperature': 0, 'max_new_tokens': 256, 'attempts_per_unique_prompt': 1,
        'student_token_range': [32, 64],
        'cross_selector_reuse': 'same_recipe_exact_source_step_type_and_prompt_including_failed_attempts',
        'different_teacher_cache_reuse_permitted': False,
        'fixed_selection_policy': 'rank0_only_relation_then_contrastive_else_original_CoT',
        'full_text_semantic_and_leakage_review_required': True,
        'one_attempt_no_truncation_no_repair_retry': True,
        'NLL_only_candidates_enter_joint_or_G4': False,
        'development_reuse': 'explicitly_adopt_original134_diagnostic_attempts_and_actual_reviews',
        'old_failed_recipe_materials_preserved': True,
        'adoption_basis': '28_of34_joint_and32_of34_NLL_fixed_explanations_usable;8_material_differences',
        'selection_bias_disclosure': 'auxiliary_teacher_selected_after_train_only_instruction_feasibility_diagnostics',
        'length_matching_disclosure': '14_joint_and15_NLL_development_pools_exceed8_token_target;'
            'no_silent_filter_or_repair;G4_length_confound_must_be_reported_and_audited_separately',
        'length_target_is_a_guarantee': False,
        'method_effectiveness_established': False, 'training_enabled': False,
        'public_test_enabled': False, 'public_test_used': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}}
    freeze(RECIPE, recipe)
    return recipe


def load_recipe(*, verify_model=False):
    recipe = read(RECIPE)
    if (recipe.get('schema') != 'tgrd_auxiliary_explainer_recipe_v2'
            or recipe.get('protocol') != PROTOCOL or recipe.get('recipe_id') != RECIPE_ID
            or recipe.get('teacher_repo') != REPO or recipe.get('teacher_revision') != REVISION
            or recipe.get('teacher_dir') != str(MODEL.relative_to(ROOT))
            or recipe.get('system') != SYSTEM or recipe.get('type_instructions') != INSTRUCTIONS
            or recipe.get('attempts_per_unique_prompt') != 1
            or recipe.get('student_token_range') != [32, 64]
            or recipe.get('root_CoT_teacher_or_outputs_changed') is not False
            or recipe.get('public_test_used') is not False):
        raise ValueError('Expected the explicitly adopted immutable v2 explainer recipe')
    verify_inputs(recipe)
    evidence = read(DIAGNOSTIC / 'semantic_review_v1.json')
    verify_inputs(evidence); validate_evidence(evidence)
    if verify_model:
        manifest = read(ROOT / recipe['model_manifest'])
        for name, spec in manifest['files'].items():
            if baseline.sha(MODEL / name) != spec['sha256']:
                raise ValueError('Adopted local explanation model changed: ' + name)
    return recipe


def diagnostic_registry(requests):
    """Never scan/reuse the old mathematical-teacher explanation namespaces."""
    load_recipe()
    old_requests = unique(read(DIAGNOSTIC / 'requests.json'), 'candidate_id')
    outputs = unique(checked_rows(DIAGNOSTIC / 'generated.jsonl'), 'candidate_id')
    registry = {}
    for request in requests:
        cid = request['candidate_id']
        reused = reusable_output(request, old_requests, outputs)
        if reused is None:
            continue
        if reused.get('scientific_pool_exported') is not False or reused.get('engineering_only') is not True:
            raise ValueError('Expected the original preserved diagnostic output')
        registry[cid] = {'requests_file': str((DIAGNOSTIC / 'requests.json').relative_to(ROOT)),
            'generated_file': str((DIAGNOSTIC / 'generated.jsonl').relative_to(ROOT)),
            'record_sha256': outputs[cid]['record_sha256'], 'recipe_id': RECIPE_ID}
    files = [RECIPE, DIAGNOSTIC / 'requests.json', DIAGNOSTIC / 'generated.jsonl',
             DIAGNOSTIC / 'full_text_reviews_v1.jsonl']
    return registry, {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}


def assemble_development(tokenizer):
    recipe = load_recipe()
    from review_general_explainer import validate_reviews
    requests = read(DIAGNOSTIC / 'requests.json')
    generated = checked_rows(DIAGNOSTIC / 'generated.jsonl')
    reviews = checked_rows(DIAGNOSTIC / 'full_text_reviews_v1.jsonl')
    _, outputs, rev = validate_reviews(requests, generated, reviews, tokenizer)
    sources, teachers = inputs(); paths = cohort_paths(COHORT)
    cases = read(paths.cases); pools = {}; files = [Path(__file__), RECIPE,
        DIAGNOSTIC / 'semantic_review_v1.json', DIAGNOSTIC / 'requests.json',
        DIAGNOSTIC / 'generated.jsonl', DIAGNOSTIC / 'full_text_reviews_v1.jsonl',
        paths.cases, paths.reviews]
    for selector in ('joint', 'nll'):
        selected = selector_paths(COHORT, selector)
        diagnostics = unique(checked_rows(selected.diagnostics), 'source_id')
        view = read(selected.pool_folder / 'requests.json')
        byid = {}
        for case in cases:
            sid = case['source_id']; teacher = teachers[sid]
            byid[sid] = {'source_id': sid, 'question_sha256': sources[sid]['question_sha256'],
                'teacher_output_sha256': teacher['output_sha256'], 'candidates': [{
                    'candidate_id': 'e0', 'completion': teacher['output'], 'student_tokens': 0,
                    'completion_sha256': teacher['output_sha256'], 'explanation_type': 'empty'}],
                'rejected': [], 'development_only': True, 'recipe_id': RECIPE_ID,
                'selector': 'nll_only' if selector == 'nll' else 'joint',
                'position_proposal': diagnostics[sid]['proposal']}
        for request in view:
            sid, cid = request['source_id'], request['candidate_id']
            candidate, reasons = reviewed_candidate(request, outputs[cid], rev[cid],
                                                    sources[sid], teachers[sid], tokenizer)
            if candidate is None:
                byid[sid]['rejected'].append({'candidate_id': cid, 'reasons': reasons})
            else:
                candidate['recipe_id'] = RECIPE_ID
                byid[sid]['candidates'].append(candidate)
        for pool in byid.values():
            lengths = [c['student_tokens'] for c in pool['candidates'][1:]]
            pool.update(nonempty_count=len(lengths),
                nonempty_token_range=max(lengths)-min(lengths) if lengths else 0)
            pool['length_matching_target_max8_satisfied'] = pool['nonempty_token_range'] <= 8
        pools[selector] = list(byid.values())
        freeze(DEVELOPMENT / f'reviewed_pools_{selector}.json', pools[selector])
        files += [selected.diagnostics, selected.pool_folder / 'requests.json',
                  DEVELOPMENT / f'reviewed_pools_{selector}.json']
    manifest = {'schema': 'tgrd_adopted_explainer_development_pools_v2',
        'protocol': PROTOCOL, 'recipe_id': recipe['recipe_id'],
        'root_ids': [c['source_id'] for c in cases],
        'all134_attempts_preserved_including_rejections': True,
        'NLL_only_candidates_enter_joint': False, 'learning_or_transfer_evidence': False,
        'training_enabled': False, 'public_test_used': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in files}}
    freeze(DEVELOPMENT / 'manifest.json', manifest)
    return pools, manifest


def freeze_readiness():
    """Material-generation readiness only; no student-method success claim."""
    load_recipe(verify_model=True)
    from full_explanations import student_tokenizer
    _, manifest = assemble_development(student_tokenizer())
    paths = [RECIPE, DEVELOPMENT / 'manifest.json', DIAGNOSTIC / 'semantic_review_v1.json']
    freeze(READY, {'schema': 'tgrd_local_explanation_recipe_readiness_v2',
        'recipe_id': RECIPE_ID, 'protocol': PROTOCOL, 'ready_for_full_reviewed_generation': True,
        'full_text_review_required_for_every_usable_output': True,
        'development_pool_manifest': str((DEVELOPMENT / 'manifest.json').relative_to(ROOT)),
        'training_enabled': False, 'public_test_enabled': False, 'public_test_used': False,
        'method_effectiveness_established': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}})
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['adopt', 'freeze-readiness'])
    args = parser.parse_args()
    value = adopt() if args.action == 'adopt' else freeze_readiness()
    print({'recipe_id': value['recipe_id'], 'training_enabled': False, 'public_test_used': False})
