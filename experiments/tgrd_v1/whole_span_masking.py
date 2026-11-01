"""Independent whole-sentence/paragraph deletion diagnostics.

This preserves syntactic spans, not certified semantic relevance. It neither
replaces v2 positions nor estimates the learning gain of an explanation.
"""
from __future__ import annotations

import random
import re
import statistics

from runtime_full import BASE
from selector_masking_core import attention_units, checked_losses, delete_context, stable_seed


def aligned_positions(offsets, start, end, shift=0):
    """Reject a span if a tokenizer token straddles either character boundary."""
    if not 0 <= start < end:
        raise ValueError('Invalid character span')
    if any(a < b and ((a < start < b) or (a < end < b)) for a, b in offsets):
        return []
    positions = [i + shift for i, (a, b) in enumerate(offsets) if start <= a < b <= end]
    if positions and positions != list(range(min(positions), max(positions) + 1)):
        return []
    return positions


def whole_units(prompt, question, completion, tokenizer):
    """Question sentences and blank-line-separated prior solution blocks.

    Sentence boundaries are a punctuation heuristic. Solution blocks can include
    headings, equations or anaphora; grammatical/semantic coherence is not proved.
    All punctuation/formatting tokens in an accepted span are included.
    """
    if prompt != BASE['student_prompt'].format(question=question):
        raise ValueError('Prompt differs from the exact question-only template')
    p = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    c = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    if len(p['input_ids']) + len(c['input_ids']) > 4096:
        raise ValueError('Whole-span diagnostic would truncate')
    qstart = prompt.index(question)
    spans = []
    for match in re.finditer(r'.+?(?:[.!?;](?=\s|$)|$)', question, re.S):
        spans.append(('question', 'question_sentence', match.start(), match.end(), qstart, 0,
                      question, p['offset_mapping']))
    # Include the following blank-line delimiter in the preceding block. Do not
    # cut a LaTeX display into separate lines at arbitrary newlines.
    cursor = 0
    for delimiter in re.finditer(r'\n[ \t]*\n', completion):
        spans.append(('prior_reasoning', 'solution_paragraph', cursor, delimiter.end(), 0,
                      len(p['input_ids']), completion, c['offset_mapping']))
        cursor = delimiter.end()
    if cursor < len(completion):
        spans.append(('prior_reasoning', 'solution_paragraph', cursor, len(completion), 0,
                      len(p['input_ids']), completion, c['offset_mapping']))
    accepted, omitted = [], []
    for kind, shape, start, end, char_shift, token_shift, text, offsets in spans:
        block = text[start:end]
        if not block.strip() or '\\boxed' in block:
            continue
        positions = aligned_positions(offsets, start + char_shift, end + char_shift, token_shift)
        unit = {'kind': kind, 'span_type': shape, 'text': block,
                'char_start': start, 'char_end': end, 'positions': positions}
        if not positions:
            omitted.append({**unit, 'reason': 'token_crosses_boundary_or_no_contiguous_tokens'})
        else:
            accepted.append(unit)
    return {'input_ids': p['input_ids'] + c['input_ids'], 'units': accepted, 'omitted_units': omitted}


def matched_controls(top, units, *, seed, key, count=3):
    """Sample whole, disjoint spans of the same source/type/exact token count."""
    if type(count) is not int or count < 1:
        raise ValueError('Positive control count required')
    occupied = set(top['positions'])
    pool = {}
    for unit in units:
        positions = tuple(unit['positions'])
        if (unit['kind'] == top['kind'] and unit['span_type'] == top['span_type']
                and len(positions) == len(top['positions']) and not occupied.intersection(positions)):
            pool[positions] = unit
    available = [pool[k] for k in sorted(pool)]
    chosen = random.Random(stable_seed(seed, key)).sample(available, min(count, len(available)))
    return chosen, len(available)


def diagnose_whole_step(ids, step, units, backend, *, seed, key, controls=3, queries=8):
    targets = step['targets']
    prefix = list(ids[:max(targets) + 1])
    prior = [u for u in units if max(u['positions']) < step['start_position']]
    for unit in prior:
        p = unit['positions']
        if p != list(range(min(p), max(p) + 1)):
            raise ValueError('A whole-span deletion must be contiguous')
    original = checked_losses(backend.score(prefix, targets), len(targets))
    chosen = sorted(zip(targets, original), key=lambda pair: (-pair[1], pair[0]))[:queries]
    predictor_queries = [position - 1 for position, _ in chosen]
    observed, weights = backend.observe(prefix, targets, predictor_queries)
    observed = checked_losses(observed, len(targets))
    difference = max(abs(a - b) for a, b in zip(original, observed))
    if difference > 1e-6:
        raise ValueError('Attention instrumentation changed original NLL')
    ranked = attention_units(weights, predictor_queries, prior, step['start_position'], len(prefix))
    mean_nll = statistics.mean(original)
    result = {'step_index': step['step_index'], 'mean_nll': mean_nll,
              'target_ids': [ids[p] for p in targets], 'targets': list(targets),
              'prefix_tokens': len(prefix), 'observation_nll_max_difference': difference,
              'evidence': ranked, 'mask_excess': None, 'status': 'no_prior_whole_span',
              'control_rule': 'whole_span_same_kind_type_exact_token_count',
              'position_distance_matched': False, 'semantic_relevance_matched': False,
              'selector_positions_changed': False, 'learning_gain_measured': False}
    if not ranked:
        return result
    top = ranked[0]
    controls_selected, pool_size = matched_controls(top, ranked, seed=seed, key=key, count=controls)
    result['matched_control_pool_units'] = pool_size
    result['requested_control_count'] = controls
    result['status'] = 'insufficient_exact_whole_span_controls'
    # Keep an attended deletion even when controls are absent, but never form an
    # excess score from fewer controls or substitute random token fragments.
    def intervention(unit):
        removed = unit['positions']
        changed, remapped = delete_context(prefix, targets, removed, step['start_position'])
        losses = checked_losses(backend.score(changed, remapped), len(targets))
        return {'kind': unit['kind'], 'span_type': unit['span_type'], 'text': unit['text'],
                'removed_positions': list(removed), 'removed_token_ids': [prefix[p] for p in removed],
                'removed_tokens': len(removed), 'target_positions_after_deletion': remapped,
                'distance_to_step': step['start_position'] - max(removed),
                'mean_nll': statistics.mean(losses), 'delta_nll': statistics.mean(losses) - mean_nll}
    result['attended_mask'] = intervention(top)
    result['whole_control_masks'] = [intervention(unit) for unit in controls_selected]
    if len(controls_selected) == controls:
        deltas = [mask['delta_nll'] for mask in result['whole_control_masks']]
        result.update(status='whole_span_diagnostic_complete',
                      random_delta_mean=statistics.mean(deltas),
                      random_delta_sd=statistics.stdev(deltas) if controls > 1 else None,
                      mask_excess=result['attended_mask']['delta_nll'] - statistics.mean(deltas))
    return result
