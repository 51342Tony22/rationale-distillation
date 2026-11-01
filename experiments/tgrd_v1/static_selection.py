"""Source-grounded RSR and LALP adaptations for the same reviewed pool.

RSR: arXiv:2601.14249v5, Eq.7; clip rank at100; smaller is better.
LALP: PMLR306 just26a, Eq.3; equal mean of token-mean step log
probabilities under prompt+preceding k steps; larger is better.
Development adaptation: deterministic paragraphs, k=2 (not GLM segmentation).
No update, selection probe, audit probe or public-test result enters these scores.
"""
from __future__ import annotations

import bisect
import math
import re


def rsr(ranks, surprisals, rank_clip=100):
    if len(ranks) != len(surprisals) or not ranks:
        raise ValueError('Rank and surprisal arrays must be nonempty and aligned')
    if any(type(r) is not int or r < 1 for r in ranks):
        raise ValueError('Ranks are one-based integers')
    if any(not math.isfinite(x) or x < 0 for x in surprisals):
        raise ValueError('Invalid token surprisal')
    # Same epsilon as the authors\' released implementation, not tokenwise ratios.
    return sum(min(r,rank_clip) for r in ranks)/max(sum(surprisals),1e-12)


def lalp(step_nlls):
    if not step_nlls or any(not row for row in step_nlls):
        raise ValueError('Every scored step needs at least one token')
    if any(not math.isfinite(x) or x < 0 for row in step_nlls for x in row):
        raise ValueError('Invalid local token surprisal')
    # Equal STEP weight, not a token-weighted concatenated average.
    return -sum(sum(row)/len(row) for row in step_nlls)/len(step_nlls)


def paragraph_spans(text):
    if not text.strip():
        raise ValueError('Cannot segment an empty response')
    boundaries = [0] + [match.end() for match in re.finditer(r'\n[ \t]*\n+',text)] + [len(text)]
    spans = []
    for start,end in zip(boundaries,boundaries[1:]):
        if start == end:
            continue
        if not text[start:end].strip() and spans:
            spans[-1] = (spans[-1][0],end)
        else:
            spans.append((start,end))
    if ''.join(text[start:end] for start,end in spans) != text:
        raise ValueError('Segmentation failed to preserve all response text')
    return spans


def scoring_inputs(tokenizer, prompt, completion, window=2):
    if type(window) is not int or window < 0:
        raise ValueError('Context window must be a nonnegative integer step count')
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    enc = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']; offsets = enc['offset_mapping']
    if not prompt_ids or not ids or len(prompt_ids)+len(ids) > 4096:
        raise ValueError('Empty/oversize scoring sequence; no truncation permitted')
    if len(ids) != len(offsets):
        raise ValueError('Tokenizer offsets do not align with response tokens')
    # Partition the SINGLE original response tokenization. No split token is
    # duplicated or silently dropped; a token crossing a paragraph cut goes next.
    ends = [end for start,end in offsets]
    if ends != sorted(ends):
        raise ValueError('Tokenizer offsets must be monotonic')
    cuts = [0]
    for _,end in paragraph_spans(completion):
        cut = bisect.bisect_right(ends,end)
        if cut > cuts[-1]:
            cuts.append(cut)
    if cuts[-1] != len(ids):
        raise ValueError('Segmentation did not cover the entire response')
    steps = []
    for index,(start,end) in enumerate(zip(cuts,cuts[1:])):
        left = cuts[max(0,index-window)]
        context = prompt_ids + ids[left:start]
        target = ids[start:end]
        steps.append({'step_index':index, 'response_token_span':[start,end],
                      'preceding_step_span':[max(0,index-window),index],
                      'context_ids':context, 'target_ids':target})
    return {'context_ids':prompt_ids, 'target_ids':ids, 'steps':steps,
            'window':window, 'segmentation':'paragraphs_preserve_all_original_response_tokens',
            'eos_scored':False}


class MLXScorer:
    """Project only short hidden-state chunks to the large output vocabulary."""
    def __init__(self, model, chunk_tokens=64):
        if type(chunk_tokens) is not int or chunk_tokens < 1:
            raise ValueError('Positive chunk size required')
        self.model = model
        self.chunk_tokens = chunk_tokens

    def statistics(self, context_ids, target_ids, include_ranks=False):
        import mlx.core as mx
        from runtime_full import check_time
        if not context_ids or not target_ids or len(context_ids)+len(target_ids)>4096:
            raise ValueError('Invalid bounded scoring input')
        self.model.eval()
        inputs = context_ids + target_ids
        hidden = self.model.model(mx.array([inputs[:-1]]))
        mx.eval(hidden)
        ranks = []; surprisals = []
        for start in range(0,len(target_ids),self.chunk_tokens):
            check_time()
            end = min(start+self.chunk_tokens,len(target_ids))
            states = hidden[:,len(context_ids)-1+start:len(context_ids)-1+end]
            if self.model.args.tie_word_embeddings:
                logits = self.model.model.embed_tokens.as_linear(states)[0].astype(mx.float32)
            else:
                logits = self.model.lm_head(states)[0].astype(mx.float32)
            target = mx.array(target_ids[start:end])
            correct = mx.take_along_axis(logits,target[:,None],axis=-1).squeeze(-1)
            nll = mx.logsumexp(logits,axis=-1)-correct
            mx.eval(nll)
            surprisals.extend(nll.tolist())
            if include_ranks:
                # Exact strict-greater tie convention from rsr_cal.py:L174-176.
                # Full comparisons with small projection chunks equal top100+clip.
                rank = mx.minimum(1+mx.sum(logits>correct[:,None],axis=-1),100)
                mx.eval(rank); ranks.extend(rank.tolist())
            del logits,states
        del hidden
        mx.clear_cache()
        if any(not math.isfinite(v) or v < 0 for v in surprisals):
            raise ValueError('Invalid model surprisal')
        return {'ranks':ranks, 'surprisals':surprisals}

    def score(self, prepared):
        global_stats = self.statistics(prepared['context_ids'],prepared['target_ids'],include_ranks=True)
        local_stats = [self.statistics(step['context_ids'],step['target_ids']) for step in prepared['steps']]
        return {'RSR':rsr(global_stats['ranks'],global_stats['surprisals']),
                'LALP':lalp([row['surprisals'] for row in local_stats]),
                'GALP':-sum(global_stats['surprisals'])/len(global_stats['surprisals']),
                'global_token_statistics':global_stats,
                'local_steps':[{**{key:step[key] for key in ('step_index','response_token_span','preceding_step_span')},
                                'context_tokens':len(step['context_ids']),
                                'target_tokens':len(step['target_ids']),
                                'surprisals':stats['surprisals']}
                               for step,stats in zip(prepared['steps'],local_stats)],
                'local_window_steps':prepared['window'], 'eos_scored':False,
                'segmentation':prepared['segmentation']}


def select_static(candidate_scores, metric):
    if metric not in ('RSR','LALP') or len({r['candidate_id'] for r in candidate_scores}) != len(candidate_scores):
        raise ValueError('Unknown metric or duplicate candidate')
    nonempty = [row for row in candidate_scores if row['candidate_id']!='e0']
    if not nonempty:
        return 'e0'
    if any(not math.isfinite(row[metric]) for row in nonempty):
        raise ValueError('Invalid candidate score')
    direction = 1 if metric=='RSR' else -1
    return min(nonempty,key=lambda row:(direction*row[metric],row['student_tokens'],row['candidate_id']))['candidate_id']
