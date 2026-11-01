"""Position diagnostics. Mask sensitivity is not an estimate of explanation gain.

No model imports, training, outcome labels or automatic parameter fitting here.
Token positions always refer to the original, unperturbed sequence.
"""
from __future__ import annotations

import hashlib
import math
import random
import statistics


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def checked_losses(values, count):
    values = [float(v) for v in values]
    if len(values) != count or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Expected one finite, nonnegative NLL per original target")
    return values


def delete_context(ids, targets, removed, step_start):
    """Delete only prior-context tokens and remap ORIGINAL, unchanged targets.

    Deletion avoids inventing a decoder [MASK] token. It also shifts positions;
    equal-token-count random deletion is a partial control, not a causal proof.
    """
    targets, removed = list(targets), sorted(set(removed))
    if not targets or targets != sorted(set(targets)):
        raise ValueError("Targets must be nonempty, unique, increasing")
    if not 0 < step_start <= min(targets) <= max(targets) < len(ids):
        raise ValueError("Invalid step/target boundaries")
    if not removed or min(removed) < 0 or max(removed) >= step_start:
        raise ValueError("Only context strictly before the current step can be deleted")
    mapping, result = {}, []
    removed_set = set(removed)
    for index, token in enumerate(ids):
        if index not in removed_set:
            mapping[index] = len(result)
            result.append(token)
    mapped = [mapping[p] for p in targets]
    if min(mapped) < 1:
        raise ValueError("Every target needs an earlier predictor token")
    if [result[p] for p in mapped] != [ids[p] for p in targets]:
        raise ValueError("Perturbation changed a scored target")
    return result, mapped


def attention_units(weights, queries, units, step_start, sequence_length):
    """Validate causal observations, then aggregate within prior semantic units."""
    if len(weights) != len(queries) or not queries:
        raise ValueError("Attention/query count mismatch")
    for query, vector in zip(queries, weights):
        if len(vector) != sequence_length or not 0 <= query < sequence_length:
            raise ValueError("Attention vector shape or query out of range")
        if not all(math.isfinite(v) and v >= 0 for v in vector):
            raise ValueError("Invalid attention weights")
        if abs(sum(vector) - 1) > 1e-5 or any(v != 0 for v in vector[query + 1:]):
            raise ValueError("Attention is not normalized or looks at future tokens")
    ranked = []
    for unit in units:
        positions = unit["positions"]
        if not positions or positions != sorted(set(positions)):
            raise ValueError("Evidence positions must be unique and nonempty")
        if min(positions) < 0 or max(positions) >= step_start:
            raise ValueError("Evidence overlaps current/future step")
        mass = statistics.mean(sum(w[p] for p in positions) for w in weights)
        ranked.append({**unit, "attention_mass": mass,
                       "attention_density": mass / len(positions)})
    return sorted(ranked, key=lambda u: (-u["attention_density"], u["positions"][0]))


def diagnose_step(ids, step, units, backend, *, seed, key, controls=3, queries=8):
    """backend.score(ids, targets); backend.observe(ids, targets, queries).

    observe returns NLLs plus mean-head/mean-layer attention per requested query.
    All forwards end at this step's last scored token; later gold text is absent.
    Random masks have the same token count and context kind as the attended unit.
    They are token subsets, not necessarily coherent full units: report this limit.
    """
    targets = step["targets"]
    prefix = list(ids[:max(targets) + 1])
    original = checked_losses(backend.score(prefix, targets), len(targets))
    chosen_targets = sorted(targets, key=lambda p: (-original[targets.index(p)], p))[:queries]
    predictor_queries = [p - 1 for p in chosen_targets]
    observed, weights = backend.observe(prefix, targets, predictor_queries)
    observed = checked_losses(observed, len(targets))
    difference = max(abs(a - b) for a, b in zip(original, observed))
    if difference > 1e-6:
        raise ValueError("Attention observation changed original target NLLs")
    ranked = attention_units(weights, predictor_queries, units, step["start_position"], len(prefix))
    mean_nll = statistics.mean(original)
    result = {"step_index": step["step_index"], "text": step["text"],
              "char_start": step["char_start"], "char_end": step["char_end"],
              "targets": targets, "target_ids": [ids[p] for p in targets],
              "predictor_queries": predictor_queries, "mean_nll": mean_nll,
              "observation_nll_max_difference": difference,
              "prefix_tokens": len(prefix), "evidence": ranked,
              "status": "no_prior_evidence", "mask_excess": None}
    if not ranked:
        return result
    top = ranked[0]
    positions = top["positions"]
    pool = sorted({p for u in ranked if u["kind"] == top["kind"]
                   for p in u["positions"]} - set(positions))
    result["random_control_pool_tokens"] = len(pool)
    rng = random.Random(stable_seed(seed, key))
    random_masks = set()
    if len(pool) >= len(positions):
        # Unique perturbations, with a finite attempt bound even for small pools.
        for _ in range(controls * 30):
            random_masks.add(tuple(sorted(rng.sample(pool, len(positions)))))
            if len(random_masks) == controls:
                break

    def intervention(removed):
        changed, remapped = delete_context(prefix, targets, removed, step["start_position"])
        losses = checked_losses(backend.score(changed, remapped), len(targets))
        return {"removed_positions": list(removed), "removed_token_ids": [prefix[p] for p in removed],
                "removed_tokens": len(removed), "target_positions_after_deletion": remapped,
                "mean_nll": statistics.mean(losses),
                "delta_nll": statistics.mean(losses) - mean_nll}

    result["attended_mask"] = intervention(positions)
    result["random_masks"] = [intervention(mask) for mask in sorted(random_masks)]
    result["status"] = "insufficient_matched_controls"
    if len(random_masks) == controls:
        deltas = [r["delta_nll"] for r in result["random_masks"]]
        result.update(status="diagnostic_complete", random_delta_mean=statistics.mean(deltas),
                      random_delta_sd=statistics.stdev(deltas) if controls > 1 else None,
                      mask_excess=result["attended_mask"]["delta_nll"] - statistics.mean(deltas))
    return result


def percentile_ranks(values):
    """Within-question average ranks; ties stay tied, not broken by list order."""
    n = len(values)
    if n < 2:
        return [0.5] * n
    return [(sum(x < v for x in values) + (sum(x == v for x in values) - 1) / 2) / (n - 1)
            for v in values]


def exploratory_choices(records, *, seed, key):
    """Frozen uncalibrated equal-rank heuristic, NEVER a validated gain predictor.

    Keep every candidate for NLL/random. Do not compare a different candidate set
    for the combined heuristic: abstain if any candidate lacks matched controls.
    A usefulness-based 'no supplement' threshold needs continuation labels later.
    """
    if not records:
        return {"status": "no_candidates", "nll": None, "random": None, "combined": None}
    best = max(records, key=lambda r: (r["mean_nll"], -r["step_index"]))
    random_record = random.Random(stable_seed(seed, key)).choice(records)
    result = {"status": "insufficient_matched_controls", "nll": best["step_index"],
              "random": random_record["step_index"], "combined": None}
    if all(r["status"] == "diagnostic_complete" for r in records):
        nll_rank = percentile_ranks([r["mean_nll"] for r in records])
        mask_rank = percentile_ranks([r["mask_excess"] for r in records])
        scores = [a + b for a, b in zip(nll_rank, mask_rank)]
        pick = max(range(len(records)), key=lambda i: (scores[i], -records[i]["step_index"]))
        result.update(status="uncalibrated_heuristic_only", combined=records[pick]["step_index"],
                      combined_scores=scores, tie_count=scores.count(scores[pick]))
    return result
