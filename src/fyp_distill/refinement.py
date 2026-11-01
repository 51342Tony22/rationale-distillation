"""Offline, auditable construction of student-guided rationale conditions.

No API calls or training occur here. Teacher responses and human quality checks
are explicit input artifacts; answer correctness alone is not a quality review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from .common import load_config, read_jsonl, resolve_path, write_jsonl

GROUPS = (
    "original_rationale", "generic_expansion", "random_expansion",
    "hint_at_start", "local_refinement", "local_unfiltered",
)
TEMPLATE = (
    "Solve the following problem step by step. "
    "End with `Final answer: <answer>`.\n\n{question}"
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def stable_seed(seed: int, *parts) -> int:
    return int(digest([seed, *parts])[:8], 16)


def write_frozen(path: Path, rows: list[dict]) -> None:
    """Never silently replace data or invalidate an existing diagnostic cache."""
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Existing artifact differs; use a new run directory: {path}")
        return
    write_jsonl(path, rows)


def unique_index(rows: list[dict], key: str = "id") -> dict:
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} values")
    return indexed


def question_key(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def segment(text: str, max_boundaries: int, seed: int) -> list[int]:
    """Offsets into the ORIGINAL text; never split decimals or rewrite equations.

    These are candidate boundaries, not verified semantic or causal breakpoints.
    A later review must exclude prefixes that already reveal the result.
    """
    final = re.search(r"(?im)^\s*Final\s+answer\s*:", text)
    if final is None:
        return []
    body = text[:final.start()].rstrip()
    offsets = [m.end() for m in re.finditer(r"\n+", body) if body[:m.start()].strip()]
    # The first incomplete prefix and the full solution are never candidates.
    return sorted(random.Random(seed).sample(offsets, min(max_boundaries, len(offsets))))


def correct(text: str, gold: str, dataset: str, protocol: str = "first_final_line_v2") -> bool:
    # Score the first solution, accepting harmless markup and math delimiters.
    if protocol == "first_answer_v3":
        from .answer_protocol_v3 import first_final_answer
    elif protocol == "first_final_line_v2":
        from .answer_protocol import first_final_answer
    else:
        raise ValueError(f"Unknown answer protocol: {protocol}")
    answer = first_final_answer(text)
    if answer is None:
        return False
    if dataset == "math":
        from .math_grader import grade_math_answer
        return grade_math_answer(answer, gold)
    from .common import extract_answer
    return extract_answer("Final answer: " + answer) == gold


def prepare(config: dict) -> dict:
    settings = config["refinement"]
    root = resolve_path(settings["root"])
    train = unique_index(read_jsonl(config["data"]["train_file"]))
    if not train:
        raise ValueError("Missing frozen training pool")
    heldout_keys = set()
    for name in ("dev_file", "test_file", "svamp_file", "math500_file"):
        for row in read_jsonl(config["data"].get(name, "__missing__")):
            heldout_keys.add(question_key(row["question"]))
    teacher = read_jsonl(config["data"]["teacher_file"])
    unique_index(teacher)
    frozen, excluded = [], []
    for row in sorted(teacher, key=lambda x: stable_seed(config["seed"], x["id"])):
        source = train.get(row["id"])
        if (not source or question_key(source["question"]) != question_key(row["question"])
                or source["gold_answer"] != row["gold_answer"]):
            raise ValueError(f"Teacher row not in TRAIN pool: {row['id']}")
        if question_key(row["question"]) in heldout_keys:
            excluded.append({"id": row["id"], "reason": "heldout_exact_overlap"})
            continue
        candidates = [c for c in row["candidates"] if c.get("is_usable") and
                      correct(c["text"], source["gold_answer"], settings["dataset"])]
        if not candidates:
            excluded.append({"id": row["id"], "reason": "no_usable_candidate"})
            continue
        candidates.sort(key=lambda x: x["candidate_id"])
        rng = random.Random(stable_seed(config["seed"], row["id"], "base"))
        candidate = rng.choice(candidates)
        boundaries = segment(candidate["text"], settings["max_boundaries"],
                             stable_seed(config["seed"], row["id"], "boundaries"))
        if not boundaries:
            excluded.append({"id": row["id"], "reason": "no_multistep_boundary"})
            continue
        record = {
            "id": row["id"], "split": "train", "dataset": settings["dataset"],
            "question": row["question"], "gold_answer": row["gold_answer"],
            "teacher_model": row["teacher_model"], "candidate_id": candidate["candidate_id"],
            "original": candidate["text"], "boundaries": boundaries,
            "random_boundary": rng.choice(boundaries),
            "review": {"correct_steps": False, "prefixes_do_not_reveal_answer": False},
        }
        record["source_hash"] = digest({k: v for k, v in record.items() if k != "review"})
        frozen.append(record)
        if len(frozen) >= settings["question_limit"]:
            break
    if not frozen:
        raise ValueError("No eligible training examples")
    # Reviews may be added after freezing; rerunning preparation must preserve
    # them while still rejecting changes to question/rationale/boundary content.
    prior = unique_index(read_jsonl(root / "bases.jsonl"))
    for record in frozen:
        previous = prior.get(record["id"])
        if previous and previous["source_hash"] == record["source_hash"]:
            record["review"] = previous.get("review", record["review"])
    write_frozen(root / "bases.jsonl", frozen)
    write_frozen(root / "prepare_exclusions.jsonl", excluded)
    return {"questions": len(frozen), "candidate_boundaries": sum(len(x["boundaries"]) for x in frozen),
            "excluded": len(excluded), "status": "answer_checked; step/prefix review required",
            "output": str(root / "bases.jsonl")}


def screen_index(bases: list[dict], screening: list[dict], expected_samples: int | None = None) -> dict:
    indexed = unique_index(screening)
    result = {}
    for base in bases:
        row = indexed.get(base["id"])
        if row is None or row["source_hash"] != base["source_hash"]:
            raise ValueError(f"Missing/stale screening: {base['id']}")
        if set(int(k) for k in row["boundaries"]) != set(base["boundaries"]):
            raise ValueError("Screening does not cover all frozen boundaries")
        for trials in row["boundaries"].values():
            if not trials or (expected_samples is not None and len(trials) != expected_samples):
                raise ValueError("Incomplete screening trials")
            if len({x["seed"] for x in trials}) != len(trials):
                raise ValueError("Duplicated screening seeds")
        result[base["id"]] = row
    return result


def hardest(base: dict, screen: dict) -> int:
    return min(base["boundaries"], key=lambda offset: (
        sum(x["correct"] for x in screen["boundaries"][str(offset)]) /
        len(screen["boundaries"][str(offset)]), offset))


def make_requests(config: dict, limit: int | None = None) -> dict:
    settings = config["refinement"]
    root = resolve_path(settings["root"])
    bases = read_jsonl(root / "bases.jsonl")
    if limit is not None:
        bases = bases[:limit]
    if not bases:
        raise ValueError("Prepare bases first")
    screens = screen_index(bases, read_jsonl(root / "screening.jsonl"), settings["screening_samples"])
    requests = []
    for base in bases:
        boundary = hardest(base, screens[base["id"]])
        failures = [x["text"] for x in screens[base["id"]]["boundaries"][str(boundary)]
                    if not x["correct"]][:2]
        for kind in ("local", "generic", "random"):
            for index in range(settings["local_candidates"] if kind == "local" else 1):
                allowed = base["boundaries"] if kind == "generic" else [
                    boundary if kind == "local" else base["random_boundary"]]
                context = {
                    "problem": base["question"], "correct_solution": base["original"],
                    "allowed_insertion_offsets": allowed,
                    "prefixes": {str(x): base["original"][:x] for x in allowed},
                }
                if kind == "local":
                    context["student_failed_continuations"] = failures
                    review_path = root / "screening_failure_review.jsonl"
                    failure_reviews = [r for r in read_jsonl(review_path)
                                       if r.get("source_id") == base["id"] and r.get("offset") == boundary]
                    if failure_reviews:
                        context["reviewed_failure_types"] = failure_reviews
                prompt = (
                    "Add one short, mathematically valid explanatory bridge to this solution. "
                    "Do not rewrite the solution. Explain a definition, relation or omitted step. "
                    "Do not reveal the final answer or complete the remaining solution. "
                    "The explanation must make sense both at the chosen position and at the start. "
                    f"Use at most {settings['max_hint_tokens']} student-tokenizer tokens. "
                    "Return only JSON with keys hint (string) and offset (integer from the allowed list).\n"
                    + json.dumps(context, ensure_ascii=False)
                )
                if kind == "local" and settings.get("request_namespace") == "trainpilot21_v3":
                    prompt = (
                        "Target the specific missing relation or rule in the student's failed continuations. "
                        "A formatting failure is not evidence of a mathematical misunderstanding. "
                        "When a failure contains incorrect arithmetic or invalid simplification, explain "
                        "the applicable general check, rather than just repeating the solution. "
                        "If no conceptual error is evident, do not invent one. Keep the mathematical "
                        "content focused and do not add chat-format instructions or the final answer.\n"
                        + prompt
                    )
                namespace = settings.get("request_namespace")
                request_id = f"{base['id']}:{kind}:{index}"
                if namespace:
                    request_id = f"{namespace}:{request_id}"
                requests.append({
                    "id": request_id, "source_id": base["id"],
                    "source_hash": base["source_hash"], "kind": kind,
                    "allowed_offsets": allowed, "model": base["teacher_model"],
                    "prompt": prompt, "request_hash": digest(prompt),
                })
    write_frozen(root / "teacher_requests.jsonl", requests)
    return {"requests": len(requests), "api_calls_made": 0,
            "output": str(root / "teacher_requests.jsonl")}


def insert(original: str, hint: str, offset: int) -> str:
    if not hint:
        return original
    if not 0 <= offset <= len(original):
        raise ValueError("Invalid insertion offset")
    return original[:offset] + hint + "\n\n" + original[offset:]


def reviewed_hint(hint: dict, base: dict, count_tokens, max_tokens: int) -> None:
    if hint["source_hash"] != base["source_hash"]:
        raise ValueError("Hint belongs to a different source rationale")
    if hint["offset"] not in base["boundaries"]:
        raise ValueError("Hint offset outside frozen boundaries")
    if not hint.get("review", {}).get("valid_explanation") or not hint["review"].get("no_answer_leak"):
        raise ValueError(f"Unreviewed hint: {hint['id']}")
    if not isinstance(hint.get("hint"), str) or not hint["hint"].strip():
        raise ValueError("Empty hint")
    if re.search(r"(?i)final\s+answer|\\boxed|####", hint["hint"]):
        raise ValueError("Answer marker in hint")
    if not 1 <= count_tokens(hint["hint"]) <= max_tokens:
        raise ValueError("Hint exceeds the student-tokenizer length limit")


def build_conditions(bases, screens, hints, verification, settings, seed, count_tokens):
    if not bases:
        raise ValueError("No frozen bases")
    unique_index(bases)
    unique_index(hints)
    screens = screen_index(bases, screens, settings["screening_samples"])
    verified = unique_index(verification)
    groups = {group: [] for group in GROUPS}
    audit = []
    for base in bases:
        if settings.get("require_review", True) and not all(
            base.get("review", {}).get(key) is True
            for key in ("correct_steps", "prefixes_do_not_reveal_answer")
        ):
            raise ValueError(f"Review the original steps and prefixes first: {base['id']}")
        selected_boundary = hardest(base, screens[base["id"]])
        pool = [h for h in hints if h["source_id"] == base["id"]]
        for hint in pool:
            reviewed_hint(hint, base, count_tokens, settings["max_hint_tokens"])
        local = sorted([h for h in pool if h["kind"] == "local"], key=lambda h: h["id"])
        if len(local) != settings["local_candidates"]:
            raise ValueError(f"Expected complete local candidate pool for {base['id']}")
        if any(h["offset"] != selected_boundary for h in local):
            raise ValueError("Local candidate was not generated for the screened boundary")
        scores = verified.get(base["id"])
        if not scores or scores["source_hash"] != base["source_hash"]:
            raise ValueError("Missing/stale independent verification")
        if scores["hint_hash"] != digest(local):
            raise ValueError("Verification was run on different hints")
        trials = scores["baseline"]
        n = settings["verification_samples"]
        if len(trials) != n or len({x["seed"] for x in trials}) != n:
            raise ValueError("Incomplete or duplicated verification trials")
        screen_seeds = {x["seed"] for xs in screens[base["id"]]["boundaries"].values() for x in xs}
        if screen_seeds & {x["seed"] for x in trials}:
            raise ValueError("Screening and verification must use independent samples")
        baseline = sum(x["correct"] for x in trials) / n
        gains = {}
        for hint in local:
            sampled = scores["hints"][hint["id"]]
            if len(sampled) != n or [x["seed"] for x in sampled] != [x["seed"] for x in trials]:
                raise ValueError("Verification arms must use matching fresh seed lists")
            gains[hint["id"]] = sum(x["correct"] for x in sampled) / n - baseline
        best = max(local, key=lambda h: (gains[h["id"]], h["id"]))
        accepted = gains[best["id"]] >= settings["min_gain"]
        unfiltered = random.Random(stable_seed(seed, base["id"], "unfiltered")).choice(local)
        chosen = {"local_unfiltered": unfiltered}
        if accepted:
            chosen["local_refinement"] = best
            chosen["hint_at_start"] = {**best, "offset": 0}
            for group, kind in (("generic_expansion", "generic"), ("random_expansion", "random")):
                controls = [h for h in pool if h["kind"] == kind]
                if len(controls) != 1:
                    raise ValueError(f"Exactly one preregistered {kind} control required")
                control = controls[0]
                if kind == "random" and control["offset"] != base["random_boundary"]:
                    raise ValueError("Random control moved away from its frozen position")
                if abs(count_tokens(control["hint"]) - count_tokens(best["hint"])) > settings["control_token_tolerance"]:
                    raise ValueError(f"Length mismatch for {control['id']}; regenerate control before training")
                chosen[group] = control
        for group in GROUPS:
            hint = chosen.get(group)
            completion = insert(base["original"], hint["hint"], hint["offset"]) if hint else base["original"]
            groups[group].append({
                "id": f"{base['id']}:{group}", "source_id": base["id"],
                "prompt": TEMPLATE.format(question=base["question"]), "completion": completion,
                "gold_answer": base["gold_answer"], "source_hash": base["source_hash"],
                "hint_id": hint["id"] if hint else None,
                "hint_offset": hint["offset"] if hint else None,
                "added_tokens": count_tokens(hint["hint"]) if hint else 0,
            })
        audit.append({"id": base["id"], "boundary": selected_boundary,
                      "baseline_rate": baseline, "gains": gains, "accepted": accepted})
    return groups, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "requests", "build"))
    parser.add_argument("--config", default="configs/refinement_pilot.yaml")
    parser.add_argument("--limit", type=int, help="Smoke-only subset for requests/build; use a new root for expansion")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        result = prepare(config)
    elif args.stage == "requests":
        result = make_requests(config, args.limit)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(resolve_path(config["mlx_training"]["model"]), local_files_only=True)
        root = resolve_path(config["refinement"]["root"])
        bases = read_jsonl(root / "bases.jsonl")
        if args.limit is not None:
            bases = bases[:args.limit]
        groups, audit = build_conditions(
            bases, read_jsonl(root / "screening.jsonl"),
            read_jsonl(root / "hints.jsonl"), read_jsonl(root / "verification.jsonl"),
            config["refinement"], config["seed"],
            lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )
        for group, rows in groups.items():
            write_frozen(resolve_path(config["data"]["variants_dir"]) / f"{group}.jsonl", rows)
        write_frozen(root / "selection_audit.jsonl", audit)
        result = {"groups": {k: len(v) for k, v in groups.items()},
                  "accepted": sum(x["accepted"] for x in audit),
                  "budget_mode": "question_exposure_pilot; not token-matched"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
