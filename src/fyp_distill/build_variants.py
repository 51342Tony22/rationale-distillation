from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import (
    LEGACY_VARIANTS,
    completion_for,
    configured_variants,
    load_config,
    prompt_for,
    read_jsonl,
    resolve_path,
    seeded_rng,
    write_jsonl,
)


def _distance_matrix(candidates: list[dict]) -> np.ndarray:
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
        [candidate["text"] for candidate in candidates]
    )
    return 1.0 - cosine_similarity(matrix)


def select_by_diversity(candidates: list[dict], k: int, highest: bool) -> list[dict]:
    if len(candidates) <= k:
        return candidates[:]
    distances = _distance_matrix(candidates)
    scored = []
    for indices in itertools.combinations(range(len(candidates)), k):
        pair_indices = list(itertools.combinations(indices, 2))
        score = float(np.mean([distances[left, right] for left, right in pair_indices]))
        stable_ids = tuple(candidates[index]["candidate_id"] for index in indices)
        scored.append((score, stable_ids, indices))
    if highest:
        _, _, selected = min(scored, key=lambda item: (-item[0], item[1]))
    else:
        _, _, selected = min(scored, key=lambda item: (item[0], item[1]))
    return [candidates[index] for index in selected]


def select_diverse(candidates: list[dict], k: int) -> list[dict]:
    return select_by_diversity(candidates, k, highest=True)


def select_low_diversity(candidates: list[dict], k: int) -> list[dict]:
    return select_by_diversity(candidates, k, highest=False)


def select_compatible(candidates: list[dict], k: int) -> list[dict]:
    scored = [candidate for candidate in candidates if candidate.get("compatibility_nll") is not None]
    return sorted(
        scored,
        key=lambda candidate: (candidate["compatibility_nll"], candidate["candidate_id"]),
    )[:k]


def mean_pairwise_distance(candidates: list[dict]) -> float:
    if len(candidates) < 2:
        return 0.0
    matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
        [candidate["text"] for candidate in candidates]
    )
    similarity = cosine_similarity(matrix)
    upper = similarity[np.triu_indices(len(candidates), k=1)]
    return float(np.mean(1.0 - upper))


def make_row(source: dict, variant: str, rationale: dict | None, row_index: int) -> dict:
    return {
        "id": f"{source['id']}__{variant}__{row_index}",
        "source_id": source["id"],
        "variant": variant,
        "question": source["question"],
        "prompt": prompt_for(source["question"]),
        "completion": completion_for(rationale["text"] if rationale else None, source["gold_answer"]),
        "gold_answer": source["gold_answer"],
        "rationale_id": rationale["candidate_id"] if rationale else None,
    }


def build(
    teacher_rows: list[dict],
    k: int,
    seed: int,
    variants: tuple[str, ...] = LEGACY_VARIANTS,
) -> tuple[dict[str, list[dict]], dict]:
    outputs = {variant: [] for variant in variants}
    exclusion_reasons = Counter()
    diversity_scores = {
        variant: []
        for variant in (
            "low_diversity_rationale",
            "multi_rationale",
            "diverse_multi_rationale",
            "compatible_rationale",
        )
        if variant in variants
    }
    compatibility_scores = []
    eligible_questions = 0
    candidate_audit = Counter()
    for source in teacher_rows:
        candidates = source.get("candidates", [])
        candidate_audit["total"] += len(candidates)
        candidate_audit["api_errors"] += sum(bool(candidate.get("error")) for candidate in candidates)
        candidate_audit["correct"] += sum(bool(candidate.get("is_correct")) for candidate in candidates)
        requires_response_content = source.get("generation", {}).get("schema_version", 1) >= 2
        valid = [
            candidate
            for candidate in candidates
            if candidate.get("is_correct")
            and candidate.get("text", "").strip()
            and candidate.get("is_usable", True)
            and (
                not requires_response_content
                or candidate.get("supervision_source") == "response_content"
            )
        ]
        candidate_audit["accepted_for_supervision"] += len(valid)
        if len(valid) < k:
            exclusion_reasons[f"fewer_than_{k}_correct_rationales"] += 1
            continue
        if "compatible_rationale" in variants and len(select_compatible(valid, k)) < k:
            exclusion_reasons[f"fewer_than_{k}_compatibility_scores"] += 1
            continue
        eligible_questions += 1
        rng = seeded_rng(seed, source["id"])
        shuffled = valid[:]
        rng.shuffle(shuffled)
        single = shuffled[0]
        multi = shuffled[:k]
        diverse = select_diverse(valid, k)
        low_diversity = select_low_diversity(valid, k)
        compatible = select_compatible(valid, k)

        selected_groups = {
            "low_diversity_rationale": low_diversity,
            "multi_rationale": multi,
            "diverse_multi_rationale": diverse,
            "compatible_rationale": compatible,
        }
        for name, scores in diversity_scores.items():
            scores.append(mean_pairwise_distance(selected_groups[name]))
        if compatible:
            compatibility_scores.append(
                float(np.mean([candidate["compatibility_nll"] for candidate in compatible]))
            )
        for index in range(k):
            if "answer_only" in outputs:
                outputs["answer_only"].append(make_row(source, "answer_only", None, index))
            if "single_rationale" in outputs:
                outputs["single_rationale"].append(
                    make_row(source, "single_rationale", single, index)
                )
            for name, selected in selected_groups.items():
                if name in outputs:
                    outputs[name].append(make_row(source, name, selected[index], index))

    report = {
        "teacher_questions": len(teacher_rows),
        "eligible_questions": eligible_questions,
        "k": k,
        "seed": seed,
        "variants": list(variants),
        "rows_per_variant": {name: len(rows) for name, rows in outputs.items()},
        "unique_questions_per_variant": {
            name: len({row["source_id"] for row in rows}) for name, rows in outputs.items()
        },
        "exclusions": dict(exclusion_reasons),
        "candidate_audit": dict(candidate_audit),
        "mean_selected_tfidf_distance": {
            name: float(np.mean(scores)) if scores else None
            for name, scores in diversity_scores.items()
        },
        "mean_selected_compatibility_nll": (
            float(np.mean(compatibility_scores)) if compatibility_scores else None
        ),
    }
    return outputs, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the four controlled training datasets.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--k", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    variants = configured_variants(config)
    default_input = (
        data_config.get("scored_teacher_file")
        if "compatible_rationale" in variants
        else data_config["teacher_file"]
    )
    teacher_rows = read_jsonl(args.input or default_input)
    if not teacher_rows:
        raise SystemExit("No teacher records found. Run generate_rationales first.")
    k = args.k or data_config["selected_rationales"]
    outputs, report = build(teacher_rows, k=k, seed=config["seed"], variants=variants)
    output_dir = resolve_path(args.output_dir or data_config["variants_dir"])
    for variant, rows in outputs.items():
        write_jsonl(output_dir / f"{variant}.jsonl", rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "build_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
