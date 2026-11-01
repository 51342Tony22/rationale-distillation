from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import load_config, read_jsonl, resolve_path


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def question_fingerprint(rows: list[dict]) -> str:
    payload = "\n".join(f"{row['id']}\t{normalize_question(row['question'])}" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_training_rows(config: dict) -> tuple[list[dict], str]:
    variants_dir = resolve_path(config["data"]["variants_dir"])
    answer_only = read_jsonl(variants_dir / "answer_only.jsonl")
    train = read_jsonl(config["data"]["train_file"])
    if answer_only:
        train_by_id = {row["id"]: row for row in train}
        rows = [train_by_id[row["source_id"]] for row in answer_only]
        return rows, "actual_variant_question_set"
    limit = config["data"].get("teacher_question_limit")
    return (train[:limit] if limit else train), "frozen_teacher_source_pool"


def compare(train: list[dict], evaluation: list[dict], threshold: float) -> dict:
    train_normalized = [normalize_question(row["question"]) for row in train]
    evaluation_normalized = [normalize_question(row["question"]) for row in evaluation]
    evaluation_lookup: dict[str, list[int]] = {}
    for index, value in enumerate(evaluation_normalized):
        evaluation_lookup.setdefault(value, []).append(index)
    exact = []
    for train_index, value in enumerate(train_normalized):
        for evaluation_index in evaluation_lookup.get(value, []):
            exact.append(
                {
                    "train_id": train[train_index]["id"],
                    "evaluation_id": evaluation[evaluation_index]["id"],
                }
            )

    corpus = [row["question"] for row in train + evaluation]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(matrix[: len(train)], matrix[len(train) :])
    nearest_indices = np.argmax(similarities, axis=1)
    nearest_scores = similarities[np.arange(len(train)), nearest_indices]
    top_indices = np.argsort(nearest_scores)[::-1][:20]
    top_pairs = [
        {
            "train_id": train[index]["id"],
            "evaluation_id": evaluation[int(nearest_indices[index])]["id"],
            "similarity": float(nearest_scores[index]),
            "train_question": train[index]["question"],
            "evaluation_question": evaluation[int(nearest_indices[index])]["question"],
        }
        for index in top_indices
    ]
    return {
        "evaluation_examples": len(evaluation),
        "normalized_exact_overlap_count": len(exact),
        "normalized_exact_overlaps": exact,
        "near_duplicate_threshold": threshold,
        "near_duplicate_count": int(np.sum(nearest_scores >= threshold)),
        "maximum_char_tfidf_similarity": float(nearest_scores.max(initial=0.0)),
        "top_nearest_pairs": top_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/evaluation overlap and near duplicates.")
    parser.add_argument("--config", default="configs/qwen3_8b_main.yaml")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    train, selection = select_training_rows(config)
    data = config["data"]
    evaluation_splits = {
        name: read_jsonl(data[f"{name}_file"])
        for name in ("dev", "test", "svamp", "math500")
    }
    missing = [name for name, rows in evaluation_splits.items() if not rows]
    if missing:
        raise SystemExit(f"Missing evaluation data: {', '.join(missing)}")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "training_selection": selection,
        "training_examples": len(train),
        "training_question_sha256": question_fingerprint(train),
        "method": "NFKC alphanumeric exact match plus character 3-5 gram TF-IDF cosine",
        "splits": {
            name: compare(train, rows, args.threshold)
            for name, rows in evaluation_splits.items()
        },
    }
    report["status"] = (
        "REVIEW"
        if any(
            result["normalized_exact_overlap_count"] or result["near_duplicate_count"]
            for result in report["splits"].values()
        )
        else "PASS"
    )
    output = resolve_path(
        args.output
        or config.get("artifacts", {}).get(
            "leakage_report", "reports_qwen3_8b/data/leakage_audit.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": report["status"],
        "training_selection": selection,
        "training_examples": len(train),
        "splits": {
            name: {
                "n": result["evaluation_examples"],
                "exact": result["normalized_exact_overlap_count"],
                "near_duplicates": result["near_duplicate_count"],
                "max_similarity": round(result["maximum_char_tfidf_similarity"], 4),
            }
            for name, result in report["splits"].items()
        },
        "report": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
