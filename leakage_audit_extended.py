from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REPO = Path("/Users/starsfalling/Documents/FYP Mini Agent")
OUT = Path("/Users/starsfalling/Documents/ChatGPT/FYP/leakage_audit_extended.json")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def question(row: dict) -> str:
    return str(row.get("question") or row.get("Body") or "").strip()


def compare(train_texts: list[str], eval_texts: list[str], top_n: int = 10) -> dict:
    train_norm = [normalize(value) for value in train_texts]
    eval_norm = [normalize(value) for value in eval_texts]
    exact = sorted(set(train_norm) & set(eval_norm))

    corpus = train_norm + eval_norm
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[len(train_norm) :], matrix[: len(train_norm)])
    best_train = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(eval_norm)), best_train]
    order = np.argsort(-best_scores)[:top_n]
    top_pairs = [
        {
            "score": round(float(best_scores[index]), 6),
            "train": train_texts[int(best_train[index])],
            "evaluation": eval_texts[int(index)],
        }
        for index in order
    ]
    return {
        "train_count": len(train_texts),
        "evaluation_count": len(eval_texts),
        "normalized_exact_overlap": len(exact),
        "nearest_pair_counts": {
            ">=0.95": int((best_scores >= 0.95).sum()),
            ">=0.90": int((best_scores >= 0.90).sum()),
            ">=0.85": int((best_scores >= 0.85).sum()),
            ">=0.80": int((best_scores >= 0.80).sum()),
        },
        "top_pairs": top_pairs,
    }


def main() -> None:
    variant = read_jsonl(REPO / "data/variants/single_rationale.jsonl")
    seen: set[str] = set()
    selected_train = []
    for row in variant:
        source_id = str(row["source_id"])
        if source_id not in seen:
            selected_train.append(str(row["question"]))
            seen.add(source_id)

    gsm_test = [question(row) for row in read_jsonl(REPO / "data/processed/gsm8k_test.jsonl")]
    gsm_dev = [question(row) for row in read_jsonl(REPO / "data/processed/gsm8k_dev.jsonl")]
    svamp = [question(row) for row in read_jsonl(REPO / "data/processed/svamp_test.jsonl")]

    report = {
        "method": {
            "exact": "lowercase alphanumeric-token normalization",
            "near_duplicate": "character 3-5 gram TF-IDF cosine similarity",
            "scope_note": (
                "Lexical overlap audit only; it cannot rule out semantic template overlap or "
                "pretraining/teacher-model contamination."
            ),
        },
        "selected_training_questions": len(selected_train),
        "comparisons": {
            "gsm8k_eval_first_100": compare(selected_train, gsm_test[:100]),
            "gsm8k_eval_full": compare(selected_train, gsm_test),
            "gsm8k_dev_full": compare(selected_train, gsm_dev),
            "svamp_eval_first_100": compare(selected_train, svamp[:100]),
            "svamp_eval_full": compare(selected_train, svamp),
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
