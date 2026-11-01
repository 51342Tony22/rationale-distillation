"""Question-only lexical overlap audit, separate from any model evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).absolute().parents[2]
DATA = ROOT / "data/math_distill_v1"


def read(name):
    return [json.loads(s) for s in (DATA / name).read_text().splitlines()]


def main():
    train = read("train.jsonl")
    dev = read("dev.jsonl")
    test = [{"source_id": f"gsm8k:test:{i}", "question": row["question"]}
            for i, row in enumerate(read("source/test.jsonl"))]
    rows = train + dev + test
    normalized = [" ".join(re.findall(r"[a-z0-9]+", r["question"].lower())) for r in rows]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(normalized)
    sections = {"train": (0, len(train)), "dev": (len(train), len(train) + len(dev)),
                "test": (len(train) + len(dev), len(rows))}
    result = {"method": "question-only char 3–5gram TF-IDF cosine; lowercase alphanumeric normalization",
              "scope": "Lexical screening only; cannot exclude paraphrases/templates/pretraining contamination",
              "comparisons": {}}
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        a, b = sections[left], sections[right]
        similarity = cosine_similarity(matrix[a[0]:a[1]], matrix[b[0]:b[1]])
        candidates = []
        for i, j in zip(*np.where(similarity >= 0.85)):
            ra, rb = rows[a[0] + int(i)], rows[b[0] + int(j)]
            candidates.append({"left_id": ra["source_id"], "right_id": rb["source_id"],
                               "similarity": float(similarity[i, j]),
                               "left_question": ra["question"], "right_question": rb["question"]})
        result["comparisons"][f"{left}_{right}"] = {
            "maximum_cosine": float(similarity.max()),
            "pairs_at_or_above_0_85": sorted(candidates, key=lambda row: -row["similarity"]),
            "counts": {str(t): int((similarity >= t).sum()) for t in [0.85, 0.90, 0.95]}}
    out = ROOT / "output/math_distill_v1/lexical_overlap.json"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if out.exists():
        assert out.read_text() == text, "Existing lexical report differs; preserve it"
    else:
        out.write_text(text)
    print(json.dumps({k: {"maximum_cosine": v["maximum_cosine"], "counts": v["counts"]}
                      for k, v in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
