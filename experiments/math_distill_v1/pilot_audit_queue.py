"""Read complete pending pilot outputs for an explicit assistant audit; never approve rows."""
import argparse
import json

from pilot_common import DATA, OUT, rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    reviewed = {r["source_id"] for r in rows(OUT / f"{args.condition}_evaluation_reviews.jsonl")}
    source = {r["source_id"]: r for r in json.loads((DATA / "dev.json").read_text())}
    records = rows(OUT / "evaluation" / args.condition / "predictions.jsonl")
    print(json.dumps({"generated": len(records), "reviewed": len(reviewed)}))
    pending = [(n, r) for n, r in enumerate(records, 1) if r["source_id"] not in reviewed][:args.limit]
    for n, row in pending:
        print(json.dumps({"n": n, "source_id": row["source_id"], "question": source[row["source_id"]]["question"],
                          "gold": row["gold"], "output": row["output"], "output_sha256": row["output_sha256"],
                          "v1": row["correct"], "v2": row["v2_correct"], "stop": row["finish_reason"]}, ensure_ascii=False))
