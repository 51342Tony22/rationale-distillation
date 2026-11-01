"""Final read-only integrity checks; requires explicit hash-bound full-text reviews."""
import json
from pathlib import Path

import baseline
import score_sensitivity_v2 as sensitive
from pilot_common import BASE, C, DATA, OUT, ROOT, digest_text, rows, sources


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    protocol = json.loads((OUT / "training_protocol.json").read_text())
    assert baseline.sha(Path(sensitive.__file__)) == manifest["sensitivity_parser_sha256"]
    for file, expected in protocol["material_sha256"].items():
        assert baseline.sha(OUT / file) == expected
    dev = sources("dev")
    expected_ids = [r["source_id"] for r in dev]
    paths = {}
    summaries = {}
    adapters = []
    for condition in ["student_base", *C["conditions"]]:
        if condition == "student_base":
            pred_path = ROOT / BASE["output_dir"] / "dev_baseline/student_base/predictions.jsonl"
            assert baseline.sha(pred_path) == manifest["baseline_predictions_sha256"]
            predictions = rows(pred_path)[:len(dev)]
            first_audit = ROOT / BASE["output_dir"] / "dev_preview_audit.json"
            audits = json.loads(first_audit.read_text())["records"]
            paths[str(first_audit.relative_to(ROOT))] = baseline.sha(first_audit)
        else:
            pred_path = OUT / "evaluation" / condition / "predictions.jsonl"
            predictions = rows(pred_path)
            audits = []
            directory = OUT / "training" / condition
            metadata = json.loads((directory / "metadata.json").read_text())
            adapter = directory / "adapters.safetensors"
            assert metadata["completed"] and metadata["actual_new_training"]
            assert metadata["microsteps"] == protocol["training"]["microsteps_per_condition"]
            assert metadata["updates"] == protocol["training"]["updates_per_condition"]
            assert metadata["adapter_sha256"] == baseline.sha(adapter)
            adapters.append(metadata["adapter_sha256"])
            data_file = OUT / "training_data" / f"{condition}.json"
            data = json.loads(data_file.read_text())
            assert baseline.sha(data_file) == metadata["data_sha256"]
            assert [r["source_id"] for r in data] == protocol["source_ids"]
            assert metadata["supervised_tokens_seen"] == C["epochs"] * sum(len(r["target_ids"]) for r in data)
            assert all(r["total_tokens"] <= C["context_tokens"] for r in data)
            assert metadata["padding_targets"] == 0
            assert baseline.sha(OUT / f"training_{condition}_signature.json") == metadata["training_signature_sha256"]
            for stage in ["training", "evaluation"]:
                signed = json.loads((OUT / f"{stage}_{condition}_signature.json").read_text())
                for relative, expected in signed["inputs"].items():
                    assert baseline.sha(ROOT / relative) == expected, f"Signed input changed: {relative}"
            for file in [adapter, directory / "metadata.json", data_file, OUT / f"training_{condition}_signature.json", OUT / f"evaluation_{condition}_signature.json"]:
                paths[str(file.relative_to(ROOT))] = baseline.sha(file)
        audit_path = OUT / f"{condition}_evaluation_reviews.jsonl"
        audits += rows(audit_path)
        lookup = {r["source_id"]: r for r in audits}
        assert len(audits) == len(lookup) == len(dev), f"Full-text audit incomplete: {condition}"
        assert [r["source_id"] for r in predictions] == expected_ids
        assert set(lookup) == set(expected_ids)
        for item, row in zip(dev, predictions):
            assert row["gold"] == item["gold"] and row["question_sha256"] == item["question_sha256"]
            assert digest_text(row["output"]) == row["output_sha256"] == lookup[row["source_id"]]["output_sha256"]
            assert all(row[key] == value for key, value in baseline.score(row["scoring_text"], item["gold"]).items())
            value, _ = sensitive.parse(row["scoring_text"])
            if condition != "student_base":
                assert row["v2_correct"] == (value is not None and value == baseline.number(item["gold"]))
            assert lookup[row["source_id"]].get("mathematics")
        summaries[condition] = {"predictions": len(predictions), "full_text_reviews": len(audits),
                                "v1_correct": sum(r["correct"] for r in predictions), "length_stops": sum(r["finish_reason"] == "length" for r in predictions)}
        paths[str(pred_path.relative_to(ROOT))] = baseline.sha(pred_path)
        paths[str(audit_path.relative_to(ROOT))] = baseline.sha(audit_path)
    assert len(set(adapters)) == 3, "Expected three separately trained distinct adapters"
    results = json.loads((OUT / "results.json").read_text())
    assert results["stage"] == "pilot_complete_full_text_audited"
    for name, summary in summaries.items():
        assert results["conditions"][name]["v1_correct"] == summary["v1_correct"]
    for file in [DATA / "manifest.json", OUT / "training_protocol.json", OUT / "results.json", OUT / "RESULTS_ZH.md", Path(__file__), Path(__file__).parent / "pilot_report.py"]:
        paths[str(file.relative_to(ROOT))] = baseline.sha(file)
    evidence = {"status": "passed", "conditions": summaries, "actual_independent_training": 3,
                "primary_scores_modified_by_auxiliary_audit": False, "formal_test_evaluated": False, "files_sha256": paths}
    baseline.freeze(OUT / "final_integrity_audit.json", evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "files_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
