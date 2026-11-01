"""Require complete hash-bound reviews and validate the new experiment in place."""
import json
from pathlib import Path

import baseline
import pilot_common as common
import score_sensitivity_v2 as sensitive
from pilot100 import C, CONFIG, DATA, OLD, OUT, ROOT, REUSE
from pilot100_report import reviews


def main():
    manifest=json.loads((DATA / "manifest.json").read_text())
    protocol=json.loads((OUT / "training_protocol.json").read_text())
    report=json.loads((OUT / "results.json").read_text())
    assert report["stage"]=="pilot100_complete_full_text_audited"
    assert baseline.sha(CONFIG)==manifest["config_sha256"]
    assert baseline.sha(Path(sensitive.__file__))==manifest["sensitivity_parser_sha256"]
    assert baseline.sha(ROOT / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl")==manifest["baseline_predictions_sha256"]
    files={}
    for stage in ["teacher_targets","locations","edits_v4"]:
        signature=OUT / f"{stage}_signature.json"
        signed=json.loads(signature.read_text())
        for path,expected in signed["inputs"].items():
            assert baseline.sha(ROOT / path)==expected,f"Material signed input changed: {path}"
        files[str(signature.relative_to(ROOT))]=baseline.sha(signature)
    preflight_path=OUT / "cpu_training_preflight.json"
    preflight=json.loads(preflight_path.read_text())
    assert preflight["status"]=="passed" and preflight["completion_loss_vs_dense_error"]==0
    assert preflight["adapter_reload_max_logits_difference"]==0
    assert preflight["padding_targets"]==preflight["prompt_targets"]==0
    assert preflight["training_runner_sha256"]==baseline.sha(Path(__file__).with_name("pilot_train.py"))
    assert preflight["preflight_sha256"]==baseline.sha(Path(__file__).with_name("pilot_preflight.py"))
    files[str(preflight_path.relative_to(ROOT))]=baseline.sha(preflight_path)
    for split in ["train","dev"]:
        path=DATA / f"{split}.json"
        assert baseline.sha(path)==manifest["splits"][split]["frozen_rows_sha256"]
        files[str(path.relative_to(ROOT))]=baseline.sha(path)
    source=json.loads((DATA / "dev.json").read_text())
    assert len(source)==250
    reuse=json.loads((OUT / "reused_material_prefixes.json").read_text())
    for name in REUSE:
        assert baseline.sha(OLD / name)==reuse[name]["sha256"]
        assert (OUT / name).read_bytes().startswith((OLD / name).read_bytes())
    for name,expected in protocol["material_sha256"].items():
        assert baseline.sha(OUT / name)==expected
        files[str((OUT / name).relative_to(ROOT))]=expected
    adapters=[]
    for name in ["student_base",*C["conditions"]]:
        prediction_path=OUT / "evaluation" / name / "predictions.jsonl"
        predictions=common.rows(prediction_path)
        audit=reviews(name)
        assert len(predictions)==len(audit)==250
        reported=report["conditions"][name]
        assert reported["full_text_reviewed"]==250 and reported["audit_complete"]
        assert reported["v1_correct"]==sum(r["correct"] for r in predictions)
        assert reported["v1_invalid"]==sum(r["invalid"] for r in predictions)
        assert reported["v2_correct"]==sum(
            sensitive.parse(r["scoring_text"])[0]==baseline.number(r["gold"])
            for r in predictions if sensitive.parse(r["scoring_text"])[0] is not None)
        assert reported["length_stops"]==sum(r["finish_reason"]=="length" for r in predictions)
        assert set(audit)=={r["source_id"] for r in source}
        assert baseline.sha(prediction_path)==report["conditions"][name]["predictions_sha256"]
        for item,row in zip(source,predictions):
            assert item["source_id"]==row["source_id"] and item["gold"]==row["gold"] and item["question_sha256"]==row["question_sha256"]
            assert common.digest_text(row["output"])==row["output_sha256"]==audit[row["source_id"]]["output_sha256"]
            assert audit[row["source_id"]].get("mathematics")
            assert all(row[k]==v for k,v in baseline.score(row["scoring_text"],item["gold"]).items())
        if name=="student_base":
            original=ROOT / "output/math_distill_v1/dev_baseline/student_base/predictions.jsonl"
            assert prediction_path.read_bytes().startswith(original.read_bytes())
        else:
            for stage in ["training","evaluation"]:
                signature=OUT / f"{stage}_{name}_signature.json"
                signed=json.loads(signature.read_text())
                for path,expected in signed["inputs"].items():
                    assert baseline.sha(ROOT / path)==expected,f"Signed input changed: {path}"
                files[str(signature.relative_to(ROOT))]=baseline.sha(signature)
            folder=OUT / "training" / name
            metadata=json.loads((folder / "metadata.json").read_text())
            adapter=folder / "adapters.safetensors"
            assert metadata["completed"] and metadata["actual_new_training"]
            assert metadata["adapter_sha256"]==baseline.sha(adapter)
            assert metadata["training_signature_sha256"]==baseline.sha(OUT / f"training_{name}_signature.json")
            assert metadata["rows"]==protocol["eligible"]
            assert metadata["microsteps"]==protocol["training"]["microsteps_per_condition"]
            assert metadata["updates"]==protocol["training"]["updates_per_condition"]
            assert metadata["padding_targets"]==0
            train_path=OUT / "training_data" / f"{name}.json"
            data=json.loads(train_path.read_text())
            assert baseline.sha(train_path)==metadata["data_sha256"]
            assert [r["source_id"] for r in data]==protocol["source_ids"]
            assert metadata["supervised_tokens_seen"]==C["epochs"]*sum(len(r["target_ids"]) for r in data)
            assert all(r["total_tokens"]<=C["context_tokens"] for r in data)
            assert metadata["adapter_sha256"]!=json.loads((OLD / "training" / name / "metadata.json").read_text())["adapter_sha256"]
            adapters.append(metadata["adapter_sha256"])
            for p in [adapter,folder / "metadata.json",train_path]:
                files[str(p.relative_to(ROOT))]=baseline.sha(p)
        for p in [prediction_path,OUT / f"{name}_evaluation_reviews.jsonl"]:
            files[str(p.relative_to(ROOT))]=baseline.sha(p)
    base_sign=json.loads((OUT / "evaluation_student_base_signature.json").read_text())
    for path,expected in base_sign["inputs"].items():
        assert baseline.sha(ROOT / path)==expected
    base_signature_path=OUT / "evaluation_student_base_signature.json"
    files[str(base_signature_path.relative_to(ROOT))]=baseline.sha(base_signature_path)
    recovery_path=OUT / "evaluation_recovery_preflight.json"
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        recovery_runner=Path(__file__).with_name("pilot100_resume_eval.py")
        assert baseline.sha(recovery_runner)==recovery["runner_sha256"]
        assert not recovery["training_repeated"] and not recovery["old_outputs_overwritten"]
        for path,expected in recovery["signed_artifacts_verified"].items():
            assert baseline.sha(ROOT / path)==expected,f"Recovery artifact changed: {path}"
        launch_path=OUT / "evaluation_resume_launch.json"
        assert json.loads(launch_path.read_text())["runner_sha256"]==recovery["runner_sha256"]
        for p in [recovery_path,recovery_runner,launch_path,OUT / "evaluation_failure_status.json",
                  OUT / "evaluation_resume.log",OUT / "pipeline.log"]:
            files[str(p.relative_to(ROOT))]=baseline.sha(p)
    assert len(set(adapters))==3
    for p in [CONFIG,DATA / "manifest.json",OUT / "training_protocol.json",OUT / "results.json",OUT / "RESULTS_ZH.md",
              OUT / "reused_material_prefixes.json",Path(__file__),Path(__file__).with_name("pilot100_report.py"),
              ROOT / "output/math_distill_v1/dev_preview_audit.json",OLD / "student_base_evaluation_reviews.jsonl"]:
        files[str(p.relative_to(ROOT))]=baseline.sha(p)
    evidence={"status":"passed","predictions":1000,"hash_bound_full_text_reviews":1000,"reused_old_student_reviews":100,
              "training_candidates":100,"eligible":protocol["eligible"],"actual_independent_new_training":3,
              "old_adapter_initialization":False,"old_trained_predictions_mixed_with_new_adapters":False,
              "primary_scores_changed_by_audit":False,"official_test_evaluated":False,"files_sha256":files}
    baseline.freeze(OUT / "final_integrity_audit.json",evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!="files_sha256"},ensure_ascii=False))


if __name__=="__main__":
    main()
