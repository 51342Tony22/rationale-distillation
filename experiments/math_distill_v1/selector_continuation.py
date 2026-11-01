"""Teacher edits -> human review gate -> paired student continuation smoke test."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path

import baseline
from pilot_common import append, digest_text, rows
import pilot_edits_v3 as edit_prompt
import pilot_edits_v4 as editor
from pilot_materials import load_teacher
import score_sensitivity_v2 as sensitivity
from selector_masking_core import stable_seed
import selector_masking_probe as probe

ROOT, HERE = baseline.ROOT, probe.HERE
CONFIG = HERE / "selector_continuation_config.json"
C = json.loads(CONFIG.read_text())
OUT = ROOT / C["output_dir"]


def material():
    cases = json.loads((probe.OUT / "cases.json").read_text())
    diagnostics = rows(probe.OUT / "diagnostics.jsonl")
    assert [r["source_id"] for r in cases] == [r["source_id"] for r in diagnostics]
    sources = {r["source_id"]: r for r in json.loads((ROOT / "data/math_distill_v1/pilot100/train.json").read_text())}
    teachers = {r["source_id"]: r for r in rows(probe.OLD / "teacher_targets.jsonl")}
    result = []
    for case in cases:
        teacher = teachers[case["source_id"]]
        assert digest_text(teacher["output"]) == teacher["output_sha256"] == case["teacher_output_sha256"]
        for candidate in case["candidates"]:
            step = candidate["step"]
            prefix = teacher["output"][:step["char_start"]]
            assert teacher["output"][step["char_start"]:step["char_end"]] == step["text"]
            result.append({"id": f'{case["source_id"]}:step:{step["step_index"]}',
                           "source": sources[case["source_id"]], "step": step,
                           "teacher": teacher, "prefix": prefix})
    return result


def sign(stage, extra_files=()):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [CONFIG, Path(__file__), probe.CONFIG, HERE / "selector_masking_core.py", Path(probe.__file__),
             baseline.CONFIG, Path(baseline.__file__), Path(sensitivity.__file__),
             Path(editor.__file__), Path(edit_prompt.__file__), HERE / "pilot_config.json", HERE / "pilot_common.py",
             HERE / "pilot_materials.py", probe.OUT / "signature.json", probe.OUT / "cases.json",
             probe.OUT / "diagnostics.jsonl", ROOT / "data/math_distill_v1/pilot100/train.json",
             probe.OLD / "teacher_targets.jsonl", probe.OLD / "teacher_reviews.jsonl",
             ROOT / probe.BASE["output_dir"] / "dev_baseline/teacher_reference/signature.json", *extra_files]
    baseline.freeze(OUT / f"{stage}_signature.json", {"config": C,
                     "inputs": {str(p.relative_to(ROOT)): baseline.sha(p) for p in files},
                     "versions": {name: importlib.metadata.version(name) for name in ["mlx", "mlx-lm", "transformers"]},
                     "ordered_ids": [m["id"] for m in material()]})


def verify_records(records, expected_ids):
    assert len(records) <= len(expected_ids)
    assert [r["id"] for r in records] == expected_ids[:len(records)]
    for row in records:
        value = dict(row)
        digest = value.pop("record_sha256")
        assert digest_text(json.dumps(value, sort_keys=True)) == digest


def store(path, value):
    value["record_sha256"] = digest_text(json.dumps(value, sort_keys=True))
    append(path, value)


def edits():
    from transformers import AutoTokenizer
    items = material()
    with baseline.gpu_lock(OUT):
        sign("edits")
        path = OUT / "edits.jsonl"
        saved = rows(path)
        verify_records(saved, [m["id"] for m in items])
        if len(saved) < len(items):
            model, tok = load_teacher()
            student_tok = AutoTokenizer.from_pretrained(str(ROOT / probe.BASE["student_dir"]), local_files_only=True)
            for item in items[len(saved):]:
                location = {"chosen": {"start": item["step"]["char_start"], "text": item["step"]["text"]}}
                prompt, prefill = edit_prompt.build_prompt(tok, item["source"], item["teacher"]["output"], location, "difficulty_only")
                result = {"id": item["id"], "source_id": item["source"]["source_id"],
                          "step_index": item["step"]["step_index"],
                          **editor.generate(model, tok, prompt, prefill, student_tok)}
                store(path, result)
                saved.append(result)
                print(json.dumps({k: result[k] for k in ["id", "explanation", "student_tokens", "output_sha256", "record_sha256"]}, ensure_ascii=False), flush=True)
                baseline.save(OUT / "status.json", {"stage": "edits_need_full_text_review", "completed": len(saved), "total": len(items)})


def reviewed_edits():
    generated = rows(OUT / "edits.jsonl")
    items = material()
    verify_records(generated, [m["id"] for m in items])
    assert len(generated) == len(items)
    reviews = rows(OUT / "edit_reviews.jsonl")
    assert len({r["id"] for r in reviews}) == len(reviews) == len(generated)
    lookup = {r["id"]: r for r in reviews}
    accepted = {}
    for row in generated:
        review = lookup[row["id"]]
        assert review["record_sha256"] == row["record_sha256"] and isinstance(review["accepted"], bool) and review["reason"]
        if review["accepted"]:
            assert row["explanation"] and 0 < row["student_tokens"] <= C["explanation_max_student_tokens"]
            accepted[row["id"]] = row
    return accepted


def continuation_prompt(item, explanation=None):
    # Deliberately excludes the quoted current step shown to the teacher editor.
    result = probe.BASE["student_prompt"].format(question=item["source"]["question"]) + item["prefix"]
    if explanation:
        result += explanation + "\n"
    return result


def evaluate():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    accepted = reviewed_edits()
    items = [m for m in material() if m["id"] in accepted]
    tasks = [(m, rep, condition) for m in items for rep in range(C["replicates"])
             for condition in ["no_explanation", "explanation"]]
    with baseline.gpu_lock(OUT):
        sign("evaluation", [OUT / "edits.jsonl", OUT / "edit_reviews.jsonl", probe.OUT / "model_signature.json"])
        checked = baseline.verify_student(probe.BASE)
        assert checked == json.loads((probe.OUT / "model_signature.json").read_text())["model_files"]
        path = OUT / "predictions.jsonl"
        expected_ids = [f'{m["id"]}:rep:{rep}:{condition}' for m, rep, condition in tasks]
        saved = rows(path)
        verify_records(saved, expected_ids)
        if len(saved) < len(tasks):
            mx.set_memory_limit(14 * 1024**3)
            mx.set_cache_limit(512 * 1024**2)
            model, tok = load(str(ROOT / probe.BASE["student_dir"]), tokenizer_config={"local_files_only": True, "trust_remote_code": False})
            model.eval()
            for item, rep, condition in tasks[len(saved):]:
                explanation = accepted[item["id"]]["explanation"] if condition == "explanation" else None
                prompt = continuation_prompt(item, explanation)
                ids = tok.encode(prompt, add_special_tokens=False)
                assert len(ids) + C["max_new_tokens"] <= C["context_tokens"]
                seed = stable_seed(C["seed"], f'{item["id"]}:{rep}') % (2**32)
                mx.random.seed(seed)
                text, last, boundary = "", None, None
                start = time.perf_counter()
                for part in stream_generate(model, tok, ids, max_tokens=C["max_new_tokens"], sampler=make_sampler(temp=C["temperature"], top_p=C["top_p"])):
                    last = part
                    text += part.text
                    boundary = re.search(r"(?:^|\n)\s*Question:", text)
                    if boundary:
                        break
                assert last is not None
                scored = text[:boundary.start()] if boundary else text
                parsed, _ = sensitivity.parse(scored)
                result = {"id": f'{item["id"]}:rep:{rep}:{condition}', "position_id": item["id"],
                          "source_id": item["source"]["source_id"], "step_index": item["step"]["step_index"],
                          "replicate": rep, "condition": condition, "seed": seed, "prompt_sha256": digest_text(prompt),
                          "output": text, "scoring_text": scored, "output_sha256": digest_text(text),
                          "gold": item["source"]["gold"], **baseline.score(scored, item["source"]["gold"]),
                          "v2_correct": parsed is not None and parsed == baseline.number(item["source"]["gold"]),
                          "prompt_tokens": len(ids), "generation_tokens": last.generation_tokens,
                          "finish_reason": "next_question" if boundary else last.finish_reason,
                          "seconds": time.perf_counter() - start}
                store(path, result)
                saved.append(result)
                baseline.save(OUT / "status.json", {"stage": "paired_continuation", "completed": len(saved), "total": len(tasks), "training_started": False})
                print(f'{len(saved)}/{len(tasks)} {result["id"]}: v1={result["correct"]} v2={result["v2_correct"]}', flush=True)
        baseline.save(OUT / "status.json", {"stage": "continuations_complete_need_full_text_audit_and_report", "completed": len(saved), "total": len(tasks), "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["edits", "evaluate"])
    args = parser.parse_args()
    edits() if args.stage == "edits" else evaluate()
