"""Local equality continuation, with explicit focus and an immutable raw trail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from transformers import AutoTokenizer

import baseline
from pilot_common import BASE, C, OUT, ROOT, append, digest_text, gpu, reviewed, rows, sign, sources
from pilot_materials import load_teacher

SYSTEM = "You are a mathematics tutor. Explain the reason for exactly one quoted equality. Do not solve the surrounding word problem, give subsequent steps, or compute a new result."
BRIDGE = "This step is valid because "


def build_prompt(tok, item, target, location, condition):
    chosen = location["chosen"]
    evidence = ""
    if condition == "difficulty_attention":
        evidence = "\nPay particular attention to these prior context units ranked by the student:\n" + "\n".join(e["text"] for e in location["evidence"])
    user = "Context for interpreting the symbols and quantities (do not solve this problem):\n" + item["question"]
    user += "\nEarlier working:\n" + (target[:chosen["start"]].strip() or "None yet.") + evidence
    user += "\nExplain ONLY this quoted equality and why its operation is appropriate:\n" + chosen["text"]
    user += "\nContinue the sentence below with the conceptual justification for that equality only."
    message = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    equality = " ".join(chosen["text"].split())
    prefill = 'The quoted step "' + equality + '" is valid because '
    return tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + prefill, prefill


def generate(model, tok, prompt, prefill, student_tok):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    ids = tok.encode(prompt, add_special_tokens=False)
    assert len(ids) + C["edit_max_new_tokens"] <= C["context_tokens"]
    output, last, end = "", None, None
    start = time.perf_counter()
    for part in stream_generate(model, tok, ids, max_tokens=C["edit_max_new_tokens"], sampler=make_sampler(temp=0.0)):
        output += part.text
        last = part
        match = re.search(r"[.!?](?=\s|$)", output)
        if match:
            end = match.end()
            break
    assert last is not None
    explanation = BRIDGE + output[:end].strip() if end is not None else None
    return {"output": output, "output_sha256": digest_text(output), "prompt": prompt, "prompt_sha256": digest_text(prompt),
            "prefill": prefill, "shared_bridge": BRIDGE,
            "extraction": "Shared generic bridge plus exact first complete generated sentence; quoted equality remains only in teacher prompt",
            "explanation": explanation, "student_tokens": len(student_tok.encode(explanation, add_special_tokens=False)) if explanation else None,
            "finish_reason": "sentence_boundary" if end is not None else last.finish_reason,
            "generation_tokens": last.generation_tokens, "seconds": time.perf_counter() - start}


def main(probe):
    targets = {r["source_id"]: r for r in rows(OUT / "teacher_targets.jsonl")}
    locations = rows(OUT / "locations.jsonl")
    reviewed("location", locations)
    lookup = {r["source_id"]: r for r in sources()}
    with gpu("edit_probe_v3" if probe else "edits_v3"):
        sign("edit_probe_v3" if probe else "edits_v3", [Path(__file__), OUT / "locations.jsonl", OUT / "teacher_targets.jsonl"],
             {"system": SYSTEM, "bridge": BRIDGE, "revision": "JSON task and generic rule task did not reliably align with the selected step; focus the assistant continuation on the exact quoted equality, retaining all earlier attempts"})
        model, tok = load_teacher()
        student_tok = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True)
        for condition in (["difficulty_only"] if probe else ["difficulty_only", "difficulty_attention"]):
            path = OUT / ("edit_probe_v3.jsonl" if probe else f"{condition}_edits_v3.jsonl")
            old = rows(path)
            expected = locations[:3] if probe else locations
            assert [r["source_id"] for r in old] == [r["source_id"] for r in expected[:len(old)]]
            prior = {r["source_id"]: r for r in rows(OUT / "edit_probe_v3.jsonl")} if not probe and condition == "difficulty_only" else {}
            for location in expected[len(old):]:
                sid = location["source_id"]
                prompt, prefill = build_prompt(tok, lookup[sid], targets[sid]["output"], location, condition)
                if sid in prior:
                    result = dict(prior[sid])
                    assert result["prompt_sha256"] == digest_text(prompt)
                    result["reused_from"] = "edit_probe_v3.jsonl; exact same signed prompt, no additional generation"
                else:
                    result = {"source_id": sid, "condition": condition, **generate(model, tok, prompt, prefill, student_tok),
                              "teacher_output_sha256": targets[sid]["output_sha256"], "location_sha256": location["output_sha256"]}
                append(path, result)
                old.append(result)
                print(json.dumps({k: result[k] for k in ["source_id", "condition", "explanation", "student_tokens", "finish_reason", "output_sha256"]}, ensure_ascii=False), flush=True)
                baseline.save(OUT / "material_status.json", {"stage": "edit_probe_v3" if probe else "edits_v3", "condition": condition, "completed": len(old), "total": len(expected)})
        baseline.save(OUT / "material_status.json", {"stage": "needs_edit_probe_v3_audit" if probe else "needs_edits_v3_full_text_audit", "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["probe", "run"])
    args = parser.parse_args()
    main(args.stage == "probe")
