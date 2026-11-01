"""Short local explanations with a fixed style demonstration and exact extraction."""
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

SYSTEM = "You are a mathematics tutor explaining one local transition. Give one conceptual sentence in words only. Do not include numerical values, compute results, state the final answer, or solve the whole problem."
PREFILL = "The underlying rule is that "
DEMOS = [
    {"role": "user", "content": "Explain why initial stock and two deliveries are added before considering sales. Give one conceptual sentence, with no numerical values."},
    {"role": "assistant", "content": "The underlying rule is that received quantities increase the same inventory, so they are added to the initial stock before any outgoing quantity is subtracted."},
    {"role": "user", "content": "Explain why an equation equating a multiple of an unknown to a known quantity is divided by that multiplier. Give one conceptual sentence, with no numerical values."},
    {"role": "assistant", "content": "The underlying rule is that dividing both sides by the same nonzero multiplier preserves equality and isolates the unknown quantity."},
]


def user_message(question, prefix, step, evidence):
    message = "Explain only the rule connecting the earlier context to the specified step. "
    message += "Use one conceptual sentence in words, following the examples. Do not include numerical values.\n\n"
    message += "Problem context: " + question + "\nEarlier reasoning: " + (prefix.strip() or "None yet.")
    message += "\nSpecified step: " + step
    if evidence:
        message += "\nStudent-attention-ranked prior context (relevance heuristic only):\n" + evidence
    return message + "\nOne conceptual sentence explaining the rule, without calculations or an answer:"


def generate(model, tok, user, student_tok):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    messages = [{"role": "system", "content": SYSTEM}, *DEMOS, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PREFILL
    ids = tok.encode(prompt, add_special_tokens=False)
    assert len(ids) + C["edit_max_new_tokens"] <= C["context_tokens"]
    output, last, end = "", None, None
    start = time.perf_counter()
    for part in stream_generate(model, tok, ids, max_tokens=C["edit_max_new_tokens"], sampler=make_sampler(temp=0.0)):
        output += part.text
        last = part
        # Exact first complete sentence. Do not cut inside a decimal or TeX
        # command; no words are added or paraphrased after generation.
        match = re.search(r"[.!?](?=\s|$)", output)
        if match:
            end = match.end()
            break
    assert last is not None
    explanation = PREFILL + output[:end].strip() if end is not None else None
    return {"output": output, "output_sha256": digest_text(output), "prompt": prompt, "prompt_sha256": digest_text(prompt),
            "prefill": PREFILL, "extraction": "fixed shared assistant prefill plus exact first complete generated sentence",
            "explanation": explanation, "student_tokens": len(student_tok.encode(explanation, add_special_tokens=False)) if explanation else None,
            "finish_reason": "sentence_boundary" if end is not None else last.finish_reason,
            "generation_tokens": last.generation_tokens, "seconds": time.perf_counter() - start,
            "contains_digits": bool(re.search(r"\d", explanation or ""))}


def main(probe):
    targets = {r["source_id"]: r for r in rows(OUT / "teacher_targets.jsonl")}
    locations = rows(OUT / "locations.jsonl")
    reviewed("location", locations)
    lookup = {r["source_id"]: r for r in sources()}
    with gpu("edit_probe_v2" if probe else "edits_v2"):
        sign("edit_probe_v2" if probe else "edits_v2", [Path(__file__), OUT / "locations.jsonl", OUT / "teacher_targets.jsonl"],
             {"system": SYSTEM, "demos": DEMOS, "prefill": PREFILL,
              "revision": "Original JSON prompt gave full solutions; first continuation probe still gave evaluated/final results; use demonstrated word-only rule explanations with exact sentence extraction"})
        model, tok = load_teacher()
        student_tok = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True)
        conditions = ["difficulty_only"] if probe else ["difficulty_only", "difficulty_attention"]
        for condition in conditions:
            path = OUT / ("edit_probe_v2.jsonl" if probe else f"{condition}_edits_v2.jsonl")
            old = rows(path)
            expected = locations[:3] if probe else locations
            assert [r["source_id"] for r in old] == [r["source_id"] for r in expected[:len(old)]]
            prior = {r["source_id"]: r for r in rows(OUT / "edit_probe_v2.jsonl")} if not probe and condition == "difficulty_only" else {}
            for row in expected[len(old):]:
                sid = row["source_id"]
                evidence = "\n".join(e["text"] for e in row["evidence"]) if condition == "difficulty_attention" else ""
                prompt = user_message(lookup[sid]["question"], targets[sid]["output"][:row["chosen"]["start"]], row["chosen"]["text"], evidence)
                if sid in prior:
                    result = dict(prior[sid])
                    assert result["user_message_sha256"] == digest_text(prompt)
                    result["reused_from"] = "edit_probe_v2.jsonl; identical user/system/demos/prefill, no new generation"
                else:
                    result = {"source_id": sid, "condition": condition, **generate(model, tok, prompt, student_tok),
                              "user_message_sha256": digest_text(prompt), "teacher_output_sha256": targets[sid]["output_sha256"],
                              "location_sha256": row["output_sha256"]}
                append(path, result)
                old.append(result)
                print(json.dumps({k: result[k] for k in ["source_id", "condition", "explanation", "student_tokens", "contains_digits", "finish_reason", "output_sha256"]}, ensure_ascii=False), flush=True)
                baseline.save(OUT / "material_status.json", {"stage": "edit_probe_v2" if probe else "edits_v2", "condition": condition, "completed": len(old), "total": len(expected)})
        baseline.save(OUT / "material_status.json", {"stage": "needs_edit_probe_v2_audit" if probe else "needs_edits_v2_full_text_audit", "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["probe", "run"])
    args = parser.parse_args()
    main(args.stage == "probe")
