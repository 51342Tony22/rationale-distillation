"""Small local-teacher instruction/continuation feasibility check on train data."""
from __future__ import annotations

import json
from pathlib import Path
import time

import baseline
from pilot_common import C, OUT, append, digest_text, gpu, rows, sign, sources
from pilot_materials import load_teacher

SYSTEM = "You are a mathematics tutor. Explain only the specified transition in at most two short sentences. Do not solve the whole problem or state its final answer."
PREFILL = "This step is justified because "


def messages(question, prefix, step, evidence=""):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        "Explain why the following single mathematical step follows from the context. "
        "Use ordinary prose, no JSON, no headings, no new evaluated arithmetic results.\n\n"
        + "Context problem: " + question + "\nEarlier reasoning: " + (prefix.strip() or "None yet.")
        + "\nStep to explain: " + step + ("\nRelevant prior context selected by student attention:\n" + evidence if evidence else "")
        + "\nGive only the conceptual reason for this step, not a solution to the context problem."}]


def run_one(model, tok, message, max_tokens=160):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + PREFILL
    ids = tok.encode(prompt, add_special_tokens=False)
    assert len(ids) + max_tokens <= C["context_tokens"]
    text, last, boundary = "", None, None
    tick = time.perf_counter()
    for part in stream_generate(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        text += part.text
        last = part
        # Stop on paragraph boundary, not on punctuation inside decimals/TeX.
        if "\n\n" in text:
            boundary = text.index("\n\n")
            break
    assert last is not None
    extracted = text[:boundary] if boundary is not None else text
    return {"output": text, "output_sha256": digest_text(text), "prompt": prompt, "prompt_sha256": digest_text(prompt),
            "prefill": PREFILL, "explanation": PREFILL + extracted.strip(), "generation_tokens": last.generation_tokens,
            "finish_reason": "paragraph_boundary" if boundary is not None else last.finish_reason,
            "seconds": time.perf_counter() - tick}


def main():
    with gpu("edit_feasibility_probe"):
        sign("edit_feasibility_probe", [Path(__file__), OUT / "locations.jsonl", OUT / "teacher_targets.jsonl"],
             {"system": SYSTEM, "prefill": PREFILL, "source_ids": [r["source_id"] for r in sources()[:3]],
              "old_attempt": "Original JSON edit queue stopped after four noncompliant responses; raw attempts preserved"})
        assert not (OUT / "edit_probe.jsonl").exists(), "Probe already saved; inspect instead of regenerating"
        model, tok = load_teacher()
        loc = {r["source_id"]: r for r in rows(OUT / "locations.jsonl")}
        targets = {r["source_id"]: r for r in rows(OUT / "teacher_targets.jsonl")}
        for item in sources()[:3]:
            chosen = loc[item["source_id"]]["chosen"]
            message = messages(item["question"], targets[item["source_id"]]["output"][:chosen["start"]], chosen["text"])
            result = {"source_id": item["source_id"], **run_one(model, tok, message)}
            append(OUT / "edit_probe.jsonl", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        baseline.save(OUT / "material_status.json", {"stage": "needs_edit_probe_audit", "training_started": False})


if __name__ == "__main__":
    main()
