"""Retain the tested v3 prompt; require confirmed whitespace for sentence stops."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

from transformers import AutoTokenizer

import baseline
import pilot_edits_v3 as v3
from pilot_common import BASE, C, OUT, ROOT, append, digest_text, gpu, reviewed, rows, sign, sources
from pilot_materials import load_teacher


def sentence_end(text):
    # During streaming, a terminal dot might still become the decimal point
    # in the next chunk. Wait for actual whitespace before accepting a stop.
    match = re.search(r"[.!?](?=\s)", text)
    return match.end() if match else None


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
        end = sentence_end(output)
        if end is not None:
            break
    assert last is not None
    if end is None and last.finish_reason == "stop" and output.rstrip().endswith((".", "!", "?")):
        end = len(output.rstrip())
    explanation = v3.BRIDGE + output[:end].strip() if end is not None else None
    return {"output": output, "output_sha256": digest_text(output), "prompt": prompt, "prompt_sha256": digest_text(prompt),
            "prefill": prefill, "shared_bridge": v3.BRIDGE,
            "extraction": "Shared generic bridge plus exact first complete generated sentence; sentence boundary confirmed by whitespace or EOS; quoted equality remains in teacher input only",
            "explanation": explanation, "student_tokens": len(student_tok.encode(explanation, add_special_tokens=False)) if explanation else None,
            "finish_reason": "sentence_boundary" if end is not None else last.finish_reason,
            "generation_tokens": last.generation_tokens, "seconds": time.perf_counter() - start}


def main():
    assert sentence_end("multiply by 1.") is None
    assert sentence_end("multiply by 1.20 preserves the ratio. ") == len("multiply by 1.20 preserves the ratio.")
    assert sentence_end("a count is 30") is None
    targets = {r["source_id"]: r for r in rows(OUT / "teacher_targets.jsonl")}
    locations = rows(OUT / "locations.jsonl")
    reviewed("location", locations)
    lookup = {r["source_id"]: r for r in sources()}
    with gpu("edits_v4"):
        sign("edits_v4", [Path(__file__), Path(v3.__file__), OUT / "locations.jsonl", OUT / "teacher_targets.jsonl"],
             {"system": v3.SYSTEM, "bridge": v3.BRIDGE,
              "revision": "v3 streamed terminal dots could be decimal prefixes; stopped its queue and retained all original rows, repeat using identical prompt and safe sentence stopping for both conditions"})
        model, tok = load_teacher()
        student_tok = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True)
        for condition in ["difficulty_only", "difficulty_attention"]:
            path = OUT / f"{condition}_edits_v4.jsonl"
            old = rows(path)
            assert [r["source_id"] for r in old] == [r["source_id"] for r in locations[:len(old)]]
            for location in locations[len(old):]:
                sid = location["source_id"]
                prompt, prefill = v3.build_prompt(tok, lookup[sid], targets[sid]["output"], location, condition)
                result = {"source_id": sid, "condition": condition, **generate(model, tok, prompt, prefill, student_tok),
                          "teacher_output_sha256": targets[sid]["output_sha256"], "location_sha256": location["output_sha256"]}
                append(path, result)
                old.append(result)
                print(json.dumps({k: result[k] for k in ["source_id", "condition", "explanation", "student_tokens", "finish_reason", "output_sha256"]}, ensure_ascii=False), flush=True)
                baseline.save(OUT / "material_status.json", {"stage": "edits_v4", "condition": condition, "completed": len(old), "total": len(locations)})
        baseline.save(OUT / "material_status.json", {"stage": "needs_edits_v4_full_text_audit", "training_started": False})


if __name__ == "__main__":
    main()
