"""Generate, locate and locally expand the fixed small-pilot training targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import baseline
from pilot_common import BASE, C, DATA, HERE, OUT, ROOT, append, digest_text, gpu, reviewed, rows, sign, sources

TEACHER_EXTRA = "\nWrite a concise solution, one complete reasoning step per line. Avoid headings and code. End with a single boxed numerical answer."
EDIT_TEMPLATE = """Explain the connection immediately before the indicated reasoning step for a student.
Write only one short explanatory paragraph, about 35-50 words. Do not solve the rest of the problem.
Explain why the quantities and the mathematical rule connect; do not merely repeat the equation.
Do not state the final answer, give new evaluated arithmetic results, use code, add headings, or change any original step.
Return JSON with one key, \"explanation\".

Problem:
{question}

Prior reasoning:
{prefix}

Next step that the student finds difficult:
{step}
{evidence}
"""
STOP = set("a an the and or but so thus therefore hence after before let we i it its this that these those to of for in on at by from with as is are was were be been being have has had then now step first next finally since if because will would can must how what which when where does do did each".split())


def content(text):
    value = text.strip().lower()
    return bool(re.search(r"[a-z0-9=+*/<>−-]", value)) and value not in STOP


def generate(model, tok, user, limit):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": BASE["teacher_system"]}, {"role": "user", "content": user}],
        tokenize=True, return_dict=False, add_generation_prompt=True)
    if hasattr(prompt, "keys"):
        prompt = prompt["input_ids"]
    assert len(prompt) + limit <= C["context_tokens"], "Would exceed context"
    text, last = "", None
    start = time.perf_counter()
    for part in stream_generate(model, tok, prompt, max_tokens=limit, sampler=make_sampler(temp=0.0)):
        text += part.text
        last = part
    assert last is not None
    return {"output": text, "output_sha256": digest_text(text), "prompt_tokens": len(prompt),
            "generation_tokens": last.generation_tokens, "finish_reason": last.finish_reason,
            "seconds": time.perf_counter() - start}


def load_teacher():
    import mlx.core as mx
    from mlx_lm import load
    mx.set_memory_limit(14 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(C["seed"])
    # Verify exact weight bytes against the already signed baseline asset record.
    signed = json.loads((ROOT / BASE["output_dir"] / "dev_baseline/teacher_reference/signature.json").read_text())
    folder = ROOT / BASE["teacher_dir"]
    for name, spec in signed["model_files"].items():
        assert baseline.sha(folder / name) == spec["sha256"]
    return load(str(folder), tokenizer_config={"local_files_only": True, "trust_remote_code": False})


def teacher(wait):
    items = sources()
    path = OUT / "teacher_targets.jsonl"
    with gpu("teacher_targets", wait=wait):
        for _ in range(10):
            state = json.loads((ROOT / BASE["output_dir"] / "pipeline_status.json").read_text())
            if state["stage"] != "dev_baselines":
                break
            time.sleep(1)
        assert state["stage"] == "needs_dev_output_audit_and_training_protocol", "Existing development pipeline has not finished normally"
        sign("teacher_targets", [Path(__file__), DATA / "train.json",
             ROOT / BASE["output_dir"] / "dev_baseline/teacher_reference/signature.json"],
             {"prompt_suffix": TEACHER_EXTRA, "source_ids": [s["source_id"] for s in items]})
        old = rows(path)
        assert len(old) <= len(items)
        for row, source in zip(old, items):
            assert row["source_id"] == source["source_id"] and row["question_sha256"] == source["question_sha256"]
            assert digest_text(row["output"]) == row["output_sha256"]
        if len(old) < len(items):
            model, tok = load_teacher()
            for item in items[len(old):]:
                result = generate(model, tok, item["question"] + TEACHER_EXTRA, C["teacher_max_new_tokens"])
                row = {"source_id": item["source_id"], "question_sha256": item["question_sha256"],
                       "gold": item["gold"], **result, **baseline.score(result["output"], item["gold"])}
                append(path, row)
                old.append(row)
                baseline.save(OUT / "material_status.json", {"stage": "teacher_targets", "completed": len(old), "total": len(items), "pid": __import__("os").getpid()})
                print(f'teacher targets {len(old)}/{len(items)}; EM={row["correct"]}', flush=True)
        baseline.save(OUT / "material_status.json", {"stage": "needs_teacher_full_text_audit", "completed": len(old), "training_started": False})


def steps(text):
    # Preserve exact text and character boundaries; every problem may have a
    # different number of nonempty lines. A boxed-only answer is not a step.
    result = []
    for match in re.finditer(r"[^\n]+", text):
        line = match.group()
        stripped = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line)
        if not stripped.strip() or not content(re.sub(r"\\boxed\{[^{}]*\}", "", stripped)):
            continue
        result.append({"start": match.start(), "end": match.end(), "text": line})
    return result


def locate():
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    generated = rows(OUT / "teacher_targets.jsonl")
    audits = reviewed("teacher", generated)
    accepted = [r for r in generated if audits[r["source_id"]]["accepted"]]
    lookup = {r["source_id"]: r for r in sources()}
    path = OUT / "locations.jsonl"
    with gpu("locate"):
        sign("locations", [Path(__file__), OUT / "teacher_targets.jsonl", OUT / "teacher_reviews.jsonl"],
             {"candidates": "Every substantive full line, including final mathematical step; boxed-only lines excluded",
              "step_score": "mean NLL over non-stopword content tokens", "queries": "up to eight highest NLL content token predictors in chosen line",
              "evidence": "top two prior complete sentence/line units by mean attention per content token; exclude current step, punctuation and stopwords"})
        old = rows(path)
        assert [r["source_id"] for r in old] == [r["source_id"] for r in accepted[:len(old)]]
        if len(old) == len(accepted):
            return
        baseline.verify_student(BASE)
        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)
        model, tok = load(str(ROOT / BASE["student_dir"]), tokenizer_config={"local_files_only": True})
        model.eval()

        class Capture(nn.Module):
            def __init__(self, original, positions):
                super().__init__()
                self.original, self.positions, self.captured = original, positions, None

            def __call__(self, x, mask=None, cache=None):
                assert cache is None
                result = self.original(x, mask, cache)
                a = self.original
                batch, length, _ = x.shape
                q = a.q_proj(x).reshape(batch, length, a.n_heads, -1).transpose(0, 2, 1, 3)
                k = a.k_proj(x).reshape(batch, length, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q, k = a.rope(q), a.rope(k)
                q = q[:, :, self.positions].astype(mx.float32)
                k = mx.repeat(k, a.n_heads // a.n_kv_heads, axis=1).astype(mx.float32)
                score = q @ k.transpose(0, 1, 3, 2) * a.scale
                causal = mx.arange(length)[None, :] <= mx.array(self.positions)[:, None]
                self.captured = mx.softmax(mx.where(causal[None, None], score, -mx.inf), axis=-1)
                return result

        for row in accepted[len(old):]:
            tick = time.perf_counter()
            item, answer = lookup[row["source_id"]], row["output"]
            prompt = BASE["student_prompt"].format(question=item["question"])
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            enc = tok._tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
            target, offsets = enc["input_ids"], enc["offset_mapping"]
            ids = prompt_ids + target
            assert len(ids) <= C["context_tokens"]
            x = mx.array([ids[:-1]])
            hidden = model.model(x)
            logits = model.model.embed_tokens.as_linear(hidden[:, len(prompt_ids) - 1:]).astype(mx.float32)
            nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, mx.array(target)[None, :, None], axis=-1).squeeze(-1)
            mx.eval(logits, nll)
            losses, original = np.array(nll)[0], np.array(logits)
            candidates = []
            for span in steps(answer):
                positions = [i for i, (a, b) in enumerate(offsets)
                             if a < span["end"] and b > span["start"] and content(answer[a:b])]
                if positions:
                    candidates.append({**span, "target_indexes": positions, "mean_nll": float(np.mean(losses[positions]))})
            assert candidates, f'No substantive steps: {row["source_id"]}'
            chosen = max(candidates, key=lambda r: r["mean_nll"])
            top = sorted(chosen["target_indexes"], key=lambda i: float(losses[i]), reverse=True)[:C["attention_queries"]]
            queries = [len(prompt_ids) + i - 1 for i in top]
            wrappers = []
            try:
                for layer in model.layers[-C["attention_layers"]:]:
                    capture = Capture(layer.self_attn, queries)
                    wrappers.append((layer, layer.self_attn, capture))
                    layer.self_attn = capture
                hidden2 = model.model(x)
                logits2 = model.model.embed_tokens.as_linear(hidden2[:, len(prompt_ids) - 1:]).astype(mx.float32)
                captures = [w.captured for _, _, w in wrappers]
                mx.eval(logits2, *captures)
                delta = float(np.max(np.abs(np.array(logits2) - original)))
                assert delta == 0
                norm = max(float(mx.max(mx.abs(mx.sum(w, axis=-1) - 1)).item()) for w in captures)
                assert norm < 1e-5
                for weights in captures:
                    for j, qpos in enumerate(queries):
                        assert float(mx.sum(weights[..., j, qpos + 1:]).item()) == 0
                attention = np.mean(np.stack([np.array(w)[0] for w in captures]), axis=(0, 1, 2))
            finally:
                for layer, original_attn, _ in wrappers:
                    layer.self_attn = original_attn
            # Map question tokens with exact offsets, then previous target lines.
            penc = tok._tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
            assert penc["input_ids"] == prompt_ids
            units = []
            qstart = prompt.index(item["question"])
            for match in re.finditer(r".+?(?:[.!?;](?=\s|$)|$)", item["question"]):
                a, b = qstart + match.start(), qstart + match.end()
                positions = [i for i, (u, v) in enumerate(penc["offset_mapping"]) if u < b and v > a and content(prompt[u:v])]
                if positions:
                    units.append({"kind": "question", "text": match.group().strip(), "input_positions": positions})
            for span in candidates:
                if span["end"] <= chosen["start"]:
                    positions = [len(prompt_ids) + i for i in span["target_indexes"]]
                    units.append({"kind": "prior_step", "text": span["text"], "input_positions": positions})
            for unit in units:
                assert max(unit["input_positions"]) < min(queries)
                unit["attention_density"] = float(np.mean(attention[unit["input_positions"]]))
            ranked = sorted(units, key=lambda u: u["attention_density"], reverse=True)
            result = {"source_id": row["source_id"], "teacher_output_sha256": row["output_sha256"],
                      "output_sha256": digest_text(json.dumps({"chosen": chosen, "evidence": ranked[:C["evidence_units"]]}, sort_keys=True)),
                      "chosen": chosen, "candidates": candidates, "evidence": ranked[:C["evidence_units"]],
                      "all_evidence_units": ranked, "target_query_indexes": top, "predictor_positions": queries,
                      "input_tokens": len(ids), "max_logits_difference": delta, "attention_sum_error": norm,
                      "future_attention_mass": 0, "seconds": time.perf_counter() - tick,
                      "peak_mlx_memory_bytes": mx.get_peak_memory()}
            append(path, result)
            old.append(result)
            print(f'located {len(old)}/{len(accepted)}: {row["source_id"]}; {chosen["text"]}', flush=True)
            del logits, logits2, hidden, hidden2, captures, original
            mx.clear_cache()
        baseline.save(OUT / "material_status.json", {"stage": "locations_complete_needs_audit", "completed": len(old), "training_started": False})


def edits():
    from transformers import AutoTokenizer
    generated = rows(OUT / "teacher_targets.jsonl")
    targets = {r["source_id"]: r for r in generated}
    locations = rows(OUT / "locations.jsonl")
    audits = reviewed("location", locations)
    locations = [r for r in locations if audits[r["source_id"]]["accepted"]]
    lookup = {r["source_id"]: r for r in sources()}
    with gpu("edits"):
        sign("edits", [Path(__file__), OUT / "locations.jsonl", OUT / "location_reviews.jsonl", OUT / "teacher_targets.jsonl"],
             {"template": EDIT_TEMPLATE, "conditions": ["difficulty_only", "difficulty_attention"]})
        model, tok = load_teacher()
        student_tok = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True)
        for condition in ["difficulty_only", "difficulty_attention"]:
            path = OUT / f"{condition}_edits.jsonl"
            old = rows(path)
            assert [r["source_id"] for r in old] == [r["source_id"] for r in locations[:len(old)]]
            for row in locations[len(old):]:
                item = lookup[row["source_id"]]
                answer = targets[row["source_id"]]["output"]
                evidence = ""
                if condition == "difficulty_attention":
                    evidence = "\nStudent-attention-ranked prior context (a relevance heuristic, not proof):\n" + "\n".join(u["text"] for u in row["evidence"])
                prompt = EDIT_TEMPLATE.format(question=item["question"], prefix=answer[:row["chosen"]["start"]].strip() or "(No earlier reasoning.)",
                                              step=row["chosen"]["text"], evidence=evidence)
                result = generate(model, tok, prompt, C["edit_max_new_tokens"])
                explanation, error = None, None
                try:
                    payload = result["output"].strip()
                    if payload.startswith("```json") and payload.endswith("```"):
                        payload = payload[len("```json"):-3].strip()
                    value = json.loads(payload)
                    assert set(value) == {"explanation"} and isinstance(value["explanation"], str)
                    explanation = value["explanation"].strip()
                    assert explanation
                except (ValueError, TypeError, AssertionError) as exc:
                    error = type(exc).__name__
                record = {"source_id": item["source_id"], "condition": condition, "teacher_output_sha256": targets[item["source_id"]]["output_sha256"],
                          "location_sha256": row["output_sha256"], "prompt_sha256": digest_text(prompt), "prompt": prompt,
                          **result, "explanation": explanation, "parse_error": error,
                          "student_tokens": len(student_tok.encode(explanation, add_special_tokens=False)) if explanation else None}
                append(path, record)
                old.append(record)
                baseline.save(OUT / "material_status.json", {"stage": "edits", "condition": condition, "completed": len(old), "total": len(locations)})
                print(f'{condition} edits {len(old)}/{len(locations)}; tokens={record["student_tokens"]}; parse_error={error}', flush=True)
        baseline.save(OUT / "material_status.json", {"stage": "needs_edits_full_text_audit", "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["teacher", "locate", "edits"])
    parser.add_argument("--wait-existing", action="store_true")
    args = parser.parse_args()
    if args.stage == "teacher":
        teacher(args.wait_existing)
    elif args.stage == "locate":
        locate()
    else:
        edits()
