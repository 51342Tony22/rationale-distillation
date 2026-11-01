"""Small student-only position diagnostics. No teacher generation or training.

prepare: tokenizer/source preflight only; run: six fixed training-source cases.
Outputs are isolated from signed pilot100 files. Existing rows resume only under
an identical signature, source order and content hashes.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path

import baseline
from pilot_common import append, digest_text, rows
from pilot_localize_v2 import mathematical_steps
from pilot_materials import STOP, content
from selector_masking_core import diagnose_step, exploratory_choices

HERE = Path(__file__).absolute().parent
ROOT = baseline.ROOT
CONFIG = HERE / "selector_masking_config.json"
BASE = json.loads(baseline.CONFIG.read_text())
C = json.loads(CONFIG.read_text())
OUT = ROOT / C["output_dir"]
OLD = ROOT / "output/math_distill_v1/pilot100"
STOP.update("text frac dfrac times div boxed left right begin end mathrm textbf".split())


def contained(offsets, start, end, text, *, only_content=True):
    # A token crossing a unit boundary is omitted, never assigned to both units.
    return [i for i, (a, b) in enumerate(offsets)
            if start <= a < b <= end and (not only_content or content(text[a:b]))]


def prepare_case(row, source, tokenizer):
    prompt, answer = row["prompt"], row["completion"]
    assert prompt == BASE["student_prompt"].format(question=source["question"])
    penc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    aenc = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    assert penc["input_ids"] == row["prompt_ids"]
    assert aenc["input_ids"] == row["target_ids"][:len(aenc["input_ids"])], "Stored target tokenization changed"
    ids = penc["input_ids"] + aenc["input_ids"]
    assert len(ids) <= C["context_tokens"], "No truncation permitted"
    p = len(penc["input_ids"])
    spans = mathematical_steps(answer)
    candidates, omitted = [], []
    for i, span in enumerate(spans):
        if i == len(spans) - 1 or "\\boxed" in span["text"]:
            omitted.append({**span, "reason": "conservative_final_step_or_box_exclusion"})
            continue
        all_positions = contained(aenc["offset_mapping"], span["start"], span["end"], answer, only_content=False)
        positions = contained(aenc["offset_mapping"], span["start"], span["end"], answer)
        if not all_positions or not positions:
            omitted.append({**span, "reason": "no_fully_contained_content_tokens"})
            continue
        step = {"step_index": i, "text": span["text"], "char_start": span["start"],
                "char_end": span["end"], "start_position": p + min(all_positions),
                "targets": [p + j for j in positions]}
        units = []
        qstart = prompt.index(source["question"])
        for m in re.finditer(r".+?(?:[.!?;](?=\s|$)|$)", source["question"]):
            positions = contained(penc["offset_mapping"], qstart + m.start(), qstart + m.end(), prompt)
            if positions:
                units.append({"kind": "question", "text": m.group().strip(), "positions": positions})
        # Prior full lines, including prose, not just preceding equalities.
        for m in re.finditer(r"[^\n]+", answer[:span["start"]]):
            if "\\boxed" in m.group():
                continue
            positions = contained(aenc["offset_mapping"], m.start(), m.end(), answer)
            if positions:
                units.append({"kind": "prior_reasoning", "text": m.group(), "positions": [p + j for j in positions]})
        assert all(max(u["positions"]) < step["start_position"] for u in units)
        candidates.append({"step": step, "units": units})
    return {"source_id": row["source_id"], "question_sha256": source["question_sha256"],
            "teacher_output_sha256": row["completion_sha256"], "input_ids": ids,
            "candidates": candidates, "omitted_candidates": omitted}


def prepare():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((OLD / "final_integrity_audit.json").read_text())
    assert audit["status"] == "passed"
    required = ["data/math_distill_v1/pilot100/train.json", "data/math_distill_v1/pilot100/manifest.json",
                "output/math_distill_v1/pilot100/training_data/ordinary_cot.json",
                "output/math_distill_v1/pilot100/teacher_targets.jsonl", "output/math_distill_v1/pilot100/teacher_reviews.jsonl"]
    for name in required:
        assert baseline.sha(ROOT / name) == audit["files_sha256"][name], f"Frozen source changed: {name}"
    material = json.loads((OLD / "training_data/ordinary_cot.json").read_text())
    source = {r["source_id"]: r for r in json.loads((ROOT / required[0]).read_text())}
    reviews = {r["source_id"]: r for r in rows(OLD / "teacher_reviews.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True, trust_remote_code=False)
    cases = []
    for row in material[:C["case_count"]]:
        assert reviews[row["source_id"]]["accepted"]
        assert digest_text(row["completion"]) == row["completion_sha256"] == reviews[row["source_id"]]["output_sha256"]
        cases.append(prepare_case(row, source[row["source_id"]], tokenizer))
    files = [CONFIG, baseline.CONFIG, Path(__file__), HERE / "selector_masking_core.py",
             HERE / "pilot_localize_v2.py", HERE / "pilot_localize.py", HERE / "pilot_materials.py",
             HERE / "pilot_common.py", Path(baseline.__file__), *[ROOT / name for name in required],
             ROOT / BASE["data_dir"] / "student_manifest.json"]
    signature = {"config": C, "inputs": {str(f.relative_to(ROOT)): baseline.sha(f) for f in files},
                 "versions": {name: importlib.metadata.version(name) for name in ["mlx", "mlx-lm", "transformers", "numpy"]},
                 "source_ids": [c["source_id"] for c in cases], "student_revision": BASE["student_revision"],
                 "cases_sha256": digest_text(json.dumps(cases, sort_keys=True))}
    baseline.freeze(OUT / "signature.json", signature)
    baseline.freeze(OUT / "cases.json", cases)
    preflight = {"status": "tokenizer_and_source_checks_passed", "model_loaded": False,
                 "signature_sha256": baseline.sha(OUT / "signature.json"),
                 "cases": [{"source_id": c["source_id"], "candidate_steps": len(c["candidates"]),
                            "input_tokens": len(c["input_ids"])} for c in cases],
                 "official_test_accessed": False, "training_started": False}
    baseline.freeze(OUT / "preflight.json", preflight)
    return cases


class MLXBackend:
    def __init__(self, model, layers):
        self.model, self.layers = model, layers

    def score(self, ids, targets):
        import mlx.core as mx
        hidden = self.model.model(mx.array([ids[:-1]]))[:, [p - 1 for p in targets]]
        logits = (self.model.model.embed_tokens.as_linear(hidden) if self.model.args.tie_word_embeddings
                  else self.model.lm_head(hidden)).astype(mx.float32)
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, mx.array([[ids[p] for p in targets]])[..., None], axis=-1).squeeze(-1)
        mx.eval(nll)
        return nll[0].tolist()

    def observe(self, ids, targets, queries):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        class Capture(nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original, self.captured = original, None

            def __call__(self, x, mask=None, cache=None):
                assert cache is None and isinstance(mask, str) and mask == "causal", "Sparse capture supports only the no-cache Qwen2 causal path"
                output = self.original(x, mask, cache)
                a = self.original
                batch, length, _ = x.shape
                q = a.q_proj(x).reshape(batch, length, a.n_heads, -1).transpose(0, 2, 1, 3)
                k = a.k_proj(x).reshape(batch, length, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q, k = a.rope(q), a.rope(k)
                q = q[:, :, queries].astype(mx.float32)
                k = mx.repeat(k, a.n_heads // a.n_kv_heads, axis=1).astype(mx.float32)
                logits = q @ k.transpose(0, 1, 3, 2) * a.scale
                causal = mx.arange(length)[None, :] <= mx.array(queries)[:, None]
                self.captured = mx.softmax(mx.where(causal[None, None], logits, -mx.inf), axis=-1)
                return output

        wrapped = []
        try:
            for layer in self.model.layers[-self.layers:]:
                wrapper = Capture(layer.self_attn)
                wrapped.append((layer, layer.self_attn, wrapper))
                layer.self_attn = wrapper
            nll = self.score(ids, targets)
            captures = [w.captured for _, _, w in wrapped]
            mx.eval(*captures)
            attention = np.mean(np.stack([np.array(w)[0] for w in captures]), axis=(0, 1))
            # score() inputs omit the last target token; pad its absent key by 0.
            return nll, np.pad(attention, ((0, 0), (0, 1))).tolist()
        finally:
            for layer, original, _ in wrapped:
                layer.self_attn = original


def run():
    import mlx.core as mx
    from mlx_lm import load
    cases = prepare()
    with baseline.gpu_lock(OUT):
        checked = baseline.verify_student(BASE)
        baseline.freeze(OUT / "model_signature.json", {"signature_sha256": baseline.sha(OUT / "signature.json"), "model_files": checked})
        path = OUT / "diagnostics.jsonl"
        existing = rows(path)
        assert len(existing) <= len(cases)
        for saved, case in zip(existing, cases):
            value = dict(saved)
            digest = value.pop("record_sha256")
            assert digest_text(json.dumps(value, sort_keys=True)) == digest
            assert saved["source_id"] == case["source_id"] and saved["teacher_output_sha256"] == case["teacher_output_sha256"]
        if len(existing) < len(cases):
            mx.set_memory_limit(14 * 1024**3)
            mx.set_cache_limit(512 * 1024**2)
            mx.random.seed(C["seed"])
            model, _ = load(str(ROOT / BASE["student_dir"]), tokenizer_config={"local_files_only": True, "trust_remote_code": False})
            model.eval()
            backend = MLXBackend(model, C["attention_layers"])
            for case in cases[len(existing):]:
                start = time.perf_counter()
                steps = []
                for candidate in case["candidates"]:
                    steps.append(diagnose_step(case["input_ids"], candidate["step"], candidate["units"], backend,
                                 seed=C["seed"], key=f'{case["source_id"]}:{candidate["step"]["step_index"]}',
                                 controls=C["random_controls"], queries=C["attention_queries"]))
                    mx.clear_cache()
                result = {"source_id": case["source_id"], "teacher_output_sha256": case["teacher_output_sha256"],
                          "steps": steps, "choices": exploratory_choices(steps, seed=C["seed"], key=case["source_id"]),
                          "seconds": time.perf_counter() - start, "peak_mlx_memory_bytes": mx.get_peak_memory()}
                result["record_sha256"] = digest_text(json.dumps(result, sort_keys=True))
                append(path, result)
                existing.append(result)
                baseline.save(OUT / "status.json", {"stage": "student_masking_diagnostics", "completed": len(existing), "total": len(cases)})
                print(f'{len(existing)}/{len(cases)} {case["source_id"]}: {len(steps)} steps; {result["choices"]}', flush=True)
        summary = {"stage": "diagnostics_complete_needs_review_and_continuation_validation", "cases": len(existing),
                   "candidate_steps": sum(len(r["steps"]) for r in existing),
                   "supported_combined_choices": sum(r["choices"]["combined"] is not None for r in existing),
                   "different_from_nll": sum(r["choices"]["combined"] is not None and r["choices"]["combined"] != r["choices"]["nll"] for r in existing),
                   "diagnostics_sha256": baseline.sha(path), "training_started": False, "explanation_gain_measured": False,
                   "limits": ["Six previously used training-source cases, engineering diagnostics only, not blind evaluation",
                              "Equalities only; conservative final-step omission; no universal semantic segmentation",
                              "Deletion shifts token positions and changes context distribution",
                              "Random deletion matches content token count and source kind, not semantic coherence",
                              "Signed sensitivity/rank scores are not explanation-gain probabilities",
                              "No teacher edits, continuations, calibrated no-supplement threshold or new training yet"]}
        baseline.save(OUT / "summary.json", summary)
        baseline.save(OUT / "status.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        print((OUT / "preflight.json").read_text())
    else:
        run()
