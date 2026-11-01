"""Functional sparse student attention probe, never an effectiveness benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import baseline

ROOT = baseline.ROOT
HERE = Path(__file__).absolute().parent

# Audited handcrafted solutions, separate from the development/test evaluation.
CASES = [
    {"id": "inventory_probe", "question": "A shop has 18 apples. It sells 7 apples and then receives 5 more. How many apples does it have now?",
     "steps": ["After selling 7 apples, the shop has 18 - 7 = 11 apples.",
               "After receiving 5 more, the shop has 11 + 5 = 16 apples."], "answer": "16"},
    {"id": "rectangle_probe", "question": "A rectangle has perimeter 50 cm. Its length is 5 cm greater than its width. What is its area in square centimetres?",
     "steps": ["Let W be the width. The length is W + 5.",
               "The perimeter gives 2(W + 5) + 2W = 50, so 4W + 10 = 50.",
               "Thus W = 10 and the length is 15.",
               "The area is 15 times 10 = 150 square centimetres."], "answer": "150"}
]


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    config = json.loads(baseline.CONFIG.read_text())
    out = ROOT / config["output_dir"] / "attention_functional_probe"
    out.mkdir(parents=True, exist_ok=True)
    baseline.freeze(out / "signature.json", {"config_sha256": baseline.sha(baseline.CONFIG),
                    "script_sha256": baseline.sha(Path(__file__)), "cases": CASES,
                    "layers": "last four", "aggregation": "mean over heads then layers",
                    "stage": "functional_only"})

    class Probe(nn.Module):
        def __init__(self, original, positions):
            super().__init__()
            self.original = original
            self.positions = positions
            self.captured = None

        def __call__(self, x, mask=None, cache=None):
            assert cache is None, "This probe only supports teacher-forced no-cache calls"
            # The returned output is exactly the existing fused forward output.
            result = self.original(x, mask, cache)
            a = self.original
            batch, length, _ = x.shape
            q = a.q_proj(x).reshape(batch, length, a.n_heads, -1).transpose(0, 2, 1, 3)
            k = a.k_proj(x).reshape(batch, length, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = a.rope(q), a.rope(k)
            q = q[:, :, self.positions, :].astype(mx.float32)
            k = mx.repeat(k, a.n_heads // a.n_kv_heads, axis=1).astype(mx.float32)
            scores = (q @ k.transpose(0, 1, 3, 2)) * a.scale
            causal = mx.arange(length)[None, :] <= mx.array(self.positions)[:, None]
            self.captured = mx.softmax(mx.where(causal[None, None], scores, -mx.inf), axis=-1)
            return result

    records = []
    with baseline.gpu_lock(out):
        baseline.verify_student(config)
        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(1024**3)
        mx.random.seed(config["seed"])
        model, tokenizer = load(str(ROOT / config["student_dir"]),
                                tokenizer_config={"local_files_only": True, "trust_remote_code": False})
        model.eval()
        assert str(model.model.embed_tokens.weight.dtype) == "mlx.core.bfloat16"
        for case in CASES:
            start = time.perf_counter()
            prefix = config["student_prompt"].format(question=case["question"])
            answer = "\n".join(case["steps"]) + "\n" + r"\boxed{" + case["answer"] + "}"
            prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
            enc = tokenizer._tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
            target = enc["input_ids"]
            offsets = enc["offset_mapping"]
            ids = prompt_ids + target
            assert len(ids) <= config["context_tokens"]
            x = mx.array([ids[:-1]])
            tick = time.perf_counter()
            hidden = model.model(x)
            # Apply the tied head only to target predictors, not to question positions.
            logits = model.model.embed_tokens.as_linear(hidden[:, len(prompt_ids) - 1:])
            logits = logits.astype(mx.float32)
            mx.eval(logits)
            original_logits = np.array(logits)
            target_mx = mx.array(target)[None, :, None]
            nll = -(mx.take_along_axis(logits, target_mx, axis=-1).squeeze(-1)
                    - mx.logsumexp(logits, axis=-1))
            losses = np.array(nll)[0]
            forward_seconds = time.perf_counter() - tick
            spans, char_start = [], 0
            # Each problem contributes its own number of complete textual steps.
            # Exclude the final reasoning step and boxed answer from location candidates.
            for index, line in enumerate(case["steps"]):
                positions = [i for i, (a, b) in enumerate(offsets) if a < char_start + len(line) and b > char_start]
                if index < len(case["steps"]) - 1:
                    worst = max(positions, key=lambda i: float(losses[i]))
                    spans.append({"step_index": index, "text": line, "target_token_indexes": positions,
                                  "mean_nll": float(np.mean(losses[positions])),
                                  "worst_target_token_index": worst})
                char_start += len(line) + 1
            chosen = max(spans, key=lambda item: item["mean_nll"])
            # Explicit off-by-one: prediction of target token t uses input query P+t-1.
            target_index = chosen["worst_target_token_index"]
            query = len(prompt_ids) + target_index - 1
            wrappers = []
            tick = time.perf_counter()
            try:
                for layer_index in range(len(model.layers) - 4, len(model.layers)):
                    layer = model.layers[layer_index]
                    wrapper = Probe(layer.self_attn, [query])
                    wrappers.append((layer, layer.self_attn, wrapper))
                    layer.self_attn = wrapper
                modified_hidden = model.model(x)
                modified_logits = model.model.embed_tokens.as_linear(modified_hidden[:, len(prompt_ids) - 1:]).astype(mx.float32)
                captures = [wrapper.captured for _, _, wrapper in wrappers]
                mx.eval(modified_logits, *captures)
                delta = float(np.max(np.abs(np.array(modified_logits) - original_logits)))
                assert delta == 0.0, f"Attention observation changed model logits: {delta}"
                errors, future = [], []
                for weights in captures:
                    errors.append(float(mx.max(mx.abs(mx.sum(weights, axis=-1) - 1)).item()))
                    future.append(float(mx.sum(weights[..., query + 1:]).item()))
                assert max(errors) < 1e-5 and max(future) == 0
                mean_attention = np.mean(np.stack([np.array(w)[0, :, 0, :] for w in captures]), axis=(0, 1))
                # Question-only evidence candidates: do not treat answer text as a premise.
                q_start = len(tokenizer.encode("Question: ", add_special_tokens=False))
                q_end = len(tokenizer.encode("Question: " + case["question"], add_special_tokens=False))
                candidates = list(range(q_start, min(q_end, query + 1)))
                best = sorted(candidates, key=lambda i: float(mean_attention[i]), reverse=True)[:8]
                evidence = [{"input_position": i, "token_id": ids[i],
                             "token": tokenizer.decode([ids[i]]), "mean_attention": float(mean_attention[i])}
                            for i in best]
                np.savez_compressed(out / f'{case["id"]}_trace.npz',
                                    nll=losses, target_ids=np.array(target),
                                    input_ids=np.array(ids[:-1]), mean_attention=mean_attention,
                                    predictor_position=np.array(query))
            finally:
                for layer, original, _ in wrappers:
                    layer.self_attn = original
            records.append({"id": case["id"], "question": case["question"], "solution": answer,
                            "solution_origin": "audited handcrafted engineering probe",
                            "input_tokens": len(ids) - 1, "candidate_steps": spans, "chosen_step": chosen["step_index"],
                            "target_token_index": target_index, "target_token": tokenizer.decode([target[target_index]]),
                            "predictor_query_position": query, "question_token_evidence": evidence,
                            "max_output_logits_difference": delta, "max_attention_sum_error": max(errors),
                            "future_attention_mass": max(future), "baseline_forward_seconds": forward_seconds,
                            "instrumented_forward_seconds": time.perf_counter() - tick,
                            "total_seconds": time.perf_counter() - start,
                            "peak_mlx_memory_bytes": mx.get_peak_memory()})
            print(f'{case["id"]}: sparse attention captured; unchanged logits and causal mask checked.', flush=True)
        baseline.save(out / "result.json", {"status": "functional_checks_passed", "records": records,
                      "limit": "Not evidence that attention identifies causal premises or improves distillation. Token evidence needs semantic grouping; hardest step is a heuristic."})


if __name__ == "__main__":
    main()
