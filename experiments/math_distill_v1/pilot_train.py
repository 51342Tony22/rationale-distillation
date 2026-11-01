"""Audited completion-only BF16 LoRA training and matched pilot evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time

import baseline
import score_sensitivity_v2 as sensitive
from pilot_common import BASE, C, DATA, OUT, ROOT, append, digest_text, gpu, reviewed, rows, sign, sources


def build():
    from transformers import AutoTokenizer
    targets = rows(OUT / "teacher_targets.jsonl")
    ta = reviewed("teacher", targets)
    locations = rows(OUT / "locations.jsonl")
    la = reviewed("location", locations)
    loc = {r["source_id"]: r for r in locations}
    edits = {name: rows(OUT / f"{name}_edits_v4.jsonl") for name in ["difficulty_only", "difficulty_attention"]}
    ea = {name: reviewed(f"{name}_edit_v4", value) for name, value in edits.items()}
    edits = {name: {r["source_id"]: r for r in value} for name, value in edits.items()}
    teacher = {r["source_id"]: r for r in targets}
    tok = AutoTokenizer.from_pretrained(str(ROOT / BASE["student_dir"]), local_files_only=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert baseline.sha(Path(sensitive.__file__)) == manifest["sensitivity_parser_sha256"]
    prepared = {name: [] for name in C["conditions"]}
    excluded = []
    for item in sources():
        sid = item["source_id"]
        reasons = []
        if not ta[sid]["accepted"]:
            reasons.append("teacher: " + ta[sid]["reason"])
        elif sid not in la or not la[sid]["accepted"]:
            reasons.append("location unavailable or rejected")
        else:
            for name in edits:
                if sid not in ea[name] or not ea[name][sid]["accepted"]:
                    reasons.append(name + ": edit missing or rejected")
        if reasons:
            excluded.append({"source_id": sid, "reasons": reasons})
            continue
        original = teacher[sid]["output"]
        variants = {"ordinary_cot": original}
        for name in edits:
            row = edits[name][sid]
            explanation = row["explanation"]
            assert explanation and row["student_tokens"] <= C["edit_max_student_tokens"]
            assert row["teacher_output_sha256"] == teacher[sid]["output_sha256"]
            assert loc[sid]["teacher_output_sha256"] == teacher[sid]["output_sha256"]
            assert row["location_sha256"] == loc[sid]["output_sha256"]
            point = loc[sid]["chosen"]["start"]
            variants[name] = original[:point] + explanation + "\n" + original[point:]
            assert baseline.score(variants[name], item["gold"])["correct"], "Inserted text changed final answer parsing"
        prompt = BASE["student_prompt"].format(question=item["question"])
        group = {}
        for name, completion in variants.items():
            prefix = tok.encode(prompt, add_special_tokens=False)
            target = tok.encode(completion, add_special_tokens=False) + [tok.eos_token_id]
            if len(prefix) + len(target) > C["context_tokens"]:
                reasons.append(name + ": would truncate")
            group[name] = {"source_id": sid, "prompt": prompt, "completion": completion,
                           "prompt_ids": prefix, "target_ids": target, "prompt_tokens": len(prefix),
                           "supervised_tokens": len(target), "total_tokens": len(prefix) + len(target),
                           "completion_sha256": digest_text(completion)}
        if reasons:
            excluded.append({"source_id": sid, "reasons": reasons})
            continue
        for name in group:
            prepared[name].append(group[name])
    count = len(prepared["ordinary_cot"])
    assert count >= 30, "Fewer than 30 jointly eligible targets: report and revise explicitly rather than silently replacing"
    signatures = {}
    for name, records in prepared.items():
        baseline.freeze(OUT / "training_data" / f"{name}.json", records)
        signatures[name] = digest_text(json.dumps([(r["source_id"], r["prompt"], r["completion"]) for r in records], ensure_ascii=False))
    assert len(set(signatures.values())) == len(signatures), "Identical supervision is not an independent treatment"
    report = {"eligible": count, "excluded": excluded, "source_ids": [r["source_id"] for r in prepared["ordinary_cot"]],
              "token_audit": {name: {"rows": len(records), "supervised_tokens_per_epoch": sum(r["supervised_tokens"] for r in records),
                            "max_total_tokens": max(r["total_tokens"] for r in records), "supervision_sha256": signatures[name]}
                              for name, records in prepared.items()},
              "training": {"epochs": C["epochs"], "microsteps_per_condition": count * C["epochs"],
                           "updates_per_condition": math.ceil(count / C["gradient_accumulation"]) * C["epochs"],
                           "loss": C["loss"], "no_truncation": True, "padding_targets": 0},
              "edit_tokens": {name: [edits[name][sid]["student_tokens"] for sid in [r["source_id"] for r in prepared["ordinary_cot"]]] for name in edits},
              "limitation": "Equal question exposure, not equal supervised tokens; attention relevance and length remain possible confounders."}
    eligible_ids = report["source_ids"]
    gaps = [abs(edits["difficulty_only"][sid]["student_tokens"] - edits["difficulty_attention"][sid]["student_tokens"]) for sid in eligible_ids]
    report["paired_edit_lengths"] = {"absolute_student_token_differences": gaps,
                                     "max_difference": max(gaps), "pairs_over_16_tokens": sum(x > 16 for x in gaps),
                                     "identical_explanations": sum(edits["difficulty_only"][sid]["explanation"] == edits["difficulty_attention"][sid]["explanation"] for sid in eligible_ids)}
    audit_inputs = ["teacher_targets.jsonl", "teacher_reviews.jsonl", "locations.jsonl", "location_reviews.jsonl",
                    "difficulty_only_edits_v4.jsonl", "difficulty_attention_edits_v4.jsonl",
                    "difficulty_only_edit_v4_reviews.jsonl", "difficulty_attention_edit_v4_reviews.jsonl",
                    "locations_signature.json", "edits_v4_signature.json"]
    report["material_sha256"] = {name: baseline.sha(OUT / name) for name in audit_inputs}
    baseline.freeze(OUT / "training_protocol.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def completion_loss(model, inputs, targets, prompt_tokens):
    import mlx.core as mx
    import mlx.nn as nn
    hidden = model.model(inputs)
    logits = model.model.embed_tokens.as_linear(hidden[:, prompt_tokens - 1:])
    assert logits.shape[1] == targets.shape[1]
    return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")


def train(condition):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint
    directory = OUT / "training" / condition
    records_path = OUT / "training_data" / f"{condition}.json"
    records = json.loads(records_path.read_text())
    protocol = json.loads((OUT / "training_protocol.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    with gpu("training_" + condition):
        sign("training_" + condition, [Path(__file__), Path(sensitive.__file__), OUT / "training_protocol.json", records_path],
             {"base_revision": BASE["student_revision"], "precision": "bfloat16", "adapter_initialization": "fresh", "grad_checkpoint": True})
        done = directory / "metadata.json"
        if done.exists():
            metadata = json.loads(done.read_text())
            assert metadata["completed"] and baseline.sha(directory / "adapters.safetensors") == metadata["adapter_sha256"]
            print(condition + ": completed signed adapter retained", flush=True)
            return
        assert not (directory / "progress.json").exists(), "Incomplete training retained; inspect checkpoint and optimizer before choosing a safe recovery, never overwrite"
        baseline.verify_student(BASE)
        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)
        mx.random.seed(C["seed"])
        model, tok = load(str(ROOT / BASE["student_dir"]), tokenizer_config={"local_files_only": True})
        assert str(model.model.embed_tokens.weight.dtype) == "mlx.core.bfloat16"
        model.freeze()
        adapter_config = {"fine_tune_type": "lora", "num_layers": C["lora_layers"],
                          "lora_parameters": {"rank": C["lora_rank"], "scale": C["lora_scale"],
                                              "dropout": C["lora_dropout"], "keys": C["lora_keys"]}}
        baseline.freeze(directory / "adapter_config.json", adapter_config)
        linear_to_lora_layers(model, C["lora_layers"], adapter_config["lora_parameters"])
        trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        assert 0 < trainable < 10_000_000
        # Identical loop and loss across conditions; no chat template or padding.
        grad_checkpoint(model.layers[0])
        optimizer = optim.AdamW(learning_rate=C["learning_rate"], weight_decay=C["weight_decay"])
        loss_and_grad = nn.value_and_grad(model, completion_loss)
        model.train()
        rng = random.Random(C["seed"])
        order, micro, updates, trained_tokens = list(range(len(records))), 0, 0, 0
        start = time.perf_counter()
        losses = []
        for epoch in range(C["epochs"]):
            rng.shuffle(order)
            accum, accumulated = None, 0
            for index, sample in enumerate(order):
                row = records[sample]
                full = row["prompt_ids"] + row["target_ids"]
                assert full[-1] == tok.eos_token_id and len(full) <= C["context_tokens"]
                inputs = mx.array([full[:-1]])
                targets = mx.array([row["target_ids"]])
                loss, grads = loss_and_grad(model, inputs, targets, row["prompt_tokens"])
                grads = tree_map(lambda g: g.astype(mx.float32), grads)
                accum = grads if accum is None else tree_map(lambda x, y: x + y, accum, grads)
                accumulated += 1
                micro += 1
                trained_tokens += len(row["target_ids"])
                if accumulated == C["gradient_accumulation"] or index == len(order) - 1:
                    averaged = tree_map(lambda g: g / accumulated, accum)
                    optimizer.update(model, averaged)
                    mx.eval(model.parameters(), optimizer.state, loss)
                    accum, accumulated = None, 0
                    updates += 1
                else:
                    mx.eval(loss, accum)
                value = float(loss.item())
                assert math.isfinite(value), "Nonfinite training loss"
                losses.append(value)
                if micro == 1 or micro % 5 == 0:
                    progress = {"condition": condition, "microsteps": micro, "updates": updates,
                                "total_microsteps": len(records) * C["epochs"], "epoch": epoch + 1,
                                "recent_loss": sum(losses[-5:]) / len(losses[-5:]),
                                "seconds": time.perf_counter() - start, "supervised_tokens_seen": trained_tokens,
                                "peak_mlx_memory_bytes": mx.get_peak_memory(), "trainable_parameters": trainable}
                    baseline.save(directory / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                del grads, loss, inputs, targets
                mx.clear_cache()
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(directory / f"epoch_{epoch + 1}_adapters.safetensors"), weights)
        mx.save_safetensors(str(directory / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        metadata = {"completed": True, "actual_new_training": True, "base_revision": BASE["student_revision"],
                    "precision": "bfloat16", "condition": condition, "seed": C["seed"], "rows": len(records),
                    "epochs": C["epochs"], "microsteps": micro, "updates": updates, "trainable_parameters": trainable,
                    "supervised_tokens_seen": trained_tokens, "elapsed_seconds": time.perf_counter() - start,
                    "peak_mlx_memory_bytes": mx.get_peak_memory(), "loss": C["loss"], "padding_targets": 0,
                    "adapter_sha256": baseline.sha(directory / "adapters.safetensors"), "data_sha256": baseline.sha(records_path),
                    "training_signature_sha256": baseline.sha(OUT / f"training_{condition}_signature.json")}
        assert micro == protocol["training"]["microsteps_per_condition"] and updates == protocol["training"]["updates_per_condition"]
        baseline.save(done, metadata)
        print(condition + ": training complete", flush=True)


def evaluate(condition):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    directory = OUT / "evaluation" / condition
    items = sources("dev")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.jsonl"
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert baseline.sha(Path(sensitive.__file__)) == manifest["sensitivity_parser_sha256"]
    adapter = OUT / "training" / condition / "adapters.safetensors"
    metadata = json.loads((adapter.parent / "metadata.json").read_text())
    assert metadata["completed"] and baseline.sha(adapter) == metadata["adapter_sha256"]
    with gpu("evaluation_" + condition):
        sign("evaluation_" + condition, [Path(__file__), Path(sensitive.__file__), DATA / "dev.json", adapter,
             ROOT / BASE["output_dir"] / "dev_baseline/student_base/signature.json"],
             {"prompt": BASE["student_prompt"], "max_new_tokens": C["eval_max_new_tokens"], "greedy": True})
        old = rows(path)
        assert len(old) <= len(items)
        for row, item in zip(old, items):
            assert row["source_id"] == item["source_id"] and row["question_sha256"] == item["question_sha256"]
            assert digest_text(row["output"]) == row["output_sha256"]
            assert all(row[k] == v for k, v in baseline.score(row["scoring_text"], item["gold"]).items())
        if len(old) < len(items):
            mx.set_memory_limit(14 * 1024**3)
            mx.set_cache_limit(512 * 1024**2)
            mx.random.seed(C["seed"])
            model, tok = load(str(ROOT / BASE["student_dir"]), adapter_path=str(adapter.parent), tokenizer_config={"local_files_only": True})
            model.eval()
            for item in items[len(old):]:
                prompt = tok.encode(BASE["student_prompt"].format(question=item["question"]), add_special_tokens=False)
                assert len(prompt) + C["eval_max_new_tokens"] <= C["context_tokens"]
                text, boundary, last = "", None, None
                tick = time.perf_counter()
                for part in stream_generate(model, tok, prompt, max_tokens=C["eval_max_new_tokens"], sampler=make_sampler(temp=0.0)):
                    text += part.text
                    last = part
                    boundary = re.search(r"(?:^|\n)\s*(?:###\s*)?Question:", text)
                    if boundary:
                        break
                assert last is not None
                scored = text[:boundary.start()] if boundary else text
                value, rule = sensitive.parse(scored)
                row = {"source_id": item["source_id"], "question_sha256": item["question_sha256"], "gold": item["gold"],
                       "output": text, "scoring_text": scored, "output_sha256": digest_text(text),
                       "generation_tokens": last.generation_tokens, "seconds": time.perf_counter() - tick,
                       "finish_reason": "next_question" if boundary else last.finish_reason,
                       **baseline.score(scored, item["gold"]), "v2_parsed_answer": str(value) if value is not None else None,
                       "v2_correct": value is not None and value == baseline.number(item["gold"]), "v2_invalid": value is None,
                       "v2_rule": rule}
                append(path, row)
                old.append(row)
                baseline.save(directory / "progress.json", {"completed": len(old), "total": len(items), "v1_correct": sum(r["correct"] for r in old),
                              "v2_correct": sum(r["v2_correct"] for r in old), "stage": "evaluating"})
                if len(old) % 10 == 0:
                    print(f'{condition}: evaluated {len(old)}/{len(items)}', flush=True)
        baseline.save(directory / "summary.json", {"completed": len(old), "v1_correct": sum(r["correct"] for r in old),
                      "v2_correct": sum(r["v2_correct"] for r in old), "v1_invalid": sum(r["invalid"] for r in old),
                      "v2_invalid": sum(r["v2_invalid"] for r in old), "length_stops": sum(r["finish_reason"] == "length" for r in old),
                      "predictions_sha256": baseline.sha(path), "full_math_review": "pending"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "train", "evaluate"])
    parser.add_argument("--condition", choices=C["conditions"])
    args = parser.parse_args()
    if args.stage == "build":
        build()
    else:
        assert args.condition
        (train if args.stage == "train" else evaluate)(args.condition)
