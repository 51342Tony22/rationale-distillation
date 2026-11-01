"""Resume-safe student prefix rollouts. Does not call a teacher API or train."""
from __future__ import annotations

import argparse
import json
import time

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import (
    TEMPLATE, correct, digest, hardest, reviewed_hint, screen_index, stable_seed, unique_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "verify"))
    parser.add_argument("--config", default="configs/refinement_pilot.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hints", help="Reviewed hint snapshot, allowing verification while the teacher queue continues")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Verify the independent unassisted arm while hints are being generated")
    parser.add_argument("--max-new-rollouts", type=int, help="Bound a timing run; resumption preserves the full sampling protocol")
    args = parser.parse_args()
    if args.baseline_only and args.stage != "verify":
        parser.error("--baseline-only is available only for verify")
    if args.max_new_rollouts is not None and args.max_new_rollouts < 1:
        parser.error("--max-new-rollouts must be positive")
    config = load_config(args.config)
    protocol = config.get("evaluation", {}).get("answer_protocol", "first_final_line_v2")
    if protocol == "first_answer_v3":
        from .answer_protocol_v3 import PROTOCOL_VERSION, completed_answer_end
    elif protocol == "first_final_line_v2":
        from .answer_protocol import PROTOCOL_VERSION, completed_answer_end
    else:
        raise ValueError(f"Unsupported diagnostic answer protocol: {protocol}")
    settings = config["refinement"]
    root = resolve_path(settings["root"])
    bases = read_jsonl(root / "bases.jsonl")
    if args.limit is not None:
        bases = bases[:args.limit]
    if not bases:
        raise SystemExit("No prepared bases")
    output = root / ("screening.jsonl" if args.stage == "screen" else "verification.jsonl")
    existing = unique_index(read_jsonl(output))
    new_rollouts = 0
    model_path = resolve_path(config["mlx_training"]["model"])
    # Bind resumption to local weights, tokenizer, decoding settings and source.
    fingerprint = [
        [str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(model_path.glob("*")) if p.is_file()
    ]
    signature = digest({"settings": {k: v for k, v in settings.items() if k != "budget_ledger"}, "stage": args.stage, "seed": config["seed"],
                        "model": str(model_path), "files": fingerprint, "template": TEMPLATE,
                        "answer_protocol": PROTOCOL_VERSION})
    if protocol == "first_answer_v3":
        from pathlib import Path
        signature = digest({"run": signature, "protocol_source":
                            Path(__file__).with_name("answer_protocol_v3.py").read_text()})
    for base in bases:
        old = existing.get(base["id"])
        if old and (old["source_hash"] != base["source_hash"] or old["run_signature"] != signature):
            raise SystemExit("Stale rollouts; use a new run directory, do not mix experiments")

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=settings["temperature"], top_p=settings["top_p"])
    hints = read_jsonl(resolve_path(args.hints) if args.hints else root / "hints.jsonl") if args.stage == "verify" else []
    screens = screen_index(bases, read_jsonl(root / "screening.jsonl"), settings["screening_samples"]) if args.stage == "verify" else {}

    def rollout(base, prefix, seed):
        nonlocal new_rollouts
        new_rollouts += 1
        messages = [{"role": "user", "content": TEMPLATE.format(question=base["question"])},
                    {"role": "assistant", "content": prefix}]
        # Continue the assistant prefix, rather than start a second assistant turn.
        tokens = tokenizer.apply_chat_template(
            messages, continue_final_message=True, add_generation_prompt=False, return_dict=False,
        )
        if len(tokens) + settings["max_new_tokens"] > config["student"]["max_length"]:
            raise ValueError("Prefix plus generation budget exceeds configured context")
        mx.random.seed(seed)
        start = time.monotonic()
        parts, last, stopped_at_answer = [], None, False
        for last in stream_generate(model, tokenizer, tokens,
                                    max_tokens=settings["max_new_tokens"], sampler=sampler):
            parts.append(last.text)
            end = completed_answer_end("".join(parts))
            if end is not None:
                parts = ["".join(parts)[:end]]
                stopped_at_answer = True
                break
        text = "".join(parts)
        result = {"seed": seed, "text": text,
                "correct": correct(text, base["gold_answer"], base["dataset"], protocol),
                "prompt_tokens": len(tokens), "generation_tokens": last.generation_tokens if last else 0,
                "finish_reason": "final_answer" if stopped_at_answer else last.finish_reason if last else "empty",
                "answer_protocol": PROTOCOL_VERSION,
                "elapsed_seconds": time.monotonic() - start}
        print(f"rollout {new_rollouts}: {base['id']}, tokens={result['generation_tokens']}, "
              f"seconds={result['elapsed_seconds']:.1f}, correct={result['correct']}, "
              f"finish={result['finish_reason']}", flush=True)
        return result

    def persist():
        write_jsonl(output, sorted(existing.values(), key=lambda r: r["id"]))

    for base in bases:
        record = existing.setdefault(base["id"], {"id": base["id"], "source_hash": base["source_hash"],
                                                 "run_signature": signature})
        if args.stage == "screen":
            for offset in base["boundaries"]:
                trials = record.setdefault("boundaries", {}).setdefault(str(offset), [])
                for index in range(len(trials), settings["screening_samples"]):
                    trials.append(rollout(base, base["original"][:offset],
                                          stable_seed(config["seed"], base["id"], "screen", offset, index)))
                    persist()
                    if args.max_new_rollouts is not None and new_rollouts >= args.max_new_rollouts:
                        print("Timing-run limit reached; incomplete sampling is resumable, not a completed diagnostic.", flush=True)
                        return
        else:
            local = sorted([h for h in hints if h["source_id"] == base["id"] and h["kind"] == "local"],
                           key=lambda h: h["id"])
            if not args.baseline_only and len(local) != settings["local_candidates"]:
                raise ValueError("Incomplete local candidate pool")
            offset = hardest(base, screens[base["id"]])
            if args.baseline_only:
                local = []
            for hint in local:
                reviewed_hint(hint, base, lambda s: len(tokenizer.encode(s, add_special_tokens=False)),
                              settings["max_hint_tokens"])
                if hint["offset"] != offset:
                    raise ValueError("Wrong local hint location")
            if not args.baseline_only and record.get("hint_hash", digest(local)) != digest(local):
                raise ValueError("Hints changed after verification began")
            if not args.baseline_only:
                record["hint_hash"] = digest(local)
            prefix = base["original"][:offset]
            for hint in [None, *local]:
                trials = record.setdefault("baseline", []) if hint is None else record.setdefault("hints", {}).setdefault(hint["id"], [])
                conditioned = prefix if hint is None else prefix + hint["hint"] + "\n\n"
                for index in range(len(trials), settings["verification_samples"]):
                    trials.append(rollout(base, conditioned,
                                          stable_seed(config["seed"], base["id"], "verify", index)))
                    persist()
                    if args.max_new_rollouts is not None and new_rollouts >= args.max_new_rollouts:
                        print("Timing-run limit reached; incomplete sampling is resumable, not a completed diagnostic.", flush=True)
                        return
        label = "verify_baseline_only" if args.baseline_only else args.stage
        print(f"{label}: {base['id']} complete", flush=True)
    print(json.dumps({"output": str(output), "questions_processed": len(bases), "teacher_api_calls": 0}))


if __name__ == "__main__":
    main()
