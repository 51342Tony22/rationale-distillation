"""One unassisted rollout per frozen calibration question; never selects hints."""
from __future__ import annotations

import argparse
import json

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import TEMPLATE, correct, digest, stable_seed, unique_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["evaluation"]["answer_protocol"] != "first_answer_v3":
        raise ValueError("This calibration probe requires v3")
    from .answer_protocol_v3 import PROTOCOL_VERSION, first_final_answer, generate_single_solution
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    root = resolve_path(config["refinement"]["root"])
    selected = read_jsonl(root / "selected_questions.jsonl")
    model_path = resolve_path(config["mlx_training"]["model"])
    from pathlib import Path
    protocol_source = Path(__file__).with_name("answer_protocol_v3.py").read_text()
    signature = digest({"config": config, "questions": selected, "protocol": PROTOCOL_VERSION,
                        "protocol_source": digest(protocol_source),
                        "model_files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                                        for p in sorted(model_path.glob("*")) if p.is_file()]})
    output = root / "unassisted_probe_v3_frozen.jsonl"
    existing = unique_index(read_jsonl(output))
    if any(r["signature"] != signature for r in existing.values()):
        raise ValueError("Stale probe output; use a new root")
    model, tokenizer = load(str(model_path))
    settings = config["refinement"]
    sampler = make_sampler(temp=settings["temperature"], top_p=settings["top_p"])
    for row in selected:
        if row["id"] in existing:
            continue
        seed = stable_seed(config["seed"], "unassisted-probe-v3", row["id"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": TEMPLATE.format(question=row["question"])}],
            add_generation_prompt=True, return_dict=False)
        if len(prompt) + settings["max_new_tokens"] > config["student"]["max_length"]:
            raise ValueError("Probe exceeds configured context")
        mx.random.seed(seed)
        result = generate_single_solution(model, tokenizer, prompt, settings["max_new_tokens"], sampler)
        existing[row["id"]] = {"id": row["id"], "seed": seed, "signature": signature,
                               "subject": row["subject"], "level": row["level"],
                               "answer": first_final_answer(result["text"]),
                               "correct": correct(result["text"], row["gold_answer"], "math", PROTOCOL_VERSION),
                               **result}
        write_jsonl(output, existing.values())
        print(json.dumps({k: existing[row["id"]][k] for k in
                          ("id", "answer", "correct", "generation_tokens", "finish_reason", "elapsed_seconds")}), flush=True)
    print(json.dumps({"questions": len(selected), "teacher_api_calls": 0,
                      "purpose": "format/runtime calibration; one sample is not a success-rate estimate"}))


if __name__ == "__main__":
    main()
