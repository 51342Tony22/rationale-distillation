"""Audit the actual MLX prompt mask and all six frozen supervision conditions."""
from __future__ import annotations

import argparse
import json

from .common import load_config, read_jsonl, resolve_path, write_jsonl
from .refinement import GROUPS
from .train_mlx import supervision_signature, training_iterations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    from transformers import AutoTokenizer
    from mlx_lm.tuner.datasets import CompletionsDataset
    tokenizer=AutoTokenizer.from_pretrained(resolve_path(config["mlx_training"]["model"]),local_files_only=True)
    root=resolve_path(config["refinement"]["root"])
    expected=[r["id"] for r in read_jsonl(root/"bases.jsonl")]
    settings=config["mlx_training"]
    rows=[]
    totals={}
    for group in GROUPS:
        data=read_jsonl(resolve_path(config["data"]["variants_dir"])/(group+".jsonl"))
        if [r["source_id"] for r in data]!=expected:
            raise ValueError("All conditions must have identical ordered question coverage")
        dataset=CompletionsDataset(data,tokenizer,"prompt","completion",True)
        for record in data:
            tokens,offset=dataset.process(record)
            # MLX uses target positions numbered from1 and an inclusive length.
            # With padded batches this also counts the first zero-padding target.
            padded_length=min(1+32*((len(tokens)+31)//32),settings["max_length"])
            target_tokens=min(len(tokens),padded_length-1)-offset+1
            if len(tokens)>settings["max_length"] or target_tokens<=0:
                raise ValueError("Pilot training would truncate or have no assistant supervision")
            rows.append({"variant":group,"source_id":record["source_id"],"full_tokens":len(tokens),
                         "prompt_mask_offset":offset,"supervised_target_tokens":target_tokens,
                         "assistant_content_and_end_tokens":len(tokens)-offset,
                         "supervised_padding_targets":int(padded_length-1>=len(tokens)),
                         "added_hint_tokens":record["added_tokens"]})
        iterations=training_iterations(len(data),settings["batch_size"],settings["num_train_epochs"],settings["gradient_accumulation_steps"])
        totals[group]={"questions":len(data),"unique_questions":len(set(expected)),
                       "target_tokens_one_pass":sum(r["supervised_target_tokens"] for r in rows if r["variant"]==group),
                       "full_tokens_max":max(r["full_tokens"] for r in rows if r["variant"]==group),
                       "changed_questions":sum(r["hint_id"] is not None for r in data),
                       "supervision_sha256":supervision_signature(data),
                       "microsteps":iterations,"optimizer_updates":iterations//settings["gradient_accumulation_steps"],
                       "effective_epochs":iterations*settings["batch_size"]/len(data)}
    write_jsonl(root/"training_token_audit.jsonl",rows)
    report=resolve_path(config["artifacts"]["token_audit"])
    report.parent.mkdir(parents=True,exist_ok=True)
    result={"assistant_only_loss":True,"no_truncation":True,"budget_mode":"same question exposure; not token-matched",
            "groups":totals,"mask_note":"MLX prompt mask excludes user tokens but its inclusive length can supervise one padding target",
            "actual_cumulative_target_tokens":"reported by MLX training logs, not inferred from epochs"}
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
