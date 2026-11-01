from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .common import artifact_dir, configured_variants, load_config, read_jsonl, resolve_path, write_jsonl
from .train_mlx import supervision_signature


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible MLX experiment matrix.")
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--split", choices=("dev", "test", "svamp", "math500"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-behavioural", action="store_true")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    representatives = {}
    aliases = {}
    if config.get("pilot", {}).get("deduplicate_identical_supervision", False):
        for variant in configured_variants(config):
            rows = read_jsonl(resolve_path(config["data"]["variants_dir"]) / (variant + ".jsonl"))
            if not rows:
                raise SystemExit("Every pilot condition requires complete supervision")
            signature = supervision_signature(rows)
            representative = representatives.setdefault(signature, variant)
            if representative != variant:
                aliases[variant] = representative
        write_jsonl(resolve_path(config["refinement"]["root"])/"condition_reuse.jsonl",
                    [{"variant":v,"representative":aliases.get(v,v),
                      "identical_supervision":v in aliases,
                      "independent_training":v not in aliases} for v in configured_variants(config)])
    for variant in configured_variants(config):
        for seed in config["seeds"]:
            if args.stage == "train":
                command = [
                    sys.executable,
                    "-m",
                    "fyp_distill.train_mlx",
                    "--config",
                    args.config,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                ]
                if variant in aliases:
                    command.extend(("--reuse-from", aliases[variant]))
            else:
                if variant in aliases:
                    root = artifact_dir(config,"mlx_report_dir","reports_mlx")/args.split
                    representative_dir=root/aliases[variant]/f"seed_{seed}"
                    destination=root/variant/f"seed_{seed}"
                    metrics_path=representative_dir/"metrics.json"
                    if not metrics_path.exists():
                        raise SystemExit("Representative evaluation has not completed")
                    metadata=json.loads(metrics_path.read_text())
                    predictions=read_jsonl(representative_dir/"predictions.jsonl")
                    expected=read_jsonl(config["data"][args.split+"_file"])
                    if args.limit is not None:
                        expected=expected[:args.limit]
                    if [r["id"] for r in predictions] != [r["id"] for r in expected]:
                        raise SystemExit("Representative evaluation question coverage differs")
                    destination.mkdir(parents=True,exist_ok=True)
                    write_jsonl(destination/"predictions.jsonl",predictions)
                    metadata.update(variant=variant,reused_from=aliases[variant],independent_evaluation=False)
                    (destination/"metrics.json").write_text(json.dumps(metadata,indent=2))
                    signature=json.loads((representative_dir/"evaluation_signature.json").read_text())
                    (destination/"evaluation_signature.json").write_text(json.dumps(signature,indent=2))
                    print(f"Reused identical condition evaluation: {variant} <- {aliases[variant]}",flush=True)
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "fyp_distill.evaluate_mlx",
                    "--config",
                    args.config,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--split",
                    args.split,
                ]
                if args.limit is not None:
                    command.extend(("--limit", str(args.limit)))
                if args.skip_behavioural:
                    command.append("--skip-behavioural")
            print(f"Running {args.stage}: {variant}, seed={seed}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
