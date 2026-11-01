#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src"

main_python="$project_dir/.venv/bin/python"
mlx_python="$project_dir/.venv-mlx/bin/python"
config="configs/qwen3_8b_pilot.yaml"

"$main_python" -m fyp_distill.generate_rationales \
  --config "$config" --limit 50 --n 5 \
  --output data/qwen3_8b/teacher/gsm8k_r1_candidates.jsonl
"$main_python" -m fyp_distill.summarize_teacher --config "$config"

"$mlx_python" -m fyp_distill.score_compatibility_mlx --config "$config"
"$main_python" -m fyp_distill.build_variants --config "$config"
"$main_python" -m fyp_distill.audit_data --config "$config"
"$main_python" -m fyp_distill.audit_leakage --config "$config"
"$main_python" -m fyp_distill.audit_tokens --config "$config"

for variant in \
  answer_only single_rationale low_diversity_rationale multi_rationale \
  diverse_multi_rationale compatible_rationale; do
  "$mlx_python" -m fyp_distill.train_mlx \
    --config "$config" --variant "$variant" --seed 7
done

for split in test svamp; do
  for variant in \
    answer_only single_rationale low_diversity_rationale multi_rationale \
    diverse_multi_rationale compatible_rationale; do
    "$mlx_python" -m fyp_distill.evaluate_mlx \
      --config "$config" --variant "$variant" --seed 7 \
      --split "$split" --limit 100 --skip-behavioural
  done
  "$main_python" -m fyp_distill.summarize_mlx --config "$config" --split "$split"
done
