# Reasoning Distillation for Compact Language Models

## Current direction (2026-10-13)

The new study is **student-guided rationale refinement**, documented in
[`EXPERIMENT_PROTOCOL_REFINEMENT_ZH.md`](EXPERIMENT_PROTOCOL_REFINEMENT_ZH.md).
Historical quantity/diversity results below remain intact. The old R1 generation
budgets are disabled; the remaining combined ¥42 allocation now funds refinement.

- `configs/refinement_pilot.yaml`: existing GSM8K/R1 data for engineering checks.
- `configs/refinement_math.yaml`: MATH main-study settings; teacher data and a
  final token-budget-controlled training protocol are still pending pilot results.
- `data/refinement_budget/teacher_ledger.jsonl`: shared estimated-cost ledger.
- `data/refinement_math/processed/`: frozen train/dev/pilot problem splits.

Run commands from this repository with `PYTHONPATH=src`. Pipeline stages:

```bash
PYTHONPATH=src .venv/bin/python -m fyp_distill.refinement prepare --config configs/refinement_pilot.yaml
PYTHONPATH=src .venv-mlx/bin/python -m fyp_distill.diagnose_refinement_mlx screen --config configs/refinement_pilot.yaml
PYTHONPATH=src .venv/bin/python -m fyp_distill.refinement requests --config configs/refinement_pilot.yaml
# Paid requests; capped by the shared ledger, default at most four new calls.
PYTHONPATH=src .venv/bin/python -m fyp_distill.request_refinement_hints --config configs/refinement_pilot.yaml
# Review source steps/prefixes in bases.jsonl and explanations in hints.jsonl.
PYTHONPATH=src .venv-mlx/bin/python -m fyp_distill.diagnose_refinement_mlx verify --config configs/refinement_pilot.yaml
PYTHONPATH=src .venv/bin/python -m fyp_distill.refinement build --config configs/refinement_pilot.yaml
```

The first saved teacher request queue is a **one-question smoke**. Do not expand
or overwrite frozen artifacts in place: choose a new run root for a larger pilot,
while preserving the shared budget ledger. Pending review flags are deliberate;
never mark them true solely because the final numerical answer matches.

The current training wrapper matches question exposure for the pilot and rejects
silent truncation. It does **not** implement exact supervised-token matching.
Do not label its results as token-budget matched. AMC/AIME preparation is pending.

This repository implements the empirical study described in
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). The current 8B main protocol is in
[`EXPERIMENT_PROTOCOL_QWEN3_8B_ZH.md`](EXPERIMENT_PROTOCOL_QWEN3_8B_ZH.md). It builds controlled training
conditions from GSM8K, fine-tunes the same compact student model with LoRA, and
evaluates answer accuracy, consistency, prompt sensitivity, and rationale length.

## Experiment map

```text
GSM8K train
  -> teacher generates N rationales/question
  -> answer-based validity filtering
  -> random or diversity-based selection
  -> four question-exposure-matched JSONL training conditions
  -> four LoRA adapters of the same student
  -> one shared evaluation protocol on GSM8K test
  -> paired bootstrap comparisons and CSV/Markdown tables
```

## Environment

Use Python 3.11 or 3.12. The pilot uses MLX QLoRA with
`mlx-community/Qwen3-0.6B-Base-4bit` on Apple Silicon. The separate Transformers
environment remains available for a later CUDA-based 1.7B experiment.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,dev]'
```

Create a separate MLX environment for the local Apple Silicon experiment:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install mlx-lm
python -m pip install -e . --no-deps
```

Copy `.env.example` to `.env` and add an OpenAI-compatible teacher API key. The
default variables also work with SiliconFlow. Never commit `.env`.

Check readiness at any point with:

```bash
python -m fyp_distill.preflight
```

## Quick pipeline

Start with a small pilot before paying to generate the full training set:

```bash
python -m fyp_distill.prepare_data --train-limit 200 --dev-limit 50 --test-limit 100
python -m fyp_distill.prepare_svamp
python -m fyp_distill.generate_rationales --input data/processed/gsm8k_train.jsonl --n 5
python -m fyp_distill.build_variants --k 3
python -m fyp_distill.audit_data
```

Train all four fixed-update/fixed-question-exposure conditions:

```bash
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.train --variant "$variant"
done
```

Evaluate each adapter and build a comparison report:

```bash
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.evaluate --variant "$variant"
done
python -m fyp_distill.summarize
```

For the MLX three-seed matrix used in the pilot:

```bash
source .venv-mlx/bin/activate
PYTHONPATH=src python -m fyp_distill.run_mlx_matrix --stage train
PYTHONPATH=src python -m fyp_distill.run_mlx_matrix --stage evaluate --split test --limit 100 --skip-behavioural
PYTHONPATH=src python -m fyp_distill.run_mlx_matrix --stage evaluate --split svamp --limit 100 --skip-behavioural
source .venv/bin/activate
PYTHONPATH=src python -m fyp_distill.summarize_mlx --split test
PYTHONPATH=src python -m fyp_distill.summarize_mlx --split svamp
```

The expanded 196-question, three-seed experiment is documented in
[`reports_mlx/FINAL_REPORT_ZH.md`](reports_mlx/FINAL_REPORT_ZH.md). The earlier
58-question pilot remains in [`reports_mlx/PILOT_REPORT.md`](reports_mlx/PILOT_REPORT.md)
for comparison.

The historical 0.6B pilot uses `configs/experiment.yaml`. The current Qwen3-8B
main study uses `configs/qwen3_8b_main.yaml`, six conditions, full GSM8K/SVAMP/
MATH-500 evaluation, mathematical-equivalence grading for MATH-500, and an
explicit leakage audit.

All commands accept `--help`.
The tiny fixture path used by automated tests does not call an API or download a
model.

## Recommended execution stages

1. **Smoke test:** local fixture only.
2. **Pilot:** 200 train / 100 test questions, one training seed.
3. **Main run:** choose the sample size after measuring pilot runtime and teacher
   acceptance rate; freeze the protocol before running it.
4. **Final evidence:** three seeds per condition, the full untouched GSM8K test set,
   and the 1,000-example SVAMP challenge set as training-free external evaluation.
   Repeated-generation behavioural metrics use one fixed 200-example subset to keep
   inference cost bounded; primary exact match still uses the full test split.

Do not interpret a single pilot run as the dissertation result.
