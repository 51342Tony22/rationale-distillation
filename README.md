# Rationale Distillation for Mathematical Reasoning

Ongoing undergraduate research on transferring teacher reasoning into compact language models. This repository brings together the packaged `fyp_distill` implementation and subsequent versioned experiments, while retaining their separate protocols and reports.

## Research stages

| Location | Scope |
|---|---|
| `src/fyp_distill/` and `configs/` | Answer-only, single-rationale, multiple-rationale, and diverse-rationale supervision; MLX and Transformers paths; student-guided local refinement. |
| `reports_mlx/` | Historical compact-student multi-seed pilot and result summaries. |
| `reports_refinement_math_trainpilot21_v3/` | Six-condition training pilot evaluated on a shared set of 70 questions: 420 question-condition evaluations, not 420 independent experiments. |
| `experiments/local_math_teacher/` | Local quantized mathematical teacher feasibility and token-logit extraction. |
| `experiments/math_distill_v1/` | Later mathematical distillation, scoring audits, pilot training, and selector diagnostics. |
| `experiments/tgrd_v1/` | Later experimental workflows, rationale variants, selector/material validation, and versioned protocols. |
| `research_notes/` | Historical status and protocol notes, with dated updates. These are records, not current execution instructions. |

This is an ongoing study. Negative results, parsing failures, output truncation, and pilot-scale limitations are retained. Proposed runs in research notes must not be interpreted as completed experiments. The older local-refinement pilot did not establish a consistent advantage from filtered local rationales.

## Install and run the packaged offline tests

Use Python 3.11 or 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

Transformers training requires the `train` optional dependencies. Apple Silicon runs use a separate MLX environment. See [the original implementation guide](README.historical.md) and the versioned protocol files for details. API-based teacher generation may incur charges and is not part of the test command.

## Working with later experiments

Later scripts live under `experiments/` and derive `data/`, `models/`, and `output/` paths from this repository root. Some historical configs and runbooks refer to original absolute local paths or archived run artifacts. Review those paths and reconstruct the required model/data/protocol artifacts before running them. Copying source code does not restore trained adapters or execution authorization state.

Local GPU/MLX and artifact-dependent checks are separate from the packaged test suite. `VALIDATION.md` records which checks were actually run for this snapshot. This upload does not launch training, teacher requests, or background supervisors.

## Data and artifacts

Use the preparation modules to obtain public datasets such as GSM8K, SVAMP, and MATH under their upstream terms. Raw datasets, downloaded teacher/student weights, adapters, detailed request/authorization logs, and large prediction dumps are excluded. Selected aggregate reports and method figures are included. Historical document links to excluded local artifacts may require reconstructing those artifacts.

`SOURCE_SNAPSHOT.json` records selected files from the two local research directories as of 2026-11-01. This is a source-and-report snapshot, not a newly completed main study.
