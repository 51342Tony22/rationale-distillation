# FYP Project Context

## Project

- **Title:** CL-1 Reasoning Distillation for Compact Language Models
- **Type:** Empirical research
- **Student:** Zizhen
- **Primary task:** Mathematical reasoning
- **Primary dataset:** GSM8K
- **Optional external evaluation:** SVAMP
- **Default student:** Qwen3-1.7B-Base (LoRA fine-tuning)
- **Default teacher:** DeepSeek-V4-Flash through the SiliconFlow API

## Research question

Does supervising a compact language model with multiple or deliberately diverse
teacher-generated rationales improve its answer accuracy and behavioural robustness
compared with answer-only or single-rationale supervision?

## Hypotheses

- **H1:** Single-rationale supervision improves exact-match accuracy over answer-only
  supervision.
- **H2:** Multi-rationale supervision improves exact-match accuracy and output
  consistency over single-rationale supervision.
- **H3:** With the same number of rationales and the same optimisation budget,
  diversity-selected rationales improve accuracy or robustness over randomly selected
  multi-rationales.

## Main experimental conditions

1. `answer_only`: question -> final answer only.
2. `single_rationale`: question -> one correct teacher rationale and final answer.
3. `multi_rationale`: question -> K correct teacher rationales sampled without an
   explicit diversity objective.
4. `diverse_multi_rationale`: question -> K correct rationales chosen using greedy
   maximum-diversity selection.

The same base student model, train/test split, prompt format, LoRA configuration,
random seed, decoding setup, and number of optimisation steps must be used for all
four main conditions. Every source question is represented K times in every fixed-
exposure condition: answer-only repeats the answer, single-rationale repeats the same
rationale, and the two multi-rationale conditions use K rationale records. This makes
the question order and exposure count identical. `multi_rationale` and
`diverse_multi_rationale` must use the same K.

## Evaluation

### Primary outcome

- Final-answer exact match / accuracy on the untouched GSM8K test split.
- Report a paired 95% bootstrap confidence interval for differences between systems.

### Secondary outcomes

- Answer consistency over repeated stochastic generations.
- Sensitivity to semantically equivalent prompt wording.
- Mean generated rationale length.
- Invalid or unextractable answer rate.
- Error categories on a manually reviewed sample: calculation, reasoning,
  extraction/formatting, and other.

## Scope decisions

- StrategyQA is excluded from the main experiment because it changes both the task
  type and evaluation protocol.
- Consistency-constrained training is an optional extension, not a required main
  condition.
- GSM8K test examples must never be used for teacher-data generation or training.
- A fixed development split is taken from the original GSM8K training split for
  pilot decisions and hyperparameter selection. The official test split remains
  untouched until final evaluation.
- Teacher rationales are retained only if their extracted final answer matches the
  gold answer.

## Reproducibility requirements

- Keep raw teacher outputs immutable.
- Save generation parameters and model names with every record.
- Use a base student rather than an already reasoning-tuned student, so prior
  post-training does not obscure the supervision effect being measured.
- Use deterministic training seeds 7, 17, and 177. Data selection uses seed 7.
- Do not tune hyperparameters independently on the GSM8K test set.
- Run at least three training seeds for the final reported table if compute permits.
- Report both the fixed-update main comparison and descriptive training-data/token
  counts. A data-expanded comparison may be included as a clearly labelled ablation.
