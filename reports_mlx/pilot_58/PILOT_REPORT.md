# FYP Pilot Experiment Report

## Research question

Does chain-of-thought rationale supervision improve a compact language model, and do multiple or explicitly diverse rationales provide more useful supervision than a single rationale?

## Experimental setup

- Teacher: `deepseek-ai/DeepSeek-V4-Flash` through SiliconFlow.
- Student: `mlx-community/Qwen3-0.6B-Base-4bit` with LoRA (`r=16`) on Apple MLX.
- Training seeds: 7, 17, and 177.
- Training source: GSM8K.
- Evaluation: 100 held-out GSM8K questions and 100 SVAMP questions.
- Primary metric: numeric exact match on the extracted final answer.
- Secondary metrics: seed standard deviation, invalid-answer rate, rationale length, and paired bootstrap confidence intervals.

The SiliconFlow balance was exhausted after 58 questions had enough valid teacher outputs. This run is therefore a pilot, not the final full-scale experiment. All four conditions use the same 58 questions and 174 training rows. Only teacher answers verified against the gold answer and returned in the public response content were accepted. In total, 284 teacher rationales were accepted; 712 attempted calls failed with HTTP 402 and were excluded.

The four controlled conditions are:

1. `answer_only`: final answer only, repeated three times per question to match row count.
2. `single_rationale`: one correct rationale repeated three times per question.
3. `multi_rationale`: three correct rationales sampled per question.
4. `diverse_multi_rationale`: three correct rationales selected by greedy TF-IDF distance.

Selected rationale diversity was 0.460 for ordinary multi-rationale and 0.545 for diverse multi-rationale.

## GSM8K results

| Condition | Seed 7 | Seed 17 | Seed 177 | Mean | SD | Invalid | Mean rationale words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untrained student | 0.030 | - | - | 0.030 | - | 0.320 | 11.3 |
| Answer only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 |
| Single rationale | 0.260 | 0.240 | 0.200 | 0.233 | 0.031 | 0.010 | 36.0 |
| Multi rationale | 0.260 | 0.330 | 0.270 | **0.287** | 0.038 | 0.000 | 39.7 |
| Diverse multi rationale | 0.280 | 0.310 | 0.260 | 0.283 | 0.025 | 0.000 | 36.2 |

Paired bootstrap differences over 100 questions, after averaging each question across seeds:

| Comparison | Difference | 95% interval |
|---|---:|---:|
| Single minus answer only | +0.233 | [0.177, 0.293] |
| Multi minus single | +0.053 | [0.003, 0.103] |
| Diverse multi minus ordinary multi | -0.003 | [-0.057, 0.050] |

The same teacher scored 0.980 exact match on these 100 GSM8K questions. Relative to the untrained student's 0.030, the best student mean (ordinary multi-rationale, 0.287) closes about 27.0% of the teacher-student accuracy gap. It does not approach or exceed the teacher in this pilot.

## SVAMP transfer results

| Condition | Seed 7 | Seed 17 | Seed 177 | Mean | SD | Invalid | Mean rationale words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untrained student | 0.020 | - | - | 0.020 | - | 0.010 | 11.6 |
| Answer only | 0.010 | 0.020 | 0.010 | 0.013 | 0.006 | 0.000 | 1.1 |
| Single rationale | 0.450 | 0.450 | 0.400 | 0.433 | 0.029 | 0.003 | 26.5 |
| Multi rationale | 0.400 | 0.430 | 0.520 | 0.450 | 0.062 | 0.003 | 26.3 |
| Diverse multi rationale | 0.490 | 0.510 | 0.450 | **0.483** | 0.031 | 0.000 | 25.8 |

| Comparison | Difference | 95% interval |
|---|---:|---:|
| Single minus answer only | +0.420 | [0.343, 0.497] |
| Multi minus single | +0.017 | [-0.053, 0.087] |
| Diverse multi minus ordinary multi | +0.033 | [-0.033, 0.100] |

No complete SVAMP teacher baseline was produced before the API balance was exhausted, so no teacher-gap calculation is reported for SVAMP.

## Interpretation

This pilot gives strong evidence that rationale supervision is much more effective than answer-only fine-tuning for this 0.6B Base student. The gain transfers from GSM8K training to SVAMP, so it is not merely memorization of the evaluation dataset.

Ordinary multi-rationale supervision improves over single-rationale supervision on GSM8K by 5.3 percentage points, with a bootstrap interval narrowly above zero. On SVAMP the gain is smaller and the interval crosses zero. Diversity selection successfully increased the measured textual diversity, but it did not reliably improve accuracy over ordinary multi-rationale sampling: the intervals cross zero on both datasets.

## Limitations and final experiment

- Only 58 GSM8K training questions were usable, selected by API completion order rather than a fresh random sample.
- Each evaluation set has 100 questions, so confidence intervals remain wide.
- The answer-only condition has equal row count and update count, but shorter target sequences and therefore fewer supervised tokens.
- TF-IDF distance measures surface-form diversity, not mathematical-strategy diversity.
- Prompt sensitivity and stochastic answer consistency were not run in this core pilot.

For the final dissertation experiment, restore teacher API credit and regenerate at least 500-1,000 randomly sampled GSM8K training questions with five candidates each. Freeze the eligible question set before training, retain the same three seeds and two evaluation datasets, and optionally add a token-matched answer-only control. The main reported conclusion should remain based on exact match mean +/- seed SD and paired bootstrap confidence intervals.
