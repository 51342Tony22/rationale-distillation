# FYP Interim Results and Student Model Scale Selection

## 1. Current Research Question

This project investigates how different forms of rationale supervision affect the mathematical reasoning performance of compact language models. It compares four training conditions: answer-only, single-rationale, multi-rationale, and diverse multi-rationale supervision. The current experiment is designed to determine whether rationale quantity and diversity affect distillation performance; it does not assume that the student must already match the teacher.

## 2. Current Experimental Setup

| Item | Setting |
|---|---|
| Teacher model | DeepSeek-V4-Flash API |
| Student model | Qwen3-0.6B-Base-4bit |
| Fine-tuning method | MLX QLoRA, rank 16 |
| Unique training questions | 196 GSM8K questions |
| Training records per condition | 588 |
| Random seeds | 7, 17, and 177 |
| Training duration | Approximately 3 epochs and 220 optimisation steps |
| Evaluation sets | 100 GSM8K questions and 100 SVAMP questions |
| Primary metric | Final numerical-answer Exact Match |

## 3. Main Results

| Training condition | GSM8K EM | SVAMP EM | Gain over answer-only on GSM8K | Gain over answer-only on SVAMP |
|---|---:|---:|---:|---:|
| Unfine-tuned student | 3.0% | 2.0% | - | - |
| Answer-only | 0.7% | 7.0% | - | - |
| Single-rationale | 32.7% | 50.3% | +32.0 | +43.3 |
| Multi-rationale | 32.3% | 47.7% | +31.6 | +40.7 |
| Diverse multi-rationale | **39.3%** | **52.7%** | **+38.6** | **+45.7** |
| Teacher model | 98.0% | 96.9% | - | - |

The SVAMP teacher score of 96.9% is conditional accuracy over the 97 successful API responses. If the three timeouts are counted as incorrect, end-to-end accuracy is 94.0%.

On GSM8K, diverse multi-rationale training exceeds ordinary multi-rationale training by 7.0 percentage points, with a 95% paired-bootstrap interval of [1.3, 12.7] percentage points. On SVAMP, the corresponding gain is 5.0 points, but the interval of [-1.0, 11.0] crosses zero.

## 4. Interpretation

1. CoT supervision is substantially more effective than final-answer-only supervision on both datasets.
2. Increasing the number of rationales alone does not guarantee an improvement; ordinary multi-rationale training does not consistently outperform single-rationale training.
3. Selecting more diverse correct rationales produces the highest mean accuracy, suggesting that rationale quality and coverage may matter more than rationale count.
4. The current student does not yet approach the teacher. The best condition closes 38.2% of the original teacher-student gap on GSM8K and 53.4% on SVAMP.
5. This is not evidence that rationale distillation failed. The current ceiling is constrained by only 196 unique questions, a 0.6B Base student, 4-bit QLoRA, and limited optimisation updates.

## 5. Selecting the Student Model Scale

The model-scale study should separate two objectives.

### Scientific Comparison

To determine whether parameter scale changes the benefit of rationale distillation, the main comparison should use Base models from the same family.

| Scale | Experimental role | Main experiment? | Rationale |
|---|---|---:|---|
| Qwen3 0.6B Base | Current compact baseline | Yes | Completed; measures method effects in a low-resource setting |
| Qwen3 1.7B Base | Primary next-scale model | **Yes, priority** | Remains compact while providing substantially greater reasoning capacity |
| Qwen3 4B Base | Scale-sensitivity condition | If resources permit | Tests whether gains increase or saturate with student capacity |
| 7B mathematics/reasoning model | Teacher-proximity upper bound | Report separately | Changes pretraining and alignment, so differences cannot be attributed to scale alone |

### Engineering Objective

If the objective changes from isolating scale effects to maximising teacher proximity, a 7B mathematics-specialised or reasoning-distilled model can be added. It should be labelled as an upper-bound experiment because it may already have strong GSM8K performance before project-specific fine-tuning and is not directly comparable with the 0.6B Base student.

## 6. Recommended Next Stage

1. Increase the number of unique training questions from 196 to approximately 2,000 initially, and use the complete GSM8K training set if the annotation budget permits.
2. Use Qwen3 0.6B Base and Qwen3 1.7B Base in the primary scale experiment; add Qwen3 4B Base if resources permit.
3. Keep the teacher, data splits, rationale-selection procedure, random seeds, evaluation prompts, and fine-tuning method fixed, changing only student scale.
4. Retain all four supervision conditions: answer-only, single-rationale, multi-rationale, and diverse multi-rationale.
5. Evaluate on the complete GSM8K and SVAMP test sets rather than 100-example subsets.
6. Predefine teacher proximity as being within five absolute accuracy points of the teacher, and also report gap closure.
7. If a 7B mathematics model is added, report it separately as a teacher-proximity upper bound rather than mixing it into the same-family Base-model scale ablation.

## 7. Summary for the Supervisor

The current experiment answers the original research question: CoT supervision substantially improves the 0.6B student; merely increasing rationale count provides no consistent benefit; and selecting more diverse correct rationales gives the strongest mean performance. The student does not yet approach the teacher, primarily because 196 unique training questions and a 0.6B Base model are insufficient for that stronger objective. The next stage should prioritise broader training-question coverage and use the 1.7B Base model as the main scale extension. The 4B model can provide a scale-sensitivity condition, while a 7B mathematics model should be treated only as a separate upper-bound experiment.
