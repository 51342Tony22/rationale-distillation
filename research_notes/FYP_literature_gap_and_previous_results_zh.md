# FYP 文献定位、明确研究缺口与已有实验结果

更新日期：2026-10-08

## 一句话结论

建议把 FYP 聚焦为：

> **When does rationale diversity help a tiny language model? A budget-controlled study of CoT distillation.**  
> 在严格匹配题目、正确性、rationale 数量、训练 token 与优化步数的条件下，研究不同类型的 rationale 多样性，以及 rationale 与学生能力的匹配程度，何时能够帮助、何时反而干扰 0.6B 小模型的数学推理。

不要把 gap 写成“多 rationale 或 diverse rationale 还没人研究”。这个表述已经被 SCoTD、Fine-tune-CoT、MCC-KD、MoRSD 和 MIND 推翻。

## 1. 文献比较矩阵

说明：这是围绕本项目问题的定向系统综述，而不是声称覆盖所有 CoT 论文的 PRISMA 全量综述。训练规模若随 benchmark 改变，表中如实写作“完整 split / 依任务而变”，不凭空合并为一个数字。

| 论文 | Teacher | Student | 训练数据/监督规模 | Rationale 监督 | 数据集 | 主要结果及与本项目的关系 |
|---|---|---|---|---|---|---|
| [Teaching Small Language Models to Reason](https://aclanthology.org/2023.acl-short.151/) (Magister et al., 2023) | PaLM 540B；GPT-3 175B | T5 Small/Base/XL/XXL（约 60M–11B） | GSM8K：原训练 6,725，过滤后有效 CoT 5,337；MAWPS 1,590；ASDiv 1,544 | 单条、答案正确的 teacher CoT；用 gold answer 条件生成并过滤 | GSM8K、MAWPS、ASDiv，另含常识/符号任务 | GSM8K 上 T5-XXL 从 8.11% 提升至 21.99%（PaLM CoT）或 18.42%（GPT-3 CoT）。证明 CoT 可以蒸馏，但没有研究多样性。 |
| [Distilling Reasoning Capabilities into Smaller Language Models](https://arxiv.org/abs/2212.00193) (Socratic CoT, 2023) | GPT-3 175B | GPT-2 124M/355M/774M/1.5B | 采用各 benchmark 的训练 split；GSM8K 约 7.5k，其他任务规模依设置变化 | 将问题分解为子问题；单模型或“问题分解器 + 子问题求解器” | GSM8K、StrategyQA、SVAMP | GSM8K 的 GPT-2 Large：answer-only 4.62%、普通 CoT 12.85%、Socratic CoT 17.89%。说明 rationale 的结构很重要，但不是数量/多样性的受控研究。 |
| [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) (Hsieh et al., 2023) | PaLM 540B；另有 GPT-NeoX 20B 消融 | T5 220M/770M/11B | 按任务使用完整或不同百分比训练 split | label prediction 与 rationale generation 两个任务的多任务学习；通常每题一条 rationale | e-SNLI、ANLI、CommonsenseQA、SVAMP | 770M T5 在一个 benchmark 上用 80% 数据即可超过 few-shot PaLM 540B；证明 rationale 是额外监督信号，但未区分数量与多样性。 |
| [Symbolic CoT Distillation](https://aclanthology.org/2023.acl-long.150/) (SCoTD, 2023) | GPT-3/code-davinci-002，175B | OPT 125M–1.3B | 默认每题采样 30 条；选择实验将各方法固定到 5 条 | 多 rationale；比较随机、SBERT 多样性、teacher likelihood、open-endedness | CommonsenseQA、QuaRel、OpenBookQA | rationale 从 1 增至 30 时，CSQA 53.0→60.2、QuaRel 62.2→73.4、OBQA 39.0→44.4（对应论文设置）；随机抽样也很强，说明“量”本身是重要混杂因素。它部分控制 K，但任务是常识选择题，且没有严格匹配 token/compute。 |
| [Large Language Models Are Reasoning Teachers](https://aclanthology.org/2023.acl-long.830/) (Fine-tune-CoT, 2023) | InstructGPT/GPT-3 175B | GPT-3 0.3B/1.3B/6.7B；另测试 T5、Flan-T5、GPT-2 | 每题 1–64 条 rationale | 多次采样“diverse reasoning”，正确答案过滤 | MultiArith、SVAMP、GSM8K 等多类推理任务 | 6.7B student 在 MultiArith 约 +26%、SVAMP 约 +17%。但随着多样性上升，rationale 数量和训练数据量也同时上升，因此不能证明增益来自 diversity 而非 quantity。 |
| [MCC-KD](https://aclanthology.org/2023.findings-emnlp.454/) (Chen et al., 2023) | GPT-3.5-Turbo | Flan-T5 3B/11B；LLaMA 7B/13B | 各任务完整训练 split；每题生成 5 条候选 rationale | 3-gram Jaccard 促进多样性；对不同 rationale 的答案分布施加双向 KL 一致性 | GSM8K、SVAMP、ASDiv、CommonsenseQA 等 | 例如 LLaMA-7B 在 GSM8K 由约 38.01% 提升到 41.58%。但 diversity 与 consistency loss 被绑在同一方法中，无法单独归因。 |
| [Investigating Mysteries of CoT-Augmented Distillation](https://aclanthology.org/2024.emnlp-main.349/) (Wadhwa et al., 2024) | Mistral-7B-Instruct | GPT-2、Phi-1.5、Gemma-2B | 各任务训练 split，每题 teacher rationale；进行顺序、打乱、删 token 消融 | 比较 rationale 在 label 前/后、打乱 token、只保留关键 token | CommonsenseQA、OpenBookQA、QuaRel | label 后放 rationale 反而更好；打乱 rationale 仍可能有收益；少数关键 token 可达到完整 rationale 的增益。关键警告：**accuracy 提升不自动等于真正的 reasoning transfer**。 |
| [Unveiling the Key Factors for Distilling CoT Reasoning](https://aclanthology.org/2025.findings-acl.782/) (Chen et al., 2025) | GPT-4o、Gemini-1.5-Flash、LLaMA-3-70B、人类 CoT | BLOOM 560M/1.1B/1.7B/3B、Gemma 2B、LLaMA-3.2 1B/3B | 每个 benchmark 的训练 split；每题生成 6 个 granularity levels，并比较多种格式 | 研究细粒度、格式与 teacher，而非固定 K 下的多样性 | SVAMP、GSM8K、AQuA-RAT、MATH、CSQA、OBQA、StrategyQA | 弱 student 可能被过细 rationale 压垮；格式影响较小；更强 teacher 不总产生更强 student。直接支持“student–rationale fit”而不是“越多越复杂越好”。 |
| [MoRSD](https://aclanthology.org/2025.findings-emnlp.413/) (Yan et al., 2025) | GPT-3/text-davinci-002，175B | Flan-T5 Small/Base/Large/XL（约 60M/250M/780M/3B） | 先生成多条；准确性过滤后用 3-gram Jaccard 从候选中保留 K=6，再按 student-specific RD 选 k=3 | 联合 accuracy、表面 diversity 与 student-specific rationale difficulty | SingleEq、AddSub、MultiArith、StrategyQA、SVAMP、Date Understanding、Shuffled Objects 等 | 七个数据集平均比 baseline 高 4.6%，且用更少 rationale。它已经研究“质量优于总量”，但 accuracy、diversity、difficulty 与候选数共同变化；不能给出固定预算下 diversity 的独立因果效应。 |
| [MIND](https://aclanthology.org/2026.acl-long.2020/) (Cui et al., 2026) | Qwen3-235B | Qwen2.5 1.5B/7B、Llama3.1 8B | 497 个 MATH 样本，每题 8 个显式视角；扩展实验达 4k–16k paths | formal-symbolic、intuitive、decomposition、analogy、Socratic、contrastive、counterfactual、planning；动态 student-aware fusion | MATH500、GSM8K、SVAMP；OOD 的 CSQA、StrategyQA、GPQA-D | Qwen2.5-1.5B 的 GSM8K 72.81→76.50、SVAMP 79.33→82.83；但直接混合 8 个视角会使小模型退化。它表明**多样性可能是帮助，也可能是冲突噪声，取决于学生匹配与融合方式**。最低 student 仍是 1.5B。 |
| [Which Reasoning Trajectories Teach Students Better?](https://aclanthology.org/2026.acl-long.1950.pdf) (Yang et al., 2026) | 11 个 reasoning teachers，4B–671B | Qwen3 14B/4B、Llama3.1 8B、Qwen2.5 7B/3B | 5,000 个数学问题；每题从多个 teacher/run 选择一条 | Rank-Surprisal Ratio 同时衡量可学习性和信息量 | AIME 2024/2025、AMC 2023、MATH500，另测 GPQA | RSR 与训练后性能的平均 Spearman 相关达到 0.86，并优于随机、长度、质量和 perplexity 选择。关注 student suitability，但 student 最小 3B、每题最终仍选单条 trajectory。 |
| [The Signal is in the Steps](https://proceedings.mlr.press/v306/just26a.html) (LALP, 2026) | 多个不同 teachers | 多个 students | 从多 teacher 长 CoT 池中选择 | 用局部步骤概率，而不是整条 trajectory 概率，评估 student 可学习性 | 数学、代码、科学 | 说明可迁移信号可能位于局部 reasoning transitions；进一步削弱“文本距离越大就越好”的简单假设。 |

## 2. 明确且可辩护的 research gap

### 核心 gap

在已审阅文献中，**仍缺少针对 sub-1B tiny language model 的低资源、预算严格匹配实验，用来分离以下因素的独立作用和交互作用**：

1. 每题 rationale 的独特数量；
2. rationale 的表面/词汇多样性；
3. 真正的解题策略或推理路径多样性；
4. rationale 对当前 student 的可学习性；
5. 总训练 token、样本曝光次数和优化步数。

已有论文通常至少混合了其中两个因素：

- Fine-tune-CoT 同时增加 rationale 数量和所谓 diverse reasoning；
- SCoTD 固定过 K，但主要是常识选择题，而且未严格匹配 token 与 compute；
- MCC-KD 把 diversity 和 consistency loss 绑定；
- MoRSD 同时使用正确性、多样性和 student difficulty 选择；
- MIND 研究显式策略多样性和动态融合，但 student 从 1.5B 起，并且方法组件复杂；
- 2026 年 student-aware work 主要研究 3B 以上 long-CoT 模型和单 trajectory 选择。

因此，最安全的 gap 不是“diversity 是否有用”，而是：

> **在固定监督预算下，什么类型、什么程度、并且与 student 能力如何匹配的 rationale diversity，才能真正帮助 sub-1B 模型；何时 diversity 只是表面改写，或会因路径冲突而伤害学习？**

### 机制 gap

Wadhwa et al. 表明打乱 rationale 或只保留少量关键 token 也可能提升 accuracy。因此，本项目不能只报告 exact match，然后直接声称“推理能力被成功蒸馏”。还需要至少一种机制/稳健性证据，例如：

- OOD transfer：只在 GSM8K 训练，在 SVAMP/ASDiv 测试；
- rationale corruption control：打乱步骤或打乱 token；
- strategy-sensitive evaluation：按算术运算类型、步骤数或解题策略分组；
- answer-first 与 rationale-first 对照；
- rationale 质量、长度和 student perplexity 与收益的关联。

### 推荐题目

**When Does Rationale Diversity Help Tiny Language Models? A Budget-Controlled Study of Chain-of-Thought Distillation**

中文可写为：

**何时推理依据多样性能够帮助微型语言模型？一项预算受控的思维链蒸馏研究**

## 3. 推荐研究问题

- **RQ1:** 在相同训练预算下，正确 CoT supervision 是否稳定优于 answer-only supervision？
- **RQ2:** 在题目曝光、记录数和 token budget 匹配时，增加独特 rationale 数量是否仍然有效？
- **RQ3:** 固定每题 K=3 时，低多样性、随机和高多样性 rationale selection 对 0.6B student 有何差异？这种差异来自词汇差异还是真实策略差异？
- **RQ4:** student-aware selection 是否比纯 maximum-diversity selection 更稳定，尤其是在 OOD 的 SVAMP/ASDiv 上？

如果时间有限，RQ1–RQ3 是 FYP 主线，RQ4 作为 extension。

## 4. 我们已经完成的实验

### 4.1 实验设置

- Teacher：`deepseek-ai/DeepSeek-V4-Flash`（SiliconFlow API）。
- Student：`mlx-community/Qwen3-0.6B-Base-4bit`。
- 训练：MLX QLoRA，LoRA rank 16；3 个 seeds：7、17、177。
- 数据：从 GSM8K train 中取 200 题，每题生成 5 条候选；196 题至少有 3 条答案正确的 rationale。
- Teacher 候选：共 1,000 条，1 次 API error，973 条答案正确并被接受。
- 每个条件：196 个相同问题、588 条训练记录（每题 K=3）、约 3 epochs、220 optimizer updates。
- 测试：未参与训练的 GSM8K test 前 100 题和 SVAMP 前 100 题；数值 exact match。
- 不确定性：3 seeds；paired bootstrap 10,000 次，以每题的跨 seed 平均正确率做比较。

四个条件：

1. `answer_only`：同一 final answer 重复 3 次；
2. `single_rationale`：同一条正确 rationale 重复 3 次；
3. `multi_rationale`：随机选择 3 条正确 rationale；
4. `diverse_multi_rationale`：用 TF-IDF 距离贪心选择 3 条最不同的正确 rationale。

选择后的平均 TF-IDF 距离：普通 multi 为 0.4566，diverse multi 为 0.5629。

### 4.2 GSM8K 前 100 题

| 条件 | Seed 7 | Seed 17 | Seed 177 | Mean ± SD |
|---|---:|---:|---:|---:|
| Base student（单次） | 3% | — | — | 3.0% |
| Answer-only | 1% | 1% | 0% | 0.7% ± 0.6 |
| Single rationale | 31% | 36% | 31% | 32.7% ± 2.9 |
| Random multi-rationale | 33% | 29% | 35% | 32.3% ± 3.1 |
| Diverse multi-rationale | 42% | 38% | 38% | **39.3% ± 2.3** |
| Teacher | — | — | — | 98% |

关键 paired-bootstrap 比较：

- single − answer-only：**+32.0 pp**，95% CI [25.3, 39.0]；
- multi − single：**−0.3 pp**，95% CI [−5.7, 5.0]；
- diverse − multi：**+7.0 pp**，95% CI [1.3, 12.7]；
- diverse − answer-only：**+38.7 pp**，95% CI [30.7, 46.7]。

### 4.3 SVAMP 前 100 题（OOD）

| 条件 | Seed 7 | Seed 17 | Seed 177 | Mean ± SD |
|---|---:|---:|---:|---:|
| Base student（单次） | 2% | — | — | 2.0% |
| Answer-only | 13% | 6% | 2% | 7.0% ± 5.6 |
| Single rationale | 53% | 45% | 53% | 50.3% ± 4.6 |
| Random multi-rationale | 52% | 48% | 43% | 47.7% ± 4.5 |
| Diverse multi-rationale | 54% | 56% | 48% | **52.7% ± 4.2** |
| Teacher | — | — | — | 96.9%（97 个有效响应）；若 3 个 timeout 算错则为 94% |

关键 paired-bootstrap 比较：

- single − answer-only：**+43.3 pp**，95% CI [36.3, 50.3]；
- multi − single：**−2.7 pp**，95% CI [−9.3, 4.0]；
- diverse − multi：**+5.0 pp**，95% CI [−1.0, 11.0]；
- diverse − answer-only：**+45.7 pp**，95% CI [38.3, 53.0]。

### 4.4 正确解释

目前可以说：

1. rationale supervision 明显优于 answer-only；
2. **普通地加入更多不同 rationales 并没有优于 single rationale**，在两个测试集上均略低；
3. 从同样数量的候选中选择 TF-IDF 更不同的三条，在 GSM8K 上相对 random multi 提升 7 pp，CI 不跨 0；
4. SVAMP 方向相同（+5 pp），但 CI 跨 0，因此只能称为趋势；
5. 当前结果更支持“selection matters”，而不是“more rationales always help”。

目前不能说：

- “diversity 已被证明普遍有效”；
- “提升来自真正的 reasoning-path diversity”；
- “模型已接近理论上应有的 90%”；
- “已经完全排除 data leakage”；
- “已经证明 reasoning 被内化，而非训练信号/关键词效应”。

## 5. 数据泄漏复核

项目原有 audit 的结论：

- GSM8K train/dev/test 原始 question exact overlap 均为 0；
- 四个条件使用相同题目集合、相同行数，没有未知或错配记录；
- 但原 audit 只做了去首尾空格后的 exact string matching。

本次补做的词法近重复检查：使用 lowercase alphanumeric normalization，以及 character 3–5 gram TF-IDF cosine similarity，将实际使用的 196 道训练题分别与 GSM8K、SVAMP 的前 100 题和完整测试集比较。

| 比较 | 标准化 exact overlap | cosine ≥ 0.80 | 最高相似度 |
|---|---:|---:|---:|
| GSM8K test 前 100 | 0 | 0 | 0.495 |
| GSM8K test 全部 1,319 | 0 | 0 | 0.725 |
| GSM8K dev 全部 500 | 0 | 0 | 0.769 |
| SVAMP 前 100 | 0 | 0 | 0.518 |
| SVAMP 全部 1,000 | 0 | 0 | 0.524 |

结论：没有发现直接复制或高词法近重复，但这仍不能排除：

- 相同数学模板的语义改写；
- student 预训练阶段已见 benchmark；
- teacher 的预训练/后训练污染；
- 评测样本仅 100 题带来的偶然性。

因此论文中应写“no exact or high lexical near-duplicate overlap was detected”，不能写“data leakage has been completely ruled out”。

## 6. 下一阶段最小实验设计

保留当前 0.6B student，不急着扩 model size。先完成以下条件：

| 条件 | K | 目的 |
|---|---:|---|
| Answer-only ×3 | 3 records | 无 rationale 基线 |
| Single rationale ×3 | 1 unique, repeated | CoT 是否有效；匹配曝光次数 |
| Low-diversity 3 | 3 | 新增关键对照：刻意选择最相似的三条 |
| Random 3 | 3 | 普通多 rationale |
| Max TF-IDF diversity 3 | 3 | 复现现有表面多样性结果 |
| Strategy-diverse 3 | 3 | 区分“词汇改写”与“真实解题路径” |
| Student-aware 3（可选） | 3 | 结合 MoRSD/RSR：选择既有信息量又可学的 rationale |

所有条件必须保持：

- 相同 196 个问题；
- teacher answer 全部正确；
- 相同记录数或明确报告有效 target-token 数；
- 相同 optimizer updates；
- 至少 3 seeds；
- GSM8K 完整 test 和 SVAMP/ASDiv OOD，而不是只测前 100；
- 除 example-matched 结果外，再做一次 target-token-matched robustness check。

## 7. 可直接对老师说的版本

> After reviewing the literature, I no longer think the gap is simply that diverse rationales have not been studied. Prior work such as SCoTD, MCC-KD, MoRSD and MIND has already explored multiple or diverse reasoning paths. However, these studies often confound rationale quantity, diversity, consistency objectives, student difficulty, and training budget. The more defensible gap is a controlled study for a sub-billion-parameter student that separates rationale quantity, surface diversity, genuine strategy diversity, and student–rationale compatibility under matched supervision and compute budgets. Our current results motivate this question: rationale supervision substantially outperforms answer-only training, random three-rationale training does not outperform repeating one rationale, while TF-IDF-selected diverse rationales improve GSM8K by 7 percentage points over random multi-rationale training. The SVAMP gain is 5 points but is not yet statistically conclusive. Therefore, the next step should be to add low-diversity and strategy-diverse controls, match token budgets, evaluate on full test sets, and strengthen the leakage audit before testing more student sizes.

