# CoT/Rationale Distillation 文献比较（论文 1–12）

更新日期：2026-10-08  
用途：回应导师要求，系统整理 **teacher/student、模型规模、训练数据量、rationale 监督类型、数据集与报告的性能**。

## 1. 阅读说明

- `B` = billion（十亿参数），`M` = million（百万参数）。闭源模型的参数量采用论文报告值。
- “训练数据量”分成三件事记录：**原始问题数、每题 rationale 数、最终训练记录数**。论文没有报告最终过滤后记录数时，明确写“未报告”。
- “代表结果”只用于比较同一论文内的方法；不同论文的数据划分、提示词、解码和评测协议不同，不能直接横向排名。
- 大多数论文的主要指标是最终答案 **Accuracy/Exact Match**。有些论文另外报告 calculator-assisted accuracy、BLEU/BERT F1、相关系数或跨任务平均准确率。

## 2. Teacher、student 与训练监督规模

| # | 论文 | Teacher（规模） | Student（规模） | 训练数据/监督规模 |
|---:|---|---|---|---|
| 1 | Teaching Small Language Models to Reason | PaLM 540B；GPT-3 175B | T5 Small/Base/Large/XL/XXL，约 60M–11B；主结果为 T5-XXL 11B | GSM8K 原训练 6,725 题，PaLM/GPT-3 正确性过滤后分别为 5,337/5,298 条；MAWPS 1,590；ASDiv 原 1,844、过滤后 1,544；StrategyQA 原 1,648、过滤后 1,319。通常每题 1 条 CoT。 |
| 2 | Distilling Reasoning Capabilities into Smaller Language Models（Socratic CoT） | GPT-3 175B | GPT-2 Small 124M、Medium 355M、Large 774M、XL 1.5B | GSM8K 约 7.5k train/1k test；SVAMP 实验用 ASDiv 761 题训练、SVAMP 237 题测试；StrategyQA 按 80/10/10 划分。每题生成一条普通 CoT 或一组 subquestion–solution pairs；过滤后的总记录数未报告。 |
| 3 | Distilling Step-by-Step | PaLM 540B；GPT-NeoX 20B（消融） | T5-Base 220M、T5-Large 770M、T5-XXL 11B | e-SNLI 549,367/9,842/9,824；ANLI-R1 16,946/1,000/1,000；CommonsenseQA 8,766/975/1,221；SVAMP 720/80/200（train/dev/test）；另用 ASDiv 2,305 条无标签题扩充 SVAMP。每题使用一条 teacher rationale。 |
| 4 | Symbolic Chain-of-Thought Distillation（SCoTD） | GPT-3 code-davinci-002，175B | OPT 125M、350M、1.3B；主结果为 OPT-1.3B | 使用 CommonsenseQA、QuaRel、OpenBookQA 的训练 split；论文未汇总问题总数。默认每题采样 30 条，监督设置按正确答案过滤后分别丢弃 40.4%、34.2%、45.0%；选择实验统一为平均每题 5 条正确 CoT。 |
| 5 | MCC-KD | GPT-3.5-Turbo（参数量未公开） | Flan-T5-XL 3B、XXL 11B；LLaMA 7B、13B（另有 250M 消融） | GSM8K 7,473/660/659；ASDiv 1,462/313/314；SVAMP 700/150/150；CommonsenseQA 8,520/1,221/1,221（train/dev/test）。过滤后每题保留 K=5 条 rationale；每个 epoch 随机取两条计算一致性损失。 |
| 6 | Unveiling the Key Factors for Distilling CoT Reasoning | GPT-4o、Gemini-1.5-Flash、LLaMA-3-70B；另用人类 CoT | BLOOM 560M/1.1B/1.7B/3B、Gemma 2B、LLaMA-3.2 1B/3B | SVAMP 700、GSM8K 7.4k、AQuA-RAT 6.1k、MATH 1.3k、CSQA 9.7k、OBQA 4.9k、StrategyQA 2k 条训练题。每题生成不同 granularity（6 级）或不同 format 的 CoT；最终生成记录总数未统一报告。 |
| 7 | Large Language Models Are Reasoning Teachers（Fine-tune-CoT） | 默认 InstructGPT text-davinci-002，175B；另比较 4 个 GPT-3 175B 版本 | GPT-3 ada 0.3B、babbage 1.3B、curie 6.7B；另测 GPT-2 124M/355M/774M、T5/Flan-T5 60M/220M/700M | 12 个任务的训练量从 258 到 10,000：例如 MultiArith 420、GSM8K 7,473、SVAMP 700、CSQA 9,741。普通设置每题 1 条；diverse reasoning 为每题 D=8，MultiArith/SVAMP 最大 D=64，并按最终答案过滤。 |
| 8 | Investigating Mysteries of CoT-Augmented Distillation | Mistral-7B-Instruct，7B | GPT-2 约 124M、Phi-1.5 约 1.3B、Gemma 2B | CSQA 9,741/1,221/1,140；OpenBookQA 4,957/500/500；QuaRel 1,948/278/514。每题基于 gold label 生成 1 条 rationale；使用完整训练 split，最终记录数与训练题数相同。 |
| 9 | MoRSD | GPT-3 text-davinci-002，175B | Flan-T5 Small/Base/Large/XL，约 60M/250M/780M/3B | SingleEq 356、AddSub 276、MultiArith 420、SVAMP 700、Date 258、Shuffled Objects 525、StrategyQA 1,603 条训练题。先正确性过滤，再保留 K=6 条多样 rationale，最后按 student-specific difficulty 选择 k=3；即最终约为每题 3 条记录。 |
| 10 | MIND | Qwen3-235B，235B | Qwen2.5 1.5B/7B、Llama-3.1 8B | 从 MATH 构造 497 道训练题，每题 8 条答案正确的不同视角 CoT，共 3,976 条候选路径；训练时动态融合/选择。 |
| 11 | Which Reasoning Trajectories Teach Students Better?（RSR） | 11 个 reasoning teachers，4B–671B（含 DeepSeek-R1、Qwen3、QwQ、Nemotron、Phi 等） | Qwen3 14B/4B、Llama-3.1 8B、Qwen2.5 7B/3B | 从 NuminaMath 选 5,000 道题。每个 teacher 独立生成 3 套、每套 5,000 条 trajectory；选择实验形成每题 33 条候选（11 teachers × 3），最终选 1 条，因此 student 最终训练集为 5,000 条。 |
| 12 | The Signal Is in the Steps（LALP） | DeepSeek-R1（约 671B）、Qwen3-32B-Instruct、QwQ-32B | Qwen2.5-7B/32B-Instruct、Llama-3.1-8B-Instruct | 主要选择实验使用 LIMO 的 817 个 prompts；每题汇集 3 个 teacher 的答案正确长推理候选，最终选 1 条，得到 817 条训练记录。另有 MATH 8,890 题等扩展实验。 |

## 3. Rationale 监督、数据集、指标与报告性能

| # | Rationale/CoT 监督类型 | 训练与评测数据集 | 报告指标 | 论文内代表性结果 |
|---:|---|---|---|---|
| 1 | gold-answer-conditioned 单条 teacher CoT；按最终答案过滤；student 输出 CoT+answer | GSM8K、MAWPS、ASDiv、StrategyQA、Last Letter、Coin Flip | 最终答案 accuracy；数学题另报 calculator-assisted accuracy | GSM8K T5-XXL：原模型 8.11%，PaLM-CoT 21.99%，加 calculator 后 38.21%；GPT-3-CoT 为 18.42%。 |
| 2 | answer-only、人工步骤、普通 CoT、Socratic CoT；后者把解题过程拆成 subquestion–solution pairs，并比较 unified 与 QG+QA 双模型 | GSM8K、StrategyQA、SVAMP | 最终答案 accuracy；子问题质量另报 BLEU、BERT-F1、问题数量匹配 | GSM8K GPT-2 Large：answer-only 4.62%，普通 CoT 12.85%，iterative SocCoT 17.89%，使用人工步骤的 SocGT 21.08%。 |
| 3 | 两个任务前缀分别训练 label prediction 与 rationale generation；总损失为两任务之和 | e-SNLI、ANLI-R1、CommonsenseQA、SVAMP；ASDiv 用于扩充 | 分类/最终答案 accuracy | T5-Base 220M：标准微调→Distill Step-by-Step：e-SNLI 88.38→89.51，ANLI 43.58→49.58，CQA 62.19→63.29，SVAMP 62.63→65.50。 |
| 4 | 多条采样 CoT；比较随机选择、SBERT diversity、teacher likelihood、open-endedness；监督设置过滤错误答案 | CommonsenseQA、QuaRel、OpenBookQA | accuracy；另有人类偏好/连贯性与 self-consistency | OPT-1.3B 全量监督的 label-only→SCoTD：CSQA 63.0→67.0，QuaRel 59.0→83.8，OpenBookQA 60.2→67.0。无 self-consistency 时，1→30 条 CoT：CSQA 53.0→60.2，QuaRel 62.2→73.4，OBQA 39.0→44.4。 |
| 5 | 多 CoT；答案正确性过滤；3-gram Jaccard 去除相似 rationale；双向 KL 约束不同 rationale 得到一致答案分布 | ID：GSM8K、ASDiv、SVAMP、CSQA；OOD：MultiArith、SingleEq、AddSub、StrategyQA、Date、Shuffled Objects、Coin Flip | accuracy，三次运行均值；另分析 Jaccard diversity | LLaMA-7B Vanilla KD→MCC-KD：GSM8K 38.01→41.58，ASDiv 64.01→65.76，SVAMP 62.67→64.67，CSQA 75.10→76.41。 |
| 6 | 系统改变三个因素：teacher 来源、CoT granularity（Level 1–6）和 format；对照为 answer-only | SVAMP、GSM8K、AQuA-RAT、MATH、CSQA、OBQA、StrategyQA | accuracy，通常报告 3 次运行 mean±std | Gemma-2B 在 GSM8K：answer-only 8.20%，不同 granularity 为 49.66%–53.45%；Llama-3.2-3B 为 59.59%–63.48%。最佳粒度通常在中间，而非越详细越好。 |
| 7 | teacher 生成 rationale+answer；按答案过滤；diverse 版本用温度采样生成多条不同 reasoning explanations | SingleEq、AddSub、MultiArith、GSM8K、AQuA、SVAMP、Date、Shuffled Objects、Last Letter、Coin Flip、CSQA、StrategyQA | 最终答案 accuracy | GPT-3 6.7B student：普通 Fine-tune-CoT→diverse reasoning：MultiArith 33.33→53.33，SVAMP 12.67→30.33，Date 60.36→83.78。 |
| 8 | 对 CoT 的位置和内容做消融：rationale 在 label 前/后、打乱 token、屏蔽 token、只保留关键 token；测试收益是否依赖完整连贯推理 | CSQA、OpenBookQA、QuaRel | accuracy；另分析 hidden-state label confidence | GPT-2 baseline→CoT-after-label：CSQA 63.11→70.92，OBQA 60.20→70.26，QuaRel 59.05→71.04；CoT-after-label 整体优于 CoT-before-label。 |
| 9 | 三阶段 selection：正确性阈值→3-gram Jaccard diversity→按 student 的 Rationale Difficulty 选较易吸收的 3 条 | SingleEq、AddSub、MultiArith、StrategyQA、SVAMP、Date、Shuffled Objects | 最终答案 accuracy；同时比较 rationale 使用量 | Flan-T5-Base Vanilla CoT→MoRSD：StrategyQA 60.84→65.65，SVAMP 12.33→13.67，Date 84.68→86.49，Shuffled Objects 67.11→99.56。 |
| 10 | 8 种明确视角：planning、formal-symbolic、counterfactual、Socratic、decomposition、intuitive、contrastive、analogy；由 student feedback 动态加权，并加一致性正则 | ID：MATH500、GSM8K、SVAMP；OOD：CSQA、StrategyQA、GPQA-Diamond | accuracy；ID/OOD 平均增益 | Qwen2.5-1.5B base→MIND：MATH500 50.40→54.09，GSM8K 72.81→76.50，SVAMP 79.33→82.83，GPQA-D 24.75→31.31。 |
| 11 | RSR（Rank-Surprisal Ratio）同时考虑 trajectory 对 student 的信息量与可学习性；用于 teacher/trajectory 选择 | AIME 2024/2025、AMC 2023、MATH500；另测 GPQA | 四个数学 benchmark 的平均 Acc@4；指标与训练后性能的 Spearman correlation | RSR 与训练后性能平均 Spearman ρ=0.86。与随机选择相比，Qwen3-14B 平均 70.2→78.6，Qwen2.5-7B 45.7→53.2，Qwen2.5-3B 27.9→34.8。 |
| 12 | 比较整条序列平均概率 GALP 与局部步骤概率 LALP；LALP 将长 CoT 分步，优先选择局部可学习但仍有信息的推理 | 数学：MATH、AIME 2025、AMC、MINERVA、KAOYAN、OlympiadBench、CN_MATH24；另测 GPQA-Diamond、LiveCodeBench | 多数学 benchmark 平均 accuracy；科学/代码 accuracy | 混合 teacher 选择中，Qwen2.5-7B 的 Random/GALP/LALP 为 40.7/41.2/44.0%；Qwen2.5-32B 为 65.1/63.2/72.6%，LALP 比 GALP 高 9.4 个百分点。 |

## 4. 给导师汇报时可以直接概括的结论

1. **基础结论已经比较稳定：** 论文 1–3、7 表明，teacher-generated rationale 通常比只训练最终答案更能提升小模型的推理准确率。
2. **监督形式会影响效果：** Socratic decomposition、多任务 label+rationale、rationale 的位置与粒度都可能改变结果，因此不能把所有 CoT supervision 当成同一种方法。
3. **多条 rationale 并非简单“越多越好”：** 论文 4、5、7 表明多条 CoT 可带来提升，但同时增加了训练记录和 token；论文 6、8 进一步说明过细、过长或完整连贯的 CoT 不一定总是最有效。
4. **较新的工作转向“选择适合 student 的 rationale”：** 论文 9–12 不再只追求更多 CoT，而是依据 difficulty、student alignment 或局部步骤可学习性筛选训练数据。
5. **文献中的 student 覆盖范围很宽：** 从约 60M 到 32B 都有，但早期基础论文常用 T5/GPT-2/OPT；真正低于 1B 的 decoder-only student 证据仍相对少。

## 5. 目前最适合本 FYP 的用法

现阶段不需要立刻复现论文 10–12 的复杂方法。可以把 1–9 作为主要实验参考，把 10–12 作为“近期趋势”放在 related work 的后半部分。下一次给导师汇报时，重点展示第 2、3 节两张表，并说明：

> 我先完成了可核查的文献事实整理，区分了原始题目数、每题 rationale 数和最终训练监督量。下一步会依据这张表界定一个规模适合 FYP 的研究问题，再决定是否需要增加模型规模或实验条件。

## 6. 本地论文文件对应关系

1. `1_CotBasic_FromLLM->LM_DIstill.pdf`
2. `2_SocraticCot.pdf`
3. `3_Disitill_Stepbystep_multitask->Siglerationale.pdf`
4. `4_SymblicCoT_Multi_rationales.pdf`
5. `5_MCC-KD_Mutiltationales_SameAnswer.pdf`
6. `6_Anaylsis_Cot.pdf`
7. `7_FineTuneCoT_MultiRationale.pdf`
8. `8_Mysteries_CoT_Distillation.pdf`
9. `9_MoRSD_RationaleSelection.pdf`
10. `10_MIND_MultiPerspective_CoTDistillation.pdf`
11. `11_RSR_InformativeAlignment.pdf`
12. `12_LALP_LocalScoring_RationaleSelection.pdf`
