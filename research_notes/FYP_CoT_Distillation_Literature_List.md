# FYP Literature List: CoT/Rationale Distillation for Small Language Models

更新日期：2026-10-08  
主题：大语言模型向小语言模型的 Chain-of-Thought（CoT）/rationale distillation，重点关注多 rationale、rationale diversity 和 student-aware selection。

## 1. 文献文件清单

### A. 原有6篇

| 编号 | 论文 | 年份/来源 | 本地文件 |
|---:|---|---|---|
| 1 | [Teaching Small Language Models to Reason](https://aclanthology.org/2023.acl-short.151/) | ACL 2023 Short | `1_CotBasic_FromLLM->LM_DIstill.pdf` |
| 2 | [Distilling Reasoning Capabilities into Smaller Language Models](https://arxiv.org/abs/2212.00193)（Socratic CoT） | arXiv 2022/2023 | `2_SocraticCot.pdf` |
| 3 | [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://aclanthology.org/2023.findings-acl.507/) | Findings of ACL 2023 | `3_Disitill_Stepbystep_multitask->Siglerationale.pdf` |
| 4 | [Symbolic Chain-of-Thought Distillation: Small Models Can Also “Think” Step-by-Step](https://aclanthology.org/2023.acl-long.150/)（SCoTD） | ACL 2023 | `4_SymblicCoT_Multi_rationales.pdf` |
| 5 | [MCC-KD: Multi-CoT Consistent Knowledge Distillation](https://aclanthology.org/2023.findings-emnlp.454/) | Findings of EMNLP 2023 | `5_MCC-KD_Mutiltationales_SameAnswer.pdf` |
| 6 | [Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning](https://aclanthology.org/2025.findings-acl.782/) | Findings of ACL 2025 | `6_Anaylsis_Cot.pdf` |

### B. 上一轮新增6篇

| 编号 | 论文 | 年份/来源 | 本地文件 |
|---:|---|---|---|
| 7 | [Large Language Models Are Reasoning Teachers](https://aclanthology.org/2023.acl-long.830/)（Fine-tune-CoT） | ACL 2023 | `7_FineTuneCoT_MultiRationale.pdf` |
| 8 | [Investigating Mysteries of CoT-Augmented Distillation](https://aclanthology.org/2024.emnlp-main.349/) | EMNLP 2024 | `8_Mysteries_CoT_Distillation.pdf` |
| 9 | [Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales](https://aclanthology.org/2025.findings-emnlp.413/)（MoRSD） | Findings of EMNLP 2025 | `9_MoRSD_RationaleSelection.pdf` |
| 10 | [MIND: From Passive Mimicry to Active Reasoning through Capability-Aware Multi-Perspective CoT Distillation](https://aclanthology.org/2026.acl-long.2020/) | ACL 2026 | `10_MIND_MultiPerspective_CoTDistillation.pdf` |
| 11 | [Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](https://aclanthology.org/2026.acl-long.1950/)（RSR） | ACL 2026 | `11_RSR_InformativeAlignment.pdf` |
| 12 | [The Signal is in the Steps: Local Scoring for Reasoning Data Selection](https://proceedings.mlr.press/v306/just26a.html)（LALP） | ICML 2026 | `12_LALP_LocalScoring_RationaleSelection.pdf` |

### C. 本轮新增6篇

| 编号 | 论文 | 年份/来源 | 本地文件 |
|---:|---|---|---|
| 13 | [SCOTT: Self-Consistent Chain-of-Thought Distillation](https://aclanthology.org/2023.acl-long.304/) | ACL 2023 Outstanding Paper | `13_SCOTT_Faithful_CoT_Distillation.pdf` |
| 14 | [Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html)（KARD） | NeurIPS 2023 | `14_KARD_Knowledge_Augmented_Reasoning_Distillation.pdf` |
| 15 | [Improve Student’s Reasoning Generalizability through Cascading Decomposed CoTs Distillation](https://aclanthology.org/2024.emnlp-main.875/)（CasCoD） | EMNLP 2024 | `15_CasCoD_Reasoning_Generalization.pdf` |
| 16 | [Logic Consistency Makes Large Language Models Personalized Reasoning Teachers](https://doi.org/10.1109/IJCNN64981.2025.11229110)（PeCoTD） | IJCNN 2025 | `16_PeCoTD_Personalized_Reasoning_Teacher.pdf` |
| 17 | [Capture the Key in Reasoning to Enhance CoT Distillation Generalization](https://aclanthology.org/2025.acl-long.21/)（EDIT） | ACL 2025 | `17_EDIT_Key_Reasoning_Step_Distillation.pdf` |
| 18 | [Skip-Thinking: Chunk-wise Chain-of-Thought Distillation Enable Smaller Language Models to Reason Better and Faster](https://aclanthology.org/2025.emnlp-main.610/) | EMNLP 2025 | `18_SkipThinking_Chunkwise_CoT_Distillation.pdf` |

以上18篇 PDF 均存放在：`/Users/starsfalling/Desktop/Resource/FYP/`。

## 2. 统一文献对比

### 2.1 Teacher、student、训练数据与结果

阅读说明：`训练数据/监督规模`优先报告论文明确给出的raw questions、rationales-per-question和总paths；论文没有报告最终过滤后记录数时明确写“未报告”，不把数据集总规模冒充实际训练记录数。`代表性结果`只用于理解每篇论文内部的增量，不应跨论文直接排名，因为split、prompt、解码和评测协议不同。

| 编号 | Teacher | Student及规模 | 训练数据/监督规模 | Rationale supervision | 数据集 | 代表性结果 |
|---:|---|---|---|---|---|---|
| 1 | PaLM 540B；GPT-3 175B | T5 Small/Base/XL/XXL，约60M–11B | GSM8K：原训练6,725题，过滤后5,337条有效CoT；MAWPS 1,590；ASDiv 1,544 | 每题一条答案正确的teacher CoT；用gold answer辅助生成并过滤 | GSM8K、MAWPS、ASDiv，另有常识和符号推理 | GSM8K上T5-XXL从8.11%升至21.99%（PaLM CoT）或18.42%（GPT-3 CoT） |
| 2 | GPT-3 175B | GPT-2 124M/355M/774M/1.5B | 使用各benchmark训练split；GSM8K约7.5k题 | 普通CoT或Socratic分解：生成子问题，再由单模型或“分解器+求解器”完成 | GSM8K、StrategyQA、SVAMP | GSM8K GPT-2 Large：answer-only 4.62%、CoT 12.85%、Socratic CoT 17.89% |
| 3 | PaLM 540B；GPT-NeoX 20B消融 | T5 220M/770M/11B | 依任务使用不同比例训练数据 | 多任务学习：同时预测label和生成rationale；通常每题一条 | e-SNLI、ANLI、CommonsenseQA、SVAMP | 770M T5在一个benchmark上只用80%数据即可超过few-shot PaLM 540B |
| 4 | GPT-3/code-davinci-002 175B | OPT 125M–1.3B | 默认每题采样30条；选择策略实验固定为5条 | 比较random、SBERT diversity、teacher likelihood、open-endedness | CommonsenseQA、QuaRel、OpenBookQA | 从1条增至30条时，CSQA 53.0→60.2、QuaRel 62.2→73.4、OBQA 39.0→44.4；random selection也很强 |
| 5 | GPT-3.5-Turbo | Flan-T5 3B/11B；LLaMA 7B/13B | 各任务训练split；每题5条候选 | 用3-gram Jaccard促进diversity，并用双向KL保持不同rationale的答案分布一致 | GSM8K、SVAMP、ASDiv、CommonsenseQA等 | LLaMA-7B在GSM8K约38.01%→41.58%；但diversity与consistency loss同时改变 |
| 6 | GPT-4o、Gemini-1.5-Flash、LLaMA-3-70B、人类CoT | BLOOM 560M/1.1B/1.7B/3B、Gemma 2B、LLaMA-3.2 1B/3B | 各benchmark训练split；每题最多6级granularity，并比较多种格式 | 系统研究granularity、format和teacher，不是固定K的diversity实验 | SVAMP、GSM8K、AQuA-RAT、MATH、CSQA、OBQA、StrategyQA | 弱student可能更适合简单CoT；格式影响较小；更强teacher不一定产生更强student |
| 7 | InstructGPT/GPT-3 175B | GPT-3 0.3B/1.3B/6.7B；另测T5、Flan-T5、GPT-2 | 每题生成1–64条reasoning explanations | 多次采样并按最终答案过滤，论文称为diverse reasoning | MultiArith、SVAMP、GSM8K等 | 6.7B student在MultiArith约+26%、SVAMP约+17%；但数量与diversity同步增加 |
| 8 | Mistral-7B-Instruct | GPT-2、Phi-1.5、Gemma-2B | 使用CommonsenseQA、OpenBookQA、QuaRel训练split | 比较rationale放在label前/后、打乱token、屏蔽token、只留关键token | CommonsenseQA、OpenBookQA、QuaRel | label-first/rationale-after可能更好；打乱CoT仍可能带来收益；少量关键token可接近完整CoT效果 |
| 9 | GPT-3/text-davinci-002 175B | Flan-T5 Small/Base/Large/XL，约60M/250M/780M/3B | 先生成多条；正确性过滤后保留K=6，再按Rationale Difficulty选k=3 | 联合accuracy、3-gram diversity和student-specific difficulty | SingleEq、AddSub、MultiArith、StrategyQA、SVAMP、Date Understanding、Shuffled Objects等 | 七个数据集平均比baseline高4.6%，并使用更少rationales；不同选择因素仍相互耦合 |
| 10 | Qwen3-235B | Qwen2.5 1.5B/7B、Llama3.1 8B | 497个MATH样本，每题8种视角；扩展实验达到4,000–16,000 paths | 8种显式策略，并使用student feedback动态融合 | MATH500、GSM8K、SVAMP；OOD：CSQA、StrategyQA、GPQA-D | Qwen2.5-1.5B：GSM8K 72.81→76.50，SVAMP 79.33→82.83；直接混合8种路径反而明显退化 |
| 11 | 11个reasoning teachers，4B–671B | Qwen3 14B/4B、Llama3.1 8B、Qwen2.5 7B/3B | 5,000道数学题；构造55个teacher-student pairs，每题最终选择一条trajectory | Rank-Surprisal Ratio同时考虑informativeness和student alignment | AIME 2024/2025、AMC 2023、MATH500，另测GPQA | RSR与训练后性能平均Spearman相关0.86；选择结果优于random、长度、质量和perplexity标准 |
| 12 | 多个不同teacher模型 | 多个student模型 | 从多个teacher生成的长推理候选中选择训练数据 | LALP用局部窗口评价每个reasoning step是否能从前提推出，而非给整条trajectory打分 | 数学、代码、科学推理 | 整体likelihood在多teacher场景会错误排序；局部step-level scoring能更稳定地选出有效训练数据 |
| 13 | GPT-NeoX 20B | T5-Base 220M、T5-Large 770M、T5-3B；主实验T5-3B | 使用CSQA、StrategyQA、CREAK、QASC官方训练集并重新划分train/dev；论文未给最终样本条数 | gold-answer-conditioned contrastive decoding生成更支持答案的rationale；再加入counterfactual rationale-answer objective | CSQA、StrategyQA、CREAK、QASC | T5-3B中SCOTT主要提升faithfulness而非峰值accuracy；例如StrategyQA约60.8%，与普通KD约61.5%接近，但LAS由约19.4升至29.4 |
| 14 | GPT-3.5-turbo | Flan-T5 250M/780M/3B；T5 250M/780M/3B；OPT 350M/1.3B-IML | MedQA-USMLE共12,723题、StrategyQA共2,780题、OpenBookQA共5,957题；用官方split或StrategyQA 7:3；每题3/5/10条rationale消融 | 多条correct rationale + Wikipedia检索知识 + student reranker；先生成rationale再答案 | MedQA-USMLE、StrategyQA、OpenBookQA | MedQA上Flan-T5-250M：reasoning distillation 31.03→KARD 38.15；K=3/5/10时为30.09/32.13/32.91，显示数量收益趋于饱和 |
| 15 | GPT-3.5-turbo-0613 | TinyLLaMA 1.1B、LLaMA2 7B/13B；另测CodeLLaMA-7B、LLaMA3-8B、Mistral-7B | BBH共6,511条，按每个子任务4:1划分，约5.2k训练条；另做12.5%–100%数据量消融 | 将标准q→rationale+answer拆成级联的q→rationale与q+rationale→answer，弱化question-answer shortcut | ID：BBH-test；OOD：BB-sub、AGIEval、ARC-E/C | LLaMA2-7B平均accuracy：Std-CoT 41.8→CasCoD 49.6；BBH-test 54.2→59.4，BB-sub 28.7→37.0 |
| 16 | 数学任务text-davinci-002；其余Qwen2.5-7B-Instruct | GPT-2 124M/354M/774M/1.6B；T5 60.5M/222M/737M | 各benchmark训练split；每题采样8条；SLC选择高/随机/低；PeCoTD每轮只重生成100题，3轮 | Self-Logic Consistency衡量student-rationale匹配，并以高SLC样例提示teacher迭代改写；另复现5/30、10/30与all-30 diversity比较 | SingleEq、AddSub、MultiArith、GSM8K、SVAMP、Date、Last Letter、Coin Flip、CSQA、StrategyQA | T5-Large上高SLC相对低SLC：Date 76.88→84.08、CSQA 65.36→67.40、StrategyQA 58.87→60.94；但sub-1B模型在数学题上仍接近随机 |
| 17 | GPT-3.5-turbo-0613 | TinyLLaMA 1.1B、LLaMA2 7B/13B；另测CodeLLaMA-7B、LLaMA3-8B、Mistral-7B | BBH共6,511条、每子任务4:1划分，约5.2k训练条；保留teacher正确和错误CoT并构造paired dual-CoTs | EDIT：纠正teacher错误CoT、破坏正确CoT形成正负对；用最小编辑距离定位约4.7%的关键步骤并加权学习 | ID：BBH-test；OOD：BB-sub、AGIEval、ARC-E/C | LLaMA2-7B平均accuracy：Std-CoT 41.8→EDIT 46.5；BBH-test 54.2→60.9，BB-sub 28.7→31.1 |
| 18 | text-davinci-002 175B；补充实验GPT-4o-mini、LLaMA3.1-70B | GPT-2 124M/355M/774M；T5 60M/220M/700M；LLaMA3.2 1B/3B；Qwen2.5-7B | 按Fine-tune-CoT划分7个benchmark；teacher每题生成多样rationale，最长128 tokens；论文未统一报告记录总数 | CWT将rationale分为语义chunk逐块训练；STT只外显关键chunk，控制长CoT中核心token被稀释和推理延迟 | SingleEq、AddSub、MultiArith、SVAMP、Tracking Shuffled Objects、Last Letter、StrategyQA；另测GSM8K | GPT-2-124M七任务平均约25.7→35.3，T5-60M约27.4→35.3；LLaMA3.2-1B在GSM8K 52.23→55.26，同时缩短推理输出 |

### 2.2 它们分别回答了什么

| 研究问题 | 最相关论文 | 已知结论 | 尚未解决的问题 |
|---|---|---|---|
| CoT能否蒸馏给小模型？ | 1、2、3、7 | 可以；通常明显优于answer-only或普通fine-tuning | 提升究竟来自推理结构，还是更丰富的训练token/特征？ |
| 多条rationale是否比一条更好？ | 4、7 | 通常更多样本带来提升 | 多数实验同时增加rationale数量、token数和优化曝光，因果关系不干净 |
| diversity是否重要？ | 4、5、9、10 | 可能重要，但random也可能很强；naive diversity甚至可能伤害小模型 | lexical diversity、strategy diversity、student compatibility没有被干净分离 |
| rationale应该如何组织？ | 2、3、6、8 | 分解、多任务、granularity和label顺序都会影响结果 | accuracy提升是否真的代表test-time reasoning仍不确定 |
| 如何选择适合student的rationale？ | 6、9、10、11、12 | student-aware、difficulty-aware、local-step-aware selection优于只看teacher强弱或整体概率 | 对sub-1B模型、短CoT和低资源数学题的结论仍不充分 |
| 更多/更强是否总是更好？ | 4、6、9、10、11 | 不是；更强teacher、更长CoT、更多diverse paths都可能不适合弱student | 需要在固定预算下建立“帮助—饱和—冲突”的边界 |
| rationale是否真的支持答案？ | 13 | SCOTT表明普通teacher CoT可能空洞或与答案不一致；faithfulness可在不明显提高accuracy时改善 | 你的实验若只看exact match，无法判断学生利用了rationale还是只学到表面相关性 |
| OOD差距来自什么训练结构？ | 15、17 | CasCoD减少question-answer shortcut；EDIT强调teacher错误中暴露的关键步骤 | 这些方法没有把surface diversity、strategy diversity和student compatibility做成同一受控比较 |
| 长CoT为什么可能伤害小模型？ | 8、17、18 | 完整文本并非每个token都同等有用；关键步骤只占少量，长rationale会稀释其梯度并增加延迟 | 固定record/update仍不等于固定target-token预算；这是现有实验和本FYP都要补的控制 |
| student compatibility是否比diversity重要？ | 9、10、11、12、16 | difficulty、SLC、alignment和local score都支持student-aware选择 | 现有指标、模型、任务不同，尚无sub-1B数学任务中的同池同预算head-to-head比较 |

## 3. 文献之间的逻辑发展

### 第一阶段：证明CoT可以被蒸馏

- 论文1说明大teacher生成的CoT能够提升小T5的推理表现。
- 论文2说明将问题分解为Socratic subquestions可能比普通线性CoT更适合小模型。
- 论文3说明rationale不仅是输出文本，还可以作为单独的multi-task supervision，提高数据效率。

### 第二阶段：增加多条rationale

- 论文4和7发现每题生成多条rationale通常有效。
- 但论文4同时发现random selection表现很强，已经暗示收益未必来自“语义多样性”，也可能来自更多数据。
- 论文5进一步要求不同rationale得到一致答案，但diversity与consistency objective被绑定，无法分别归因。

### 第三阶段：分析什么rationale真正有用

- 论文6说明弱student并不总能吸收更细、更复杂的CoT。
- 论文8进一步挑战“模型是在学习完整推理过程”的解释：打乱token或只保留关键token仍可能有效。
- 论文9开始依据正确性、多样性和student difficulty选择更少但更合适的rationale。

### 第四阶段：从diversity转向student-aware selection

- 论文10表明显式策略多样性本身不够；对小模型直接混合冲突路径可能造成退化。
- 论文11提出informativeness与alignment之间的平衡。
- 论文12将分析单位进一步从整条trajectory下沉到局部reasoning steps。

### 第五阶段：从“选哪条”转向“为什么可学”

- 论文13提醒：答案正确并不保证rationale忠实，accuracy也不能替代faithfulness分析。
- 论文15说明训练目标本身会产生question-answer shortcut，OOD结果是必要证据。
- 论文16直接研究student-rationale distribution gap，说明“最diverse”不一定等于“最适合student”。
- 论文17和18进一步表明真正有用的信号可能集中在少量关键步骤；完整长CoT还会造成核心token梯度被稀释。

整体发展方向是：

> **single CoT → multiple CoTs → diverse/consistent CoTs → faithful/student-compatible trajectories → key, locally learnable reasoning steps**

## 4. 对本FYP最重要的比较

### 4.1 你的实验和已有论文的对应关系

| 你的条件 | 最接近的已有研究 | 主要区别 |
|---|---|---|
| Answer-only | 论文1、4中的label-only baseline | 你在0.6B Qwen上测试数学生成，而不是T5/OPT常识选择题 |
| Single rationale重复3次 | 论文1、2、3的单rationale监督 | 你匹配了每题训练记录数，使其能与multi-rationale更公平比较 |
| Random multi-rationale | 论文4、7 | 你固定为K=3并保持相同问题数和optimizer updates |
| TF-IDF diverse multi-rationale | 论文4、5、9 | 你当前衡量的是surface-form diversity，而不是明确的数学strategy diversity |
| 下一步low-diversity 3 | 现有文献中较少见的直接控制组 | 可以建立low→random→high的diversity梯度，是回答主问题的关键 |
| 下一步strategy-diverse 3 | 论文10的multi-perspective思想 | 你的目标是做简单受控消融，而不是引入复杂MetaNet |
| 下一步student-compatible 3 | 论文9、11、12、16 | 用0.6B student自身的perplexity/difficulty或简化SLC从同一候选池选择，而不是只看teacher/文本距离 |
| 下一步token-matched controls | 论文8、17、18 | 除records和updates外，再匹配总target tokens与长度分布，排除“多看了更多文字”的解释 |
| 下一步key-step coverage分析 | 论文12、17、18 | 检查高diversity组是否只是更容易覆盖关键算式/转换步骤，而非diversity本身直接起作用 |

### 4.2 现有实验结果

Student：Qwen3-0.6B-Base-4bit；Teacher：DeepSeek-V4-Flash；196道GSM8K训练题；每题5条候选，选择3条；3个seeds。

| 条件 | GSM8K前100 | SVAMP前100 |
|---|---:|---:|
| Base student | 3.0% | 2.0% |
| Answer-only | 0.7% ± 0.6 | 7.0% ± 5.6 |
| Single rationale | 32.7% ± 2.9 | 50.3% ± 4.6 |
| Random multi-rationale | 32.3% ± 3.1 | 47.7% ± 4.5 |
| Diverse multi-rationale | **39.3% ± 2.3** | **52.7% ± 4.2** |

关键比较：

- GSM8K：diverse − random multi = **+7.0 pp**，95% CI [1.3, 12.7]；
- SVAMP：diverse − random multi = **+5.0 pp**，95% CI [−1.0, 11.0]；
- random multi没有优于single：GSM8K为−0.3 pp，SVAMP为−2.7 pp。

因此当前结果支持：

> More rationales are not automatically better; rationale selection appears more important than rationale count.

它还不能证明：

- lexical diversity等于真实reasoning-path diversity；
- diverse rationale在所有任务上普遍有效；
- accuracy提升必然代表reasoning被内化；
- 性能应该接近90%。

## 5. 18篇是否足够，以及最终research gap

### 5.1 是否足够

这18篇已经足以支撑一个**聚焦于小模型数学CoT蒸馏的FYP literature review和实验设计**。它们覆盖：基础CoT蒸馏、多rationale、diversity/consistency、训练机制分析、student-aware selection、faithfulness、OOD泛化、关键步骤与长CoT效率。

它们并不等于“CoT全部文献”，但继续无边界扩充清单的边际价值已经较低。下一步应以这18篇为core set，通过它们的references/citations只补充与最终实验直接竞争的论文。

### 5.2 已经不能再声称的gap

不应写成：

> Few studies have investigated multiple or diverse rationales.

原因是：

- 论文4、7、14已经研究rationale数量和数量饱和；
- 论文4和16已经比较random与embedding-based diversity，并包含固定数量选择；
- 论文9、10、11、12、16已经研究student-aware或step-aware选择；
- 论文8、17、18已经研究关键token/步骤、rationale长度和训练机制。

因此，单纯研究“more rationales是否更好”或“diverse rationales是否更好”，创新性不足。

### 5.3 可以成立的精确gap

建议写成：

> **Prior work has separately studied rationale quantity, embedding-based diversity, student-aware selection, and key-step supervision. However, within the literature reviewed here, we did not identify a controlled comparison for sub-billion-parameter decoder-only models on mathematical reasoning that selects rationales from the same correct candidate pool while jointly controlling the number of unique rationales, question exposure, target-token/length budget, training records, and optimization steps. Consequently, it remains unclear whether the observed benefit of a “diverse” set is caused by surface-form variation, genuinely different solution strategies, greater coverage of key reasoning steps, or better compatibility with the student model.**

对应中文：

> 现有工作已经分别研究了rationale数量、基于embedding的多样性、student-aware选择和关键步骤监督；但在本次纳入的文献中，我们尚未发现一个面向十亿参数以下decoder-only数学模型的统一受控比较：从同一个答案正确的候选池中选择rationale，同时控制不同rationale数量、题目曝光、target-token/长度预算、训练记录数和优化步数。因此，所谓“diverse set”的收益究竟来自表面表达变化、真正不同的解题策略、对关键推理步骤更充分的覆盖，还是与student能力更匹配，仍不清楚。

这比旧版gap更窄，但也更能经受老师追问。论文4和16意味着“固定K比较random与embedding diversity”本身也不是新贡献；你的贡献必须进一步落在**同池、同预算、不同选择信号的因果比较与机制解释**上。

## 6. 推荐研究问题与实验设计

### 6.1 主研究问题

> **RQ：在相同候选池、K值和监督预算下，surface-form diversity、reasoning-strategy diversity与student–rationale compatibility中，哪一种选择信号最能提升0.6B模型的数学推理与OOD泛化；其收益能否由关键推理步骤覆盖率解释？**

对应英文：

> **Under a matched candidate pool, rationale count, and supervision budget, which rationale-selection signal—surface-form diversity, reasoning-strategy diversity, or student–rationale compatibility—most improves the mathematical reasoning and OOD generalization of a 0.6B model, and can the gains be explained by coverage of key reasoning steps?**

### 6.2 子问题

- **SQ1（复现）:** 当前TF-IDF high-diversity相对random的GSM8K收益，在完整评测集、更多训练题和严格长度匹配后是否仍成立？
- **SQ2（机制）:** TF-IDF/SBERT距离是否真的对应不同解题策略，还是只对应措辞变化？
- **SQ3（选择）:** 基于student loss/difficulty或SLC近似的compatibility选择，是否比maximum-diversity更稳定？
- **SQ4（解释）:** 哪个条件覆盖更多正确关键步骤；该指标能否解释ID和OOD accuracy差异？
- **SQ5（次要）:** 固定最佳选择原则后，K=1/2/3是否呈现收益饱和？Quantity应作为次要消融，而不是论文主gap。

### 6.3 最小可行实验矩阵

所有条件从**同一批、答案正确且通过格式检查的5条候选rationale**中选择，主实验固定K=3：

| 条件 | 选择规则 | 回答的问题 |
|---|---|---|
| Single-repeat | 1条rationale重复3次 | unique rationale数量是否重要 |
| Random-3 | 随机选3条 | 无选择策略基线 |
| Lexical-diverse-3 | TF-IDF或SBERT最大距离 | 表面/语义文本差异是否有效 |
| Strategy-diverse-3 | 按算式、解法步骤或人工/规则strategy标签选择 | 真正解题策略差异是否有效 |
| Student-compatible-3 | 依据0.6B student的长度归一化loss、difficulty或SLC proxy选择 | 匹配student能力是否更重要 |
| Low-diversity-3 | 选择最相似的3条 | 构造diversity梯度并排除“随机波动” |

必要控制：相同训练题、K、records、updates、epochs、seed集合与生成候选池；通过约束选择使总target tokens和长度分布尽量匹配，并逐条件报告实际token数，而不是只声称“预算相同”。

### 6.4 评测与报告

- Student主模型：Qwen3-0.6B；先不要把多模型规模设为主变量。
- ID：完整GSM8K test；OOD：完整SVAMP，并可加入ASDiv。
- 至少3个seeds；报告mean、standard deviation、paired difference与bootstrap confidence interval。
- 除exact match外，报告rationale长度、输出token数、valid-answer rate、关键步骤/算式覆盖、student loss或SLC proxy。
- 当前“196训练题 + 每个评测集前100题”只能作为pilot，不能作为最终结论。
- 在扩大实验前先完成train/eval exact、lexical与semantic overlap审计，并记录所有prompt、split、seed和候选生成日志。

## 7. 推荐阅读顺序

如果时间有限，按下面顺序读：

1. **论文1**：最基本的CoT distillation。
2. **论文4**：quantity、固定数量diversity selection及其早期结论。
3. **论文7和14**：多rationale规模扩大、数量饱和与知识增强。
4. **论文16**：直接理解diversity与student-rationale distribution gap。
5. **论文8**：为什么accuracy提升不能直接解释成完整reasoning transfer。
6. **论文13**：为什么答案正确不代表rationale faithful。
7. **论文9、10、11、12**：student-aware、trajectory/step-level selection。
8. **论文17、18**：关键步骤、长CoT token稀释和效率机制。
9. **论文15**：OOD泛化与question-answer shortcut。
10. **论文2、3、5、6**：结构化分解、多任务监督、一致性、granularity与teacher/student capacity。

与本FYP主gap最直接、必须精读的是：**4、8、9、11、12、13、14、15、16、17、18**。其中论文4、14、16是quantity/diversity的主要先行工作；论文9、11、12、16是compatibility/selection的主要先行工作；论文8、13、17、18提供机制解释。

## 8. 术语统一

| Canonical term | 中文解释 | 本项目使用方式 |
|---|---|---|
| chain-of-thought (CoT) | 思维链/逐步推理文本 | 首次写全称，之后使用CoT |
| rationale | 解释最终答案的推理依据或步骤 | 保留英文rationale，避免与一般“理由”混淆 |
| rationale quantity | 每题不同rationale的数量 | 与重复次数和总record数分开报告 |
| surface-form diversity | 词汇或文本形式多样性 | TF-IDF、n-gram、SBERT距离属于这一类 |
| reasoning-strategy diversity | 解题策略/推理路径多样性 | 不应仅由TF-IDF距离推断 |
| student–rationale compatibility | rationale与student学习能力的匹配程度 | 可由difficulty、perplexity、RSR或local score近似 |
| in-distribution (ID) | 同分布评测 | 本项目中的GSM8K test |
| out-of-distribution (OOD) | 分布外迁移评测 | 本项目中的SVAMP/ASDiv |
| supervision budget | 监督预算 | 包括题目数、unique rationales、target tokens、records和updates |
