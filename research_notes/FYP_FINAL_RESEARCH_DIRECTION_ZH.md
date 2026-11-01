# FYP最终研究方向与实验设计

> 2026-10-13 更新：本文以下保留的是旧 quantity/diversity/compatibility 方案，已不再是当前主实验。
> 当前工作标题为 **Where to Explain: Student-Guided Rationale Refinement for Reasoning Distillation**。
> 新协议见 [EXPERIMENT_PROTOCOL_REFINEMENT_ZH.md](</Users/starsfalling/Documents/FYP Mini Agent/EXPERIMENT_PROTOCOL_REFINEMENT_ZH.md>)。
> 新实验比较原始解答、普通补充、随机位置补充、同解释前置、student 引导局部补充，以及不按续写收益筛选的消融组。
> 旧实验结果作为前期可行性证据保留；新主实验尚未产生结果，不能沿用旧分数。

## 1. 最终研究主线

原来的Quantity和Diversity方向不需要推翻，但应统一到一个更明确的问题之下：

> **在相同的答案正确候选池、相同rationale数量和基本一致的训练预算下，应该如何为student选择rationale？rationale数量、词汇多样性和student compatibility分别有多重要？**

研究重点不再是笼统地问“多条rationale是否更好”，而是进一步区分：

1. 收益是否来自不同rationale的数量；
2. 收益是否来自rationale之间的词汇或表层多样性；
3. 收益是否来自rationale与student已有语言建模能力的匹配程度。

这样可以保留已经完成的Quantity和Diversity工作，同时把它们组织成一个统一的
**rationale selection**问题。

---

## 2. 建议研究题目

### 英文题目

> **Rationale Selection for Chain-of-Thought Distillation: A Controlled Study of Quantity, Lexical Diversity, and Student Compatibility**

### 中文题目

> **思维链蒸馏中的Rationale选择：数量、词汇多样性与Student Compatibility的受控研究**

标题不直接写“0.6B小模型”或“8B模型”，原因是模型是实验对象，而不是研究问题本身。
这种写法既能准确表达贡献，也避免模型调整后整个题目失效。

---

## 3. 最终模型选择

### Teacher

- 模型：`deepseek-ai/DeepSeek-R1`
- 作用：为每道GSM8K训练题生成多个候选rationale；
- 仅使用最终回答中的简洁、可读推导作为训练监督；
- 不把API返回的私有或冗长`reasoning_content`直接用于训练；
- 只保留最终答案与GSM8K gold answer一致、没有被截断且满足长度要求的rationale。

### Student

- 模型：`Qwen/Qwen3-8B-Base`
- 使用Base而不是Instruct或Thinking版本，减少已有指令微调和推理后训练对实验效应的干扰；
- 使用4-bit MLX权重和LoRA训练；
- 所有实验组从完全相同的base checkpoint开始；
- student compatibility必须使用训练前的Qwen3-8B-Base计算。

Qwen3-8B不应被严格描述成“十亿参数以下的小语言模型”。论文中更稳妥的说法是
`student model`或`compact student relative to the teacher`。

### 已完成的可行性验证

- Qwen3-8B-Base已成功转换为本地4-bit MLX模型；
- compatibility NLL计算已经通过端到端测试；
- 8B LoRA训练峰值内存约8.12GB，在32GB Apple Silicon设备上可以运行；
- adapter保存、重新加载、生成和答案评分链路均已跑通。

---

## 4. 最终Research Questions

### Main RQ

> **在训练问题、每题监督记录数和优化预算保持一致时，随机选择、词汇多样性选择和student-compatible选择中，哪一种rationale选择方式最有利于Qwen3-8B-Base的数学推理？**

英文：

> **Under a matched question-exposure and optimization budget, which rationale-selection strategy—random selection, lexical-diversity selection, or student-compatible selection—most effectively improves the mathematical reasoning performance of Qwen3-8B-Base?**

### RQ1：Quantity

> **在每题训练记录数相同时，使用三条不同rationale是否优于将同一条rationale重复三次？**

比较：

- `Single×3`：同一条正确rationale重复三次；
- `Random-3`：从同一候选池随机选择三条不同的正确rationale。

两组具有相同问题、相同训练记录数和相同optimizer updates，因此主要差别是每题包含的
**unique rationale数量**。

### RQ2：Lexical Diversity

> **当每题都使用三条rationale时，低多样、随机和高多样的rationale组合是否产生不同的数学推理效果？**

比较：

- `Low-diversity-3`
- `Random-3`
- `High-diversity-3`

这里研究的是`lexical/surface-form diversity`，不能直接称为“真实推理策略多样性”。

### RQ3：Student Compatibility

> **使用训练前student的长度归一化NLL选择更容易建模的rationale，是否优于随机选择或只追求词汇多样性的选择？**

比较：

- `Compatible-3`与`Random-3`；
- `Compatible-3`与`High-diversity-3`。

这是论文最核心的新增部分，因为它把研究从“rationale越多或越不同是否越好”推进到
“什么样的rationale更适合特定student”。

---

## 5. 核心实验组

每道题都从同一个DeepSeek-R1候选池中选择监督信号。

| 实验组 | 每题记录数 | 选择方式 | 主要作用 |
|---|---:|---|---|
| Answer-only×3 | 3 | 最终答案重复3次 | 检查CoT监督本身是否有价值 |
| Single×3 | 3 | 选择1条正确rationale并重复3次 | 单rationale基线 |
| Low-diversity-3 | 3 | 选择平均TF-IDF距离最低的3条 | 低表层多样性 |
| Random-3 | 3 | 固定随机种子选择3条 | 普通多rationale基线 |
| High-diversity-3 | 3 | 选择平均TF-IDF距离最高的3条 | 高表层多样性 |
| Compatible-3 | 3 | 选择student NLL最低的3条 | student-aware选择 |

对于196道最终训练题，每组均为：

- 相同的196道问题；
- 每题3条训练记录；
- 共588条训练记录；
- 相同问题曝光次数；
- 相同epoch和optimizer updates；
- 相同LoRA配置、batch size和训练种子集合。

Answer-only天然具有更少的target token，因此它是辅助基线。Quantity、Diversity和Compatibility
的主要因果比较应集中在包含rationale的组之间。

---

## 6. 候选Rationale池

### 生成方式

对每道GSM8K训练题：

1. 使用同一个DeepSeek-R1版本生成5条候选rationale；
2. 使用不同的策略提示和随机采样增加候选差异；
3. 提取每条候选的最终答案；
4. 仅保留答案正确、8–180词、未被截断的最终推导；
5. 正式实验优先要求每题具有5条可用候选；
6. 不满足候选池要求的题目从所有实验组同时排除，而不是只从某一组排除。

所有原始API响应、token usage、错误、候选ID和筛选结果都需要保存，从而保证实验可以审计和
断点续跑。

### 为什么必须使用同一候选池

如果不同实验组使用不同teacher请求生成的数据，那么teacher采样差异会与选择方法混在一起。
因此六组必须从每题同一个、已经冻结的答案正确候选池中选择。

---

## 7. Diversity选择方法

对每道题的5条正确rationale：

1. 使用unigram和bigram TF-IDF表示每条rationale；
2. 计算候选之间的cosine distance；
3. 5条选3条共有10种组合；
4. 计算每个组合中3条rationale的平均两两距离。

选择规则：

- 平均距离最小：`Low-diversity-3`；
- 固定种子随机组合：`Random-3`；
- 平均距离最大：`High-diversity-3`。

论文中必须明确：

> TF-IDF距离测量的是lexical或surface-form diversity，而不是对“不同真实解题策略”的直接证明。

如果最终出现`High > Random > Low`，可以支持表层多样性与效果相关；如果没有出现该顺序，
则应如实报告TF-IDF diversity不足以稳定预测蒸馏收益，而不能在看到测试结果后更换定义。

---

## 8. Student Compatibility选择方法

使用训练前的Qwen3-8B-Base，对每条正确候选rationale计算长度归一化负对数似然：

\[
\operatorname{NLL}(r\mid q)
=
-\frac{1}{|r|}
\sum_{t=1}^{|r|}
\log P(r_t\mid q,r_{<t})
\]

解释：

- NLL较低：student更熟悉这种表达和推导形式；
- NLL较高：student认为该rationale较陌生或较难建模。

对每道题选择NLL最低的3条，构成`Compatible-3`。

计算时必须满足：

- loss只覆盖assistant中的rationale token；
- 不计算最终答案token；
- 使用token-mean NLL进行长度归一化；
- 使用任何训练开始前的base student；
- 不使用dev、GSM8K test或SVAMP信息参与选择；
- 所有实验组确定后才进行student训练。

该方法不需要额外训练选择器，因此适合FYP规模，也能形成清晰的student-aware baseline。

---

## 9. “监督预算基本一致”的准确含义

### 完全一致的部分

- 相同训练问题；
- 相同teacher候选池；
- 每题3条监督记录；
- 相同训练记录数；
- 相同epoch和optimizer updates；
- 相同LoRA配置和batch size；
- 相同训练随机种子集合。

### 必须测量和报告的部分

- rationale平均词数；
- target token总量；
- 每条记录的token长度分布；
- 超过最大序列长度的比例；
- 各组平均TF-IDF距离；
- 各组被选rationale的平均student NLL。

如果rationale组之间target token总量差异小于约5%，可以写成`approximately token-matched`。
如果差异较大，只能称为record-matched和update-matched，并将token量作为潜在混杂变量报告，
必要时补充token-matched sensitivity analysis。

不能笼统声称所有六组“监督token预算完全相同”。

---

## 10. 数据范围与两阶段实验

### 阶段A：展示版Pilot

- GSM8K训练题：50道；
- 每题生成5条候选；
- 训练种子：7；
- 六个实验条件；
- 评测：GSM8K前100题和SVAMP前100题；
- 不运行MATH-500；
- 不做重复采样行为评测。

目的：

- 验证新teacher和student的完整流水线；
- 得到可用于阶段汇报的初步对比表；
- 检查Low、Random、High的多样性操作是否真正拉开；
- 检查Compatible选择的NLL是否显著低于其他组；
- 估计正式实验的训练和评测时间。

Pilot只能标记为`preliminary results`，不能作为最终论文结论。

### 阶段B：FYP主实验

- 固定约200道GSM8K源训练题；
- 目标得到196道满足候选池质量要求的共同训练题；
- 每组196×3＝588条训练记录；
- 训练种子：7、17、177；
- GSM8K官方完整test：1,319题；
- SVAMP完整固定测试集：1,000题；
- MATH-500、AIME、AMC暂不纳入主实验。

这样既保留原实验规模和可比性，也避免把FYP扩张成大规模benchmark工程。

---

## 11. 数据泄漏检查

按照老师要求，正式实验前必须完成：

1. 训练题与GSM8K test的标准化exact duplicate检查；
2. 训练题与SVAMP的标准化exact duplicate检查；
3. 使用character 3–5 gram TF-IDF cosine进行近重复检查；
4. 使用预先固定的0.90阈值标记可能的near duplicate；
5. 人工检查每个split最相似的约20–50对问题；
6. 确认任何测试题都没有用于rationale生成、筛选、模型选择或调参。

目前对初始冻结训练池的自动检查没有发现标准化完全重复或相似度达到0.90的近重复，但在最终
196道eligible question set确定后仍需重新运行并保存最终报告。

Teacher预训练阶段是否见过公开benchmark通常无法完全验证，应作为研究限制说明；但自己的
训练pipeline必须保证没有直接泄漏。

---

## 12. 评价与统计方法

### 主要评价

- GSM8K final-answer exact match；
- 三个训练种子的mean和sample standard deviation；
- paired bootstrap 95% confidence interval；
- invalid answer rate。

### 主要预注册比较

1. `Random-3 − Single×3`：不同rationale数量的作用；
2. `High-diversity-3 − Low-diversity-3`：多样性的总体作用；
3. `High-diversity-3 − Random-3`：主动多样性选择是否优于普通随机选择；
4. `Compatible-3 − Random-3`：student-aware选择是否有效；
5. `Compatible-3 − High-diversity-3`：compatibility是否优于纯表层多样性。

### 外部分布评价

- SVAMP exact match；
- 所有模型不在SVAMP上训练或选择；
- 比较各选择策略在GSM8K分布内性能和SVAMP泛化性能是否一致。

### 次要分析

- rationale平均输出长度；
- 无法提取答案的比例；
- calculation、reasoning、extraction和other错误类型；
- 如时间允许，对固定小样本进行prompt robustness和答案一致性分析。

---

## 13. 旧0.6B结果如何使用

此前结果为：

| 条件 | GSM8K前100 | SVAMP前100 |
|---|---:|---:|
| Single×3 | 32.7% ± 2.9 | 50.3% ± 4.6 |
| Random-3 | 32.3% ± 3.1 | 47.7% ± 4.5 |
| High-diversity-3 | **39.3% ± 2.3** | **52.7% ± 4.2** |

可以暂时解释为：

1. Random-3没有明显优于Single×3，因此rationale数量本身未必足够；
2. High-diversity-3在GSM8K上高于Random-3约7个百分点，说明选择方式可能比数量更重要；
3. SVAMP方向相似，但只有100题，证据不足；
4. 尚未测试Low-diversity和Compatible，因此无法判断多样性梯度或student compatibility。

但这些结果来自旧teacher、旧0.6B student和旧候选池，只能作为：

> **Preliminary 0.6B pilot evidence motivating the rationale-selection hypothesis.**

它们不能与新的DeepSeek-R1/Qwen3-8B结果直接合并，也不能作为新主实验的最终结果。

---

## 14. 当前Literature Gap

基于目前整理的12篇核心论文，可以采用以下谨慎表述：

> 现有研究已经分别探索了多rationale监督、rationale diversity和student-aware selection，
> 但较少在同一个答案正确候选池中，同时固定每题K值、训练问题、训练记录和优化步数，直接比较
> 随机选择、词汇多样性选择与简单student-compatible选择，并检验这些结论能否从GSM8K泛化到SVAMP。

英文：

> **Prior work has investigated multi-rationale supervision, rationale diversity, and student-aware selection, but relatively few studies directly compare random, lexical-diversity-based, and student-compatible selection from the same answer-correct candidate pool while controlling the number of rationales, question exposure, training records, and optimization steps.**

这是当前的`working gap`。最终论文中应在完成老师要求的系统文献对比表后，再确认“relatively few”
是否有足够证据，避免使用“no previous work”之类过强的表述。

---

## 15. 预期研究贡献

本项目不需要发明复杂的新算法，其贡献可以是严格、透明的受控实证研究：

1. 从同一个答案正确候选池出发，分离rationale数量与训练记录数；
2. 在固定K值下直接比较Low、Random和High lexical diversity；
3. 测试简单的base-student token-mean NLL能否作为有效的rationale选择指标；
4. 比较selection方法在GSM8K分布内性能和SVAMP外部分布泛化上的差异；
5. 系统报告训练记录、target token、多样性、compatibility和数据泄漏审计结果。

无论最终哪一组获胜，都可以形成有意义的结论：

- `Random > Single`：不同rationale数量本身有价值；
- `High > Low/Random`：表层多样性选择有价值；
- `Compatible > Random/High`：student-aware匹配比单纯追求多样性更重要；
- `High ≈ Compatible`：两种选择原则可能选择出同样有效的监督；
- 不同方法分别在GSM8K和SVAMP占优：分布内性能和OOD泛化可能需要不同选择原则；
- 所有差异都很小：在8B student上，选择策略的边际收益可能有限。

---

## 16. 明确不研究的内容

为了保证FYP能够完成，本项目暂不研究：

- student model size scaling；
- 7B、8B、14B等多个student同时比较；
- MIND、RSR、LALP等复杂选择算法的完整复现；
- teacher model之间的系统比较；
- AIME、AMC和完整MATH训练；
- embedding diversity、LLM judge strategy diversity等额外定义；
- Compatible与Diverse的复杂混合算法。

只有在六组主实验完成且时间充足时，才考虑增加`Compatible+Diverse`作为扩展实验。

---

## 17. 最终执行顺序

1. 完成50题展示版pilot，确认方向与代码正确；
2. 整理老师要求的系统文献对比表；
3. 根据pilot确定并冻结正式超参数；
4. 固定约200道源训练题和最终共同eligible question set；
5. 用DeepSeek-R1生成并冻结答案正确候选池；
6. 完成最终train–test leakage检查；
7. 用未训练Qwen3-8B-Base计算所有候选的compatibility NLL；
8. 构建六个实验条件；
9. 审计训练记录、token量、长度、多样性和NLL；
10. 使用种子7、17、177完成六组训练；
11. 最后统一运行完整GSM8K和SVAMP评测；
12. 计算mean、standard deviation、paired bootstrap CI并进行错误分析；
13. 根据主结果决定是否需要一个最小的补充实验。

---

## 18. 向老师汇报时的核心表述

可以将整个项目概括为：

> 我在文献回顾和前期实验后，决定不再笼统研究“多条rationale是否更好”，而是研究如何为
> student选择有效rationale。前期0.6B实验显示，三条随机rationale没有明显优于重复一条
> rationale，但高多样性选择取得了更好的初步结果，因此选择方式可能比数量本身更重要。
> 新实验使用DeepSeek-R1生成同一个答案正确候选池，固定Qwen3-8B-Base作为student，在相同
> 问题、每题三条记录和相同更新步数下，比较Single、Random、Low-diversity、High-diversity
> 和Compatible选择。第一阶段使用50题验证完整流程，正式实验计划使用约196题、三个种子，
> 并在完整GSM8K和SVAMP上评测。

---

## 19. 最终建议

> **保留Quantity和Diversity实验，但将FYP主线统一为“如何为student选择有效rationale”。使用DeepSeek-R1作为teacher、Qwen3-8B-Base作为固定student，先完成Low-diversity和Compatible两个关键条件，再在GSM8K与SVAMP上进行受控比较。旧0.6B结果作为研究动机，新8B实验作为正式证据。**

这个题目能够直接回应老师提出的“rationale diversity是否比quantity更重要”以及“CoT知识能否
有效蒸馏到较小student”两个方向，同时把研究范围控制在一个FYP可以完成的规模内。
