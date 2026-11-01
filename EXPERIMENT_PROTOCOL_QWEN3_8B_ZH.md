# Qwen3-8B rationale distillation 主实验协议

## 1. 研究问题

本实验不再把“多条 rationale”视为一个不可拆分的处理，而是从同一个教师候选池中区分：

1. **Quantity**：一题只有一条独特 rationale，和一题有三条独特 rationale 是否不同？
2. **Diversity**：同样三条 rationale 时，低多样、随机和高多样选择是否不同？
3. **Compatibility**：由未微调学生模型认为更容易生成（较低 token-mean NLL）的 rationale，是否比随机或纯 diversity 选择更适合学生？

主张的“公平预算”具体指：相同源题集合、每题相同训练记录数、相同 epoch、optimizer
update 与训练配置。各组实际 assistant token 数不一定完全相等，因此必须在训练前用同一 tokenizer
报告 token 总量和长度分布，不能把本实验表述成严格的等 token 实验。

## 2. 冻结模型

- 教师：`deepseek-ai/DeepSeek-R1`（SiliconFlow 当前对应 R1-0528）；
- 学生：`Qwen/Qwen3-8B-Base`，本机转换为 4-bit MLX 权重；
- 微调：MLX LoRA，所有实验组使用同一个 base checkpoint；
- 训练种子：7、17、177；先用 seed 7 在 dev 做流程验证，参数冻结后才跑其余种子和 test。

选择 Base 而非 Instruct/Thinking checkpoint，是为了减少学生已有 reasoning post-training
对监督信号差异的遮蔽。

## 3. 数据范围

- GSM8K 官方 train：7,473；固定 seed 7 划出 500 dev，留下 6,973 训练池；
- 教师生成固定使用训练池前 380 题，每题 5 条候选；
- GSM8K 官方 test：完整 1,319 题；
- SVAMP：完整 1,000 题，只做跨数据集评测；
- MATH-500：数据和评分程序已准备，但因本轮预算暂缓，不纳入第一阶段结果；
- 不从 dev、test、SVAMP 或 MATH-500 生成训练监督。

这里“完整数据集”指第一阶段完整评测 GSM8K test 和 SVAMP，不再截取前 100 题。主训练集固定为 380 题，是本机
32GB Apple Silicon 上可完成六组 8B 消融的研究设计，不声称覆盖全部 6,973 道训练题。

## 4. 教师生成与过滤

- 每题请求 5 条候选，使用 5 个策略提示；
- 温度 0.6，top-p 0.95，thinking budget 4096，final-response limit 768；
- 仅使用 `content` 中简洁、自包含的最终推导，不把私有 `reasoning_content` 当监督；
- 最后一行必须能提取出 `Final answer`；
- 只保留答案等于 GSM8K gold、8–180 词且 `finish_reason != length` 的候选；
- 一题少于 3 条可用候选时，从所有组同时排除；
- API 原始返回、token usage、错误与选择 ID 全部保存，可断点续跑。

5×5 pilot 得到 23/25 条可用候选，5/5 题均达到至少三条；两条失败来自 API 超时，没有
数学错误或长度截断。根据后续 85 条成功响应的实际用量，380×5 的生成约需人民币 39.5 元；
扣除已生成部分后预计还需约 37.7 元，含约 5% 重试按 39.6 元预留。

## 5. 六个训练条件

所有保留题目在每组都形成 3 条记录：

| 条件 | 每题三条记录的内容 | 控制目的 |
|---|---|---|
| Answer-only | 同一 final answer ×3 | 无 rationale 基线 |
| Single | 同一条正确 rationale ×3 | 只有一条独特 rationale |
| Low-diversity | 候选池中平均 TF-IDF 距离最低的三条 | 三条但策略相似 |
| Random-multi | 固定随机种子选择三条 | 普通多 rationale 基线 |
| High-diversity | 平均 TF-IDF 距离最高的三条 | 纯 diversity 选择 |
| Compatible | base student token-mean NLL 最低的三条 | student–rationale compatibility |

Low/High 使用候选组合的精确枚举，而非贪心近似。构建报告必须验证 Low < Random < High
的总体距离操作是否成立；若随机组偶然不在两者之间，保留真实结果，不事后改选择规则。

## 6. 训练顺序

1. 完成教师生成；
2. 用冻结的未训练 Qwen3-8B-Base 为每条可用 rationale 计算 token-mean NLL；
3. 构建六组数据并运行数据审计、泄漏审计和 tokenizer token 审计；
4. seed 7 六组短训练 smoke test；
5. seed 7 完整训练，在 dev 只检查明显故障，不根据 test 调参；
6. 冻结配置，完成 seed 17、177；
7. 第一阶段最后一次性评测完整 GSM8K test 与 SVAMP；MATH-500 留到余额和时间允许时再跑。

当前 LoRA 配置：3 epochs、micro-batch 1、gradient accumulation 8、rank 16、前 16 层、
max sequence length 1024。若 8B smoke test 因内存失败，只允许统一减少可训练层数或 rank，
并在任何正式 condition 训练前重新冻结配置。

## 7. 指标与比较

主要指标：

- GSM8K test exact match；
- 三种子 mean、sample standard deviation；
- paired bootstrap 95% CI。

主要预注册比较：

1. Random-multi − Single（quantity）；
2. High-diversity − Low-diversity（diversity）；
3. Compatible − Random-multi（学生适配选择）；
4. Compatible − High-diversity（compatibility 是否超越纯 diversity）。

次要结果包括 SVAMP exact match、invalid-answer rate、生成 rationale 长度、随机采样答案
一致性和 prompt robustness。生成上限固定为 256 token、评测 batch size 为 4；行为评价只用
每个 split 的前 50 题、每题 3 次随机采样。MATH-500 的 LaTeX/SymPy 等价评分实现保留为后续扩展。

## 8. 泄漏控制

训练题与 dev/GSM8K test/SVAMP（以及已准备但暂缓评测的 MATH-500）分别执行：

- NFKC + 仅保留字母数字后的 exact duplicate 检查；
- character 3–5 gram TF-IDF cosine 近重复检查，预警阈值 0.90；
- 保存每个 split 的 top-20 最近邻题对供人工复核。

初始 500 题冻结池的审计结果为四个 split 均 0 exact、0 个相似度 ≥0.90 的近重复。
构建最终 eligible question set 后必须重新运行同一审计。

## 9. 不能越界的结论

- 结果只支持“在本模型、本候选生成方法和本预算下”的因果比较；
- 不能把 380 训练题称作 full GSM8K training；
- 不能预设最终准确率应达到 90%；
- 如果 High-diversity 不优于 Random，结论应是“表面词汇 diversity 不足以保证收益”，而不是
  临时更换 diversity 定义；
- MATH-500 是难度迁移测试，不是本研究训练分布内的主要性能指标。
