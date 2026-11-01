# Student-guided rationale refinement：新主实验

更新日期：2026-10-13。状态：实现与 pilot 准备；尚无新实验结果。

旧的 quantity/diversity 实验及其结果保留为历史可行性实验。本协议对应新的、独立输出目录中的实验，不沿用旧 adapter 作为初始化。

## 研究问题

对已经正确的 teacher 解答，student 的前缀续写表现能否指导有限解释预算的分配，并提升训练后在新题上的无辅助准确率？

工作标题：Where to Explain: Student-Guided Rationale Refinement for Reasoning Distillation。

相近工作已涉及 student-aware 编辑、失败提示与轨迹修复。本项目不宣称该宽泛方向首次提出，也不把续写失败直接解释为不理解或因果断点。

## 分组

每题冻结同一条原始正确解答，每组每题一条记录。

| 标识 | 组别 | 构造 |
|---|---|---|
| original_rationale | A | 原始解答 |
| generic_expansion | B | teacher 自行选择位置，普通补充 |
| random_expansion | C | 在预先确定的随机候选边界补充 |
| hint_at_start | D | 使用 E 的同一段解释，移到解答开头 |
| local_refinement | E | student 定位、局部补充、独立续写收益筛选 |
| local_unfiltered | F | 同一困难位置、同一候选池，随机选合格解释，不看收益 |

B/C/D 与 E 使用相同的接受题目集合；E 拒绝补充的题四组均保持原始解答。F 保留未筛选解释。B/C 必须在新增 token 数的预注册容差内匹配 E；缺失或不匹配的控制不能静默替换为原文。D/E 解释文本完全相同。F 的增补比例可能不同，须报告，不能归因为单一因素。

## 数据与阶段

1. 免费工程检查：复用现有 R1 的 GSM8K 训练候选，不代表 MATH 主实验。
2. MATH pilot：训练侧约 100 题，最多 3 个候选边界；初筛每边界 4 次续写。
3. 正式实验：计划 MATH train 约 1,000 题、独立 validation 约 200 题；按 subject/level 分层抽样。MATH500 为主要测试，AMC23/AIME24/25 为待准备的补充测试。
4. 核对来源 ID 与标准化题目重叠。test 不用于诊断、解释筛选、阈值或长度调参。近重复和预训练污染不能仅靠精确去重排除。

## 模型与监督

固定 DeepSeek-R1 teacher 的实际返回版本及参数；Qwen3-8B-Base 的同一份本地 4-bit 权重作为冻结诊断模型与每组训练初始化。沿用 MLX LoRA rank=16、scale=2、dropout=0.05、学习率初值 1e-4；超参数在 validation/pilot 上冻结，训练种子 7/17/177。

解答分步需要保留原文，不由模型改写整个基线。候选边界不能包含最终答案行；公式或前文已暴露答案的情况必须人工审查。最终答案正确并不保证每个步骤正确。

teacher 每个困难位置最多生成两段简短解释。筛选使用与定位不同的采样种子，并重新测原始前缀基线。测试仅提供题目；SFT 仅对 assistant 目标计算 loss。无外部提示测试不等于模型不会自行生成解释。

## 长度与预算

新实验初始训练长度 4096、输出上限 2048，仅为 pilot 起点；正式值须按验证集截断率与资源测量冻结。禁止静默截断训练解答或沿用旧 256-token 正式评测上限。

第一轮工程 pilot 使用相同题目曝光和更新次数，并报告真实 target token 数，不能称为 token-matched。正式证据需要独立的训练 token 预算控制实验；相同 epoch 不能同时保证相同 token 数和曝光次数。D/E 可作文本和长度高度匹配的主要位置对照。

分别记录 teacher 请求/输入输出 tokens、student 定位/筛选 tokens、训练监督 tokens、耗时。用户已确认转移原预算：历史 R1 响应按 response_id 去重的记录估算费用为 ¥2.549316，新增请求共用 ¥39.45 估算上限，合计不超过原 ¥42。实际扣费以平台账单为准；不确定失败保留费用预留，不自动重试。pilot/main 共用同一账本，不能各自再花一份额度。

## 验证与结果解释

先做数据完整性、分组配对、独立采样与答案泄露检查，再运行训练。主指标为新题无辅助正确率，报告各 seed、均值与 SD、按题配对 bootstrap；多次采样平均正确率不是 pass@k。

E>A 支持整体收益；E>B 支持超越普通扩写；E>C 支持定位价值；E>D 支持局部插入价值；E>F 支持候选选择与拒绝模块的整体价值。若 E≈D，则不能宣称局部位置重要。

## 旧实验复用边界

旧报告 reports_mlx/TEACHER_UPDATE_ZH.md 与 FINAL_REPORT_ZH.md 对应 DeepSeek-V4-Flash → Qwen3-0.6B、196 题，不是 R1 → 8B 新主结果。2026-10-13 文件审计发现 R1 主候选文件实际为 25 题、120 条候选、115 条可用；不能把旧计划中的 380 题当作已完成。

保留旧 config、data、adapter 和 reports。新代码必须写入 refinement 命名的独立目录。
