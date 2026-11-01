# RSR / LALP 静态评分适配（开发版 v1）

**2026-10-27 最新五组修订：** 本轮固定 G1普通CoT（M1）、G2独立单NLL（NLL）、G3联合选位（M3）、G4迁移收益筛选（M7）、G5原题CoT与Vsel验证推导交替（原M6），各seed42一次、每次4epoch，共5次正式训练。详见 [五组修订](FIVE_GROUP_AMENDMENT_ZH.md) 与 `output/tgrd_v1/five_group_amendment_v1.json`。旧四组/七组范围保留为历史，遇冲突以五组修订为准。

**2026-10-27 最新四组修订：** 本轮仅 G1普通CoT（M1）、G2独立单NLL（NLL）、G3联合选位（M3）、G4联合选位＋迁移筛选（M7），各seed42一次、每次4epoch，共4次正式训练。独立NLL从全部合格步骤选位。RSR/LALP、随机、直接变式组移出本轮。详见 [四组修订](FOUR_GROUP_AMENDMENT_ZH.md) 和 `output/tgrd_v1/four_group_amendment_v1.json`；本文下方七组/旧预算内容作为历史设计保留，遇冲突以新修订为准。

这两个组在原始学生 S0 上评分同一个已经完成质量审核的解释池，不更新学生，不使用 Vsel、Vaudit 或公开测试成绩。当前实现通过 CPU 检查，真实模型 GPU 对照检查已另编写，等待当前全量教师任务结束后执行。不得据此声称已完成论文复现或正式实验。

## RSR-adapted：保留公式、截断与排序方向

来源：Yang 等，*Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment*，arXiv:2601.14249v5，§3.4，式 (7)。

`RSR = sum(min(rank_t, 100)) / sum(-log p_t)`，越小越好。

- 分子采用每个真实目标 token 的一基排名，严格按 logits 大于目标 logit 的词表项数加一；相同 logit 不算排在目标之前。
- 排名截断为 100；不是用未截断的词表排名。
- 使用“和的比值”，不是“每 token 比值的平均”。
- 分母 epsilon 为 1e-12，与作者代码一致，不是人为丢弃低 NLL token。
- 已核对作者仓库 commit `59a7c4cdbbb7c26b93f91472da5d79498e29b5c0` 中 `rsr_cal.py`：token 计算约第 159–176 行、样本聚合约第 197–225 行。
- 本机通过小块输出投影后计算严格排名，再截断；数学上等于作者 top100 后计数截断的策略，避免一次保留全部上下文的词表 logits。

## LALP-adapted：局部上下文及步骤等权

来源：Just、Ko、Jia，*The Signal is in the Steps: Local Scoring for Reasoning Data Selection*，ICML 2026，PMLR 306，§6，式 (3)，PDF 第 4 页。

先计算每步在“完整题目提示 + 最近 k 个前序步骤”下的 token 平均 log probability，再对步骤平均。**越大越好**。各步权重相等，不按步骤 token 数加权。

本开发适配固定 `k=2`；这是针对短 GSM8K CoT 的实验选择，不冒称论文唯一或官方推荐窗口，也尚未冻结为最终公开测试配置。本文主实现采用 GLM-4.5-Air 进行语义步骤切分，我们采用确定的双换行段落切分；该规则属于原论文 Appendix C.15 中明确研究过的替代方式，但不是其最优主切分器。

- 原始问题提示始终保留；较早的推导段落移出局部上下文。
- 在同一个完整 completion 的 tokenization 上切分，不分别重新分词各段，不重复或遗漏跨段 token。
- 跨原始段落边界的 token 归入后一段，记录实际 token 边界。
- 全部响应内容参与评分，包括解释、格式文本及最终答案。段落是确定的近似单位，不把它写成经过人工确认的语义步骤。
- 不因某条候选分数低而改分段、换窗口或补写解释。

## 共同实验边界

- 对完整 completion 评分，而不只给新增解释打分；源提示使用当前 Qwen2.5-Math-1.5B Base 的统一学生提示。
- 本适配不计 EOS；最终训练仍采用 completion + EOS 交叉熵。两个静态组的评分范围完全一致，另保存 token/步骤统计。
- 主 M4/M5 从合格非空共享池选一条；只在没有合格非空解释时回退 e0。会保存 e0 分数作为诊断，但不因 e0 静态分数更好而改变主组的补充政策。
- 固定分数相同时优先较短解释，再按候选 ID 排序。
- 原论文研究长推导、多教师、大模型等设定。本实验研究同一教师短推导的局部解释选择，必须标注 adapted；结果只在本实验统一设置内比较，不直接对发表成绩排名。
- CPU 测试包含独立 NumPy 参考与实际 MLX CPU 内核的下一 token 对齐、同值排名、100 截断和输出分块；尚不替代真实学生模型 GPU 检查。

本地来源快照和哈希见 `output/tgrd_v1/references/static_selectors/source_manifest.json`。原始论文和作者代码作为研究证据保存，没有执行下载的代码。
