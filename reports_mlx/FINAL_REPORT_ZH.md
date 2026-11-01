# FYP 扩大数据实验最终报告

## 1. 研究问题

本实验研究紧凑语言模型的推理蒸馏，重点回答：

1. CoT 推理监督是否优于只监督最终答案？
2. 同一道题使用多条 rationale 是否优于只使用一条 rationale？
3. 主动选择差异更大的 rationale 是否优于普通的多 rationale 抽样？
4. 蒸馏后的小模型在多大程度上缩小了与教师模型的差距？

## 2. 实验设计

- 教师模型：`deepseek-ai/DeepSeek-V4-Flash`（SiliconFlow API）。
- 学生模型：`mlx-community/Qwen3-0.6B-Base-4bit`。
- 微调方式：MLX QLoRA，LoRA rank 16。
- 随机种子：7、17、177。
- 教师生成：200 道 GSM8K 训练题，每题请求 5 条 rationale。
- 数据过滤：只接受最终答案与 gold answer 一致，并且位于公开 response content 中的 rationale。
- 合格题目：196/200 道，每道至少 3 条正确 rationale。
- 每个条件：196 道相同题目、588 条训练记录、约 3 epoch、220 次优化更新。
- 测试：100 道未参与训练的 GSM8K，以及 100 道 SVAMP。
- 主要指标：最终数值答案 Exact Match（EM）。
- 统计方法：三个随机种子的均值和样本标准差；在每题跨种子平均后进行 10,000 次配对 bootstrap，报告 95% 区间。

四个训练条件：

1. `answer_only`：每题只学习最终答案，重复三行以匹配训练行数。
2. `single_rationale`：每题使用一条正确 rationale，并重复三行。
3. `multi_rationale`：每题随机使用三条正确 rationale。
4. `diverse_multi_rationale`：每题通过贪心 TF-IDF 距离选择三条差异较大的正确 rationale。

普通 multi-rationale 的平均选中距离为 0.457，diverse multi-rationale 为 0.563，说明筛选方法确实提高了表层文本多样性。

## 3. GSM8K 结果

| 条件 | Seed 7 | Seed 17 | Seed 177 | EM 均值 | SD | 无效答案率 | 平均推理词数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未微调学生 | 0.030 | - | - | 0.030 | - | 0.320 | 11.3 |
| Answer only | 0.010 | 0.010 | 0.000 | 0.007 | 0.006 | 0.000 | 0.0 |
| Single rationale | 0.310 | 0.360 | 0.310 | 0.327 | 0.029 | 0.000 | 32.8 |
| Multi rationale | 0.330 | 0.290 | 0.350 | 0.323 | 0.031 | 0.000 | 34.2 |
| Diverse multi rationale | 0.420 | 0.380 | 0.380 | **0.393** | 0.023 | 0.000 | 35.8 |
| 教师模型 | - | - | - | 0.980 | - | 0.000 | - |

关键配对比较：

| 比较 | EM 差值 | 95% bootstrap 区间 |
|---|---:|---:|
| Single rationale - Answer only | +0.320 | [0.253, 0.390] |
| Multi rationale - Single rationale | -0.003 | [-0.057, 0.050] |
| Diverse multi - Multi rationale | **+0.070** | **[0.013, 0.127]** |
| Diverse multi - Answer only | +0.387 | [0.307, 0.467] |

GSM8K 上，普通 multi-rationale 并没有超过 single-rationale。经过多样性选择后，diverse multi-rationale 比普通 multi-rationale 高 7.0 个百分点，且置信区间不跨 0。

## 4. SVAMP 跨数据集结果

| 条件 | Seed 7 | Seed 17 | Seed 177 | EM 均值 | SD | 无效答案率 | 平均推理词数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未微调学生 | 0.020 | - | - | 0.020 | - | 0.010 | 11.6 |
| Answer only | 0.130 | 0.060 | 0.020 | 0.070 | 0.056 | 0.000 | 0.0 |
| Single rationale | 0.530 | 0.450 | 0.530 | 0.503 | 0.046 | 0.000 | 22.5 |
| Multi rationale | 0.520 | 0.480 | 0.430 | 0.477 | 0.045 | 0.000 | 27.1 |
| Diverse multi rationale | 0.540 | 0.560 | 0.480 | **0.527** | 0.042 | 0.000 | 24.8 |
| 教师模型 | - | - | - | 0.969 | - | 0.000 | - |

教师模型成功返回 97/100 道题，其中 94 道正确，所以条件准确率为 94/97 = 96.9%，API 覆盖率为 97%。若将三次 API 超时保守计错，端到端准确率为 94.0%。

关键配对比较：

| 比较 | EM 差值 | 95% bootstrap 区间 |
|---|---:|---:|
| Single rationale - Answer only | +0.433 | [0.363, 0.503] |
| Multi rationale - Single rationale | -0.027 | [-0.093, 0.040] |
| Diverse multi - Multi rationale | +0.050 | [-0.010, 0.110] |
| Diverse multi - Answer only | +0.457 | [0.383, 0.530] |

SVAMP 上 diverse multi-rationale 仍是均值最高的条件，但其相对普通 multi-rationale 的区间略微跨 0，因此这里只能报告方向一致，不能宣称统计上已确定胜出。

## 5. 与 58 题 Pilot 的比较

| 条件 | GSM8K Pilot | GSM8K 196题 | 变化 | SVAMP Pilot | SVAMP 196题 | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Answer only | 0.000 | 0.007 | +0.007 | 0.013 | 0.070 | +0.057 |
| Single rationale | 0.233 | 0.327 | +0.094 | 0.433 | 0.503 | +0.070 |
| Multi rationale | 0.287 | 0.323 | +0.036 | 0.450 | 0.477 | +0.027 |
| Diverse multi rationale | 0.283 | 0.393 | **+0.110** | 0.483 | 0.527 | +0.044 |

增加训练题后，三个 CoT 条件整体提高。GSM8K 上提升最大的是 diverse multi-rationale，增加 11.0 个百分点。Pilot 中普通 multi-rationale 领先 single-rationale 的现象没有在扩大数据后保留，说明原结论存在小样本不稳定性。

## 6. 师生差距

使用公式：

`Gap Closure = (蒸馏后学生 - 未微调学生) / (教师 - 未微调学生)`

| 条件 | GSM8K 差距缩小率 | SVAMP 差距缩小率 |
|---|---:|---:|
| Answer only | -2.5% | 5.3% |
| Single rationale | 31.2% | 50.9% |
| Multi rationale | 30.9% | 48.1% |
| Diverse multi rationale | **38.2%** | **53.4%** |

学生仍未接近教师的绝对准确率：GSM8K 为 39.3% 对 98.0%，SVAMP 为 52.7% 对 96.9%。但 diverse multi-rationale 已分别缩小约 38% 和 53% 的原始师生差距。

## 7. 最终结论

1. **CoT 蒸馏明显有效。** 在两个数据集上，任何 rationale 监督都远优于 answer-only，且区间明显不跨 0。
2. **仅增加 rationale 数量并不足够。** 普通 multi-rationale 没有稳定优于 single-rationale。
3. **rationale 的选择质量比数量更重要。** diverse multi-rationale 在两个数据集上均取得最高均值；GSM8K 上相对普通 multi 的优势具有统计证据，SVAMP 上为相同方向但证据尚不充分。
4. **研究目的已部分达到。** 实验成功证明推理蒸馏能显著改善 0.6B 学生，并证明多样性选择具有潜在价值；但学生仍没有接近或超过教师。

因此，论文中最稳妥的核心表述是：

> Rationale-based distillation substantially improves a compact student over answer-only supervision. Merely adding multiple rationales does not guarantee further gains, whereas selecting more diverse correct rationales yields the strongest overall performance and a significant improvement on GSM8K, although the cross-dataset advantage remains less conclusive.

## 8. 限制

- 训练规模从 58 道扩展到 196 道，但对最终毕业论文仍属于小规模实验。
- 测试集各为 100 道，差值的置信区间仍较宽。
- TF-IDF 衡量的是表层文本差异，不一定等同于数学策略差异。
- Answer-only 虽匹配了行数和优化步数，但其目标序列更短，监督 token 数更少。
- 本轮核心评估没有加入多次随机生成一致性和 prompt sensitivity；这些适合作为后续行为分析。
