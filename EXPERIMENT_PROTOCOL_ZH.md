# FYP 实验执行协议

## 1. 本项目要回答什么

在学生模型、题目集合、训练步数和提示词保持一致时，比较以下监督信号：

1. 只有最终答案；
2. 每题一条正确 rationale；
3. 每题 K 条随机选择的正确 rationale；
4. 每题 K 条经过多样性选择的正确 rationale。

主要问题不是“更多数据是否更好”，而是“在相同题目曝光次数下，不同 rationale
监督是否更有效”。

## 2. 已冻结的数据划分

- GSM8K 原始训练集：7,473 题；
- 固定开发集：500 题，只用于试运行和超参数选择；
- 剩余训练候选：6,973 题；
- 官方测试集：1,319 题，最终实验之前不用于选择参数；
- SVAMP challenge test：1,000题，只做跨数据集评价，不参与训练或选参；
- 划分随机种子：7；
- 原始文件使用官方 SHA-256 校验。

## 3. 推荐模型

- 教师：`deepseek-ai/DeepSeek-V4-Flash`；
- 学生：`Qwen/Qwen3-1.7B-Base`；
- 微调方式：LoRA，所有条件使用完全相同的配置；
- 不使用已经完成 reasoning post-training 的学生模型，以减少已有推理能力对实验效应的遮蔽。
- 本机 MPS 已完成兼容性验证，但实测吞吐不足以支撑1.7B正式实验；论文结果应在同一台
  CUDA GPU/同一运行环境中完成，避免硬件变化形成额外变量。

教师和学生模型一旦开始主实验就不得中途更换。

## 4. 教师数据生成

每道训练题生成 N 条候选 rationale。候选使用不同的解题策略提示和随机采样，
每条限制在约120词、最多256个生成 token，避免教师冗长推理成为混杂变量。
程序提取每条输出的最终答案，只保留与 GSM8K gold answer 完全匹配的候选。

- Pilot：先做 200 题，每题 N=5；
- 观察正确候选接受率和候选间相似度；
- Main：建议每题 N=8、最终选择 K=3；如果预算有限，可在 protocol freeze
  之前根据 pilot 决定主实验训练题数；
- 每题少于 K 条正确 rationale 时，该题从四组中同时移除，不能只从某一组移除。

原始 API 输出必须保留，生成失败可用 `--resume` 继续。

## 5. 多样性选择

对每题的正确候选 rationale 计算 unigram + bigram TF-IDF cosine similarity。

- Multi：用固定随机种子选择 K 条；
- Diverse Multi：先选平均相似度最低的一条，再逐步选择与已选集合最大相似度最低的候选；
- 两组 K 相同；
- 构建报告记录两组的平均 pairwise TF-IDF distance，确认多样性操作确实改变了监督数据。

TF-IDF 是预注册的主要多样性定义。语义 embedding 多样性可以作为扩展消融，不能在看到
测试结果后替换主定义。

## 6. 公平训练

每个保留问题在所有组中都形成 K 条训练记录：

- Answer-only：同一答案重复 K 次；
- Single：同一条 rationale 重复 K 次；
- Multi：K 条随机 rationale；
- Diverse Multi：K 条多样性 rationale。

因此四组具有相同的行数、题目顺序、题目曝光次数、epoch 数和优化步数。默认训练 3 epochs，
有效 batch size 为 8。Pilot 后可使用开发集统一调整一次，之后冻结配置。

## 7. 评价

主要结果：

- 官方 GSM8K test 的 final-answer exact match；
- 三个预先指定的配对差异：Single−Answer、Multi−Single、Diverse−Multi；
- 每个差异报告 paired bootstrap 95% CI。

次要结果：

- 无法提取答案的比例；
- rationale 平均词数；
- 同题 5 次随机生成的答案一致性；
- 三种等价 prompt 下的答案一致性和准确率；
- 手工抽样分析 calculation / reasoning / extraction / other error。

行为评价固定使用前 200 个测试样本，避免产生过量重复推理；exact match 使用全部 1,319 题。

## 8. 执行顺序

```bash
source .venv/bin/activate
python -m fyp_distill.preflight

# Pilot 教师数据（需要先配置 .env）
python -m fyp_distill.generate_rationales --limit 200 --n 5
python -m fyp_distill.build_variants --k 3
python -m fyp_distill.audit_data

# 四组训练
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.train --variant "$variant" --seed 7
done

# 先在 dev 上比较，冻结配置后才跑 test
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.evaluate --variant "$variant" --seed 7 --split dev
done
python -m fyp_distill.evaluate --variant base_student --seed 7 --split dev
python -m fyp_distill.summarize --seed 7 --split dev

# 最终测试
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.evaluate --variant "$variant" --seed 7 --split test
done
python -m fyp_distill.evaluate_teacher --split test
python -m fyp_distill.summarize --seed 7 --split test

# 外部泛化测试
for variant in answer_only single_rationale multi_rationale diverse_multi_rationale; do
  python -m fyp_distill.evaluate --variant "$variant" --seed 7 --split svamp
done
python -m fyp_distill.evaluate_teacher --split svamp
python -m fyp_distill.summarize --seed 7 --split svamp
```

最终实验使用 7、17、177 三个训练种子。Pilot 的数值不能作为论文最终结论。
