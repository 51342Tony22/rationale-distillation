# 全体合格训练根题的独立 NLL／联合选位入口

`full_selector.py` 为已确认的 `selector_first_transfer128_seed42_v1` 实现全量候选准备和两种选位诊断。它直接读取共同教师质量集合，不需要每根题有关系模板、变式、Vsel/Vaudit 或迁移试学结果。

该入口不会自动接受教师推导，不读取公开 test，不启动正式训练，也不把 34 根开发题当全量。

## 质量输入

准备入口等待 `output/tgrd_v1/gates/full_teacher_quality_frozen_v1.json`。真实格式为：

- `schema: tgrd_full_teacher_quality_v1`、当前 `protocol`、`status: frozen`、`candidate_root_count: 7473`、`public_test_used: false`。
- `inputs`：将完整 `train7473.json`、完整教师 JSONL、教师原签名、当前128修订、具体审核规则文件及实际审核证据文件逐个绑定 SHA。
- `review_policy`：具体规则文件的 `path` 和 `sha256`，同时必须出现在 `inputs` 中。
- `reviews`：7,473 个不同根题的实际全文决定，含 source_id、question/output SHA、accepted 布尔值、all_text_read、review_kind 和数学理由。不能用最终数值匹配替代全文审核。
- `accepted_root_ids`：所有 accepted=true 的根题，按原始全量训练文件的冻结顺序排列。拒绝/隔离理由继续保留；不按组删除题。

当前全文审核未完成，规则边界仍需冻结，因此该质量文件尚未创建。历史简化审核记录必须保留并进行有来源的规范化或补充审核，不能填造 all_text_read，也不能静默修订标签。入口只消费真正完成的质量清单。

## 命令与输出

```sh
python experiments/tgrd_v1/full_selector.py prepare
python experiments/tgrd_v1/full_selector.py run --selector nll
python experiments/tgrd_v1/full_selector.py run --selector joint
```

使用配置中的 MLX Python。准备阶段只加载本地 tokenizer；诊断阶段才在共享 GPU 锁内加载学生模型，不能与活跃教师重叠。

输出独立存入 `output/tgrd_v1/full_selector_v1/`：完整共同 `cases.json`、SHA 绑定的 `cases_manifest.json`、`positions_nll_v1.jsonl`、`positions_joint_v1.jsonl` 和各自进度文件。进度不覆盖活跃教师的全局 status.json。改变冻结输入时拒绝复用旧签名，不覆盖历史结果。

NLL 对共同解析器得到的每个合格步骤评分，不要求 attention/遮蔽对照有效。联合选位保留既有 v2：末四层 attention、最多八查询、三个同类同长度随机对照、signed masking excess 与 NLL 的 rank-sum；诊断 seed7 是既有控制随机数，不是本轮训练 seed42 的变更。缺少联合对照的步骤继续报告，不混入独立 NLL 的资格过滤。

候选准备与既有 34 根题的真实 tokenizer 输出逐字段相同，包括 91 个候选步骤；该一致性由新增检查验证。续跑核对根题顺序、完整候选覆盖、原教师文本 SHA 和可重算的提议；不得悄悄漏步骤或改成别的选位法。

这是全量材料链的一部分。固定解释池的全量生成/审核、G1–G3 的全量材料装配与分阶段正式训练/公开评测门槛仍待完成，不能仅凭本入口存在就说全量选位已执行或正式训练已开始。
