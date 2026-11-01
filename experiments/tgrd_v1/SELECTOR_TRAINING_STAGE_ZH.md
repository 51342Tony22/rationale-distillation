# 前三组正式训练入口

当前协议是 `selector_first_transfer128_seed42_v1`。`train_selector_stage.py` 单独执行 G1/M1、G2/NLL、G3/M3，各 seed42、4 epoch。旧 `train_formal.py`、旧五组协议和原输出均保留，不能把新材料交给旧入口冒充同一实验。

```bash
python experiments/tgrd_v1/train_selector_stage.py freeze-gate
python experiments/tgrd_v1/train_selector_stage.py train --condition M1 --seed 42
python experiments/tgrd_v1/train_selector_stage.py train --condition NLL --seed 42
python experiments/tgrd_v1/train_selector_stage.py train --condition M3 --seed 42
```

冻结入口实际验证全量教师质量清单、完整 34 题开发诊断、独立 NLL 对所有合格步骤的覆盖、重建并审核的两类解释池、前三组共同材料，以及真实学生模型的训练续跑和推理适配器加载预检。开发观测报告如实记录选位覆盖、无法遮蔽对照的根题、与单 NLL 第一名的差异；不以正收益为训练条件，也不声称这些工程检查证明方法有效。解释缺失可按已冻结策略回退，但缺失生成结果或必要语义审核不能伪装成回退。

前三组训练不读取 G4/G5 材料、128 题变式、试学结果或迁移校准收益。G4/G5 仍属于之后的已授权阶段。训练调用原已验证的优化器/随机状态断点引擎，使用相同根题顺序、初始化及更新预算；说明性文本增加的实际 token 数独立报告。当前协议输出在 `output/tgrd_v1/training/selector_first_transfer128_seed42_v1/`，不会覆盖历史训练。

入口会拒绝 CPU 报告冒充 GPU 预检、旧协议或开发预览材料、缺组、根题顺序变化、输入和审核记录改动。真实 GPU 预检签名需绑定本入口后才能执行。此文件及 CPU 检查通过不表示正式训练已启动；实际门槛只会在所有真实材料和检查完成后生成。

该门槛仅开启训练，不开启公开评测。公开 1,319 题的独立入口及门槛仍需适配当前分阶段协议，且在查看结果前冻结后续迁移抽样与配置规则。不得直接沿用要求旧五组齐备的公开评测门槛。
