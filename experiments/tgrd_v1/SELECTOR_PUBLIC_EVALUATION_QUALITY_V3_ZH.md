# 解释自动准入 v3 的公开评测适配

当前训练使用`train_selector_quality_v3.py`及已冻结的新训练门槛。旧公开评测入口引用旧训练门槛，因此新增`evaluation_selector_quality_v3.py`与`evaluate_selector_quality_v3.py`，保留所有活跃训练签名绑定的原模块。新入口仍只评估前三组M1/NLL/M3，seed42，各完整1319test，不读取解释或教师文本辅助自由解题。

适配验证全部三组真实最后epoch4权重、优化器断点、相同S0、根题与日程、材料哈希、训练签名及v3训练入口哈希。旧质量模式或旧公开门槛不能重命名代用。原实际GPU加载证据对应未变的推理API；本新增公开入口尚未实际GPU执行，不把CPU检查当作公开评测通过。

公开test前仍需真实general34全网格重建所选配置、完整128抽样证据重放、完整迁移规则和具体限制披露。新披露额外要求`automatic_explanation_quality_noise`文字及`explanation_quality_evidence_sha256`，绑定实际固定128报告和自动准入导出。保留109接受/19严格标记（并非全为算错）、8192原机械可用解释未读、不宣称全量语义正确；原教师噪声及历史91严格标记另行披露。不会据test表现修改阈值或checkpoint；全开发非正则如实保留无支持/不自动扩量结果。

新门槛文件使用`_pretest_transfer_quality_v3.json`和`_selector_public_evaluation_quality_v3.json`。当前两文件均未创建，正式test没有读取。准备源码或CPU测试不授予公开test权限。

七项新集成检查使用真实微型MLX CPU训练引擎产生的三组epoch4产物，并检查v3签名、错误材料版本、未请求组未完成、旧公开门槛拒绝、公开数据加载前阻断、真实网格缺失与质量披露绑定。总计164项集成检查通过；测试均不使用官方test题面。

待真正三组训练完成及科学网格完成后，用新入口`freeze-pretest --roster ... --rules ... --disclosures ...`，再`freeze-gate`和`run --condition M1|NLL|M3 --seed 42`。不得提前写布尔字段解除等待。G4/G5仍需有限128真实试学及各自正式材料、训练、评测适配，不自动假定已实现。
