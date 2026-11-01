# 全量固定解释与前三组材料入口

当前协议为 `selector_first_transfer128_seed42_v1`。这两个入口接续全量选位，不需要 128 题变式、迁移试学结果或 G4/G5 材料；仍需真正完成全量教师质量审核和两套选位诊断。

## 固定解释

`full_explanations.py prepare` 重建独立 NLL 与联合选择器的全量诊断，只取各自第一名位置的 relation、contrastive 两种解释。NLL 全部合格步骤的分数必须完整，联合缺少匹配遮蔽对照不影响 NLL。无位置的根题保留 e0。

对两个选择器建立共享的规范请求表：同一根题、同一步、同一解释类型且前缀与提示完全一致时，只调用教师一次。请求只包含题目、当前步之前的解题文本及当前步，不包含后续解答、变式或探针标签。每个规范请求只尝试一次，生成温度 0，上限 256 token，解释按学生 tokenizer 要求 32–64 token。

当前局部解释教师采用明确冻结的`local_explanation_general7b_v2`配方，详见`AUXILIARY_EXPLAINER_V2_ZH.md`。
准备时只检索该配方已完成的134条开发诊断输出，将同一精确提示的旧输出及其记录SHA纳入冻结复用表。
旧解释即使曾为第二名位置生成，只要提示完全相同也直接复用；失败输出也复用，不另生成一次择优。
旧数学教师缓存与本配方隔离，不能只因候选ID相同而复用。开发解释能否进入正式池仍须精确输出的语义审核。
新全量输出目录为`output/tgrd_v1/full_explanations_general7b_v2/`，原题CoT教师及其输出不变。

`generate` 在共享 GPU 锁内运行，逐条保存输出、请求散列和机械拒绝证据。恢复时检查已完成请求顺序、输出 SHA 和重新计算的机械检查；相同请求不重新抽样。该入口不写教师的全局 `status.json`。

`assemble` 从规范输出与 `full_text_reviews.json` 重建两份池。机械不合格可直接拒绝；机械可用的每条解释必须有绑定精确输出的全文语义审核，不能把“尚未审核”当作失败回退以跳过审核。联合与 NLL 仅共享同位置同类型的文本，各自池仍只收各自选到的候选，NLL 独有候选不会进入联合池。

全量生成第一名两类解释已经足够维持 G2/G3 的预定策略：第一名位置优先 relation，不合格再 contrastive，均无合格解释才普通 CoT。此入口不是 G4 的完整试学池。G4 之后仍须在已冻结的 128 根题内补齐联合第二名位置两类解释，保持最多四解释加 e0；不能把这里的 rank0 池直接冒充完整迁移池。

## 三组材料

`full_selector_materials.py` 先从生成结果、请求和实际审核重建固定解释池，再从共同合格 N 与共同根题顺序构建 M1、NLL、M3 三份四轮材料。每条材料重新用真实学生 tokenizer 编码，包含 EOS，不允许截断；材料覆盖缺失是错误，不按组丢题。覆盖报告分别保存解释覆盖、普通 CoT 回退数及监督 token 总量。相同根题/优化更新预算不意味着三组监督 token 总数相同，实际差异须披露。

材料保存于 `output/tgrd_v1/formal_materials/selector_first_transfer128_seed42_v1/selector_first/`，输入与输出都绑定 SHA。这里明确记录 `formal_training_enabled=false`：材料准备完成不等于正式训练门槛通过，真实模型工程验证和新版分阶段训练/公开测试门槛仍须单独完成。旧五组协议的训练入口不可直接接收这些材料。

## 命令与前置条件

使用本项目 MLX 环境中的 Python，依次执行：

```sh
python experiments/tgrd_v1/full_explanations.py prepare
python experiments/tgrd_v1/full_explanations.py generate
# 完成实际输出的 full_text_reviews.json 后：
python experiments/tgrd_v1/full_explanations.py assemble
python experiments/tgrd_v1/full_selector_materials.py
```

所有生成、质量审核和后续训练仍使用官方训练侧数据。本入口不加载公开 test、不冻结或挑选 128 名单，不声明任何方法有效性。
