# GSM8K 优先执行修订

**2026-10-27 最新五组修订：** 本轮固定 G1普通CoT（M1）、G2独立单NLL（NLL）、G3联合选位（M3）、G4迁移收益筛选（M7）、G5原题CoT与Vsel验证推导交替（原M6），各seed42一次、每次4epoch，共5次正式训练。详见 [五组修订](FIVE_GROUP_AMENDMENT_ZH.md) 与 `output/tgrd_v1/five_group_amendment_v1.json`。旧四组/七组范围保留为历史，遇冲突以五组修订为准。

**2026-10-27 最新四组修订：** 本轮仅 G1普通CoT（M1）、G2独立单NLL（NLL）、G3联合选位（M3）、G4联合选位＋迁移筛选（M7），各seed42一次、每次4epoch，共4次正式训练。独立NLL从全部合格步骤选位。RSR/LALP、随机、直接变式组移出本轮。详见 [四组修订](FOUR_GROUP_AMENDMENT_ZH.md) 和 `output/tgrd_v1/four_group_amendment_v1.json`；本文下方七组/旧预算内容作为历史设计保留，遇冲突以新修订为准。

日期：2026-10-26。用户已授权20小时完整实验，并要求先集中GSM8K，时间允许再做SVAMP/GSM-Plus。

## 优先级

1. GSM8K材料质量、开发校准及正式训练。
2. 官方完整GSM8K test，固定1,319题，不抽样、不只跑前100题。
3. 满足主要组的训练种子和必要对照，再考虑另外两个公开数据集。
4. 原20小时截止不因本修订自动延长。

“完整数据集”可能表示完整test，也可能同时表示全部7,473道train。已异步请求澄清；现有40题校准与500题材料准备是两者都需要的工作，因此继续运行。未收到答复前不把500题训练称为7,473题全训练，不推断允许超过20小时运行。

## 官方种子核查

核查来源为OpenAI官方GSM8K仓库：

- https://github.com/openai/grade-school-math/blob/master/README.md
- https://github.com/openai/grade-school-math/blob/master/grade_school_math/train.py
- https://github.com/openai/grade-school-math/blob/master/grade_school_math/sample.py
- https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

在核查的README和示例训练/采样代码中，没有推荐固定训练随机种子，也没有指定“三个官方种子”。示例训练使用DataLoader(shuffle=True)，不设置固定seed。README明确该示例流水线用于演示，没有用于原论文实验。

因此：

- 官方固定train/test划分直接沿用，无需自己用seed重新划分官方测试集。
- 训练种子暂定7、17、27，为本实验预声明选择，不是官方推荐。
- 已冻结500训练候选的seed7保持不变，不用最终训练seed重新抽数据。
- 短程解释选择seed701独立记录；三个最终训练seed不能冒充三次独立选材。
- 最终GSM8K采用greedy全量评测，不利用不同抽样seed挑最好分数。

原始官方文件快照保存在 `output/tgrd_v1/references/gsm8k_official/`，各次获取的Git blob SHA另存来源manifest。官方仓库未规定通用种子不表示所有复现平台都没有各自默认seed；其他平台默认值不能写成GSM8K官方规定。

## 耗时边界

此前约229.5小时的估算是七组×三个训练seed加参照、三个评测数据集的总生成量级，不是单个GSM8K测试集。

单模型1,319题，按旧8.224秒/题锚点约3.0小时；七组×三个seed共27,699条GSM8K测试生成约63.3小时，尚不含构建材料和选材。以新实测吞吐更新，不承诺20小时可完成所有组合，也不把减少组或种子隐藏在“全量”中。

## 当前实际执行

顺序GPU队列已启动校准教师生成与500题训练教师目标。新材料仍需全文质量审核，不能把数字答案一致直接标成推理合格。正式test尚未开始，旧pilot保持原样。
