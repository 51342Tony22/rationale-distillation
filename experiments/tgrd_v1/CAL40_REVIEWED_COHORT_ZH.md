# 校准材料扩展：cal40_reviewed34_v2

官方 train 内原 cal40 的40题教师输出已逐条进行助手全文核查，其中34题通过，6题隔离。这里的审核不是独立人工复核，也不是全7473题的质量结论。当前六题排除仅用于内部开发；最终各组共同训练样本的排除/回退规则尚未冻结。

新增24套显式关系模板，与最初10套合并，每题有6道 Vsel、6道 Vaudit，共204+204道。每份模板分别提供 numeric、wording、structural 的选择/审计题面；结构题改变未知量，标签重新计算。原始公式必须恢复源参考答案。物理计数、余量和货币/时间单位分别处理，未执行教师提供的代码。

这是手工构建的校准覆盖，不能声称已有适用于整个GSM8K的可靠关系解析器。所有生成题面来自经过助手语义核查的模板；求解器和独立数量核算验证算术，并不构成另一个独立人对408道实例逐条复核。

文件：

- `data/tgrd_v1/cal40_reviewed_relations_v2.json`
- `data/tgrd_v1/cal40_reviewed_variants_v2.json`
- `data/tgrd_v1/cal40_reviewed_manifest_v2.json`
- `output/tgrd_v1/calibration/cal40_reviewed34_v2/positions_v2_cases.json`
- `output/tgrd_v1/preflight/cal40_reviewed34_v2_material_selection_cpu_v1.json`

CPU预检验证34题教师审核/输出绑定、全部408道标签、跨集合参数隔离、重复题面、原10题实例保持一致、训练上下文和解释请求可见边界。已准备91个候选计算步骤，其中26题至少有两个候选；所有候选对应182份假设解释请求。GPU实际诊断最多选两个位置，实际生成请求上界120份，具体数量尚未知。最长教师解释提示637 tokens，最长普通CoT训练上下文含EOS617 tokens。

## JSON复现问题及修复

初次未入队的扩展 v1 在读回JSON后预检失败。原因是冻结工具排序字典键，但原抽样按参数字典的迭代顺序逐一消耗随机数。种子相同不保证参数顺序改变后仍抽到相同实例。此前10题的旧快照也没有显式记录该顺序。

v1文件完整保留，未用于任何学习、公开测试或质量放行。v2为全部34套关系显式加入 `sampling_parameter_order`，读回JSON后整套抽样可精确重放；原10题采用最初源码中的参数顺序，保留其原有实例。该元数据不改变问题的数学关系。

对于缺少抽样顺序的历史快照，验证器不重新抽题，而是用每道冻结实例的参数和模板精确恢复题面、答案、派生量及哈希，同时检查域、身份和集合隔离。新快照还要求完整随机抽样重放一致。测试覆盖排序JSON往返、标签/题面/模板/哈希篡改和跨集合实例重复。25项测试通过，包括新增24题的独立数量核算和原来的MLX CPU算子检查。

## 执行与边界

所有校准执行器支持 `--cohort cal40_reviewed34_v2`，输出使用独立命名空间，保留bootstrap10默认路径与旧输出。后续队列改为34题选位、共享解释生成、审核后静态评分；不重复运行旧10题的同类GPU阶段。原待运行队列另存备份。全量7473教师任务和签名输入未改变。

共享池仍须实际生成并逐条检查数学正确性、解释目标步骤、未引入新结果/最终答案及未循环解释。通过机械检查不等于语义合格。实际attention诊断、解释生成、解释审核、短训练配置网格、真实试训驱动验证、正式七组多seed训练均不能由这次CPU预检代替。

现有mask诊断仍是v1部分token删除对照，不是语义匹配删除的因果证据。Vaudit在此calibration中允许用于超参数开发，不能作为独立验证结果。公开test未访问。最终训练如回收原train内开发题，须披露其不再独立。

复现CPU准备：

```sh
"/Users/starsfalling/Documents/FYP Mini Agent/.venv-mlx/bin/python" experiments/tgrd_v1/calibration_relations.py
"/Users/starsfalling/Documents/FYP Mini Agent/.venv-mlx/bin/python" experiments/tgrd_v1/positions_v2.py --prepare-only --cohort cal40_reviewed34_v2
"/Users/starsfalling/Documents/FYP Mini Agent/.venv-mlx/bin/python" experiments/tgrd_v1/preflight_calibration_cohort.py --cohort cal40_reviewed34_v2
```
