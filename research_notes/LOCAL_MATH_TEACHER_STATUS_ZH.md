# 本地数学教师：运行前准备完成

最新状态（2026-10-22，取代下文等待启动状态）：用户授权“运行”后，五题教师试跑、给定序列的logits导出及5/5全文助手审核全部完成。教师加载到生成/导出结束共42.18秒，MLX内存峰值4.44GB；五题答案和推导均正确、无截断/漂移/矛盾。没有学生训练或付费模型API调用。运行目录为`output/local_math_teacher/run_20261022T084957Z`，最终状态`complete_with_assistant_math_audit`。五题只作链路检查，不能视为正式数据集准确率或方法效果。完整结果见[试跑报告](output/local_math_teacher/RESULT_ZH.md)。下文保留运行前准备记录，勿据此重复启动。

更新日期：2026-10-22。状态：`ready_for_explicit_teacher_run`，等待用户明确启动。真实数学教师尚未加载、尚未生成解答、尚未训练学生。

本次准备的是 Qwen2.5-Math-7B-Instruct 的社区 MLX 4bit 版本，用于本机可行性试跑。固定版本、参数和五道自编题已保存；原六组实验的结果与账本不改。

已完成的检查：

- 4,284,346,187 字节权重下载完整，SHA256 与固定版本公开记录一致；其余模型文件亦逐一核验。
- safetensors 中735个张量的索引及数据边界通过检查。
- 32GiB内存、Apple M5、Metal GPU可用；检查时无其他模型训练任务。
- 分词器模板与assistant边界正常；五题prompt为55–62 tokens，加768生成预算均低于4096。
- 教师与学生候选共享151,643个普通词元的映射；输出维数分别为152,064/151,936，EOS分别为151645/151643。特殊词元映射及正式KL训练尚未实施。
- 微型随机4bit Qwen2完成GPU生成接口、指定序列的logits导出、有限性与概率归一化检查。逐token和批量预测的概率最大绝对差约9.98e-8；刻意错位一个token时明显不一致，位置检查通过。
- 启动脚本、配置及来源清单的SHA256与最终预检记录相符。没有创建真实试跑的run目录。

数学教师的实际加载、速度、内存峰值和解题质量仍待试跑；微型模型的结果不能代替这些测量。

下一步固定为五题本地试跑、一个给定序列的软分布导出，以及五题全文数学审核。暂估运行5–15分钟、审核5–10分钟；首题后更新估计。该阶段不自动开展新一轮蒸馏训练，也不调用付费模型API。

文件入口：

- [运行计划](experiments/local_math_teacher/PLAN_ZH.md)
- [固定参数与自编题](experiments/local_math_teacher/config.json)
- [启动脚本](experiments/local_math_teacher/teacher_trial.py)
- [完整预检证据](output/local_math_teacher/preflight.json)
- [模型来源清单](output/local_math_teacher/reference_manifest.json)

用户说“运行”后，执行计划中的`run`命令。脚本会先重新核验资产和环境、检查其他模型进程并获取独占锁，之后才加载教师。失败会保留日志，不自动重发或重复试跑。
