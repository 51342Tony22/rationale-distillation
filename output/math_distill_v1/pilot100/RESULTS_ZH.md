# 100候选训练／250开发题扩展结果

状态：pilot100_complete_full_text_audited。共同有效训练97/100；三个新adapter均独立从同一原始1.5B学生初始化。

本报告为开发探索结果。原48题训练小试和所有原输出保留，正式1319题test未运行。未训练200输出复用，只新增50；新训练模型各自重新生成全部250题。

教师：mlx-community/Qwen2.5-Math-7B-Instruct-4bit（本地4bit）；学生：Qwen/Qwen2.5-Math-1.5B（原始BF16）。数据：GSM8K，版本b0bb162abedc65e1fdd8e93ed090fd7598ee68bc，训练与开发均取官方train的冻结互斥子集。模型固定revision与prompt见results.json。

| 条件 | v1正确 | 固定v2正确 | v1/v2 invalid | 长度停止 | 全文审核 |
|---|---:|---:|---:|---:|---:|
| 未训练学生 | 166/250 | 199/250 | 55/21 | 0 | 250/250 |
| 普通CoT | 224/250 | 224/250 | 1/1 | 0 | 250/250 |
| 仅难度补充 | 225/250 | 225/250 | 1/1 | 0 | 250/250 |
| 难度＋attention | 221/250 | 221/250 | 2/2 | 0 | 250/250 |

v1/v2为同一输出的两种固定解析，解析差异不是训练提升。v2在查看最初20道开发输出后制定，不能称盲测主指标。数字命中不保证解释正确，原gold和单位/措辞问题按辅助审核披露。

## attention相对其他组的同题配对比较

| 参照 | v1净变化／改对／改错 | v2净变化／改对／改错 | v2按题bootstrap95%区间 | v2精确McNemar p |
|---|---:|---:|---:|---:|
| 未训练学生 | +22.0pp / 68 / 13 | +8.8pp / 38 / 16 | [3.2, 14.4]pp | 0.003838 |
| 普通CoT | -1.2pp / 7 / 10 | -1.2pp / 7 / 10 | [-4.4, 2.0]pp | 0.6291 |
| 仅难度补充 | -1.6pp / 5 / 9 | -1.6pp / 5 / 9 | [-4.4, 1.2]pp | 0.424 |

区间使用10000次按题重采样、seed7；不衡量训练seed方差。原始逐题改善/退步ID和v1区间见results.json。

## 训练量扩充：共同前100评测题

| 条件 | 原48题训练v1/v2 | 新训练v1/v2 |
|---|---:|---:|
| 未训练学生 | 62/80 | 62/80 |
| 普通CoT | 91/91 | 91/91 |
| 仅难度补充 | 88/88 | 90/90 |
| 难度＋attention | 89/89 | 90/90 |

此表各数字分母均为100，仅作同题训练规模探索；不能把旧100题总分与新250题总分直接比较。

## 如何解释本轮结果

主要机制对照是attention对仅难度：固定v2净差-1.6个百分点。区间跨过或接触0，本轮未建立attention相对仅难度的额外优势。
普通CoT对未训练学生检验本轮监督训练的整体变化；attention对普通CoT检验完整补充方法的净收益。它们与attention自身的增量不是同一结论，全部配对数据见results.json。

97个共同有效样本中，30对解释完全相同，6对长度差超过16tokens、最大差40。前文线索相关性分类（辅助主观判断、非盲审）为{'partial': 32, 'direct': 62, 'weak': 3}。不能将本轮称为严格长度控制、因果attention定位或通用数学步骤发现。

## 训练、生成与行为

| 条件 | 一遍监督tokens | 训练秒 | 本轮新生成秒 | 平均生成tokens | 多桥句输出数 |
|---|---:|---:|---:|---:|---:|
| 未训练学生 | — | 0.00 | 294.91 | 208.65 | 0 |
| 普通CoT | 26807 | 107.12 | 2056.05 | 295.35 | 0 |
| 仅难度补充 | 29685 | 115.35 | 2552.89 | 304.51 | 4 |
| 难度＋attention | 29671 | 116.29 | 2058.56 | 302.95 | 10 |

4轮、batch1、梯度累积5、seed7；仅completion及EOS监督，无prompt/padding目标、无训练截断。实际微步、更新、adapter哈希和费用见results.json及metadata。付费API支出0，未核算电力和人工成本。

新增材料生成50条教师目标及两组各49条补充，均保留原始输出并逐条审核；旧50条前缀原字节复用。各阶段生成秒数见results.json，不含加载、定位、审核与未保存工作，不能等同整批墙钟时间。

训练全部完成后，原入口在新增基线生成前因旧signature的SHA字符串格式读取错误停止；单独的恢复入口修正校验格式，验证已签名材料与adapter后接续评测。未修改已签名代码、重训或覆盖旧输出，失败日志保留。

多桥句指标仅计算共同句式出现次数，不将解释不同步骤一概视为重复错误。全文记录另审计算、条件、任务目标、前后矛盾、循环、单位与截断。

## 全文审核的具体问题示例

- 未训练学生 / gsm8k:train:2005（v1=False、v2=False）：既有全文审核分类：incorrect_omitted_first_half；原审核保留，未补写旧记录。
- 未训练学生 / gsm8k:train:5888（v1=False、v2=False）：全文审核：先将5天与周薪相乘又重复乘4，最后还把缺一天误当5小时，得到4775；按参考四周假设应(6×4−1)×8×5=920。
- 未训练学生 / gsm8k:train:908（v1=False、v2=False）：全文审核：列式及代码应为12×5000+14×6000=144000，但模拟output和最终boxed为156000；未真实执行代码，属于计算/模拟结果矛盾。
- 普通CoT / gsm8k:train:1203（v1=False、v2=False）：全文审核：1.75+1.25误化为2而非3，导致原款8而正确9；真实算术错误。
- 普通CoT / gsm8k:train:2645（v1=False、v2=False）：全文审核：染发净100已扣20，又从总260扣20，重复扣成本得290；应120+40+120−20+50=310。
- 普通CoT / gsm8k:train:5888（v1=False、v2=False）：全文审核：把月缺一天误说每周只五天，又算20工作日却用原240×4=960；按四周月应(24−1)×40=920，前后矛盾。
- 仅难度补充 / gsm8k:train:267（v1=True、v2=True）：全文审核：最终2正确、公式420分钟及42饼正确；桥句声称720分钟对应7小时错误，解释与后式矛盾。
- 仅难度补充 / gsm8k:train:5032（v1=False、v2=False）：全文审核：把平板充电53分钟误用手机26，且桥句肯定此错误；正确53+13=66，非39。
- 仅难度补充 / gsm8k:train:1203（v1=True、v2=True）：全文审核：最终9及1.75+1.25=3正确，但桥句说1.75是两项合计，和题意/公式矛盾。
- 难度＋attention / gsm8k:train:2005（v1=True、v2=True）：全文审核：方程x/2−18−9=1得56正确；第1步补充却称18杯是总量一半，与实际28矛盾。
- 难度＋attention / gsm8k:train:267（v1=False、v2=False）：全文审核：已算7小时可做42个，却未将x=42代回，最终停在44−x而非剩余2个；非截断。
- 难度＋attention / gsm8k:train:5032（v1=False、v2=False）：全文审核：把平板全充53误用手机26，再加手机半充13得39；补充说明引用手机支持平板，正确应66。

所有逐题审核见各evaluation_reviews.jsonl；原200未训练预测的前100审核分别复用dev_preview_audit.json与pilot50原80条，其余150条保存在本目录。这些辅助判断不重写主分，也不是独立人工盲审。

源题问题保留：2867速度关系应得40.5而gold23；2882按共享一加仑应剩0而gold4；5134逆百分比应为300/0.85，原gold983保留；7274的6%联合概率需额外独立性假设；1138的月累计与当周金额、3155的返还/比较措辞均有歧义。单位等价、错误抵消与正确最终数值但推理矛盾另记，不修改主分。

## 适用边界

- One seed; confidence intervals resample questions, not independent training runs.
- Exploratory paired comparisons are not adjusted for multiple testing.
- The development expansion was decided after inspecting the earlier pilot; not an untouched final test.
- Attention only highlights prior context for explanations at the same NLL-selected position.
- No hard-answer, soft-distribution, random-evidence or strict prompt/token-budget controls in this run.
- Same question exposure; unequal supervised token budgets and explanation lengths.
- No tool execution; any generated code/output is model text.
- Full-text assistant audit is auxiliary, not independent blind review or a mathematical proof.
- Source gold and wording issues retained without changing the frozen primary scores.
- Sensitivity parser v2 was developed after inspecting the first 20 development outputs; v1 remains primary and both are reported unchanged.
- Teacher is pinned community MLX4bit, original upstream conversion commit unspecified; student original BF16.
- Exact and lexical overlap checks do not exclude semantic templates or pretraining contamination.
