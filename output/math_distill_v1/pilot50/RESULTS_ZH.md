# 50题探索小试结果

状态：四条件共400条输出已全文辅助审核。100题为开发集，尚未进行1319题正式测试。

普通CoT在同100开发题的v2数字分由未训练的80提升到91；这支持该小样本训练有方向性收益，不能全归因于数学推理改善，格式风格也改变。
attention组相对仅难度组净+1.0个百分点，相对普通CoT净-2.0个百分点；这些是固定开发题上的结果。
尚不能证明attention带来可靠额外收益：只有一个训练seed，开发集已被查看，且没有随机线索与完整提示/长度预算对照。本版attention提供前文线索，不参与改变NLL所选位置。
保留解释重复、任务目标误读、漏条件、算术错误、单位等价和源gold问题；冻结数字主分不因辅助审核改动。

教师：Qwen2.5-Math-7B-Instruct的固定社区MLX4bit版本；学生：Qwen2.5-Math-1.5B Base原BF16。数据：官方GSM8K冻结划分，前50训练候选共同审核后48题，固定前100开发题；全部本地调用。

| 条件 | 原v1正确 | v2敏感性正确 | v1/v2 invalid | 长度停止 |
|---|---:|---:|---:|---:|
| 未训练学生 | 62/100 | 80/100 | 27/9 | 0 |
| 普通CoT | 91/100 | 91/100 | 0/0 | 0 |
| 仅难度补充 | 88/100 | 88/100 | 2/2 | 1 |
| 难度＋attention补充 | 89/100 | 89/100 | 1/1 | 0 |

v1与v2是同一输出的两种解析，二者差值不是训练收益。源gold速度题2867保留原23，数学审核认为应40.5，单列披露。

## attention组相对其他组的配对变化

| 参照 | v1净变化/改善/退步 | v2净变化/改善/退步 | v2 bootstrap95%差值区间 |
|---|---:|---:|---:|
| 未训练学生 | +27.0pp / 32 / 5 | +9.0pp / 15 / 6 | [0.0, 18.0]pp |
| 普通CoT | -2.0pp / 2 / 4 | -2.0pp / 2 / 4 | [-7.0, 3.0]pp |
| 仅难度补充 | +1.0pp / 5 / 4 | +1.0pp / 5 / 4 | [-5.0, 7.0]pp |

## 数据、训练与边界

共同有效训练题48/50，排除原因和每组监督token数见results.json。

| 条件 | 全文审核 | 一遍监督tokens | 训练秒 | 生成秒 |
|---|---:|---:|---:|---:|
| 未训练学生 | 100/100 | — | 0.00 | 509.51 |
| 普通CoT | 100/100 | 12934 | 53.52 | 769.71 |
| 仅难度补充 | 100/100 | 14397 | 58.76 | 825.90 |
| 难度＋attention补充 | 100/100 | 14385 | 59.71 | 803.46 |

3组均独立从同一原始学生训练，4轮、192微步、40更新；LoRA参数1,245,184。无旧adapter初始化、无padding监督、无截断。训练同题曝光，不是同监督token预算。

位置均由学生NLL启发式选择；attention在这个版本中为局部解释提供前文线索，尚未比较不同位置选择器。48对补充中15对解释相同，4对长度差超过16tokens（最大40），相关性和长度效应尚未完全分离。

材料准备保存的新生成共261次，约819.79秒，包括失败提示尝试及修正流式小数截断后的两组重生成；这些是本地调用，付费API支出0元，未核算电力和人工成本。该秒数不含模型加载、定位和人工审核，不能视为全部墙钟耗时。

源gold/措辞问题见results.json与全文审核；2867、2882等不改冻结主分。生成的Python/output均为文本，未执行工具；数值正确仍可能含单位错误、自我纠错或前后矛盾，辅助分类单列。

## 输出行为与全文审核

| 条件 | 平均生成tokens | 含补充桥句的输出 | 桥句多次出现的输出 | 代码风格输出 |
|---|---:|---:|---:|---:|
| 未训练学生 | 185.37 | 0 | 0 | 22 |
| 普通CoT | 281.97 | 0 | 0 | 0 |
| 仅难度补充 | 305.24 | 67 | 12 | 0 |
| 难度＋attention补充 | 296.93 | 67 | 11 | 0 |

桥句计数是文本行为指标，多次出现并不一概代表错误；具体重复与推理问题已逐条记录。例：仅难度组4130循环至1024 tokens而未完成；补充组2005多次复述题设但最终数值仍正确；5182把卖给两位朋友误算三根；700输出300美分等价3美元但数字主分拒绝。6481中普通CoT/仅难度组有真实算术错误，attention组则正确输出等价的2/3立方英尺，冻结数字分同样拒绝，不能将三者一律归为格式问题。

attention组还出现5032将平板充电时间误套手机、230把速度与时间反比写反；4441最终22正确但桥句把150+16说成168；4857最终196正确但桥句把240只幼鸽称为总鸽数，与后续280矛盾。1400最终22正确，但桥句解释加10却放在2×6之前，位置不对齐。以上均保持原分，保留具体文本问题。

辅助审核是本次助手对完整文本的复核，非独立盲评者或形式证明；各组记录绑定原输出SHA256，详见同目录*_evaluation_reviews.jsonl。

## 适用边界

- Single seed; intervals resample questions, not training runs.
- Question exposure matched; supervised tokens and explanation lengths differ.
- Development outputs were previously inspected and the sensitivity parser was designed on development data.
- Attention is a relevance heuristic; random and complete budget controls deferred.
- Teacher is a pinned community 4bit conversion; exact upstream original-weight commit unspecified. Student is original BF16; no paid API.
- No exact question overlap was found, but semantic duplication and benchmark pretraining contamination cannot be excluded.
- Numerical exact match does not validate the whole reasoning; full-text audit coverage must be read separately.
