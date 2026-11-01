# 新实验运行说明

2026-10-15 21:43最终断点（取代下文未完成状态）：本批完成，六次独立训练、六组同70题420/420评测、420/420全文辅助审核与既有base70/70审核全部齐全。最终报告reports_refinement_math_trainpilot21_v3/TRAINING_PILOT_REPORT_ZH.md已补接受率、主/学科分数、E配对区间及数学/格式解释、监督token、复用映射（均自身，无复用）、时间费用和局限；逐条报告TRAINING_OUTPUT_AUDIT_ZH.md、机器记录training_output_manual_audit.jsonl及分批01–07/summary保留。冻结主分base7/A13/B13/C17/D13/E14/F18（各70）；E−A/B仅+1且区间跨0，E−F−4，未达到扩大条件，不直接扩大本版筛选。E对A增分主要含格式/完成度，E对F四失都有答案数学差异，但F两题仍坏理由；单seed不能估训练方差。
六adapter/metadata/监督签名及评测adapter哈希、70有序ID、v3源码hash全部核查通过；无runner/teacher/MLX进程，不重跑训练/评测/merge或新请求。40条审核快照SHA256 1b9fd3bea51b55b32f7a0fc3a250eb6ecf5862095b5995ac90aa7d1f0deaad81、原34字节与408复测哈希保持。纠正旧断点：hints逐字段与reviewed40完全相同，但merge标准JSON空格序列化使hints字节SHA256为90014cae314a0730ac8413c7da816b1da1683608d68e09d9b76fb4825aa7e2d9；未改任一原文件。
费用含历史8.779300元，其中未知1.850596保留，没有新增teacher调用；不扩题、不多seed、不调冻结评分或阈值。1971原解答两处人工数学修正、宽泛提示与prealgebra236 independent术语局限、MLX一padding target、覆盖不足/非token匹配已披露。final_completion.jsonl保存完成证据与哈希，run_status_before_final_completion.jsonl保留旧状态。rationale在完整报告交付时按授权暂停，后续新实验需新授权。下文均为历史，不再执行旧命令。


2026-10-15 21:30后最新断点：全部六组独立训练和420/420训练后评测已于21:17:38结束，runner38173及GPU进程已只读确认退出，禁止重跑训练/评测。run_status=needs_final_audit，自动报告reports_refinement_math_trainpilot21_v3/TRAINING_PILOT_REPORT_ZH.md已生成。冻结主分Base7/A13/B13/C17/D13/E14/F18（各70题）；E−A仅+1题，E−F−4题，不作方法有效或稳定性结论。全文辅助审核已完成索引0–19的120/420条，保存在training_output_manual_audit_01/_02.jsonl；剩余索引20–69共300条，不能标整批完成。发现广泛UIControl等无关尾缀引发格式拒绝，另有符号、枚举、任务项数等真实错误，逐条记录并保持主correct不变。Base70已有完整审核不重复。费用仍含历史8.779300元，其中未知1.850596保留，无新teacher调用。继续余下全文审核→补完整报告/结论→同步文档并暂停rationale；不扩题不多seed。

2026-10-15 18:22阶段更新：六组A–F均于18:18:45前完成独立训练，全部adapters.safetensors与run_metadata.json齐全，每组168微步21updates。实际训练耗时A531.108/B522.037/C521.796/D522.698/E553.689/F578.503秒，累计约53分50秒，六组无复用。原runner38173已自动进入evaluate_six_groups，当前唯一GPU evaluate_mlx PID55143、matrix55137，A原解答已保存8/70预测；其余组尚未完成评测，不给部分分数结论。勿重复启动训练/评测或第二MLX。继续同70独立题评测→报告→全文辅助审核，不改冻结主分/参数，整批仍未完成。

2026-10-15 17:25训练已启动：两geometry98控制长度修复实审通过，generic48/E61差13（offset354）、random63/E61差2（冻结offset116）；random公共响应前置说明+唯一JSON已逐字免费恢复，hint_response_recovery_reviews_7.jsonl保留区间/hash，旧error/完整响应/费用不删。新增2请求成本0.107032元；共享新估算/预留6.229984、含历史8.779300元，未知1.850596保留。control_length_repair_reviews_1.jsonl逐条实审；merge --checkpoint controls_length_repair --expected-count40完成，勿重跑，原38字节保留。最终40审核快照SHA256=1b9fd3bea51b55b32f7a0fc3a250eb6ecf5862095b5995ac90aa7d1f0deaad81，hints一致且全部verification local hash不变，旧账本字段/费用校验保留。
六组builder严格通过，每组17题：A变0，B/C/D/E各变3，F变17，六组有序监督SHA均不同，condition_reuse均独立，必须实际训练6次。训练前102条真实token审计通过，最大1317≤4096、无截断、assistant-only；一遍监督目标token A9194/B9290/C9338/D9328/E9328/F9926，同题曝光非token预算匹配。MLX inclusive mask可能监督1个padding目标，审计已披露。每组168微步/21updates、约9.882实际等效轮，seed7/冻结LoRA配置不改。
已启动runner train-evaluate PID38173，日志train-evaluate_pipeline.log；token audit完成，当前train_six_groups/A original_rationale已进入Training，19.399M可训参数。不要重复runner或并发第二MLX，旧teacher35807已退出。后续流水线自动六组训练→同70独立题greedy无hint评测→汇总/报告，结束仍需完整辅助人工输出审核，不改冻结主分。rationale ACTIVE负责接续，阶段推进/失败/需决定/完成才通知，完整报告后暂停。尚无已完成adapter/训练效果，不把启动当成功。

2026-10-15 17:17恢复：用户今日明确“今天继续，开始训练”，解除下文昨日暂停。已核查无GPU/教师进程，并冻结两条geometry98长度修复control_repair_requests_1.jsonl（新ID为旧:length_target:1追加:repair:length:1，旧付费原响应/拒绝记录/费用保留）。只读连接98models、0生成，预检control_repair_preflight_1.jsonl：当前共享新6.122952元、最多新增预留0.204732元，含历史最高8.877元≤42，旧未知1.850596保留。唯一teacher PID35807，controls_length_repair_pipeline_1.log；先实审两响应数学/位置/无泄漏和相对E61tokens差≤16，锁释放后新checkpoint合并40条，再六组build/token审核与train-evaluate。38条旧审核SHA bfeebb1de7f903fb90c8cadf3690cd9305f8c03ca964249e86193748e6bfc89c及34local/408verification冻结不改。不重复旧/新ID，不人工扩写付费文本，不放宽容差。rationale已恢复ACTIVE，仅阶段推进/失败/需决定/完成通知，整批报告后暂停。尚未训练。

2026-10-14 23:22今日停止断点：按用户23:05“先别开始训练，完成这一步今天到此为止”，已完成且退出当前控制teacher队列12159，共6新请求全部保存stop响应/已知usage，生成累计1194.716943秒（约19分55秒），本队列成本0.392420元。6提示均全文对照题面/原解答/Asymptote、数学/位置/无泄漏实审：4控制通过、2几何控制因长度不匹配拒绝，不追加收费修复、不人工扩写原响应、不放宽16容差。geometry98 generic25/E61差36、random37/E61差24待后续；IA1229 generic26/E42差16、random48/E42差6合格；prealgebra392 generic22/E31差9、random33/E31差2合格。两IA1229格式失败的完整公共响应含两份完全相同JSON，按顺序固定首个免费恢复，0新增API，溯源hint_response_recovery_reviews_5/_6.jsonl、原响应/error/usage/费用保留。
control_hint_review_progress_1至_4保存6条实审；reviewed_control_hints.jsonl保存4合格；rejected_control_hint_reviews.jsonl保存2长度拒绝。merge_refinement_hint_reviews新增显式control拒绝来源校验；已在锁释放后执行--checkpoint controls_day_stop --expected-count38，勿重复此冻结checkpoint。合并34local+4合格control并免费恢复2IA控制，原2不合格付费候选保留paid_hints_before_controls_day_stop.jsonl；ledger_before_controls_day_stop.jsonl/hint_merge_controls_day_stop.jsonl溯源。最终reviewed_hints/hints38完全一致，SHA256=bfeebb1de7f903fb90c8cadf3690cd9305f8c03ca964249e86193748e6bfc89c，最初34local原字节SHA256=b404d3219be47c77dce0de8c67802f2c2162bb97794c0bc38ee63325f851e71b及全部verification local hint_hash不变；所有旧账本字段/原响应/错误/用量/费用均校验保留，仅两条控制追加免费恢复status/provenance。共享新估算/预留6.122952元，加历史共8.672268元，其中旧未知1.850596不释放。
control_review_summary.jsonl保存完整收尾；run_status原needs_controls快照在run_status_before_day_stop.jsonl，当前run_status=paused_by_user/control_queue_and_review_finished，training_started=false、ready_controls4/pending_control_repairs2，整批未完成，六组未build且尚不具备完整可训控制集合。已确认无teacher/GPU/训练进程。本自动化rationale按用户指令PAUSED，今夜不自启，等待用户明确继续。后续若用户恢复，先处理几何2长度控制（旧响应与费用保留、source/kind核查、不重复旧ID，新修复ID及预算/连接追踪），完整builder/token审核之后才训练；不要因旧授权要求整批完成忽略今天的新暂停指令。


用户2026-10-14 23:05最新明确指令：先别开始训练，完成当前这一步后今天到此为止。这条取代此前本次接续必须继续训练评测的要求。立即禁止启动train-evaluate、train_mlx、run_mlx_matrix训练、任何新GPU/训练进程；不标整批完成。当前控制teacher PID12159已在运行，仅让已发送/既有6请求队列完成并审核保存响应/数学/位置/长度，保留不合格/失败/费用及旧未知预留，不自动重试、不在今天扩大请求或切换模型。停止点为当前控制队列收尾及审核记录；如果控制长度/数学不合格明确留待后续，不能绕过builder或称数据已全部可训。只有6条控制全部通过且锁释放时可离线合并/构建六组，但不启动GPU token audit或训练。今天结束后保留断点等待用户明确继续。本接续按用户指令暂停，不能因旧完整任务未完自行恢复。


2026-10-14 22:57最新阶段：408/408独立复测于22:55:03完成，旧runner95421/GPU95423已确认退出，不重启；当前无GPU任务。全17题source_hash、本地原文hash、34local snapshot/hint_hash、fresh8种子匹配及与screen独立校验通过，verification.jsonl SHA256=78d4226f12f7007813bd0b7f2f9f68c0342a967d673cfe4ffcdb6a675bf80a69。复测生成累计8144.813839秒（包括136baseline与272hint）；baseline84/136主正确，两个local合计146/272，不能把接受题的正增益当整个池普遍改善或独立训练效果。按冻结0.25和builder相同tie-break接受3/17：geometry98 local0，baseline0/8->3/8，61tokens；intermediate_algebra1229 local1，4/8->7/8，42tokens；prealgebra392 local1，4/8->8/8，31tokens。verification_acceptance.jsonl固定全部17决策；不调阈值、不换题。
只对3接受题保存6条generic/random于control_requests.jsonl，基于未发冻结teacher_requests加明确长度目标E±8（最终builder容差16），新ID各原generic/random ID+:length_target:1，replaces_request_id关联原未发请求、derived_from_frozen_unsent_request=true；原未发请求hash不改。冻结控制前核查新旧same-kind无账本/响应，ledger锁已释放，control_request_preflight.jsonl记录预算，0API准备；连接只读ok98models、0generation、账本不改。已启动唯一teacher控制队列PID12159（已确认存活），日志controls_pipeline.log，记录control_queue_launch.jsonl，仅发6请求，不重复旧/新ID。API持共享锁期间不要编辑hints/账本。启动前共享新5.730532+历史8.279848元，最多新增预留0.596476、合计最大8.876324<=42；旧未知1.850596保留。
merge_refinement_hint_reviews工具仅新增加载control_requests.jsonl/control_repair_requests_*.jsonl来源验证，未改审核/冻结策略；语法检查通过。控制响应到达逐条读取题面/原解答/位置/完整原response，真实审核数学、无泄漏、<=128tokens且E长度差<=16；不能自动标true、不能因数学答案对就忽略坏理由。优先免费恢复格式问题，数学或长度必要修复必须核查保存响应/连接/费用后新ID，旧响应/费用保留。控制实审记录追加reviewed_hints时原34字节/hash必须保留（b404d3219be47c77dce0de8c67802f2c2162bb97794c0bc38ee63325f851e71b），旧每题local hint_hash保持。队列退出锁释放后用独立新checkpoint合并预计40条；若同kind替代有旧active候选需要显式拒绝追踪，不能重复收费。六组builder通过后runner train-evaluate先真实token审核再六组训练/独立70评测。尚未训练无提升结论。runbook及workspace同步，此复测完成/3接受/控制启动阶段已通知，普通无变化静默。


2026-10-14 22:08最新阶段：前9题144次hint复测完成，旧GPU80521和CPU等待器91305已确认退出，不重启。接续器在22:06:27成功校验前9题完整缓存、无其他MLX后自动启动完整verify runner95421/GPU95423，当前仅GPU95423，日志verify_pipeline.log、full_verification_dispatch_2.log；启动交接full_verification_handoff.jsonl记录复用136baseline+144hint、仅新增128。最新保存136baseline+146hint=282/408、9题全部三臂完成，全部source_hash/fresh8匹配seed/hint_hash及34审核快照SHA256校验通过，原结果不重采样。当前GPU继续、勿启动第二完整verify/训练或重启旧PID。34提示/原响应/费用保持，teacher无在途，共享新5.730532+历史=8.279848元，未知1.850596不释放。完整结束后先核查408/seed/hash，再冻结接受集合并只为接受题发generic/random控制、实审长度匹配，再六组build/token审核/完整训练70独立评测。尚未训练、不按部分复测改阈值、教师或超参。用户询问R1/Flash，仅讨论下批同题对比，无本批切换授权；冻结本批仍R1。runbook与workspace同步，完整接续已通知，普通无变化静默。


2026-10-14 21:54接续器修正：首CPU接续器90737因启动时解析.venv/bin/python符号链接，绕过venv依赖并缺yaml立即退出，未启动GPU/未改采样缓存/0API。失败日志full_verification_dispatch.log及启动记录保留。已核查venv路径import yaml/common正常，修正为保留.venv/bin/python路径，启动新CPU等待器91305并确认存活且无报错；日志full_verification_dispatch_2.log，记录full_verification_dispatch_launch_2.jsonl。该等待器沿用/private/tmp/fyp_trainpilot_verify_after_subset.py，仍等待原GPU80521结束、前9题完整144hint缓存校验后自动runner verify；勿重复启动。最新已有136baseline+108hint=244/408保存，当前仍一个GPU80521。新34提示及费用/未知预留按21:52断点；未训练。


2026-10-14 21:52最新断点（取代21:44修复在途描述）：必要数学修复87637已结束，34条local全部逐条实审合格并完成merge_refinement_hint_reviews --checkpoint local_repair --expected-count34，不要重跑该冻结checkpoint。reviewed_hints/hints完全一致，每17题各2条，16–83实际student tokens；34快照SHA256=b404d3219be47c77dce0de8c67802f2c2162bb97794c0bc38ee63325f851e71b。前33条原始字节hash81572247463beca3ef16268cfc6e35dd897af2e487dc049c2b329672a63d5cf5保持，现有复测hint_hash一致；ledger_before_local_repair与当前账本每条原字段/响应/费用完全相同。新修复ID trainpilot21_v3:math_train_intermediate_algebra_1971.json:local:1:repair:positive_ratio:1成本0.039808元、53tokens，n/x=9/10>0与x>0推出n>0，位置461无答案泄漏。旧拒绝提示/错误/原付费全文保留于paid_hints_before_local_queue及rejected_local_hint_reviews；不要重发旧/新ID。7免费恢复仍全部溯源、无API重发。hint_merge_local_repair.jsonl、paid_hints_before_local_repair.jsonl、ledger_before_local_repair.jsonl记录最终合并。当前共享新估算/预留5.730532元，加历史8.279848元，未知1.850596元保留，无teacher在途。
唯一GPU80521前9题复测仍运行（约105/144新hint采样），不要重启。已启动CPU接续器90737，/private/tmp/fyp_trainpilot_verify_after_subset.py，日志full_verification_dispatch.log，记录full_verification_dispatch_launch.jsonl：等待80521退出，验证前9题144hint+136全体baseline、匹配seed/源hash/hint_hash、无其他MLX进程后，写full_verification_handoff.jsonl并自动启动完整runner verify --background，仅余128hint采样。不要另启动完整verify或其他GPU；接续时先查90737/80521、full_verification_handoff与run_status/verify_pipeline.log。若接续器失败先读日志查原因，确认所有GPU退出后可正常runner verify续跑，不能重采样已保存结果。接续器最长等1小时，未结束就报失败且不启动第二GPU。完整408之后按冻结阈值接受集合、控制提示/实审、六组build/token审核/训练独立70评测，任务尚未完成、尚未训练无提升结论。此提示审核完成与安全接续已通知，无变化静默。


2026-10-14 21:44最新断点：34local队列于21:35:31结束，原locals66381/teacher66384已确认退出，不重启。34条响应均保存，7条JSON格式失败全部免费逐字恢复并审核、1条1971local1理由不足拒绝，33条实审合格，实际16–83tokens。reviewed_hints SHA256=81572247463beca3ef16268cfc6e35dd897af2e487dc049c2b329672a63d5cf5，旧31条/22条及已verify的各题hint_hash不变。merge_refinement_hint_reviews新工具已成功锁内合并33条到hints，先冻结paid_hints_before_local_queue.jsonl（27原候选含被拒绝1条）和ledger_before_local_queue.jsonl，hint_merge_local_queue.jsonl追踪；原账本7条追加免费恢复溯源并complete，所有原response/error_type/费用和未知预留逐字段校验保留。恢复审核分布hint_response_recovery_reviews.jsonl及_2/_3/_4，共7条；原JSON文本及重新编码字段的hash已校验，不再收费恢复。prealgebra236local0的independent术语不精确，审核注释限定为两阶段可选数量恒定的乘法原理、不是概率独立，最终报告披露宽泛/冗余提示及术语局限。
1971local1已核查完整已保存JSON不能免费得到数学合格原文、只读连接ok（98models、0生成）、预算通过。保存local_repair_requests_1.jsonl和local_repair_preflight_1.jsonl，只有新ID trainpilot21_v3:math_train_intermediate_algebra_1971.json:local:1:repair:positive_ratio:1，明确正比值+n/x与x>0共同推n正、旧提示/费用/候选保留。已启动唯一API修复PID87637，日志local_repair_pipeline_1.log，不重复发此新ID，不自动重试；teacher持锁时不改hints/账本。修复结束先免费恢复可用格式响应、实质审核数学位置无泄漏token，追加reviewed_hints新记录并保留已有33条hash，再用merge_refinement_hint_reviews --checkpoint local_repair --expected-count34合并（若未补回不能标完整池）；先查恢复/新ID已保存状态，不能以新ID认重复缺失。
唯一GPU仍为前9题提示复测80521，最新约90/144新hint生成，136baseline均复用。后续完整runner verify必须确认80521退出，复用已有hint_hash与fresh8结果；不并发第二GPU。当前共享新估算/预留5.828932，加历史最保守8.378248元，未知1.988804含本次在途0.138208与旧1.850596，全部保留。stage仍数据准备/复测，未训练无提升结论。下一步修复审核完、GPU子集结束后完整408复测续跑，控制解释/六组build/token审核/完整训练独立评测，不能停在提示测试。此队列结束与修复启动已通知，普通无变化静默。

21:25例行接续：原locals66381/teacher66384与唯一GPU80521均正常保留，提示复测已保存27/144次。新增1030 local0实审合格66tokens，reviewed_hints共22条，SHA256=5213fad05bef836fc2779d424ef14979cffa568d0802c47b9bbfcc04cb35cd9c；此前21条字节hash不变，已开始verify的各题hint_hash校验一致。无新队列/收费修复/训练启动，无阶段改变，按通知规则静默。其余接续按21:23断点，尤其1971 local1待修复、四条免费恢复在两个恢复审核文件中，teacher锁释放前不合并账本/hints。

2026-10-14 21:23最新推进：无提示baseline136/136完成，84主评分正确，生成总2458.4224秒；这是训练题前缀续写，不是70独立评测或训练效果。旧GPU66416已确认退出。已后台启动已审核前9题提示复测PID80521、144新hint续写、复用72baseline，日志verify_reviewed_subset_pipeline.log，启动记录reviewed_subset_verification_launch.jsonl；当前唯一GPU，后续任何完整verify/训练须确认80521退出，复用全部136baseline和已有hint结果/hash，不重新采样。locals66381/teacher66384仍运行，不重启。reviewed_hints21条实际16–62tokens，SHA256=b7358731012d2306d45af31226f4525b8d410cef8a8ce786ec079a564bba4583；此前19条字节hash21acfe5a8e5536a0c3eb546818ef750130e185c62879d2640af00ca6658ff44e保持，包括自动审核拒绝过的时间戳，不再修改。四条格式异常已免费审核恢复：393local0、911local1，以及1229local0重复相同JSON/1229local1唯一JSON原LaTeX未转义；后两条恢复记录在hint_response_recovery_reviews_2.jsonl，前两条在hint_response_recovery_reviews.jsonl。teacher锁期间未改hints/账本，结束后按这两个恢复文件合并审核快照和追加原账本恢复溯源，原错误/全响应/费用保留，不重复收费。1971local1虽数学结论n>=1正确，但首句仅用x>0和x0未定义推n非零，缺正比值关系，按与原解答相同标准拒绝、不标true；原记录及SHA256在rejected_local_hint_reviews.jsonl，不能verify它，队列结束核查连接/保存响应/预算后准备新可追溯修复ID，旧候选保留，不用新ID认重复缺失。1971local0实审合格。当前共享新估算/预留5.365004，加历史7.914320，未知1.972988含在途0.122392与旧1.850596不释放。runbook及workspace状态同步，尚未训练，继续完成小批六组训练独立评测；本阶段推进已通知，无变化静默。

2026-10-14 21:08接续：locals runner66381与baseline GPU66416仍运行，不重复启动。已实审16/34条local（前8题完整），实际token16–62，reviewed_hints.jsonl SHA256=878aa30cfdcff428e6fa7e086699125bb93da33fe4ea2a7f5c71f7e81ac0e9e6；之前9条/7条内容及字节hash不变。393 local0完整公共响应前置说明+唯一JSON已免费提取，911 local1完整响应含3个JSON，人工全文核查三候选后在任何hint复测前按响应顺序固定第一条，未依据gain择优，余2条在hint_response_recovery_alternatives.jsonl留档，不增加local数量。两条原响应/失败/已知费用保存，hint_response_recovery_reviews.jsonl记录原响应hash、提取区间、0新增API；teacher持锁期间账本/付费hints未改。队列结束后必须把这两条已冻结审核记录合并到hints并为原账本条目追加免费恢复溯源，保留旧错误/全响应/所有费用；不要收费新ID重发可恢复项。baseline保存100/136次，后续GPU先确认66416退出；可用已审核快照对完整前8题verify --limit8，必须复用fresh baseline且保留hint_hash/内容，此后完整verify续跑不重采样。共享新估算/预留5.010884元，加历史7.560200元，未知1.960764含在途0.110168与旧1.850596，不释放。部分提示只是通用/宽泛规则，不能称诊断到真实概念缺口或有收益；尚未训练，无提升结论。runbook与workspace状态同步。继续全部小批训练评测，普通无变化保持安静。

20:56接续：原locals66381与baseline66416保留。新增360 local1实审合格22tokens；393 local0返回完整stop响应但因前置说明+末尾唯一JSON解析失败，已全文核查并免费提取JSON、用原请求parse_hint验证，31tokens、约分规则正确、位置285合格且不泄漏最终分数。独立reviewed_hints现9条，SHA256=5d88bd4069e9dc7ea6088cea402f92dc7b7f0179146a3e333d651099c52a8ad1；此前7条字节hash不变。原响应/失败/已知费用0.070288保留，hint_response_recovery_reviews.jsonl记录响应SHA256及提取区间，0API重发；teacher锁期间账本/付费hints未改，队列结束后需合并这条审核快照并追加免费恢复溯源，不能新ID重复收费。baseline已保存73/136次，仍不并发第二GPU。当前新共享估算/预留4.668592，加历史7.217908，未知1.963312含在途0.112716与旧1.850596，未释放；训练未开始。

20:49接续：原locals PID66381与独立baseline GPU PID66416均仍在运行，无重复启动。已返回7/34条local，7条均对照题面/冻结原文/插入位置实质审核，并用实际学生tokenizer验证25–52 tokens、数学正确、无最终答案泄漏；单独保存reviewed_hints.jsonl，付费hints和共享账本未修改。七条快照SHA256=f37c68d55fa21d2fb037fa74e20b4a5dae33d19f91af10123b97f0e3768e2a31；后续只追加新实审记录，已有记录内容/hash不变。代数2425/325提示是通用公式强化，其screen无失败，不能称已发现概念缺口或已有收益。baseline已保存44/136次，尚未结束、不能启动第二GPU任务。当前共享新估算/预留4.552428元，加历史7.101744元；未知预留1.954460元包含在途0.103864和旧未知1.850596，不释放。未训练，继续完成本批所有阶段。

最新阶段20:37：204次screen于20:35:39结束，主正确140、失败64、上限9；64失败全部全文核查并绑定hash，判对140条未全部人工逐步审核。快照screen_review_summary.jsonl，详情SCREEN_REVIEW_ZH.md；原PID50702退出。已启动locals runner PID66381仅发34local（68请求冻结），同时独立baseline verify --baseline-only PID66416采样136次，日志verify_baseline_pipeline.log、启动记录baseline_verification_launch.jsonl。只一个GPU进程，完整verify/训练须等baseline退出，同fresh种子结果后续复用，不重复采样。teacher锁期间不改账本/hints，到达解释逐条实质审核。启动前历史+新队列保守总估算10.517616≤42元，未知旧预留不释放。尚未训练。

20:26原screen进程继续，20:25已有158/204次、13题完成；55条评分失败实质审核及hash记录在screening_failure_review.jsonl，详见SCREEN_REVIEW_ZH.md。数论336有素性/奇偶/无序计数及虚构代码执行结果错误，1030有进位/列约束错误，715须区分正确等式格式拒绝与位系数输出错误。不改主correct/阈值/题目，完成筛查后补齐其余审核再发local；未启动训练。

20:08筛查原PID50702继续，已保存98/204次、8题完整。25条评分失败已全文逐条审核，screening_failure_review.jsonl保存source_hash/offset/seed/输出hash/数学与格式区分，SCREEN_REVIEW_ZH.md说明进行中边界；未改变correct/阈值/数据选择。包括真实错约分、Stirling递推漏k和距离方程错误，也有正确推导被格式拒绝、近似数及生成循环。后续需补齐其他失败审核后才能冻结local请求；未生成local、未开始训练。

最新断点（2026-10-14 19:49）：最后原解答队列已结束，原PID43681/43683退出。17条完全审核的原解答已通过freeze_reviewed_trainpilot冻结为bases.jsonl（base_freeze_manifest.jsonl记录hash/覆盖）；固定21题中源题矛盾821及保存响应不完整的6200/72/884排除，不换题、不再付费修复原解答。884最终数值正确但缺开头且截断，无三条合格非泄漏边界，不能说最终答案算错。17题学科分布3/3/2/2/3/3/1，七学科均有覆盖但难度不齐。51个真实token前缀最大565，含生成上限2613≤4096。

screen runner PID50702已启动，204次冻结位置续写，日志data/refinement_math_trainpilot21_v3/screen_pipeline.log。当前训练未开始，无训练后提升结论。接续应保留该进程，结束后逐条审核失败、生成34local并实质审查，独立408复测，再按接受集合生成控制解释、构建六组、token审核、完整训练/70题无hint评测。17有效行预计168微步/21更新，实际以metadata为准。共享新估算/预留4.191468元、加历史共6.740784元，未知预留1.850596元不释放。审核与排除详见ORIGINAL_REPAIR_REVIEW_ZH.md及rejected_original_response_reviews.jsonl；人工数学修正继续披露。

## 2026-10-14新授权：完整七学科小批次训练实验

新配置`configs/refinement_math_trainpilot21_v3.yaml`、根`data/refinement_math_trainpilot21_v3`。21训练题七学科各难度3/4/5一题；70独立dev各学科10题、难度1–4，不能称全难度覆盖。完整协议在新根PILOT_PROTOCOL_ZH.md。用户此次明确要求完整训练与评测，先前“不训练”的两题阶段限制已结束。

bootstrap runner PID19014启动原解答16请求与base70题评测并行，复用旧5解答，2旧超时补发有新ID与replace关系。所有费用继续共用原总¥42，失败预留保留。数据审核后3边界x4筛查，local2/题，baseline/local各8独立复测，阈值0.25与v3固定。local提示在新请求前强调具体错误规则，旧两题不重写。

source intermediate_algebra821严格正数条件与gold零变量等号条件冲突，已在本次teacher该题返回前排除、不换题；有效训练目标至多20，道数和难度覆盖据实报告，七学科仍覆盖。probability360按题要求把金额gold规范为0.67并保留原混合字段与说明；已核查草稿在draft_bases_reviewed.jsonl，质量说明在DATA_QUALITY_REVIEW_ZH.md，不能只看candidate.is_usable字段丢掉这条正确teacher回答。

18:46接续：原队列PID19016保持运行，已核查11原解答草稿、保留先前hash，未提前冻结bases。base70/70完成，冻结主评分7/70、无解析答案44/70；辅助分类61条全文加9条首尾片段，在reports_refinement_math_trainpilot21_v3/base_output_manual_audit.jsonl绑定输出hash，说明见BASE_OUTPUT_AUDIT_ZH.md。8条正确计算被占位或标签等拒绝，任务漂移35条、仅回显11条，不能把低分全部归因数学理解。原队列5超时（含已排除821），无响应可恢复，不自动重试；共享历史+新估算/预留5.483440元，队列释放锁后再处理必要显式修复。本批筛查、local与训练仍未启动，不能称有训练增益。

18:47草稿累计13条：新增prealgebra392已按图形代码核查120+90=210，原响应孤立think结束标记与重复解答保留，草稿规范化为标记后完整解答并保存原文本hash；三边界在方形角之前。1164容量题各步正确，边界在通分求解前。已有11条不变。

18:51草稿14条，新增precalculus359倍角方程题，原think响应保留、草稿规范化并固定非泄漏边界；已有13条不变。基线9条片段的中段已补读，70条全文审核类别不变，仍不改主评分。独立源题核查在新根PENDING_SOURCE_REVIEW_ZH.md，不代替teacher审核。

18:54原队列仍在，72新增超时、884在途。只读连接检查成功。4条此前必要失败已显式保存original_repair_requests_1.jsonl但尚未发送，待原队列锁释放复核后发；72还需另行核查保存新ID，勿重复原4条修复。详细新ID和预算说明在ORIGINAL_REPAIR_REVIEW_ZH.md。

19:01最新：首轮18:58结束，14原解答已审核，7超时中821排除、6有效题必要修复。已核查无响应、锁释放、新ID未发送、连接与预算通过；6条显式合并original_repair_queue_1.jsonl启动repair-originals runner PID35756、teacher PID35758。不能重复发送原4条或当前6条。启动前历史+新估算/预留5.762968元、最保守新增0.633332元；仍未筛查或训练。详细断点与追踪见ORIGINAL_REPAIR_REVIEW_ZH.md；新增只读连接/后台修复阶段的相关34项回归通过。

19:11概率911补回并逐步审核，草稿15条、原14条不变；三个完整前缀在S52/S53和最终和之前。几何6200本次修复仍超时，未知预留保留、未重复发。PID35756/35758继续处理剩余，共享历史+新估算/预留6.046996元含在途；完整训练目标仍在推进，尚未冻结bases。

19:25最新：6修复2返回4超时，草稿16条。取整1971两个错误正性理由已新候选显式人工修正，原响应不改，最终报告必须披露（ORIGINAL_MATH_CORRECTIONS_ZH.md）。四缺失无响应、锁释放、连接/费用检查通过，最后一轮original_repair_queue_2.jsonl以新ID与600秒传输等待启动PID43681/43683；不改模型或student协议，不自动重试。启动前历史+新6.342960元、最多额外0.424840元。此轮结束仍缺失就记录覆盖不足、冻结实际审核16–20题继续训练全流程，不无限重发、不换题。详情ORIGINAL_REPAIR_REVIEW_ZH.md；尚未筛查/训练。

19:31geometry6200最后响应数学错误并截断，全文审核hash保存在rejected_original_response_reviews.jsonl，原response_content与已知用量费用0.101008元保留；不免费拼补、不再收费补该题。实际有效上限19、已有16题，几何仅level3/4、七学科仍有覆盖。原PID43681/43683处理余三题，仍不提前冻结。报告模块加入实际覆盖/排除以及人工数学修正的披露。

六组seed7、同8B Base独立LoRA初始化、10轮题目曝光（21题208微步26updates），不称token-matched，不用评测挑epoch。最后比较训练后无hint70题准确率和未训练base；完全相同监督条件显式复用并披露，避免把标签不同当不同干预。run_refinement_trainpilot分bootstrap/screen/locals/verify/train-evaluate后台阶段，遇审核点需实质阅读审核，不自动标true。接续任务rationale已恢复为完成这整批训练评测和报告后暂停；不能只完成解释检查就标成功，也不自动扩大。

## 两题检查完成：未见稳定净收益（2026-10-13 23:33）

24次筛查、48次独立复测完成。代数baseline/local0/local1均8/8；概率分别6/8、5/8、6/8。原增益0.25下接受0/2。人工阅读概率24条后，核心数学正确且无错约分/漏计三组均7/8；两个解释均修复同一漏计，但分别产生另一个错误约分。解释一自动判对的一条还含错误等价答案，正式分数保留。无训练、无六组比较，不能外推方法无效或训练提升。

完整报告`reports_refinement_math_quick2_v3/QUICK_RESULT_ZH.md`，机器汇总及逐输出审核在quick2数据根。72次本地采样17分22.5秒，无生成上限触发；teacher4条合计10分23.7秒等待，部分并行，总计至汇总约25分。此次提示¥0.173992；共享旧+新估算/预留¥4.571704，其中未知预留¥0.687492不释放。原七题队列5成功2超时，不自动重复请求。完成两题报告后暂停接续。下面条目为运行历史。

## 当前优先：两题局部解释机制检查

用户要求先跑几个题看方法效果。`configs/refinement_math_quick2_v3.yaml`与`data/refinement_math_quick2_v3`复用已付费且核查的代数2425、概率393原解答，不等七题完整加工；原队列已结束，5成功2超时，未知扣费预留保留，不重试。七题后续筛查暂缓。

两题已完成24次筛查：代数12/12、概率5/12；固定最差候选偏移188与285。概率7个失败中3个数学错误、4个正确推导后的答案表达问题，详情见quick2根`SCREEN_REVIEW_ZH.md`。独立baseline每题8次已完成：代数8/8、概率6/8；后者1个真实数学错误（把C(4,3)误算0）、1个答案表达问题。

只先付费生成4条local解释，ID以quick2_v3开头，保存完整未发送控制请求于teacher_requests、实际发送local队列于local_requests。所有批次共享原预算。解释审核后各候选8次匹配种子复测；无训练、无六组构建。`diagnose_refinement_mlx verify --baseline-only`允许提前采样无解释对照，完整verify会复用这些独立baseline，不重复生成。评分代码与阈值固定，两个候选全部报告。两题便利样本只能提供提示辅助机制证据，不代表训练或泛化提升。

接续任务rationale已改为优先完成两题、报告后暂停，不自动继续七题付费加工或训练。

## 2026-10-13 23:00 第二批校准已启动

用户授权继续。新配置`refinement_math_calibration7_v3.yaml`、新根`data/refinement_math_calibration7_v3`：7道训练侧题覆盖7学科、难度3–5，选题不看模型输出。原始R1队列已于22:53启动，PID95776，最多7新请求，共享原总预算；22:59有2道返回正确并实质审核。

v2保持原样，新校准使用`first_answer_v3`，支持多行LaTeX/boxed和markup，停止新题。协议冻结校验和与57项测试记录在新根protocol_frozen.json；旧144条仅离线回放修复7个格式误判，不改旧分数。单次无辅助探测1/7正确，其他多为跑题/未完成，不当作理解失败，也不据此换题。初稿探测保留debug，冻结探测为unassisted_probe_v3_frozen.jsonl。

共享旧+新账本费用/预留截至22:59约¥4.109712；仍共用¥42。10分钟接续任务已恢复，完成这7题校准后暂停，不自动启动训练。详情见新根CALIBRATION_PLAN_ZH.md及工作区FYP_PHASE2_STATUS_ZH.md。以下为已完成首批与历史记录。

## 2026-10-13 22:34 首批加工完成

4题、16条解释、48次筛查与96次独立复测完成，六组各4条已冻结。接受0/4，A–E解答内容相同，F保留4条未筛选解释，不能据此做位置与扩写优势结论。协议失败明显混有格式问题，下一校准批次需新版本评分与跨学科/难度抽样；本批分数不变。不启动正式训练、不扩大付费样本。

新账本费用/未知预留¥1.358392，含旧历史合计¥3.907708；其中未知预留¥0.473580保持占用。144次学生生成合计952.60秒，2次达到上限；从21:36筛查至22:33构建约57分14秒。完整报告：`reports_refinement_math_pilot5_v2/PHASE1_REPORT_ZH.md`；机器汇总：`data/refinement_math_pilot5_v2/phase1_summary.json`。首批完成后暂停接续任务rationale。下面时间条目为历史进展。

## 2026-10-13 22:25 重新连接后的进展

原16条解释队列已结束，12条解释已取得并完成内容核查（其中2条免费恢复保存响应），最后一题4条连接错误解释正在新ID修复队列中补齐。配置仍为 `refinement_math_pilot5_v2.yaml`，不启动训练、不扩大样本。原失败费用预留保留；初始队列结束时新账本费用/预留¥1.243052，含历史共¥3.792368，修复继续使用同一账本。

48次筛查完成，生成时间266.70秒。首题24次独立复测完成：baseline8/8、local0 8/8、local1 7/8，E不接受。其余题继续复测。唯一筛查“错误”已有正确boxed20但协议失败，不能声称真实推理困难。

解释核查详见 `data/refinement_math_pilot5_v2/EXPLANATION_REVIEW_ZH.md`。修复请求：`repair_requests_1.jsonl`；修复日志：`repair_pipeline.log`；已核查快照：`reviewed_hints.jsonl`。恢复模块 `recover_refinement_responses.py` 不发送API；显式修复队列准备模块 `prepare_refinement_repairs.py` 不发送API。后台已有本聊天10分钟接续任务完成首阶段后暂停。

## 2026-10-13 启动状态

用户已授权开始运行。首批五道 MATH 原始解答请求已发送：四道返回且答案正确，一道 API 超时；没有重复请求，未知扣费保留预算预留。共享新账本当前记录/预留合计 ¥0.333288，新增额度暂余 ¥39.116712（不含旧历史 ¥2.549316）。

初始计时发现旧格式检查误判 `x=2`、加粗 Final answer、数学括号，以及答完后继续编题。新增 `answer_protocol.py`：接受这些答案格式，评分第一条最终答案，在其完整行结束后停止生成。新主配置与诊断统一采用 `first_final_line_v2`。旧 timing 和 v1 pilot 数据保留为调试记录，不作为方法结果或与 v2 混用。

四道解答的步骤已核查；候选边界调整到完整步骤之前，剔除已经给出最终答案或可直接抄出答案的前缀。这次四道题都是简单代数题，只用于计时和流程检验，不能代表完整分层抽样 pilot。

正在运行的配置：`configs/refinement_math_pilot5_v2.yaml`。后台 runner 为 `run_refinement_pilot.py`，顺序执行四题 student screening、保存解释请求、生成最多 16 段 teacher 解释。所有付费调用仍共用预算账本；解释生成完成后状态标记为待解释审查，随后才进行独立验证与训练。

实时状态：`data/refinement_math_pilot5_v2/run_status.jsonl`。
日志：`data/refinement_math_pilot5_v2/pipeline.log`。
后台进程绑定 caffeinate 抑制闲置睡眠，worker 退出后释放。

运行时间初估：小批量完整 pilot 约 1–2 小时；100 题 pilot 约 1–2 天；完整 1,000 题/六组/三种子方案为数天到一两周量级。简单题测量不能外推所有难题；原始解答、续写、解释、人工核查、训练与评测都计入总耗时。¥42 合计预算能加工的正式题数仍需用 pilot 测定。

## 当前状态与旧实验的关系

旧实验目录就是本目录 `/Users/starsfalling/Documents/FYP Mini Agent`。
老师汇报为 `reports_mlx/TEACHER_UPDATE_ZH.md`；完成结果为 0.6B、196 道 GSM8K、四组、三个训练种子。
这份结果保留，用作早期动机。R1→8B 的旧数据生成实际只完成 25 题，不是计划中的 380 题。

新研究协议见 `EXPERIMENT_PROTOCOL_REFINEMENT_ZH.md`，A–F 六组有独立名称、目录和统计比较。

## 已可复用

- `models/Qwen3-8B-Base-4bit`：已下载的 4-bit student，不重新下载。
- MLX LoRA 训练与 adapter 保存/加载；新增长序列截断拒绝和训练签名检查。
- 数学等价评分；新 MATH 配置在 dev 上也使用数学等价，而不只在名为 math500 的 split 上启用。
- 逐题配对 bootstrap；新增 E 对 A/B/C/D/F 的比较，并要求完整种子和相同评测题目。
- 旧 R1 数据只用于新流程的 GSM8K 工程检查，不冒充 MATH 主实验。

## 新增模块

| 文件 | 用途 |
|---|---|
| `refinement.py` | 冻结原解答、候选步骤偏移、解释请求、配对六组构建 |
| `diagnose_refinement_mlx.py` | assistant 前缀续写、独立复测、逐次断点保存和 token/时间统计 |
| `request_refinement_hints.py` | 串行 R1 解释生成、共享预算、失败预留、原响应保存 |
| `prepare_refinement_math.py` | 固定数据版本、剔除原始 test/ 来源、按学科和难度分层抽样 |
| `prepare_refinement_teacher.py` | MATH 原始解答请求，支持分数/表达式并使用统一费用账本 |

这些 Python 文件位于 `src/fyp_distill/`。

## 预算

不再继续旧多-rationale 生成。原 ¥42 预算转移：历史记录估算 ¥2.549316，新增额度 ¥39.45。
两个新配置共享 `data/refinement_budget/teacher_ledger.jsonl`，费用按照原配置输入 ¥4/百万 tokens、输出 ¥16/百万 tokens 估算；实际收费由平台决定。
不能把 ¥39.45 分别分给每个 run。不得通过换目录或重复请求绕过失败请求的预留费用。
当前只授权额度内执行，不保证这个额度能完成 1,000 道题的全部加工。

## 已冻结的 MATH 题目

`data/refinement_math/processed/` 中包含 1,000 train、200 dev、100 pilot（pilot 为 train 子集）。
上游 PRM800K 文件共有 12,000 题，其中 4,501 个原始 test/ 题目被排除；7,499 个原始 train/ 中再排除 2 个题面重复，得到 7,497 题候选。
它不是完整原版 7,500 题 MATH train；论文应准确报告所用固定版本和可用数量。
与本地 MATH500 做了来源 ID 和标准化题面精确去重；这不能排除所有近重复或预训练污染。
来源 URL 与校验和在 `prepare_refinement_math.py` 中固定，manifest 记录实际源文件校验和。

## 继续正式实验前的必要工作

1. 先从 100 道 MATH pilot 中的小批量题目生成并核查 R1 原始正确解答。已接入原始解答请求入口及统一账本；首批 5 题请求保存于 `data/refinement_math/original_requests.jsonl`，已发送（4 成功、1 超时），后续需要覆盖更多 subject/level，而不是只取按学科排列的前几题。这条新入口支持数学表达式，不沿用旧 GSM8K 的数字-only 提示与 180 词限制。
2. 核查原解答每一步正确、候选前缀不包含最终答案。当前换行分步是保守工程起点；没有换行的 6 条旧解答被排除，不能声称支持所有语义分步。
3. 独立验证选出的提示；通过质量审查不等于通过续写收益筛选。
4. B/C 若长度不匹配，重新生成控制解释并计入成本；不能以无补充静默替代。
5. 冻结 pilot 阈值、训练/生成长度及截断规则后扩大样本。
6. 当前训练器只实现题目曝光预算；确切的 supervised-token budget 控制尚未实现。主论文若使用“同监督 token 预算”的结论，需要额外实现并验证调度；不能改个配置标签当作已经完成。
7. AMC23/AIME24/25 仍是后续补充评测，尚未准备。

## 运行顺序

全部命令在本目录执行，使用 `PYTHONPATH=src`。普通准备使用 `.venv/bin/python`，MLX 续写和训练使用 `.venv-mlx/bin/python`。
README 列出准备、定位、解释请求、独立验证、构建六组的命令。
首次保存的请求队列仅包含一题 smoke，不要直接把它当成完整 19 题 pilot。
需要扩大时使用新 run root，费用账本保持同一个路径。

数学验证集用于调参；测试集不得用于选提示或看分数后修改阈值。无辅助测试只提供题目。

原始 MATH 解答生成命令（产生费用，使用共享剩余额度）：

```bash
PYTHONPATH=src .venv/bin/python -m fyp_distill.request_refinement_hints --config configs/refinement_math.yaml --requests data/refinement_math/original_requests.jsonl --limit 5
```

## 首题工程检查（不能作为主实验效果）

- 首题：GSM8K train 的 Samanta/Mark/Eric 分数题，独立 gold=32。
- 三个候选前缀定位采样：4/4、4/4、3/4；因此选择最后一个候选。
- 独立验证每臂 8 次：无解释 8/8，局部候选 0 为 4/8，候选 1 为 5/8。
- 两个候选均未达到收益阈值，E 拒绝补充；B/C/D 同样保留原始解答。F 保留预定随机解释。
- 六组均构建 1 条记录，仅用于接口检查。接受分支由离线 fixture 测试覆盖，不能谎称本次真实样本证明了位置配对的性能优势。
- 四个 R1 解释请求估算费用总计 ¥0.1313。新增预算估算剩余 ¥39.3187，历史+本轮合计约 ¥2.680616。
- 定位阶段 3/12 次生成达到 512-token 上限；独立验证 2/24 次达到上限。部分达到上限的输出仍含正确最终答案，因此答案正确率与截断率须分别报告。
- 普通扩写控制解释为 44 tokens，两个局部解释为 20/18 tokens；若该题的 E 接受补充，现有普通控制会因长度不匹配被阻止训练，需要重新生成。此题 E 拒绝补充，B/C/D 沿用原文。
- `local_unfiltered` 已完成单题 16 micro-iterations / 2 optimizer updates 的训练 smoke，第二次更新学习率 1e-4，耗时约 18.4 秒，峰值内存约 6.81 GB，adapter 已保存到 `outputs_refinement_pilot/local_unfiltered/seed_7/adapter`。这只能证明训练接口和参数更新正常，不能报告为模型性能结果。
- 自动测试 35 项通过；新代码语法和两份六组配置另作检查。正式 MATH 训练尚未启动。
