# 七学科小批次训练实验

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

21:25例行接续：原locals66381/teacher66384和唯一GPU80521正常保留，提示复测保存27/144次；1030首条提示实审66tokens，reviewed_hints累计22条，SHA256=5213fad05bef836fc2779d424ef14979cffa568d0802c47b9bbfcc04cb35cd9c。此前21条及verify题hint_hash保持一致，未新增收费修复或开始训练，未发生阶段变化。下一步继续剩余提示审核，teacher结束后处理免费恢复合并及1971理由不足提示的新ID必要修复；不重启现有队列、不并发GPU。

2026-10-14 21:23最新推进：无提示baseline136/136完成，84主评分正确，生成总2458.4224秒；这是训练题前缀续写，不是70独立评测或训练效果。旧GPU66416已确认退出。已后台启动已审核前9题提示复测PID80521、144新hint续写、复用72baseline，日志verify_reviewed_subset_pipeline.log，启动记录reviewed_subset_verification_launch.jsonl；当前唯一GPU，后续任何完整verify/训练须确认80521退出，复用全部136baseline和已有hint结果/hash，不重新采样。locals66381/teacher66384仍运行，不重启。reviewed_hints21条实际16–62tokens，SHA256=b7358731012d2306d45af31226f4525b8d410cef8a8ce786ec079a564bba4583；此前19条字节hash21acfe5a8e5536a0c3eb546818ef750130e185c62879d2640af00ca6658ff44e保持，包括自动审核拒绝过的时间戳，不再修改。四条格式异常已免费审核恢复：393local0、911local1，以及1229local0重复相同JSON/1229local1唯一JSON原LaTeX未转义；后两条恢复记录在hint_response_recovery_reviews_2.jsonl，前两条在hint_response_recovery_reviews.jsonl。teacher锁期间未改hints/账本，结束后按这两个恢复文件合并审核快照和追加原账本恢复溯源，原错误/全响应/费用保留，不重复收费。1971local1虽数学结论n>=1正确，但首句仅用x>0和x0未定义推n非零，缺正比值关系，按与原解答相同标准拒绝、不标true；原记录及SHA256在rejected_local_hint_reviews.jsonl，不能verify它，队列结束核查连接/保存响应/预算后准备新可追溯修复ID，旧候选保留，不用新ID认重复缺失。1971local0实审合格。当前共享新估算/预留5.365004，加历史7.914320，未知1.972988含在途0.122392与旧1.850596不释放。runbook及workspace状态同步，尚未训练，继续完成小批六组训练独立评测；本阶段推进已通知，无变化静默。

2026-10-14 21:08接续：locals runner66381与baseline GPU66416仍运行，不重复启动。已实审16/34条local（前8题完整），实际token16–62，reviewed_hints.jsonl SHA256=878aa30cfdcff428e6fa7e086699125bb93da33fe4ea2a7f5c71f7e81ac0e9e6；之前9条/7条内容及字节hash不变。393 local0完整公共响应前置说明+唯一JSON已免费提取，911 local1完整响应含3个JSON，人工全文核查三候选后在任何hint复测前按响应顺序固定第一条，未依据gain择优，余2条在hint_response_recovery_alternatives.jsonl留档，不增加local数量。两条原响应/失败/已知费用保存，hint_response_recovery_reviews.jsonl记录原响应hash、提取区间、0新增API；teacher持锁期间账本/付费hints未改。队列结束后必须把这两条已冻结审核记录合并到hints并为原账本条目追加免费恢复溯源，保留旧错误/全响应/所有费用；不要收费新ID重发可恢复项。baseline保存100/136次，后续GPU先确认66416退出；可用已审核快照对完整前8题verify --limit8，必须复用fresh baseline且保留hint_hash/内容，此后完整verify续跑不重采样。共享新估算/预留5.010884元，加历史7.560200元，未知1.960764含在途0.110168与旧1.850596，不释放。部分提示只是通用/宽泛规则，不能称诊断到真实概念缺口或有收益；尚未训练，无提升结论。runbook与workspace状态同步。继续全部小批训练评测，普通无变化保持安静。

20:56接续：原locals66381和baseline66416继续。360第二条提示实审22tokens；393第一条因前置说明+末尾唯一JSON解析失败，完整付费响应已免费提取并核查数学/位置/无泄漏31tokens，未重发。reviewed_hints现9条，SHA256=5d88bd4069e9dc7ea6088cea402f92dc7b7f0179146a3e333d651099c52a8ad1，前7条字节hash不变；免费恢复溯源在hint_response_recovery_reviews.jsonl。teacher持锁期间不编辑付费hints/账本，队列结束后合并该审核记录并记录恢复，原响应/已知费用/失败保留。baseline保存73/136次，不能并发第二GPU；共享含历史估算/预留7.217908元，未知1.963312含在途0.112716，旧1.850596继续保留。尚未训练。

20:49接续：原locals runner PID66381和独立baseline GPU PID66416均继续运行，无重复启动。已到7/34条local，7条逐条核查数学、插入位置、无泄漏并验证实际学生token25–52；保存独立reviewed_hints.jsonl，付费hints与账本保持不动。七条快照SHA256=f37c68d55fa21d2fb037fa74e20b4a5dae33d19f91af10123b97f0e3768e2a31，后续已核查内容/hash保留。代数2425/325提示属通用规则强化，screen无失败，不能称诊断到缺口或已有收益。baseline保存44/136次，后续完整verify/训练先确认其退出。共享含历史估算/预留7.101744元，未知1.954460元含当前在途0.103864及旧1.850596不释放；仍未训练、无增益结论。

最新阶段20:37：204次筛查全部完成（140主评分正确、64失败、9次上限），64失败全部全文审核并绑定hash，140判对未全部逐步人工复核。筛查实际约48分11秒。已启动34条local解释生成runner PID66381；同时启动独立baseline复测PID66416、136次，日志verify_baseline_pipeline.log。只有baseline使用GPU，后续完整verify/训练必须先检查其退出；baseline结果后续复用，不重复采样。原screen PID50702已退出，不能重启。68请求已冻结、只发送34local；局部解释需逐条数学/位置/泄漏/真实token审核。启动前保守含历史预算最多10.517616元≤42，未知预留保留。当前未训练，无方法增益结论。

20:26筛查原进程继续，20:25读取已保存158/204次、13题完成。已实质审核55条评分失败并保留hash；新发现数论素性/奇偶约束/无序计数错误、六进制进位权重错误，亦保留正确答案伴随错误理由及格式拒绝的分类。详情SCREEN_REVIEW_ZH.md，未改变correct/阈值/选择；等完整筛查和余下审核结束后发送34local，未启动训练。

20:08筛查继续原PID50702，已保存98/204次、8题完成；在生成期间逐条审核25个评分失败输出，输出hash/原因在screening_failure_review.jsonl，说明在SCREEN_REVIEW_ZH.md。已见错约分、Stirling递推漏k、距离方程常数/求根错误，也有正确推导被格式拒绝、近似答案和生成循环；不可把全部失败称理解断点。审核记录hash与原输出逐条校验通过，主correct不改。尚未准备teacher local队列或启动训练，等待完整筛查后继续。

最新状态（2026-10-14 19:49）：最后原解答队列已结束，17条逐步审核原解答与三个不泄漏完整步骤边界已冻结。固定21题中排除题面无最小值的821，以及已保存但截断/无完整正确解答的6200、72、884；不换题。学科实际分布：代数3、计数概率3、几何2、中级代数2、数论3、初等代数3、前微积分1。七学科都有覆盖，但并非每学科难度3/4/5都齐。

已启动screen后台runner PID50702，204次续写（17×3位置×4次），日志screen_pipeline.log；不要重启已运行的队列。之后审核失败类型、生成34条local、独立408次复测，再按接受集合加工B/C控制解释、构建六组、审核真实监督token后完整训练与70题独立评测。训练尚未开始，无方法训练效果结论。17有效题同曝光配置预计168微步、21优化更新，实际以训练metadata为准。

现有17条hash已核对，51个真实学生token前缀最长565，含2048生成预算2613≤4096。原解答阶段共享新增估算/预留4.191468元，加历史共6.740784元；未知预留1.850596元保留。R1最后884虽重复给正确向量(-6,8)，仍缺开头推导且末尾截断，没有三条合格前缀，所以按不完整解答排除，不说其最终答案错误。所有付费响应/旧失败/人工修正溯源保留。

2026-10-14开始，用户明确授权本次完整完成训练与实验。前两题辅助解释检查已完成、结果无稳定净收益；本批是新的训练后无辅助泛化检查，不能把旧两题作为训练结果。

规模预先冻结：21训练题，MATH七学科各3题，难度3/4/5；70独立评测题，各学科10题，难度1–4（评测未覆盖5）。配置`configs/refinement_math_trainpilot21_v3.yaml`，实际项目`/Users/starsfalling/Documents/FYP Mini Agent`。

六组：原解答A、普通扩写B、随机位置扩写C、E同解释放开头D、筛选局部解释E、未筛选局部解释F；再加未训练base参照。各组seed7、10轮固定题目曝光，同一8B Base 4bit LoRA初始化。21有效训练题时208微步、26次优化更新。此批单seed小样本只判断是否值得扩大，不估训练方差，不把阴性结果当整个方向无效。

已实际启动bootstrap runner PID19014：16条新的teacher原解答请求（5旧原解答复用，2原超时用新ID显式补发）与70题未训练模型评测并行；teacher PID19016、评测PID19017。run_status和日志在`data/refinement_math_trainpilot21_v3`。不重复启动。原¥42预算继续共享，所有旧失败预留保留，不读/输出.env。

local请求在本批新输出前固定改为针对实际缺失规则/关系；旧两题提示与结果原样保留。解释位置/内容实质审核、独立采样和原0.25阈值固定。六组文本如果完全一致，明确记录训练/评测复用，不制造伪独立结果；不同文本仍实际训练。

接续任务rationale已恢复为“完成七学科小批次训练实验”，10分钟接续，完成数据审核→训练→70题评测→报告，不会仅停在数据加工。只在真实阶段变化、失败或完成时通知；完成后暂停，不自动扩大。当前只有基线评测和原解答生成启动，尚无本批训练效果结果。

[完整冻结协议](</Users/starsfalling/Documents/FYP Mini Agent/data/refinement_math_trainpilot21_v3/PILOT_PROTOCOL_ZH.md>)。

## 数据核查进展（2026-10-14 18:46）

已实质审核11条原解答并保存draft_bases_reviewed.jsonl，已有10条内容和hash保持不变，未冻结不完整bases。70题base评测全部完成，冻结主评分7/70（10%），44条无可解析答案；辅助核查发现任务漂移、回显以及8条数学正确却因占位/标签等被拒绝，不能称63条都数学错误。辅助记录绑定输出hash；61条全文、9条首尾片段初步分类，不声称70条全步骤全文审核。详见项目reports_refinement_math_trainpilot21_v3/BASE_OUTPUT_AUDIT_ZH.md。评测不用于调参，尚未启动本批筛查、解释生成或训练，尚无本方法训练效果。

原teacher队列继续运行，已出现5条超时（包含已按源题矛盾排除的821），均无可免费恢复的响应；保持原进程，未知费用预留保留，不自动重试。全共享账本截至本次读取新累计估算/预留2.934124元，含历史共5.483440元，其中未知预留1.321628元包含当前在途请求。队列结束后再核查连接并以显式新ID处理仍必要的缺失解答，排除题不修复。

训练候选intermediate_algebra821存在源题矛盾：题面严格正数，参考解答用97个零达到等号，给出的值只是下确界而非可达到最小值；该题在本次teacher输出返回前已独立证明并记录排除、不替换。因此有效训练目标至多20题，七学科仍覆盖，中级代数缺level5，20有效行时200微步25updates。probability360金额gold混合表达已按题要求规范为0.67，原gold、请求与candidate记录不改，在student筛查前保存理由。不修改评分协议、独立评测数据或旧实验结果。

数据核查说明：[质量审核](</Users/starsfalling/Documents/FYP Mini Agent/data/refinement_math_trainpilot21_v3/DATA_QUALITY_REVIEW_ZH.md>)。配置名中的21是固定选择数，最终报告必须注明20或实际有效数，不能宣称21全部有效。

18:47又核查prealgebra392和1164，草稿累计13条。392响应含孤立`</think>`与前后重复解答；原付费响应完整保留，仅训练草稿可追溯地保留标记后完整正确解答，附raw_candidate_text_hash。图形已按Asymptote确认六边形120度、正方形90度；三边界均在方形角计算与最终和之前。1164容量题x/6+130=3x/5得300，三边界在通分求解前。先前11条草稿hash不改，尚不冻结bases。

18:51已核查新返回的precalculus359，草稿14条；同样仅规范化孤立think标记前的重复内容、原响应保持，三边界在sin4x与角度求解之前。基线审核另已补读9条长输出中段，现在70条均有全文阅读与辅助分类，主评分保持7/70。等待teacher时独立复核缺失源题的斜向十字面积、素数枚举和角度分支，记录在PENDING_SOURCE_REVIEW_ZH.md，不能当成teacher已审核。

18:54原teacher队列仍运行，precalculus72又超时，累计6条（含已排除821），最后884在途。连接已用worker新增只读--check-connection确认可达，无生成、无账本写入。先前4条必要失败已核查无保存响应并准备original_repair_requests_1.jsonl，只有新可追溯ID、尚未发送；等共享锁释放后复查再发送。72尚未加入该冻结修复队列，不能因已存在修复队列就遗漏此题，也不能重复准备原4条。说明在ORIGINAL_REPAIR_REVIEW_ZH.md，费用估算/未知预留保留，未按表现换题。

## 最新断点（2026-10-14 19:01）

原队列18:58结束：16新请求9成功7超时，加复用5条，14条原解答均已核查。除源题矛盾排除821外，6有效题缺失（概率911、几何6200、中级代数1971、数论336、前微积分72/884）。逐条确认无保存响应，锁已释放、连接检查通过，6个新ID保持原prompt/hash与旧失败追踪关系，已合并保存original_repair_queue_1.jsonl并于19:00启动唯一后台runner PID35756/teacher PID35758；不能再重复发此前4条或当前6条修复。日志repair-originals_pipeline.log，状态teacher_original_repairs。

补齐前共享历史+新估算/预留5.762968元；6条新增最保守预留0.633332元，合计6.396300元，仍共用42元总上限，旧未知预留不释放。原解答与边界审核后才冻结20有效题并筛查，本批训练尚未开始；完整训练与独立评测目标继续。已有相关34项回归通过。

19:11补齐队列PID35756/35758继续，概率911已补回并实质核查集合分拆S51=1、S52=15、S53=25、总41，三边界在S52/S53计算与总数前。草稿现15条，先前14条hash保持；几何6200本次新ID再次超时，旧预留仍保留，不自动重复请求。其余修复在队列内，未冻结部分bases、未启动筛查。当前共享历史+新估算/预留6.046996元（含在途），用户已获解释：原解答后还需局部解释筛选、六组构建才进入训练，先前准备耗时估计过于乐观。

## 最新断点（2026-10-14 19:25）

六条修复已结束，仅概率911/取整1971返回，其余几何6200、数论336、前微积分72/884超时。草稿累计16条；1971原最终80/9正确但正性推断两处理由不足，原候选保留，新人工修正候选保存在manual_corrected_candidates.jsonl，详情ORIGINAL_MATH_CORRECTIONS_ZH.md；最终报告披露人工辅助修正，不冒充全自动R1原文。已有15条hash不改。

四条缺失已核查无保存响应、共享锁释放、连接及预算通过，保存original_repair_queue_2.jsonl并于19:25启动runner PID43681/teacher PID43683，日志repair-originals_pipeline.log、状态teacher_original_repairs。新ID后缀repair:extended_timeout:1，只有传输等待从240秒延到600秒，模型/提示/采样与所有student设置不改。旧未知预留保留，共享历史+新启动前6.342960元，本次最保守额外0.424840元，总6.767800元。34项相关回归通过。

这是四条原解答最后一次显式付费补齐，不能无限重试或重复该队列。结束后免费恢复可用响应、实质审核；仍缺失则记录服务失败和实际16–20有效题/学科难度覆盖，冻结完全审核集合继续完整训练评测，不换题、不称20全成功。七学科当前仍都有题。尚未筛查/训练，需后续局部解释及六组构建。

19:31最后补齐仍在原PID43681/43683运行。geometry6200返回length截断响应，已全文核查：前半误套水平十字得4%，后半自我修正到斜向图形坐标但在C4中途截断，没有完整正确解答。原付费响应及已知用量费用0.101008元保留，审核hash在rejected_original_response_reviews.jsonl；不能当格式恢复、不能拼补冒充R1解答，不再收费补该题。当前有效目标上限19（已有16合格），几何只3/4难度，七学科仍覆盖。其余3题队列未结束，不重复启动、不提前冻结部分bases；报告模块已加入实际学科难度、源题/响应/处理排除和人工修正披露。
