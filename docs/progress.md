# 项目进度与交接（Progress / Handoff）

> 本页保留各段日期对应的工程与训练快照。训练目录、原始测量和旧 Git 提交未随公开源码分发；“当前”指该段记录时点。发布进度见 [发布流程](public-release-preparation.md)，历史数字不是此副本的模型或棋力保证。

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

> **当前入口**：[当前状态与下一步](status.md)集中列出当前事实、工具、产物和待办。本页保留按时间积累的修复与实验记录；历史“当前状态”“训练中”和下一步须按日期解读，不能用旧 A/B 冻结当前参数。

> **2026-10-08 短CPU/网络归因**：冻结iter28、原预算/批次/FP32，广范围36图+144根+32整局对精确，但插桩开销固定根12.52%/整局19.60%；保留原始结果并预声明收紧，只包原评估AST调用，重新36图/144根精确、开销1.35%。普通/强evaluation墙钟54.35%/49.42%，同步策略下载30.43%/27.47%，不是纯传输或可直接消除的等待；CPU算子trace另测且精确。Nsight/WPR需要管理员权限，原生CPU栈/核时间线未取得，驱动忙等只是待证假设。新增9项定向通过，全量 **530 passed / 141.33s**，两套各18项保护哈希测后一致、文档本地链接通过。暂不扩大规则编译，新候选均未实现；无训练/生产改动/提交，见 [结果、限制与工作顺序](reviews/2026-10-08-selfplay-attribution.md)。

> **2026-10-08 全部保留加速联合轮**：冻结iter28并恢复selfplay RNG，rules/legal=Cython、nodes=Python、1024局/128强局/800+3200模拟/8 worker，正式调度与完整样本IPC；自博弈 **5696.47s / 94.94min、9.455步/s**，53,862原始/215,448增强样本，增强11.20s、保存5.35s。GPU1107采样均值84.54%，worker任务占用97.81%、进程CPU7.46核等效（含驱动CPU），排空尾部128.95s/2.26%。另测18根72对搜索精确，网络边界46%--51%、合法着/扩展各约10%--11%，嵌套/等待/5.48%插桩开销不能当纯CPU占比。普通/强包装、顺序/seed流、保存往返和18个保护哈希通过；入口/原生快路定向 **17 passed / 5.98s**，全量 **521 passed / 142.22s**。不更新生产权重/优化器/数据、不提交Git；没有旧/新配对联合加速率，未测4M常驻或梯度更新轮。**零散微优化降优先级**，未来先拆CPU/驱动/CUDA边界再决定结构性试验，见 [协议、结果与限制](reviews/2026-10-08-joint-selfplay-round.md)。

> **2026-10-08 中间快照削减**：NeuralMCTS从合成行动者账本直接生成原信息集键，只留根/非终局合法叶的完整观测。修正旧函数对照的模块访问后从头重测：36根条件快照201,539→64,486，完整图/网络流精确；固定根288对无明确收益；192局平均配对省1.25%、合计省1.34%，描述性区间[0.77%,1.70%]，96对样本/RNG精确，六组均更快，通过预声明门槛并保留。原只读图插桩漏记直接键已适配，全量 **512 passed / 142.77s**，全Python回退 **70 passed、1 skip / 19.10s**；27个保护哈希不变。无新参数、原生重建、精度/预算/版本变化、生产训练或Git提交；仅约1%小收益，非1024局稳态保证，见 [完整协议、测量修正与回归](reviews/2026-10-08-information-snapshots.md)。

> **2026-10-08 完整机会表复用**：冻结iter28、rules/legal=Cython、nodes=Python、8 worker、800/3200模拟。819,678次根请求仅1,017种计数对，修正微测以保留754,035个原快照身份。纯Python防护192局合计慢0.54%，不保留；只编译精确整数防护后另声明新种子/同门槛，192局平均配对省2.55%、合计省2.75%，描述性区间[1.77%,3.66%]，96对样本/RNG及36根的完整图/网络流全部精确一致。保留有限不可变LRU，实际整局命中99.8458%；不缓存网络/世界/统计，不改概率运算/精度/预算。全量 **502 passed / 148.91s**；本机规则扩展重建，原训练五个主要产物哈希一致、未续训或提交Git。并非1024局稳态保证，见 [协议、失败版本、实测与回退](reviews/2026-10-08-chance-support-reuse.md)。

> **2026-10-08 当前CPU热点复测**：最新iter28、攻击/合法着Cython、nodes=Python、lazy开，18根覆盖开中残局及私有捕获。既有只读工具的800/3200模拟边界计时：动作边扩展约11%、合法着9%--10%、编码6%--7%、选子3%--5%，不是纯CPU或整轮占比。首套串行/并行精确对照及第二套288次参考搜索通过，14/16个保护哈希不变。机会对象已延迟但完整分布仍重复构造；优先研究其精确准备复用，再看信息态快照与整数规则热点，新优化未实现、收益未证明。FP16降优先级，不改生产代码、配置、模型或提交Git，见 [报告](reviews/2026-10-08-current-cpu-hotspots.md)。

> **2026-10-08 FP16独立复验**：8批×32局/模式，共512局、61.17分钟，预先声明新种子/平衡顺序/5%省时及区间下界门槛，记录384个GPU样本。网络微测省20.98%，完整局合计慢1.87%，平均配对省时-5.21%（描述性区间-26.77%至17.02%）。位置/叶吞吐点估计+3.84%/+8.25%但区间含零，FP16强局更长、任务计时占用更低；筛选未过，不启动棋力/训练。新增独立受门槛控制的换色对弈原型，仅单测验证；联合回归正常本机目录 **50 passed / 5.28s**，沙箱首跑两项旧硬链接权限失败未改断言。22个保护文件哈希不变、未改生产精度或提交Git，见 [复验报告](reviews/2026-10-08-fp16-followup.md)。

> **2026-10-08 推理精度首测**：保留FP32训练配置，新增独立TF32/FP16/BF16测量工具及8项定向测试；冻结第22轮模型、336搜索/模式、64局/模式，共192局长测29.02分钟。FP16固定根平均配对省时19.61%，完整局合计省14.60%，但四批区间含零且局长/strong构成改变；TF32固定根约1.74%，64局合计慢2.33%。主线卷积原已允许TF32，不是额外开启可保证完整FP32精度。输出/搜索/样本非逐位相同，棋力及数据效率待独立验收，不接续训。22个保护文件哈希不变，未训练或提交Git，见 [报告](reviews/2026-10-08-inference-precision.md)。

> **2026-10-08 加速阶段总结**：只读核对最新28轮/28,672局、1000更新步/4M buffer，最新run仍legal=Python。整理 [阶段交接](reviews/2026-10-08-acceleration-stage-summary.md)，涵盖环境、配对基准/指标、实测与暂缓原因、保留改动、数值/信息/版本约束和后续实验。最新五个非评估轮自博弈占日志完整轮耗时93.40%，不是单变量加速归因。保留基准F1可选二进制依赖未知/未修复边界；不改运行代码、不训练、不测速、不提交Git。

> **2026-10-07 独立图搜索研究**：整理 [压缩交接与Gate A/B/C](experiments/0003-graph-search.md)，冻结当前v5第22轮模型。14局面、800/3200预算、两搜索种子，56组原始/插桩搜索的完整图、策略、RNG及网络流精确相同；95,339个展开节点未发现不同公开路径的完整输入/保守新增换位。真实轨迹棋盘碰撞3.75%/9.80%不能直接合并。新增12项反例/一致性测试通过；本机三Cython后端全量 **472 passed / 129.71s**，沙箱临时文件权限失败经正常本机目录重验通过。Gate A本轮未通过，不实现图引擎、不做棋力或训练对照，不修改生产搜索与训练产物。报告与原始证据见 [首测](reviews/2026-10-07-graph-search-feasibility.md)。相关默认值整理、用户分析文件与审查报告保留。

> **2026-10-06 加速整理**：on/k=0 的机会对象按需分配设为默认，新增 `--no-lazy-chance-allocation` 回退，旧 off/k>0 入口保持兼容。当前 v5 目录 `data/train_rules5_20261004` 已完成16轮/16,384局，历史 v3 的13轮目录保留。加速开关/实测证据统一到 [performance](performance.md)，工具分为日常、验收、测量、原型和历史入口，见 [tools/README](../tools/README.md)。不移动或删除训练产物、历史结果、测试与用户分析脚本；集中推理/缓存仍不接生产。

> **2026-10-04 算法审查后续验收**：当前规则v5/搜索v3/评估协议v2。修正长捉并列与同码身份误认、双闪将锚点、根噪声、有界策略裁剪、验证工具、合并重复增强/丢边界/版本重标，以及评估洗牌/RNG/未完成配对/对手分组。全量 **300 passed（52.66s）**；200局随机加2条固定路径、28,651状态时刻严格翻面零偏差；小池未知损失及双揭示回传精确枚举通过。根条件Q/计数信念近似保留，无forced playouts，不声称官方复杂裁判或均衡正确性。未训练、迁移、改既有产物或提交Git；旧v3训练目录不能直接完整续训。见 [审查清单](reviews/2026-10-03-augmentation-audit.md)、[ADR-0020](adr/0020-algorithm-audit-followup.md)。

> **2026-10-03 iter=12诊断检查**：新目录已完整提交13轮/13,312局、缓冲2,872,040增强样本；13个数据哈希、8轮固定诊断、模型/父检查点链及六个核心源文件核验通过。留出KL/value误差从诊断开始下降约1.4%/9.6%，参照value下降56.7%，泛化差距扩大；索引9以后留出value主要波动。当前偏差转为+0.03744，两端仍过于确信；32局整局bootstrap对零基线的改善区间包含0，不判断棋力。原产物哈希未变，另存当前纯权重基准；未训练、未改运行代码。建议原参数再新增7轮至索引19，结合首次对弈评估复查。见 [iter12审查](reviews/2026-10-03-iter12-diagnostics.md)。

> **2026-10-02 固定诊断**：已接入一次性整局留出，在增强前排除，另存固定训练参照；每轮训练前/后记录策略熵/KL、value误差/校准和期望复用。完整检查点恢复固定集与最近报告，报告按attempt区分提交状态；旧五轮不追认为留出。新增 `--run-iterations` 限制本次新增轮数，保留2000轮cosine计划。新增15项回归，全量 **235 passed（53.28s）**；真实238通道网络/CUDA隔离短测验证保存恢复和诊断不改变参数/BN/RNG，用户产物哈希未变。正式训练仍停在5轮；下一轮用32局留出先新增1轮，再累计新增3至5轮检查。见 [续训说明](reviews/2026-10-02-fixed-diagnostics.md)、[ADR-0018](adr/0018-fixed-training-diagnostics.md)。

> **2026-10-02 五轮训练与Ctrl+C**：用户在 `data/train_fixed_20261001` 完成5轮/5,120局，缓冲1,099,924增强样本，数据哈希匹配；平均8,279s/轮，搜索占95.8%。训练loss略升，尚无固定留出或对弈评估，不判定棋力/过拟合。Ctrl+C现场仍有主进程及七个worker，已核实后结束且产物未变。隔离复现发现worker导入torch时退出、父进程阻塞模型传输；补提前信号屏蔽和Windows本次训练Job清理，约0.3s退出无worker残留。新增12项回归，全量 **220 passed（50.71s）**。见 [五轮检查](reviews/2026-10-02-five-iterations-and-interrupt.md)、[ADR-0017](adr/0017-windows-training-interrupt.md)。未启动续训，下一项仍为持久化训练诊断。

> **2026-10-01 运行来源**：每次训练/独立评估创建 run ID，记录完整配置/架构/语义版本、源码哈希与 dirty 状态、父模型/检查点哈希。每轮独立 attempt ID 连接自博弈生成模型、训练后评估模型及提交标记；完整检查点提交后才确认，标记写入中断可在续训时补回。旧历史明确来源未记录。新增17项回归，全量 **208 passed（37.13s）**；真实旧模型双进程隔离通路检查通过，两局均未终局，无棋力结论。主线 latest/best/状态/最近数据哈希未变，未训练或迁移。见 [ADR-0016](adr/0016-run-provenance.md)；下一项为持久化训练诊断。

> **2026-10-01 完整续训**：已实现原子 `training_checkpoint.pt`，保存 AdamW/调度/计数/RNG/配置和版本/缓冲文件清单。补每局边界、有序 worker 返回、未提交数据隔离和兼容导出修复；18 项新回归及真实主线网络 CUDA 微测通过，权重最大差0。全量 **191 passed（31.49s）**。主线未运行训练/迁移，旧产物保留。使用与限制见 [ADR-0015](adr/0015-complete-training-checkpoint.md)。下一项为独立运行来源记录。

> **2026-10-01 轻量整理**：新增当前状态/待办/工具与产物入口；README 路线图及问题单状态同步当前实现，旧实验标为历史。仅整理文档，未改核心代码、模型、训练数据或状态；`pytest --collect-only` 确认仍收集 173 项，完整回归沿用同日最近通过记录。

> **2026-10-01 训练预算检查**：用户补充实际 1024 局/800+3200 搜索/1000 步配置，并决定把长训留待更合适的硬件。最近六轮日志平均 7,415s，搜索占约 95%、训练约 3.3%。旧/新数据短探针 batch1024 均无 OOM（新数据单步302ms、CUDA 分配/保留约 1.71/1.97GiB）；新搜索数据的小样本能拟合，留出策略却变差，不能据此判定棋力或容量上限。仅隔离诊断目录写入，主线模型、数据、状态哈希一致。全量 **173 passed（30.51s）**。完整 4M/8-worker 峰值及完整预算续训仍未测。报告见 `docs/reviews/2026-10-01-training-budget.md`。

> **2026-10-01 第四批更新**：机会节点固定支持偏差已修复（§S1.29、ADR-0014），全量 **173 passed**。k 只限制初始预建数量，完整世界抽样后按需补建，默认仍 k=0。编码/数据版本不变；未训练、未覆盖历史产物。下一步优先完整训练吞吐/峰值内存检查，再决定算法改造。

> **2026-10-01 第三批更新**：公开规则历史编码已补全（§S1.28、ADR-0013），当前 **238 通道 / DATA_VERSION=3 / 全量 167 passed**。旧模型补零加载、v2 数据显式标记缺失历史；未重置训练或覆盖既有产物。下一项为 chance_k 截断偏差，默认仍 k=0。以下快照中的 71 通道说明已由本条更新。

> **2026-10-01 第二批更新**：私有捕获的信息边界与共享决策已修复（§S1.27、ADR-0012），并补明吃暗机会路径。全量 **159 passed**。叶评估现在使用根一致粒子的合成行动者观测，真实对手私有槽仍不可读；不要恢复历史“所有叶都用根观测”约束。计数信念、根条件样本与非均衡局限仍保留。

> **2026-10-01 更新**：第一批搜索修复完成（§S1.26）。NeuralMCTS off 与 ISMCTS 按公开翻子结果隔离子树；训练/GUI/评估默认 `chance_nodes=True, chance_k=0`。全量 **151 passed**。此前 A/B 记录不作为本次算法判断依据；私有捕获信息集约束仍待后续改造。`train_state.json` 当前记录 iteration=269、total_games=51,200；本次未启动训练、未改模型或训练数据。
> 以下 2026-09-20 训练状态与评估是历史快照，当前搜索状态以本条及 §S1.26 为准。

> 更新：2026-09-20 · `RULESET_VERSION = 3` · 测试：**143 passed** · Gate v3 停在 iter560（验收 0.5725 vs RIS-800）；**v4 chance-aware 主线**训至 iter220（32,000 局），训练内 100 局@800 评估 **0.725**（iter219），正式 200 局验收待再训 40 轮后进行（§S1.25 / §5）
> 历史记录说明：本文件保留原始进度证据；先读 `docs/status.md`，再按需查本页、`docs/adr/` 与调研报告。
> 2026-09-16（晚）：调研 P1 三项落地——S1.14 批内叶评估去重（默认开）、S1.15 PCR（默认关）、S1.16 编码 64→71 通道 R2 修复（`DATA_VERSION=2`，旧 v1 数据须归档重生成；旧模型经 pad-load 可继续用）。
> 2026-09-14 修复：信息泄露审查 R3 修复——`NeuralMCTS.search` 与 `ris_ismcts` 均改为根视角单次确定化（+叶评估用根公共视图），见 `docs/issues/0003`；下表历史评估数值均为修复前基线，正式评估前建议按 ADR-0005 重跑刷新。
> 2026-09-15：v4 调参 A/B 完成（见 §3b 筛选记录）——LR 胜出 **1e-5**；chance-k4、lr-high、sims800 均淘汰（sims800 无收益且成本翻倍）；`buffer-250k` 判无效未跑。**注意（2026-09-16）**：该批早于 P0-1（叶 observer 槽）修复，结论须重跑复核后才能用于冻结配置（见 §3b 首行注记）；下一步不变，但“冻结配置”前先复核。
> 2026-09-16：对抗审查（`docs/reviews/2026-09-16-adversarial-review.md`）10 条缺陷修复并探针验证，新增 17 个回归测试（82→99 passed）。
> 2026-09-18（晚）：续训加载修复（Issue 0006）——`load_selfplay_data` 默认按 `replay_buffer_size` 从最新文件回填，消除 `--max-load-files 32` 导致的 buffer 重建不完整；全量 136 passed，真实数据验收 512,920。见 §S1.21。
> 2026-09-19：新增**楚河汉界翻面数据增强**（行翻面+红黑互换+暗池互换；`augment_flip` 默认开、`--no-augment-flip` 关）。与左右镜像同开每样本 ×4（完整棋盘+颜色对称群）。数学证明 + 3 独立 agent + 裁判复核 + lockstep 实证（500 局 70,805 ply 合法性/胜负零偏差；已知 chase tie-break 记账近似不影响禁着）见 `docs/adr/0011`；全量 141 passed。见 §S1.22。
> 2026-09-20：v4 主线续训中（起点迁移/配置/进度见 §S1.25）；**Ctrl+C 中断硬化**——终止子进程后 ~1s 硬退出（退出码 130），不再卡在 executor 关闭流程，全量 143 passed。
> 2026-09-20：用户决策——**继续训练 40 轮**（至 ~iter260）后再跑正式 200 局验收；验收 **≥75% 视为停训目标**，停训后计划见 §5。

## 1. 历史状态速览（2026-09-20）

| 项 | 状态 |
|---|---|
| Stage 1（修复/评估/吞吐/工程卫生） | **已完成并完成验证前审查修复**（S1.1–S1.8） |
| Gate（v3 自博弈，iter0→560 / 35,840 局） | **已停训**（2026-09-14 用户决定）；320→560 对 RIS-200 评估窗口 0.633→0.750 后走平；`latest.pt`=iter560、`best.pt`=iter479 |
| Stage 1 验收（200 局@800 vs RIS-800） | 第一次（iter303 best.pt）：0.2925（CI 0.234–0.359）**未通过**；第二次（iter560 latest.pt，`acceptance_iter560.json`）：**0.5725**（CI 0.503–0.639，113W/3D/84L），对随机 0.9975——仍 <0.65，但接近一倍提升 |
| v3 最终模型 | `data/models/v3_iter560.pt`（存档，后续 h2h / A-B 基线；别被新训练覆盖） |
| Stage 2A | **代码已就绪**：暗子揭示机会节点（`--chance-nodes --chance-k 0`，9 个新测试）；`tools/ab_runner.py` A/B 串行实验 runner |
| v4 主线（chance-aware） | **训练中（可断点续训）**：iter220 / 32,000 局；训练内评估 100 局@800 vs RIS-800：iter139 0.5375 → 199 0.65 → **219 0.725**；配置与进度见 §S1.25 |
| 调研 P1 优化 | **已实现，待 A/B/复审**：叶评估去重（S1.14，默认开）、PCR（S1.15，默认关）、R2 编码 71 通道（S1.16，`DATA_VERSION=2`，旧数据须重生成） |
| 训练数据 | `data/selfplay/` v4 迭代文件（起点 128 个来自 chance-full-128，现至 0220）；`data/eval/train_eval.jsonl`；正式验收文件 `data/eval/acceptance_iter*.json`（待生成） |
| 旧数据 | v0 → `data/legacy_20260911/`；v2 → `data/legacy_ruleset_v2_20260911/`（均含日志切片） |
| 当前进程 | 无（用户手动断点续训；Ctrl+C 已可干净退出，见 §S1.25） |

## 2. Stage 1 完成清单

### S1.1 规则对齐（`jieqi.py`）
- `RulesConfig`（frozen dataclass）+ 预设 `RULESET_TIANTIAN`（默认）与 `RULESET_LEGACY`；`RULESET_VERSION = 2`，`ruleset_version` 写入训练状态与数据 meta。
- 判定顺序修复：将死/困毙优先于长将（原 bug：连将覆盖将死）。
- 长将/长捉 = **禁着**：连续 6 回合（12 ply）后继续同一行为被禁止；无任何替代着法判负。实现：`_move_scan()`、`_update_chase()`、`_banned_for_side()`、`chase_streak/chase_target`。
- 重复：**公共局面**（`(side, tuple(board))`，不含暗池）第 3 次判和（`RULESET_LEGACY` 保持第 4 次判负）。
- 无吃子和棋：`RULESET_TIANTIAN` 80 ply（`RULESET_LEGACY` 50 ply）。
- FEN/记谱：`to_fen/from_fen`、`square_to_name/name_to_square`、`format_move_uci`（`X/x` 暗子、走子后缀揭示类型、`[被吃子]`）。
- 被吃暗子信息不对称保持：吃方知类型、原主不知。
- 依据：`docs/adr/0002`、`0003`、`0004`。

### S1.1b 回归测试（`test_jieqi.py`）
- 新增：判定顺序（将死优先）、长将禁着且存在替代着法、长将无替代判负、长捉禁着与脱追、重复和棋、FEN 初始/中局往返、坐标与记谱。
- 旧行为用 `RULESET_LEGACY` 复现（长将判负、无吃子和、重复判负等）。

### S1.2 GUI 泄漏修复（`gui.py`）
- 暗盒面板改用 `belief.pool_counts / pool_total / unknown_removals` 渲染（未知被吃只显示总数）；终局展示真实池。
- 被吃暗子按**吃方**视角显示 `[真实类型]`，其余 `[暗]`（`_capturer_sees_type`）。
- 文案修正：长将/长捉="禁着判负"，重复="重复局面和棋"。

### S1.3 评估协议（`tools/eval.py` + `test_eval.py`）
- 同 seed 双色配对；对手接口 `random` / `ris:N` / `model:path[:sims]`；Wilson 95% CI；JSONL（`run/game/summary`）。
- CLI：`--quick`（50 局@400 vs `ris:400`）、正式 200 局@800；`--workers`（默认 8，spawn 进程池）。
- 依据：`docs/adr/0007`。

### S1.4 训练卫生（`azero.py`）
- policy target pruning：`pruned_policy_from_root()`，自博弈开启（`prune_policy=True`）。
- `train_epochs=1.0` + `compute_train_steps()` 替代固定 2000 步（**2026-09-13 修正：该折算导致每轮仅约 15 步、Gate 训练不足，已由 ADR-0010 改为固定 1000 步/轮**）。
- `TrainingState.ruleset_version`；`load_selfplay_data(..., ruleset_version=...)` 跳过不匹配数据；resume 时版本不符则重置。
- 评估重写为配对（同 seed 双色），训练内评估追加 `data/eval/train_eval.jsonl`。
- 依据：`docs/adr/0005`、`0008`。

### S1.5 吞吐优化（目标 ≥10×，已达成）
- 规则：`_iter_legal_moves_fast()`（轻量几何合法性，无抽子/追踪/计数），有禁着时退回原 apply/undo 路径；`_move_scan()`/`has_legal_moves()` 惰性化。**对拍 40 局随机对局、5830 状态 0 失配**。
- 网络推理：`encode_state_numpy()` + `NeuralMCTS._evaluate_batch()`（一次前向评估一批叶）。
- MCTS：叶并行 + 虚拟损失（`leaf_batch_size=64`、`virtual_loss=3.0`；设 `leaf_batch_size=1` 可退回旧行为）。
- 基准（未训练网络，RTX 4060 Laptop）：单进程 batch=1 约 326 evals/s；单进程 batch=64 约 3129；8 workers × 64 约 **7500–8200 evals/s**（约 23× 单进程、3–4× 旧 8-worker 日志推算）。工具：`tools/bench.py`。

### S1.6 工程卫生
- 新增 `requirements.txt`、`.gitignore`、`README.md`、`CONTEXT.md`（术语表）、`docs/adr/0001–0008`；删除 Windows 保留名文件 `nul`；`git init` 并完成初始提交（`352d15d`，35 文件；训练数据/模型不入库）。

### S1.7 C++ 端口验证（报告：`docs/cpp_port_verification.md`）
- 结论：归档 `jieqi-cpp-prototype.7z（历史归档，未随源码分发）` 的 C++ 端口**不可用**（预计算表初始化缺失、暗池索引不一致、≥10 迭代挂起；README 性能声明不复现）。
- 若要 C++ 后端须作为正式移植工程（数天）+ conformance 向量；现阶段维持 Python。依据：`docs/adr/0006`。

### S1 后补充修复
- **pickle 兼容 bug**：`python azero.py` 保存的数据引用 `__main__.CompactSample`，导致 `import azero` 的工具（merge_selfplay 等）读不了。已加 `_CompatUnpickler`（`azero.py` 中 `load_compact_payload` 附近）+ 回归测试；现有 6 个迭代文件已验证可读。

### S1.8 验证前审查修复（2026-09-11，见 ADR-0009）
- 规则（升 v3，旧数据作废）：F2 马腿判定方向错误（`is_square_attacked` 改用 `KNIGHT_ATTACKS` 反向索引）；F3 长将/长捉按“同一枚棋子”判定（新增 `check_anchor` / `chase_anchor`）。
- 搜索/评估：F1 RIS-ISMCTS 回传视角错误（改用 `node.player_to_move`，消除约半数模拟值反转）；F4 确定化携带长将/长捉/重复历史与 `rules`；F5 merge 产物 meta 补 `ruleset_version`；F6 Dirichlet 噪声改用实例 RNG；F7 `ply_limit` 未终局不计分；F8 `model:C:\...` 盘符解析；F9 FEN 恢复规则状态；F10 GUI 搜索移入后台线程。
- 回归测试已补充（马腿假阳性/假阴性、按子长将、按子长捉、FEN 规则状态、RIS 奇数 rollout 回传、Dirichlet 实例 RNG、merge 版本 meta、tally ply_limit），**尚未运行**（按用户要求，验证由用户手动执行）。

### S1.9 Gate 失败复盘与训练步数修复（2026-09-13，见 ADR-0010）
- **Gate 结果（未通过）**：320 轮 × 64 局（共 20,480 局）；对 RIS-800 200 局 58W/1D/141L，score 0.2925；对随机 0.9525；训练内快评 320 轮无上升趋势；和棋率 iter0 的 8/64 涨至 iter100-300 的 30-39/64。
- **根因**：S1.4 `compute_train_steps()` 按“新增样本 1 epoch”折算，1M buffer 下每轮仅 13-25 步（平均 16），单样本生命周期平均只被训练约 1 次；value 头退化（corr≈0.16、pred std≈0.20）、policy loss 仅 6.9→3.0（目标熵 ~2.3）。
- **修复**：改为固定步数制，`train_steps=1000`（约 0.5 epoch/轮，AlphaZero 同量级）；删除 `train_epochs`/`--train-epochs`；自博弈汇总日志增加“新增样本数”。
- **旧管线日志仅作旁证**：其含已修复的规则/搜索缺陷且从未验证强度；修复依据为样本复用算术 + 研究笔记的 epoch 建议 + value 头退化事实。

### S1.10 Pilot 验证与 Gate 续训决策（2026-09-13）
- 修复后从 iter 320 续训 32 轮（固定 1000 步/轮；resume 快进 LR 至 2e-5）：对 RIS-800（50 局@800，同种子）**0.20 → 0.33**（配对 13 好/6 差/31 平）；value loss 0.424→0.241；和棋率 43%→34%。
- 决策：训练信号恢复，续跑完整 Gate 至 672 轮（`--lr 5e-5` 软续训，见第 3 节）。注意 `best.pt` 由 16 局快评选出、噪声大不可信，最终验收用 `latest.pt`。

### S1.11 Gate 续训中期评估与停训（2026-09-13/14）
- 320→560 分四阶段（评估=16 局@400 vs RIS-200，窗口合计）：P1 320-383 **0.633** → P2 384-447 **0.695** → P3 448-511 **0.750** → P4 512-560 **0.750**；收益集中在 P1/P2，P3 起走平（窗口差在噪声内，单点 16 局 σ≈12%）。
- value loss 呈 U 形：0.330 → 谷底 0.151（iter 400-431）→ 0.192（iter 544-559，+27%）；同期和棋率 35.4%→13.5%。归一化 value/(1−和棋率) 从谷底 0.186 升到 0.222（+19%）：约 1/4 由“目标变难”解释，其余与高重放比（每样本一生被训 ≈62 次、每 batch 仅 ~1.6% 新数据）与目标漂移的“适应滞后”一致；policy loss 仍持续降（锐化）。
- 决策：iter560 停训，重设参数再训；先做正式验收拿诚实基线。
- **验收（2026-09-14，`data/eval/acceptance_iter560.json`，latest.pt，200 局@800 同种子）**：vs RIS-800 **0.5725**（113W/3D/84L，CI 0.503–0.639）——较 iter303 best.pt 的 0.2925 几乎翻倍，同开局配对 82 好/25 差/93 平（净 +112 对局分）；vs random 0.9975（0 负）。仍未达 65% 验收线，但趋势明确。
- 归档：`data/models/v3_iter560.pt` = v3 最终模型（勿覆盖；后续 h2h / A-B 基线）。

### S1.12 Stage 2A 暗子机会节点（2026-09-14 代码就绪，未跑训练）
- 语义：走暗子/吃暗子时把揭示结果作为**显式随机分支**；每个机会节点按信念计数采样至多 K=4 个揭示结果（候选 ≤K 时全量枚举、概率精确；否则按权重无放回采样并归一化），同一着法在不同揭示下分别回传价值（Rao-Blackwell 方差缩减，见 `docs/adr/0005` #3）。
- 实现：`azero.py` 的 `sample_chance_outcomes` / `chance_spec_for_move` / `PUCTNode.is_chance` / `NeuralMCTS._chance_specs` / `_apply_chance_move`；强制揭示在当前模拟世界不可用时回退自然抽样；回传时机会边不翻转视角。
- 开关：`--chance-nodes --chance-k 4`（默认关闭，v3 行为不变）；另新增 `--replay-buffer-size` CLI。
- 验证：新增 9 个单元测试（枚举/加权采样/被吃双揭示/展开/回传视角/端到端/回退/CLI），全量 **74 passed**；整局自博弈冒烟通过。
- A/B：`tools/ab_runner.py` + `tools/ab_experiments.json`（见 §3b）。

### S1.13 机会节点实现修复（2026-09-16，调研驱动）
- **背景**：9-15 的 `chance-k4` A/B 无效——该批早于 P0-1 修复，且机会节点实现存在叠加缺陷：①结果选择误用通用 `select_child`（`-q+u` 迫使行动方"挑有利揭示"，探针实测有利结果访问占比 99.3%、节点价值乐观偏置 +0.19~+0.79）；②`chance_k=4` 截断丢概率质量（实测 82.8% 暗子着法候选 >4、吃暗子 100% >4）；③吃暗子双揭示按 36 组合平铺（未分层）。
- **修复**：新增 `PUCTNode.sample_outcome()`——机会节点按揭示概率采样，不再做 bandit 选择；`sample_reveal_outcomes()` 以 `k<=0` 表示全枚举（`chance_k` 默认改 0，`k>0` 保留截断近似）；双揭示拆两层机会节点（移动子 ≤6 → 被吃子 ≤6）；搜索循环按机会链逐层采样并回传。
- **验证**：新增/更新 7 个单测（概率采样回归、两层展开等），全量 **101 passed**；端到端统计：捕获机会节点 178 次访问 `mean|share-prior|=0.022`；30 步搜索出现 L1 机会节点 30,992、L2 双揭示节点 1,976。
- **待办**：新增 `chance-full`（`--chance-k 0`）A/B 臂，与 `control` / `lr-low-1e-5` 同批重跑（见 §3b 复核提示）。

### S1.14 批内叶评估去重（P1-4，2026-09-16）
- **依据**：调研报告 §3.1 实测同输入重复占叶评估 6.27%（400 sims）——低 sims 下更高（64 sims 时 54%），PCR fast 臂收益更大；§4 第 4 行 P1。
- **实现**：新增 `leaf_input_signature()`（覆盖编码全部信息：棋盘/轮次/双方池/未知被吃/无吃子/长将/长捉/重复计数/history）；`NeuralMCTS.search` 在 `_evaluate_batch` 前按签名去重，重复项复用同一 `(policy, value)`；同一叶节点在同批内只 `expand` 一次（避免重造 children 并多消耗机会节点采样 RNG），`_backup` 仍对每条 path 各执行一次；`leaf_evals_total/unique` 计数汇总进自博弈日志（`叶评估去重: 唯一=/总计=/节省=x%`）。
- **开关**：`AZConfig.leaf_dedup=True`（默认开）+ CLI `--no-leaf-dedup`。`chance_k<=0` 全枚举时去重不改变 RNG 流（单元测试证明搜索结果逐位一致）；`chance_k>0` 截断采样时机会节点的 RNG 流会因去重而改变（同 seed 仍可复现，但与去重前不同流）。
- **验证**：新增 3 个单测（签名覆盖规则字段/history、去重计数与结果一致、CLI 开关），全量 **104 passed**。固定局面（v3_iter560、seed 20260916、CUDA、5 次均值）：
  - 400 sims：重复率 **7.0%**（372/400），耗时 133.4→126.4 ms；
  - 100 sims：重复率 **27.0%**（73/100），耗时 28.1→25.6 ms；
  - 两种 sims 下首选着法一致（批量归约浮点噪声使访问分布偶差 1–4 次访问，属混沌漂移不是语义变化）。
- **A/B**：新增 `leaf-dedup-off` 臂（`--no-leaf-dedup`，control 默认已开去重）：
  `python tools/ab_runner.py --config tools/ab_experiments.json --only control,leaf-dedup-off`（判据：单位时间产局数/评估量，去重开应不劣于关）。

### S1.15 PCR（playout cap randomization，P1-5，2026-09-16）
- **依据**：KataGo 2019 论文 §3.1（消融约 1.37×）；本项目 sims800 A/B 已统计证"加深无收益"（p=0.502），样本量是主约束；调研报告 §5 ②。
- **实现**：普通自博弈（非 strong 局）每步按 `pcr_full_prob` 抽档——full=`mcts_simulations`（策略目标权重 1.0）/ fast=`pcr_fast_sims`（权重 0.0）；`train_step` 改为加权策略损失（策略目标只由 full/strong 回合样本产生，KataGo 语义），value 目标仍使用全部样本；Dirichlet 与温度退火两档一致（不因 PCR 降低探索）；既有"每 8 局 1 局 strong(1600)"机制**保持不变**（整局强搜索，作为低频高质量样本与既有对照；PCR 只作用于普通局，避免替换后失去该对照）。
- **开关**：`AZConfig.pcr_enabled=False`（默认关，训练现状不变）+ CLI `--pcr / --pcr-fast-sims / --pcr-full-prob`；`--pcr` 时校验 `fast < mcts_simulations` 且 `full_prob ∈ (0,1]`。样本新增 `policy_weight` 字段（0=fast 档），旧格式样本按 1.0 处理（`getattr` 兼容 + 单测）。
- **验证**：新增 8 个单测（档位分布/自博弈样本权重/加权训练只吃 full 策略/旧样本无字段/镜像/序列化往返/CLI 校验），全量 **112 passed**。吞吐冒烟（v3_iter560、固定种子各 2 局、单进程 CUDA）：
  - control（400 sims）：6.75 步/s、388 评估/步、去重 5.4%；
  - pcr25（fast 100 / full 400 / p 0.25）：**15.62 步/s（2.3×）**、166 评估/步（预算 0.25×400+0.75×100=175）、full 回合策略样本 23.3%、去重 16.4%；
  - 按"局/秒"仅 1.21×：该次 PCR 两局长 86 步 vs control 45 步（浅搜走法不同，局长度不可比）；训练口径应看步/s（≈样本/s）。
- **风险**（调研报告 §6.5）：fast 策略目标噪声（靠 value 目标 + ≥20% full 兜底）、`p_full` 需 A/B 标定。
- **A/B**：新增 `pcr-25` / `pcr-50` 臂：
  `python tools/ab_runner.py --config tools\ab_experiments.json --only pcr-25,pcr-50`
  判据：h2h 配对分为主（对 v3_iter560，120 局@400）。注意固定 32 轮时 PCR 臂更快但策略目标样本更少——若要按时间对齐，可对 PCR 臂单独把 `iterations` 调大（如 48–64）再比同等真机耗时后的 h2h。若 A/B 无益，保留开关默认关闭并在本节记录否决。

### S1.16 编码 64→71 通道（R2 修复，P1-6，2026-09-16）
- **依据**：`docs/issues/0002`——编码缺长捉/重复等**公开规则状态**，训练输入非 Markov（搜索侧不受影响：禁着仍在 `legal_moves` 过滤，编码只影响训练输入）。频率背景：随机局 chase≥1 占 18.8%；强评估 900 局中 repeat/no_capture 有实际终局占比。
- **实现**（按 issue 0002 方案）：
  - 样本：`RawSample`/`CompactSample` 新增 `check_anchor / chase_streak / chase_target / chase_target_code / chase_anchor / rep_count`；`raw_sample_from_info` / `finalize_samples` / `mirror_sample` 全链路透传；`compact_to_tensors` 以 `position_counts={(side, tuple(board)): rep_count}` 无损还原（`rep_count` 从 `position_counts.get(公共局面键, 1)` 取，与 `jieqi._position_key` 一致）。
  - 编码：`STATE_CHANNELS=71`，通道 64=`rep_count/3`、65~66=`chase_streak`（红/黑）`/6`、67~68=`chase_target` 单点 1.0（红/黑）、69~70=`chase_anchor` 单点 1.0（红/黑）；`PolicyValueNet` 首层 conv 输入 64→71。
  - 镜像增强：`check_anchor / chase_target / chase_anchor` 经 `mirror_index` 翻转（-1 保留）；`chase_streak / chase_target_code / rep_count` 原样复制（棋子码与计数非空间量）。
  - 数据版本：`DATA_VERSION=1→2`，`load_compact_payload` 校验拒绝旧文件（不静默降级）；`ruleset_version` 不变（规则语义未变）。
  - 模型兼容：新增 `load_state_dict_padded()`——旧 64 通道检查点首层 `conv.weight` 新增输入通道补零、其余权重逐元素不变，非首层形状不符则 `RuntimeError`；`load_model_if_exists` 自动走它，覆盖训练/评估/GUI/bench/test_window。
- **开关与默认**：R2 是结构性改动（无布尔开关），由 `DATA_VERSION=2` 全局生效；旧 64 通道模型经 pad-load 可继续评估/热启动（新通道初始零权重，待训练利用；首层仅 +11% 参数）。
- **验证**：新增 7 个单测（71 通道逐通道断言、chase/rep 差异、样本→张量无损、镜像翻转、pad-load 逐元素一致+前向、形状不兼容报错、v1 数据拒绝），全量 **119 passed**。冒烟（v3_iter560.pt、CUDA）：
  - pad-load：`conv.weight (64,64,3,3)→(64,71,3,3)`，非首层逐元素一致、新增 7 通道全零、前向 OK（0.06s）；
  - 短自博弈（16 sims、15 步）：规则字段写入正常、张量 (71,10,9)、save/load 往返一致；
  - `data/selfplay/` 352 个 v1 文件全部以 `data version mismatch: 1 != 2` 拒绝（预期）。
- **数据迁移/重生成（必须）**：v1 与 v2 数据不可混用。建议先归档：
  ```pwsh
  Move-Item data\selfplay data\selfplay_v1_archive
  New-Item -ItemType Directory data\selfplay
  ```
  再从 v3 热启动重生成（pad-load 自动生效；保留 `data/models/latest.pt` 与 `train_state.json` 以便续训；`--iterations` 是**绝对目标**，当前 `train_state.iteration=352`，故 352→383 写 384）：
  ```pwsh
  python azero.py --iterations 384 --self-play-games 64 --mcts-simulations 400 --self-play-workers 8 --self-play-log-interval 8 --train-steps 1000 --lr 1e-5 --lr-schedule none
  ```
  若要彻底从零（不复用 352 轮状态），把 `data/models`、`train_state.json` 一并归档后用 `--no-resume --iterations 32`；`v3_iter560.pt` 只作评估基线、不要删。
- **A/B（未跑）**：64 vs 71 通道为结构性改动，需两个代码快照各自从同一 v3 热启动、各自重生成 32 轮数据后 h2h 配对（成本 ≈2×3.5h 训练 + 评估）。建议提交后用「回退编码 commit 的 worktree」方式跑：主判据为两臂 h2h 配对分（同洗牌 120 局@400），辅以 chase/rep 终局转化率；若无显著差异，保留 71（更符合 Markov 假设且首层参数仅 +11%）。

### S1.17 v4 筛选：12→64 轮复核（2026-09-16/17）
- **背景**：S1.14–S1.16 落地后旧 A/B 结论全部作废（代码语义已变）。用 `data/ab_short` 先跑 5 臂 12 轮短筛，再对 4 臂续跑至 **64 轮**（`prepare_arm` 检测 `train_state` 自动续训；续跑配置给 base_args 加 `--max-load-files 32` 温缓冲，因 `max_load_files` 默认 0、缓冲本身不落盘）。`pcr-50` 停留在 12 轮（与 pcr-25 同假设且短筛更差，不再投入）。
- **评估**：h2h-v3（`model:data/models/v3_iter560.pt:400`，120 局@400，同 seed 20260911，全部 arm 共享配对）。为提灵敏度，对逐局 JSONL 做**跨臂配对分析**（脚本在 `<LOCAL_TEMP>\opencode\ab_paired.py`，只读；逐局换算神经侧得分后按 pair 配对），显著性以配对 CI95 为准。

| arm（LR 2.5e-5 基准=control） | 12 轮 | 64 轮 | 与 control 配对差（64 轮） | 判读 |
|---|---|---|---|---|
| `control` | 0.517 | **0.492** | — | 基准 |
| `lr-low-1e-5` | 0.425 | 0.450 | −0.042 [−0.156, +0.073] | 无差异（旧“1e-5 胜出”结论不成立） |
| `chance-full` | 0.400 | 0.450 | −0.042 [−0.159, +0.076] | 无差异（评估侧未开机会节点，见待办） |
| `pcr-25` | 0.463 | 0.388 | −0.104 [−0.228, +0.020] | 倾向更差（n.s.） |
| `pcr-50` | 0.400 | （停 12 轮） | — | 淘汰 |

- **结论**：64 轮内**没有任何新项超过 control**；`lr-low` 与 `chance-full` 与 control 统计不可分；`pcr-25` 点估计更差。v4 冻结候选维持 **control 配置**（常量 LR 2.5e-5、400 sims、chance 关、PCR 关、去重开）。
- **解释性观察（控制混淆）**：
  1. control 自身 64 轮后也只有 0.492（vs 父模型）——空缓冲 + 7 个新通道零初始化 + 修复后搜索语义带来的数据分布偏移是**全臂共享**的；且父模型也按修复后搜索重新评估，**不可与旧 0.5725 直接对照**。
  2. loss 缓升：policy 2.05→2.12、value 0.087→0.112（iter31→63），列为观察项，可能与分布偏移/目标变难有关。
  3. 等轮数 ≠ 等墙钟（实测每轮：chance-full +32%、PCR −17%）；pcr-25 的策略目标吞吐仅 0.44×，其劣势部分来自这一点，时间对齐复测未做。
- **待办（chance-full 的公平评估）**：当前 `tools/eval.py` 只构造 `AZConfig(mcts_simulations=sims)`、**无机会节点开关**，所以所有评估一律 chance 关，chance-full 的模型从未在“机会节点搜索”下评估过。建议给 `tools/eval.py` 补 `--chance-nodes/--chance-k` 参数，并做 **2×2**（control / chance-full 两个 64 轮检查点 × eval 开/关机会节点，各 120 局同 seed 配对）再对 chance 训练下最终结论；不要只给 chance-full 单侧开启（会把搜索差异混进模型比较）。

### S1.18 机会节点世界条件采样修复（Issue 0005，2026-09-17）
- **背景**：机会节点数学审查发现**回退污染**——搜索按边际先验独立采样 reveal 类型，与世界池不符时 `_apply_chance_move` 静默回退自然抽样，样本错记到分支。探针实测单次 reveal 回退概率：missing=4 → 3.9%、missing=8 → **18.4%**、missing=12 → **50.8%**；真实搜索（missing=6）**5.67%**。边际概率本身没有错误（`E_world[pool[t]/total] == pool_known` 归一化，探针验证），问题在 (世界, 揭示) 联合分布不匹配。
- **修复**（`azero.py::NeuralMCTS._navigate_chance`）：机会链导航改为从**当前模拟世界池计数**采样 reveal（双揭示两层分别用两色池、独立），(世界, 揭示) 联合分布与真实规则一致；`_apply_chance_move` 可用性检查降级为防御性兜底（世界条件采样命中时必然可用）；新增计数器 `chance_applied`/`chance_nav_miss`；`sample_outcome` 保留为边际采样工具与概率测试。
- **回归修复（同日）**：主实验首跑在 pair 5 崩溃（`TypeError: cannot unpack non-iterable int object`）——机会节点由先前叶展开按当时信念建立，不同模拟的世界池演化不同，支持集可能不相交，miss 后原代码继续对机会节点 `select_child`，把揭示子节点键（int）当着法传入 `apply_move`。修复：miss 时按世界自然抽样揭示并 **break 进入叶评估**（不再从机会节点选择）；新增回归测试 `test_search_chance_nav_miss_does_not_crash`。
- **记录**：`docs/issues/0005`（问题/修复/回归修复/验证）；探针入库 `tools/probe_chance.py`（4 段证据，含修复前后回退对比）。
- **验证**：新增 5 个单测（受控世界池导航、计数加权、双揭示两层、搜索回退=0、miss 不崩溃）；`test_azero` **77 passed**、`test_eval+test_ismcts+test_jieqi` **53 passed**（合计 130）；探针复验 missing=0/6 回退均 0（修复前 5.67%）；40 局 @400 sims 非对称 chance 评估无崩溃（on 侧 0.537，W17/D9/L14）。
- **工具**：`tools/eval.py` 新增 `--opponent-chance-nodes {inherit,on,off}` / `--opponent-chance-k`（默认 inherit=双方一致；非对称仅用于搜索 A/B）；`test_eval.py` 新增解析、解析函数、worker 独立配置测试（16 passed）。
- **兼容性**：修复改变机会节点搜索的采样流与语义——修复前的 chance 相关结果（含 `data/ab_short` 2x2-chance 批次）仅对旧版有效；chance on/off 配对主实验在修复后版本进行（见下条）。
- **方差副实验（Rao-Blackwell 机制验证）**：`tools/chance_variance_probe.py` 固定模型自博弈生成 21 个局面快照（ply 0→110，暗子 30→1），每局面 on/off 各重复 32 次 @400 sims，比较根 Q 跨次标准差。结果：配对差（off−on，>0=on 更稳）**+0.00130 ± 0.00179（95% CI，未达显著）**；暗子≥8 局面 **+0.00246**、暗子<8 局面 ≈0；单次搜索耗时 on/off = **1.23×**。方向符合分层降方差预测，优势集中在暗子多的局面。数据 `data/eval/chance_variance.jsonl`。

### S1.19 机会节点 on/off 配对主实验（Issue 0005，2026-09-17）
- **目的**：同模型（`v3_iter560.pt`）固定预算下，chance on 侧 vs off 侧配对 A/B；H1：on 侧期望得分 > 0.5（更宽的分层降方差 vs 更深的预算摊薄）。
- **工具**：`tools/eval.py` 对手独立开关（`--opponent-chance-nodes off`）+ 统计 `tools/paired_stats.py`（配对 d_i、bootstrap CI、配对 t、符号翻转置换）。
- **档 1（1600 局 @400 sims，800 pairs，双方同模型）**：on 侧 W/D/L=**637/335/628**，d̄=**0.5028，CI95=[0.4806,0.5244]**，t=+0.255，置换 p=0.818 → **无显著差异**；换色 red 0.490 / black 0.516。数据 `data/eval/chance_ab/s400_n1600.jsonl(.stats.json)`。
- **档 2（800 局 @800 sims，400 pairs）**：on 侧 W/D/L=**336/159/305**，d̄=**0.5194，CI95=[0.4900,0.5487]**，t=+1.281，置换 p=0.217 → 点估计 +1.9% 偏向 on，**未达显著**。数据 `data/eval/chance_ab/s800_n800.jsonl(.stats.json)`。
- **档 3（等时间对照：on@400 vs off@492，1600 局，800 pairs）**：492 ≈ 400 × 1.23（方差副实验实测耗时比）。on 侧 W/D/L=**597/301/702**，d̄=**0.4672，CI95=[0.4459,0.4888]**，t=−3.00，置换 p=**0.0026** → **off 侧显著更优（−3.3 个百分点）**。数据 `data/eval/chance_ab/s400on_vs_off492_n1600.jsonl(.stats.json)`。
- **解读**：同预算下"更宽 vs 更深"不可分（400 打平；800 点估计 +1.9% 不显著）；等时间约束下 off 显著占优——机会节点 1.23× 的单次开销未被宽度收益抵消（方差副实验显示分层降方差方向正确但幅度小、集中于暗子多的局面）。工程结论：默认关闭 `chance_nodes` 合理，保留为研究/诊断开关。
- **报告**：`docs/experiments/0001-chance-node-ab.md`（预注册设计+三档结果+机制副实验+局限）；复现命令与产物路径见报告第 6 节。
- **范围限定（2026-09-17 补记）**：`v3_iter560` 为 64 通道时代模型（评估时 pad-load 补零，通道 64–70 权重恒零）且自博弈训练全部由 off 搜索生成 → on 侧处于训练分布外，共适应偏倚方向利于 off；主实验结论仅限"该模型在手时的部署选择"。
- **相近证据复用判断**：`data/ab_short` 的 chance-full（自博弈开 chance，64 轮微调）与 control（off 对照）两次 h2h-v3 为 0.40/0.45 vs 0.5167/0.4917，但训练（9/16 20:25–9/17 13:22）与 2x2 评估（19:22–19:32）均在 Issue 0005 修复（azero.py 20:16 写入）之前、且为各臂 vs 基线（120 局）而非直接配对 h2h → 不能替代 chance-aware 对照实验。
- **待办（可选后续）**：`chance_k` 截断消融；暗子数阈值启用；机会节点在其他场景（更深预算/价值目标质量）的价值分析。

### S1.20 机会节点训练对照（chance-aware A/B，2026-09-18）
- **目的**：回答 0001 报告留下的范围限定——用 Issue 0005 修复后代码训练 chance-aware 模型，检验机会节点在**训练侧**是否有净收益。
- **配置**：`tools/ab_chance128.json`（两臂 v3_iter560 热启、128 轮、同超参，唯一变量自博弈 chance on/off）。`chance-full-128` 0→128 轮（全 chance 语义，13.7h）；`control-128` 复用 `data/ab_short/control` 的 64 轮 off 成果续训至 128（6.1h）。
- **瑕疵（已决策接受）**：`control-128` 续训按 `--max-load-files 32` 只重建最近 32 文件温缓冲（加载 254,288 = 32×2 augment；全量应为 512,920 = 64×2）——复制完整、属加载策略；接受差异（buffer 起步 254k、末期 ~77 万 vs 理想 512k→~102 万）。
- **结果**：各自 vs v3_iter560:400（120 局）——chance 0.550 [0.461,0.636]、control 0.579 [0.490,0.664]（CI 重叠）。**两臂直接配对 h2h（1600 局 @400 off-off）**：chance 侧 W/D/L=646/262/692，d̄=**0.4856**，bootstrap CI95=[0.4637,0.5075]，t=−1.29，置换 p=**0.202** → **无显著差异**（点估计 −1.4%）。
- **结论**：训练侧未观察到机会节点净收益；与 S1.19（搜索侧：等预算不可分、等时间 off 优）合观，机会节点在当前配置下无实用优势，默认关闭合理，保留为研究/诊断开关。
- **报告**：`docs/experiments/0002-chance-aware-training.md`；数据 `data/ab_chance128/`、`data/eval/chance128_ab/`。
- **待办（可选后续）**：2×2 交叉评估（on-on/混合配置）；256 轮长训练；`chance_k` 消融；机会节点用于价值目标质量的独立分析。

### S1.21 续训加载修复：按回放缓冲容量自适应（Issue 0006，2026-09-18）
- **问题**：`load_selfplay_data` 旧实现只支持"最近 N 个文件"（`--max-load-files`），与 `ReplayBuffer` 容量脱节。`control-128` 续训沿用了复制来的 `--max-load-files 32`，只重建 254,288 个样本（全量应为 512,920），buffer 起步偏离连续训练轨迹，削弱 A/B 公平性（背景见 S1.20 瑕疵）。
- **修复**：`max_files=0`（默认）时按 `buffer.capacity` 自适应——按 mtime 从新到旧扫描并累计（augment 每文件 ×2），达到容量即停（不再反序列化更旧文件），选中文件按旧→新灌入，FIFO 截断后保留的样本集合与连续训练一致；返回 `min(灌入数, 容量)`。`max_files>0` 保留旧语义（最近 N 个文件、不做容量裁剪）供历史配置复现。不新增独立预算参数——直接复用 `replay_buffer_size`，需要控量时调小容量或用 `max_files`。
- **验证**：新增 3 个单测（自适应保留最新、augment 预算 ×2、旧语义回归）；全量 `test_azero+test_eval+test_ismcts+test_jieqi` **136 passed**；真实数据 `data/ab_short/control/selfplay`（64 文件）：容量 1M → loaded=512,920（修复前 254,288）、容量 200k → 200,000、`max_files=32` → 254,288（旧行为不变）。
- **兼容性**：历史实验（`ab_short`、`ab_chance128`）结果基于旧加载策略，**结论不变、不重算**；修复只影响未来训练的启动加载。`tools/ab_chance128.json` 的 `--max-load-files 32` 保持原样，未来新实验不要复制该参数（默认自适应即可）；详见 `docs/issues/0006`。

### S1.22 楚河汉界翻面数据增强（2026-09-19，见 ADR-0011）
- **背景**：原有增强只有左右镜像（×2）；完整「棋盘+颜色」对称群（Klein 四元群）中「行翻面（楚河汉界翻转）+红黑互换」此前缺失。单独行翻面非法（黑在上红在下、兵方向/九宫/过河按色区分），必须绑定红黑互换+暗池互换。
- **实现**：`azero.py` 新增 `flip_vertical_*` 函数组与 `augment_samples(mirror, flip)` 组合入口；`AZConfig.augment_flip=True`（默认开）+ `--no-augment-flip`；训练循环 / `load_selfplay_data` / `merge_selfplay_files`（`merge_selfplay.py --augment-flip`）共用；两增强同开每样本 ×4（原样/镜像/翻面/镜像∘翻面），容量自适应预算同步按 ×4 计。实现要点：`chase_target_code` 存含颜色位的完整棋子码，换槽后必须翻转颜色位，否则翻面样本续捉断开。
- **验证**：数学（T=左右镜像∘(180°旋转+换色)，两个因子均为自同构，故 T 为自同构）+ 3 独立 agent + 裁判复核；lockstep 实证 500 局 70,805 ply：`legal_moves`/`game_outcome`/value 语义零偏差；改用正式实现 200 局复验未引入额外偏差。已知引擎索引序 tie-break 与翻面不同变（多子将军 `min(checkers)`、长捉同值目标扫描序；0.33%/ply 记账分歧），裁判 8192 局 509,298 ply 归档回放确认**禁着状态差异 0**。复现工具 `tools/verify_flip_symmetry.py`。新增 6 个单测，全量 **141 passed**。
- **待办（可选）**：`--no-augment-flip` 消融（先 50 局@400 快评、200 局@800 验收），判读 ×4 冗余 vs buffer 对局多样性在 Gate 规模下的净效应。

### S1.23 off 模式树节点 children 跨世界复用缺陷（Issue 0007，2026-09-19）
- **来源**：审查"翻红中兵出炮后黑方动作变少"——暗子揭示结果改变合法着法集合（翻炮将军：黑方 legal 44→6），而树节点 children 只由**首次展开世界**决定并全搜索共享；`apply_move` 不做合法性校验。
- **证据**：探针 `tools/probe_chance_tree.py`（12 seeds：children 6/44 波动，与翻炮概率 2/15 吻合；on 模式 6 个揭示分支各自独立 children=44/6）；污染量化 `tools/probe_world_mismatch.py`（3 局/模式 @400 sims）：random 基线 **0**，**off 走后自将执行 181/128,236 = 0.141%**（每 ~300 条模拟 1 次非法着法执行；受限方向：children=6 时 38 个着法永不探索），**on（修复后）0/123,836**。
- **影响**：off 搜索结果存在系统性偏差（非法局面进入叶评估 + 关键节点动作空间错误压缩）；S1.19/S1.20 等旧结果不能用于判断当前实现的收益。on 的公开翻子分支没有此问题，私有信息约束仍须单独审查。
- **状态**：2026-10-01 已修复并回归，见 §S1.26；上述数据保留为历史缺陷证据。
- **记录**：`docs/issues/0007-off-tree-world-mismatch.md`；探针 2 支 + 数据 `data/eval/probe_world_mismatch.jsonl(.log)`。

### S1.24 训练规模参数调研：sims / buffer / 每轮数据量（2026-09-19）
- **调研**：4 路并行 agent 对照 AlphaZero 系（AGZ/AZ/ELF）、Leela Zero/Lc0/KataGo、不完美信息（DouZero/PerfectDou/ReBeL/AlphaHoldem/Suphx）、象棋暗棋类（CCZero/icyChessZero/JiangJun/Hsueh CDC 2023/RNaD-JunQi/Pikafish jieqi），一手来源优先，全文 `docs/research_jieqi_training_scale.md`（含全部来源 URL 与不确定项）。
- **核心发现**：sims=400 与同类 9×10 项目（AZ/CCZero/icyChessZero/JiangJun 普遍 800）同区间、无需大动；**每轮 64 局是同类最低**（CCZero 窗口 5000 局、icyChessZero 攒 1 万局更新、AGZ 25k 局/iter、Hsueh 暗棋 1000 局/iter）；由此样本重复率 ~32×（新数据 16k/轮 vs 每轮训练展示 512k）高于 KataGo 建议的 4–8×，games:minibatches=0.064:1 远低于 ELF 硬约束 ≥10:1。揭棋无公开 AZ 先例，最可比学术参照为 Hsueh 2023（2×4 暗棋，含 chance）；其"N_sim 50 够用"结论因动作空间（40 vs 2550）不可照搬。
- **建议（按优先级）**：① `self_play_games` 64→128/256（256 为 `AZConfig` 原默认；提到 256 后重复率 8×、buffer 覆盖 62→~15 轮，两项同时进入推荐区间，代价是自博弈耗时 ×4）；② sims 保持 400–800、评估固定 800，不降到 200 以下；③ buffer 保持 1M（ELF 警告小 buffer 过拟合、Lc0 大窗口方差更低）；④ train_steps 暂保持 1000（ADR-0010 决策，局数 ×4 后重复率自动合规）；⑤ 后续 A/B 轴：局数、sims800（已有 arm）、PCR、MCTS 初始 W/N=0.5（Hsueh 低 sims 稳定化技巧，需先验证语义）。
- **状态**：仅调研与建议，未改任何训练/搜索默认参数。

### S1.25 v4 chance-aware 主线（2026-09-19 起，训练中）
- **起点迁移（已完成）**：`data/ab_chance128/chance-full-128` 的 128 个自博弈文件、`latest.pt`、`train_state.json` 迁至主位置（`data/selfplay/`、`data/models/`）；旧 v3 产物归档（`data/models/v3_iter479.pt`、`train_state_v3_iter560.json`、`logs/train_v3_iter560.log`、`eval/train_eval_v3.jsonl`）；起点模型另存 `data/models/v4chance_iter128.pt`。选择续训而非从 0：起点为 Issue 0005 修复后全枚举语义 + `RULESET_VERSION=3` + `DATA_VERSION=2`，迁移预演验证通过（模型可载入、LR 快进、buffer 自适应重建 1M ≈19s）。
- **配置**：`--iterations 2000 --self-play-games 256 --mcts-simulations 400 --strong-mcts-simulations 1600 --strong-self-play-interval 8 --chance-nodes --chance-k 0 --eval-interval 20 --eval-games 100 --eval-mcts-simulations 800 --ris-iterations 800 --lr 5e-5 --lr-schedule cosine --train-steps 1000 --self-play-workers 8 --self-play-log-interval 128`（LR cosine 5e-5→1e-5，T_max=2000；**2000 为上限非硬目标**）。
- **与 A/B 结论的关系**：S1.17/S1.19/S1.20 显示 chance 无显著优势（搜索侧等预算打平、等时间 off 更优；训练侧无净收益），但**无有害证据**；主线按用户决策采用 chance-full（训练/自博弈/评估同语义）。粗评收益主要来自每轮 256 局、增强 ×4 与长训。
- **进度（iter128→220）**：累计 32,000 局（train_state，含起点 128 轮）；~17min/轮（自博弈 ~850s + 训练 1000 步 ~107s，含每 20 轮评估均摊）；buffer 1M 饱和；avg_moves 67→52、和棋率 ~17%→5.5%（iter220：红 125 / 黑 117 / 和 14）。
- **评估曲线（训练内，100 局@800 vs RIS-800；标注者除外）**：iter139 0.5375（40 局）→159 0.575（40）→179 0.595（100）→199 0.65（100）→**219 0.725（100）**；`best_score=0.725`。**尚未做正式 200 局验收**（ADR-0001/0007 要求 ≥200 局；100 局 Wilson CI ≈ ±9pp）。
- **Ctrl+C 中断硬化（2026-09-20）**：根因＝Windows 下 Ctrl+C 同时发给主进程与 spawn worker，worker 死亡后主进程卡在 `ProcessPoolExecutor.shutdown(wait=True)`；修复＝`install_interrupt_handler()`（终止子进程→`os._exit(130)`）、worker 忽略 SIGINT、池异常路径 `shutdown(wait=False)`、等待循环 1s 轮询。真实 `GenerateConsoleCtrlEvent` 复现：退出码 130、延迟 0.9s、零残留；新增回归测试，全量 **143 passed**。
- **下一步（用户 2026-09-20 决策）**：再训 40 轮（至 ~iter260）后跑正式验收 200 局@800 vs `ris:800,random`（加 `--chance-nodes --chance-k 0`，与训练同语义）；**验收 ≥75% 视为停训目标**，停训后计划见 §5。

### S1.26 第一批搜索修复与回归（2026-10-01）

- **公开结果隔离**：NeuralMCTS 非枚举路径与 ISMCTS 增加按 `moved_reveal` 惰性创建的结果节点，分别保存各公开局面的后续动作。原动作节点汇总访问/价值；结果层不换手，修正 ISMCTS 价值回传符号，验证 NeuralMCTS 虚拟损失回收。
- **配置一致**：训练、GUI、评估默认机会节点开启且 `chance_k=0`；训练/评估支持 `--no-chance-nodes`。评估 CLI/Python/worker 对手默认继承己方开关与 k；保留显式覆盖。旧 A/B 配置固定控制臂 off、机会臂 on、评估 off，避免默认变化使实验臂相同，但不能复现缺陷版的搜索语义。
- **回归**：新增 8 个测试（`test_search_reveals.py` 5 个、`test_eval.py` 3 个），全量 **151 passed in 30.42s**。双首展顺序、on/off/ISMCTS 均验证炮 6 / 兵 44 个完整动作集合，每个搜索着法经过当前模拟世界合法性核验。
- **真实模型探针**：短局 @120 sims，off/on/RIS 搜索走后自将分别 **0/6,513、0/6,509、0/275,731**（旧 off 同口径 11/7,040）。文件 `data/eval/search_fix_20260930_world_mismatch.jsonl`；探针现排除合法性与终局查询内部试走，并支持 RIS 模式。只验证搜索正确性，未做棋力对比。
- **追加完整对局**：off/on 各 2 局 @400 sims，均自然终局，走后自将 **0/168,073、0/101,469**，对局实际着法同为 0；独立文件 `data/eval/search_fix_20260930_world_mismatch_long.jsonl`。
- **范围与风险**：未改规则/编码/网络结构，已有模型可加载；未启动训练、未覆盖历史产物、保留用户原 batch=1024/buffer=4M 改动。公开分支增多会改变 off/RIS 搜索和样本目标，可能增加内存并降低同模拟预算深度。明吃暗机会缺项、私有捕获信息集/对手知识建模、编码历史补全仍待后续，详见 `docs/reviews/2026-09-30-algorithm-assessment.md`。

### S1.27 合成行动者观测与信息集决策图（2026-10-01）

- **观测**：新增公共池账本，只扣公开揭示。每条模拟仍仅根确定化一次，`BeliefModel.for_simulation` 从根信息与粒子重建对手观测，不克隆真实对手私有槽。对手知道粒子中自己先前吃掉的类型，不知道自身被吃类型；叶策略/价值用合成行动者观测。
- **共享**：环境揭示结果连接按行动者公开/私有观测序列和计数信念共享的决策节点。红私知吃黑车/马不再拆黑决策；黑私知吃红车/马不拆红决策。吃方自己的捕获知识及搜索内私有事件顺序仍可区分。
- **机会与回传**：补明子吃暗子的私有随机结果。ISMCTS 的共享节点可有多条入边，改沿实际 path 回传；NeuralMCTS 保留价值视角及虚拟损失逐路径回收。
- **验证**：新增 `test_search_information_sets.py` 8 个测试，保持第一批合法性/完整动作集回归，强化真实秘密不可影响根结果、采样对手知识可影响自身策略等断言。全量 **159 passed in 29.50s**。
- **当前模型探针**：加载 `data/models/latest.pt`，短局 @120 sims off/on/RIS 搜索走后自将 **0/3,073、0/3,089、0/279,154**；@400 sims off/on 完整对局为 **0/13,483、0/9,171**，所有实际对局着法自将同为 0。文件 `data/eval/information_fix_20261001_world_mismatch.jsonl` 及 `_long.jsonl`；没有棋力对比结论。
- **兼容/限制**：71 通道、数据/规则版本未变，现有模型和数据可加载；未训练、未覆盖历史产物。更多决策键增加 CPU/内存成本。根前私有捕获仍为组成计数抽象，Q 仍按根采样分布回传，未实现策略条件后验或均衡求解。后续优先规则历史编码（check_anchor/重复历史）；`chance_k>0` 偏差等另排期。完整设计与复现见 `docs/adr/0012-search-information-sets.md`。

### S1.28 公开规则历史编码与产物兼容（2026-10-01）

- **修复**：将军锚点、长捉目标码进入网络；保留所有仍可能重复的公共棋盘、轮次与计数。吃子/翻明不可逆，剔除更多棋子/暗子的旧键；默认 80 ply 规则下最多 81 个相关键，超容量显式拒绝。叶去重签名包含完整相关历史，镜像/翻面变换所有历史棋盘。
- **格式**：238 通道保留旧 71 通道前缀，历史棋盘用公共码标量平面，样本用压缩字节记录。DATA_VERSION=3；读 v2 时缺失历史置 None，保留已有当前计数并编码完整性为 0，不伪造历史。旧 64/71 通道模型新增首层权重补零，三个 worker 初始化入口同步兼容；RULESET_VERSION=3 不变。
- **回归**：新增 8 项测试覆盖实际禁着/和棋反例、超过八步的历史、去重、不可逆裁剪、容量拒绝、序列化/增强、旧 slots pickle 与旧模型前向；全量 **167 passed in 31.23s**。翻面 20 局 2,803 ply 合法性/胜负零差异，12 ply 既有 chase 记账近似未放大。
- **真实产物**：latest.pt 加载、FP32 前向兼容及一次实际 train_step 通过（仅内存），新增通道梯度非零；真实 v2 文件 62,088 样本可读，缺失历史正确标记。10 局随机样本 1,559 个完整历史往返/增强通过。模型 off/on 各 @64 sims/30 ply 搜索走后自将均 0/1,920，实际走子为 0；无棋力结论。
- **成本/范围**：4060 Laptop batch=64 单次前向约 3.98→4.19 ms（约 +5%，不是完整训练基准）；batch=1024 FP32 输入增加约 58.7 MiB，4M buffer 新历史字节内容估约 422 MiB，另有对象开销。默认 TF32 存小幅数值漂移，严格 FP32 本次输出一致。保留用户 batch/buffer，不启动持续训练，不覆盖模型/数据/训练状态。仍未实现策略条件信念或完整私有历史。
- **证据/后续**：`data/eval/rule_history_20261001.json`、`rule_history_20261001_world_mismatch.jsonl`；设计与复现见 ADR-0013、`tools/probe_rule_history.py`。下一项优先 chance_k 固定支持偏差（默认 k=0）。

### S1.29 完整机会抽样与分支按需补建（2026-10-01）

- **修复**：导航先从当前模拟世界完整暗池抽取移动/捕获类型，再查找或补建结果节点；候选遗漏不重抽、不归一化到固定子集。k>0 改为各层初始预建上限，节点可超过 k；默认 k=0。预建先验保留原始概率质量，入口机会节点保存计数快照；先验不决定实际环境抽样。
- **回传/边界**：新增分支正常进入共享信息集决策图并沿路径回传，保持玩家视角及虚拟损失回收。明吃暗只抽被吃方池；被吃方仍不能按私有捕获结果拆策略。新增 chance_outcomes_added 计数；正常缓存遗漏不计 chance_nav_miss。
- **回归**：新增六项测试，穷举验证 k=0/1/2 的正确概率与平均价值、双揭示联合分布、缓存无交集、明吃暗、边际先验与世界概率分离、私有共享及 VL。第一批公开翻子合法性回归扩展至 on/k=1/2 和双首次顺序，保持炮 6 / 兵 44 完整着法集。全量 **173 passed in 30.47s**。
- **真实模型**：latest.pt @400 sims，自然终局四局。k1 两局 95 ply，搜索走后自将 0/59,541、机会导航 28,722/miss 0、补建 12,097；k4 一局 10 ply，0/6,076、4,482/0、补建 869；k0 一局 12 ply，0/7,219、4,318/0、补建 0。实际走子自将同为 0。文件 data/eval/chance_lazy_k{0,1,4}_20261001_world_mismatch.jsonl；不比较棋力或吞吐。
- **范围/后续**：238 通道、DATA_VERSION=3、RULESET_VERSION=3 不变；用户 batch=1024/buffer=4M 保留，未训练或覆盖历史产物。旧 k4 实验语义已改变，结果不可直接复用。动态缓存会增长，快照也有成本，先测完整训练吞吐和峰值内存；策略条件后验、完整私有历史仍为算法工作。设计与复现见 ADR-0014。

## 3. Gate 运行手册

> **2026-09-14 更新**：v3 Gate 已于 iter560 停训（见 S1.11）；下面的完整 Gate 命令是历史记录，后续调参/训练走 §3b 的 A/B runner。

> 注意：旧 v2 数据与模型已归档；`data/selfplay` 现有 352 轮产物（`train_state.json` iteration=352）。默认 **续训**：`--iterations 672` 即从 352 跑到 671。若要干净复现，先将 `data/selfplay`、`data/models` 归档后从 iter 0 重跑。

```pwsh
cd <PROJECT_ROOT>
$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8'
python azero.py --iterations 672 --lr 5e-5 --self-play-games 64 --mcts-simulations 400 --strong-self-play-interval 8 --eval-interval 16 --eval-games 16 --eval-mcts-simulations 400 --ris-iterations 200 --eval-workers 8 --self-play-workers 8 --self-play-log-interval 8 --train-steps 1000
```

- 配置含义：总目标 672 迭代（本次续训 +320 轮 ≈ 2 万局，累计 43,008 局）；每 8 局 1 局强自博弈（1600 sims）；每 16 迭代评估一次（16 局@400 vs RIS-200）；每轮固定 1000 步训练（ADR-0010）。
- **LR（软续训）**：`--lr 5e-5` 使 cosine（T_max=672）从 ≈2.85e-5 平滑衰减至 1e-5；若不加 `--lr`，默认 5e-4 会在快进后跳回 ≈2.4e-4（warm restart）。`--lr-schedule none` 可切恒定 LR。
- **Pilot 结果（2026-09-13 已执行）**：续训 32 轮（iter 320→351，`--iterations 352`）后对 RIS-800 50 局@800：0.20（10W/0D/40L）→ **0.33（16W/1D/33L）**；配对 13 好/6 差/31 平；和棋率 43%→34%；value loss 0.424→0.241。判据通过 → 续跑完整 Gate。
- 实测速度：pilot 32 轮平均约 6.3 分钟/迭代（含周期性评估）→ +320 轮预计 **约 34–37 小时**。
- 续训：默认 `resume=True`，从 `train_state.json` 的迭代继续；Ctrl+C 或杀进程后重跑即可。
- 监控与进度（逐行复制**命令本身**，不要带后面的中文标点，否则 `UTF8；` 会被当成编码名报错）：
  ```pwsh
  Get-Content data\logs\train.log -Tail 20 -Encoding UTF8        # 日志快照
  Get-Content data\logs\train.log -Tail 20 -Wait -Encoding UTF8  # 实时跟踪，Ctrl+C 退出
  Get-Content data\models\train_state.json -Encoding UTF8        # 当前迭代/局数/best_score
  ```
- 首次评估在第 16 迭代（`iteration=15`）写入 `data/eval/train_eval.jsonl`；此前 `best_score=-1.0` 属正常。
- 停止：**直接 Ctrl+C 即可**（2026-09-20 起：终止子进程后 ~1s 内硬退出、退出码 130；当前未完成的迭代不保存，重跑从上一个完成迭代继续）。若用旧版本或异常残留，再手工清理：
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'spawn_main|azero' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`

**Gate 判据**（`docs/adr/0005`）：快评（50 局@400 vs RIS-400）胜率曲线显著上升、对随机基线 >90%、loss 趋势正常；约 2–5 万局后评估。

**Stage 1 验收结果（2026-09-12）**：`python tools/eval.py --model data/models/best.pt --opponents ris:800,random --games 200 --sims 800 --workers 8 --out data/eval/acceptance.json`（3.5h）→ vs RIS-800 **0.2925**（58胜/1和/141负，CI 0.234–0.359；红方 0.345 / 黑方 0.240）；vs random **0.9525**（183胜/15和/2负，CI 0.914–0.974）。**未达 ≥65% 验收线**（信号成立但强度不足）。注意训练内评估（16 局 vs RIS-200）显著乐观：对手更弱、样本小、`best` 取最大分有偏。

## 3b. v4 A/B 实验手册（Stage 2A / 调参）

```pwsh
cd <PROJECT_ROOT>
$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8'
python tools/ab_runner.py --config tools\ab_experiments.json --dry-run          # 先看命令
python tools/ab_runner.py --config tools\ab_experiments.json --only control     # 单跑一个 arm
python tools/ab_runner.py --config tools\ab_experiments.json                    # 串行跑全部
```

- 机制：每个 arm 在 `data/ab/<name>/` 独立目录里从 `data/models/v3_iter560.pt` 热启动、**重放缓冲清空**、常量 LR 2.5e-5（保证只变一个因素）；跑完自动评估 h2h（vs v3_iter560，120 局@400）与 RIS-800（100 局@800），结果写 `data/ab/results.jsonl` 并打印汇总表。中断后重跑会自动跳过 `.done` 的 arm。
- 默认 arm（在 `tools/ab_experiments.json` 里改）：`control` / `chance-full`（修复后全枚举机会节点，2026-09-16 新增）/ `chance-k4`（旧截断近似，留档对照）/ `lr-low-1e-5` / `lr-high-5e-5` / `buffer-250k`（250k+500 步）。
- 成本：每 arm 32 轮 ≈3.5h 训练 + ≈3h 评估；建议分批（先 `control,chance-full`）。
- 判读：主判据是 **h2h 配对分**（对旧模型，灵敏度高）；RIS-800 100 局只做粗筛（±10pp 噪声）。ARM 差异要超过噪声才有意义。

### A/B 筛选记录（2026-09-15，R3 修复后的同码批次）

> **⚠ 2026-09-16 复核提示**：本批运行于 P0-1 修复（见 `docs/reviews/2026-09-16-adversarial-review.md`）**之前**——叶评估读取的是行动方 observer 槽（对手节点含私有观测），搜索值/Q 被污染；`chance-k4` 的叶机会先验同受影响。**本表结论只能作参考**，冻结 v4 配置前须用修复后代码重跑关键 arm（至少 `control` + `lr-low-1e-5`）；重跑结果不可与本表逐位比较（配对评估协议 P1-5 同时已变更）。另：机会节点实现已于 2026-09-16 修复（概率采样 + 全枚举 + 双揭示两层，见 S1.13），本表 `chance-k4` 行结论同时受该修复影响，重测请用新增的 `chance-full` 臂。

| arm | h2h-v3（120 局） | 配对均值 | ris800（100 局） | 训练耗时 |
|---|---|---|---|---|
| control（2.5e-5） | 0.438 [0.352, 0.527] | −0.125 | 0.51 | 3.0h |
| chance-k4 | 0.342 [0.263, 0.430] | −0.317 | 0.52 | 3.3h |
| lr-low-1e-5 | **0.529** [0.440, 0.616] | **+0.058** | 0.53 | 3.1h |
| lr-high-5e-5 | 0.383 [0.301, 0.473] | −0.233 | 0.45 | 3.1h |
| sims800（800/3200 sims） | 0.475 [0.388, 0.564] | −0.05 | 0.445 | 5.7h |

- sims800 附加口径：`h2h-v3-s800`（双方均 800 sims）= 0.442 [0.356, 0.531]。
- 配对符号检验（同种子 120 对）：lr-low vs lr-high **50/26, p=0.006**；lr-low vs control 47/32, p=0.091；chance-k4 vs control 27/42, p=0.071；**sims800 vs control 43/37, p=0.502（无显著差异）**。
- 结论：**LR 是唯一有单调趋势的因素**（1e-5 > 2.5e-5 > 5e-5，h2h 与 ris800 方向一致）→ 下一阶段默认 `lr 1e-5`；`chance-k4` 淘汰（两指标均不占优，交互于单次确定化可能削弱）；`lr-high` 淘汰；`sims800` 淘汰（统计上无差异、成本翻倍——搜索精度的收益被数据多样性下降/价值头误差放大/目标分布偏移抵消；32 轮微调预算不足以体现深度收益）。
- 注意：control（2.5e-5、32 轮）h2h=0.438 **低于起点 v3**（0.5 基准线下），提示 v3 权重上继续 2.5e-5 微调 32 轮已负收益；v4 正式训练需关注 LR/轮数配置。
- `buffer-250k` **未运行且判定为无效实验**：1M 默认缓冲在 32 轮内零淘汰（iter30 末 buffer≈258k），且该 arm 同时改 `--train-steps 500`（双因素）；要测 buffer 需 50k 级激进值或 ≥80 轮，并保持 train-steps 一致。

### sims800 arm（已跑完，2026-09-15；结论：无收益，淘汰）

- 配置（`tools/ab_experiments.json`）：`{"name": "sims800", "args": ["--mcts-simulations", "800", "--strong-mcts-simulations", "3200"]}`，其余同 control（64 局、1000 步、LR 2.5e-5）；评估 spec `h2h-v3-s800`（arm 与 v3 对手均 800 sims）。
- 实际成本：训练 5.7h（自博弈约 8min/轮，约 control 的 2 倍）。
- 口径说明（重要）：
  - 本 arm 同时改变**搜索次数**与**总计算量**（普通 400→800、强局 1600→3200），按"同时间预算下哪种配置更优"解读。
  - **根视角单次确定化是正确的信息模型**（玩家只能按己方观测推理、模拟对手时只给它该知道的），不是缺陷；400→800 增加的是**每条模拟的根世界采样数**（每条模拟重新 `root_info.determinize`），不是"在同一世界里推得更深"。据此，"搜索更深→学到更好知识"在 32 轮微调 + 未收敛价值头下不成立；无收益不是搜索原则问题。
- 历史运行命令（留档）：
  ```pwsh
  python tools\ab_runner.py --config tools\ab_experiments.json --only sims800
  python tools\ab_runner.py --config tools\ab_experiments.json --only control --force --skip-train   # 为 control 补 800-sims 评估（旧评估文件复用）
  ```
- 副作用记录：`--force` 重写了 control 行的 `train_seconds`（如需可从 `data/ab/results.jsonl` 更早的行恢复）。

## 4. 已知问题 / 注意事项

- **数据版本（2026-10-01）**：`DATA_VERSION=3`（238 通道，ADR-0013）；v2 可读，完整重复历史标记缺失；v1 仍拒绝。旧 64/71 通道模型 pad-load 可继续评估/热启动，用户现有产物未归档或覆盖。
- **中断行为（2026-09-20）**：Ctrl+C 会立即终止全部 spawn worker 并以 130 退出（~1s），不再出现进程挂起；未完成的迭代不保存（见 §S1.25）。
- 叶并行 + 虚拟损失会改变搜索统计（与 `batch=1` 非严格等价）；质量差的 A/B 验收放在 Gate 训练信号中验证。
- RIS 速度约 137 iters/s → RIS-800 约 5.8s/步；200 局验收必须并行（`--workers`）。
- 完整续训现在从 `training_checkpoint.pt` 恢复 AdamW/调度/RNG/轮计数；`latest.pt` 仍为纯权重兼容导出。只有旧权重时仍无法恢复过去的动量/RNG；cosine 改配置需显式 `--reset-optimizer`。见 ADR-0015。
- 评估触发条件为 `(iteration + 1) % eval_interval == 0`。
- 弱模型快评可能长期接近 0；Gate 看的是曲线趋势而非绝对值。
- Pikafish `jieqi` 分支需自行构建（Stage 2，注意 GPL-3：仅作数据/对手，不复制源码）。
- value loss 的“U 形回升”（S1.11）与 LR 衰减 / 重放陈旧 / 目标变难三者纠缠，用 §3b 的 `lr-*`、`buffer-250k` arms 分离；训练日志建议后续补 value–结果相关等 scale-free 指标。**2026-09-15 更新**：LR 因素已分离（见 §3b 筛选记录，1e-5 最优）；buffer 实验因双因素+无淘汰被判定无效、未运行。

## 5. 历史 v4 主线计划及后续更新

本节 2026-09-20 的续训/验收计划保留为历史记录；当前用户安排和开发顺序以 [status.md](status.md) 为准，不自动启动以下长训或大规模评估。

1. **当前（2026-09-20）**：v4 主线训至 iter220（32,000 局），训练内 100 局@800 vs RIS-800 评估 **0.725**（iter219）；**正式验收尚未做**。用户决定再训 40 轮（至 ~iter260）后按 ADR-0007 跑 200 局验收（与训练同语义）：
   `python tools/eval.py --model data/models/latest.pt --opponents ris:800,random --games 200 --sims 800 --workers 8 --chance-nodes --chance-k 0 --out data/eval/acceptance_iter260.json`
2. **停训判据**：2000 iter 是上限而非目标；每 ~200 轮看评估增量，连续两次窗口增量 <2pp 视为平台期即可停训；**正式验收 ≥75% 视为停训目标**（用户 2026-09-20 意向；ADR 验收线仍为 ≥65%）。
3. **验收 ≥75% 后的计划（待确认）**：
   - **收尾归档**：发布模型另存（`data/models/v4_iter*.pt`）、验收报告入 `data/eval/`、更新 ADR/CONTEXT/README（git 提交由用户决定）。
   - **产品化（推荐，符合 ADR-0001 主线目标）**：GUI 试玩反馈、打包分发、README/玩法说明、可选难度档。
   - **上限提升（可选，成本高）**：2B 蒸馏（Pikafish-jieqi 教师 → 标注数据 → 监督预训练 + 自博弈微调；GPL-3 仅作数据/对手，不复制源码）；搜索改造（Issue 0007 修复、Gumbel 根搜索、chance_k 消融）——仅在需要继续冲强度时启动。
4. **验收 <75%**：按停训判据继续训练；若 2000 轮内仍 <75%，转到 2B 蒸馏或接受当前强度并产品化（二选一）。
5. **当前后续**：公开分支、合成对手知识/共享决策、规则历史编码及固定机会支持偏差四批已完成（§S1.26–29）。短时资源/学习信号验证见本文件顶部与 `docs/reviews/2026-10-01-training-budget.md`；用户暂不在笔记本投入长训。以后在新目录热启动、采集修复后数据，并测完整 4M/8-worker 峰值与统一口径配对评估。策略条件后验属于后续算法改造；旧对比不作为当前算法选择依据。
6. C++ 后端若重启：先写 conformance 测试向量（perft + 随机对局回放）再移植。

## 6. 文档地图

| 路径 | 内容 |
|---|---|
| `docs/status.md` | 当前事实、验证边界、开发优先级、工具与产物入口 |
| `docs/adr/0001–0011` | 目标/验收、规则基准、禁着、重复与和棋、训练路线与 Gate、C++ 定位、评估协议、数据版本管理、v3 规则修正与审查修复、每轮训练步数、楚河汉界翻面数据增强 |
| `docs/adr/0012–0014` | 合成行动者观测与共享决策图、公开规则历史编码及模型/数据兼容、完整机会抽样及动态分支 |
| `docs/adr/0015` | 完整续训、原子轮次提交点、缓冲清单与每局边界、旧纯权重兼容 |
| `docs/research_jieqi_azero.md` | 揭棋规则/AlphaZero/不完美信息/开源生态调研（含来源 URL） |
| `docs/research_jieqi_search_training.md` | 搜索/训练优化调研：LC0/KataGo 谱系核实、机会节点、MCGS 量化否决、优化选项表（2026-09-16） |
| `docs/cpp_port_verification.md` | C++ 端口验证报告与结论 |
| `docs/progress.md` | 本文件（进度与交接） |
| `CONTEXT.md` | 领域术语表与关键常量 |
| `README.md` | 使用说明（GUI/训练/评估/基准） |
| `tools/ab_runner.py` + `tools/ab_experiments.json` | A/B 串行实验 runner 与 arm 定义（产物在 `data/ab/`，不入库） |
| `data/legacy_20260911/` | v0/legacy 规则下的模型、数据与日志切片（`logs/train_legacy_20260113-20260226.log`、`sum_command.txt`）归档（不删除、不再加载） |
| `data/legacy_ruleset_v2_20260911/` | v2（含两处规则缺陷）的模型、训练状态、384 局自博弈数据、smoke 评估记录与日志切片（`logs/train_ruleset_v2_20260911.log`） |
| `data/logs/train.log` | 仅保留当前 v3 run（2026-09-11 16:27 起）；重训/换版本前先归档切片，避免混入历史 |
