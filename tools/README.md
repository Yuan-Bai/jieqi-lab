# 工具索引

更新：2026-10-22。当前训练入口是根目录 `azero.py`，GUI 入口是 `gui.py`。
本目录同时包含日常工具、验收工具和隔离实验；当前加速状态以 [加速配置](../docs/performance.md) 为准。
当前结论、停止清单与剩余工作见 [研究总览](../docs/research-status.md)；
[阶段总结](../docs/reviews/2026-10-08-acceleration-stage-summary.md)保留当时环境与历史复现限制。旧工具对旧源码的对照不自动是当前版本的单变量实验。

## 日常与构建

| 入口 | 用途与边界 |
|---|---|
| `eval.py`、`paired_stats.py` | 冻结历史池/Pika配对评估与bootstrap统计，输出使用新路径；使用自己的冻结清单固定预算并校验权重；日期清单仅作历史参考，模型未分发，见[清单说明](../configs/README.md) |
| `run_status.py` | 只读核对运行来源、attempt 与提交状态 |
| `generate_example_record.py` | 无模型、无搜索的合成分支棋谱生成器；`--out` 只写新文件，见 [示例](../examples/README.md) |
| `diagnose_training.py` | 只读检查固定训练参照/整局留出诊断 |
| `bench.py` | 自博弈吞吐探针，不执行梯度更新 |
| `build_rules.py`、`build_legal.py` | 构建攻击判定/合法着扩展 |
| `build_search.py` | 构建可选节点实验扩展，不推荐默认启用 |

历史模型池、Pikafish `jieqi_old` 外部对手、配对 bootstrap 统计和命令见
[对手池评估](../docs/evaluation.md)。训练用 `azero.py --eval-pool`，默认只报告，
不按旧RIS高分自动更新 `best.pt`；原有训练参数必须沿用，详见该文档。

先看 `--help` 与报告中的运行条件；带模型/数据参数的工具必须核对目标目录和版本。
所有模型、Pickle 数据与实验 fixture 必须来自可信来源，见 [SECURITY](../SECURITY.md)。
构建脚本会更新本机二进制，构建期间不要同时开始使用旧二进制的新训练进程。
原生采样/预检脚本从 `JIEQI_PYTHON`、`JIEQI_NSYS` 或 PATH 解析工具；准备文件会记录实际路径，换机器需重新准备。

## 正确性与专项诊断

| 入口 | 检查内容 |
|---|---|
| `verify_flip_symmetry.py`、`audit_augmentation.py` | 翻面对称性、增强与根噪声 |
| `probe_input_audit.py`、`probe_rule_history.py` | 输入与公开规则历史 |
| `probe_world_mismatch.py`、`audit_information_probe.py` | 模拟世界合法性与信息边界 |
| `probe_gui_replay_privacy.py` | GUI 回放的私有信息隔离 |
| `probe_web_background.py` | Web 服务 Windows 调度策略的 AZ 固定根 ABBA 短测；预热、预算/叶批/种子固定，不切窗口、不改系统设置、不训练，不等同于终端焦点切换复现 |
| `probe_chance.py`、`probe_chance_tree.py`、`chance_variance_probe.py` | 机会分支、树结构与价值方差 |
| `probe_interrupt.py` | Windows 训练中断通路，使用隔离隐藏控制台 |
| `probe_training_budget.py` | 资源与短时学习信号，不是棋力验收 |

专项结果只支持其检查的机制，不能把探针胜负或单次资源测量当作正式棋力结论。

## 加速测量

| 入口 | 对应范围 |
|---|---|
| `benchmark_encoding.py` | CPU 编码/装配与精确更新对照 |
| `benchmark_exact_encoder.py` | 仅向data/pilots构建实验编码pyd；36对全图/网络与实际输入位级验收、局部微测及双池576根筛选。候选根吞吐+0.411%未过门槛，不接生产、不做整局；不改变重复历史准备 |
| `benchmark_exact_selector.py` | 仅向data/pilots构建对象算术PUCT选择实验pyd；201408次选择、36对图/网络与576根精确。局部省15.05%，根吞吐-0.401%，按门槛停止，不接生产、不做整局 |
| `measure_repetition_preparation.py` | 原函数影子统计：36根1719703旧棋盘身份查询、有界复用/过滤/转换比例及组件成本，结果精确；不缓存或改变生产输出 |
| `benchmark_repetition_memo.py` | 隔离公共不可变棋盘身份FIFO原型，仅data/pilots构建；局部三模式消融、64479条目/36对图网络/576冷根精确，根吞吐-0.967%未过门槛，不接生产/做整局 |
| `benchmark_selfplay.py` | 策略输出数组化与批内历史复用 |
| `benchmark_native_rules.py` | 攻击判定编译 |
| `benchmark_native_legal.py` | 合法着快路编译 |
| `benchmark_native_search.py` | 节点扩展/初始化编译 |
| `benchmark_compiled_paths.py` | 三项编译路径的组合长测 |
| `benchmark_chance_allocation.py` | eager/lazy 机会对象分配精确配对 |
| `benchmark_chance_support.py` | 完整 k=0 机会表重复率、原快照查表成本、完整图/网络流与整局精确配对；不更新模型 |
| `benchmark_information_snapshots.py` | 改动前搜索函数与中间层直接信息集键的完整图/网络流、快照次数和整局精确配对；不更新模型 |
| `measure_selfplay_round.py` | 全部保留加速同时开启的1024局基线：正式调度/完整样本IPC、任务/尾部/低频资源测量、隔离增强保存及另测18根热点；不更新模型，不重载完整buffer |
| `measure_selfplay_attribution.py` | 短CPU/网络边界归因：嵌套墙钟/线程CPU、原评估语句CUDA events、CPU算子时间线与精确/开销对照；evaluation模式可只测固定根，不训练 |
| `native_profile_preflight.ps1` | 历史2022.4.2预检，不再重跑；新版实际版本/帮助/签名已独立核对，见归因报告 |
| `profile_native_selfplay.py` | 普通权限先验收仅NVTX标签的36图/网络流及四组精确固定根对照；采样模式仅复跑冻结根并核对策略/RNG/计数，不训练 |
| `capture_native_selfplay.ps1` | 固定新版2026.5.1与CompatibilityFile/准备哈希；采集全生命周期，仅分析预热后NVTX测量窗口；cuda模式普通令牌可跑，CPU模式需管理员；SQLite另步导出，不终止目标或改策略 |
| `probe_native_runtime.ps1` / `.py` | 新版合成兼容性门禁：marker/single/spawn，triggered/lifecycle，cuda/cpu/combined；准确验证小矩阵/卷积，无棋局/模型/训练，CPU模式需管理员 |
| `inspect_native_runtime.py` | 只读合成SQLite，按metadata选择隔离cuda/cpu：核对PID/闭合NVTX窗口，CUDA查launch correlation/计数，CPU查目标主线程栈/模块/调度；保留诊断，不支持combined；使用`python -m tools.inspect_native_runtime --probe-dir ...` |
| `inspect_native_selfplay.py` | 只读生命周期隔离cuda/cpu固定根SQLite，检查144根/8主线程/阶段；CUDA汇总API/核/拷贝，CPU仅周期叶样本按模块/阶段归因并由调度转换重建区间；不把累加当父墙钟分解，不证明忙等或无丢失 |
| `measure_sync_baseline.py` | 消费原v5哈希参照；无profiler/events/新增同步，双8-worker池三模式/六种顺序复核时序与纯主机阶段计时，36图/网络流及1296根精确；不实现等待优化，不训练 |
| `benchmark_host_wait.py`、`benchmark_host_wait_games.py` | 仅消费通过的哈希参照，独立blocking主机等待的36网络/图/576根筛选与追加无hook图/128完整局筛选，原worker完整样本IPC；不改生产/精度/预算/系统设置，无训练开关 |
| `profile_python_selfplay.py` | 固定参照的Python栈采样工作负载、双子进程兼容性探针和本地Speedscope解析；外部隔离py-spy 0.4.2，按已知PID/主线程/正式搜索栈标记排除热身，576根精确；条件GIL样本而非纯CPU时间，不训练 |
| `probe_native_launch.ps1` | 历史2022.4.2启动异常诊断，不再重跑；三次PSModulePath对照证据保留于归因报告 |
| `stop_failed_native_capture.ps1` | 默认只读保存失败采样快照；显式StopProcesses且管理员权限才按日志、路径、创建时间、父子/预热身份核对后定向结束冻结采样进程；不按名称批量杀进程 |
| `profile_selfplay.py`、`profile_search_expansion.py` | 多局面剖析与低扰动边界计时 |
| `measure_search_reuse.py` | 未访问机会对象、网络输入复用与查表成本 |
| `measure_network_cache.py` | 跨批次缓存重复率、成本与数值门槛 |
| `benchmark_inference_precision.py` | 冻结模型TF32/FP16/BF16推理数值、搜索与整局隔离测量，不更新生产模型 |
| `eval_inference_precision.py` | 吞吐筛选通过后，同权重FP16对基线的同洗牌换色棋力试验；不作为训练验收 |

这些工具可能进行真实 CUDA 前向或隔离优化更新，并消耗较长时间；具体副作用边界见脚本和关联报告。
它们不是生产训练的启动入口。`data/pilots/` 存放隔离结果，禁止覆盖旧输出或写回生产权重。
当前Python函数栈的冻结输入、隔离采样器哈希、复现命令和警告/开销口径见
[函数热点报告](../docs/reviews/2026-10-09-python-hotspots.md)；不上传报告、不读取locals、不安装全局依赖。
两档粗排名一致，不把条件GIL排名当作可省时间；后续直接编码/PUCT编译均未通过整体收益筛选。
编码直接编译随后完成，局部省时24.27%但根吞吐仅+0.411%，按门槛停止，见
[编码筛选报告](../docs/reviews/2026-10-09-exact-encoder.md)。实验扩展没有生产加载器，
`JIEQI_EXPERIMENT_ENCODER_DIR`仅供编译专测，普通训练不会读取；实际启用状态不变。
PUCT选择直接编译也已筛选，局部省15.05%但根吞吐-0.401%，停止，见
[选择筛选报告](../docs/reviews/2026-10-09-exact-selector.md)。`JIEQI_EXPERIMENT_SELECTOR_DIR`
同样只用于编译专测，无生产开关；不改回传、信息集或模型精度。
重复历史随后完成先测后原型：身份复用88.65%但转换重复33.55%；局部准备省69.03%，
双池根吞吐-0.967%，未过门槛，见[历史准备报告](../docs/reviews/2026-10-09-repetition-preparation.md)。
`JIEQI_EXPERIMENT_REPETITION_DIR`只供专测；缓存不含可变信息态、计数、世界或网络，不读入普通训练。
三个CPU候选停止，不扩大本轮散点微优化；局部热缓存成本不外推完整局/续训收益。
旧源码快照与依赖路径属于报告复现条件，单有工具文件不代表旧实验在任意机器上可直接重跑。
混合精度首测与限制见 [推理精度实验](../docs/reviews/2026-10-08-inference-precision.md)：FP16固定根有收益，完整局收益尚不稳定，棋力/收敛未验收；没有生产精度开关。
后续 [512局FP16复验](../docs/reviews/2026-10-08-fp16-followup.md) 未过预声明吞吐筛选；网络微测仍快，完整局合计慢1.87%。`eval_inference_precision.py`拒绝该报告启动棋力；原型只有单测通过，不是已完成棋力验收。
机会表复用见 [独立实验报告](../docs/reviews/2026-10-08-chance-support-reuse.md)：它只缓存不可变计数推导的概率表，
不缓存网络输出或合并搜索统计。报告区分微测、固定根和整局吞吐，并保留失败候选与预声明门槛。
中间层快照削减见 [独立实验报告](../docs/reviews/2026-10-08-information-snapshots.md)：根/叶完整观测保留，
只削去生成相同信息集键之前的中间层复制；两模式都使用前一轮机会表缓存，避免混入其收益。
完整联合轮的协议、结果与未测边界见 [联合测量](../docs/reviews/2026-10-08-joint-selfplay-round.md)。
该工具要求新隔离目录和psutil；输出样本只用于测量，不导入生产buffer。单轮当前基线不是旧/新配对加速率。
已完成的联合轮为1024局/128强局，5696.47s；热点插桩另测，不能把其边界占比当整轮纯CPU占比。
工具拒绝串行worker/非cuda:0设备及不匹配的leaf64/virtual loss3，避免静默走不受任务日志覆盖的路径。

### 原生采样历史与当前状态

新版Nsight2026.5.1隔离CUDA/CPU采集和无跟踪复核均已完成，不再等待管理员采样。
以下保留权限限制、旧版本失败与启动器演变，按各次实验时点解读，不是待执行步骤。
短归因的权限、计时口径与开销见 [归因报告](../docs/reviews/2026-10-08-selfplay-attribution.md)。
本机无法在当前令牌下取得Nsight/WPR原生栈；CUDA events不是纯核执行时间，CPU算子trace也不是CUDA trace。
工具要求固定18局面/8 worker/现有预算及新隔离目录，不返回生产全样本IPC，不用其cohort计算生产加速率。
管理员预检已经确认本机Nsight 2022.4.2的Windows参数；准备/采样工具和后续操作也见归因报告。
CPU采样符号先不解析，不启动符号服务器；CLI成功仍需核对实际子进程CUDA覆盖、栈表和丢失事件。
准备工具和管理员采样均保持rules/legal=Cython、nodes=Python、原精度/预算；不要与训练并行。
首次管理员启动报`bad conversion`但Nsight返回0，未取得采样；已修正提前报错。
三组诊断证实移除子进程PSModulePath可消除转换异常，恢复后重现；第二组报告生成但未捕获Python打印。
启动器只移除该子进程变量，并增加实际Python文件标记自检；父/系统环境与精度变量不动。
`native_profile_process.ps1`是启动器共享的隐藏进程/实时异步日志与有限等待辅助函数，加载本身无副作用；
`test_native_profile_process.ps1`只用普通Python验收环境/真实标记、实时双管道、继承管道故障和超时不杀目标；
只清理测试自己创建的休眠进程，不运行Nsight/训练/GPU。
v3正式采样及v4 CUDA-only均在Processing events阶段发生同一Nsight崩溃，不能当成有效trace；退出保护已生效，但工具内部原因未定。
停止旧2022.4.2重试；用户已清理残留并独立安装2026.5.1，marker/单CUDA/双spawn验收完成。
父进程NVTX触发丢过一个子区间，生命周期模式连续三次已知操作覆盖通过；所有警告仍保留，不宣称无丢失。
新版改用全生命周期采集并按闭合窗口分析，必须重新完整准备v5；旧v2/v3/v4采样命令不再重复运行。
v5准备和普通令牌8-worker CUDA-only已完成；144根/8PID/4510阶段组和已记录CUDA对应通过，24项保护哈希不变。
旧CUDA报告的额外Perf事件源拒绝访问仍保留；管理员CPU-only双spawn及8-worker冻结根均已验收，CPU报告无该错误，不接生产优化。
CPU探针按目标PID/主线程/闭合窗口核对，不能使用报告中的其他进程或加载/预热记录判断搜索热点。
composite事件混有上下文切换栈；本机官方导出文档规定`cpuCycles=1`是周期样本标记，0为其他栈，并非实际周期数。
CPU正式归因只取周期样本`stackDepth=0`叶模块，调度区间按`isSchedIn`转换重建；`kernelMode=NULL`仍未知，不按threadState独立判定on-CPU。
小探针1.1246ms仅证明链路；正式24.9161s窗口有178,221个根内周期叶样本但符号全未解析，不证明具体忙等、无丢失或可省比例。
诊断severity按本机枚举：2=warning、3=error、4=verbose，不把verbose当error。无跟踪双池复核已通过预声明门槛，不再管理员采样。
纯主机计时回读累计27.21%，与原生trace的阶段差异不能相减；候选阻塞事件仅在隔离工具实施，不修改全局device flags或生产。
`benchmark_host_wait.py`消费通过的无跟踪基线，36网络/图审计与576根精确、合计吞吐+3.06%，根门槛通过。
`benchmark_host_wait_games.py`只消费通过的根结果，保留原worker完整样本IPC，128局精确但cohort吞吐-2.37%、CPU累计-27.00%，筛选未过；
两者均要求新隔离目录/来源哈希/无其它Python/Nsight/WPR工作，无生产开关，见[等待实验](../docs/reviews/2026-10-09-host-wait.md)。
`test_stop_failed_native_capture.ps1`只用模拟对象验证15项身份/歧义反例，不枚举或终止真实进程。

## 未接入生产的原型

`central_inference.py` 是独立推理服务原型；`benchmark_central_inference.py` 做原批次精确性/吞吐对照；
`probe_merged_inference.py` 做离线反事实合批检查。只有实验工具/测试引用服务原型，`azero.py` 不依赖它。
合批未通过严格数值门槛，原批集中执行收益有限，当前不提供生产开关。
证据见 [集中 GPU 推理报告](../docs/reviews/2026-10-06-central-inference-pilot.md)。

## 独立算法研究

`measure_graph_transpositions.py` 只旁路测量不同公开路径的分层换位候选，并核对原始/插桩搜索的图、策略、RNG与网络输入输出；不合并节点、不写训练数据。
中间层直接键接入后，插桩同时观察快照/直接键入口，仅审计时重建完整信息态以保留规则/输入证据；
`measured_seconds`包含这些额外复制，不代表生产快照开销或剩余热点比例。
研究顺序见 [EXP-0003](../docs/experiments/0003-graph-search.md)：有效换位 → 固定模型棋力 → 独立训练收敛。
第22轮模型首测未发现满足完整状态约束的新增候选，Gate A未通过；当前没有图引擎或生产开关。
报告及冻结审计输入边界见 [首轮结果](../docs/reviews/2026-10-07-graph-search-feasibility.md)。

## 历史入口

`ab_runner.py`、`ab_experiments.json`、`ab_chance128.json` 和 `test_window.py` 保留旧版本实验配置。
这些配置可能引用旧规则模型/目录，不是当前长训推荐方案；使用前必须按当前规则重新核对。
历史结果见 [progress](../docs/progress.md)，日常使用先读 [status](../docs/status.md)。
