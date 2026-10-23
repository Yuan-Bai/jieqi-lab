# 自博弈 CPU / 网络边界短归因

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

日期：2026-10-08。状态：边界测量、精确性/开销对照和回归完成；原生栈/核trace因权限未获取。
独立测量，不是新加速、不更新模型、不接生产。

## 范围

接续 [1024局联合基线](2026-10-08-joint-selfplay-round.md)，先识别可观的集中成本，
不再主动追逐约1%的散点微优化。本次不重跑完整轮，不重载4M buffer，不训练/评估棋力/提交Git。

冻结iter28、18个可信局面，rules/legal=Cython、nodes=Python、8 worker，
800/3200模拟、leaf64、virtual loss3、chance完整/lazy、去重开、PCR关。
FP32张量，matmul TF32关、cuDNN TF32原样开；模型/批次/运算顺序不变。
主机沿用联合轮的i9-13900HX / RTX4060Laptop；本次再次记录torch2.6.0+cu118与GPU名称。

新增 [measure_selfplay_attribution.py](../../tools/measure_selfplay_attribution.py)。
通过AST包裹当前评估函数中的原始调用，不重写编码、前向、softmax或输出数值逻辑。
分别计时编码、H2D、前向调用、softmax、策略D2H、值D2H，外围evaluation计时包含其子阶段。
广范围版本另包合法着、局面更新、确定化、信息态/键、扩展、选择、机会准备等。
嵌套作用域同时保存inclusive/exclusive墙钟与主线程CPU时间，不重复相加。

CUDA events放在已有GPU阶段前后，搜索/整局结束后才读取，**不逐段同步**。
它们是当前stream的事件跨度，包含排队、其他进程/stream抢占和主机提交间隙，
不是纯核函数时间；跨worker之和不等于GPU独占工作量。
[NVIDIA事件计时说明](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
也明确提示其他stream工作可改变事件跨度。

## 原生采样限制

- 已安装Nsight Systems目录为2022.4.2，CLI启动遇到需要提升权限；当前Windows令牌不是管理员。
  [对应版本说明](https://docs.nvidia.com/nsight-systems/2022.4/pdf/UserGuide.pdf)要求Windows管理员运行CLI。
- WPR开始CPU采样失败：`0xc5585011 / Failed to enable the policy to profile system performance`。
  前后`wpr -status`均为未录制，未保留系统采样会话、修改采样策略或安装工具。
- 本机`torch.profiler.supported_activities()`只有CPU。所录CPU算子JSON没有CUDA核时间线，
  `with_stack=False`，也不是原生CPU栈；没有绕过权限或改PyTorch安装。
- `time.thread_time()`使用GetThreadTimes，标称分辨率100ns，但本机0.3s探测的20个正增量均为15.625ms。
  仅汇总多次搜索/整局CPU估计，不解读单次微秒级CPU开销。
- **原生栈上的驱动忙等占比、CUDA核执行/排队/抢占的完整分解仍未取得。**
  下载边界消耗CPU不等于已证明其中全部是驱动自旋。

## 复现

输出必须是`data/pilots`内不存在的新目录；不要同时运行训练或其他GPU任务。
pickle只接受本机可信fixture。默认使用此前冻结输入，不随生产新轮权重变化。

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
python tools/measure_selfplay_attribution.py --output-dir data/pilots/selfplay_attribution_20261008_iter28
```

广范围预声明见 protocol（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_protocol.md`），
原始 JSON（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28/result.json`）、
事件（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28/events.jsonl`）、
控制台（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28.console.log`）。
该测量版 工具副本（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28/measured_tool.py`）
在增加evaluation-only入口前归档，其哈希与主结果保护项相同。
副本用于源码核对，不是可直接在该归档路径运行的完整仓库副本。

广范围开销较高，保留所有结果，随后预声明收紧边界，只补固定根：

```powershell
python tools/measure_selfplay_attribution.py --output-dir data/pilots/selfplay_attribution_20261008_iter28_narrow --measurement-level evaluation --game-repeats 0
```

见 收紧协议（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow_protocol.md`）。
两个目录均受生产五个产物、冻结检查点、fixture、核心/原生及工具共18项哈希保护。
worker只返回审计摘要，不是生产的完整样本IPC；不能把队列吞吐当生产加速率。
没有生成可导入buffer的训练样本文件。

## 广范围结果

- 36组完整图/全部网络输入输出/策略/RNG/计数精确，共1129对网络请求。
- 4轮交替顺序，每模式144个计时根：144对策略/RNG/计数精确；先预热两模式，
  不保证每worker的每batch形状都已充分保温。
- 2轮交替顺序，每轮16局/模式，14普通+2强：共32对/64次整局执行。
  两组分别806和769步/原始样本，整局样本摘要、结果、步数及RNG精确。
- 两段另测CPU算子时间线所在8根cohort也与原路径策略/RNG/计数一致，不纳入吞吐对照。
- 固定根cohort插桩合计开销 **12.52%**，四组分别14.44%、15.80%、12.75%、7.22%。
- 完整局cohort开销 **19.60%**，两组分别17.96%、21.66%；原路径合计281.32s，插桩336.45s。
- 全部耗时864.28s / 14.40min，包括图/网络审计、预热、对照及CPU trace，不是自博弈加速率。

下表只表示32次**插桩完整局**的主线程边界，分母为worker整局墙钟合计1222.08s，
CPU分母为主线程合计1172.08s。GPU事件跨度不能与墙钟相加。

| 互斥阶段 | 插桩墙钟占比 | 主线程CPU估计占比 | CUDA事件跨度合计 |
|---|---:|---:|---:|
| 策略D2H调用 | 20.11% | 20.25% | 6.09s |
| 动作边扩展 | 12.26% | 12.32% | 不适用 |
| 前向调用 | 8.13% | 7.82% | 335.03s |
| 编码 | 6.48% | 6.38% | 不适用 |
| 合法着（扣内部命名子阶段） | 5.47% | 5.52% | 不适用 |
| 选子 | 4.42% | 4.48% | 不适用 |
| 局面更新 | 3.90% | 3.90% | 不适用 |
| H2D调用 | 3.77% | 3.93% | 43.25s |
| 确定化 | 2.74% | 2.74% | 不适用 |
| 机会准备（扣命名子阶段） | 2.31% | 2.30% | 不适用 |
| evaluation其他命名外工作 | 2.06% | 2.08% | 不适用 |

信息键约1.38%、叶签名1.35%、完整快照0.75%、observe_move0.65%；值D2H约0.20%、
softmax调用约0.13%。余量包括回传/导航/分配、未包逻辑和插桩自身，不能全部归为某个热点。
原先合法着inclusive约10%不能与本表exclusive约5.5%直接比较，两者定义不同。

广范围数值**不能直接当生产百分比，也不能按这些比例承诺提速**。尤其策略`.cpu()`会等待之前
提交的前向；其20%墙钟不等于PCIe传输成本，也不能与前向CUDA跨度相加。

## 收紧结果与判断

原始 JSON（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow/result.json`）、
事件（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow/events.jsonl`）、
控制台（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow.console.log`）。

收紧版完全不包规则/树方法，重新验证36组图/网络/策略/RNG/计数精确，1129对网络请求；
4组每模式36根，144对计时搜索精确，另测CPU trace也精确。
没有新增完整局，不能声称此模式已量化整局开销或补做了整局样本门槛。

cohort插桩合计开销 **1.35%**，四组分别3.52%、0.61%、-0.17%、1.50%；
相较广范围12.52%明显更低，优先使用本版本解读网络边界。
全套耗时229.66s / 3.83min，包含审计/预热/trace，不是单一根搜索耗时。
当前工具SHA256为`e74053463925578d48765a76557bd7852c84363505fd3e8de8957cf9f4084461`。

下表分母分别是72次普通根、72次强根的worker搜索墙钟37.35s / 149.31s，
预算构成为1:1根数量，不是生产7:1局数；不得投射为完整轮的精确百分比。

| 网络阶段 | 普通800模拟：墙钟占比 | 强3200模拟：墙钟占比 |
|---|---:|---:|
| evaluation inclusive（下列阶段合计） | **54.35%** | **49.42%** |
| 策略D2H调用（包含等前向完成） | **30.43%** | **27.47%** |
| 前向主机调用（不是GPU执行） | 10.00% | 8.38% |
| 编码 | 7.03% | 7.02% |
| H2D调用 | 3.99% | 3.93% |
| evaluation其他命名外工作 | 2.50% | 2.28% |
| 值D2H调用 | 0.25% | 0.22% |
| softmax主机调用 | 0.15% | 0.13% |

全部144次插桩根合计墙钟186.65s、主线程CPU177.63s；约50.72%的主线程CPU估计落在
evaluation边界，策略D2H约28.39%（普通31.74%、强27.55%）。没有负的汇总exclusive CPU阶段。
外部CPU归因仍存在GetThreadTimes量化误差，不能断言某阶段每调用消耗这些比例。

CUDA事件前向跨度合计66.38s，策略D2H跨度0.96s，H2D跨度6.95s。
策略D2H主机墙钟却有52.38s：显然不能把主机`.cpu()`边界全当PCIe拷贝成本；
但两类钟的差也不是精确GPU等待时长，不能按`52.38-0.96`计算可省时间。
必须保留排队、事件提交、抢占、同步、主机工作和插桩的口径差异。

另测CPU算子trace中，`aten::copy_`是这两个根最大的命名host算子：普通0.160865s/89次，
强0.261746s/178次。包含H2D/D2H及其他copy，不能只归策略下载；
profiler字段`self_cpu_seconds`是host事件自耗**墙钟**，不是GetThreadTimes线程CPU。
两个trace的总搜索墙钟分别0.575s/1.184s，不放进开销/吞吐对照：
普通根trace（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow/cpu_trace_15380_202610083_0.json`）、
强根trace（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow/cpu_trace_664_202610100_1.json`）。

## 已验证与未验证

**已验证（本机固定条件、有限样本）**：网络评估包含约一半搜索边界，很多主线程CPU消耗
发生在同步策略下载而非棋规/树逻辑。收紧测量开销约1.35%，图/网络/策略/RNG有限对照精确。
广范围规则exclusive数据只能用作粗定位，不能拿19.6%开销的路径声称生产的单函数比例。

**仍是推测**：下载CPU中有大量驱动忙等；阻塞等待可能减少CPU浪费；减少网络提交可能帮助吞吐。
尚无原生栈、核时间线或候选开/关对照，不能证明这些方案可带来10%收益或不改变结果。
CPU时间降低也不一定降低墙钟，MCTS原本就需要等待网络结果，GPU执行时间不会自动消失。

这不是新的旧/新联合加速率或全局最优证明，不能把此处CPU比例乘7.46核直接当整轮归因。
18局面/短整局的模型、局长、GPU竞争与生产4M常驻/梯度更新条件均有边界。

## 工作顺序

1. 不扩大棋规编译、不追加节点/字段/表容量微优化。保留当前精确加速，回到稳定训练与固定诊断；本次没有自动续训。
2. 继续性能研究时，先取得短管理员原生栈/CUDA trace，确认同步下载的CPU到底在哪里以及GPU排队/执行。
   仅对本工具的隔离子进程采样，记录开销；不改正式训练、系统调度策略或驱动设置。
3. 如果证据确认驱动忙等是可观成本，才独立验证“相同网络批次/原算术、不同同步读回等待方式”。
   pinned buffer/阻塞event等只是待测候选，不保证收益；保留原策略数组只读/值Python float及完整graph/network/RNG/sample门槛。
4. 如确认主机提交是瓶颈，可另试CUDA Graph提交图，不是MCTS换位图；它主要减少CPU提交，
   要求静态形状与长寿命输入输出地址，见 [PyTorch说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)。
   不合批/不填充改变真实batch，不启用AMP或算术融合；需显存、所有实际shape和整局数值/吞吐验收。
   当前前向host边界约8%--10%本身不能证明其全可省，**本轮未实现任何候选**。

## 验证

新增9项定向测试通过（6.05s），包括CPU网络逐位一致/float类型/只读策略、AST改形防护、
scope互斥与异常恢复、方法恢复、原生快路不回退、事件读取不额外同步、入口/预算防护、
evaluation-only不包装规则/树以及开销汇总。
广范围36图+144根+32整局对、收紧36图+144根对及四个CPU trace通过。
推荐rules/legal=Cython、nodes=Python全量 **530 passed / 141.33s**，
回归日志（本地实验产物，未随源码分发：`data/pilots/selfplay_attribution_20261008_iter28_narrow/regression.log`）使用正常系统临时目录，
未修改保存逻辑或放宽测试。此前组合定向25项通过；增加evaluation-only后9项专项再次通过。

两套结果各18项保护哈希在全量回归后重新核验一致；广范围工具项对应归档的实测副本，
其余保护项及收紧工具均为当前文件。原生产检查点/latest/状态/log/iter27 replay未变，
没有核心代码或原生重建、生产更新、正式训练、棋力评估或Git提交。
`git diff --check`没有错误（既有LF/CRLF警告）；五份更新文档本地链接全部可解析，
新工具/测试/报告无行尾空格。测量/回归进程已结束，GPU0%/0MiB，WPR未录制。

## 管理员采样准备

后续用户已同意继续原生采样。当前会话Windows令牌仍不是管理员，不能把执行授权
当作OS提权，也不建议关闭UAC/沙箱或长期以管理员运行整个应用。
OpenAI的`elevated`沙箱仍使用专门的低权限账号，见 [官方说明](https://learn.chatgpt.com/docs/windows/windows-sandbox)。

先由用户在开始菜单对PowerShell选择“以管理员身份运行”，接受Windows UAC，再执行：

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\native_profile_preflight.ps1"
```

[预检脚本](../../tools/native_profile_preflight.ps1)只核对管理员身份/安装工具签名，收集Nsight的
version/profile help/stats help/export help与WPR只读状态。只写新的
`data/pilots/native_profile_preflight_<timestamp>/`，不覆盖旧数据、开始录制、运行Python或修改策略。
用户已完成管理员预检：结果（本地实验产物，未随源码分发：`data/pilots/native_profile_preflight_20261008_202231_959/result.json`）
记录管理员身份为true，Nsight **2022.4.2.1-df9881f**；五个命令均返回0，WPR没有正在录制。
预检本身没有采样。当前Codex普通令牌仍不能直接执行需要管理员的CPU栈采样。

后续使用独立[NVTX准备/工作负载](../../tools/profile_native_selfplay.py)和
[管理员采样脚本](../../tools/capture_native_selfplay.ps1)，不改生产搜索。
先由普通令牌执行完整36图/网络输入输出对及四组普通/NVTX固定根配对；
管理员脚本只消费已成功的准备结果，核对保护哈希，再复跑相同144个固定根。
仅给原始网络评估的六个调用加NVTX范围，不加CUDA events、阶段同步或规则包装。
两个模式都先在八个worker预热，单一父进程NVTX标记只覆盖之后的采样cohort。

参数来自本机Windows帮助：`--sample=process-tree --cpuctxsw=process-tree --trace=cuda,nvtx`、
1000Hz/backtrace auto、`--capture-range=nvtx --capture-range-end=stop`，最长45秒，
`--kill=false --wait=all`，保留ETW并导出SQLite。非注册字符串触发需要
`NSYS_NVTX_PROFILER_REGISTER_ONLY=0`，见本机安装的官方2022.4 UserGuide。
Windows的kill是布尔值；不使用Linux的`--kill=none`或帮助未列出的`osrt`。
符号先不解析，不启动符号服务器，也不启动WPR/全系统GPU指标或改系统策略。

预声明协议（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_protocol.md`）区分标签配对开销与
跨运行的采样/参照时间差；后者混有机器状态差异，不能称纯profiler开销或生产加速率。
固定根普通/强数量1:1，仍不能替代7:1完整局/4M常驻训练轮。
首轮需检查老Nsight/新驱动的实际CUDA与八worker覆盖、栈/模块/丢失事件后才能归因；
有报告文件和成功退出码不等于采样完整，驱动模块驻留也不自动证明忙等。

首次普通权限准备：旧准备结果（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared/result.json`）。
36组完整图、网络输入输出、策略和RNG对齐（每模式1129次网络请求）；
四组144个普通/NVTX根对齐，两个预热都覆盖全部8 worker，23项保护哈希一致。
标签合计开销 **2.4326%**、平均配对2.4663%，四组分别5.7991%/2.6983%/-0.2029%/1.5705%；
普通cohort合计24.7033s，标签cohort25.3042s。全部准备226.97s，包含昂贵的另测完整图/网络审计。
这只量化标签扰动，不是Nsight录制开销或加速收益；首次新增工具5项+既有归因9项 **14 passed / 6.05s**。
PowerShell AST语法通过；普通令牌在管理员检查处拒绝、没有启动Nsight或写采样目录。
本次没有修改核心搜索/精度/模型或重建扩展；此前530项全量回归未在本次工具准备中重新运行。

首次无Nsight的采样入口试跑在跨进程记录比较处误报：内存走子为元组，JSON参照为列表，
两者直接比较必然不等。失败目录
`data/pilots/native_profile_20261008_iter28_capture_path_smoke/`原样保留，不作为通过结果。
现已在记录层按相同JSON序列化边界比较，并记录真实失败时的期望/实际策略摘要；
未放宽策略/RNG哈希和搜索计数。补充容器类型及真实字段变化/非有限值反例后，
定向 **16 passed / 6.08s**。新准备记录必须重新生成，旧记录因工具哈希变化已失效，
不用于管理员采样；此修正不涉及搜索算法。

修正后已完成有效准备v2（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared_v2/result.json`）：
重新跑36图/每模式1129次网络请求、144个普通/NVTX根对，全部精确。
普通cohort合计24.9201s、标签25.1723s，标签合计开销 **1.0121%**、平均配对1.0375%；
四组分别4.2246%/1.2033%/-0.4258%/-0.8520%，整个准备224.66s。
记录层修正不是性能优化；两次标签开销不同也提示短测机器/调度波动，不能据此声称新版更快。

采样入口普通权限重试（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_capture_path_smoke_v2/result.json`）
也已完成：重新启动8 worker并预热、144根全部与v2 JSON参照精确，23项保护哈希未变，
任务68.71s。它**没有启动Nsight或取得原生栈/CUDA trace**，结果kind的capture仅指采样工作负载路径。
这一步证明参数/哈希校验、父NVTX范围及跨进程参照可用；正式采样兼容性仍待管理员运行验证。

以下是当时v2的历史操作，启动器已修正，不再使用此参照；当前命令见后面的v3部分：

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\capture_native_selfplay.ps1" -ReferenceFile "<PROJECT_ROOT>\data\pilots\native_profile_20261008_iter28_prepared_v2\result.json"
```

新输出为 `data/pilots/native_profile_capture_<timestamp>/`；包含启动元数据、stdout/stderr、
`native.nsys-rep`、`native.sqlite`、保留ETW与工作负载最终审计结果。
父范围预计约25s，45s只是采样上限，目标进程仍完整结束四个cohort再落盘；
预热、报告导出不算在这个窗口内，总任务可能数分钟。
**真实管理员采样仍待用户执行，尚无新的原生CPU/CUDA结论。**

### 首次管理员启动失败

用户实际运行已返回失败：启动元数据（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_205312_713/capture.json`）、
Nsight stderr（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_205312_713/nsys.stderr.txt`）。
Nsight约2秒内报告`Unexpected exception thrown while launching the application`、
`std::range_error: bad conversion`，却退出0；stdout为空，没有workload目录或原生报告。
旧脚本只检查退出码，因此继续等了5分钟才报工作负载未完成。
已增加此类明确异常的提前拒绝，不再把退出0视为成功启动；没有重试、终止进程或放宽精确性门槛。
检查时没有Python/Nsight残留、GPU0%/0MiB、WPR未录制；生产五项和准备23项哈希在修复启动器前均一致。

该问题位于启动/工具层，不是训练结果失败，也没有进入已标记的自博弈热窗口。
[NVIDIA支持线程](https://forums.developer.nvidia.com/t/error-using-nsight-system-to-analyze-pycuda/256996)
记录了同类Windows `bad conversion`，建议先关闭跟踪/采样定位；它没有证明本机的具体原因。
当前环境唯一含非ASCII值的变量是`PSModulePath`，编码因素只是待验证假设。
本机普通令牌调用最小Nsight启动仍被Windows拒绝（requires elevation），不能在这里完成真实对照。

已准备[无采样启动诊断](../../tools/probe_native_launch.ps1)：在管理员终端执行三组相同参数，
`--trace=none --sample=none --cpuctxsw=none`，目标只打印`_native_launch_ok_`。
仅第二组从`ProcessStartInfo`的子进程环境移除`PSModulePath`，第三组重新按原环境启动，
父进程/用户/系统环境与精度变量均不修改；没有模型加载、CUDA导入或训练。
成功依据实际Python标记而非单独退出码。输出只写新`data/pilots/native_launch_probe_<timestamp>/`。
此前普通令牌试跑在Nsight启动前因权限拒绝，遗留空诊断目录原样保留，不作为启动对照结果。
正式启动器的提前报错修复改变了保护哈希，**v2参照暂不能用于重跑正式采样**；
先完成诊断、修复实际启动原因，再重新生成匹配启动器的准备记录，不直接更新旧哈希来掩盖变化。

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\probe_native_launch.ps1"
```

### 环境对照与启动器修正

用户已完成三组管理员诊断（本地实验产物，未随源码分发：`data/pilots/native_launch_probe_20261008_213033_191/result.json`）：

| 子进程环境 | Nsight转换异常 | 报告生成 | Python打印标记 |
|---|---|---|---|
| 保留PSModulePath | `bad conversion`，退出0 | 否 | 未捕获 |
| 仅移除PSModulePath | 无，退出0 | 是，132,753字节 | 未捕获 |
| 再次保留PSModulePath | 同一`bad conversion`，退出0 | 否 | 未捕获 |

三组均关闭全部跟踪/采样，父进程PSModulePath未变；唯一非ASCII环境值仍是PSModulePath。
这是本机该变量触发启动转换异常、仅移除可消除异常的直接对照，
**不是已获得CPU/CUDA采样，也未证明第二组Python目标正确执行**：
即使报告生成，目标stdout并没有可用的打印标记，不能忽略原验收条件。

修正[启动器](../../tools/capture_native_selfplay.ps1)，只在`ProcessStartInfo`的子进程副本中
移除PSModulePath；其他环境变量、父进程/系统环境、精度与算法均不改。
[公共辅助函数](../../tools/native_profile_process.ps1)由采样器与诊断器共享，
隐藏进程、同时异步排空stdout/stderr、将明确异常和非零退出码视为失败；加载辅助文件本身不启动进程。
没有选择性更新旧参照哈希，准备工具把新增辅助文件也纳入保护，并重新生成完整v3记录。

正式脚本先用相同子环境执行一次无跟踪/无采样的普通Python，它必须在新实验目录创建
`python_started.marker`才允许进入自博弈。标记是目标进程实际执行的证明，
不依赖Nsight是否转发stdout，不加载模型或导入CUDA；诊断工具也增加同样的文件标记。
输出多一个`startup_probe.nsys-rep`及日志，不把这个空事件报告当正式采样。
正式采样仍固定144根/45s上限/kill=false；如果Python已经退出却没有最终结果，立即报错，
不无条件等待5分钟，也不终止其他进程。

[辅助函数测试](../../tools/test_native_profile_process.ps1)已在普通令牌实际通过：
只删除子变量、其余环境值/父环境保留、隐藏窗口、异常退出0识别、正常stderr接受、
引号拒绝、普通Python创建文件标记/输出重定向。
测试输出（本地实验产物，未随源码分发：`data/pilots/native_process_helper_test_20261008_213647_786/python.stdout.txt`）
有实际标记；未运行Nsight/模型/GPU。四个PowerShell文件AST语法通过，
Python定向 **16 passed / 6.06s**。实际管理员启动标记与CUDA/CPU覆盖仍待重试检查。

### v3准备与重试命令

新准备v3（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared_v3/result.json`）已完整重跑，
没有直接更新v2哈希：36图/每模式1129次网络请求、144个普通/NVTX根对全部精确，
两种预热均覆盖8 worker，24项保护哈希一致（新增公共进程辅助文件）。
普通cohort合计25.1184s、标签25.5591s，标签合计开销 **1.7544%**、平均配对1.7580%；
四组分别4.2845%/1.7294%/1.6187%/-0.6005%，整个准备227.63s。
这些仍是标签开销，不是管理员Nsight开销或生产加速收益。

另用同一公共进程辅助函数，只移除子PSModulePath，在**普通令牌**直接启动Python，
完成过滤环境复跑（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_filtered_smoke_v3/result.json`）：
144根的策略/RNG/计数全部与v3参照一致，24项保护哈希未变，任务51.44s。
这验证了过滤环境下实际八worker工作负载/引用路径与搜索结果，不是Nsight采样；
没有取得CPU原生栈或CUDA核时间线，实际管理员启动自检与完整采样仍需下面的重试。
原生产五项、核心搜索/规则/原生扩展未变化，无模型更新、全量回归或Git提交。

以下是历史v3重试命令，**现在不要执行**：第二次采样已崩溃，后续修复改变了保护哈希，
当前入口与命令见文末v4章节。

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\capture_native_selfplay.ps1" -ReferenceFile "<PROJECT_ROOT>\data\pilots\native_profile_20261008_iter28_prepared_v3\result.json"
```

脚本先自检实际Python文件标记，再做新隔离目录的预热/短采样；
期间不要启动训练或其他Python/Nsight任务。失败输出仍保留，不自动安装/升级工具，
不修改Windows系统区域设置/永久环境变量，不杀进程，也不接入生产优化。

### 第二次采样崩溃与有界收尾

用户的v3管理员采样（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_214644_241/capture.json`）
实际Python标记通过，两个预热覆盖8 worker，四组共144根逐批精确比较完成；
事件日志（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_214644_241/workload/events.jsonl`）
最后停在51.1073s的cohort3。没有最终`workload/result.json`、正式报告或SQLite，
因此不是已完成的原生采样，也不能用它宣称驱动忙等/可回收时间。

Windows Application Error在**2026-10-08 21:47:49**记录同一Nsight进程
`PID=23276/0x5AEC`崩溃：`nsys.exe`、`ucrtbase.dll`、异常`0xc0000409`。
之后Windows Error Reporting记录BEX64；这证明工具崩溃，不证明某个训练/驱动内部缺陷。
旧脚本后来落盘的Nsight stderr（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_214644_241/nsys.stderr.txt`）
停在`Capture range ended... / Processing events...`，退出`-1073740791`，对应同一异常码。
最初9个Python进程和采样PowerShell仍滞留，GPU0%；**22:19复查时均已退出，未由本轮工具终止**。
无法从现有事件确定Python曾卡在NVTX退出还是pool收尾，不能补写成功结果。

独立故障备份（本地实验产物，未随源码分发：`data/pilots/native_capture_failure_20261008_221908_213/diagnosis.json`）
保存Windows事件、当时进程状态、修复前24项保护哈希，以及原始中间流副本
`native-incomplete.qdstrm`（1,286,336字节，原文件未改）。24项修复前哈希无变化。
同版本`QdstrmImporter.exe`未通过既定签名检查，**未执行导入器**，不以放宽检查强行恢复报告。

公共启动器原先先`WaitForExit()`再取两个`ReadToEndAsync().Result`，
可在退出的CLI子进程仍持有管道时无限等待，且只有收尾才写日志。
现在按收到的字节持续刷盘，并立即记录PID、开始时间和参数到`*.process.json`：

- Python启动自检最多60s、采样CLI最多300s，超时不自动杀目标或重试。
- CLI结束后的管道EOF只等3s；未关闭、超时或日志任务失败均拒绝作为成功采样。
- 管道未完成时延后资源释放，不在finally再次无限等待；异常退出码与显式启动错误仍拒绝。
- 输出已存在时在启动前拒绝覆盖；父/系统环境不变，仅移除子PSModulePath。
- 工作负载在热窗口外增加`capture_window_closed`及pool shutdown begin/end标记；
  最终结果仍必须经过pool结束和保护哈希核对，不提前假定成功。

普通Python最终回归结果（本地实验产物，未随源码分发：`data/pilots/native_process_helper_test_20261008_222924_625/result.json`）
**13项检查通过**：实际标记、实时日志、双管道大输出、环境/错误/引号/覆盖防护，
父进程退出0或23但子进程继承管道的有界失败，以及活进程超时但未被启动器终止。
另外三项直接执行脚本的局部参数构建，核对CUDA/CPU/combined隔离、kill=false/45s及无自动SQLite导出，
不执行管理员检查或采样入口。
故障用例只清理测试自己创建的普通Python休眠进程，不运行Nsight/模型/GPU。
四个PowerShell入口AST语法通过；Python定向 **16 passed / 6.06s**，未重跑全量回归。
这修复的是启动器挂起，**没有修复或确定Nsight自身崩溃的内部原因**。

下一次采样拆分为显式`-TraceMode cuda`或`cpu`；原`combined`仍保留但不直接重试。
先CUDA/NVTX且sample/cpuctxsw/backtrace=none，取得有效报告再单独测NVTX+CPU进程树。
各模式仍运行同一144根/800与3200预算、相同精度/信息边界与逐组精确检查。
SQLite不再与profile自动捆绑，只有工作负载和`.nsys-rep`通过后，才执行独立、有120s上限的export。
本机2022.4.2随附UserGuide的CLI Export Command Switch Options已核对参数。
拆分结果只能描述其启用子系统，不能拼成同时的CPU/CUDA分解。
新准备必须完整重新跑入v4，不能直接改v3哈希；尚无新的生产加速或有效原生trace结论。

### v4准备与CUDA-only验证命令

完整v4准备（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared_v4/result.json`）
已通过36图/每模式1129个网络请求、144个普通/NVTX根对；两种预热都覆盖8 worker，
24项保护哈希无变化，包含所有结束标记，总耗时210.98s。
普通cohort合计24.8848s、标签25.3596s，标签合计开销**1.9083%**、平均配对1.9923%；
四组7.4620%/-0.7787%/0.0730%/1.2130%。这是标签准备的短测，不是Nsight或生产加速率。

过滤环境v4复跑（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_filtered_smoke_v4/result.json`）
使用新公共启动器，仅移除子PSModulePath，直接普通Python完成144根精确比较，51.74s；
stdout/stderr正常实时落盘，退出0、未超时、管道完成，pool收尾标记和最终结果齐全。
这没有启动Nsight；其`kind=native_profile_capture`只是工作负载运行分支，不能当成有效原生trace。

修复后保护核对（本地实验产物，未随源码分发：`data/pilots/native_capture_failure_20261008_221908_213/after-repair.verification.json`）
还比较了v3/v4两模式共8组/288次搜索，策略/RNG/搜索计数全部相同。
与v3相比受保护文件仅三个工具源变化：Python标记、采样PowerShell、公共日志启动器；
其余源码、原生扩展、运行时、冻结输入和生产五项均未变。
没有优化器更新、Git提交、自动清理用户进程或工具升级；未重跑全量测试。

以下是已经执行并失败的v4 CUDA-only命令，**现在不要重复运行**；
后续处置见下一节，旧v2/v3命令同样不重跑。

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\capture_native_selfplay.ps1" `
  -ReferenceFile "<PROJECT_ROOT>\data\pilots\native_profile_20261008_iter28_prepared_v4\result.json" `
  -TraceMode cuda
```

脚本仍先验证实际Python标记，再做两次预热和45s上限热窗口；
启动60s、采样CLI300s、管道收尾3s、结果补等至多5分钟、独立导出120s分别有限等待，
失败时保留日志/进程身份，不自动结束目标或循环重试。期间不启动训练或其他Python/Nsight任务。
准备时没有管理员Windows令牌，实际CUDA-only由用户随后执行，结果见下一节。
只有CUDA-only报告有效才考虑CPU-only，不能从成功准备推断Nsight崩溃已修复或新加速有效。

### CUDA-only再次崩溃与停止旧工具重试

实际CUDA-only配置（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/capture.json`）
确认`sample/cpuctxsw/backtrace=none`，只跟踪CUDA/NVTX，未捆绑SQLite导出。
启动自检通过，四组144根逐批精确通过；工作负载事件（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/workload/events.jsonl`）
停在50.7600s的cohort3，没有`capture_window_closed`、pool结束或最终结果。
这缩小了阻塞位置，但没有目标线程栈，不能断言具体卡在NVTX range_pop或某个注入函数。

CLI进程记录（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/nsys.process.json`）
表明PID27956退出`-1073740791/0xc0000409`，没有超时；继承管道3s后仍未关闭，启动器正常失败返回。
实时stderr（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/nsys.stderr.txt`）
仍停在`Capture range ended... / Processing events...`。
故障快照（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/failure_diagnostic_20261008_224547_365.json`）
保存Windows 22:37:36的Application Error：同一`nsys.exe/ucrtbase.dll`、同一异常和模块偏移，
随后BEX64。22:45核对仍有根PID19572及8个worker，CLI已不存在，未由本轮工具终止。
没有正式报告/SQLite，不算有效采样，也没有训练结果或性能收益结论。

已备份原始中间流并核对保护哈希（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/raw_stream_backup.json`）：
1,413,445字节，原文件保留；24项v4哈希全部一致，生产模型/状态/日志未变。
已验证的缩小范围：**CPU采样不是本次复现的必要条件，独立SQLite导出根本未执行**。
未证实的原因包括旧分析器/当前环境兼容性及CUDA/NVTX多进程事件处理问题，
没有证据把本机异常对应到某条具体的官方已修复缺陷，不能保证升级必定解决。

[定向恢复工具](../../tools/stop_failed_native_capture.ps1)默认只读保存新的故障快照，
显式`-StopProcesses`且管理员权限才会结束核实过的冻结工作负载。
它核对失败记录、CLI退出、根程序/工作负载路径、解释器、创建时间、父子关系及预热记录的worker，
拒绝PID复用、未知子进程、活跃/歧义CLI、根已退出但worker身份不完整等情况。
不使用按名称批量终止或递归tree kill，不终止训练进程/系统会话；证据保存后按worker再根顺序处理。
[纯模拟测试](../../tools/test_stop_failed_native_capture.ps1)**15项通过**，不枚举或终止真实进程；
当前只执行了工具默认只读模式，未替用户执行清理。

以下清理命令已由用户完成，作为历史记录保留，**现在不要再次执行**：

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\stop_failed_native_capture.ps1" `
  -CaptureDirectory "<PROJECT_ROOT>\data\pilots\native_profile_capture_20261008_223633_532" `
  -StopProcesses
```

达到预声明的“隔离模式仍失败就停止并重估工具”条件，不继续旧2022.4.2的CPU-only/combined参数试错。
建议由用户安装独立新版Nsight Systems，不改变PyTorch、CUDA Toolkit、驱动或训练环境。
2026-10-08核对[NVIDIA官方Windows下载](https://developer.nvidia.com/nsight-systems/get-started)
当前为**2026.5.1**；[官方系统要求](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#system-requirements)
列出Windows11和CUDA10.0+支持，这只说明公开支持范围，不是本机修复保证。
没有自动下载安装、接受EULA、改变策略或上传故障转储。

新版安装后先核对签名/实际CLI路径/版本与Windows帮助，旧硬编码路径及SHA不能直接重用。
验证顺序为普通Python文件标记→单进程小CUDA/NVTX→少量spawn子进程→8-worker冻结根；
每步均限时并核对实际报告，失败不扩规模。修改采样入口后必须重新完整准备哈希参照，
不编辑旧v4哈希；在可靠原生trace取得之前，不扩大编译或接入CUDA Graph/读取方式原型。

### 新版兼容性验收与生命周期窗口

用户已运行定向清理；清理结果（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_223633_532/failure_diagnostic_20261008_225059_614.json`）
确认旧失败采样的9个Python PID均被核实并结束，没有清理错误。随后用户安装了独立新版，
未升级训练解释器、PyTorch、CUDA Toolkit或驱动；旧安装仍保留但不再重试。

本轮核验的实际CLI为`<HISTORICAL_NSYS>`，
版本`2026.5.1.161-265138896106v0`，NVIDIA签名有效，SHA256
`e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291`。
只读能力记录（本地实验产物，未随源码分发：`data/pilots/native_2026_capabilities_20261008_230640_037/result.json`）
保存version/profile/export/stats帮助，均退出0。CUDA-only的版本/启动/采集在普通Windows令牌通过，
无需管理员；CPU process-tree采样仍需管理员。帮助参数与实测成功只证明所测配置，不能推广到CPU模式。

[合成兼容性入口](../../tools/probe_native_runtime.ps1)不加载棋局/训练模型/优化器，
只使用小矩阵乘法和卷积、准确验证结果、FP32及原TF32开关，预热之后8次计算/两次回读。
两个spawn子进程通过独立Pipe握手保证各自预热和执行，不自动终止目标或重试。
[只读SQLite核验](../../tools/inspect_native_runtime.py)按PID、闭合窗口、计数、完整回读字节数与CUDA launch correlation检查，
保留所有诊断，不用忽略警告的方式通过门禁。它的16核/两回读预期仅针对当前固定torch/cuDNN/4060，非通用CUDA不变量。

| 探针 | 实测窗口内证据 | 结论 |
| --- | --- | --- |
| 实际Python启动（本地实验产物，未随源码分发：`data/pilots/native_2026_marker_20261008_231257_456/probe.json`） | 文件标记/结果/正式report/独立SQLite均存在 | 启动通过，不代表CUDA覆盖 |
| 单进程NVTX触发（本地实验产物，未随源码分发：`data/pilots/native_2026_single_20261008_231321_502/coverage.json`） | 16核、2回读、36 runtime API、10 NVTX；结果准确；2条可能不完整警告 | 已知操作覆盖通过，不宣称无丢失 |
| 双子进程NVTX触发（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_20261008_231832_984/coverage.json`） | 32核/4回读/72 API，但18 NVTX而非19；PID27532只有7个forward区间 | **覆盖失败**，退出0/结果正确不足以接纳 |
| 生命周期对照1（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_20261008_232008_701/coverage.json`） | 测量窗口内32核/4回读/72 API/19 NVTX，两个PID均完整 | 已知操作覆盖通过 |
| 生命周期对照2（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_20261008_232138_600/coverage.json`） | 同上，逐子进程与launch correlation均通过 | 重复通过，不宣称普遍无丢失 |
| 最终脚本对照（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_cuda_20261008_232410_268/coverage.json`） | 同上，修改后的脚本新鲜运行，源哈希及二进制固定 | 可作为8-worker采集的兼容性门禁 |

生命周期对照保留15条severity2诊断，包括父/其他未执行CUDA进程的无CUDA/NVTX事件提示；
它们不能被一概视为本轮worker丢核，也不能删除。**父触发到子记录之间存在边界问题是推测**，
目前可靠事实是触发方式少了一个子区间，而全生命周期方式连续三次覆盖已知操作。
本机GPU不支持新版硬件跟踪，诊断明确回退到软件CUDA trace；这不是训练精度变化。
没有发生旧异常，但仍不能把小探针成功写成所有规模/所有模式的崩溃已修复。

因此采样入口固定为新版并改为**全生命周期采集，分析仅取预热后闭合的`jieqi_native_window`**。
原始trace包含加载/预热/收尾，测量指标必须排除它们，不能直接使用全报告stats总计；
CPU/CUDA隔离仍保留，CPU模式仍检查管理员令牌。采集180s上限、CLI300s/管道3s/独立导出120s，
kill=false、不修改系统策略/永久环境、不自动清理。需要显式`-CompatibilityFile`，核对二进制与证据文件哈希。
修改准备所保护的采集脚本后，完整准备重跑到新的v5，不覆盖v4或更改其哈希。

v5完整准备与8-worker CUDA-only随后完成，结果见下节。CPU栈/忙等归因尚未取得，
不接入生产优化。最终辅助脚本13项检查、两个修改后入口AST通过、Python定向26 passed / 2.13s；
包含只读分析的漏子进程/区间/阶段、CUDA PID/correlation及诊断保留反例，未重跑全量回归。

### v5准备与首份8-worker CUDA报告

完整v5准备（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared_v5/result.json`）
36组图/网络流内部精确、每模式1129个网络请求、四组144个普通/NVTX根对均通过，
两种预热覆盖8 worker。跨v4核对（本地实验产物，未随源码分发：`data/pilots/native_profile_20261008_iter28_prepared_v5/verification_v4.json`）
确认两版共8组/288次搜索的策略/RNG/计数全部相同，图验收元数据与请求计数相同；
没有保存旧版完整图供跨版本逐对象比较，不能扩大成这种保证。
24项受保护文件与v4相比只有`capture_native_selfplay.ps1`变化，其余核心/扩展/输入/生产产物未变。

准备耗时649.57s；普通cohort合计53.9070s，NVTX53.5145s，合计差异-0.7280%，
四对-1.6946%/-2.8496%/+0.1672%/+1.4471%。负值是运行波动，不是标签加速。
无跟踪参照也明显慢于v4（24.8848/25.3596s）；运行状态差异原因未确认，
**本轮不以跨v4耗时比较宣称训练提速/退化，也不把下面差异当纯采样开销。**

实际8-worker配置（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_233729_457/capture.json`）
普通Windows令牌、CUDA/NVTX-only、全生命周期、无CPU sample/cpuctxsw/backtrace，
实际Python标记/CLI结束/管道完成/四组精确/窗口闭合/pool正常退出/正式报告/独立SQLite均通过。
工作负载结果（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_233729_457/workload/result.json`）
102.57s；四个测量cohort14.1917/14.2130/14.3349/14.2951s，合计57.0347s，
相对本轮NVTX参照+6.5780%，其中还包含不同运行状态，非隔离测得的工具开销。
报告78,903,447字节，SQLite201,482,240字节，旧崩溃未重现，无人工/自动终止进程。

最终只读覆盖结果（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_233729_457/coverage_v2.json`）
与采后保护核对（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_233729_457/post_capture_verification.json`）确认：

- 预热后闭合窗口57.0469s；8个目标PID各自普通/强根数均与结果144条一致（72/72）。
- 4510个evaluation与六个阶段各4510条，目标主线程身份/区间结束均符合；所有worker均有CUDA核/拷贝。
- 713,128个已记录计算核与kernel-launch API数量一致，13,736个拷贝与MemcpyAsync API数量一致；
  每个GPU事件均能按**全局进程身份+correlationId**找到窗口内runtime API，未匹配0，未见跨窗口GPU事件。
- 全生命周期有36条severity2诊断，均非8个目标worker；还有一条severity3
  `AnalysisService: failed to start event source Perf: AccessDeniedException`，原样保留。
  未验证该额外Perf源的具体用途，不将它掩盖，也不把CUDA-only报告当成CPU原生栈成功。
- 两次只读分析的SQLite SHA256相同；24项源码/扩展/运行时/冻结与生产文件采后均未变。

这些证明**已知根/阶段以及已记录CUDA API和GPU事件的覆盖与对应**；不能独立证明
API/GPU事件从未成对丢失、所有未标记事件完整或任意训练局面结果不变。
`coverage.json`是最初分析，保留不覆盖；`coverage_v2.json`增加CUDA对应和warning/error分类。

本轮窗口内初步聚合（跨8 worker累加的阶段/API时间，**不是父墙钟分解**）：

| 指标 | 实测 | 可解释的范围 |
| --- | ---: | --- |
| root区间累计 | 405.5504s | 分母仅这144个固定根的worker区间 |
| evaluation累计 | 166.8601s / 41.1441% | 包含编码/主机调用/传输/等待，非纯GPU计算 |
| encode / forward / H2D累计 | 42.3048 / 93.3669 / 17.8920s | 主机区间时间；不能直接判定每段为CPU工作或GPU等待 |
| policy D2H累计 | 3.3474s / 0.8254% | 跟踪下占比，明显不同于早前27%--30%边界计时 |
| cudaLaunchKernel API累计 | 31.0066s / 690,588次 | 不含另22,540次LaunchKernelExC，不能直接推算CUDA Graph收益 |
| H2D GPU拷贝 | 4510次 / 21.7537GB / 13.5913s累计 | GPU活动跨度累加，非独立PCIe带宽基准 |
| D2H GPU拷贝 | 9020次 / 2.5907GB / 0.2077s累计 | 不包含此前GPU任务的主机同步等待 |
| 已记录kernel+copy区间并集 | 24.5540s / 窗口43.0419% | 不含memset/未记录活动，不是GPU利用率或可回收56.96%时间 |

阶段分布与旧边界计时明显不同，且本轮无跟踪参照也变慢，**原因尚未拆分**。
可能来自当前运行状态和/或CUDA软件跟踪对排队/同步位置的扰动；这只是候选解释。
下一步先做同版本CPU-only小spawn探针，再以同v5参照采CPU-only冻结根；
检查真实栈/模块和NVTX阶段分布后，再决定是否重测边界计时、研究CUDA Graph或读取方式。
仍不确认驱动忙等，不根据这份窄根报告改生产worker、精度、batch或搜索预算。

本轮可复现命令（新目录才能运行，不与其他Python/Nsight/WPR任务并行）：

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
python tools/profile_native_selfplay.py --output-dir data/pilots/native_profile_20261008_iter28_prepared_v5_REPRO

powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\capture_native_selfplay.ps1" `
  -ReferenceFile "<PROJECT_ROOT>\data\pilots\native_profile_20261008_iter28_prepared_v5_REPRO\result.json" `
  -CompatibilityFile "<PROJECT_ROOT>\data\pilots\native_2026_spawn_lifecycle_cuda_20261008_232410_268\coverage.json" `
  -TraceMode cuda
```

然后使用采集打印的新目录运行`python -m tools.inspect_native_selfplay --capture-dir <新采集目录>`。
本轮已完成，无需现在重复以上CUDA采集。下一步只在原管理员PowerShell运行小CPU探针：

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\probe_native_runtime.ps1" `
  -Stage spawn -CaptureScope lifecycle -TraceMode cpu
```

小CPU报告须检查子PID/采样表/栈/模块/丢失诊断后才扩大到8 worker。
CPU模式未在本轮执行，不声称管理员下已通过，也不直接运行combined或生产续训。

### CPU小探针验收（2026-10-09）

用户在原管理员PowerShell完成2026-10-08 23:58开始的
CPU-only双spawn探针（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_cpu_20261008_235809_505/probe.json`）。
同一签名/哈希固定的2026.5.1，全生命周期NVTX+process-tree sample/cpuctxsw、auto backtrace，
无CUDA trace/符号服务器；它执行的仍是原FP32合成CUDA操作，不加载训练模型。
CLI、管道、正式report、独立SQLite、两个子进程准确回读和正常退出均通过。
报告12,436,592字节，SQLite380,899,328字节；不是正式自博弈耗时基准。

只读覆盖证据（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_cpu_20261008_235809_505/coverage.json`）
按父闭合窗口、工作负载记录的子PID与主线程核对：

| 指标 | 子PID24772 / 主TID16168 | 子PID24236 / 主TID27032 |
| --- | ---: | ---: |
| 测量forward / readback标签 | 8 / 1 | 8 / 1 |
| 窗口内composite栈事件 | 22 | 14 |
| 有frame / module记录的事件 | 22 / 22 | 14 / 14 |
| frame行数 / 未解析行数 | 954 / 954 | 460 / 460 |
| `kernelMode=NULL`行数 | 682 | 115 |
| 窗口内调度事件 | 7 | 20 |
| `cpuCycles>0`事件 | 0 | 1 |

父测量窗口**1,124,561ns（1.124561ms）**，19个已知测量NVTX标签完整；
全生命周期21个标签含两个预热回读。CPU表/栈/module/IP/调度都实际存在，
模块包含Python、torch、CUDA/cuBLAS/cuDNN、NVIDIA驱动和系统模块。
9条severity2 warning完整保留，含可能不完整的NVTX提示；severity3 error为0，
此前`Perf: AccessDeniedException`未出现。**通过CPU采集兼容性门禁，不宣称无丢失或性能结论。**

限制与解释边界：

- 约1.12ms窗口、36个composite事件不足以估计搜索热点；采样与上下文切换回溯可能混合，不能用栈事件/帧数算CPU时间占比。
- `kernelMode=NULL`是未知，不能当user或kernel；未解析IP/module不是已解析函数。
  几乎全零的`cpuCycles`不能当CPU未运行，也不能据此计算可靠周期占比。
- 驱动/库栈驻留不证明忙等或可优化时间。CPU-only没有CUDA活动表，但目标仍使用CUDA计算；不能把两份隔离报告拼成同步CPU/GPU分解。
- 原始SQLite还含非目标进程记录；只分析目标PID/主线程/闭合窗口，排除加载/预热/收尾，原始数据保留本地、不上传。
- 没看到CPU丢数据诊断，不等于证明无丢数据；可能不完整提示仍在，`lossless_trace_verified=false`。

采后保护核对（本地实验产物，未随源码分发：`data/pilots/native_2026_spawn_lifecycle_cpu_20261008_235809_505/post_probe_verification.json`）
确认v5准备24项、探针4项来源、被分析的SQLite/工作负载哈希均未变；Nsight签名有效且二进制不变。
核对时没有Python/Nsight残留，WPR未录制。只读核验器新增CPU分支和反例，诊断按本机
枚举`2=Warning, 3=Error, 4=Verbose`分类；定向 **33 passed / 2.22s**，未重跑全量回归。
采集入口/冻结负载/生产代码与产物未改，未训练、重建或提交。

下一步仅采**8-worker CPU-only冻结根**，沿用未变的v5准备与此CPU门禁。
仍需原管理员PowerShell、无并行训练/其他Python/Nsight/WPR录制；不重复小探针或CUDA采集：

```powershell
powershell.exe -NoProfile -File "<PROJECT_ROOT>\tools\capture_native_selfplay.ps1" `
  -ReferenceFile "<PROJECT_ROOT>\data\pilots\native_profile_20261008_iter28_prepared_v5\result.json" `
  -CompatibilityFile "<PROJECT_ROOT>\data\pilots\native_2026_spawn_lifecycle_cpu_20261008_235809_505\coverage.json" `
  -TraceMode cpu
```

全生命周期采集上限180s、CLI300s、独立导出120s、kill=false；命令打印新隔离目录。
较长窗口的开销/表大小尚未知，若超限或结果不完整，保留日志/PID，不盲目重试或终止目标。
完成后先验收144根/8个主线程/阶段与策略/RNG/计数、窗口覆盖及诊断，再分析栈/调度。
当前固定根只读分析工具尚仅支持CUDA；CPU正式报告要按实际表结构扩展核验，不能套用CUDA计数门禁。
此CPU-only冻结根尚未执行，热点、忙等和下一项生产优化仍未确定。

### 首份8-worker CPU报告（2026-10-09）

用户在原管理员PowerShell完成上述命令，实际目录为
native_profile_capture_20261009_001250_716（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261009_001250_716/capture.json`）。
新版二进制、v5冻结准备和CPU双spawn门禁不变；NVTX+CPU process-tree采样/调度，1000Hz，
不启用CUDA trace或符号解析。实际Python标记、CLI/管道、pool正常结束、正式report和独立SQLite通过，
未触及180s采集上限。report116,844,994字节，SQLite2,428,760,064字节，ETW保留本地。

工作负载（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261009_001250_716/workload/result.json`）
耗时52.9445s；四个热cohort6.1122/6.0162/6.4321/6.3521s，合计24.9126s。
相对v5的NVTX参照53.5145s为-53.44699%，明显不同于先前CUDA-only采集57.0347s。
**没有新优化，这是分开的运行/不同跟踪子系统，不能宣称53%加速或负采样开销。**
机器状态与工具扰动的贡献未分离，需无跟踪复核；不能根据三份报告耗时定量推算可省时间。

最终只读覆盖（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261009_001250_716/coverage_v2.json`）
与独立采后核对（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261009_001250_716/post_capture_verification.json`）确认：

- 闭合窗口24.916102269s，144个根（普通/强各72）、8个记录中的主线程，4510个evaluation及六个子阶段各4510条。
- 144条策略/移动/RNG/计数与v5相应repeat/index/strong/seed逐项一致；这不是重新取得完整搜索图或任意局面的证明。
- 八个主线程均有周期叶样本、module/IP、完整窗口调度区间；根/阶段无交叉或未闭合，调度没有连续sched-in异常或未覆盖的根区间。
- 27条warning完整保留，其中8条来自目标worker，内容为NVTX可能不完整；已知根/阶段计数完整不等于所有事件无丢失。0条Error，无旧Perf权限错误。
- 24项受保护源码/原生扩展/运行时/冻结与生产文件，以及准备、兼容性、分析输入哈希均未变；不修改原`capture.json`的未分析标记。

本机安装版官方`AnalysisGuide/index.html`的**cpuCycles attribution**和**SCHED_EVENTS gotchas**
明确了两个口径，文档路径/哈希保存在采后核对中：
`cpuCycles=1`是周期CPU采样标记，0是其他原因（如调度转换）的栈，不是实际周期数或CPU闲置。
因此本轮只统计`cpuCycles=1, stackDepth=0`的**叶模块采样占比**，不累计所有帧或所有composite事件。
这也补充前一节小探针的解释：它仅0/1个周期样本，36个composite栈事件不能用于热点估计。
调度时间由实际`isSchedIn`转换及窗口边界重建，不能把`threadState=Running`直接当on-CPU。
`threadBlock`只在sched-out时解释，和线程状态一起报告。

全生命周期有2,198,013个composite事件、45,647,092个frame行、4,274,191个调度记录，
含预热和非目标进程，不能直接用于搜索归因。八个目标主线程在窗口内有178,277个周期样本，
其中178,221个在已标记根内，56个在根外；周期叶均有module/IP但**全部符号未解析**。

| 根内周期叶模块 | 样本 | 占178,221样本 | 可解释范围 |
| --- | ---: | ---: | --- |
| `python311.dll` | 78,947 | 44.30% | Python解释器执行，不能对应到具体规则/搜索函数 |
| `nvcuda64.dll` | 66,760 | 37.46% | 驱动模块执行，不直接证明具体轮询循环 |
| `ntoskrnl.exe` | 10,890 | 6.11% | 系统内核，不能全部归因到CUDA等待 |
| `nvlddmkm.sys` | 4,276 | 2.40% | NVIDIA内核驱动模块 |
| `_jieqi_rules...pyd` | 3,005 | 1.69% | 已编译规则内核的叶样本，不含Python包装/调用方 |
| NumPy `_multiarray_umath...pyd` | 2,833 | 1.59% | NumPy模块，不是纯编码或数据复制计时 |
| `_jieqi_legal...pyd` | 1,484 | 0.83% | 已编译合法着内核，不等于完整合法着调用的成本 |

下表是**同一CPU-only报告内跨8 worker累计的根/阶段墙钟**，不是24.9161s父墙钟分解：

| 阶段 | 累计墙钟 | 占根累计182.3800s | 根内周期样本占比 | 观测scheduled-on / 阶段墙钟 |
| --- | ---: | ---: | ---: | ---: |
| search_other（evaluation之外） | 80.8829s | 44.35% | 43.96% | 99.7670% |
| encode | 11.2512s | 6.17% | 6.07% | 99.7430% |
| H2D主机边界 | 6.4136s | 3.52% | 3.51% | 99.7278% |
| forward主机边界 | 14.6812s | 8.05% | 7.98% | 99.6775% |
| policy D2H主机边界 | 64.9902s | 35.63% | 36.34% | 99.8362% |
| value D2H主机边界 | 0.2817s | 0.15% | 0.14% | 99.5720% |
| softmax主机边界 | 0.0928s | 0.05% | 0.05% | 99.8010% |
| evaluation_other（六阶段之外） | 3.7864s | 2.08% | 1.95% | 99.7021% |

evaluation总计101.4971s/55.65%，是六阶段与evaluation_other之和，不能再次加到表中。
根内观测scheduled-on181.9787s/99.7800%，off0.401319s，其中0.398502s记录为
`Running/NonBlocked`，0.002720s为`Interruptible/NonBlocked`，0.000097s为`Interruptible/PushLock`。
这是转换重建的调度占据时间，不是有效逻辑计算或纯PythonCPU时间；中断/内核等可发生在被调度区间。
工作负载计时器累计process CPU172.90625s与scheduled-on相差约9.07s，两者定义和计时边界不同，
差异未完整解释，不拿其中一个校正另一个或声称精确CPU时间分解。

**已验证事实与推断分开：**

- policy D2H内64,767个周期叶样本，有62,455个（96.43%）落在`nvcuda64.dll`；scheduled-off仅0.106454s。
  因此“这一边界几乎全是worker睡眠/未被调度”不符合本次记录，存在大量被调度的驱动执行。
- 具体是否驱动忙等、等待哪个GPU任务、能否通过改变主机等待方式加速，**仍是待验证推断**。
  未解析符号、没有同次CUDA活动，不能指定驱动函数或宣称35.63%墙钟都能消除。
  policy回读还承担此前异步提交任务的同步，不能当纯PCIe拷贝成本。
- search_other仍有78,348个周期样本，其中68,975个落Python解释器；存在Python层工作，但当前报告
  不知道是局面更新、信息集键、节点组织、选择/回传或其它哪个函数主导。
- 规则与合法着编译模块合计约2.52%周期叶样本，单独再磨这两个C内核的优先级降低。
  Python包装和数据结构成本不计入它们；不能用该比例否定此前已验证的编译收益，或当完整合法着成本/生产加速上限。
- CUDA-only报告的policy D2H为0.83%根累计墙钟，本次35.63%；不能把两份不同运行的阶段或GPU时间相减。
  只证明当前分阶段归因对运行/跟踪状态敏感，不确认其具体原因。

最终审计（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261009_001250_716/final_analysis_verification.json`）
保存核验器/测试来源，定向 **42 passed / 2.18s**。CPU只读分析重复完全相同；
同一核验器对旧CUDA报告的实际回归（本地实验产物，未随源码分发：`data/pilots/native_profile_capture_20261008_233729_457/coverage_regression_cpu_extension.json`）
除新增来源哈希字段外与原覆盖报告完全相同，保留原Perf错误。未重跑全量，原SQLite均只读、不改报告或生产。
核对时无Python/Nsight残留、WPR未录制；没有新采集、优化、训练、重建或Git提交。

下一步不需要继续管理员/combined采样：

1. 先以现有固定模型/18局面/800+3200预算，在无Nsight下交替复核普通/NVTX及轻量网络边界计时，
   报告每cohort和开销，分离明显的运行状态漂移；不把不同日期/令牌/trace模式当优化A/B。
2. 基线稳定后，独立试验**主机同步等待方式**；保持每次网络batch/输入/运算顺序、FP32/TF32开关、
   机会支持和搜索预算。先网络输出/完整图/策略/RNG精确，再完整局配对吞吐，任何门槛失败不接生产。
   这是候选方向，尚无原型或收益证明，不预承诺10%。
3. 若等待方式无净收益，再用低扰动Python函数定位选择较大的逻辑段；不根据模块采样直接重写
   局面更新或追加散点Cython编译。当前训练仍可使用已验收版本，不调整训练参数。

### 无跟踪稳定性复核（2026-10-09）

继续上一节第1步，先保存执行前协议（本地实验产物，未随源码分发：`data/pilots/sync_baseline_20261009_iter28_protocol.md`），
新增隔离入口[measure_sync_baseline.py](../../tools/measure_sync_baseline.py)，不编辑受v5保护的采样/搜索源码。
同一iter28模型/18局面/8 worker/800+3200/leaf64/FP32及原TF32开关，三个模式为普通、原NVTX标签、
缓存AST的纯主机wall/thread计时。无Nsight/WPR/PyTorch profiler、CUDA events或新增阶段同步；
保留`run_case`原有搜索边界同步，不包装规则/树逻辑。

新计时器36组完整图/网络输入输出/策略/RNG精确，与v5图验收元数据/1129网络请求相同。
原NVTX完整图证据沿用源码哈希未变的v5，本次没有重做NVTX完整图对照。
两个独立spawn池分别预热三个模式，各覆盖8个PID，然后各做六种模式排列，使每模式在每个位置各出现两次；
第二池逆序排列。每cohort36根，**主测1296次搜索**，另有72次图审计、216次预热，均不是完整自博弈局。

结果（本地实验产物，未随源码分发：`data/pilots/sync_baseline_20261009_iter28/result.json`）427.9540s完成，
独立核对（本地实验产物，未随源码分发：`data/pilots/sync_baseline_20261009_iter28/verification.json`）确认所有跨模式/池的策略、RNG、计数一致，
前四repeat共864条根记录还逐项匹配v5；27项保护文件（原24项+新工具/测试/参照JSON）未变。

| 指标 | 新池0 | 新池1 | 预声明门槛/结论 |
| --- | ---: | ---: | --- |
| 普通cohort六次墙钟/s | 6.3634 / 6.2449 / 6.1920 / 6.3268 / 6.3270 / 6.3300 | 6.2567 / 6.3164 / 6.2303 / 6.3202 / 6.2853 / 6.2829 | 原始每组均保存 |
| 普通cohort中位数/s | 6.3269 | 6.2841 | 跨池-0.6761%，绝对差<=10%，通过 |
| 普通cohort CV | 1.0300% | 0.5499% | 每池<=5%，通过 |
| 后三组/前三组平均漂移 | +0.9768% | +0.4522% | 每池绝对值<=10%，通过 |
| 普通六组合计/s | 37.7840 | 37.6919 | 固定根parent cohort墙钟 |
| NVTX六组合计/s | 37.8154 | 37.8817 | 两池合计相对普通+0.2929% |
| 主机计时六组合计/s | 37.7930 | 38.2398 | 两池合计相对普通+0.7378% |
| evaluation / 累计worker根墙钟 | 49.6845% | 49.1643% | 含等待，不是纯GPU时间 |
| policy回读 / 累计worker根墙钟 | 27.5360% | 26.8765% | 每池>=20%、相差0.6596个百分点<=5，通过 |
| policy回读threadCPU / wall | 97.6536% | 96.2213% | 每池>=80%，通过；不是纯CPU/GPU分解 |

NVTX/主机计时的合计绝对开销均<=3%，通过；两池主机计时累计根556.5467s，
policy回读151.4089s（27.2051%），threadCPU读数146.78125s（回读wall的96.9436%）。
各池六阶段/evaluation分别6769次，全部计数匹配，短value/softmax占比很小。
Windows本次threadCPU累计读数呈15.625ms粒度特征，短阶段甚至有CPU/wall>1；
这些短阶段不能当精密纯CPU计时，长回读边界仍应结合上一节周期样本/调度证据解读。

五次GPU温度/功耗/时钟快照仅在审计前、两个池前后取得：温度35/44/58/58/60度，
后三次SM时钟2640MHz；快照不构成连续热态历史或功耗控制，也不能解释旧v5运行的慢因。
原来的13s左右cohort慢基线不能复现为当前常态，**本次没有优化，不能把现在6.3s写成约53%加速**。
本次无跟踪回读约27%、CPU-only trace约36%、CUDA-only trace约1%的阶段分布差异仍不能相减；
已验证的是当前两个新池的基线稳定与较大活跃回读边界复现，不是长期/整轮稳定性或具体驱动忙等。

本次预声明的稳定性、开销和候选门槛全部通过。**可以进入独立等待方式A/B，尚未实施或接入生产**。
候选只在原policy `.cpu()`之前用当前流记录可复用的
`torch.cuda.Event(enable_timing=False, blocking=True)`，执行`event.synchronize()`后仍调用原`.cpu()`；
不合批、不缓存输出、不改softmax/value转换、精度、机会支持或预算。
[PyTorch v2.6.0绑定](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/cuda/Event.cpp)
确认`blocking`映射到`cudaEventBlockingSync`；[CUDA11.8官方接口](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__EVENT.html)
规定这种事件的主机同步采用阻塞等待。`event.wait()`是GPU流等待，不是本候选的CPU等待方式，见
[同版本CUDAEvent实现](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CUDAEvent.h)。
本机已安装Python包装层也核对过，不升级任何组件或修改全局device scheduling flags。
这只是**API可行性**：减少CPU活动不保证墙钟变快，新增事件和唤醒可能抵消收益，调度变化也必须实际检验数值。
下一实验先完整网络/图/策略/RNG精确及固定根筛选，再完整局配对吞吐；未过任何门槛就不接生产。

可复现命令（仅当v5保护文件仍匹配、无其它Python/Nsight/WPR工作；输出目录须新建）：

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
<HISTORICAL_PYTHON> tools/measure_sync_baseline.py `
  --output-dir data/pilots/sync_baseline_20261009_iter28_REPRO `
  --reference-file data/pilots/native_profile_20261008_iter28_prepared_v5/result.json
```

用户续训后生产哈希变化，应建立新的冻结参照，不能编辑旧v5哈希绕过门禁。本轮无需重复以上基准。
新增9项反例/计时器测试，连同复用模块定向 **60 passed / 7.03s**，未重跑全量。
没有生产修改/训练/重建/Git提交；两个池正常退出，核对时无Python/Nsight残留、WPR未录制。

后续隔离等待实验已完成，见[2026-10-09等待报告](2026-10-09-host-wait.md)：根吞吐+3.06%，
128完整局精确但cohort吞吐-2.37%、CPU累计-27.00%，吞吐门槛未过，不接生产。
短cohort低在途比例不能证明1024局收益，不从本报告的35.63%或27.21%阶段占比推算加速。
