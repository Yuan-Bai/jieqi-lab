# 当前 Python 函数栈热点（2026-10-09）

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

## 结论

完成当前保留加速全部开启后的函数级采样，不修改生产搜索或训练配置。
576 次正式搜索逐项匹配冻结参照；50/100 Hz 均覆盖 8 个 worker，分别留下 5,377/11,168 个正式搜索栈样本。
两档前十名函数集合相同；热点集中在节点扩展、编码、合法着调用、PUCT 选择，而非长将/长捉的小循环。
这是**持有 GIL 的搜索栈条件分布**，不是纯 Python CPU 占比、墙钟分解或可节省时间。

下一步优先筛选**编码/重复历史准备的小范围精确原型**，PUCT 选择作为第二候选。
不因初始化/合法着调用的排名高就重启已失败的编译方案，不继续扩大阻塞等待实验。
本次没有新加速率，也没有证据支持某个剩余方案能提高整轮吞吐 10% 以上；不能据此证明已达到最优。

后续同日完成[精确编码编译筛选](2026-10-09-exact-encoder.md)：局部省时24.27%，576根精确，
但双池根吞吐仅+0.411%，未过预声明门槛，不接生产；历史准备未改，下一候选PUCT选择。
后续同日[PUCT选择编译筛选](2026-10-09-exact-selector.md)也完成：201408次选择、36对图/网络、576根精确，
局部省时15.05%，双池根吞吐-0.401%，未过门槛，停止。下文优先级是采样时的判断，
不能当作当前尚未验证的任务清单；两个直接编译候选都不接生产，重复历史准备仍未改。
随后[重复历史准备测量/身份复用原型](2026-10-09-repetition-preparation.md)也完成：身份复用88.65%，
局部省69.03%，64479条目/36对图网络/576根精确，但根吞吐-0.967%，未过门槛。
生产历史准备仍未改，三个CPU候选均停止，不凭下面条件GIL排名继续扩大本轮散点编译。
本报告以下排名/优先级保留为该后续实验之前的测量记录。

## 环境与输入

- Python `<HISTORICAL_PYTHON>`，3.11.5；PyTorch 2.6.0+cu118，CUDA 11.8。
- i9-13900HX、RTX 4060 Laptop 8GB、32GB RAM；同前阶段环境，不调整功耗、前后台或进程优先级。
- 后端 rules/legal=Cython、nodes=Python；原 FP32、matmul TF32=false、cuDNN TF32=true。
- 叶 batch64、virtual loss3、批内去重开、机会 on/k=0/lazy、完整机会表复用和中间快照削减开，PCR 关。
- 冻结 iter28 检查点：`data/pilots/chance_support_20261008_iter28/frozen_inputs/models/training_checkpoint.pt`。
  SHA256 `46c9ebf2c76ef45ba355bbeaa58c7d3d541484b992e3fd34523637a94d640def`。
- 同目录上层 `measure.fixtures.pkl` 的 18 个既有诊断局面，SHA256
  `eb4d1773c445ad66bedeae6e0647df24986855514128a92d12ac7535f6f316ef`。
- 普通/强预算 800/3200，各局面都执行两档；诊断根数量为 1:1，不是生产整局的 7:1，也不是生产频率抽样。
- 参照为完成的 `data/pilots/host_wait_games_20261009_iter28/result.json`，沿用其 35 项保护文件。
  加上参照自身、新工具、测试、预声明协议，共 39 项哈希前后未变。

## 采样方法与验收

使用官方 py-spy 0.4.2 Windows x64 wheel，隔离安装在 `data/profilers/py_spy_0_4_2/`，不修改全局环境或项目依赖。
wheel SHA256 `8b06a353c177677e4e1701b288d8c58e2f8d4208ee81a8048d9f72ba800918f8`，
exe SHA256 `6e83ab095f40453742d01a2719a178f308534015da519ec0659d2584139dc995`。
版本与轮子来源见 [官方 PyPI](https://pypi.org/project/py-spy/0.4.2/)。

先验证两个 8 秒 Python CPU 循环子进程，50 Hz 各得到 418 个有效样本，叶函数全部为预设循环。
随后依次执行四个全新的 8-worker 池：无采样前置、50 Hz、100 Hz、无采样后置。
每池两组 36 根热身，再执行四组 36 根正式搜索，共 144 根/池、576 次正式搜索；热身另计 288 次。
正式 seed=`202610110 + repeat*1000 + fixture_index`，与 `sync_baseline` 的 session0/plain 精确比较。

工具只在原 `profile_selfplay.run_case/plain` 外增加每根一次的 Python 栈标记，复用原 `native.init_worker`。
不替换搜索函数，不加网络 hook、CUDA events、计时阶段同步，不改变网络 batch 或随机流。
原有 root 前后同步仍保留；cohort 包含原任务分发和小结果 IPC，不包含本次结果文件写入。
本次只比策略/着法/RNG/叶与机会计数，**没有重新进行全网络流/全图或完整局对照**。

采样参数为 `--gil --subprocesses --threads --full-filenames`、Speedscope 输出、默认一致性快照；
不采 native/locals，不使用 nonblocking，不连接符号服务器或上传报告。
根据每根记录的 PID/native TID，仅保留 worker 主线程，并要求同时出现工具的 `measured_task`、
唯一的普通/强标记、原 `azero.search`。父进程、热身、导入、退出、结果哈希及其他线程全部排除。

GIL 过滤会漏掉释放 GIL 的原生活动，持有 GIL 的 C/Cython 工作可能落在 Python 调用者上；
默认采样暂停目标读取一致栈，仍有扰动，idle 判定也不是精确 CPU 调度证明。
方法边界见 [py-spy 官方说明](https://github.com/benfred/py-spy)。
Speedscope `weights/endValue` 是样本表示，不是可直接累加的实测 CPU 时间；本工具只统计样本，
见 [固定版本导出实现](https://github.com/benfred/py-spy/blob/v0.4.2/src/speedscope.rs)。
inclusive 每样本对同名同文件递归帧去重；匿名/同名函数仍可能合并，定位时必须看 current-line 表。

预声明协议：`data/pilots/python_hotspots_20261009_iter28_protocol.md`。
每 worker 至少 200 个正式搜索样本、总计至少 2,000；无采样池内 CV<=5%、跨池中位差<=10%，
采样相对前后无采样总耗时均值增幅<=10%。这是诊断接受门槛，不是生产优化门槛或统计置信保证。

## 开销、漂移与覆盖

| 组 | 四个正式 cohort 耗时 / s | 合计 / s | 中位 / s | CV | 正式搜索样本 |
|---|---|---:|---:|---:|---:|
| 无采样前置 | 5.9400 / 5.9529 / 6.0320 / 6.2148 | 24.1397 | 5.9924 | 2.10% | -- |
| 50 Hz | 6.2369 / 6.2465 / 6.4513 / 6.1757 | 25.1105 | 6.2417 | 1.91% | 5,377 |
| 100 Hz | 6.2408 / 6.2408 / 6.3069 / 6.3090 | 25.0975 | 6.2739 | 0.62% | 11,168 |
| 无采样后置 | 6.1998 / 6.3169 / 6.2776 / 6.2981 | 25.0924 | 6.2879 | 0.82% | -- |

前后无采样合计均值 24.6161s；50/100 Hz 比它分别慢 **2.01%/1.96%**。
但前后无采样中位漂移 **+4.93%**，与各自前置相比采样慢 4.02%/3.97%，与后置相比仅慢 0.07%/0.02%。
因此这些只是跨新池的开销估计，混有机器漂移，不把约 2% 归因为纯采样暂停成本。
诊断门槛通过，不以这四池结果声称没有采样偏差或精确区间。

50 Hz 每 worker 635--730 个正式样本；100 Hz 为 1,250--1,466，均超过覆盖门槛。
全生命周期导出分别 8,590/17,686 个样本，筛去 3,213/6,518 个非正式搜索样本。
50 Hz 报告 0 个栈读取 Error、5 条 warning；100 Hz 有 1 个栈读取 Error、6 条 warning，全部日志保留。
其中一次失败涉及已知 worker 主线程，未建立失败的绝对时间与测量阶段对应，不断言它在热身或对排名无影响。
两档前十名集合一致只支持粗粒度排序；不证明零丢失，也不保证低占比函数的精确先后。

## 当前排名

以下均是**正式搜索中持有 GIL 的条件样本比例**，分母不是整轮墙钟或全部 CPU 活动。
“叶帧”指栈顶 Python 帧，可能正在调用 C/Cython；“含后代”是出现该函数的栈比例，列中存在嵌套，不能相加。

| 函数/路径 | 50 Hz 叶帧 | 100 Hz 叶帧 | 50/100 Hz 含后代 | 判断 |
|---|---:|---:|---:|---|
| `NeuralMCTS._expand_node` | 3.61% | 3.59% | 28.68% / 27.30% | 大边界，含归一化、分配和机会包装，不是单一可编译整数热点 |
| `PUCTNode.expand` | 7.46% | 7.04% | 20.64% / 19.26% | 主要仍为 Python 对象/字典与先验处理，旧对象式编译额外收益不稳定 |
| `PUCTNode.__init__` | 9.93% | 9.19% | 9.93% / 9.19% | current-line 证实主要是三份空字典分配；不等于用 Cython 搬运能省掉 |
| `_encode_state_into` | 7.72% | 8.95% | 9.34% / 10.28% | 小范围精确编码候选，浮点转换与历史通道不可改变 |
| `_iter_legal_moves` | 9.63% | 8.51% | 9.76% / 8.59% | 叶帧主要是 line980 `yield from` 调用已启用 Cython 快路，不是未编译 Python 循环占比 |
| `PUCTNode.select_child` | 5.06% | 5.29% | 7.79% / 7.84% | 第二候选；包含求访问和、Q/U 运算、顺序比较 |
| `_repetition_entries` | 4.89% | 4.33% | 6.51% / 5.94% | 历史准备候选；键/顺序/完整性/容量异常语义必须保留 |
| `GameState.apply_move` | 1.95% | 2.15% | 6.10% / 6.91% | 包含规则、公开历史、揭示及重复计数，不建议整函数重写 |
| `_full_chance_outcomes` | 1.62% | 1.35% | 1.66% / 1.41% | 已启用机会表复用，不能再拿旧机会表构造排名做依据 |
| `_piece_threat_targets` | 1.23% | 1.49% | 1.28% / 1.61% | 小整数循环有可能编译，但当前不足以支持大收益预期 |

采样叶帧 `azero.py` 约68%--70%、`jieqi.py`约19%、`ismcts.py`约7%；不能乘上上一份 Nsight 的 Python 模块比例推算整轮收益。
`_evaluate/_evaluate_batch` 栈约14.15%/15.46%，不与之前约50%的网络边界墙钟矛盾：大量释放 GIL 的计算/等待未进本分母。
普通预算的样本占比约19%，强预算约81%；当前总体排序主要受强预算影响，`analysis.json`另存各预算排名。
例如选子含后代的总体比例约7.8%，不能直接外推为生产节省7.8%。

## 下一步实验与约束

1. **首选编码/重复历史准备**：先逐语句细分现有路径，筛选一个不改网络形状、历史顺序与浮点运算的精确原型。
   先做位级输入对比、缺失/不完整/容量超限历史和所有规则字段反例，再验收固定模型网络流、图、policy/RNG，最后独立完整局吞吐。
2. **第二候选 PUCT 选择**：仅独立原型，不共享多父统计，不改回传；求和顺序、运算舍入、`>`平局规则、字典遍历顺序必须相同。
   不启用 fast-math/FMA 改写，不用较低精度或向量化规避精确门槛；先证明数值与搜索轨迹一致。
3. **暂缓旧节点编译/泛化规则重写**：已有初始化编译更慢、已开 legal 后扩展边际不稳定的证据。
   三个空字典的分配是对象成本，直接 Cython 搬运不能消除；惰性容器属于另一个改动，需新证明，不在本次实现。
   `_checker_squares`/长捉小循环只保留候选，当前不优先扩大编译范围。

这里只确定研究优先级，尚未实现/验收上述新优化，不承诺收益超过10%。
后续仍采用预声明的稳定性与完整局精确/吞吐门槛；失败候选不接生产，不为了提速改变预算、叶 batch、精度或训练数据量。

## 产物、复现与检查

工具：[profile_python_selfplay.py](../../tools/profile_python_selfplay.py)，测试：[test_profile_python_selfplay.py](../../test_profile_python_selfplay.py)。
结果：`data/pilots/python_hotspots_20261009_iter28/` 下四组 `result.json`、两份 `*.speedscope.json`、
两份 `analysis.json`、stdout/stderr、`launch.json`和`summary.json`；兼容性探针在同前缀的 `_probe/` 与相邻原始文件。
`summary.json`记录各池耗时、覆盖/开销门槛、精确根数、前十集合和分析输入哈希；公式为
`slowdown = sampled_cohort_sum / mean(plain_before_sum, plain_after_sum) - 1`，CV使用样本标准差。
源码、测试与预声明协议受本次哈希保护；修改后不要直接用旧报告充当新工具的验收。

用新目录，不覆盖现有结果，不与训练、其他 Python 工作或 Nsight/WPR 录制并行。
先安装/核对上述隔离采样器，随后一次只执行一个命令；完整复现按前置 plain、50、100、后置 plain 的顺序。

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
$py='<HISTORICAL_PYTHON>'
$spy='<PROJECT_ROOT>\data\profilers\py_spy_0_4_2\runtime\bin\py-spy.exe'
$reference='data/pilots/host_wait_games_20261009_iter28/result.json'
& $py tools/profile_python_selfplay.py --workload --reference-file $reference --output-dir data/pilots/python_hotspots_new_plain_before
& $spy record --format speedscope --subprocesses --threads --full-filenames --gil --rate 50 --output data/pilots/python_hotspots_new50.speedscope.json -- $py tools/profile_python_selfplay.py --workload --reference-file $reference --output-dir data/pilots/python_hotspots_new50
& $spy record --format speedscope --subprocesses --threads --full-filenames --gil --rate 100 --output data/pilots/python_hotspots_new100.speedscope.json -- $py tools/profile_python_selfplay.py --workload --reference-file $reference --output-dir data/pilots/python_hotspots_new100
& $py tools/profile_python_selfplay.py --workload --reference-file $reference --output-dir data/pilots/python_hotspots_new_plain_after
& $py tools/profile_python_selfplay.py --analyze data/pilots/python_hotspots_new50.speedscope.json --workload-result data/pilots/python_hotspots_new50/result.json --analysis-file data/pilots/python_hotspots_new50/analysis.json
```

100 Hz 分析同样使用对应路径。采样报告仅本地解析，不打开/上传到在线 Speedscope。
工具要求固定原输入、源码与本机扩展哈希；更换机器、后端或源码时需另建输入协议，不能绕过哈希校验。
本阶段新增分析/兼容性定向测试和已有测量测试共 **68 passed / 6.16s**；未重跑生产全量回归。
没有修改生产文件、续训、buffer、二进制、模型或 Git 历史；采样后无残留 Python/Nsight/py-spy 进程，WPR未录制。
