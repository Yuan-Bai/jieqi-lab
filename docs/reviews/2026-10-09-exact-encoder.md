# 精确编码编译筛选（2026-10-09）

## 结论

完成一个独立 Cython 编码原型的正确性、局部成本和双池固定根筛选，**不接生产**。

- 36 对全图/网络流搜索精确；63,447 次实际编码输入逐位一致，全部满足原型快路条件。
- 576 次正式根搜索与冻结参照的 policy、着法、RNG、叶/机会计数精确一致。
- 576 个实际编码观察的微测，编码耗时下降 **24.27%**，局部吞吐提高 **32.06%**。
- 双池根搜索合计耗时仅下降 **0.409%**、吞吐提高 **0.411%**；两池分别提高 **0.246%/0.575%**。
- 稳定性门槛通过，但合计>=3%、各池>=1%的实用吞吐筛选未过，按协议不进入整局或生产接入。

局部代码更快已经验证，生产自博弈收益没有验证。不能将24%的编码省时当作24%的搜索加速，
也不能据此证明编码完全没有优化空间或整体已经最优。
本轮**没有更改 `_repetition_entries`**、历史顺序/过滤、网络形状、精度、预算或训练样本量。

## 原型与安全边界

新增 [experimental_encoder.pyx](../../native/experimental_encoder.pyx) 与
[benchmark_exact_encoder.py](../../tools/benchmark_exact_encoder.py)，没有生产加载器。
扩展仅位于 `data/pilots/..._build_v2/lib/`，不生成/替换仓库根目录的 `.pyd`，普通训练不会自动启用。
测试环境变量 `JIEQI_EXPERIMENT_ENCODER_DIR` 也不是训练开关。

原型只替换 `_encode_state_into`：标准整数棋盘的独热平面索引使用 C 整数循环，
剩余流程按原顺序编译执行；浮点除法仍使用 Python 对象，NumPy cast、切片、广播和重复历史准备调用保持原样。
没有用 float32 做原来 float64/Python float 的算术，没有 fast-math 或表达式重排。
使用已隔离安装的 Cython **3.3.0**、MSVC **14.42.34433**；构建命令额外设置 `/fp:strict`。

初始化核对源码、核心编码/helper code 指纹和二进制身份。快路检查棋盘类型/码范围、数组dtype、
位布局、编码尺寸/历史容量、helper身份、最近走子与显式重复历史记录格式；不支持的这些情况调用原函数。
这是隔离原型对标准 `InformationState` 的验收，不宣称任意自定义状态类/属性回调的完整兼容证明。

首版预检发现畸形 move 的 Cython 解包错误文字不同于 Python，未进行性能测量；
随后增加历史/显式记录格式回退，另建 v2 扩展并重跑全部门槛。v1 构建和 `source_snapshot.pyx` 保留，
当前加载器因源码哈希不符会拒绝旧扩展，不绕过保护。
另一个既有保存测试曾因沙箱临时目录的 `os.replace` 权限失败；正常本机临时目录重跑通过，未修改保存实现。

测试覆盖：全部238通道、缺失/不完整/显式历史、81槽容量及溢出错误、未写通道/非零预填、
signed zero、非连续数组、只读/float64数组、超大Python整数、非原生整数/棋盘、修改helpers/位常量/坐标表、
畸形move/显式记录的异常与部分输出、现有增强/紧凑批次装配、输入不变与吞吐门槛拒绝。
现有增强测试沿用原三种样本变换集合，不将其说成额外完整四向增强验收。

## 冻结输入与协议

沿用[函数热点采样](2026-10-09-python-hotspots.md)的 iter28 和18局面，以及其
`data/pilots/python_hotspots_20261009_iter28/plain_before/result.json` 的39项保护文件。
增加工具、测试、原型、预声明协议、参照自身、build记录、实验二进制和实际微测输入，共 **47项哈希未变**。

Python `<HISTORICAL_PYTHON>` / 3.11.5，PyTorch2.6.0+cu118、CUDA11.8；
RTX4060 Laptop、i9-13900HX、32GB，环境同此前基线，不调功耗/前后台/进程优先级。
rules/legal=Cython、nodes=Python，FP32、matmul TF32=false、cuDNN TF32=true，
leaf64/virtual3、去重/完整lazy机会表复用/中间快照削减开、PCR关。
诊断根800/3200各一半，不是生产7:1整局频率，也不含4M replay常驻。
冻结检查点SHA256 `46c9ebf2c76ef45ba355bbeaa58c7d3d541484b992e3fd34523637a94d640def`。

协议：`data/pilots/exact_encoder_20261009_iter28_protocol.md`，先于性能测量写入。
局部吞吐>=10%才进入双池根筛选；原型修正发生在反例预检，不根据吞吐调整实现或门槛。
根筛选要求普通基线池内CV<=5%、前后漂移<=10%、跨池中位差<=10%；
合计吞吐>=3%、各池>=1%、各池4对至少3胜。通过也只意味着优先整局筛选，不允许直接生产接入。

## 正确性与局部成本

每个局面两档预算做原编码/候选成对搜索，共36对、72次审计搜索。
候选审计调用每次同时计算原编码并比较所有float32字节，网络hook记录完整批次输入/输出顺序，
另比较完整逻辑图、policy/着法/RNG/计数。两模式各1,129次网络请求，63,447次候选编码全部相同。
所有插桩、额外参考计算、图补全/序列化都在吞吐计时之外，不拿审计耗时宣称提速。

从每个候选审计根保留头8/末8个真实输入，共576份（允许重复，不是576个独立随机样本）；
显式冻结已准备的重复历史条目，8个顺序交替微测对、每组循环8次，每模式36,864次调用。
只测填充同一份预分配float32输出，不含清零/分配、历史条目准备、网络前向或传输。
微测开始前逐输入位级验收，结束后输入pickle哈希不变。

| 模式 | 编码调用合计 / s | 每次编码均值 / us | 口径 |
|---|---:|---:|---|
| 原Python编码 | 0.79558 | 21.581 | 原float/NumPy流程，显式历史已准备 |
| 编译候选 | 0.60245 | 16.343 | 包含快路防护，显式历史已准备 |

按合计编码调用算，省时24.27%、吞吐+32.06%，局部门槛通过。
所有576份输入都走快路，不把Python回退的相同结果算成原型生效证明。

## 双池根吞吐

每池8个spawn worker，两模式各36根热身，然后4个平衡A/B对，每组36根。
第二池整体反转先后；正式seed=`202610110 + repeat*1000 + fixture_index`。
原 `run_case/plain` 的根前后同步、网络批次、固定模型、原有任务/小结果IPC保持不变；
正式计时不带网络hook、逐编码参考、额外CUDA event或编译调用计数。
576次正式执行覆盖144种根/预算/seed条件，所有模式/池匹配此前冻结plain参照。

| 池/对 | 原编码 / s | 候选 / s | 吞吐变化 |
|---|---:|---:|---:|
| 0/0 | 6.18018 | 6.10324 | +1.261% |
| 0/1 | 6.16027 | 6.36851 | -3.270% |
| 0/2 | 6.30773 | 6.19917 | +1.751% |
| 0/3 | 6.37831 | 6.29426 | +1.335% |
| 1/0 | 6.35672 | 6.21751 | +2.239% |
| 1/1 | 6.33521 | 6.26141 | +1.179% |
| 1/2 | 6.18917 | 6.28337 | -1.499% |
| 1/3 | 6.33855 | 6.31319 | +0.402% |
| 合计 | 50.24615 | 50.04066 | **+0.411%** |

池0/1基线CV1.66%/1.23%，前后漂移+2.80%/-1.29%，跨池中位变化+1.49%，稳定性门槛通过。
两池各3/4对快，但各池合计增益只有0.246%/0.575%，合计0.411%远低于预声明3%门槛。
不把6对为正解释成统计显著，也不把576根当576个独立硬件试验。没有预声明的置信区间或因果分解。

推测：部分局部CPU省时可能被并行推理/等待/其他CPU工作覆盖；这只是可能解释，
本次没有逐阶段或CPU调度trace来确认，不把GPU/Windows后台功耗当作已证明原因。
结论仅为当前候选未达到实用根吞吐门槛，不是已证明真实收益严格等于0.411%或为零。

## 保留产物与复现

- `data/pilots/exact_encoder_20261009_iter28_build/`：失败预检v1构建及源码快照；没有正式性能结果。
- `data/pilots/exact_encoder_20261009_iter28_build_v2/`：`build.json`、源码快照、C/object、实验`.pyd`。
  最终原型源码SHA256 `bfe3391e3b92990a7da6c476a97cd06eb1e4f95de6f24a2adad601cc2cc4878e`；
  `.pyd` SHA256 `e949d1c558a52897ed6c27b067970a9e20230e62cae7c657ab0d31fcdcb6c3eb`。
- `data/pilots/exact_encoder_20261009_iter28_screen/`：`result.json`、逐事件/分组`events.jsonl`、
  `micro_cases.pkl`、最终核验记录。正式screen耗时 **290.993s**，含审计/微测/双池热身/退出，不等于生产轮耗时。

以下用全新输出目录；先确认没有其他Python/GPU训练/Nsight任务、WPR不录制。
构建使用既有 `data/build_deps` 中的Cython，不联网安装依赖，失败时保留目录并换新目录，不覆盖旧结果。

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
$py='<HISTORICAL_PYTHON>'
& $py tools/benchmark_exact_encoder.py --phase build --output-dir data/pilots/exact_encoder_new_build
$env:JIEQI_EXPERIMENT_ENCODER_DIR=(Resolve-Path data/pilots/exact_encoder_new_build).Path
& $py -m pytest -q tests/test_exact_encoder_experiment.py tests/test_encoding_acceleration.py tests/test_rule_history_encoding.py
& $py tools/benchmark_exact_encoder.py --phase screen --output-dir data/pilots/exact_encoder_new_screen --build-dir data/pilots/exact_encoder_new_build --reference-file data/pilots/python_hotspots_20261009_iter28/plain_before/result.json
```

源码或核心方法改变后需新构建/协议，旧哈希参照不能当作新实验的同变量验收。
新构建的二进制哈希可能不同，不以本机pyd哈希替代源码/版本/实际加载路径验证。
默认测试未设置实验目录时跳过编译专测；本次显式加载v2的定向 **29 passed / 3.90s**，
再连同共享测量工具共 **97 passed / 4.65s**，未跑生产全量回归。
生产源码/扩展/检查点/状态/日志/replay等47项保护哈希在测量前后未变；未进行生产训练、更新冻结模型、重置buffer或提交Git。

## 下一步

此直接编码编译候选停止，不增加整局或训练试验，不修改生产编码。
下一候选优先独立筛选 **PUCT选择循环**，必须保留浮点运算顺序/舍入、访问求和与平局选择顺序；
重复历史准备仍未实施，保持待筛，不借编码局部收益推断其整轮潜力。
如果后续候选也只有约1%的不稳定收益，收束零散微优化，优先训练诊断与正式固定模型评估。
