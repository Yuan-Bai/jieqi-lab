# 完整机会分布复用实验

日期：2026-10-08。范围：减少 lazy/on/k=0 路径中 `_full_chance_outcomes`
的重复概率表构造，不缓存网络结果，不合并树/图节点，不调整搜索预算。

**结论：保留第二版编译防护 + 不可变概率表 LRU。** 192 局平均配对省时 2.55%，
合计省时 2.75%，描述性配对区间 [1.77%, 3.66%]；样本/RNG 精确对齐，
通过原先声明的吞吐筛选。第一版纯 Python 防护未过筛选，不作为生产路径。
这是本机冻结模型的小批验收，不是任意输入的数学证明或 1024 局整轮保证。

## 环境与边界

- 冻结已完成 iteration 28 的 v5 模型；原目录 `data/train_rules5_20261004/` 不写入。
- i9-13900HX、RTX 4060 Laptop 8GB、约 32GB RAM；Python 3.11.5、
  PyTorch 2.6.0+cu118、NumPy 2.0.2；Cython 3.3.0 / MSVC 14.42.34433 构建。
- rules/legal=Cython、nodes=Python；8 worker、每进程 torch CPU threads=1。
- 普通/强局 800/3200 模拟，强局间隔 8；leaf batch=64、dedup 开、
  chance on/k=0/lazy 开，PCR 关。FP32 张量、matmul TF32 关、cuDNN TF32 开，未更改。
- 最新生产 manifest 仍是 legal=Python；用户计划下次显式设置
  `$env:JIEQI_LEGAL_BACKEND='cython'`，本实验已以 legal=Cython 为基线。
- 冻结目录只含最新 replay，并不是可续训的完整 buffer 副本。
- 不运行优化器、生产训练、棋力评估或数据增强，不修改精度设置；未提交 Git。

## 实现候选与安全约束

只缓存两份不可变计数快照推导出的完整揭示概率表。每进程 LRU 上限 4096；
条目是 tuple 嵌套的整数/概率，不持有局面、历史、信念、模型、树节点或搜索统计。
命中复用的仅是确定性数学结果，实际揭示仍从当前模拟世界抽样。

概率运算、类型枚举顺序和 RNG 全部沿用原 Python 枚举器。不能把“计数相同”
用来共享决策价值、真实暗池、网络输出或父边访问量。

- 仅精确内置整数的不可变 tuple 行进入缓存；bool、float、NumPy 数值、
  mutable 行、其它 duck type 沿用原路径。相等哈希的 int/float 不保证相同除法结果。
- 枚举函数或 `POOL_TYPES` 被覆盖时绕过缓存。
- 第一版用 Python `all`/类型检查；第二版只编译这段检查，不转 C 整数，
  不进行 sum/division/概率计算，没有 fast-math。
- 第二版没有有效规则扩展时直接用旧构造路径，不启用收益不明的 Python 防护。
- k>0/off/eager 的既有流程不改；规则、数据、搜索版本不改。
- 验收要求原输入状态、完整物化逻辑图、每次网络输入/输出、策略、走子、
  chance/leaf 计数、整局 compact 样本和最终 RNG 精确一致。
- 现有 `generate_selfplay_results` 按任务索引归序返回；不修改这个流程，
  worker 完成时序不会改变样本入库顺序。缓存不进入检查点，不影响恢复 RNG。

## 重复率与局部成本

18 根来自同模型两局真实轨迹（75/77 ply）、吃暗子偏置根和受控残局；
每根各测 800/3200 模拟，包含已有私有吃子信息。

| 测量 | 结果 |
|---|---|
| 机会表请求 | 819,678 次 |
| 不同计数对 | 1,017 种 |
| 真实快照对象 | 754,035 个 |
| 重复率 / 冷 LRU 命中率 | 99.8759%，本样本所有键均容纳于 4096 上限 |
| 原构造，六次 replay 均值 | 第一版复测 1.0575s；第二版 1.0505s |
| Python 防护缓存 | 0.6942s，平均配对省时 34.35% |
| 编译防护缓存 | 0.2305s，平均配对省时 78.06% |
| 无防护 LRU 查找（诊断，禁止直接集成） | 约 0.0935s / replay |
| 本样本缓存增量内存 | retained 636,816 bytes，约 0.61 MiB / 进程 |

初测 `measure.json` 将请求换成了去重后的键对象，不能视为真实查表成本。
已修正为保留原快照类型/身份，新增回归测试；修正后的重复率、概率位模式与
搜索结果不变，局部省时仍约 34%。因此该测量偏差不是第一版整局失败的主要原因。
类型防护和 Python 调度开销显著，编译检查有局部收益；对整体关键路径的影响仍需整局测量。

微测不包含计数快照创建、局面更新、GPU 等待或完整自博弈；每次从冷缓存开始。
内存是捕获完成后 tracemalloc 的增量，不含保存请求快照的测量内存，也不是最大 4096 项的峰值。
不能把微测的 78% 或 4.56 倍换算为自博弈的提速倍数。

## 第一版完整对照：拒绝集成

计时前声明：六组配对、每组每模式 16 局，96 局/模式，192 局总计；
每组 14 普通局与 2 强局，模式顺序交替，每局缓存清空。
门槛是整局平均配对省时至少 1%、总耗时下降、配对 cohort bootstrap 下界 >0，
且所有精确性检查通过。不能事后调低门槛。

| 指标 | 第一版结果 |
|---|---|
| 36 根/预算条件的完整图与网络流 | 全部精确一致 |
| 8 组固定根配对，共 288 对搜索 | 全部精确一致；平均省时 -0.083%，区间 [-1.51%, +1.03%] |
| 固定根合计耗时 | 原构造 50.4295s，缓存 50.4516s |
| 96 对完整局 | compact 样本、结局、局长、叶数、RNG 全部一致 |
| 六组原构造/缓存墙钟合计 | 766.9596s / 771.1138s |
| 平均配对整局省时 | -0.5378%，即略慢 |
| 合计整局省时 | -0.5417% |
| 描述性配对 bootstrap 95% 区间 | [-1.4461%, +0.1566%] |

结论：局部更快不足以证明 8 worker 自博弈更快。第一版未通过筛选。
不是改变采样/精度导致的局长变化：这 96 对局的样本和局长均完全一致。
上述 bootstrap 只有六个 cohort，不是针对所有硬件/模型的性能置信保证。

## 第二版：通过预声明筛选

在整局计时前另存 `protocol_native_guard.md`。仅更换为编译防护，沿用相同冻结
模型/18 根/预算/缓存大小；两种计时模式都使用同一新规则二进制，避免攻击编译器变化混入收益。
新种子 offset=10,000，不复用第一版的整局计时种子。

仍先做 36 根/预算的完整图与网络流，再八组固定根，最后六组 ×16 局/模式。
仍采用原门槛，不基于中途结果提前停止或降低标准；失败将撤去本次生产接线和原生检查。
定向检查已完成：核心机会/信息集/自博弈 49 passed，编译原生专项 35 passed 加
一项沙箱临时目录权限失败；该项及节点专项在正常系统临时目录下 14 passed；
纯 Python 下新增专项 12 passed、1 项编译专项 skip。
没有修改保存逻辑或弱化断言。整局计时完成后，在正常系统临时目录执行
rules/legal=Cython、nodes=Python 的全量回归：**502 passed / 148.91s**。
新增缺失 guard 拒绝加载和 tuple/int 子类绕过用例也已包含于全量验证。
三个后端均强制 Python 的定向回退回归另通过 **53 passed、1 项编译专项 skip / 17.97s**。

| 指标 | 第二版结果 |
|---|---|
| 36 根/预算条件完整图与每次网络流 | 72 条模式记录全部精确一致 |
| 八组固定根、288 对搜索 | 全部精确一致；平均省时 1.02%，区间 [-0.22%, +2.17%]，不能据此单独确认收益 |
| 固定根合计墙钟 | 原构造 51.6496s / 缓存 51.1157s |
| 六组完整对局 | 96 对 / 192 局；每模式 84 普通局、12 强局 |
| 整局精确性 | 每对 compact 样本、结局、局长、叶数、最终 RNG 均一致 |
| 每模式总 ply / 叶评估请求 | 5,042 / 5,492,729 |
| 六组整局墙钟合计 | 原构造 906.3746s / 缓存 881.4909s |
| 平均配对整局省时 | 2.5527% |
| 合计整局省时 / 吞吐比 | 2.7454% / 1.02823 |
| 描述性配对 bootstrap 95% 区间 | [1.7705%, 3.6597%] |
| 每组省时，按预声明顺序 | 2.2739%、1.2338%、2.5956%、2.0277%、5.1127%、2.0724% |
| worker 内部墙钟合计 | 3,402.8744s / 3,336.6767s，省约 1.95%；不是多进程端到端墙钟 |
| worker process CPU 合计 | 3,261.7500s / 3,199.6563s，省约 1.90%；不是局部算子的占比 |
| 完整局缓存请求 / 命中 | 71,595,154 / 71,484,774，命中率 99.8458% |
| 单局观测最大缓存项数 | 3,567，低于 4,096 上限；本轮没有达到容量上限 |
| 该次验收总用时 | 2,077.43s，约 34.62min，包含图/网络检查与固定根测量 |

门槛三项均通过，六组都更快；不据大于平均值的第五组夸大收益。
尚未做 early/late ply 分段归因，不把某组的长尾耗时单独归因于机会表。
六个 cohort 的 bootstrap 仍是有限同机证据，不能保证未来模型/局长/机器负载都有同样收益。

## 保留代码与续训

- `azero.py`：保留原枚举器作为 miss/fallback，新增每进程有限 LRU 与不可变表。
- `native/jieqi_rules.pyx`：新增 tuple 行 / 精确 Python int 检查；攻击逻辑未改。
- `jieqi.py`：加载有效规则扩展时接入检查函数，缺失函数拒绝加载；源码/二进制来源继续记录。
- 规则扩展本机已重建；合法着二进制和节点方法指纹未改，不需要因为此改动重建它们。
- 新进程规则后端有效为 Cython 且走 lazy/k=0 时自动使用缓存，无新增训练 CLI 配置。
  规则 Python/缺失扩展回退旧计算；`JIEQI_RULES_BACKEND=python` 也会关闭攻击编译，
  不能把这种整体回退当作缓存的单变量测速。工具用原 helper 作控制，保留攻击/合法着编译。
- 不需要重训、清空 buffer、重置 optimizer；缓存不是检查点内容，不改变模型/数据/搜索版本。
- 原生产 `training_checkpoint.pt`、`latest.pt`、`train_state.json`、`train.log`
  和最新 replay 文件的 SHA256 均与改动前一致；其它 replay 没有写路径，但未逐个做哈希核对。
- 测量保护的候选源码、原生二进制、冻结模型/日志/状态/replay 和工具/fixture 在计时期间全部未变。

有限精确对照不等于全未来轨迹的证明；已有 CUDA 非确定性设置没有更改。
本轮没有棋力/学习率/收敛实验，因为没有改精度、网络批次、采样或训练目标；
仍不宣称已经完成 1024 局续训验证。

## 复现与原始产物

本机隔离目录：`data/pilots/chance_support_20261008_iter28/`。
JSON 保留模型配置、实际后端/二进制、源文件哈希、逐根/逐局精确摘要、计时原始值；
JSONL 是逐步追加的旁路日志。工具结束时验证受保护文件哈希未变化。

- `protocol.md`、`protocol_native_guard.md`：各版本预声明协议。
- `measure.json`：初测，仅能参考重复数；键对象复用的成本口径已更正。
- `measure_snapshots.json`：真实快照的 Python 防护成本。
- `measure_native_guard.json`：真实快照的编译防护成本。
- `validate.json`：第一版固定根、完整图/网络和 192 局结果。
- `validate_native_guard.json`：第二版完整结果，不以 JSONL 中途数据替代最终结果。
- `measure.fixtures.pkl`：可信本地测试局面，SHA256
  `eb4d1773c445ad66bedeae6e0647df24986855514128a92d12ac7535f6f316ef`。
- `azero_before.py`、`jieqi_before.py`、`rules_before.pyx`、`rules_before.pyd`：
  改动前对照快照；不应整文件覆盖用户其它未提交修改。
- `azero_candidate.py`：第一版 Python 防护；`benchmark_source.py`：初测工具；
  `benchmark_snapshots_source.py`：修正成本口径的工具。
- `azero_native_guard.py`、`jieqi_native_guard.py`、`rules_native_guard.pyx`、
  `benchmark_native_source.py`：第二版计时对应源码。冻结输入/源码不是 Git 提交，
  被忽略的本地文件不会自动随仓库分享。

当前第二版复查命令如下，`$out` 必须是尚不存在的文件名；不要与生产训练争用硬件。
只执行 measure 不会跑梯度更新；validate 需要较长时间，不能据提前观察的统计停止。

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
python tools/build_rules.py
$frozen='data/pilots/chance_support_20261008_iter28/frozen_inputs'
$fixtures='data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl'
$out='data/pilots/chance_support_20261008_iter28/new_measure.json'
python tools/benchmark_chance_support.py --phase measure --run-dir $frozen --fixtures $fixtures --output $out
$out='data/pilots/chance_support_20261008_iter28/new_validate.json'
python tools/benchmark_chance_support.py --phase validate --run-dir $frozen --fixtures $fixtures --seed-offset 10000 --search-rounds 8 --game-rounds 6 --games-per-round 16 --workers 8 --output $out
```

第一版历史结果需要对应的源快照与旧二进制，不是在当前源码上运行同一命令即可复现。
重建时不要让旧二进制被正在运行的 Python 进程占用。
