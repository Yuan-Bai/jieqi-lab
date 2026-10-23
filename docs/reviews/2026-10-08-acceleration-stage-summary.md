# 训练加速阶段总结

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

> 后续机会表复用、快照削减、联合轮/原生归因与三个CPU原型已完成，当前结论见
> [研究总览](../research-status.md)。本页保留2026-10-08整理时点，不用旧“下一步”覆盖后续决策。

核对日期：2026-10-08。工作区：`<PROJECT_ROOT>`。这是截至本日的阶段快照，不替代原始报告或未来基准。
同日随后新增 [推理混合精度隔离实验](2026-10-08-inference-precision.md)：数值/搜索变化不满足原逐位等价门槛，尚未验收棋力或学习，不启用主线；下文保持本次整理时的阶段事实。

## 结论与证据等级

保留精确编码/批次装配、策略数组/批内历史复用、攻击编译、可选合法着快路，以及默认机会对象按需分配。
节点编译、集中推理、跨批缓存不默认接入；图搜索仍是独立研究，未通过新增有效换位门槛。
**最新生产实际上尚未开启合法着快路**：当前run记录为 rules=Cython、legal=Python、nodes=Python，不能把推荐组合写成实际已启用组合。

- **本次核对**：直接读取当前checkpoint/config、manifest、日志、源码、Git状态、环境与文件哈希；本轮没有重跑性能实验。
- **历史实测**：取自已保存的配对实验与原始JSON，注明模型/负载/分母；不保证在当前模型上重现相同比例。
- **推测**：原因或潜在收益尚未完成因果实验；不作为启用依据。
- **待验证**：没有测量或没有达到验收门槛。未通过逐位等价不等于已证明棋力/收敛变差。

没有同一冻结模型、同一完整生产负载的“全部优化前→全部优化后”联合吞吐对照。
以下不同实验的百分比不能相加，也不能保证训练需要的数据量或达到目标棋力的时间按同样比例改善。

## 1. 训练环境与现状

| 项目 | 核对结果及边界 |
|---|---|
| CPU | i9-13900HX，本次注册表核对；历史测量也使用此CPU |
| GPU | RTX4060 Laptop；本次nvidia-smi显示8188 MiB显存、驱动610.47；历史报告通常称8GB |
| RAM / GPU功耗 | 约32GB RAM来自10-05报告，本次未重新核定总物理内存；GPU功耗上限查询N/A，实际TGP未知 |
| 系统 | Windows，当前平台API报告build26200；不据API字符串判断Windows商品版本 |
| Python | Anaconda Python3.11.5，`<HISTORICAL_PYTHON>` |
| 运行库 | NumPy2.0.2、PyTorch2.6.0+cu118，CUDA构建11.8；本次直接核对 |
| 编译链 | 历史构建记录Cython3.3.0/MSVC14.42；本轮未重建、未重新核定编译工具安装状态 |
| 精度 | 历史基准FP32/NCHW，matmul TF32=false、cuDNN TF32=true；本次新进程查得同值、cuDNN benchmark/deterministic=false。最新生产manifest未单独记录这些精度开关，不能当作对运行中进程的直接采样 |
| 并行 | 生产8个spawn自博弈worker，各worker源码设置Torch CPU线程数1；每worker独立GPU模型，未启用集中推理 |
| 语义版本 | rules=5、data=3、search=3、eval protocol=2；238输入通道、15残差块/64主干通道，2550动作 |
| 搜索配置 | 普通800/强3200，强局间隔8；leaf batch64、virtual loss3、leaf dedup开、chance on/k=0、lazy开、PCR关 |
| 学习配置 | batch1024、1000更新步、AdamW、LR5e-5、cosine总计划2000轮、buffer4M；镜像/翻面增强开，固定诊断32局/batch128 |
| 最新训练 | 完成28轮/28,672局，下轮索引28；最新提交轮为iter27，2026-10-08 08:12结束；最近一次启动新增6轮 |
| 训练内评估 | 当前eval interval=5，不是早期命令中的20；100局、NN/RIS预算800。iter19 score=.91，iter24为88胜/2和/10负、score=.89；均不代替独立棋力/收敛验收 |

来源：当前 `models/train_state.json`、`models/training_checkpoint.pt`、`logs/train.log`，以及
最新run manifest（本地实验产物，未随源码分发：`data/train_rules5_20261004/runs/420917973e9f4fcab8d528ed9a612c35/manifest.json`）。
manifest明确记录 `legal_backend=python`，且记录的 `azero.py` 哈希与本次源码吻合；不是由新诊断进程的默认后端反推生产状态。
历史测速驱动、CPU温度、功率模式、GPU持续频率、背景进程没有统一完整留证，不能默认与今天完全相同。

### 最新六轮：运行观察，不是单变量提速验收

| iter | 自博弈秒 | 平均步数 | 新增增强样本 | 更新步 | 完整轮秒 | 含训练内评估 |
|---|---:|---:|---:|---:|---:|---|
| 22 | 5919.66 | 54.1 | 221652 | 1000 | 6334.09 | 否 |
| 23 | 5920.55 | 53.3 | 218288 | 1000 | 6336.78 | 否 |
| 24 | 5768.24 | 53.5 | 218952 | 1000 | 6918.13 | 是 |
| 25 | 5732.80 | 52.8 | 216316 | 1000 | 6148.17 | 否 |
| 26 | 6103.45 | 55.1 | 225592 | 1000 | 6520.02 | 否 |
| 27 | 5968.05 | 53.2 | 217752 | 1000 | 6400.56 | 否 |

六轮平均自博弈5902.13s（98.37分钟）。五个非评估轮完整耗时平均6347.92s（105.80分钟），
按总耗时加权，自博弈占93.40%；其余约419.02s/轮，包含采集后处理、增强/保存、诊断、训练与检查点等，不是纯GPU更新耗时。
完整轮秒由日志“迭代开始→迭代结束”计算；自博弈秒取程序汇总字段，两者计时边界不同。
“自博弈汇总→训练日志”约265.51–270.36s，也不直接等于1000次GPU更新的纯耗时。

相较旧模型/旧steps的8127s等历史生产数字，不能把当前5902s之差全归因于某个优化：权重、局长、配置、机器状态都不同。
4M buffer与1000步已经真实运行，但尚无同模型的1024局联合A/B或全过程峰值RAM/显存观测。
当前每轮1024000次抽样，每个4M缓冲样本当轮期望0.256次；抽样数/新增样本的流量比约4.66，二者不是“每条样本每iter训练4次”。

## 2. 已尝试方案与实测结果

表中“耗时下降”按对应报告口径；完整训练步、固定根、微函数、完整对局小批次分开看。

| 方案 / 状态 | 模型与负载 | 历史实测结果 | 验证与限制 |
|---|---|---|---|
| 编码/批次预分配，保留 | iter10；固定1024样本训练批次，6组；8-worker固定根 | CPU构批173.23→66.51ms（-61.6%）；完整FP32步343.89→251.39ms（-26.9%）；固定根-2.7% | 三步CUDA损失/梯度/权重/BN/AdamW逐位一致；不是整轮-26.9%。[报告](2026-10-05-exact-acceleration.md) |
| policy数组/批内历史复用，保留 | iter10；已优化编码基线，384对固定根 | batch64叶评估10.128→7.497ms（-26.0%）；8-worker固定根8.3631→7.5970s（-9.2%） | 原批大小/顺序/softmax不变，policy只读FP32、value仍Python float；完整对局仅2对作等价检查。[报告](2026-10-05-selfplay-lowrisk-acceleration.md) |
| 攻击判定Cython，保留 | iter10；已含编码/输出优化，384对固定根 | 攻击微测-74.0%；8-worker固定根7.335→6.957s（-5.15%，合计均值比） | 平均配对降幅5.04%，一对变慢；不是自博弈3.84倍。[报告](2026-10-05-native-rules-acceleration.md) |
| 合法着快路Cython，保留可选，生产未开 | iter15；攻击已编译，8轮固定根+32对完整局 | 合法着微测-38.07%；固定根平均配对-1.82%；完整局平均配对-3.10%，四批区间2.58%–3.61% | 合计551.834→534.480s，对应耗时-3.145%/吞吐+3.247%；不是整轮-38%。[长测](2026-10-06-compiled-path-long-benchmark.md) |
| 节点扩展Cython，默认不启用 | 同一iter15组合长测 | 合法着已开后额外平均仅-0.17%，区间-0.32%至+0.86% | 有限等价测试通过，但额外收益不稳定；单独扩展+1.60%受首批影响。[长测](2026-10-06-compiled-path-long-benchmark.md) |
| 节点初始化编译，不接生产 | 同一组合长测 | 60000节点构造41.57→44.72ms（+7.60%耗时）；完整局边际收益-0.83% | 当前表示/调用结构下更慢，不据此否定所有原生节点设计。初始化只在基准替换。[长测](2026-10-06-compiled-path-long-benchmark.md) |
| 机会对象按需分配，默认保留 | iter16；24根/两预算，64对完整局、8 worker | 64局合计569.80→502.65s：耗时-11.78%、吞吐+13.36%；四批平均降幅12.27% | 配对均值区间9.91%–15.18%；逻辑图、网络流、样本/RNG一致；不是整轮保证。[报告](2026-10-06-lazy-chance-allocation.md) |
| 原批集中GPU，不接生产 | iter16；16对逐请求审计+64对吞吐，2组 | 不计启动402.785→393.872s：耗时-2.21%、吞吐+2.26% | 当前测试逐请求/样本精确一致；只有两组，启动优势不能按比例外推1024局。[报告](2026-10-06-central-inference-pilot.md) |
| 跨请求合批，排除当前接入 | iter16；14根实际叶请求离线2/4合批 | 112个原请求比较中110个输出不一致；最大value差.00657003 | 没有测合批速度/学习效果；不能满足当前精确性要求。[报告](2026-10-06-central-inference-pilot.md) |
| 跨批缓存，停在测量 | iter16；32对局旁路，8 worker | 同局跨调用重复上限7.215%；LRU4096命中6.436%；签名+LRU约5.613us/输入 | 105743次输出比较中40402次不同；最大value差.00771169。没有实际缓存加速实测。[报告](2026-10-06-network-cache-feasibility.md) |
| 批内叶去重，保持原有开关 | iter10早期6组；后续签名已优化 | 早期叶数-3.61%，墙钟8.4774→8.5827s（约慢1.2%，区间含零）；最新日志叶数减少约3.1%–3.2% | 早期签名2.28 worker秒抵消约1.69评估秒；该净收益已过时，当前净墙钟收益未知；开关改变CUDA批形状。[早期报告](2026-10-05-performance-review.md) |
| 图搜索，独立Gate A未过 | iter22；14根、两预算/两种子，56组 | 95339个展开节点中保守新增候选0；真实棋盘碰撞3.75%/9.80% | 不同公开路径的完整输入态候选也为0；没有图引擎、棋力或训练实验；不能推断所有图搜索无效。[报告](2026-10-07-graph-search-feasibility.md) |

机会对象方向先有证据再实施：eager下113133075个累计机会子节点99.16%未访问；不是瞬时驻留数量。
lazy审计中全部物理节点累计减少72.07%，机会子对象减少99.09%；3200单worker进程峰值RSS1034.7→759.4MiB。
这是单进程特定根的RSS，不是显存、8-worker峰值或4M buffer总内存，不能直接乘8。[访问率](2026-10-06-search-reuse-measurement.md)、[内存验收](2026-10-06-lazy-chance-allocation.md)。

### 精度等价门槛与主动暂缓项

- 缓存即使加batch大小，23756次比较仍有5316次policy不同；完整批大小+行顺序+全部输入相同仅2/29647次，全部为根，没有叶整批复用。不是“命中率7%就能节省7%自博弈”。
- 合批/缓存差异的具体CUDA内核或TF32因果尚未定位。输出不同已验证，“一定让训练更难/需要更多数据”没有验证。
- 早期AMP FP16完整步341.1→243.1ms，但改变数值路径，未接scaler完整续训/棋力收敛验收；推理短测10.15→9.97ms，未见有意义的整体优势。
- AMP+channels_last完整步380.4ms，比该轮FP32更慢；不推荐在当前软件栈直接开启。该结论来自旧编码阶段，当前版本的收益未知。[探针](2026-10-05-performance-review.md)。
- PCR、降低模拟预算、扩大叶batch、量化、近似缓存和跨根Q复用改变搜索/监督或数值边界，不作为当前精确加速；PCR已有实现但默认关闭，不声称已做等棋力验收。
- 原批集中服务精确性通过，但约2%的小样本收益未足以抵消服务故障/超时/生命周期维护成本；不是技术上不可行，后续可独立复查。

## 3. 可复现基准与指标

### 3.1 模型/代码必须先冻结

确认存在的审计副本：`data/pilots/graph_transpositions_20261007_iter22/frozen_inputs/`，含checkpoint/latest/state/log/最后一个replay。
它足够供下列只读测量工具使用，但**缺少完整buffer清单，禁止用作训练续训目录**。
历史iter10/15/16检查点是否另有完整存档，本次未全面核实；不能用今天iter28模型冒称复现其数值结果。

| 历史模型 | checkpoint SHA256 | 对应实验 |
|---|---|---|
| iter10 | `aee9016a8b4e0b443feb87dc891ee1d8a642a6ed6953c48b2a9ef7227490cf31` | 编码、输出/历史、攻击、旧热点 |
| iter15 | `5d21e89f09cdbea7390a392f04c8c3bd379837c481f3258533adbb522a6dccba` | 90分钟编译组合 |
| iter16 | `74f72ee9a9689bef759cedfca5374983013a55c0e0a894b5889d48d414cfec7d` | 访问率、lazy、缓存、集中GPU |
| 冻结iter22 | `4d22aa4d04991313e2cce77a4ea3c12c391c1cf8767a54cf7e75b57ab90e9dc8` | 当前可用独立重测基准、图搜索首测 |
| 当前iter28 | `46c9ebf2c76ef45ba355bbeaa58c7d3d541484b992e3fd34523637a94d640def` | 本日生产快照；以后训练会更新 |

当时使用私有工作区分支与快照（未随本仓库分发），工作区有未纳入快照的改动。
本日 `azero.py` SHA256为 `3a238d66fed014ad0de1b4847c5e26de65a41f1f51ed0033876404f44607a1f1`。
每次结果保留config、种子、模型/源/实际原生二进制哈希、精度及计时边界；旧源码、平台ABI、二进制不齐时标记“条件未恢复”，不绕过加载校验。

### 3.2 当前基准配方：不在本次执行

以下在项目根目录、生产训练空闲时选择性运行；可能耗时数十分钟至数小时。
测量均用新输出文件，拒绝覆盖；当前冻结iter22重测属于新条件，不承诺重现iter10/15/16的数值。

```powershell
$benchRun='data/pilots/graph_transpositions_20261007_iter22/frozen_inputs'
$out=Join-Path 'data/pilots' ('accel_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
if (Test-Path -LiteralPath $out) { throw 'Output directory already exists; choose a new path.' }
New-Item -ItemType Directory -Path $out
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
python -c "import jieqi as j,azero as a; print(j.RULES_BACKEND,j.LEGAL_BACKEND,a.SEARCH_BACKEND); print(j.RULES_BACKEND_NATIVE,j.LEGAL_BACKEND_NATIVE); print('lazy',a.AZConfig().lazy_chance_allocation)"
```

需要匹配的攻击/合法着扩展。缺失时另行按 [构建说明](../performance.md#编译后端) 构建，不在训练或基准运行中重建。

**A. eager/lazy精确整局与内存对照**：64局/4批、800/3200与1/8强局，读取池总秒数、按批配对降幅、样本/网络流/RNG、RSS和逻辑图一致性。

```powershell
python tools/benchmark_chance_allocation.py --run-dir $benchRun --workers 8 --search-rounds 4 --game-rounds 4 --games-per-round 16 --output (Join-Path $out 'chance.json')
```

**已确认的复现限制F1/P2**：该工具仍无条件读取 `_jieqi_search.cp311-win_amd64.pyd`，即使nodes=Python。
当前本机有此文件；仅构建两个所需扩展的机器会预检失败。不要为修复依赖而开启生产节点编译；该工具问题尚未修复，见 [审查](2026-10-06-four-accelerations-code-audit.md)。

**B. 合法着/节点8种组合长测**：6144次固定根计时、128次整树检查、32种子×8组合完整局；按配对轮/批而非单次搜索计算区间。

```powershell
$env:JIEQI_SEARCH_BACKEND='cython'
python tools/benchmark_compiled_paths.py --run-dir $benchRun --output (Join-Path $out 'compiled.json') --search-rounds 8 --search-blocks 4 --game-rounds 4 --games-per-round 8 --micro-pairs 8 --workers 8
$env:JIEQI_SEARCH_BACKEND='python'
```

本项确实需要当前可选节点扩展；初始化只由工具在各组合内替换。
当前冻结22默认lazy开，历史iter15长测尚未开启lazy，不能直接比较两个实验的绝对耗时或照搬节点边际收益。

**C. 当前混合版本多局面热点、缓存与集中推理**：分别输出不同文件；trace是精确性审计，不计作吞吐结果。

```powershell
python tools/profile_selfplay.py --run-dir $benchRun --games 3 --roots-per-game 6 --repeats 2 --workers 8 --parallel-pairs 2 --output (Join-Path $out 'profile.json')
python tools/measure_network_cache.py --run-dir $benchRun --games 32 --workers 8 --output (Join-Path $out 'cache.json')
python tools/benchmark_central_inference.py --run-dir $benchRun --games 8 --rounds 2 --workers 8 --trace --output (Join-Path $out 'central_exact.json')
python tools/benchmark_central_inference.py --run-dir $benchRun --games 32 --rounds 2 --workers 8 --output (Join-Path $out 'central_timing.json')
python tools/probe_merged_inference.py --run-dir $benchRun --output (Join-Path $out 'merged.json')
python tools/measure_graph_transpositions.py --run-dir $benchRun --trajectory-games 2 --roots-per-game 4 --seed-blocks 2 --output (Join-Path $out 'graph.json')
```

**D. 历史编码/输出/攻击配方**：参数按当时报告，当前源码再执行时，旧Git基线/源码快照与当前实现之间包含后续改动，尤其lazy；不再自动是单变量编码/输出实验。
若要单独重新归因，先恢复相应源码条件或修订隔离工具并验收。本轮未执行，未承诺旧工具在所有新条件下断言通过。

```powershell
python tools/benchmark_encoding.py --run-dir $benchRun --baseline-ref <PRIVATE_BASELINE_REF> --pairs 6 --blocks 4 --workers 8 --output (Join-Path $out 'encoding.json')
python tools/benchmark_selfplay.py --run-dir $benchRun --baseline-source data/pilots/perf_20261005/before_lowrisk_selfplay.py --pairs 6 --blocks 4 --workers 8 --complete-games 2 --output (Join-Path $out 'selfplay.json')
python tools/benchmark_native_rules.py --run-dir $benchRun --pairs 6 --blocks 4 --workers 8 --complete-games 2 --output (Join-Path $out 'rules.json')
```

旧攻击报告的合法着后端为Python；若要恢复该单变量条件，应先设 `JIEQI_LEGAL_BACKEND=python`，不要混入后续新增的合法着编译变量。当前原生合法着仍调用 `self.is_square_attacked`，保留攻击回调，并非绕过它。
这些命令保存的是隔离探针输出；编码/输出工具会对隔离网络做AdamW测试，不会保存其优化后权重或写生产buffer。

### 3.3 生产命令与推荐组合

下面主体对应当前保存配置；这是供下次正常续训参考的命令，**本次没有执行**。
开始前的环境块是推荐组合，与最新manifest的legal=Python不同；合法着启用需确认加载校验和新manifest记录为Cython。
更新步数1000和评估间隔5来自当前checkpoint，不自动恢复为旧的750/850或20。

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
$run='data/train_rules5_20261004'
python azero.py --iterations 2000 --run-iterations 6 --diagnostic-games 32 --diagnostic-batch-size 128 --self-play-games 1024 --self-play-workers 8 --self-play-device cuda --self-play-log-interval 256 --mcts-simulations 800 --strong-mcts-simulations 3200 --strong-self-play-interval 8 --leaf-batch-size 64 --virtual-loss 3 --chance-nodes --chance-k 0 --eval-interval 5 --eval-games 100 --eval-mcts-simulations 800 --ris-iterations 800 --lr 5e-5 --lr-schedule cosine --train-steps 1000 --replay-buffer-size 4000000 --data-dir $run --model-dir "$run/models" --log-dir "$run/logs"
```

batch1024、leaf dedup开、PCR关、镜像/翻面开是当前默认，执行前核对解析结果和manifest；新进程在on/k=0自动lazy开。
`--no-lazy-chance-allocation`仅回退对象分配；合法着回退用 `JIEQI_LEGAL_BACKEND=python`。不清空buffer、不加 `--reset-optimizer`、不热替换正在运行的进程。
本轮不修改用户训练参数或环境变量；也不把此命令的运行范围视为新训练授权。

### 3.4 必须同时报告的指标

- 速度：启动/热身/稳态分开，完整轮与自博弈段分开；秒/局、原始ply/s、普通/强局步数、实际叶输入/调用数。强局更长时，不能用游戏数比例直接估计每ply平均模拟预算。
- 统计：耗时下降 `1-Tnew/Told`；吞吐提升 `Told/Tnew-1`；配对均值与合计比例分别标注。少批区间仅描述本次波动，不当总体保证。
- 等价：编码及原始网络输出/批形状、合法着顺序、根policy/着法、规则字段、RNG、逐字段CompactSample、叶/chance计数、完整逻辑图与共享引用。
- 内存/资源：父进程+各worker的峰值RAM/显存、常驻buffer、树存活/释放；累计对象数不是峰值，Tensor.cpu时间不是纯D2H，评估时间不是纯GPU核时间。
- 学习：固定诊断和独立配对棋力分开；若改变算法/精度，需要学习对照，不凭loss、NaN缺失或一轮吞吐验收。

## 4. 当前保留的改动与版本控制

| 范围 | 保留实现 |
|---|---|
| `azero.py`编码/训练 | 原位编码、预分配NCHW输入/目标、精确只读查表，保留标量参考与旧样本兼容；不缓存4M展开张量 |
| `azero.py`叶评估 | 原批完整softmax、policy只读NumPy数组、Python float value；重复历史只在当前pending批内复用 |
| `jieqi.py`/`native/jieqi_rules.pyx` | 只读攻击编译、Python回退、API/规则/原生源哈希校验、实际加载文件身份 |
| `jieqi.py`/`native/jieqi_legal.pyx` | 合法着快路，保留禁着apply/undo慢路、生成器恢复时机、回调/非标准输入回退；默认Python |
| `native/jieqi_search.pyx`/加载器 | 可选扩展原型与方法指纹校验；不默认开启，生产不编译初始化 |
| 机会分配 | 不可变完整支持快照、稳定原枚举键、按世界采样后按需建对象；辅助补全保留共享引用/RNG，默认仅on/k=0生效 |
| 原型/测量 | `tools/central_inference.py`及独立验收、缓存/机会复用测量、图换位旁路与对应测试；主训练不依赖服务/缓存/图原型 |

已提交基线为三个私有历史快照（未随本仓库分发）；均不代表全部当前工作区。
筛选后的运行代码只保留在 `azero.py`：on/k=0 默认 lazy、BooleanOptionalAction 回退、CLI 解析/日志及叶去重数值边界说明。
相关测试覆盖默认解析、eager/lazy 物理树假设、共享决策/揭示、旧检查点续训；没有放宽数值/信息边界断言。
历史图研究工具/测试/文档、performance/tools 索引和审查报告保留；`analysis_longcheck.py` 作为独立分析脚本保留。
本次只添加本总结并更新文档入口；不提交/推送、不修改已有运行代码、不删除旧产物。
Git不保存被忽略的模型、4M数据、本机二进制或冻结审计输入，代码回退也不会回退训练数据。

已修复的审查问题见 [加速审查](2026-10-05-acceleration-code-audit.md)、[编译路径审查](2026-10-06-compiled-path-code-audit.md)；
仍保留F1基准可选二进制依赖问题，不能把“核心联合实现已通过审查”写成所有工具已无复现问题。
最近全量证据为2026-10-07本机三Cython后端 **472 passed/129.71s**；本次仅文档整理，未重跑全量。
沙箱临时写入/硬链接权限失败后来在正常本机临时目录重验通过，不改保存逻辑来迁就沙箱。

## 5. 重要约束

1. 当前目标是不改变既有训练结果的工程加速，不以降低预算、减少新数据或删规则/历史特征换速度。
2. 保留浮点计算顺序/Python float累加、动作平分顺序、完整前向/softmax与CUDA批大小/行顺序。相同单行输入不意味着不同批上下文输出逐位相同。
3. 所有合法动作边、完整机会支持、边际先验、模拟世界条件抽样和原RNG次数/顺序不变。lazy是对象表示，不是降低chance_k。
4. 行动者信息隔离、根条件采样、公开/私有历史、长将/长捉身份、重复计数与规则配置不可遗漏。相同棋盘不等于可合并搜索状态。
5. 不跨批复用可变信息态，不让数组行意外持有整批缓存或改value精度；估算缓存内存包括键、容器与输出副本。
6. worker调度须保持按任务索引排列数据/种子；改变并发组织仍需重验CUDA流与整局样本，不能因为游戏RNG相同就自动承诺等价。
7. 完整恢复AdamW、scheduler、RNG、计数和replay清单。规则v3旧目录不可直接在v5完整续训；冻结审计目录不可当完整训练目录。
8. 后端在导入时选择；二进制ABI/原生源/受保护方法过期时重建或回退，不绕过校验，不在长训中热替换。
9. 测量用新输出目录、冻结模型、独立随机流、硬件空闲与前后哈希；不往生产buffer加入探针对局，不用改变后的模型复用旧统计。
10. 精确性只覆盖有限测试域和本机软件栈，不是任意模型/平台的形式化保证；也不证明当前近似裁判或搜索具有均衡收敛保证。

## 6. 剩余瓶颈与未知

**已验证的瓶颈**：当前普通轮93.40%是自博弈，因此只加速训练更新的整体上限有限。
即使消除日志边界下全部非自博弈时间，普通轮最多省约6.60%时间；实际仅优化GPU更新会更少。

**过时但有价值的热点证据**：iter10、lazy尚未实施的低扰动探针中，expand占worker累计搜索33.50%/34.05%，叶评估35.91%/32.03%，合法着约11%。
函数边界有嵌套，包含等待且不是纯CPU/GPU时间；Cython小函数profile还有扰动。lazy显著减少对象后，不能把这些旧占比当作当前热点。[剖析](2026-10-05-compiled-hotspot-profile.md)。

**本阶段未确认**：当前完整组合的热点比例、GC/内存分配独占成本、CUDA排队/真实kernel占比、1024局长强局尾部占比、父进程样本IPC/压缩/增强分项、4M驻留的全过程峰值、worker最佳数量。
可见的真实轮后处理间隔说明存在预算，但不能直接认定是磁盘、内存带宽或GC导致。
4M输入全展开约343GB（十进制，仅FP32输入），不能用整buffer张量缓存作为本机方案。
GPU显存副本、context数量减少或更大batch是否会净省时间均需测量，不能由GPU利用率快照推出。
近期诊断value/KL仍波动；0.91训练内成绩和几轮loss不提供严格收敛/棋力结论，本次没有做学习诊断专项审查。

## 7. 下一步实验顺序与停止条件

| 顺序 | 独立任务 | 验收 / 停止条件 |
|---|---|---|
| P0 | 修复机会基准F1，仅保护实际加载/存在的可选文件，加缺失节点扩展回归 | 不修改生产算法；两必需扩展即可完成预检，其他ABI文件名不写死。尚未实施 |
| P1 | 正常重启时确认合法着快路启用；用冻结模型做Python/Cython配对，再观察生产归一化耗时 | manifest证实legal=Cython；输入/输出/策略/样本/RNG/逻辑图一致；生产日志只是补充，不代替配对 |
| P2 | 对lazy+合法着后的版本重做低扰动多局面profile，并补父进程后处理/常驻buffer峰值 | 区分等待与计算、普通与strong、累计与峰值；先定位当前热点，再选择局部精确优化 |
| P3 | 冻结模型扫描worker数，先4/6/8，内存余量充分才试12；固定每局种子/排序，冷热分开 | 同budget/真实完整局，多组平衡顺序；报告吞吐与总内存/尾延迟；等价失败、OOM或收益落在噪声内就不接入 |
| P4 | 只对新的实测热点尝试原位编码/分配复用或小范围Cython | 先微测再完整局；按现有精度/排序验收，不把新复杂度建立在旧33%占比上 |
| 低优先 | 原批集中推理补更多配对批次/稳态及服务故障、内存测量 | 逐请求数值门槛、可靠性和稳定收益同时通过；不附带合批/缓存变量。2组约2%暂不足 |
| 独立研究 | 图搜索补独立轨迹、深搜索/长循环根的有效换位；非零后再固定模型棋力，最后训练收敛 | 保守键不删信息/网络历史；重复来源复现、净成本值得才建原型。当前Gate A未过，后两阶段不启动 |

图搜索的后续预案见 [EXP-0003](../experiments/0003-graph-search.md)：先固定模型32配对通路，正式预注册400配对/800局、配对区间；
有棋力收益再做至少3组训练种子的独立tree/graph目录。它是算法实验，不混入正在续训的搜索或buffer。
AMP/PCR/合批/近似缓存如重新考虑，必须另设数值或学习门槛；本阶段未授权实施、长训或硬件升级实验。

## 8. 原始证据与本次整理验收

已确认下列本机文件存在；本次核对报告及关键元数据，没有把全部旧JSON逐记录重算，也没有重测旧结果：

| 原始文件 | 内容 |
|---|---|
| `data/pilots/perf_20261005/encoding_acceleration.json` | iter10编码/训练步/固定根 |
| `data/pilots/perf_20261005/selfplay_lowrisk.json`、`native_rules.json` | 输出/历史、攻击编译 |
| `data/pilots/perf_20261005/compiled_multi_profile.json`、`compiled_expansion_profile.json` | 编译后旧热点 |
| `data/pilots/compiled_paths_long_20261006_112409/results.json`、`.jsonl` | 节点/合法着组合长测 |
| `data/pilots/search_reuse_20261006_154246/results.json` | 访问率/输入重复与键成本 |
| `data/pilots/chance_allocation_20261006_162356/results.json` | lazy等价/吞吐/RSS |
| `data/pilots/network_cache_20261006_204415/results.json` | 输出数值门槛和LRU |
| `data/pilots/central_exact_20261006_211736/results.json`、`central_timing_20261006_212712/results.json`、`central_merge_20261006_214230/results.json` | 原批集中、吞吐与离线合批 |
| `data/pilots/graph_transpositions_20261007_iter22/results.json`、`.jsonl` | 图候选首测；同目录保留冻结22审计输入 |

本次文档整理验收：5份新增/更新文档的82个本地链接与行尾空白检查通过，`git diff --check`通过；
3个核心源码、用户分析脚本和5个生产产物共9个文件的SHA256前后一致，冻结iter22检查点哈希与记录一致。
未启动训练或性能基准，未重跑全量测试；不以历史472项记录冒充本日重跑。
完整4M peak、1024局单变量联合收益、当前叶去重净秒数、跨平台数值保证、等棋力训练所需数据量与目标硬件收益均未确认。
