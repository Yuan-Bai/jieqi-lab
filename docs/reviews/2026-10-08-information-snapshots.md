# 中间层信息态快照削减

日期：2026-10-08。范围：NeuralMCTS 中间层只为信息集键而生成的完整
`InformationState`。这是精确工程优化，不是新的图搜索算法。

**结论：保留。** 修正对照工具后，192局平均配对省时1.25%、合计省时1.34%，
描述性配对区间[0.77%,1.70%]，通过事先声明的吞吐门槛；96对样本/RNG精确一致。
固定根没有明确收益，不能把快照次数减少约68%当作同幅度的训练提速。

## 实现与约束

- `BeliefModel.information_set_key` 从当前模拟局面的公开字段和合成行动者账本
  直接生成不可变键，与原快照键共用一个字段格式化函数，保持原 15 字段/schema。
- NeuralMCTS 根快照不改；中间层不再复制棋盘、暗池、规则数组、重复计数字典
  与公共账本；有合法着的非终局叶仍生成独立完整快照供机会先验/编码/去重使用。
- 零步到叶保留原根快照，不能以确定化后新增的重复计数替代根输入。
- 决策键只读取 `for_simulation` 重建的行动者视图，不读真实对手私有槽，
  不用真实暗池作为决策键。键中的路径、棋盘和账本仍冻结为 bytes/tuple。
- 不删网络需要的最近历史/重复历史，不缓存可变局面或信息态，不合并节点；
  机会采样、概率、访问量、回传路径、网络批次与 RNG 运算不改。
- 本轮不把快照削减接入 RIS rollout 搜索，其原快照流程保留。
- 不修改语义版本、训练配置、精度、预算、样本排序、buffer 或 optimizer。
  没有新训练参数；纯 Python 也可使用，不涉及原生受保护方法，无需重建扩展。

等价性依据：旧中间快照只用于生成键，之后不再消费；其复制不消耗RNG或做概率运算。
直接键对同一公开状态和同一合成账本执行原bytes/tuple转换，复用同一字段格式。
叶延迟复制依赖现有合法着枚举的apply/undo完整恢复约定；本轮以边界单测、
真实局面完整网络流和整局样本对照验证该约定，没有弱化历史/隐私或数值断言。

## 对照与环境

控制不是“候选多调一次 `info_state`”：保存改动前 `azero.py` 和 `ismcts.py`，
只提取原 `NeuralMCTS.search` 与原 `information_set_key` 的 AST 函数执行。
它在每层生成完整快照，并在叶复用最后那份快照，没有人为增加叶复制。
执行全局名称连接当前未改的模型/机会/规则等辅助函数；不执行旧模块导入。
旧模块名称空间使用真正的 `ModuleType`，保持CPython的模块属性访问路径。

沿用同机 i9-13900HX、RTX 4060 Laptop 8GB、约32GB RAM，Python3.11.5、
torch2.6.0+cu118、NumPy2.0.2，冻结 iteration28 模型与18根 fixture。
rules/legal=Cython、nodes=Python；8 worker，每进程torch CPU threads=1；
800/3200模拟、强局间隔8、leaf batch64、dedup开、chance on/k0/lazy、PCR关。
FP32张量、matmul TF32关、cuDNN TF32开，设置不变。

**两种模式都保留已经验收的完整机会表缓存**，每根/局均清空，避免把其收益
混入本次优化。冻结输入沿用前一实验的只读副本，只含最新replay，不是完整续训buffer。
当前真实训练目录最新manifest仍legal=Python；本轮以用户下次计划启用的legal=Cython测量。

先验协议保存在 `data/pilots/information_snapshots_20261008_iter28/protocol.md`：
36根/预算图与网络流精确检查，八组固定根，再六组×16局/模式，共192局。
模式顺序交替，新种子；整局门槛为平均配对省时至少1%、合计耗时下降、
描述性配对cohort bootstrap下界大于零且全部精确检查通过。不基于中途结果提前结束。

## 已完成检查

| 项目 | 结果 |
|---|---|
| 定向信息集/揭示/机会/快照回归 | 42 passed / 19.32s |
| 补充终局/无合法着、真实模块访问边界后新增专项 | 10 passed / 5.74s |
| 36根/预算的完整图、每次网络输入输出、策略/走子、计数、RNG | 两模式全部精确一致 |
| 修正对照后36条件完整快照总次数 | 201,539 → 64,486，减少68.0032% |
| 八组固定根，288对搜索 | 全部精确一致 |
| 修正后固定根平均配对省时 | -0.44112%，描述性区间[-1.97306%,+0.79646%]，无明确收益 |
| 修正后固定根墙钟合计 | 原快照50.11796s / 直接键50.32297s，合计慢0.40906% |

计数与网络/完整图插桩在吞吐计时之外，不能用快照减少68%宣称自博弈快68%。
固定根预算普通/强各半，也不能当作生产强局7:1的整轮加速率。

首版对照用 `SimpleNamespace` 承载旧 `ismcts` 名称，可能改变CPython属性访问成本。
两组整局配对后为修正测量方法而中止，并保留部分JSONL/console，不能用于保留门槛。
精确性结果仍有效；已另声明 `protocol_module_reference.md`，仅修正为真实模块对象，
候选不变、门槛不变，新seed offset=10,000，从头完成全套图/固定根/192局验收。
不混合两轮耗时或根据观察到的计时结果调低门槛。
首版已记录64局完整结果，其余中止任务未统计，不计入下面的192局验收。
两次测量的种子/机器状态也不同，不能把性能点估计变化全归因于模块对象修正。

## 整局与全量回归

最终六组，每模式每组16局，84普通+12强局/模式，合计96对、192局。
先做完整图/网络流，再固定根，最后整局；没有基于中途计时提前结束或改变门槛。

| 指标 | 修正对照后的结果 |
|---|---|
| 整局精确性 | 96对compact样本、策略/历史字段、局长、胜负、叶数、最终RNG全一致 |
| 每模式总ply / 样本 / 叶评估请求 | 4,993 / 4,993 / 5,112,394 |
| 六组原快照/直接键墙钟合计 | 724.89670s / 715.15457s |
| 平均配对整局省时 | 1.24920% |
| 合计整局省时 / 吞吐比 | 1.34393% / 1.013622 |
| 描述性配对cohort bootstrap 95%区间 | [0.77489%,1.70455%] |
| 每组省时，按声明顺序 | 0.81565%、1.16917%、0.33986%、1.86911%、2.01846%、1.28297% |
| worker内部墙钟合计 | 3,200.80529s / 3,185.01857s，省约0.4932% |
| worker process CPU合计 | 3,069.4375s / 3,044.7500s，省约0.8043% |
| 最终验收总用时 | 1,715.04s，约28.58min，不含被中止的首版测量 |
| 来源保护 | 27个核心/原生/工具/输入/协议/产物哈希全部未变 |

门槛三项通过、六组均较快；保留候选，但收益级别只有约1%。cohort端到端墙钟与
worker合计并非同一口径，包含不同尾部/调度影响；没有分段归因，不把差额归因到某个算子。
不是大幅提速，也不保证未来模型、长强局尾部、机器负载或1024局生产轮都有同样收益。

原生产checkpoint/latest/state/log/最新replay的SHA256与改动前一致；其它replay
没有写入路径，但未逐文件哈希。规则/合法着二进制未改，源/方法身份校验有效。

推荐后端rules/legal=Cython、nodes=Python的全量回归：**512 passed / 142.77s**。
三个后端强制Python的快照/信息集/揭示/机会/自博弈/图审计回退专项：
**70 passed、1项原生防护skip / 19.10s**。使用正常系统临时目录支持已有原子写入/硬链接测试。

首跑全量511 passed、1 failed：原只读图测量工具只挂钩快照键，漏记新的中间直接键。
已适配 `TranspositionAudit.install` 同时观察两个入口，只有审计时重建完整快照以记录
规则/网络输入证据；生产不引入额外复制。增强异常后双挂钩恢复及非根决策覆盖断言，
适配后定向30 passed / 6.54s，再完成上述全量。未放宽精确性断言；失败日志仍保留。
全量结束后再次复核27个保护哈希，变化数0。

## 保留与续训

- `ismcts.py`：新增直接行动者键，完整信息态入口与它共用原字段格式。
- `azero.py`：中间层用直接键，仅根和非终局合法叶生成完整快照。
- `test_information_snapshots.py`：全部键字段、私有信息隔离、不可变性、
  两色向量投影、叶独立性、零深度、终局/无合法着、对照工具真实性。
- 新测量工具、协议、原始结果与文档入口；前后核心源码另存在隔离目录中。
- 只读图换位测量工具同步适配直接键入口，不合并节点或修改任何搜索统计。
- 不改变概率、网络批次、浮点设置、模拟预算、搜索/数据版本或恢复格式。
  无需新参数、重训、清buffer或重置optimizer；新进程导入后生效，不热切换。
- 未启动生产训练、棋力/收敛实验或Git提交。本轮未测RIS吞吐或4M buffer峰值。

## 复现

```powershell
$env:JIEQI_RULES_BACKEND='cython'
$env:JIEQI_LEGAL_BACKEND='cython'
$env:JIEQI_SEARCH_BACKEND='python'
python tools/benchmark_information_snapshots.py --run-dir data/pilots/chance_support_20261008_iter28/frozen_inputs --before data/pilots/information_snapshots_20261008_iter28/before --fixtures data/pilots/chance_support_20261008_iter28/measure.fixtures.pkl --output data/pilots/information_snapshots_20261008_iter28/recheck.json --seed-offset 10000 --workers 8 --search-rounds 8 --game-rounds 6 --games-per-round 16
python -m pytest -q tests/test_information_snapshots.py tests/test_search_information_sets.py tests/test_search_reveals.py tests/test_chance_support.py tests/test_chance_allocation.py
```

仅加载本机可信fixture pickle；旧来源快照、冻结输入与本机有效二进制是复现条件，
工具源码本身不包含这些被忽略的产物。输出必须用新文件，避免覆盖旧结果。

`validate_module_reference.json`/`.jsonl` 保存原始逐根/逐局结果、配置、源码与输入/二进制哈希；
工具保护生产checkpoint/latest/state/log/最新replay、冻结副本、核心/原生源码、
二进制、依赖工具、fixture与旧函数来源，结束时复查未变。
`before/`保留改动前两份核心源码，`candidate/`保留候选源码和测量工具。
`regression_native.log`保留首次失败，`regression_native_final.log`和`regression_python.log`
保存最终回归结果；候选与图插桩适配的测试源码也另存于`candidate/`。

## 证据边界

这是有限冻结模型/输入/种子下的精确对照，不是所有未来轨迹的数学证明；
没有更改现有CUDA非确定性设置。小批cohort的bootstrap仅描述本机测量。
不含优化器、完整样本保存/数据增强；cohort墙钟包含样本摘要及compact IPC。
未启动生产训练或棋力/收敛评估，不宣称完成1024局生产轮验收。
