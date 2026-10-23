# Issue 0005：机会节点回退污染（已修复，2026-09-17）

> 当前补充（2026-10-01）：固定缓存支持不足也已处理；完整世界抽样后按需补建，`chance_k` 只限制初始预建，见 [ADR-0014](../adr/0014-lazy-chance-support.md)。下文保留当时的修复证据。

- 状态：**已修复（私有工作区 azero.py，未随本仓库分发）**，本文件记录问题、修复语义与验证证据
- 来源：机会节点数学审查（主实验前置检查）
- 关联：`azero.py` 的 `NeuralMCTS._navigate_chance` / `_apply_chance_move` / search 机会链；`test_azero.py` 回归测试；探针 `tools/probe_chance.py`

## 问题：边际先验独立采样 → 世界与 reveal 分裂 → 回退污染

修复前 search 在机会链上逐层调用 `PUCTNode.sample_outcome()`：按存储的**边际先验**
（expand 时用 `pool_known` 上界归一化写入）采样揭示类型 t，再由 `_apply_chance_move`
在当前模拟世界池中强制 reveal t。两条关键事实：

1. `pool_known` 归一化确实是"根信念下 reveal 的**边际**分布"——探针验证
   `E_world[pool[t]/total] == pool_known[t]/Σpool_known`（missing=4/8 均在 MC 噪声内），
   即分支概率本身没有数学错误。
2. 但模拟中世界 θ 与采样 t 是**独立**的；当 t 在 θ 的池中不存在（该类型已被未知被吃），
   `_apply_chance_move` 静默回退自然抽样，样本的 reveal 类型 ≠ 分支 t，却被计入 t 分支，
   导致 E[V|a,t] 的条件化被污染，且污染率随 `unknown_removals` 增大。

探针实测（初始局面，单次 reveal 采样的回退概率；分支不可用率 ≈ 世界池中该类型缺失率）：

| missing（未知被吃数） | 单次 reveal 回退概率 | 各分支不可用率（车/马/象/士/炮） |
|---|---|---|
| 2 | 0.66% | ~1% |
| 4 | 3.90% | ~5.8% |
| 8 | 18.37% | ~26.5% |
| 12 | 50.77% | ~62.7% |

真实搜索（初始局面 + missing=6，200 sims）回退 **14/247 = 5.67%**。

## 修复：世界条件采样（world-conditional sampling）

搜索机会链不再用 `sample_outcome`（边际先验），改为 `NeuralMCTS._navigate_chance()`：

- 从**当前模拟世界**的池计数采样 reveal 类型（第一层用移动方池；双揭示第二层用被吃方池，
  两池异色、独立）；
- 导航到树中对应分支；全枚举（`chance_k=0`）时目标分支必然存在；
- 每条模拟的 (θ, reveal) 联合分布与真实规则一致，分支 value 的样本条件化正确。

配套改动：

- `_apply_chance_move` 的可用性检查保留为防御性兜底（世界条件采样命中时目标类型必然可用）；
- 导航 miss（树信念与世界池支持集不一致，见下节）时按世界自然抽样并直接进入叶评估，
  计入 `chance_nav_miss`；
- `sample_outcome` 保留（边际分布采样工具 + 概率回归测试），不再参与搜索导航；
- 新增搜索计数器 `chance_applied` / `chance_nav_miss`（主实验报告用）；
- 机会分支先验（prior）仍为边际概率，作用于着法节点对机会节点的 UCT 探索项与展示。

## 验证

- 新增 4 个单测（`test_azero.py`）：
  - `test_navigate_chance_follows_world_pool`：受控世界池（红只有车）下导航必须选车；
  - `test_navigate_chance_weighted_by_world_counts`：车:卒=3:1 时采样频率 ≈75%；
  - `test_navigate_chance_double_reveal_two_layers`：双揭示逐层用对应色池，visited 含中间层；
  - `test_search_chance_never_falls_back_with_full_enum`：missing=6 全枚举搜索中
    reveal 类型始终可用、`chance_nav_miss==0`。
- 全量：`test_azero.py` **76 passed**；`test_eval.py + test_ismcts.py + test_jieqi.py` **50 passed**。
- 探针复验（`tools/probe_chance.py` 第 4 段）：修复后 missing=0 / missing=6 的搜索回退均为 **0 次**
  （修复前 missing=6 为 5.67%）。

## 修复后回归：导航 miss 的树结构缺陷（同日发现并修复）

主实验首跑在 pair 5 崩溃：`TypeError: cannot unpack non-iterable int object`
（`azero.py` search → `sim_state.apply_move(move)`，传入的 move 是 int）。根因：

- 机会节点由**先前的叶展开**按当时模拟的信念建立（`leaf_info.pool_known`），
  而每条模拟的世界池由 `root_info.determinize` 独立采样（含 missing 扣除）加各自路径上的
  揭示消耗决定，二者支持集可能不相交：某条模拟走到既有机会节点 C 时，C 的所有分支
  类型在该世界池中计数均为 0 → `_navigate_chance` 返回 `(None, [])`。
- 原 miss 处理继续 while 导航：下一轮对机会节点 `select_child`，其 children 键是
  **揭示类型（int）**而非着法 `(from, to)`，被当作普通着法传给 `apply_move` → 崩溃。
- 该路径与 `chance_k` 无关（全枚举同样发生），是树信念演化不一致的固有现象，
  与旧版 `_apply_chance_move` 的池不足回退是同一"世界-树不匹配"现象的不同表现。

修复：miss 时按当前世界自然抽样揭示（此时揭示类型必不在树分支中——否则不会 miss），
记录路径后 **break 出导航循环**直接进入叶评估；机会节点此后不再被 `select_child`，
缺失分支的访问由叶评估展开补偿。

回归测试：`test_search_chance_nav_miss_does_not_crash`（patch `_navigate_chance`
恒返回 `(None, [])`，断言搜索完成、`chance_nav_miss>0`、`chance_applied==0`）。

验证：全量 `test_azero.py + test_eval.py + test_ismcts.py + test_jieqi.py` **130 passed**；
40 局 @400 sims workers=8 真实评估（非对称 chance）无崩溃，on 侧 0.537（W17/D9/L14）。

## 语义与兼容性

- `missing=0`（开局）修复前后等价（边际与世界条件分布一致）；`missing>0`（中残局）
  修复前有偏、修复后无偏。
- 修复改变了机会节点搜索的采样流与语义：修复前的评估/实验结论（如 `data/ab_short`
  的 2x2-chance 批次）仅对修复前版本有效，不可与修复后数据混用。
- 主实验（chance on/off 配对 A/B）在修复后版本进行。
