# Issue 0004：chance 节点虚拟损失永不冲销（已修复，登记核验）

- 状态：**已修复（私有工作区 azero.py，未随本仓库分发）**，本文件记录核验结论与历史误判纠正
- 来源：多 agent 分析 R3 时发现的独立问题（agent C 实测）
- 关联：`azero.py:950-968`（机会节点选择与 path 构造）、`azero.py:846-862`（`_backup`）

## 问题 A：机会节点 VL 不冲销（真实，已修复）

修复前：进入机会节点时对机会节点与揭示子节点各加 `virtual_loss`，但 `path` 只
append 揭示子节点 → 机会节点 VL 永不冲销，`q ≡ virtual_loss`（复核实测 1413 个
机会节点中已访问者 `value_sum == visits * 3.0`，Stage 2A 实质失效）。

修复（工作区 diff）：把机会节点也 append 进 `path`（`azero.py:966-968`）。
核验（探针 `probe_visits_recheck.py`，sims=32/batch=8、sims=16/batch=4）：
已访问机会节点 q 分布 [0.0010,0.0092] / [-0.0046,-0.0007]，不再恒为 3.0；
`_backup` 的“同视角层不翻转”约定（`azero.py:861`）保证了机会节点与揭示子节点
间的价值传递正确。

## 问题 B：root.visits 翻倍（**此前误判，已纠正**）

此前静态推断“选择阶段 `child.visits += 1` 与 `_backup` 双重计数导致 root.visits
≈2×sims”。实测证伪：

| 配置 | root.visits | Σchild.visits | sims |
|---|---|---|---|
| 32/batch=8 | 32 | 32 | 32 |
| 32/batch=1 | 32 | 32 | 32 |
| 64/batch=16 | 64 | 64 | 64 |
| 32/batch=8（chance） | 32 | 32 | 32 |

原因：每次模拟选择阶段最多只把 root 的**一个**孩子 `visits += 1`；`_backup` 中
root（`i==0`）恰好也加 1，两者合计每模拟 root 增量 = 1（同孩子的选举计数 +1 由
`_backup` 的 i>0 分支冲正）。此前把“选择阶段对路径节点的 +1”与“_backup 对同一
批节点的 +1”误判为双重计数，忽视了选择阶段每模拟只覆盖单一子树这一事实。
GUI 胜率显示（`root.value_sum / root.visits`）口径正常。

## 回归建议

- 补一条“机会节点 q 不恒等于 virtual_loss”的回归测试（修复前必失败）；
- 补一条 `root.visits == sims 且 Σchild.visits == sims` 的搜索统计测试
  （chance 开关两种配置）。
