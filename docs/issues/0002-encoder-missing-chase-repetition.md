# Issue 0002：网络输入缺长捉/重复等公开规则状态（审查项 R2）

- 状态：**已修复并补全（2026-10-01）**；当前编码 238 通道、`DATA_VERSION=3`，保留完整相关公开重复历史。旧模型补零加载，v2 数据可读并标记历史缺失，v1 拒绝；见 [ADR-0013](../adr/0013-rule-history-encoding.md)。下文保留最初 64→71 通道修复的背景。
- 来源：搜索信息泄露审查（2026-09）遗留风险 R2
- 关联：`azero.py:177-246`（64 通道编码）、`azero.py:258-287`（样本）、
  `jieqi.py:617-708`（规则状态维护/禁着）、`docs/adr/0003`、`docs/adr/0004`

## 问题

`encode_state_numpy` 的 64 通道只含规则相关平面：`no_capture_plies`(45)、
`consecutive_checks`(46-47)。以下公开规则状态**不进入编码**：

| 字段 | GameState | InformationState | 样本 | 编码 |
|---|---|---|---|---|
| consecutive_checks | ✓ | ✓ | ✓ | ✓ 46-47 |
| check_anchor | ✓ | ✓ | ✗ | ✗ |
| chase_streak / target / target_code / anchor | ✓ | ✓ | ✗ | ✗ |
| position_counts（当前局面重复次数） | ✓ | ✓ | ✗ | ✗ |
| no_capture_plies | ✓ | ✓ | ✓ | ✓ 45 |

这些均为双方可见的公共信息（`info_state` 对两观察者拷贝同一份；ADR-0003/0004 将
长将/长捉/重复定义为公开行为；`to_fen` 第 7 字段已全量序列化这些状态）。
`compact_to_tensors` 重建 `InformationState` 时这些字段回落默认值，训练无法接触。

## 影响

- **搜索：不受影响。** 禁着过滤在 `legal_moves()`（`jieqi.py:924-942`）与终局
  `game_outcome()` 中精确处理；`PUCTNode.expand` 只在合法着法集合上归一化先验
  （`azero.py:694-729`），被禁着法永不进树、不进策略目标。
- **训练：输入对规则不 Markov。** 长捉需 12 ply 历史而输入历史仅 8 ply，
  网络看不见 `streak→6` 的临禁；`position_counts` 缺失使重复判和压力不可见
  （短周期重复可由 history 平面间接暴露，跨窗口次数不可知）。后果：
  临禁局面的策略目标支持集依赖不可见历史（梯度近似噪声）；重复/无吃子和棋的
  value 标签在关键输入缺失下产生估值偏差（平面 45 可部分代偿）。
- **频率实测**（agent B）：随机 300 局/88609 ply 中 chase≥1 占 18.8%、
  ≥4 仅 0.03%；900 局强评估终局 reason：repetition 9、no_capture 12、
  long_chase/long_check 0。→ 实际是残局/和棋估值的小幅强度税，非非法着法 bug。
- 该缺陷在现有测试中无法暴露（搜索侧掩码自消；无输入完备性测试）。

## 解决思路（按推荐序）

1. **补样本字段 + 扩通道 64→71（推荐）**
   - `RawSample`/`CompactSample` 增加 `check_anchor/chase_streak/chase_target/
     chase_target_code/chase_anchor/rep_count`；
   - 编码追加：`rep_count/3`、`chase_streak/6`×2、`chase_target` 空间平面×2、
     `chase_anchor` 空间平面×2（`check_anchor` 供 value 侧参考）；
   - `mirror_sample` 必须镜像空间字段（`chase_target/chase_anchor/check_anchor`），
     `chase_target_code` 为棋子码不镜像；
   - 兼容性：`DATA_VERSION 1→2` 并在 `load_compact_payload` 校验（当前 `version`
     写了不读，`azero.py:31/1076/1118-1126`）；旧数据（~560 文件/202.6MB）字段缺失
     需归档并重新生成；模型仅首层 `conv.weight` 形状变（`[C,64,3,3]→[C,71,3,3]`），
     可 pad-load `latest.pt/best.pt` 后热启动，无需从零重训；规则语义未变，
     `ruleset_version` 不必升级。
2. **只补 value 侧**：规则特征走独立小 MLP 接 value head，主体 64 通道不动 →
   旧模型可直接加载，但策略仍盲，修复不完整。
3. **不改架构**：把缺失登记为接受缺口，靠搜索掩码与 `no_capture` 平面代偿；零成本。

## 验证计划（修复时）

- 构造 `chase_streak=5 vs 0`、`rep=2 vs 1` 仅此差异的两局面，断言编码不同
  （当前实测编码逐元素相同，可作回归测试）；
- A/B：同 checkpoint 分别 64/71 通道微调同数据，比较 Gate 曲线（先行小样本）。

## 不确定性

- 未做 A/B，无法断言 Elo 损失幅度；8 ply history 与 `no_capture` 平面可能已部分代偿。
- 自博弈长捉意图可能略高于随机统计，但强评估 900 局 0 次 `long_chase` 支持"罕见"。
