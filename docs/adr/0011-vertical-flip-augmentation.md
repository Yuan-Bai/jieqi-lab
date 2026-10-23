# ADR-0011：楚河汉界翻面数据增强（行翻面 + 红黑互换 + 暗池互换）

- 状态：已接受
- 日期：2026-09-19
- 相关：ADR-0008（数据与版本）、ADR-0005（训练路线）；验证测试 `test_azero.py::test_flip_vertical_lockstep_with_engine`；实现 `azero.py` 翻面增强函数组

## 背景

> 2026-10-03 审查更正：下文“不影响合法着法/胜负”的推断已被默认规则下的确定性禁着反例推翻。左右镜像和翻面都能使长捉计数在 12 步后变成 6 对 5，随后合法动作集合不同；字段变换正确不等于规则转移严格同构。原随机回放记录保留为历史证据，修复与验收待办见[本次审查](../reviews/2026-10-03-augmentation-audit.md)。

原有数据增强只有左右镜像（`augment_mirror`，每样本 ×2）。棋子走法与棋盘几何允许的完整「棋盘 + 颜色」对称群是 Klein 四元群：

1. 恒等；
2. 左右镜像（列 `c→8−c`，颜色不变）；
3. **行翻面（楚河汉界翻转，`r→9−r`）+ 红黑互换**；
4. 180°旋转 + 红黑互换（= 2∘3 的复合）。

其中第 3 个元素此前缺失。单独行翻面不是对称——黑在上红在下、兵的前进方向、九宫归属、过河判定都按颜色区分——必须与「红黑互换 + 暗池互换」绑定。

## 决策

在 `azero.py` 实现翻面增强，并接入全部样本入口：

- 函数组（`azero.py:627–755`）：`flip_vertical_index/move/square`、`swap_piece_color`、`flip_vertical_board/history/policy/sample(s)`、`augment_flip_vertical`；`augment_samples(mirror, flip)` 统一组合语义。
- `AZConfig.augment_flip=True`（默认开，与 `augment_mirror` 一致）+ CLI `--no-augment-flip`；训练循环、续训 `load_selfplay_data`、`merge_selfplay_files`（`merge_selfplay.py --augment-flip`）共用 `augment_samples`。
- 两个增强同开时每样本 4 份（原样 / 镜像 / 翻面 / 镜像∘翻面），恰好覆盖完整对称群；只开一个时 ×2。续训自适应加载预算按 `(2 if augment else 1) * (2 if augment_flip else 1)` 计。
- 变换清单：棋盘棋子位置套 σ 且颜色位取反；`side` 互换；`pool_known / pool_total_known / unknown_removals` 两色槽互换；`consecutive_checks / chase_streak` 槽互换；`check_anchor / chase_target / chase_anchor` 槽互换 + 格子套 σ；`chase_target_code` 槽互换 + **颜色位翻转**；`history / policy_idx` 动作套 σ；`value / policy_prob / no_capture_plies / rep_count` 不变。
- 实现要点（易错）：`chase_target_code` 存的是含颜色位的完整棋子码（`jieqi._update_chase`，jieqi.py:731），交换颜色槽后必须同时翻转颜色位，否则翻面样本的续捉立即断开；`value` 是「轮到走棋方」视角，颜色互换后数值不变（不取负）。

## 验证（数学 + 多 agent + 实证）

- 数学：T = 左右镜像 ∘ (180°旋转+红黑互换)，两个因子均为博弈自同构，故 T 为自同构：着法合法性、照面、九宫/河界、观察结构（含「被吃暗子只向吃方揭示」的不对称信息）、value/policy 标签语义均精确等变。
- 多 agent（3 独立 + 1 裁判）：几何/着法分区 217,895 项断言零失败；规则时间演化与信息结构分区逐字段核对；对抗红队未推翻核心命题。
- 实证 lockstep（原局与翻转局同步走映射着法，逐字段比对）：
  - 本机 500 局 70,805 ply：`legal_moves`、`game_outcome`、value 语义**零偏差**；
  - 改用正式实现后 200 局 28,621 ply 复验：样本/信息态/张量偏差数（112）与状态偏差数（112）完全相等，即实现未引入额外偏差。
  - 复现工具：`python tools/verify_flip_symmetry.py --games 200`（退出码 0=通过；预期 chase 记账近似出现但合法着法/胜负/样本层不放大）。
- 已知稀有近似：引擎两处索引序 tie-break 与 σ 不同变——`_update_check` 多子将军取 `min(checkers)`（jieqi.py:685）、`_update_chase` 同值目标取扫描序首个（jieqi.py:715–732）。翻面后可能追踪另一枚**等价值**棋子（实测 0.33%/ply）；裁判在 `control-128` 归档 8192 局 509,298 ply 回放中确认**禁着状态差异 0**（不改合法着法/胜负），且既有左右镜像存在同类问题，T 不引入新风险类别。该近似只影响 chase 记账字段，已在 `flip_vertical_sample` docstring 记录；修复需改规则语义（升 `RULESET_VERSION`、作废数据），成本不成比例。
- 单测：新增 6 个（逐字段变换、对合、群大小、60 ply 引擎 lockstep、加载 ×2/×4、容量预算 ×4）；全量 **141 passed**（test_azero + test_eval + test_ismcts + test_jieqi）。

## 后果

- 默认训练样本量由 ×2 变 ×4；`replay_buffer_size` 固定时 buffer 内**不同对局数量变为 1/4**（冗余 vs 对称覆盖的权衡）。消融用 `--no-augment-flip` / `--no-augment-mirror`。
- 网络编码是绝对颜色平面（0–27 按色分组、通道 28=轮到红方、动作空间绝对坐标），无内建颜色对称，因此翻面样本对网络是真信息而非冗余副本；若未来改为按走棋方规范化编码（永远把走棋方画在下方），此增强才退化为冗余。
- T(初始局面) 是「同盘面、黑先」，从红先标准根不可达，但它是规则下的合法局面且标签由同构精确给出，不影响训练语义；是否划算（×4 冗余 vs 对局多样性）建议按 ADR-0005 的 A/B 协议用 `--no-augment-flip` 消融判读。
