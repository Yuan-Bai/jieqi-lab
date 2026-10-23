# Issue 0001：确定化未显式使用 unknown_removals（审查项 R1）

- 状态：已分析，**证伪**（当前实现正确）；仅建议加固
- 来源：搜索信息泄露审查（2026-09）遗留风险 R1
- 关联：`ismcts.py:41-55`、`ismcts.py:89-119`、`azero.py:196-210`

## 原始疑点

`InformationState.determinize()` 在确定化时不读取 `unknown_removals`，而是以
`missing = sum(pool_known) - pool_total_known` 作为"未知被吃数"，再由
`sample_unknown_removals()` 抽走。疑点：这样是否把"未知被吃"当作**按类型均匀**
抽签，从而偏离真后验（某类型被吃概率应正比于其剩余数量）？

## 分析结论（证伪）

`sample_unknown_removals` 的实现是：

```python
roll = rng.randrange(total)      # 按“棋子”均匀，而非按类型均匀
acc += counts[p_type]            # 命中概率 = counts[t]/total
counts[p_type] -= 1; total -= 1  # 多步无放回
```

多步无放回 ⇒ **按剩余计数加权的多元超几何分布**，即
`P(u | 已揭示 k) = ∏_t C(C0_t-k_t, u_t) / C(Σ(C0-k), m)` 的精确实现。
"初始先验×似然"化简后与之相同；反而"按 POOL_INIT 抽签"才是错的。

实验证据（agent A，脚本见临时目录 `probe` 系列）：

| 场景 | TV(经验分布, 超几何) | TV(经验分布, POOL_INIT 先验) |
|---|---|---|
| 已揭示 1 车 1 马，m=3 | 0.0038（噪声≈0.005） | 0.2125 |
| 已揭示 2 车 2 马 1 炮，m=4 | 0.0030 | 0.7846 |
| m=1 | 0.0018 | 0.1119 |

独立真值（初始池洗牌 + 条件化拒绝采样）与超几何的 TV = 0.0128（n=3e4，噪声≈0.017），
端到端 `BeliefModel→info_state→determinize` 同分布。`sum(pool_known)-pool_total == unknown_removals`
恒成立（`test_azero.py` 已有断言），故不读 `unknown_removals` 不影响数值。

真实近似在更上层：belief 不条件化对手行动历史/对手模型；不属本项。

## 待办（不修复，登记）

1. 在 `sample_unknown_removals`/`determinize` 补 docstring：说明"按剩余计数无放回 =
   多元超几何精确后验"，避免再次误报。
2. （可选加固）把 `missing` 改为显式读 `self.unknown_removals[side]` 并断言与
   `sum(pool_known)-pool_total` 一致；数值与 RNG 消耗不变，可复现性不受影响。
3. 禁止改为按 `POOL_INIT` 先验抽签（会引入 TV 0.11~0.78 的系统偏差）。

## 不确定性

- 交换性假设（揭示位置与隐藏类型独立）在对手据私知调整走法时会被上层打破；
  当前采样仍是该模型内的正确实现。
- 外部构造 `pool_total != sum(pool_known)` 且 `unknown_removals=0` 的 belief 会被静默接受
  （加固项 2 的断言可暴露）。
