# ADR-0007：评估协议（配对、Wilson CI、JSONL、不门控）

> 当前实现：协议v2使用双方/环境独立RNG；统计v2优先整对bootstrap区间。
> 2026-10-22后历史池接入训练，默认只报告，RIS的65%门槛仅保留为历史记录。
> 下文原始决定及独立接入时点不代表最新训练政策，见[ADR-0021](0021-opponent-pool-and-selection.md)。

- 状态：已接受
- 日期：2026-09-11
- 相关：需求访谈 Q8；实现 `tools/eval.py`

## 背景

揭棋含随机洗牌（暗子组成）与随机揭示，逐局胜负噪声大。若双方各用不同洗牌，强度差异会被洗牌运气掩盖；单色固定先后又引入先后手偏差。需要可复现、成对、带统计区间的评估协议。

## 决策

1. **配对评估**：同一初始洗牌（固定 seed）双方各执红/黑一局，成对统计；`games` 向上取偶。每局双方（NN 与对手）共用同一 rng，保证同 seed 可复现。
2. **两档**：
   - 快评：50 局 @400 sims vs `ris:400`（信号检查，不作为验收）；
   - 验收：200 局 @800 sims vs `ris:800`（+ 随机基线 `random`），验收线 **≥65%**。
3. **统计**：得分 = (胜 + 0.5 平) / 总局；Wilson 95% 置信区间随 summary 一并写入。
4. **结果格式**：JSONL，逐行 `type` 为 `run` / `game` / `summary`；每局记录 seed、neural_side、outcome、reason、plies、score；超 `max_plies`（默认 600）记 `ply_limit`（score 不计）。
5. **固定评估种子集**：默认 `seed=20260911`；正式对比必须显式记录（写入 run 行）。
6. **不做晋升门控**：`latest.pt` 始终参与自博弈；`best.pt` 按验收分更新（`train_state.json.best_score`）。训练循环内评估追加写入 `data/eval/train_eval.jsonl`。
7. **对手接口**：`random` / `ris:N` / `model:path[:sims]`，便于历史版本自对比。

## 后果

- 报告的比分必须附 CI 与局数，禁止只报点估计。
- 任何评估结果可用 `--seed` 与 JSONL 完整复现。

## 2026-10-22：独立评估工具补充

- 原 §1 的共享 RNG 描述已由评估协议 v2 取代：环境、双方搜索各用独立 RNG，
  同 seed 换色共享预物化洗牌；一局未终局则剔除整对。
- `tools/eval.py` 新增冻结历史模型池的配对 bootstrap 摘要及 `pika:budget` 对手。
  `tools/paired_stats.py` 的整体和颜色统计同步只使用完整有效配对。
- 旧 Wilson 字段保留兼容，但优先报告 `bootstrap_ci95`；零方差/样本不足必须标明。
- Pika 使用观察者计数信念及项目根合法着；内部历史规则与项目不完全相同。
- 训练循环、最新模型自博弈及 `best.pt` 的原更新逻辑不由此补充改变。

完整协议、参数与限制见 [独立对手池评估](../evaluation.md)。
