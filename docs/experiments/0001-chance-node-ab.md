# 实验 0001：机会节点 on/off 配对 A/B（已运行，2026-09-17）

> 历史实验：本次所用 off 搜索、私有信息处理和编码均早于 2026-10-01 修复。保留原始结果，不能据此判断当前 on/off 棋力或配置；当前状态见 [status.md](../status.md)。

- 状态：**已运行并出结论**（私有工作区快照，未随本仓库分发）
- 模型：`data/models/v3_iter560.pt`（冻结，两侧同一模型）
- 契约：评估协议沿用 ADR-0007（配对同洗牌红黑互换、未终局不计分、默认 seed 20260911、max_plies 600、评估贪心温度 1e-6）
- 前置：本实验全部在 Issue 0005（世界条件采样）修复后的代码上运行；修复前的 chance 结果不参与对比
- 数据：`data/eval/chance_ab/`（`s400_n1600` / `s800_n800` / `s400on_vs_off492_n1600`，各含 `.jsonl` 与 `.stats.json`）
- 工具：`tools/eval.py`（对手独立 chance 开关）、`tools/paired_stats.py`、`tools/chance_variance_probe.py`

## 1. 问题与假设

NeuralMCTS 的机会节点把暗子揭示显式建模为随机分支（`chance_k=0` 全枚举；分支先验为根信念边际概率）。
理论上（Rao-Blackwell 分层）它能降低根价值估计的方差，但树更宽会摊薄有限模拟预算、并增加单次搜索耗时。
两种效应方向相反，"谁更优"不可先验判定，只能实验。

- H1：固定模拟预算下，chance-on 侧期望得分 > 0.5（宽度收益 > 深度损失）。
- H0：E[d] = 0.5（不可分或 off 更优，双侧检验）。

## 2. 设计（运行前锁定）

- 唯一变量：己方 `--chance-nodes --chance-k 0`（on）vs 对手 `--opponent-chance-nodes off`（默认实现）；其余搜索参数完全一致。
- 每 pair 使用同一开局洗牌、红黑各一局，配对均值 `d_i = (score_red + score_black)/2`，正负抵消先后手与洗牌差异。
- 三档：
  - 档 A（同预算）：1600 局 @400 sims。
  - 档 B（同预算）：800 局 @800 sims。
  - 档 C（等时间）：on@400 vs off@492。492 ≈ 400 × 1.23，其中 1.23× 为方差副实验实测的单次搜索耗时比（同机同时段）。
- 统计：配对 bootstrap 95% CI、配对 t 检验、符号翻转置换检验（20000 次）、红/黑分层得分、整体 Wilson CI。
- 命令模板：

```pwsh
python tools\eval.py --model data\models\v3_iter560.pt --opponents model:data/models/v3_iter560.pt:<opp_sims> `
  --chance-nodes --chance-k 0 --opponent-chance-nodes off `
  --games <N> --sims 400 --seed 20260911 --workers 8 --out data\eval\chance_ab\<name>.jsonl
```

## 3. 结果

配对统计（`d̄` 为 on 侧得分率；>0.5 表示 on 占优）：

| 档 | 设定 | 局数 | W/D/L（on 侧） | d̄ | bootstrap CI95 | t | 置换 p | 结论 |
|---|---|---|---|---|---|---|---|---|
| A | 同 400 sims | 1600 | 637/335/628 | 0.5028 | [0.4806, 0.5244] | +0.25 | 0.818 | 不可分 |
| B | 同 800 sims | 800 | 336/159/305 | 0.5194 | [0.4900, 0.5487] | +1.28 | 0.217 | +1.9%，不显著 |
| C | on400 vs off492（等时间） | 1600 | 597/301/702 | **0.4672** | [0.4459, 0.4888] | **−3.00** | **0.0026** | **off 显著更优** |

- 换色分层（on 侧得分）：A red 0.490 / black 0.516；B red 0.526 / black 0.513；C red 0.448 / black 0.486。
- 全部三档 0 未终局、0 崩溃。
- 机制副实验（`tools/chance_variance_probe.py`，21 个局面快照 ×32 次重复 @400 sims）：根 Q 跨次 SD 配对差（off−on，>0 = on 更稳）**+0.00130 ± 0.00179（95% CI，含 0）**；暗子 ≥8 的局面 **+0.00246**，暗子 <8 ≈ 0；单次搜索耗时比 on/off = **1.23×**。

## 4. 结论

> **范围限定（2026-09-17 补记）**：被测模型 `v3_iter560.pt` 存在模型-搜索共适应偏倚——它是 64 通道编码（`DATA_VERSION=1`）时代训练/保存的（当前代码为 71 通道，加载时 pad-load 补零，通道 64–70 权重恒为 0），且其全部自博弈数据由 **off** 搜索生成（机会节点默认关闭，且当时的实现未修复，见 Issue 0005）。因此 on 侧搜索处于该模型的训练分布之外，偏倚方向利于 off。以下结论仅适用于"该模型在手"时的部署选择；对"机会节点的内在价值 / 训练收益"不可外推（见 §5 与 §7）。

1. **同预算下棋力不可分**：400 sims 打平（+0.3%）；800 sims 点估计 +1.9% 但未达显著。若有宽度净收益，效应量上限约在 2–5% 量级，本实验未能分辨。
2. **等时间下 off 显著更优**：在相同 wall-clock 约束下，把省下的时间换成更高模拟数（off@492）可显著赢过 on@400（−3.3 个百分点，p≈0.003）。机会节点的 1.23× 单次成本没有被宽度收益抵消。
3. **方差方向符合理论但幅度小**：on 的根 Q 跨次抖动略小（Rao-Blackwell 分层方向），优势集中在暗子多的局面；该机制性收益不足以转化为对局强度。
4. **工程结论（限定于该模型）**：拿 `v3_iter560` 系模型部署时，默认关闭 `chance_nodes` 是合理的（现状默认 False）；机会节点保留为研究/诊断开关。

## 5. 局限

- **模型-搜索共适应偏倚（关键）**：`v3_iter560` 训练于 64 通道编码（早于 S1.16 的 64→71 扩展），评估时经 pad-load 补零、7 个规则通道权重恒零；其自博弈数据全部来自 off 搜索（机会节点默认关闭且当时实现未修复）。on 侧搜索处于分布外状态，共适应偏倚方向利于 off。本实验因此只回答"该模型下的部署选择"。
- **已有相近证据不足以替代 chance-aware 对照**：`data/ab_short` 的 `chance-full` arm（自博弈开机会节点，64 轮微调，同一 runner 的 `control` 为 off 对照、同起点/同数据量/同 LR）两次 h2h-v3 得 0.40 / 0.45，`control` 两次得 0.5167 / 0.4917；但（i）该批次的训练（9/16 20:25–9/17 13:22 自博弈）与 2x2 评估（19:22–19:32）均在 Issue 0005 修复前完成（旧语义，与当前实现不同），（ii）评估为各 arm 各自 vs `v3_iter560` 基线（120 局、CI 重叠、无结论），（iii）不是 chance-trained vs off-trained 的直接配对对比。
- 单模型（v3_iter560）单 seed（20260911）；其他训练阶段/模型强度下结论可能不同。
- 等时间比率 1.23× 依赖硬件（RTX 4060 Laptop）、实现与 sims 档位；档 C 的 off@492 是对 off@400 的等时间近似换算，其优势中混合了"更多 sims"本身的增益，解释限定为"等时间约束下"。
- 评估为确定性贪心；训练分布（Dirichlet 噪声、温度采样）下未测。
- 自博弈配对只回答"同模型对抗强度"，未对比 RIS/外部基线。
- `chance_k` 截断近似（>0）未做消融。

## 6. 复现与产物

```pwsh
# 档 A / B / C（每条约 1 小时）
python tools\eval.py --model data\models\v3_iter560.pt --opponents model:data/models/v3_iter560.pt:400 --chance-nodes --chance-k 0 --opponent-chance-nodes off --games 1600 --sims 400 --seed 20260911 --workers 8 --out data\eval\chance_ab\s400_n1600.jsonl
python tools\eval.py --model data\models\v3_iter560.pt --opponents model:data/models/v3_iter560.pt:800 --chance-nodes --chance-k 0 --opponent-chance-nodes off --games 800 --sims 800 --seed 20260911 --workers 8 --out data\eval\chance_ab\s800_n800.jsonl
python tools\eval.py --model data\models\v3_iter560.pt --opponents model:data/models/v3_iter560.pt:492 --chance-nodes --chance-k 0 --opponent-chance-nodes off --games 1600 --sims 400 --seed 20260911 --workers 8 --out data\eval\chance_ab\s400on_vs_off492_n1600.jsonl

# 统计与机制副实验
python tools\paired_stats.py data\eval\chance_ab\s400_n1600.jsonl
python tools\paired_stats.py data\eval\chance_ab\s800_n800.jsonl
python tools\paired_stats.py data\eval\chance_ab\s400on_vs_off492_n1600.jsonl
python tools\chance_variance_probe.py --out data\eval\chance_variance.jsonl
```

前置修复记录：`docs/issues/0005-chance-world-conditional-sampling.md`；
进度记录：`docs/progress.md` 的 S1.18（修复/工具/副实验）与 S1.19（主实验）。

## 7. 后续可选

- **chance-aware 训练对照（回答"内在价值"的公平实验）**：用**修复后**代码重跑 chance 训练臂（`--chance-nodes --chance-k 0` 自博弈；`control` 的 off 结果不受修复影响可复用），随后两臂**直接配对 h2h**（可交叉 on/off 评估搜索，2×2：{chance-trained, off-trained} × {on, off}），配对多局统计。成本约：训练 4–6h + 评估 1–2h。
- `chance_k` 截断消融（k=4/8）在暗子多局面下是否更划算；
- 仅在暗子数高于阈值时启用机会节点。
