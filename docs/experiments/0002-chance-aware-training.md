# 实验 0002：chance-aware 训练对照（已运行，2026-09-18）

> 历史实验：2026-10-01 的搜索/信息边界/编码修复改变了训练与评估语义。保留当时数据和结论，不能据此冻结当前配置；当前状态见 [status.md](../status.md)。

- 状态：**已运行并出结论**（私有工作区快照，未随本仓库分发）
- 背景：实验 0001 的范围限定指出被测模型 `v3_iter560` 与 off 搜索共适应（64 通道 pad-load、自博弈全 off），故 0001 不能回答"机会节点的内在价值"。本实验用 **Issue 0005 修复后**的代码训练 chance-aware 模型，做训练侧公平对照。
- 起点：`data/models/v3_iter560.pt`（两侧同一模型热启）
- 契约：ADR-0007（配对、红黑互换、未终局不计分、seed 20260911、max_plies 600、评估贪心）
- 配置：`tools/ab_chance128.json`；数据：`data/ab_chance128/`、`data/eval/chance128_ab/`

## 1. 设计

- 两臂同起点、同超参（64 局/轮、400 sims、每 8 局 1 局强自博弈、常量 LR 2.5e-5、1000 步/轮、8 workers），唯一变量：
  - `chance-full-128`：自博弈 `--chance-nodes --chance-k 0`（修复后世界条件采样），0→128 轮，buffer 全部为 chance 语义数据（13.7h）；
  - `control-128`：自博弈 off，前 64 轮复用 `data/ab_short/control`（off 语义不受修复影响），64→128 续训（6.1h）。
- 评估：
  1. 各自 vs `v3_iter560:400` 基线（120 局，ab_runner 内置）；
  2. **两臂直接配对 h2h**：1600 局 @400 sims，双方统一 **off** 搜索平台，`data/eval/chance128_ab/h2h_off_1600.jsonl`。

## 2. 已知瑕疵（实施记录）

- `control-128` 续训时按 `--max-load-files 32` 只重建最近 32 个文件的温缓冲（加载 254,288 = 32 文件 × augment 镜像 2×）；全量应为 512,920（64 文件 × 2）。该参数继承自 ab_short 批次（当时文件数 ≤32 恰好全量）。经决策（2026-09-18）**接受差异继续**：control 的 buffer 起步 254k、末期约 77 万，而非理想轨迹的 512k → ~102 万。
- 数据与模型本身完整（64 个 shard 全在），此瑕疵只影响 control 臂早期样本的多样性。
- 修复（2026-09-18，Issue 0006）：续训加载已改为按 `replay_buffer_size` 从最新文件自适应回填（`max_files=0` 默认）。本实验为历史产物，**按原样保留、不重跑**，上述结论不受影响；未来实验不要复制 `--max-load-files 32`。

## 3. 结果

| 评估 | chance-full-128 | control-128 | 差异与检验 |
|---|---|---|---|
| vs v3_iter560:400（120 局） | 0.550（53W/26D/41L）CI95=[0.461, 0.636] | 0.579（58W/23D/39L）CI95=[0.490, 0.664] | 点估计 −2.9%，CI 重叠，不显著 |
| 直接 h2h（1600 局，off-off，配对 800） | d̄=**0.4856**，W/D/L=646/262/692 | — | bootstrap CI95=[0.4637, 0.5075]，t=−1.29，置换 p=**0.202** → **不显著** |

- h2h 换色（chance 侧得分）：red 0.4931 / black 0.4781；0 未终局、0 崩溃。
- 统计功效：h2h 的 95% CI 上限 0.5075 → 可排除 chance 训练带来超过 ~+0.75 个百分点的优势（在此样本量与 400 sims 配置下）。

## 4. 结论

1. **训练侧未见 chance-aware 收益**：直接 h2h 点估计 −1.4% 偏向 control，统计不显著；对基线评估方向一致（−2.9%，不显著）。
2. 结合实验 0001（搜索侧：同预算不可分、等时间 off 显著更优），机会节点在当前配置（400 sims、64 局/轮、128 轮、单模型强度）下**既无搜索侧净收益、也未显示训练侧净收益**。
3. 工程结论维持：`chance_nodes` 默认关闭；保留为研究/诊断开关（可复查机会节点的期望建模价值需另寻场景，如更深的预算、暗子极多的局面或价值目标质量分析）。

## 5. 局限

- `control-128` 的 buffer 起步瑕疵（见 §2）：其训练数据分布偏向近期，方向上可能削弱 control（缺早期多样性），但实际结果 control 略强，故不改变"无机会节点收益"的结论方向。
- 两臂训练 wall-clock 不对称（chance 13.7h vs control 6.1h 续训）：这是机会节点搜索开销 1.23× 的必然后果；按数据预算（128 轮 ×64 局）对齐。
- 单 seed（20260911）、120 局基线 CI 宽、h2h 仍有 ±2.2% 的 95% CI；更小效应需更大样本。
- 未做 2×2 交叉（双方 on-on、混合配置）与更长训练（256 轮）的消融。

## 6. 复现

```pwsh
# 训练（chance 主臂；control 前 64 轮已存在于 data/ab_short/control）
python tools\ab_runner.py --config tools\ab_chance128.json --only control-128,chance-full-128

# 两臂直接配对 h2h（off-off，1600 局）
python tools\eval.py --model data\ab_chance128\chance-full-128\models\latest.pt --opponents model:data/ab_chance128/control-128/models/latest.pt:400 --games 1600 --sims 400 --seed 20260911 --workers 8 --out data\eval\chance128_ab\h2h_off_1600.jsonl
python tools\paired_stats.py data\eval\chance128_ab\h2h_off_1600.jsonl
```

相关：`docs/experiments/0001-chance-node-ab.md`（搜索侧主实验）、`docs/issues/0005-chance-world-conditional-sampling.md`（修复记录）、`docs/progress.md` S1.19/S1.20。
