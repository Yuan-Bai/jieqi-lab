# 揭棋训练规模参数调研：MCTS sims / Replay Buffer / 每轮新增数据

> 公开整理说明（2026-10-22）：本文保留原调研日期与范围，不是持续更新的生态综述。检索未找到不等于不存在；未做统一棋力排名或首创性证明。旧提交、临时探针和原始实验输入未随此独立源码快照分发；历史问题的当前修复状态请看 [发布流程](public-release-preparation.md) 与对应 ADR。

> 调研日期：2026-09-19。方法：4 个并行调研 agent，一手来源优先（论文原文/官方仓库配置/官方 wiki/官方服务端实时配置），每条数字后标注来源；无法核实的一律写"未找到"。相关既有文档：`docs/research_jieqi_azero.md`、`docs/research_jieqi_search_training.md`、ADR-0010（train_steps）。
>
> **结论先行**：本项目当前 `sims=400 / buffer=1M 增强 / 64 局/轮 / 1000 train steps` 与同棋盘规模项目（CCZero、icyChessZero、JiangJun）的 sims 一致，但**每轮新增数据量比同类项目低 1–2 个数量级**、**样本重复训练率（~32×）比 KataGo 建议值（4–8×）高 4–8 倍**。最优先动作是提高每轮自博弈局数（64 → 256，即回到 `AZConfig.self_play_games` 默认值），而不是调 sims 或 buffer。

## 0. 本项目现状（对照基线）

| 参数 | 当前值 | 说明 |
|---|---|---|
| 自博弈 sims | 400/步（strong 局 1600，每 8 局；`sims800` arm 为 800/3200） | `tools/ab_experiments.json` |
| 每轮自博弈 | 64 局（`AZConfig` 默认 256） | 实验配置主动调小 |
| 每轮新增原始样本 | ~4k（64 局 × 揭棋均长 ~62 ply，由 `data/ab_short/control/selfplay` 256,460/64 文件反推） | ×4 增强后 ~16k |
| Replay buffer | 1,000,000（按**增强后**样本计，仅 25 万原始样本） | ×4 增强（ADR-0011）后 64 轮即溢出 |
| 每轮训练 | 1000 steps × batch 512 = 51.2 万样本展示量 | ADR-0010 固定步数制 |
| 样本重复率 | 新数据 16k/轮 在 62 轮生命周期内被看 ~32 次 | 与下文 KataGo 4–8× 对比 |
| 生成:训练 | 每 1 个 minibatch（512）仅对应 0.064 局新对局 | ELF 警告 games:minibatches < 10:1 会过拟合 |

## 1. AlphaZero 系（完全信息）

| 项目 | 自博弈 sims | Buffer / 窗口 | 每轮生成 | 每轮训练 | 来源 |
|---|---|---|---|---|---|
| AlphaGo Zero (2017) | 1600 | 最近 **500k 局** FIFO（≈20 轮） | **25,000 局/iter** | 每 1000 步存 checkpoint，batch 2048 | Nature 2017（UCL 作者稿） |
| AlphaZero (2018) | **800** | 论文未指定 | 无 iter 概念；chess 全程 44M 局/9h（≈63 局/step） | 700k steps × batch 4096 | arXiv:1712.01815 |
| ELF OpenGo (2019) | 1600 | 500k 局 FIFO（50 队列×10k） | 无 iter；全程 ~20M 局 / 1.5M minibatch | batch 2048 | arXiv:1902.04522 |

关键比例（ELF 原文）：**games : minibatches 不低于 10:1**（AGZ 7:1、AZ 30:1、ELF 13:1），低于 10:1 会显著过拟合；新数据平均训练次数 ≤1.5（总数据量是我们的 100–1000 倍时的数字）。可迁移结论：**buffer 容量以"覆盖最近 10–20 个数据生产周期"为准**；探索常数、LR 三段形状与规模无关；网络要按数据量缩（ELF 消融用 64f×5b）。

## 2. Leela Zero / Lc0 / KataGo

| 项目 | 自博弈 sims | 窗口 | 每轮新增 | 训练量 / 重复率 |
|---|---|---|---|---|
| Leela Zero (Go) | 1600 visits（早期 3200） | **最近 250k 局**，每轮重新 dump | 1.6万–11万局/网 | ≤128k steps/session（batch 256）≈32.8M 样本；sampling ratio 未公开 |
| Lc0 (Chess) | **现行 Run2=800**（Run1 曾 10000） | 最近 N chunks（=局）；RL 后期 250–600k 局 | 2018 主 run 26.7k 局/网（3.6M positions） | 50k steps，batch 1024→256，**sampling ratio 14.2→3.55** |
| KataGo (Go) | **600 全搜索 + 100 cheap**（25% 着法记录），后期 1000/200 → 2000/350 | 幂律增长：250k → 22M rows（N^0.65~0.75），纯 recency | 连续异步；24h ≈ 156k 局 / 9.1M rows | batch 256，1M 样本/epoch，**训练 bucket = 新增数据 ×4~8**（doc 明确 >4 有过拟合风险） |

KataGo 的两个直接可复用项：
1. **playout cap randomization**：75% 着法只跑 cheap 搜索且**不写入训练数据**，25% 全搜索记录。同等算力对局产量 ×2–3。本项目已有对应实现（PCR 臂 + chance 节点）。
2. **`max-train-bucket-per-new-data = 4`**：每轮训练采样量不应超过新增数据的 4–8 倍。本项目当前为 512k 展示量 / 16k 新增 = **32×**，明显偏高；把每轮局数提到 256（新增 64k 增强）后降为 8×，恰好进入推荐区间。

## 3. 不完美信息游戏（扑克/斗地主/麻将）

| 项目 | 搜索 | 数据循环 | Buffer | Batch | 总账本 |
|---|---|---|---|---|---|
| DouZero（斗地主） | 无（DMC） | 45 actor 连续产局，按位置入 3 个共享队列 | 50 entries × 100 样本（消费型，非长期 replay） | 3200 | 30 天 / 4×1080Ti |
| PerfectDou | 无（PPO） | IMPALA 式异步 | 论文未给容量 | 1024（8 GPU） | 2.5e9 frames |
| ReBeL（扑克） | 每子博弈 **1024 次 CFR**（深度 2 / 到本街） | 解子博弈→数据入环形缓冲→采样叶节点[继续] | **12M 样本环形**（repo 默认 2M） | 1024 | HUNL 1750 epochs × 2.56M ≈ 4.5e9 样本 |
| AlphaHoldem | 无（PPO） | 每 iter 131,072 步（8×128 env×128 step） | 未给 | 16,384 | 50k iters ≈ 2.7e9 手 |
| Suphx（麻将） | 无（PG；pMCPA 非搜索） | 持续异步 | 未给 | 未给 | 每版 1.5M 局 / 44 GPU×2 天 |
| DeepStack / Libratus | CFR continual re-solving | 无自博弈循环（随机局面监督/一次性求解） | — | — | **参数不可比**，勿直接对照 |

可迁移结论：不完美信息阵营普遍**无 MCTS**（DouZero 等），有搜索的 ReBeL 成本极高（1024 CFR 迭代/子博弈）——本项目 400 sims 的确定化 ISMCTS 已明显低于 ReBeL 档位；这些工作的 buffer 容量普遍不是敏感参数（多家未披露），**小项目 1e5–1e6 样本环形缓冲 + 均匀采样**足够；更值得借鉴的是**对手多样性机制**（AlphaHoldem 的 vs 历史 Top-K、ReBeL 的 25% 随机动作）——本项目目前已通过温度/噪声部分覆盖。

## 4. 象棋 / 暗棋 / 军棋 / 揭棋（棋盘与规则最接近）

| 项目 | 棋盘 | 自博弈 sims | 数据窗口 / 触发 | 训练量 |
|---|---|---|---|---|
| **CCZero**（NeymarL/ChineseChess-AlphaZero） | 9×10 象棋 | **800** | 数据文件上限 **5000 局**；`min_games_to_begin_learn=5000` | batch 1024，192f×10b，LR 0.03→1e-4 |
| icyChessZero（北邮） | 9×10 象棋 | 800 | 攒 **>10000 局**才更新一轮 | batch 256，128f×7b |
| JiangJun（华为，2023） | 9×10 象棋 | 800（亦提 1800） | 每 block **≥5000 局** | batch 2048，192f×10b，90×V100 |
| **Hsueh et al. 2023（IEEE Access）** | 2×4 暗棋（含 chance） | 默认 800；扫描 25–25600 | 每 iter **1000 局**；表格版窗口 2×iter；DNN 版 buffer **40k 局** | batch 256，5-block ResNet |
| aa8100225/chinese_dark_chess | 暗棋 | 600 | — | README 自述"效果≈随机"（反面案例） |
| RNaD-JunQi | 迷你军棋 | 非 AZ | — | 32ch CNN，batch 500，12k steps → 对随机 82% |
| Pikafish `jieqi` 分支（本次调研参考的揭棋引擎） | 揭棋 | αβ+NNUE，非 MCTS | — | 无自博弈训练循环 |

**Hsueh 2023 的关键实测结论**（与本项目机制最接近的学术工作）：
- "足够用"的 N_sim 远小于直觉：2×4 暗棋 PPPP 只需 **50**，固定时间预算下 N_sim=800 跑 300 iter 的最终胜率反而低于小 N_sim 多跑几轮（高 sims 挤占了对局产量）。
- 低 N_sim 时 MCTS 树节点初始 W/N 用 **0.5（和棋先验）** 明显优于 0（视为输）。
- c_puct∈[0.5,4]、Dirichlet α∈[0.03,12]、ε∈[0.1,0.5] 都鲁棒——超参敏感性低于预期。
- 局限：2×4 动作空间仅 40，揭棋 2550，不能直接照搬"50 sims 够用"的结论。

**揭棋（Jieqi）本身**：截至 2026-09-19，本次 GitHub/arXiv/中文源检索未找到可直接对照的公开 AlphaZero/MCTS 训练项目或论文；这不是不存在相关工作或本项目首创的证明。本次核对的引擎包括采用 αβ/NNUE 路线的 Pikafish jieqi 分支。以下暂以“暗棋 Hsueh + 象棋 CCZero”作有限参考，不能视为揭棋的通用训练标准。

## 5. 跨项目对照速查

| 维度 | 行业常见值 | 本项目现状 | 建议 |
|---|---|---|---|
| 自博弈 sims（9×10 类棋盘） | 800（AZ/CCZero/icyChessZero/JiangJun） | 400（strong 1600） | **保持 400–800 区间**；算力允许时评估 800；更优先考虑 PCR 保持产量 |
| Buffer 覆盖周期 | AGZ ≈20 轮；Lc0 250–600k 局；KataGo 幂律窗口 | ≈62 轮（1M 增强 / 16k 每轮） | 保持 1M（安全性优先）；局数提升 4× 后自然降到 ~15 轮，进入推荐区间 |
| 每轮新增数据 | 同类 1000–25000 局（棋盘/单局成本不同） | **64 局 ≈ 4k 唯一样本** | **提到 128–256 局**（256 = 仓库默认值） |
| 训练:新增 重复率 | KataGo 4–8×；Lc0 sampling ratio 3.5–14× | **~32×** | 局数 ×4 后自动降到 8×；或等效把 train_steps 降到 250–500 |
| games : minibatches | ELF 硬约束 ≥10:1 | 0.064:1（远低于） | 无法在当前算力下满足；用"重复率 4–8× + 大 buffer + gating"近似替代 |
| 网络规模 | 数据 1/100 时用 64–128f×5–10b（ELF 消融） | 需查（当前网络配置未在本报告核实） | 数据量未大幅提升前**不要加大网络** |
| gating | LZ SPRT 55%、KataGo 200 局可选 | 已有 200 局@800 验收 + best 模型机制 | 保留现有 gating，无需额外改动 |

## 6. 针对本项目的建议（按优先级）

1. **每轮自博弈局数 64 → 128 或 256（最高优先）**。依据：CCZero 起步阈值 5000 局、icyChessZero 10000 局、JiangJun ≥5000 局/block、AGZ 25000 局/iter、Hsueh 1000 局/iter；本项目 64 局/轮是全部同类中最低。提到 256（恰好是 `AZConfig.self_play_games` 原默认值）后：每轮新增 ~16k 唯一 / 64k 增强，重复率降到 8×、buffer 覆盖降到 ~15 轮，两项同时落入行业推荐区间。代价：自博弈耗时 ×4（ab_chance128 当前 128 轮约需数天，需按算力权衡 128 vs 256）。
2. **sims 维持 400（或经 A/B 升 800）**，不要为提速降到 200 以下。依据：9×10 类项目普遍 800；Hsueh 的"低 sims 划算"在动作空间 40 的小棋盘上成立，2550 空间的项目（CCZero 等）均用 800。降 sims 的替代方案是多确定化采样（已经做了根视角单次确定化 + chance 节点）。
3. **buffer 保持 1M 增强样本**。依据：ELF 警告小 buffer 早期过拟合；Lc0 实践中大窗口显著降低 train/test MSE 方差（issue #1197）；1M 的内存代价可接受，且局数提升后覆盖率自动合理化。
4. **train_steps 暂保持 1000，观察局数提升后的效果**。1000 步是 ADR-0010 的显式决策（小数据下保证训练信号）；局数 ×4 后重复率 8× 已符合 KataGo `4~8×` 建议，无需再改。若未来算力允许进一步加大数据量，train_steps 按 `新增增强样本 × 4~8 / 512` 校准即可。
5. **可选的下一批 A/B 轴**（按调研证据强度排序）：a) `self_play_games` 128 vs 256 vs 64；b) `sims` 400 vs 800（已有 sims800 arm）；c) PCR 参数（fast=100/full 比例，与 KataGo cheap 25% 对应的已验证技巧）；d) MCTS 节点初始 W/N=0.5（Hsueh 的低 sims 稳定化技巧，需先在规则/搜索层验证语义）；e) 网络规模消融（若数据量提升后再考虑加大）。
6. **不建议**：引入对手池/历史 Top-K（本项目 best 模型 + 温度噪声已部分覆盖，收益未验证）；模仿 DouZero 的"无搜索"路线（与现有 ISMCTS 基础设施冲突）；按 AZ/ELF 的总量数字缩放到本项目的每次训练步数（它们的新数据训练次数 ≤1.5 依赖 100–1000 倍的数据量，小项目照搬等于不训练——ADR-0010 已有此结论）。

## 6b. 学习率（LR）对照与建议（2026-09-19 补）

| 项目 | Optimizer | LR 与调度 |
|---|---|---|
| AGZ / AZ / ELF | SGD + momentum 0.9 | AZ 0.2→0.02→0.002→0.0002；AGZ/ELF 1e-2→1e-3→1e-4 三段台阶（ELF 消融：每段步数减半近似） |
| KataGo | SGD + momentum 0.9 | 2e-5（前 5M 样本）→ 6e-5 → 6e-6（17.5 天后），按样本量推进 |
| CCZero | TF（未注明） | 0.03→0.01→0.003→0.001→0.0003→0.0001（每 100k steps） |
| icyChessZero | — | 0.01 |
| JiangJun | — | 三角循环 0.03 ↔ 0.0001（warm restart） |
| DouZero | RMSprop | 1e-4 |
| PerfectDou / ReBeL / AlphaHoldem | Adam | 3e-4（ReBeL 每 800 epochs 减半） |
| **本项目** | **AdamW** | 默认 5e-4 + cosine→1e-5（T_max=iterations）；A/B 冻结候选=常量 2.5e-5 微调；Gate 软续训=5e-5 cosine（≈2.85e-5→1e-5） |

要点：
- **optimizer 不可直接对照**：AlphaZero 系（含象棋类 CCZero/icyChessZero/JiangJun）全部用 SGD，LR 数值比 Adam 系大 1–2 个数量级；Adam 系项目（不完美信息阵营）普遍 3e-4，本项目默认 5e-4 同量级、热启动 2.5e-5 更保守，均为合理区间。
- **本项目 LR 维度已有实验证据**：早期 v4 批"1e-5 胜出"在 P0-1 修复后的复核批（S1.17，v2 数据 64 轮）不成立——没有任何新项超过 control（2.5e-5），冻结候选维持常量 2.5e-5。LR 在当前规模下接近饱和，继续扫 LR 的边际收益低于数据侧（局数/重复率）。
- **若采纳"局数 64→256"**：数据新鲜度与去相关度提升，最优 LR 可能上移（`lr-high-5e-5` 配置可直接复用重扫），但须按现行评估协议重跑，不能沿用旧 32 轮微调批的结论。
- JiangJun 的三角循环（warm restart）对"反复从检查点热启动"的场景有潜在价值（与本项目实验流程相似），但无本项目证据，暂不引入。

## 7. 主要来源

- AlphaGo Zero: Nature 2017, https://www.nature.com/articles/nature24270（作者稿 PDF: https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf）
- AlphaZero: https://arxiv.org/abs/1712.01815（HTML 全文 https://arxiv.org/html/1712.01815v1）
- ELF OpenGo: https://arxiv.org/abs/1902.04522v5；https://github.com/pytorch/ELF
- Leela Zero: https://github.com/leela-zero/leela-zero（FAQ.md、training/tf/parse.py、issue #484/#1197/#1569）；https://zero.sjeng.org/
- Lc0: https://training.lczero.org/training_runs；https://github.com/LeelaChessZero/lczero-training（README、tf/train.py、docs/loader.md）；https://github.com/LeelaChessZero/lc0/wiki/Neural-Net-Training
- KataGo: https://github.com/lightvector/KataGo（SelfplayTraining.md、python/train.py、python/selfplay/shuffle.sh、cpp/configs/training/selfplay1.cfg）；https://arxiv.org/abs/1902.10565；https://katagotraining.org/
- DouZero: https://arxiv.org/abs/2106.06135；https://github.com/kwai/DouZero
- PerfectDou: https://arxiv.org/abs/2203.16406；https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou
- ReBeL: https://arxiv.org/abs/2007.13544；https://github.com/facebookresearch/rebel
- AlphaHoldem: https://ojs.aaai.org/index.php/AAAI/article/view/20394（DOI 10.1609/aaai.v36i4.20394）
- Suphx: https://arxiv.org/abs/2003.13590
- CCZero: https://github.com/NeymarL/ChineseChess-AlphaZero
- icyChessZero: https://github.com/bupticybee/icyChessZero
- JiangJun: https://arxiv.org/abs/2308.04719
- Hsueh et al.: IEEE Access 2023, DOI 10.1109/ACCESS.2023.3246638（免费 PDF: https://dspace.jaist.ac.jp/dspace/bitstream/10119/18238/1/I-IKEDA-K0405-13.pdf）
- RNaD-JunQi: https://github.com/JimZhouZZY/RNaD-JunQi
- 揭棋复杂度: https://arxiv.org/abs/2112.02989；Pikafish jieqi 分支: https://github.com/official-pikafish/Pikafish/tree/jieqi

## 8. 不确定项与可信度说明

- AGZ 学习率切换精确步数、AZ 的 buffer 规格与网络 filters/blocks：论文未给或未核实（标"未找到"）。
- LZ 服务端训练调度器未开源，250k 局窗口来自维护者 issue 讨论（一手但非代码）。
- Lc0 2018 主 run 的 sampling ratio 14.22→3.55 来自官方 wiki 表格（注明数字可能不精确）。
- 项目中所有"重复率/覆盖率"推导基于揭棋均长 ~62 ply 与当前实验配置，若自博弈局长分布变化需重算。
- Hsueh 的实验棋种（2×4 暗棋）与揭棋差异显著，其 sims 结论仅作方向性参考，不作为降 sims 的依据。
