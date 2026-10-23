# 揭棋搜索/训练优化调研报告（2026-09-16）

> 公开整理说明（2026-10-22）：本文保留原调研日期与范围，不是持续更新的生态综述。检索未找到不等于不存在；未做统一棋力排名或首创性证明。旧提交、临时探针和原始实验输入未随此独立源码快照分发；历史问题的当前修复状态请看 [发布流程](public-release-preparation.md) 与对应 ADR。

> 方法与纪律：只读调研，未 commit；换位统计脚本 `<LOCAL_TEMP>\opencode\transposition_probe.py`（+`.json`），chance 选择偏置探针为同目录临时脚本。本文件由调研会话生成，供后续参考。

---

## 0. 执行摘要（TL;DR）

1. **命题核实**：LC0 不是 KataGo 的后代——LC0（2018-01，源自 Leela Zero）比 KataGo（2019-02，独立实现）更早；MCGS 的学术源头是 Czech 等 2020，KataGo v1.11.0（2022-03-20）提供了公开的生产级图搜索实现，LC0 的 `dag-preview` 直到 2025-08 才发布。两者是 AlphaZero 谱系下的两条独立分支，共享的"原版三件套"（虚拟损失/批叶/树复用）来自 AGZ 2017。
2. **新发现（评审未覆盖的 P0 缺陷）**：当前机会节点用通用 `select_child`（`score=-q+u`）来"选"揭示结果，而 outcome 子节点价值是走后行动方视角——等于让行动方"选一个对自己有利的揭示"，而不是按概率抽样。探针实测：两个等概率结果、价值 ±0.8 时，行动方有利结果的访问占比 99.3%，节点采样价值偏置 +0.79（±0.2 时 +0.19）。会系统高估暗子着法。
3. **旧 A/B 不可外推**：`chance-k4` 那条臂同时踩了 4 个缺陷——P0-1 叶观察者（已修复）、issue 0004 VL 不冲销、k=4 截断（实测 82.8% 的暗子着法候选 >4、吃暗子 100% >4；评审实测平均丢概率质量 28.7%/84.5%）、以及上述选择偏置。0.342 vs 0.438 的结论既不是"机会节点没收益"，也不是"K=4 可行"。
4. **MCGS 否决（有量化）**：实测 20 局/400 sims 自博弈中，以 `(side,board)` 为键的真实换位仅占叶评估 2.26%（理论上限节约≈1.36%），且键不是信息态（缺私有吃子知识/重复/无吃子/长将长捉计数）→ 合并违反两个正确性条件；叠加 KataGo "0~100 Elo 高度可变、搜索越大收益越大"与纯 Python 成本，不建议在 400 sims 下做图搜索。
5. **最值得先做**：① 修复机会节点三处问题（按概率采样代替 PUCT 选择、去 k 截断、吃暗子双揭示拆两层），并与"修复后重跑 control + lr-low-1e-5"合成一批 A/B；② PCR（playout cap randomization）作为第一个训练效率改造（sims800 已证"加深无收益"，预算该转向"加宽"）。免费项：批内叶评估去重（~6.3% 前向）。

---

## 1. 技术谱系核实（任务 1）

### 1.1 结论
**"LC0 是基于 KataGo MCGS 的优化后代" = 不成立（时间方向颠倒 + 归属错误）。**
- LC0 生于 Leela Zero；时间线：LZ 仓库 2017-10-24 → `glinscott/leela-chess` 2017-12-30（fork 自 LZ）→ LC0 公布 2018-01-09 → KataGo 2019-02-26/27（仓库非 fork，论文自述"借鉴 Leela Zero 的少量技巧"）。
- MCGS 概念源头：[Czech, Korus, Kersting 2020, arXiv:2012.11045](https://arxiv.org/abs/2012.11045)（ICAPS 2021 版）。KataGo v1.11.0（2022-03-20）提供了公开生产级 MCGS 实现（[release](https://github.com/lightvector/KataGo/releases/tag/v1.11.0)，原文"0 到近 100 Elo，高度可变"）；LC0 由 Etcaqab 自 2021 年独立开发实验分支（[2022 进展博客](https://lczero.org/blog/2023/01/2022-progress/)称其"胜过常规 Leela、已参赛"，但"尚未进发布版"），[PR #2155](https://github.com/LeelaChessZero/lc0/pull/2155) 2025-04-17 合并，[v0.32.0](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.0) 2025-08-21 以 `dag-preview` 发布。

### 1.2 机制归属（谁先引入/谁采用）

| 机制 | 源头 | LC0 | KataGo |
|---|---|---|---|
| 虚拟损失 | Chaslot 2008（并行 MCTS） | ✓ | ✓ |
| 批量叶评估 | AGZ 2017（mini-batch 8） | ✓（batch 256） | ✓ |
| 树复用 | AGZ 2017 | ✓（自博弈曾禁用） | 自博弈刻意禁用 |
| MCGS/换位表 | Czech 2020（概念）；AGZ 论文注记过 TT | 2025 `dag-preview`（独立） | 2022 v1.11.0（公开生产级实现） |
| PCR（playout cap randomization） | KataGo 2019 论文 §3.1 | — | ✓（消融≈1.37× 效率） |
| Policy target pruning + forced playouts | KataGo 2019 论文 §3.2 | — | ✓ |
| 辅助目标（ownership/score、soft policy） | KataGo 2019 §4.1 + 后续 | — | ✓（消融 1329→1139 Elo） |
| Subtree value bias correction | KataGo 2020 原创 | — | ✓ |
| 动态方差 cPUCT | KataGo v1.9.0（2021） | — | ✓ |
| Gumbel AlphaZero/MuZero | Danihelka 2022 | — | 仅取论文附录网络架构（v1.13.0） |
| FPU 折减 | AGZ 原作理念 | ✓ `--fpu-reduction=0.2` | ✓（v1.11 新启发式） |

来源：[KataGo 论文 arXiv:1902.10565](https://arxiv.org/abs/1902.10565)、[GraphSearch.md](https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md)、[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)、[AGZ PDF](https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf)、[Gumbel OpenReview](https://openreview.net/forum?id=bERaNdoegnO)。

### 1.3 对揭棋的适用性
- **值得**：policy target pruning（项目已实现）、PCR、辅助价值目标、FPU 折减、Dirichlet 总 α 随合法动作数缩放、树复用（谨慎）。
- **不适用/低优先**：MCGS（见 §3）、围棋 ownership 目标、动态方差 cPUCT（依赖短视目标，成本高）、Gumbel（理论建立在确定性 MDP 上，改造大）。

---

## 2. 随机性与隐藏信息（任务 2）

### 2.1 随机事件（机会节点）：三范式对比

| 维度 | 单结果采样（UCT 默认） | 期望化/全枚举（Rao-Blackwell） | K 结果截断采样 |
|---|---|---|---|
| 偏差 | 无偏 | 无偏 | 有偏（丢低概率质量） |
| 方差 | 最高 | 最低 | 中 |
| 成本 | 最低 | ×候选数（揭棋单翻 ≤6） | 可控 |
| 判据 | 通用基线 | 概率可精确算+候选少时最优 | 仅当候选集中时才有意义 |

依据：UCT 2006；[MCMS, Lanctot et al., IJCAI 2013](https://www.ijcai.org/Proceedings/13/Papers/093.pdf)（单结果采样是常见做法但性能未知；采 c 个结果等权平均）；[Veness et al., NIPS 2011](http://papers.neurips.cc/paper/4288-variance-reduction-in-monte-carlo-tree-search.pdf)（公共随机数 CRN 降方差）；[Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1)（chance 边始终按机会概率采样）。

**同类实践**：Pikafish 的 `jieqi` 分支 `flip_search` 对剩余池全枚举 + 按计数加权胜率 + 杀棋优先（[repo](https://github.com/official-pikafish/Pikafish/tree/jieqi)，`src/search.cpp:1670`）；`misty-banqi` 用 Star1 期望 + depth-1 翻子缓存（[repo](https://github.com/brianhliou/misty-banqi)）；CLAP_CDC（AlphaZero + 机会采样）赢 2022/2024 电脑奥林匹克 CDC（DOI 10.3233/ICG-230220）；Cazenave 在 CDC 上证明 chance-nodes 是最优的树约简（[PDF](https://www.lamsade.dauphine.fr/~cazenave/papers/mctrsg.pdf)）。

### 2.2 隐藏信息：谱系与边界
- **PIMC/确定化**：有 strategy fusion 与 non-locality 两个经典病（Frank & Basin 1998；[Long et al. AAAI 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7562)）。Long 的三参数判据说：disambiguation 高时 PIMC 表现反而好——揭棋暗子一走/一吃立即揭示，恰好是高 disambiguation。
- **ISMCTS（SO）**：[Cowling et al. 2012](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)——节点=搜索者视角的信息集，跨确定化共享统计是设计本身；determinized UCT 被显著击败。项目"根视角单次确定化"与之一致。
- **RIS-ISMCTS（每节点重确定化）**：[Goodman 2019](https://arxiv.org/abs/1902.06075) 作者自认会 "back-propagate impossible game results… pollutes the tree statistics"；且泄漏在"隐藏信息关于对手"时才严重。→ 项目 issue 0003 的修复方向（弃 RIS、改根公共视图）与文献一致，不要回头。
- **CFR/PBS 类**（Deep CFR、DeepStack/ReBeL/SoG、SePoT）：2p0s 有 Nash 保证，但严格 PBS 要求公共信念是共同知识；揭棋"吃子方知被吃暗子类型"造成后验分叉，严格 PBS 不成立——要做需先设计"公共信念近似"，属研究级工程（[ReBeL](https://arxiv.org/abs/2007.13544)、[SePoT IJCAI 2024/0480](https://www.ijcai.org/proceedings/2024/0480)）。
- **另一条路**：R-NaD/DeepNash（Stratego 无搜索达人类水平，[arXiv:2206.15378](https://arxiv.org/abs/2206.15378)；军棋复现 [RNaD-JunQi](https://github.com/JimZhouZZY/RNaD-JunQi)）；Suphx 的 oracle guiding（先给网络完整池真值、再逐步 drop；[arXiv:2003.13590](https://arxiv.org/abs/2003.13590)）。

### 2.3 对项目现状的判定
- `chance-k4` 旧 A/B 同时踩 4 个缺陷，完全不可外推；`sims800` 无收益（p=0.502）与文献"增加根世界采样≠加深搜索"一致——先修 chance，再谈加预算。
- 实测补充：400 sims 下 82.8% 的普通暗子着法候选 >4、吃暗子 100% >4 → K=4 是系统性有偏截断；候选=6 时全枚举仅 ×≤6 分支，成本可控（吃暗子双揭示 36 组合必须拆成"先翻移动子、再翻被吃子"的两层，否则分支灾难）。

---

## 3. MCGS 专项（任务 3）

### 3.1 量化实验（只读、可复现）
配置：`v3_iter560.pt`、20 局自博弈、400 sims/步、seed 20260916、CUDA；1021 次搜索、390,727 次叶评估（平均 382.7 叶/搜索）。

| 指标 | 结果 |
|---|---|
| 叶评估重复（去重后） | 7.95%（多为同批同节点碰撞） |
| 同输入重复（同 key+唯一 history+唯一隐藏签名） | 6.27% ← 批内去重可省的前向 |
| 公共键 `(side,board)` 被 ≥2 条不同历史到达 | 3,518 键（0.90% 叶），覆盖 2.26% 叶评估 |
| 其中同时隐藏签名不同 | 仅 128 键（3.64% 的换位键） |
| 图搜索理论上限 | ≈ 1.36% 叶评估节约 |
| 真实对局 | 20 局 1021 ply、均值 51.05、全部将死、repeat_plies=1（0.1%）、从未出现全明池空阶段 |
| 机会候选规模 | 暗子着法 >4 占 82.83%；吃暗子 >4 占 100% |

### 3.2 正确性条件（有出处）
1. **完美信息确定性**：合并键必须是完整 Markov 状态（含 ko/重复/无吃子/长将长捉等历史计数），否则 Graph History Interaction（Kishimoto & Müller 2004）。KataGo 用"历史敏感哈希 + `graphSearchRepBound=11`"换"部分可证明的保证"，且文档明确未处理真实环。
2. **chance 节点**：两个机会节点可合并 iff 结果集与逐项概率完全相同（expectimax 定义直接推出）。
3. **不完美信息**：节点必须是搜索者的信息集（Cowling 2012）；按公共棋盘键跨确定化合并，会把 m(w)≠b(w) 的世界分布混入统计。
4. **算法层面**：正确的图上 MCTS 需要边访问 N(n,a) + 幂等 Q 重算 + PUCT 用边计数；"朴素 DAG + 运行统计"连无限搜索都不保证收敛（[KataGo GraphSearch.md](https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md) 两个反例）。

### 3.3 结论
不建议引入以 `(side,board)` 为键的通用图搜索：键缺私有吃子知识/规则计数（违反 1、3）；每条模拟采样新世界使"世界特定键"几乎零碰撞（真换位只有 2.26%）；KataGo/Czech 一致表明"节点少收益小"，而你是 400 sims；纯 Python 复刻"边统计+幂等+历史哈希"成本高。替代：树复用、NN 评估缓存键覆盖完整输入、CDC 已验证的非图增强——以及最高性价比的"批内叶去重"。

---

## 4. 优化选项表（任务 4）

| # | 机制 | 依据 | 预期收益 | 成本 | 风险 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | chance 结果改按概率采样（替换 PUCT `select_child`），或显式期望回传 | Stochastic MuZero；expectimax | 消除行动方乐观偏置（探针 +0.19~+0.79/节点） | 低 | 低（需回归视角/VL 记账） | P0 |
| 2 | 去 k=4 截断：单翻全枚举（≤6）；吃暗子两层机会节点（6+6 而非 36） | Pikafish flip_search；MCMS；82.8%/100% 实测 | 机会节点无偏、方差↓ | 中 | 分支膨胀（可配自适应） | P0 |
| 3 | 修复后重跑 A/B：control+lr-low-1e-5+chance-full（修复版） | progress.md 复核提示；ADR 0001 | 恢复 v4 冻结依据；首次公平检验 chance | 低（跑批） | 需要算力时间 | P0（流程） |
| 4 | 批内叶评估去重（同输入合并一次前向） | 实测 6.27%（64 sims 时 54%） | ~6% 前向免费；PCR 低 sims 下更高 | 低 | 低 | P1 |
| 5 | PCR（playout cap randomization）：每 8 局 1 局 full，其余 fast | KataGo 2019 §3.1，消融 1.37× | 同时间产出更多对局（样本量是主约束） | 中 | fast 臂策略目标噪声（保留 full≥20%） | P1 |
| 6 | R2：编码 64→71 通道（chase/repetition 状态入网） | issue 0002 | 训练输入 Markov 化；长捉局面（18.8%）行为修正 | 中（数据重生成+pad-load） | 旧数据/版本兼容 | P1 |
| 7 | 辅助价值目标（soft-Z / 终局子力差 / 短视 value） | KataGo 2019 §4.1（消融 1329→1139） | value 头更稳、样本效率↑ | 中 | 权重需调 | P2 |
| 8 | Dirichlet 总 α 随合法动作数缩放 + FPU 折减 | KataGo Methods；LC0 `--fpu-reduction` | 2550 动作下探索更合理 | 低 | 低 | P2 |
| 9 | 树复用（同局面内保留子树、重定根） | AGZ；LC0 | 省重搜（估 20-40%） | 中 | value 目标偏差（KataGo 自博弈刻意禁用） | P2 |
| 10 | CRN / 固定确定化集合（ensemble determinization） | Veness 2011 | 根候选间方差↓、比较更公平 | 中 | 确定化世界数需权衡 | P2 |
| 11 | Gumbel 根选择（Top-k + Sequential Halving） | Danihelka 2022 | 低 sims 策略改进（理论在确定性环境） | 高 | 理论不覆盖随机/隐藏 | P3 |
| 12 | Oracle guiding（特权池真值逐步 dropout） | Suphx | 弱模型续训提速 | 中 | 训练曲线漂移 | P3 |
| 13 | R-NaD/蒸馏路线 | DeepNash；项目 2B 备选 | 突破 ISMCTS 上限 | 高 | 新管线 | P3 |
| — | MCGS/图搜索 | KataGo/Czech | 理论上限≈1.36% 叶 | 高 | 正确性风险高 | 否决 |
| — | PBS/CFR 类搜索 | ReBeL/SePoT | 理论更强 | 研究级 | 吃子私信破坏 PBS | 暂缓 |

---

## 5. 最值得先做的 1-2 项

**① 修复机会节点 + 合成一批 A/B（P0）**
- 三个修复：`search` 中机会节点按 prior(=概率) 比例采样 outcome（或幂等期望回传 `Q=Σ p_o Q_o`）；去 k 截断改全枚举；吃暗子双揭示拆两层。
- 同一批跑：`control`（2.5e-5）、`lr-low-1e-5`（复核 v4 冻结依据）、`chance-full`（修复版）。
- 理由：机会节点是揭棋最核心的随机结构；现有"无收益"结论建立在 4 个叠加缺陷上（探针偏置 99.3% 访问占比），而本次参考的揭棋引擎 Pikafish 使用"全枚举期望化"。

**② PCR（P1）**
- 理由：`sims800` 已统计证"加深无收益"（p=0.502），预算应转向加宽/加样本；3.5 万局下样本量是主约束；KataGo 消融 1.37×；与现有"每 8 局 strong"机制同构。
- 免费附带：批内叶去重（~6.3% 前向，且对 PCR 的 fast 臂收益更大）。

---

## 6. 不确定性/反证

1. chance 偏置幅度未实测：探针为纯逻辑层面（机制置信度高），真实搜索中的偏置多少、对胜率影响多大未测——修复后须用配对 A/B 验证。
2. "修复后机会节点一定有收益"不成立：FuzzyBear 分析表明机会节点会加宽→变浅→收益被 bootstrap 稀释；旧 A/B 无效 ≠ 新 A/B 一定正向。
3. 换位统计的边界：20 局/400 sims/单模型（v3_iter560）；"隐藏签名"未含规则计数（重复/无吃子/长将长捉）→ 2.26% 是乐观上界；`hash(history)` 用 Python hash（非 Zobrist 级严谨）。
4. 样本代表性：20 局全部将死、无全明池空阶段（训练日志 avg_moves 52–61 表明分布接近，但"完美信息阶段永不出现"的外推谨慎）。
5. PCR 收益外推自围棋：揭棋 fast 搜索的策略目标在隐藏信息下更差；full 臂比例（20%?）需实验标定。
6. Gumbel 理论前提：确定性 MDP；随机+隐藏信息下的策略改进保证失效，若要试建议只借"根 Top-k 采样"。
7. MAPLE（arXiv:2605.24139）仅验证存在性（摘要未读，中置信度）；子 agent 曾误引"Gumbel IS-MCTS (Kurpiewski)"，已剔除。
8. 未验证：LC0 是否在实现层看过 KataGo 代码（无公开证据；两者都指向 Czech 2020 与 AGZ 换位表注记）。
9. policy target pruning 已实现，但 KataGo 配套的 forced playouts 未实现——本项目下收益未验证（低优先）。
10. MCGS 否决基于"当前 400 sims + 当前键"：若未来搜索规模上到数千 sims、且暗池清空阶段常见，可在"全键（含规则计数）+ 边统计"前提下重估（本分布暂不支持）。

---

## 7. 来源清单（关键链接）

**谱系/机制**：[KataGo 论文 arXiv:1902.10565](https://arxiv.org/abs/1902.10565)｜[GraphSearch.md](https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md)｜[KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)｜[KataGo v1.11.0](https://github.com/lightvector/KataGo/releases/tag/v1.11.0)｜[LC0 2022 进展博客](https://lczero.org/blog/2023/01/2022-progress/)｜[LC0 PR #2155](https://github.com/LeelaChessZero/lc0/pull/2155)｜[LC0 v0.32.0](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.0)｜[AGZ PDF](https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf)｜[Gumbel 论文](https://openreview.net/forum?id=bERaNdoegnO)

**随机/隐藏信息**：[MCMS IJCAI 2013](https://www.ijcai.org/Proceedings/13/Papers/093.pdf)｜[Veness NIPS 2011](http://papers.neurips.cc/paper/4288-variance-reduction-in-monte-carlo-tree-search.pdf)｜[Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1)｜[Cowling ISMCTS 2012](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)｜[Goodman RIS 2019](https://arxiv.org/abs/1902.06075)｜[Long AAAI 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7562)｜[Frank & Basin 1998](https://doi.org/10.1016/S0004-3702(97)00082-9)｜[ReBeL](https://arxiv.org/abs/2007.13544)｜[SePoT IJCAI 2024](https://www.ijcai.org/proceedings/2024/0480)｜[MAPLE arXiv:2605.24139](https://www.alphaxiv.org/abs/2605.24139)｜[DeepNash](https://arxiv.org/abs/2206.15378)｜[Suphx](https://arxiv.org/abs/2003.13590)｜[DouZero](https://proceedings.mlr.press/v139/zha21a.html)

**同类实践**：[Pikafish jieqi 分支](https://github.com/official-pikafish/Pikafish/tree/jieqi)｜[misty-banqi](https://github.com/brianhliou/misty-banqi)｜[AB-JChess](https://github.com/lxsgx23/AB-JChess)｜[RNaD-JunQi](https://github.com/JimZhouZZY/RNaD-JunQi)｜[DarkKnight (DOI 10.1109/TCIAIG.2014.2329034)](https://doi.org/10.1109/TCIAIG.2014.2329034)

**MCGS**：[Czech arXiv:2012.11045](https://arxiv.org/abs/2012.11045)｜[LC0 Transposition tests](https://github.com/LeelaChessZero/lc0/wiki/Transposition-tests)｜[LC0 issue #1091](https://github.com/LeelaChessZero/lc0/issues/1091)｜[GHI Kishimoto & Müller](https://aaai.org/papers/00644-aaai04-102-a-general-solution-to-the-graph-history-interaction-problem)

**本地证据**：`docs/progress.md`、`docs/adr/0001~0010`、`docs/issues/0001~0004`、`docs/reviews/2026-09-16-adversarial-review.md`、`azero.py`（`select_child`/`_backup`/`search` 机会节点路径）、单测 `test_backup_chance_edge_keeps_perspective`。

---

## 附录 A：离线引擎仲裁——回放末手分歧（2026-09-16）

**背景**：回放 `data/replays/jieqi_20260916_130612.json` 末手（黑方 i9f9，暗子假类型车，实际揭示炮 → 杀）中，RIS-ISMCTS-3200 首选 i9f9，而 v3_iter560 + NeuralMCTS（chance 关与修复后 chance-full）均首选 f3d2。为判断“i9f9 是否只是赌赢”，引入离线引擎仲裁。

**引擎可用性**（结论重要）：
- Pikafish `jieqi`（NNUE 分支）**不可用**：分支代码期望 7560 维特征（`FeatureSet::Dimensions=7560`、TF=2048；编译期期望 hash=0x6E24CA4A），而官方发布的 net 是 25920 维（desc 中 hash 常量碰巧相同但维度不同）→ 加载在 layer0 失败；官方与社区**均未发布** 7560 维揭棋 net（mistboard 注释自述 “our own-trained net”；维护者以官方 issue #110 关闭此类请求）。
- 实际使用：**Pikafish `jieqi_old`（classical eval、无需 net）**，也是 mistboard 线上所用构建（分支 `jieqi_old`，手动编译：mingw-w64 g++ 16.2.0，`-O3 -std=c++17 -DNDEBUG -DNOMINMAX -static -march=native`）。
- 引擎与源码在仓库外（`%LOCALAPPDATA%\jieqi-engines\`），不污染项目。

**仲裁局面**（重演回放前 75 手；黑走；黑仅剩 1 枚暗子在 i9、假类型车；黑自视池 = 炮1、兵2）：

```
FEN_PRE（黑走）:
1P6x/2a6/3k4b/r2a5/6b2/P1N1B4/2P2n2p/9/5K3/3X1X1X1 b R2A1C2P1N0B0r0a0c1p2n0b0 0 1

FEN_PAWN（反事实：若该暗子为兵、落 f9，红走；黑池 = 炮1兵1）:
1P3p3/2a6/3k4b/r2a5/6b2/P1N1B4/2P2n2p/9/5K3/3X1X1X1 w R2A1C2P1N0B0r0a0c1p1n0b0 0 1
```

**结果**（双方均 3200 sims 的 RIS/AZ；MultiPV 引擎 Threads 8 / Hash 512；score 为走子方视角）：

| 引擎/配置 | 首选 | i9f9 | f3d2 |
|---|---|---|---|
| jieqi_old depth 20 MultiPV6 | **i9f9**（+1576） | 第 1 | 第 5（+1481，差 ~95cp） |
| jieqi_old depth 24 MultiPV3 | d6c7（+1650） | 第 2（+1649，差 1cp） | 未入前 3 |
| jieqi_old depth 24 searchmoves i9f9 | — | +1635（黑全胜 WDL 1000/0/0） | — |
| RIS-ISMCTS-3200（seed 777 / 1234） | **i9f9**（1284 / 1116 visits，Q(走子方) ≈ +0.48） | 第 1 | — |
| v3_iter560 + NeuralMCTS-3200（chance 关） | f3d2 | policy=0.0216（不在 top4），根值 −0.004 | 第 1 |
| v3_iter560 + NeuralMCTS-3200（chance-full 修复后） | f3d2 | policy=0.0887（第 4），Q=+0.050 | 第 1 |

反事实世界（FEN_PAWN）：红方视角 −658cp（WDL 0/0/1000）→ 黑方仍胜。
即 **i9f9 在两种揭示结果下均获胜**（1/3 炮→立杀；2/3 兵→仍胜势），并不是赌博。

**结论**：
1. 末手 i9f9 是裁判引擎的顶级选择（d20 第 1；d24 与第 1 差 1cp、与第 3 差 7cp）→ RIS 的选择有据可依；AZ v3 的分歧是**模型评估弱**，不是搜索实现问题（搜索次数同为 3200）。
2. v3 与裁判引擎的差距在**价值头**（根值 ≈ 0 vs 裁判 +16.5）与**暗子着法的策略先验**（i9f9 policy 2.2%）；修复 chance 后 policy 提升约 4 倍（0.0216 → 0.0887）但仍排第 4 → 修复让搜索“看见”揭示分支，剩余差距要靠网络（更多/更好的数据、R2 编码、训练时长）；若自博弈始终 chance 关，会形成“暗子着法盲区”的自强化偏差。
3. 局限：classical 裁判（无 7560 维 net 可用）、depth 20–24、单局面；该局面黑本已胜势（合理着法均 ≈ +16，WDL 1000/0/0），f3d2 是“胜势下次优”而非败着。
4. 复现：引擎 `pikafish-old.exe`（仓库外）+ `<LOCAL_TEMP>\opencode\jieqi_uci_eval.py <exe> <workdir> <fen> [depth] [multipv] [searchmoves]`；局面重建脚本 `<LOCAL_TEMP>\opencode\replay_probe.py`（三引擎均为 3200 sims）。

---

## 附录 B：P1 三项实施结果（2026-09-16）

| 项 | 状态 | 关键结果（冒烟/单测） | 详见 |
|---|---|---|---|
| P1-4 批内叶评估去重 | 已实现，默认开（\--no-leaf-dedup\ 关） | 400 sims 重复率 7.0%、耗时 −5%；100 sims 重复 27.0%、耗时 −9%；\chance_k<=0\ 时 RNG 流不变 | progress §S1.14 |
| P1-5 PCR | 已实现，默认关（\--pcr\ 开） | pcr25 步/s 2.3×（166 vs 388 评估/步），full 策略样本 23.3%；策略目标只取 full/strong | progress §S1.15 |
| P1-6 编码 64→71 通道（R2） | 已实现（\DATA_VERSION=2\） | 旧模型 pad-load 前向不崩且旧通道逐元素一致；v1 数据拒绝；新增 7 单测 | progress §S1.16 |

- 全量测试 **119 passed**；改动未提交（机会节点修复基线上继续）。
- 未决：三项 A/B 均未跑（命令与判据见 progress 对应小节）；R2 需先归档并重生成 v2 数据；64 vs 71 通道对比需两个代码快照各自重生成数据。
