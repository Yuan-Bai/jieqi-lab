# 揭棋 AI 项目调研笔记（规则 / RIS-ISMCTS / AlphaZero / 开源生态）

> 公开整理说明（2026-10-22）：本文保留原调研日期与范围，不是持续更新的生态综述。检索未找到不等于不存在；未做统一棋力排名或首创性证明。旧提交、临时探针和原始实验输入未随此独立源码快照分发；历史问题的当前修复状态请看 [发布流程](public-release-preparation.md) 与对应 ADR。

- 调研日期：2026-09-11
- 调研范围：本项目代码理解、揭棋规则一手来源、规则实现逐条对照、不完美信息 MCTS 理论、AlphaZero 训练实践、可借鉴开源项目
- 核验标记：
  - ✔ 本次调研亲自抓取原文/源码验证
  - ○ 来自搜索聚合或二手页面（已附来源 URL，未逐字核验）

---

## 0. 项目速览

| 文件 | 行数 | 职责 |
|---|---|---|
| `jieqi.py` | 725 | 规则引擎：位编码棋子、表驱动走法生成、暗盒(pool)、走子/悔棋、终局判定 |
| `ismcts.py` | 496 | ISMCTS 智能体（根视角单次确定化；函数名 `ris_ismcts` 沿用）+ `BeliefModel`（双方视角信念）+ 确定化 |
| `azero.py` | 1417 | AlphaZero 式训练：2550 动作空间、64 通道特征、ResNet、PUCT、自博弈、评估、断点续训 |
| `gui.py` | 413 | Tkinter 人机对弈界面（默认对手为 1000 次迭代的 RIS-ISMCTS） |
| `test_*.py` | 35 用例 | 全部通过（本次运行验证 ✔） |
| `merge_selfplay.py` | 34 | 自博弈数据合并工具 |

关键设计点：

- **棋子编码**：`type | color<<3 | dark<<4`（`jieqi.py:21-24`）。
- **暗子规则**：棋盘上暗子按其“初始位置对应的兵种”走子（假车/假炮……），走子或吃子时从暗盒随机抽真实类型并翻开；明士可出九宫、明象可过河（`iter_pseudo_moves`，`jieqi.py:451-538`）。
- **暗盒 (pool)**：双方各 2 车 2 马 2 象 2 士 2 炮 5 卒；暗子每次翻开/被吃都从池中抽取；`pool_total` 等于场上仍暗的棋子数。
- **隐藏信息模型**：`BeliefModel`（`ismcts.py:105`）为每个观察者维护“已知类型计数 + 已知总数 + 未知被吃数”；吃暗子时**吃方知道类型、被吃方只知总数少 1**；`determinize()` 按缺失数随机采样。搜索自 2026-09-14 R3 修复起为“每条模拟只按根玩家视角确定化一次”（不再每节点按行动者重抽）。
- **AlphaZero**：动作空间 2550（位移向量表，`azero.py:35-70`）；输入 64 通道（明/暗×7 兵种×2 方 + 轮次 + 暗盒统计 + 未知被吃 + 无吃子 + 长将 + 最近 8 步历史）；ResNet 64 通道 ×15 残差块；PUCT + Dirichlet；温度线性退火；多进程自博弈（spawn，需 CUDA else 单进程）；评估对手为 RIS-ISMCTS；镜像增强；replay buffer 1M。

训练现状（仓库内数据）：

- `data/models/latest.pt`（37MB）存在；`train_state.json` = `iteration: 1, total_games: 32, best_score: -1.0`。
- `data/logs/train.log`：历史曾跑到 iter 21/22（best_score 0.000，自博弈 64 局/迭代，iter21 用时 7331s）；2026-02-26 重置重跑：iter0 32 局（红 2 / 黑 2 / 和 28，平均 262.3 ply，含 4 局 strong），训练 2000 步。
- `data/selfplay/` 目前仅 1 个文件 `iter_20260226_162628_0000.pkl.gz`。
- 观察：iter0 和棋率 28/32 偏高，与“无吃子 50 ply 判和 + 长将/重复简化”强相关（见 §2）。

---

## 1. 揭棋规则（一手来源）

### 1.1 核心规则（多来源一致 ✔）

来源：天天象棋官方规则页、中文维基百科、Chess Programming Wiki（Jeiqi）、Mistboard 规则页（链接见 §6）。

1. **初始摆放**：将/帅明置并留在九宫原点；其余每方 15 子洗混后背面朝上随机摆到该方的象棋初始位置。
2. **暗子首步**：按其**所在初始位置的兵种**走子/吃子（角上为“假车”，依次类推），受该兵种在象棋中的常规限制（蹩马腿、塞象眼、炮架、士不出九宫、象不过河）；该步走完后立即向双方翻开。
3. **翻开后**：按真实身份走子；**明士可出九宫、明象可过河**；兵/卒按当前位置决定：过河前只进，过河后可横走，永不后退。
4. **被吃暗子**：
   - 天天象棋官方：“己方暗子被吃后，自己是不能看到暗子是什么棋，明子被吃则可见”；
   - Mistboard 明确“只有吃子方知道其身份，原主看不到”；
   - 中文维基措辞为“吃子方将该暗子背面朝上放置，被吃子的一方不能翻看”。
   - 注：Chess Programming Wiki 的 FEN 章节写成“被吃即公开”，与上面矛盾；以平台规则（吃方知、原主不知）为准 ✔。
5. **将帅照面**（对脸将）仍有效；暗子同样遮挡纵线、并按“假兵种”参与将军判定与应将。
6. **胜负**：将死或困毙均判负（Mistboard、天天象棋一致）。

### 1.2 平台规则差异（重要）✔

| 规则点 | 天天象棋（官方） | Mistboard | 复杂度论文的自对弈设定 | 本项目 `jieqi.py` |
|---|---|---|---|---|
| 无吃子和棋 | 40 回合（80 ply） | 60 步（120 ply） | 40 ply | **25 回合（50 ply）** |
| 长将 | 同一棋子连续将军 >6 回合后**禁着** | 未提 | — | **连续将军 >3 次直接判负** |
| 长捉 | 连续捉同一子 >6 回合后**禁着** | 未提 | — | **未实现** |
| 重复局面 | 未提 | **不自动判和** | — | 第 4 次相同局面（含暗盒）**判刚走方负** |

结论：本项目采用了一套自定义简化规则，与线上主流平台并不一致；这直接影响自博弈数据分布与棋力对标（详见 §2.2）。

### 1.3 与 `jieqi.py` 逐条对照

**一致 ✔**：初始摆法；暗子按原位兵种走子；明士/明象解除区域限制；兵卒过河逻辑按当前位置；吃暗子的信息不对称（吃方知）；困毙判负；将帅照面；暗子按“假兵种”参与攻击判定（`is_square_attacked`）。

**差异与风险**：

1. **终局判定顺序错误（已实测复现）**：`game_outcome()`（`jieqi.py:670`）先判长将、再判重复、最后判将死。后果：若第 4 次连续将军同时形成将死，会被判成“长将负”。
   实测（合成局面：黑将 (0,4)，红车 (1,3)/(1,5)/(2,4) 已构成将死）：
   - `consecutive_checks[RED]=0` → `(1, 'checkmate')`（红胜）✔
   - `consecutive_checks[RED]=4` → `(-1, 'long_check')`（红负）✘
   建议：**先判将死/困毙，再做长将/重复裁决**；更贴近平台规则的做法是把长将/长捉实现为“禁着”（剔除该走法）而不是直接判负。
2. **长将阈值**：实现为 4 次连续将军即负；平台为 6 回合后禁着。且实现不区分“同一棋子”，也不要求重复局面，会把正常连续攻杀误判为长将。
3. **长捉完全缺失**；GUI 把 `REASON_REPETITION` 标注为“长捉负”（`gui.py:236`），概念混用。
4. **无吃子和棋门槛** 50 ply vs 平台 80/120 ply，导致训练对局和棋显著偏多（日志 iter0：28/32 和棋）。
5. **重复判定包含暗盒（隐藏信息）**：`position_key` 含双方 pool（`jieqi.py:313-314`），即以“真实但双方不可见”的信息参与重复裁决；且判负而非判和。
6. **暗象首步缺少河界检查**（代码层面）：`ELEPHANT_MOVES` 无过河限制。经几何验证，暗象只存在于 (0,2)/(0,6)/(9,2)/(9,6)，其第一步所有落点都在己方半场，因此**实际不可达、无影响**；但建议补检查做防御性编程。
7. **GUI 信息泄漏**：`update_pool_labels()`（`gui.py:259-275`）直接展示真实 `state.pool`。当 AI 吃掉人类的暗子时，人类不应知道自己池里少了哪一类子，但界面会显示出来；`BeliefModel` 已正确维护“未知被吃”，UI 未使用。（暗子被吃不进“被吃棋子”列表，`gui.py:206-216`，这部分是安全的。）
8. **缺少 FEN/记谱**：Chess Programming Wiki 的 Jeiqi 规范（暗子记 `X/x`、走子后缀标注揭示类型、`[被吃子]` 组）已可用作与 Pikafish-jieqi 互通、导入导出与复盘的格式 ✔（链接见 §6）。

---

## 2. RIS-ISMCTS 与不完美信息搜索

### 2.1 理论要点（附出处）

- **ISMCTS**：在信息集上建树，每轮迭代抽取一个确定化世界，只沿与该世界一致的分支搜索（Cowling, Powley, Whitehouse, *Information Set MCTS*, IEEE TCIAIG 2012）。
- **PIMC/确定化的固有缺陷**：*strategy fusion*（同一信息集内对不同真实状态分别选最优，等于作弊）与 *non-locality*（对手可引导博弈偏离确定化假设）——Frank & Basin 1998。
- **RIS-ISMCTS**：在每个决策节点从“该行动玩家自己的视角”重新确定化，缓解根玩家信息泄漏（Goodman, *Re-determinizing ISMCTS in Hanabi*, CIG 2018, arXiv:1902.06075）。作者自述其代价：会产生与根信息集不兼容的“假确定化”，把不可能的结果回传、污染树统计。本项目曾按该路线实现（`ismcts.py` 每节点 `determinize`）；2026-09-14 的 R3 修复后，`ris_ismcts()`（函数名保留）与 `azero.py::NeuralMCTS` 均改为**根视角单次确定化**（SO-ISMCTS 语义），详见 `docs/issues/0003`。
- **揭棋的隐藏信息类型**：属于 **symmetric hidden information**（连持有者也不知道自己暗子的身份；Mistboard 原文、复杂度论文都如此描述）。这与扑克/军棋“私有信息只有自己知道”不同，DeepStack/ReBeL 式“从行动者视角重建私有信息”部分失效；但揭棋的公共结构好（暗子位置公开、剩余池可公共推导），显式建“翻子机会节点”是可行且更干净的方案。

### 2.2 暗棋/揭棋专门研究

- **《On the Complexity of Dark Chinese Chess》**（Cong Wang, Tongwei Lu, Wuhan Institute of Technology，arXiv:2112.02989 / IEEE CCDC 2022）✔ 已抽取原文：
  - 结论：**AlphaZero 不能直接用于暗中国象棋**（无法直接处理随机 + 不完美信息）；CFR 及抽象技术也难以直接套用（动作空间大、子力价值强依赖位置）。
  - 自对弈统计：分支因子约 34–35，平均对局长度约 133 ply，开局信息集大小约 10^17，博弈树复杂度约 10^205。
  - 其自对弈程序采用“**40 ply 无吃子判和**”（与本项目 50 ply 接近，与平台 80/120 ply 不同）。
- 中文论文思路（○ 转载页）：在极大/极小层之间插入“**暗子扩张层**”（对暗子翻面后的每种类型建机会节点，贝叶斯加权回传），并为暗子设计“位置灵活度 + 平均子力价值”的评估。
- 暗棋（Banqi/CDC）社区：小棋盘（2×4）上 tabular AlphaZero 收敛到理论最优（TAAI 2018 / IEEE Access 2023），CLAP_CDC 获 2024 ICGA 暗棋冠军（○，来源见 §6）→ 支持“先在可解小局面验证信念与搜索”。

### 2.3 与本项目的差距

- 自对弈/搜索中的“翻子”没有显式 **chance node**；依靠隐式重确定化。可对照 **Pikafish-jieqi** 的做法 ✔（已抓取 `jieqi` 分支 `src/search.cpp`）：
  ```cpp
  // Dive into flip search when the last move is moving a dark pieces
  if ((ss - 1)->currentMove.is_ok() && pos.is_dark((ss - 1)->currentMove.to_sq()))
      return flip_search<nt>(pos, ss, alpha, beta, ...);
  ...
  for (auto& [piece, num] : restPieces) {          // 枚举剩余暗盒中每种可能
      Piece flipped_piece = pos.do_flip(to_sq, piece, ...);
      value = search<nodeType>(pos, ss, alpha, beta, depth, cutNode);
      pos.undo_flip(...);
  }
  ```
  即：**把“翻出什么”当作机会节点，按剩余池逐一搜索**。这是本次调研参考的 Pikafish 揭棋分支（NNUE）的处理方式，不构成棋力排名。
- 温度/评估注意：2550 动作 vs 800 模拟，模拟数远小于动作数，Gumbel MuZero 的“少量模拟下保证策略改进”在此类设置收益最大（Danihelka et al. 2022）。
- Suphx（麻将）的 **oracle guiding**（先上帝视角学，再逐步遮蔽隐藏信息）对揭棋这种“隐藏信息拖慢学习”的游戏非常对口。

---

## 3. AlphaZero 训练调研

### 3.1 原版超参数

**AlphaGo Zero（Nature 2017，`agz_unformatted_nature.pdf`）**：

- 每步 1600 次模拟；前 30 步 τ=1，之后 τ→0；根节点 Dirichlet η~Dir(0.03)、ε=0.25；允许认输（自动标定，误判率 <5%，10% 对局禁用认输）。
- 每迭代 25,000 局；训练从最近 500,000 局采样；minibatch 2048；SGD momentum 0.9、L2 1e-4；学习率阶梯 1e-2→1e-4。
- 门控：新模型对当前最优下 400 局（1600 模拟、τ→0、无噪声），胜率 >55% 才晋升。

**AlphaZero（Science 2018 / arXiv:1712.01815）**：

- 训练时每步 800 次模拟；按根节点访问次数比例选步；评估时对访问次数取 argmax。
- Dirichlet 浓度与合法着法数成反比：α = {0.3（国象）, 0.15（将棋）, 0.03（围棋）}，ε=0.25。
- LR 0.2，在 100k/300k/500k 步降到 0.02/0.002/0.0002；国象/将棋各约 700k mini-batch；国象自对弈 44M 局。
- AZ 实际上**不做门控**（ELF OpenGo 对照表：AGZ "55% win rate"，AZ "none"）；原文未给出 c_puct 数值（ELF 表标 `?`），复现常取 1.5–5（AGZ 风格取 5；ELF OpenGo 调出 1.5、virtual loss 1.0；CCZero 取 1.5）。

**已验证的开源象棋 AZ 配置（CCZero，`configs/normal.py` ✔）**：

```
simulation_num_per_move = 800      c_puct = 1.5 (self-play) / 1.0 (eval)
noise_eps = 0.15                   dirichlet_alpha = 0.2
tau_decay_rate = 0.9               virtual_loss = 3
resign_threshold = -0.98           min_resign_turn = 40
enable_resign_rate = 0.5           batch_size = 512
lr_schedules = [(0,0.01),(150000,0.003),(400000,0.0001)]
loss_weights = [1.0, 1.0]
```

### 3.2 工程实践与常见坑

1. **value target 必须是“该步行动方”视角**，否则价值头学反（AGZ Methods；CCZero 等实现一致）。
2. **policy target 用根节点访问次数分布**；**KataGo 的 policy target pruning**：把根 Dirichlet 噪声从训练目标中剥离（噪声只用于选择），消融显示约 1.25× 等价加速（KataGo, arXiv:1902.10565）。
3. **Dirichlet 噪声只加在根、只用于自对弈**；评估/正式对局无噪声。
4. **温度调度**：前期 τ=1 保多样性，后期 τ→0 贪心；评估一律贪心。揭棋对局长度不同，需按对局长度重标定“前 N 步”。
5. **门控 vs 不门控**：门控防策略退化但拖慢迭代；小规模本地训练可学 AZ 不门控、只记录强度曲线。
6. **Replay buffer 新鲜度**：AGZ 用最近 500k 局；数据太少时训练会迅速过拟合（KataGo/ELF 的消融都证实）。每迭代只新增几 k 样本时，训练步数必须相应受限。
7. **BatchNorm running stats 陈旧**（ELF OpenGo 专门报告）：推理必须 `eval()`，自对弈/评估/训练前向一致。
8. **对称增强要逐规则验证**：揭棋左右镜像严格成立（规则左右对称、无颜色翻转）；上下/旋转不成立。
9. **认输要保守**（AGZ 的 <5% 误判 + 10% 禁用；CCZero `-0.98/40 回合/50% 对局`）。
10. **规模现实**：AZ 国象 44M 局/5000 TPU；本地 32–64 局/迭代的规模下，靠 Gumbel/PCR/oracle guiding 等样本效率手段更实际。

### 3.3 对照本项目 `azero.py` 的观察

**做对的部分**：

- value 视角、PUCT 选择时父节点视角转换（`select_child` 取负号）、以及测试用例覆盖（`test_azero.py` 全过 ✔）。
- 训练/推理 BatchNorm 模式切换；评估无 Dirichlet、低温；自博弈多进程 spawn + CUDA 回退单进程；镜像增强正确（保色）。
- “strong”对局（每 8 局 1 局，1600 模拟、无噪声、极低温度）用于产出更高质量目标。

**建议改进（按影响排序）**：

1. **训练步数 vs 新增数据量失衡**：`train_steps=2000`、batch 512 → 每迭代 100 万次采样；而每迭代新增约 32 局 × ~130 ply ≈ 4k 样本，buffer 约 3.5 万 → 每迭代约 28 个 epoch，极易过拟合。建议按新增样本量设 `train_steps`，或改为固定 epoch 数（如 1–4 epoch/迭代）。
2. **policy target 包含 Dirichlet 噪声**（`search()` 里噪声先于访问统计），建议实现 KataGo 式 target pruning。
3. **规则失真会污染数据**：50 ply 无吃子判和导致 28/32 和棋；长将判负先于将死判定会产生错误标签（§1.3）。修规则后再训练。
4. **翻子未建机会节点**：自博弈 target 的质量受 RIS 假确定化影响；可参照 Pikafish `flip_search` 在关键节点显式枚举剩余池。
5. **评估协议**：目前只有 vs RIS-ISMCTS 的 score（32 局，噪声大），建议加：固定随机种子棋盘的双色配对局、随机走子基线、历史 best 对最新（Elo 近似）、以及评估局数增加。
6. **样本存储**：每迭代一个 `pkl.gz`，resume 时全量加载（日志见 356k 样本）。可定期 `merge_selfplay.py` + 控制 `max_load_files`，或改流式训练。
7. **工程补齐**：`requirements.txt`/README 缺失；`nul` 残留文件；GUI 信息泄漏（§1.3）。

---

## 4. 现有开源项目与可借鉴点

### 4.1 揭棋专用（最相关）

| 项目 | 说明 | 借鉴点 |
|---|---|---|
| `official-pikafish/Pikafish` 的 **`jieqi` 分支** ✔ | 调研中参考的揭棋引擎（UCI + NNUE，未做统一棋力排名）。`src/search.cpp` 中有 `flip_search`：对暗子走子后的翻面结果枚举剩余池逐一搜索 | **机会节点建模的工程范本**；可作为评估对手/标注器 |
| `Velithia/JieqiBox` ○ | Tauri+Vue3 揭棋 GUI（46★），兼容 Pikafish 揭棋 FEN | 界面与 FEN/协议参考 |
| `miaosiSari/Jieqi`（musesfish）✔ 仓库存在 | Python/PyPy，PVS alpha-beta + 手工评估（26★）；搜索中刻意屏蔽对手吃掉的暗子信息（`discount_red/black`）；作者建议用 MCTS 处理大信息集 | 揭棋专属启发（空头炮/暗炮进四/禁着/开局库）；隐藏信息处理思路 |
| `arthuryangcs/JieqiAI` ○ | C++，基于 ElephantEye/UCCI | 传统搜索基线 |
| 《**揭棋人工智能探秘**》○ | 书籍（CSDN `zhyjie2008` 预告，Google Books 有书目；作者疑似与皮卡鱼揭棋相关）。B 站有“皮卡鱼揭棋 23.1.11 达揭9水准”的软件介绍 | 揭棋 AI 的中文专著线索 |
| Jeiqi FEN/记谱规范 ✔ | Chess Programming Wiki：`X/x` 暗子、走子后缀标注揭示类型（`b9c7c`）、`[被吃子]` | 与现有引擎互通、复盘、数据集格式 |

### 4.2 中国象棋 AlphaZero/引擎生态

| 项目 | 说明 |
|---|---|
| `NeymarL/ChineseChess-AlphaZero`（CCZero）✔ 配置已核 | 完整分布式 AZ 流水线（self/opt/eval + 监督预训练）；超参见 §3.1 |
| `chengstone/cchess-zero` ○ | 教学向 AZ 实现；作者自评“训练规模不足时棋力很差”，规模预期管理参考 |
| `official-pikafish/Pikafish` / PX0 生态 ○ | 象棋引擎 + Lc0 系 AZ 训练流水线（pxzero-training/lczero-client） |
| `xqbase/eleeye`（象眼）、`xqbase/xqwlight` ○ | 经典 alpha-beta 走法生成/搜索实现 |
| `lightvector/KataGo`、`opendilab/LightZero`、`suragnair/alpha-zero-general` | 训练技巧（policy pruning、辅助头）、统一 MCTS/RL 框架、最小可读 AZ 模板 |

### 4.3 不完美信息棋牌参考

- `kwai/DouZero`（斗地主，DMC + 动作编码 + 历史 LSTM，不做显式信念）；`datamllab/rlcard`；`Equim-chan/Mortal`（日麻，历史 Transformer）。
- `JimZhouZZY/RNaD-JunQi`（军棋，DeepNash/R-NaD 复现）；`google-deepmind/open_spiel`（含 `dark_hex` 等不完美信息环境）。
- 暗棋（Banqi/CDC）：DarkKnight（progressive bias 的 MCTS）、CLAP_CDC（2024 ICGA 冠军）；Kriegspiel/Recon Blind Chess 社区的经验：“国际象棋式的局面评估在盲棋里经常误导，必须对信息集做评估聚合”。

---

## 5. 建议清单（优先级从高到低）

1. **修终局判定顺序 + 对齐规则参数**：先判将死/困毙，再判长将/重复；建议把长将/长捉实现为“禁着”，并把无吃子和棋、长将阈值对齐目标平台（如天天象棋：40 回合、6 回合）。用 §1.3 的合成局面加一个回归测试。
2. **翻子显式机会节点**：自博弈与搜索中按剩余池枚举翻面结果（学 Pikafish `flip_search`），替代/补充 RIS 重确定化的假确定化路径；这一步对策略目标质量影响最大。
3. **训练循环卫生**：KataGo policy target pruning；train_steps 与新增样本量挂钩；评估加配对开局/随机基线/更大局数；记录 ELO 近似曲线。
4. **小局面正确性验证**：构造 2-3 个可解析的小残局（固定暗盒），验证：无策略融合、无信息泄漏（人类 belief 与引擎 belief 一致）、value 视角正确。
5. **工程与体验**：修正 GUI 暗盒泄漏（显示人类 belief 而非真实 pool）；补 `requirements.txt`/README；可选支持 Jeiqi FEN 导入导出以便与 Pikafish 对拍；清理仓库残留文件。

---

## 6. 参考来源

**规则**

- 天天象棋官方揭棋规则（腾讯）：https://image.qqchess.qq.com/test_160324/activity/html/jieqiguize.html ✔
- 中文维基百科·揭棋：https://zh.wikipedia.org/wiki/%E6%8F%AD%E6%A3%8B ✔（含暗子走法/明子走法/被吃暗子段）
- Chess Programming Wiki·Jeiqi（含 FEN/记谱规范）：https://chessprogramming.org/Jeiqi ✔
- Mistboard 揭棋规则（含被吃暗子只告知吃方、60 步判和）：https://mistboard.com/zh-hans/rules/jieqi ✔
- 知乎/CSDN 规则科普（二手）：https://zhuanlan.zhihu.com/p/86913264 、https://blog.csdn.net/qq_36591160/article/details/147163413 ○

**论文与理论**

- Wang & Lu, *On the Complexity of Dark Chinese Chess*：https://arxiv.org/abs/2112.02989 ✔（PDF 原文已抽取）
- Cowling et al., *Information Set Monte Carlo Tree Search*, IEEE TCIAIG 2012：https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf
- Frank & Basin, *Search in games with incomplete information*（strategy fusion），AI 1998：https://www.sciencedirect.com/science/article/pii/S0004370297000829
- Goodman, *Re-determinizing ISMCTS in Hanabi*, CIG 2018：https://arxiv.org/abs/1902.06075
- Silver et al., *Mastering the game of Go without human knowledge*（AGZ）：https://www.nature.com/articles/nature24270
- Silver et al., *A general reinforcement learning algorithm…*（AZ）：https://arxiv.org/abs/1712.01815
- KataGo 论文（policy target pruning 等）：https://arxiv.org/abs/1902.10565
- ELF OpenGo（AZ 对照表、BatchNorm 问题）：https://arxiv.org/abs/1902.04522
- Gumbel MuZero：https://openreview.net/forum?id=bERaNdoegnO
- ReBeL：https://arxiv.org/abs/2007.13544 ；DeepNash：https://www.science.org/doi/10.1126/science.add4679
- Suphx（oracle guiding）：https://arxiv.org/abs/2003.13590
- DouZero：https://arxiv.org/abs/2106.06135
- 暗棋 AlphaZero（TAAI 2018 / IEEE Access 2023 / CLAP_CDC 2024）：https://scholar.nycu.edu.tw/en/publications/alphazero-for-a-non-deterministic-game 、https://ieeexplore.ieee.org/document/10049064 、https://journals.sagepub.com/doi/10.1177/13896911251315096 ○

**代码与工具**

- Pikafish `jieqi` 分支（`flip_search`）✔：https://github.com/official-pikafish/Pikafish/tree/jieqi （源码：`src/search.cpp`）
- miaosiSari/Jieqi（musesfish）✔：https://github.com/miaosiSari/Jieqi
- arthuryangcs/JieqiAI：https://github.com/arthuryangcs/JieqiAI
- Velithia/JieqiBox：https://github.com/Velithia/JieqiBox
- NeymarL/ChineseChess-AlphaZero ✔：https://github.com/NeymarL/ChineseChess-AlphaZero
- chengstone/cchess-zero：https://github.com/chengstone/cchess-zero
- Pikafish：https://github.com/official-pikafish/Pikafish ；PX0：https://github.com/official-pikafish/px0
- LightZero：https://github.com/opendilab/LightZero ；alpha-zero-general：https://github.com/suragnair/alpha-zero-general ；KataGo：https://github.com/lightvector/KataGo
- DouZero：https://github.com/kwai/DouZero ；RNaD-JunQi：https://github.com/JimZhouZZY/RNaD-JunQi ；OpenSpiel：https://github.com/google-deepmind/open_spiel
- 《揭棋人工智能探秘》书目：https://books.google.com/books/about/%E6%8F%AD%E7%A7%98%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD.html?id=Ds5SzwEACAAJ ○

> 说明：`docs/` 目录为本报告首次创建；引用的 ○ 条目未逐字核验，仅作为线索保留，落地前建议二次确认。
