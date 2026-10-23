# 对抗式代码审查缺陷清单（2026-09-16）

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

> 用途：本文件是 2026-09-16 对抗式审查的**唯一缺陷记录来源**，供后续修复 agent 逐条处理。
> 方法：4 路并行子 agent（规则引擎 / 搜索与信念 / 训练评估管线 / 编码与 GUI）+ 主审代码复核 + 独立红队逐条证伪。
> 基线：`python -m pytest test_azero.py test_ismcts.py test_jieqi.py test_eval.py -q` → **82 passed**（审查时）。
> **仓库零改动**：本次审查只读 + 临时探针；本文件是唯一新增产物。工作区原有 R3/R4 未提交改动未被触碰。
> 探针目录（本机保留）：
> - `<LOCAL_TEMP>\opencode\rules_probe\`
> - `<LOCAL_TEMP>\opencode\mcts_probe\`
> - `<LOCAL_TEMP>\opencode\train_probe\`
> - `<LOCAL_TEMP>\opencode\encode_probe\`
> - `<LOCAL_TEMP>\opencode\redteam_probe\`
>
> 终端中文输出前缀（Windows/pwsh 7，每条 python 命令前加）：
> `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;$OutputEncoding=[System.Text.Encoding]::UTF8;$env:PYTHONIOENCODING='utf-8';$env:PYTHONUTF8='1';`
>
> **2026-09-20 追记**：本清单 P0–P3 共 10 条**全部已修复并探针验证**（私有历史修复及后续；回归测试 82→99，当前全量 143 passed；逐条证据见 `docs/progress.md` 头部与 S1 各节）。评审后新发现项：Issue 0007（off 模式树跨世界复用，已记录未修；只影响 chance-off 搜索，主线 chance-on 不受影响）。

## 修复顺序建议（有依赖）

1. **先修 P0-1（R3 叶 observer 槽）**：这是搜索信息模型的核心漏项，修复后 v4 A/B 结论（lr 1e-5 胜出、chance-k4 淘汰等）才能被重新解读；chance-k4 arm 的叶机会先验同样受污染。
2. P0-2/P0-3（评估 fail-open、resume 静默降级）：修复后重跑 A/B/验收时才能保证结果可信。
3. P1/P2/P3 相对独立，可按序处理。
4. 每条修复都应补回归测试（见各条「回归建议」）；全部完成后重跑 82 项基线 + 新增测试。

---

## P0-1【高·搜索】R3 修复不完整：叶评估/机会规格按"叶行动方"读 observer 槽

**位置**：`azero.py:1006`（`leaf_info = root_view.info_state(sim_state, sim_state.side)`）；配套 `azero.py:931-946`（`root_belief` 构造）、`ismcts.py:217-229`（`observe_move` 按 mover/other 分槽更新）。

**机理**：`root_belief` 两个观察者槽初始都复制根玩家行，看似恒等；但模拟中一旦发生"暗子被吃/被揭开"，`observe_move` 把 mover 槽更新为"知道真实类型"、另一槽更新为"只知少一子"。此时 `info_state(sim_state, sim_state.side)` 在叶轮到对手时读取对手槽（含根不可见的私有知识），或缺失根已知信息——与 `azero.py:931-932` 注释及 `docs/issues/0003` 声明的"叶评估与机会规格只使用根玩家公共视图"直接矛盾。`leaf_specs`（`azero.py:1007`）复用同一 `leaf_info`，一并被污染。

**最小复现**：
```pwsh
python <LOCAL_TEMP>\opencode\mcts_probe\probe_root_view_minimal.py
# → leaf input depends on hidden captured type (leak): True ; root-observer input invariant: True
python <LOCAL_TEMP>\opencode\mcts_probe\probe_leaf_leak.py
# → 279 次非根 observer 叶评估中 38 次"叶知道根不可知的被吃类型"、161 次"叶缺失根已知信息"
python <LOCAL_TEMP>\opencode\mcts_probe\probe_realnet_400.py
# → 真实 v3_iter560.pt 400 sims：2/2 局面选招翻转，policy L1 0.35/0.33
```
红队补充：`redteam_probe\c1_realnet.py` 800 sims **8/12 局面输出改变**，seed=2 argmax 翻转（max Δpolicy 0.406）。

**影响**：根 Q / 策略目标依赖根不可测信息 → 自博弈 policy target 噪声；评估与 v4 A/B（R3 修复后的同码批次）结论被污染；`chance-k4` 的叶机会先验同样读错槽。

**置信度**：高（机制 + 确定性复现 + 真实模型复现）。

**去重**：R3 已修复项的**意图**是"根视角单次确定化 + 根公共视图"；本条是其实施漏项（取槽用了 `sim_state.side` 而非 `root_state.side`），不是重复报告。

**修复建议**：叶取 observer 固定为根玩家（`root_view.info_state(sim_state, root_state.side)`）；值标签的"行动方视角"约定不变（信息态里的 `side` 字段仍是 `sim_state.side`）。检查 `_chance_specs` 调用处的 info 来源。

**回归建议**：增强 `test_azero.py::TestSearchInformationSafety::test_opponent_private_belief_does_not_leak_into_search`——现有测试只改初始对手槽（`root_belief` 只复制根槽，恒通过、无判别力）；应改为在**模拟过程中**制造"暗子被吃导致两槽分叉"的局面，断言叶评估 trace 不随对手私信变化。

---

## P0-2【高·评估】`tools/eval.py` 模型缺失时用随机权重照常评估并写 summary（fail-open），A/B `.done` 放大危害

**位置**：`tools/eval.py:382-390`（只 print 警告、EXIT 0，结果文件无任何标记）；`tools/ab_runner.py:217-233`（无 summary 也写结果行 + `.done`；`--skip-train` 直接触发）；对照 `tools/eval.py:119`（`model:` 对手缺失会抛 `FileNotFoundError`）。

**复现**：
```pwsh
python tools\eval.py --model C:\nonexistent\nope.pt --opponents random --games 2 --sims 4 --workers 1 --out $env:TEMP\missing.jsonl
# → 警告: 未找到模型...将使用随机初始化权重进行评估；score 正常输出；EXIT=0；JSONL 无模型缺失标记
python tools\ab_runner.py --config <config> --skip-train   # arm 无 latest.pt → 仍写 .done + results 行
```
红队 `redteam_probe` C4 两步复现实跑确认；典型触发：新 arm 直接 `--skip-train`、模型归档/改名、路径写错。

**置信度**：高。

**去重**：ADR-0007/0008、S1.10 均未登记；非 R1-R4。

**修复建议**：主模型不存在→非零退出（或至少写 `model_missing`/模型路径到 run 行并拒绝写 summary）；ab_runner 仅在 summary 有效且评估退出码为 0 时写 `.done`。

**回归建议**：`tools/eval.py` 缺模型时返回非零；ab_runner 空 summary 不写 `.done`。

---

## P0-3【高·训练】resume 静默降级三连：损坏数据崩溃 / latest.pt 缺失静默随机续训 / train_state 损坏静默重置

**位置**：
- `azero.py:1194` `except (OSError, pickle.UnpicklingError, ValueError)` 漏 `EOFError`（`save_compact_iteration` 直接写最终路径 `azero.py:1115-1124`，非原子）。
- `azero.py:1649-1651` `if load_model_if_exists(...)` 无 `else` 警告：`latest.pt` 缺失时用随机权重从 `iteration=N` 继续，并覆盖写 `latest.pt`。
- `azero.py:313-327` 损坏 `train_state.json` 静默返回默认 `TrainingState()`（默认 `ruleset_version=3`，连版本告警也不触发）；`save_training_state`（330-340）非原子写。

**复现**：
```pwsh
python ...\train_probe\probe2_truncated_file.py        # 截断/0字节 → EOFError 整体崩溃（契约是"跳过并告警"）
python ...\train_probe\probe11_silent_random_resume.py # 无"模型缺失"警告，latest.pt 被随机权重覆盖
python ...\train_probe\probe12_corrupt_state.py        # 损坏 state 静默 iteration=0，非原子写
```
红队 `redteam_probe\c5_probe.py` / `c6_probe.py` 独立复现（含真实 `training_loop` 三个场景）。

**置信度**：高（机制与输出确定；触发前提为非正常终止/文件损坏）。

**去重**：已登记项只有"resume 不恢复优化器动量"，与本条不同。

**修复建议**：`except` 补 `EOFError`（及 `zlib.error`/`AttributeError`/`TypeError` 兜底）；`except` 分支已有 warning 逻辑，保持"跳过坏文件"契约。`latest.pt` 加载失败且 state.iteration>0 时告警或拒绝续训。`save_training_state`/`save_compact_samples` 改临时文件 + `os.replace` 原子写；损坏 state 读取失败时显式告警（不静默用默认值）。

**回归建议**：截断 `.pkl.gz` 被跳过不崩；缺 latest.pt 的 resume 产生告警；损坏 state 触发告警。

---

## P1-4【中高·GUI】两处 AI 落子竞态：暂停无效 + 旧局延迟回调在新局用旧设置落子

**位置**：`gui.py:1042-1066`（`_poll_ai_search` 无 `paused` 检查）；`gui.py:1002`（`after(delay, lambda: self._start_ai_search(settings))` 无对局代次守卫）；`gui.py:1004-1010`（`_start_ai_search` 自身递增 `_ai_token`，反而作废新局在途搜索）。

**复现**：
```pwsh
python ...\encode_probe\p4_gui_thread.py       # [FAIL] 暂停后仍落子 1 步（状态显示"已暂停"）
python ...\encode_probe\p7c_gui_stale_clean.py # [BUG] 旧局 A 搜索结果在新局 B 落子
```
红队 `redteam_probe\c8_probe.py` 用真实 Tk 状态机独立复现。真实搜索耗时更长（450–1200ms 延迟窗口），更易命中。

**置信度**：高。

**去重**：S1.8-F10 只覆盖"搜索移入后台线程"，未覆盖暂停/对局代次。

**修复建议**：`_poll_ai_search` 回填前检查 `paused`（暂停时丢弃或挂起，需产品裁定）；`after` 回调带对局代次（generation/token）守卫并在返回菜单/重开时作废旧回调；`_start_ai_search` 不要绕过代次检查自行递增 token。

**回归建议**：桩 agent 下"暂停后不再落子"；"返回菜单后旧回调不启动搜索"。

---

## P1-5【中·评估协议】配对评估声称的"同一初始洗牌"未实现

**位置**：`tools/eval.py:135-138`（pair 两局同 seed）+ `jieqi.py:349-353`（`GameState.initial` 不消耗 rng、无洗牌）+ `jieqi.py:416-433`（真值在 `apply_move` 时懒抽）。

**复现**：
```pwsh
python ...\train_probe\probe5_pair_shuffle.py   # 同 seed 两局在同一格揭示真值不同
python ...\redteam_probe\c7_probe.py
```
`('move',6): A=7 !== B=6` 等。ADR-0007 §1 与 CONTEXT「配对评估」均写"同一初始洗牌"。

**影响**：无系统偏差，但配对降方差目标落空、"同开局配对/配对符号检验"表述被高估；Wilson CI 独立性假设本就因同 seed 配对受损（此点独立于本条）。

**置信度**：事实高、定性中。

**去重**：非 R1-R4。

**修复建议**：开局预物化整副暗子真值（或提供确定性洗牌接口），再让两局共享该分配；至少修正文档措辞并说明统计口径。

**回归建议**：同 seed 配对两局在相同格子的揭示真值一致。

---

## P1-6【中·规则】长捉不追踪移动中的受害子，`chase_streak` 永远停在 1

**位置**：`jieqi.py:650-695`（续捉要求 `from_idx == chase_anchor` **且** `board[target] == chase_target_code`，target 是格子）。

**复现**：
```pwsh
python ...\rules_probe\p13c_flee.py
# → 红车往返追同一只逃跑黑车 12 步：streak=1 恒定，banned=(False,False)
# 对照：受害子静止时可达 streak=6 并禁着
```
红队 `redteam_probe\c9_probe.py` 复现。与 ADR-0003「同一枚棋子连续攻击同一枚敌方非将、非兵棋子」字面冲突（代码已存 `chase_target_code`，说明意图按子身份）。平台原文不可得，存在"跑子是否算同一目标"的规格歧义。

**置信度**：机制高、规则定中。

**去重**：非 R1-R4。

**修复建议**：续捉判定改为按受害子身份（`chase_target_code`）在**新格子**上匹配；需先与用户确认规则口径（可能需要平台原文）。

---

## P2-7【中低·规则】`fast_outcome()` 与 `game_outcome()` 优先级不一致（将死/困毙可被和棋阈值遮蔽）

**位置**：`jieqi.py:1019-1041`（fast 从不检查无合法着法）vs `jieqi.py:989-1017`（game：无合法着法最优先）。调用路径：`azero.py:968/991`（在 `leaf_moves` 空判断之前）、`ismcts.py:284-293`。

**复现**：
```pwsh
python ...\rules_probe\p3_repro_stalemate_draw.py
# FEN: 5x2e/1e3k3/4p4/9/9/4p3p/9/4K1c2/3r5/9[A2E2R2C2H2P5a2r1c1h2p1] w - - 80 1
# → legal_moves=[]; game_outcome=(-1,'stalemate'); fast_outcome=(0,'no_capture')
```
自然命中率极低：4000 局随机对局 3 次（0.075%）；红队 300 局随机对局 0 命中（无吃子计数到达 80 的那一步本身就会终结对局，需"和棋阈值与无着可走恰好同步"）。训练最终标签用 `game_outcome` 不受影响。

**置信度**：高（语义不一致 + FEN 精确复现）；实际频率低。

**去重**：非 R1-R4；ADR-0004 第 5 条要求 fast/game 语义一致。

**修复建议**：`fast_outcome` 无法便宜地检测无合法着法时，至少让 `game_outcome` 的优先级检查在调用方先于 fast（搜索叶需在 fast 判和后仍检查 `leaf_moves` 是否有将死）；或文档/注释明确二者非严格等价并补测试固化边界。

**回归建议**：将死/困毙与 no_capture/repetition 同触发的 FEN 断言两函数语义一致（或按修复后的约定）。

---

## P2-8【中低·规则】合法性/终局"查询"消耗 `state.rng`，对局揭晓随查询次数改变

**位置**：`jieqi.py:934`/`960`（禁着激活时 `apply_move(move)`，`rng` 缺省落 `self.rng`（`jieqi.py:461`））→ 暗子 `_draw_from_pool` 推进随机流；`undo_move`（`jieqi.py:513-542`）只恢复池计数，不恢复 rng（全文件无 getstate/setstate）。

**复现**：
```pwsh
python ...\rules_probe\p7b_clean_repro.py
# → "no query -> reveal 2 | after one legal_moves() -> reveal 4"；构造 FEN 下合法性可随随机流变化
```
自然禁着率约 0.002%（红队 200 局 55787 ply 仅 1 ply 激活）；但机制上"查询函数修改游戏状态"，破坏同 seed 复现（GUI 轮询频率会影响实际揭晓）。

**置信度**：高（机制），频率低。

**去重**：R3/R4 处理的是搜索内 rng 用法，未涉及规则查询副作用。

**修复建议**：`_iter_legal_moves`/`_move_scan` 的探测用 `apply_move(move, rng=临时 rng)`（或保存/恢复 `self.rng.getstate()`），保证查询无副作用。

**回归建议**：任意查询前后 `state.rng.getstate()` 不变；同 seed 加/减 `legal_moves()` 调用不改变后续揭示。

---

## P3-9【低·健壮性】参数边界无校验导致除零

**位置**：`azero.py:1805`（`(iteration+1) % config.eval_interval`；`--eval-interval 0` 崩溃，`parse_args` 无 >0 校验，对比 `strong_self_play_interval` 有守卫）；`azero.py:1033-1035`（已终局根 / `sims=0` 时 `total_visits==0` → `ZeroDivisionError`；`tools/eval.py --sims 0` 亦可达）。

**复现**：
```pwsh
python ...\train_probe\probe8_eval_interval_zero.py
python ...\mcts_probe\probe_misc.py
```

**置信度**：高。**去重**：未登记。

**修复建议**：`parse_args` 校验 `--eval-interval/--sims>0`；`search` 在 `total_visits<=0` 时返回空 policy/None（与 `no root.children` 同路径）。

---

## P3-10【低·数据】`DATA_VERSION` 写了不读

**位置**：`azero.py:31` 定义、`1102` 写入 payload；`load_compact_payload`（`1144-1152`）/`load_selfplay_data` 均不校验。

**复现**：
```pwsh
python ...\train_probe\probe1_data_version.py   # version=999 照常加载
```

**影响**：未来增删 `CompactSample` 字段会静默互载（slots 缺字段 → 延迟 `AttributeError` 或不一致语义）。

**置信度**：高。**去重**：ADR-0008 只要求 `ruleset_version`。

**修复建议**：`load_compact_payload` 校验 `version == DATA_VERSION`（未来升版时提供迁移或拒绝+告警）。

---

## 已排查但排除的疑点（为何不是 bug）

| 疑点 | 排除理由 |
|---|---|
| `_backup` 视角翻转、chance 边不翻转 | 7 种普通/机会交替拓扑 + 桩网符号实验与手算一致；`root.visits==sims`、`Σchild.visits==sims` 实测成立（同 R4 记录） |
| VL 改变搜索统计 | ADR-0005 明确承认的非严格等价，非缺陷 |
| `_evaluate_batch` 掩码/补齐 | `pending` 只含实际模拟，无 padding；批序确定 |
| Dirichlet 注入点与 `pruned_policy_from_root` | 仅根注入、只作用合法着法；剪枝语义与 KataGo 一致；`sqrt(total)` vs `sqrt(total+1)` 仅常数差异 |
| 镜像增强/动作空间 | 3 个真实数据文件、298 万样本：`mirror∘mirror=id`、编码=左右翻转、镜像 policy 落在镜像合法集、`LEGAL_MOVE_TABLE` 镜像封闭 |
| 64 通道索引 | 逐通道扰动核对与规格一致；历史平面不足 8 步不越界；池类/连将平面 side 顺序相反仅为风格 |
| 序列化往返 | 727 文件/297.8 万样本池不变量、policy 范围、value、坐标全部干净；`_CompatUnpickler` 兼容旧格式 |
| Undo/clone/FEN 字段完整 | 400 局×150 ply 逐字段对拍 0 失配；重复边界（第 3 次判和）正确 |
| 马腿判定 | ADR-0009 修复后四实现一致；残留"王沿横线当车"误报只在非将帅格，引擎不变用 |
| R1 确定化采样 | 已证伪（超几何精确），未重报 |
| R2 编码缺 chase/repetition | 已登记未修；本次仅审查其与镜像/序列化交互（未发现额外 bug） |
| 机会节点 k 截断重归一化 | `azero.py:615-619` docstring 明示"概率在采样结果上归一化"，是设计近似（k=4 单暗子实测平均丢弃 28.7%，复合吃子 84.5%），非实现错误 |
| 暗象无河界 / 暗士威胁表不过九宫 | 代码事实存在（`jieqi.py:750-756`、`575-578`），但自然对局不可达（暗子一步即翻开），`is_square_attacked` 独立正确，无消费者影响；属外部构造/移植潜缺口 |
| eval worker 数影响结果 / Wilson CI / unfinished 分母 / scheduler 快进 / buffer 淘汰与镜像 | 逐项对照 ADR 并通过 1 vs 2 workers 逐字段对拍、fast-forward 奇偶性实验，均正确 |
| GUI 暗池泄漏 | S1.2 修复有效：人类视角只显示未知被吃，AI 吃子按吃方显示真类型，终局才揭示真池 |
| `merge_selfplay_files` 无版本来源合并后静默跳过 | 仓库无 merged 产物；口径隐患而非 bug |
| `test_azero.py:773` 注释把通道 37 标为"红车"（实为红象） | 测试逻辑正确，仅注释错标，可顺手修 |

## 不确定性 / 反证

1. **P0-1 的量化影响未完成**：真实模型翻转证据充分（400/800 sims），但"policy target 噪声 → 强度损失"未做修复后 A/B；进度 §3b 的 A/B 结论（lr 1e-5 胜出、chance-k4 淘汰）产生于该缺陷之下，**建议修复后重跑受影响 arm 再解读**。
2. **P2-7/P2-8 的频率仅由随机对局估计**：策略驱动的长将/长捉局面频率、GUI 轮询复合影响、自博弈残局中"困毙与无吃子同步"概率均未量化。
3. **P0-3 的触发前提**：只证明了"产物损坏 → 崩溃/静默降级"和非原子写；实际发生频率取决于非正常终止/磁盘故障。
4. **P1-5 的统计影响**：配对失效对 Wilson CI 覆盖率与符号检验功效的损失未做蒙特卡洛量化。
5. **P1-6 的规则裁定**：天天象棋官方原文不可得，受害子移动是否算"同一目标"存在歧义；但实现与 ADR-0003 字面不一致是确定事实。
6. **P1-4 的暂停语义**："暂停时完成在途思考"可辩解为产品选择，但**旧局回调在新局落子**无辩解空间。
7. **机会节点 k 截断**（已排除为设计）：若启用 `chance_nodes` 训练，其机会先验偏差对 Stage 2A 信号的影响未量化。
8. **未覆盖面**：`gui.py` 全量交互、`tools/bench.py`、`merge_selfplay.py`、C++ 端口复刻一致性、模型膨胀/推理性能。

## 探针索引（按缺陷）

| 缺陷 | 探针 |
|---|---|
| P0-1 | `mcts_probe\probe_root_view_minimal.py`、`probe_leaf_leak.py`、`probe_realnet_400.py`；`redteam_probe\c1_probe.py`、`c1_realnet.py` |
| P0-2 | `redteam_probe` C4 两步复现；`train_probe\missing_model.jsonl` |
| P0-3 | `train_probe\probe2_truncated_file.py`、`probe11_silent_random_resume.py`、`probe12_corrupt_state.py`；`redteam_probe\c5_probe.py`、`c6_probe.py` |
| P1-4 | `encode_probe\p4_gui_thread.py`、`p7c_gui_stale_clean.py`；`redteam_probe\c8_probe.py` |
| P1-5 | `train_probe\probe5_pair_shuffle.py`；`redteam_probe\c7_probe.py` |
| P1-6 | `rules_probe\p13c_flee.py`；`redteam_probe\c9_probe.py` |
| P2-7 | `rules_probe\p3_repro_stalemate_draw.py`；`redteam_probe\c2b_probe.py` |
| P2-8 | `rules_probe\p7b_clean_repro.py`；`redteam_probe\c3_probe.py` |
| P3-9 | `train_probe\probe8_eval_interval_zero.py`；`mcts_probe\probe_misc.py` |
| P3-10 | `train_probe\probe1_data_version.py` |
