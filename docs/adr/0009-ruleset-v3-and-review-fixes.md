# ADR-0009：RULESET_VERSION = 3（马腿判定与按子长将/长捉修正）

- 状态：已接受
- 日期：2026-09-11
- 相关：ADR-0002（版本管理）、ADR-0003（禁着语义）；Stage 1 验证前的代码审查

## 背景

Stage 1 验证前的全面审查发现两处规则引擎缺陷（均有最小复现与量化证据）：

1. **马腿判定方向错误**：`is_square_attacked` 用“目标格”方向查马腿，而 `iter_pseudo_moves` 用“攻击方”方向。
   二者在 300 盘随机对局 / 173,038 次将帅格评估中产生 675 次（0.39%）不一致，其中 347 次（0.20%）
   改变“轮到方是否被将军”的判断。S1.5 的快/慢路径对拍无法发现（两条路径都调用同一个错误函数）。
2. **长将/长捉未按“同一枚棋子”判定**：`consecutive_checks` 对任何将军都累加；`_update_chase`
   在换子接手后仍延续同一目标的计数。与 ADR-0003 定义的“同一枚棋子”不符，会过度禁着。

两处缺陷都会进入合法着法集合，进而污染训练数据。

## 决策

1. 修正马腿判定：新增 `KNIGHT_ATTACKS[target] = [(attacker, leg), ...]` 反向索引，
   `is_square_attacked` 改为按攻击方马腿判定；补充假阳性/假阴性回归测试。
2. 修正按子判定：新增 `check_anchor` / `chase_anchor`（该棋子最近一次落点），只有同一枚棋子
   继续将军/追同一目标才累加；换子接管重置为 1。`Undo`、`clone`、`reset_history`、FEN 序列化同步该状态。
3. **`RULESET_VERSION` 由 2 升至 3**；依据 ADR-0002/0008，版本不匹配的数据/模型不加载。
4. v2 产物（6 个迭代文件共 384 局、`latest.pt`、`train_state.json`、smoke 评估记录）归档至
   `data/legacy_ruleset_v2_20260911/`（归档不删除），训练从 iteration 0 重新开始。
5. 同批审查还修复了若干非规则问题（不升 `RULESET_VERSION`，但影响搜索/评估正确性）：
   RIS-ISMCTS 回传视角改为按节点执棋方记账（消除 rollout 奇偶性导致的值反转）；
   确定化时携带长将/长捉/重复等规则历史；merge 产物 meta 携带 `ruleset_version`；
   Dirichlet 噪声改用实例 RNG（同 seed 可复现）；`ply_limit` 未终局对局不计分；
   `model:<path>` 支持 Windows 盘符；FEN 恢复长将/长捉/重复计数；GUI 搜索移出 Tk 主线程。

## 后果

- 规则引擎与天天象棋基准进一步对齐；长将/长捉仅在真正“同一枚棋子”继续时累加。
- 旧 v2 自博弈数据与模型不可复用；Gate 从零开始（v2 数据仅约 40 分钟，重跑成本低）。
- 新增回归测试：马腿假阳性/假阴性、按子长将、按子长捉、FEN 规则状态往返、
  RIS 奇数 rollout 回传视角、Dirichlet 实例 RNG、merge 版本 meta、tally ply_limit。
