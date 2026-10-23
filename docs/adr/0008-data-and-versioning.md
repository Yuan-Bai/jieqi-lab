# ADR-0008：数据与版本管理（ruleset_version；归档不删除）

- 状态：已接受
- 日期：2026-09-11
- 相关：需求访谈 Q9；实现 `TrainingState` / `load_selfplay_data`

## 背景

规则从 version 0（legacy）变更为 version 2（天天象棋基准）属于语义不兼容修改：用旧规则生成的自博弈数据训练的模型，其策略目标与新合法着法集合可能冲突。需要版本化机制防止混用，同时保留旧数据备查。

## 决策

1. **版本标记**：
   - `train_state.json` 增写 `ruleset_version`（缺省视为 -1/legacy）；
   - 自博弈数据文件（`iter_*.pkl.gz`）的 meta 中写入 `ruleset_version`；
   - 网络模型随训练状态使用（resume 时校验版本）。
2. **加载策略**：`load_selfplay_data(..., ruleset_version=N)` 跳过不匹配文件并 `logger.warning`；resume 时若 `train_state` 版本不匹配：警告、重置训练状态、不加载旧模型与旧数据（从零训练）。
3. **归档而非删除**：旧模型/数据移动到 `data/legacy_20260911/{models,selfplay}/`；`data/logs/train.log` 保留作历史。
4. **训练产物不入库**：`.gitignore` 排除 `data/models/`、`data/selfplay/`、`data/eval/`、`data/logs/`、`*.pt`、`*.pkl.gz`、`*.7z`。
5. **Git 策略**：仓库已 `git init`；当前阶段只做本地版本化，不提交（是否提交由用户决定）。

## 后果

- 规则升级流程固定为：改规则 → 升 `RULESET_VERSION` → 更新 ADR/测试 → 归档旧产物 → 从零训练。
- 误用不兼容数据/模型会被自动拦截而非静默出错。
