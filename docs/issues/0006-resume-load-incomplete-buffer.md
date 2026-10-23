# Issue 0006：续训时历史自博弈数据加载不完整（与回放缓冲容量脱节）

- 状态：**已修复**（2026-09-18；`load_selfplay_data` 默认按 `replay_buffer_size` 自适应回填）
- 发现：S1.20 chance-aware A/B 实施记录（`docs/experiments/0002`）
- 关联：`azero.py::load_selfplay_data`、`AZConfig.max_load_files`、`--max-load-files`、
  `azero.py::ReplayBuffer`、`docs/progress.md` S1.21

## 问题（根因）

续训启动时 `load_selfplay_data()` 把 `data_dir/selfplay/*.pkl.gz` 灌回内存回放缓冲
`ReplayBuffer(capacity=replay_buffer_size)`。旧实现只支持 `max_files`：

```python
files.sort(key=os.path.getmtime)
if max_files > 0 and len(files) > max_files:
    files = files[-max_files:]   # 只取最近 N 个文件
```

`max_files` 与 buffer 容量没有任何关系：

- 实测 `data/ab_short/control/selfplay` 有 64 个迭代文件（原始 256,460 样本，
  augment 镜像 ×2 = 512,920）。续训配置里写着 `--max-load-files 32`（从早期批次
  复制而来，当时文件数 ≤32 恰好等于全量），于是只重建了 254,288 个样本。
- 结果：buffer 从 254k 起步而非 512k，续训后的缓冲轨迹与"连续训练"不一致
  （后者为全部历史数据按时间序灌入、`ReplayBuffer` 超容量时 FIFO 淘汰最旧），
  破坏 A/B 对照的公平性。数据文件本身完整，缺陷纯在加载路径。

## 修复语义

`load_selfplay_data(..., max_files=0)`（默认，CLI 不传即为 0）：

1. 文件按 mtime 从**新到旧**扫描，累计样本数（`augment=True` 时每文件 ×2）
   达到 `buffer.capacity` 即停止，更旧的文件不再反序列化；
2. 选中文件再按**旧→新**顺序 `add_game`，由 `ReplayBuffer` 的 FIFO 截断
   （`data[-capacity:]`）保留最新的容量内样本——与连续训练的最终保留集合一致；
3. 规则版本不符 / 损坏的文件被跳过，**不消耗预算**（预算按实际加载成功计数）；
4. 返回值 = `min(实际灌入样本数, capacity)`；容量大于全部数据时为全量。

向后兼容：显式 `max_files > 0` 时保留旧语义（只取最近 N 个文件、不做容量裁剪、
返回全部灌入数），历史配置可原样复现。

不新增 `--load-max-samples` 之类的独立预算参数——直接复用 `replay_buffer_size`
即可表达"重建到连续训练同款缓冲"，参数越少语义越清晰；如需减少加载量，显式
调小 `--replay-buffer-size` 或用 `--max-load-files N`。

## 验证证据（2026-09-18）

- 新增 3 个单测（`test_azero.py`）：容量自适应保留最新、augment 预算 ×2、
  `max_files>0` 旧语义回归（含"旧语义下加载数可超容量"断言）。
- 全量：`python -m pytest test_azero.py test_eval.py test_ismcts.py test_jieqi.py -q`
  → **136 passed**。
- 真实数据验收（`data/ab_short/control/selfplay`，64 文件）：
  - `capacity=1_000_000, augment=True, ruleset_version=3` → **loaded == 512,920
    == len(buffer)**（修复前 254,288），耗时 21.5s；
  - `capacity=200_000` → loaded == 200,000（从最新文件回填，未读全部文件）；
  - `max_files=32` → loaded == 254,288（旧语义保持）。

## 旧配置迁移建议

- `tools/ab_chance128.json` 等历史配置里的 `"--max-load-files", "32"` 是复制粘贴
  产物，**不要删改历史配置**（其对应的 ab_short/ab_chance128 结果已出且不重跑）。
  未来新实验不要带该参数；默认的容量自适应即可。
- 历史实验（`ab_short`、`ab_chance128`）的结论不变、不重算：修复只影响**未来**
  训练的启动加载路径，不改变已产出的模型、数据与评估数值。
- 仅当刻意复现"温缓冲减半"等历史条件时，才显式传 `--max-load-files N`。
