# ADR-0016：运行来源与轮次提交记录

- 状态：已实现并验证（2026-10-01）
- 实现：`run_records.py`、`azero.py`、`tools/eval.py`、`tools/run_status.py`
- 回归：`test_run_records.py`（17 项）、原完整续训测试
- 关联：[当前状态](../status.md)、[完整检查点](0015-complete-training-checkpoint.md)

## 问题

只有轮号和文件名，不能判断数据来自哪份权重、哪种搜索配置，或评估发生在优化前还是优化后。工作区未提交改动使 Git HEAD 也不足以识别实际代码。评估先于检查点提交，日志可能留下失败轮次；重新执行同一轮后，两条记录不能直接合并。

## 决策

每次进入训练循环创建独立 `run_id`；续训也创建新 ID，并记录父运行、恢复前检查点的文件 SHA-256 与实际加载模型的张量 SHA-256。`fresh`、`legacy_weights`、`full`、`reset_optimizer` 分开标记。仅加载旧文件时，父运行 ID 为 null、历史来源为 `unrecorded`，不推断其生成算法。

不可变 `manifest.json` 保存完整 AZConfig、网络架构、规则/数据/搜索版本、起始训练计数、起始重放文件清单、Python/Torch/NumPy/CUDA 与设备信息。代码记录包含 HEAD、dirty/porcelain 状态、运行 Python 源文件和工具配置的逐文件哈希及汇总哈希，覆盖未跟踪的运行源文件。Git 不可用时显式记录未知状态，文件内容哈希仍保留。

源文件清单覆盖根目录运行 `.py`、`tools/**/*.py`、`tools/*.json`、requirements/pyproject；它是代码身份记录，不是完整环境或源码归档。哈希不能恢复丢失的未提交修改，长期保留实验仍需自行备份对应源码。当前工作未进行 Git 提交。

每轮执行创建独立 `attempt_id`，包括生成模型张量哈希和轮前已提交检查点的文件哈希。轮次中断后重试使用新 attempt ID。模型张量哈希覆盖参数和 BN 缓冲、名称/形状/类型，不依赖 `torch.save` 的文件包装；旧 71 通道文件哈希与补零后 238 通道模型哈希分别记录。

- 自博弈文件的 `meta.provenance` 指向 attempt，记录生成模型。
- 训练内评估的 `provenance` 指向同一 attempt，另记 `evaluated_model_sha256`，对应优化后的模型。补录配对 base seed、pair 数及实际机会开关，完整搜索配置由 manifest 提供。
- 完整检查点嵌入 manifest、attempt、提交后的模型哈希、该轮数据文件哈希与评估摘要；核对配置/架构/版本/轮计数/数据链接和模型哈希，拒绝不一致的来源。
- 先原子提交完整检查点，再写 `commits/<attempt_id>.json`。提交记录包含该检查点文件哈希、模型哈希、完成轮数与数据文件引用，随后导出 latest/best/JSON。

元数据写入采用同目录独立临时文件、flush/fsync、原子排他链接。已存在且内容相同可幂等重读；不同内容明确拒绝覆盖。运行 manifest 深拷贝调用者输入，后续缓冲文件追加不会修改运行身份。ID 使用 OS 熵，不消耗训练 Python/NumPy/Torch RNG。

## 提交边界与兼容

完整检查点仍是训练提交依据。若检查点提交后、提交标记落盘前中断，该轮已经提交；不能仅凭标记缺失认定失败。只读状态工具可结合当前检查点识别此情况，续训在开始下一轮前从其嵌入的来源补回 manifest/attempt/标记。修复失败会停止继续写轮次，避免丢失最后提交的证明。

评估 JSONL 中 `status_at_write="unconfirmed"` 表示写出时尚未提交，不是之后的最终状态。查询 run/attempt 的提交记录；标记缺失且当前检查点也没有该 attempt 时为 `unconfirmed`，不能用于验收。旧无来源记录的数据/评估不能据此追认。历史已提交的标记会保留，即使文件已被 FIFO 缓冲淘汰。

schema version=1 的旧完整检查点没有来源字段，仍可完整恢复优化状态，新运行明确记录过去的来源未知。添加元数据没有改变搜索、编码、样本内容或纯权重模型格式，规则/数据/搜索版本保持 3/3/1。

独立 `tools/eval.py` 同样创建运行 manifest，并把来源附到 run/game/summary 记录。记录己方文件及加载后模型哈希、模型对手文件哈希、双方机会配置、预算/种子/最大步数/worker 数。串行或 worker 加载对手前后核对原文件哈希，文件改变明确报错，缓存已加载的原模型继续属于原来源。JSONL 完整关闭并 fsync 后才创建 `complete.json`，同时记录输出哈希。

独立评估 `complete` 只表示运行写完；逐局可能仍未终局，不能把它当作有效棋力验收。输出已存在时拒绝覆盖，默认文件名加随机后缀。CLI 调用者使用新输出路径；A/B runner 若复用旧输出也会明确失败。

## 位置与查看

训练默认记录位于 `data/runs/<run_id>/manifest.json`、`attempts/*.json`、`commits/*.json`。首次建立轮前检查点的提交记录为 `commits/bootstrap.json`。

独立评估记录位于输出 JSONL 所在目录的 `runs/<run_id>/`，并有 `complete.json`。例如默认 `data/eval/runs/`；run header 也嵌入 manifest。

```powershell
# 只读检查；旧主线当前尚无完整检查点，可先省略 --checkpoint
python tools/run_status.py --data-dir data

# 以后生成完整检查点后，同时核对最后提交但尚无标记的轮次
python tools/run_status.py --data-dir data --checkpoint data/models/training_checkpoint.pt

# 独立评估的运行记录；工具同时核对完成标记引用的输出哈希
python tools/run_status.py --data-dir data/eval
```

迁移机器时，除完整检查点及所需自博弈文件外，还保留 `runs/` 和评估 JSONL。检查点只能补回其当前嵌入的运行/attempt 记录，不能恢复全部历史提交记录。独立评估完成记录中的原输出路径是来源记录；搬迁后原路径不存在会显示 `output_missing`，不改写旧记录。输出被修改会显示 `output_changed`。

## 验证

- 17 项新测试覆盖 RNG 不变、不可变清单拷贝、权重/BN 哈希、未跟踪源码/Git 不可用、原子排他写入、生成/训练模型关联、新运行父链接、评估后提交失败及重试、提交标记失败后恢复、旧完整检查点和旧纯权重来源、优化器重置标记、来源损坏拒绝、独立评估完成/中断/输出改变、禁止覆盖和串行/worker 对手来源检查。
- 全量 `python -m pytest -q`：**208 passed（37.13s）**。包括原 CPU/CUDA 优化等价与连续/中断恢复测试。
- 真实旧主线模型的隔离双进程 CLI 检查：`data/pilots/provenance_20261001_a5760b35/`。两局、1 次搜索、每局最多两步；己方/对手文件哈希、补零后实际模型哈希、逐局来源、输出哈希与完成标记一致。两局均未终局，不给棋力结论。
- 主线 latest/best、训练状态及最近自博弈文件哈希未变；未运行主线训练或迁移。代码与记录位于工作区，未提交。

下一项为持久化训练诊断：按整局固定划分留出集，再增强；通过来源记录核对训练与留出数据隔离，记录策略 KL/目标熵、value 误差/校准和样本复用。
