# 固定留出诊断与续训说明（2026-10-02）

实现已接入训练。用户实验 `data/train_fixed_20261001` 仍完成5轮、当前停止；正式留出集将在下一轮新自博弈中建立。过去五轮不能作为独立留出。

## 先新增一轮

在项目根目录运行；沿用原核心参数和现有目录，新增诊断和本次限轮参数：

```powershell
python azero.py --iterations 2000 --run-iterations 1 --diagnostic-games 32 --diagnostic-batch-size 128 --self-play-games 1024 --mcts-simulations 800 --strong-mcts-simulations 3200 --strong-self-play-interval 8 --chance-nodes --chance-k 0 --eval-interval 20 --eval-games 100 --eval-mcts-simulations 800 --ris-iterations 800 --lr 5e-5 --lr-schedule cosine --train-steps 1000 --self-play-workers 8 --self-play-log-interval 256 --replay-buffer-size 4000000 --data-dir data/train_fixed_20261001 --model-dir data/train_fixed_20261001/models --log-dir data/train_fixed_20261001/logs
```

从现有完整检查点恢复AdamW、cosine、RNG和缓冲，不需要重新复制权重或初始化新目录；不添加 `--no-resume` 或 `--reset-optimizer`。总计划保留2000；本次只新增1轮（索引5），完成后 `train_state.iteration=6` 并正常退出。按近期日志约2小时18分，加上诊断开销；正式留出推理开销尚未测量。

本轮1024局中一次性留出32局，约3.1%；其余992局进入训练。固定留出包含4局高强度、28局普通搜索，对局全部结束后划分、再增强。另32局训练参照正常进入训练，原先五轮缓冲保留。之后每轮生成的1024局均参与训练，继续使用首次保存的固定目标。

## 查看结果

日志新增“固定诊断已建立”“固定诊断”“样本复用”。完整数据和报告位置：

- `diagnostics/sets/<建立轮attempt_id>/heldout.pkl.gz`
- 同目录 `training_reference.pkl.gz`
- `diagnostics/reports/<每轮attempt_id>.json`
- `models/training_checkpoint.pt` 嵌入固定集清单和最近报告，续训自动恢复。

这些路径均位于 `data/train_fixed_20261001` 内。只读核对诊断及提交状态：

```powershell
python tools/diagnose_training.py --data-dir data/train_fixed_20261001 --checkpoint data/train_fixed_20261001/models/training_checkpoint.pt
```

首次建立之前返回dataset=null、reports=[]，是正常状态。之后检查 `status=committed`，并只在相同 `dataset_sha256` 下比较曲线；中断轮可能有unconfirmed报告或孤立数据，恢复不会自动采用。最近报告被中断写出时可以从完整检查点恢复。

确认新增一轮正常后，把原命令的 `--run-iterations 1` 改为 **2至4**，其他参数保持不变；累计新增3至5轮，即总完成8至10轮，再检查趋势。也可以先提供第六轮日志确认通路。`--diagnostic-games 32` 保留即可；已有集时省略也会自动恢复，不会每次再抽32局。

## 如何解释

| 观察 | 含义与下一步 |
|---|---|
| 策略目标熵固定，KL下降 | 网络更贴近该批搜索策略；比只看动态训练CE更可比 |
| 参照改善，留出连续恶化 | 提示泛化或优化问题；检查更新次数/LR，不能凭单轮噪声决定 |
| 参照与留出都改善 | 固定目标拟合改善；仍需要同口径对弈验证棋力 |
| 两边指标突变 | 检查BN统计、优化稳定性、数据与提交状态，先确认原因再扩网 |
| value MSE降低但校准仍差 | 按预测区间核对偏差，结合常数/零预测基线判断 |

指标按局面加权，同一局的局面有相关性。32局不是大样本验收，固定目标也会随训练变成历史分布。固定训练参照最终可能被FIFO淘汰；长期差距不宜直接称为过拟合。算法近似和棋力评估边界保持不变。

设计、回归和隔离CUDA短测见 [ADR-0018](../adr/0018-fixed-training-diagnostics.md)。短测使用历史诊断素材，只证明路径可用；本次没有运行上述长训命令，也没有修改已有训练产物。
