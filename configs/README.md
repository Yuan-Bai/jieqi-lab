# 评估对手池清单

这里的两份 JSON 是 2026-10-22 历史实验的冻结清单：旧基准为 800 模拟/叶批 64，
新基准为双方 1600 模拟/叶批 16。它们保留原哈希与相对路径，所引用的模型**未随仓库分发**，
不能在新克隆中直接运行，也不应把历史哈希替换后继续沿用原清单 ID。

使用自己的模型时，新建 `data/eval/my_pool.json`（`data/` 已被 Git 忽略），例如：

```json
{
  "version": 1,
  "id": "my-rules5-pool-v1-1600-leaf16",
  "ruleset_version": 5,
  "search": {"mcts_simulations": 1600, "leaf_batch_size": 16},
  "models": [
    {"path": "models/reference-v1.pt", "sha256": "<替换为实际文件的64位小写SHA256>", "sims": 1600}
  ]
}
```

`path` 相对清单所在目录解析：上例对应 `data/eval/models/reference-v1.pt`。
先将自己有权使用且来源可信的历史纯权重冻结到新文件，核对训练规则版本与来源记录，再计算哈希：

```powershell
(Get-FileHash -Algorithm SHA256 data/eval/models/reference-v1.pt).Hash.ToLowerInvariant()
```

把结果填入清单。可按相同格式添加其他冻结模型；不要使用随训练覆盖的 `latest.pt` 或 `best.pt`
作为池成员。历史清单里的哈希不能用于你自己的权重。
运行参数与 `search` 不一致、文件缺失或哈希不匹配时，程序会中止，不会回退 RIS。

候选模型单独通过 `--model` 指定。配对协议、训练接入和可选 Pika 对手见[评估说明](../docs/evaluation.md)。
