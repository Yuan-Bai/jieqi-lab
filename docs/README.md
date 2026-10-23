# 文档索引

先读 [README](../README.md) 的快速开始；本目录保存当前状态、专题说明与带日期的研究记录。

## 当前状态与使用

| 文档 | 内容 |
|---|---|
| [status.md](status.md) | 当前工程状态、版本兼容与工作顺序 |
| [research-status.md](research-status.md) | 研究结论、停止清单与剩余边界 |
| [performance.md](performance.md) | 加速配置、构建方法、回退开关与证据索引 |
| [evaluation.md](evaluation.md) | 冻结对手池、配对评估、Pika 挑战 |
| [web.md](web.md) | 本地网页、HTTP API、Host/Origin 边界 |
| [gui.md](gui.md) | 棋谱树、变例研究、摆棋与棋谱兼容 |
| [public-release-preparation.md](public-release-preparation.md) | 发布与验证流程 |

## 决策与问题记录

- [adr/](adr/)：架构决策记录（规则、搜索、信息集、检查点、来源链、评估协议）。
- [issues/](issues/)：已修复或已记录的具体算法问题与验证结论。
- [experiments/](experiments/)：隔离实验的协议、结果与停止条件。

## 历史报告

- [progress.md](progress.md)：实现演变与历史实验日志。
- [reviews/](reviews/)：按日期归档的审查、基准与性能报告。
- [research_jieqi_azero.md](research_jieqi_azero.md)、
  [research_jieqi_search_training.md](research_jieqi_search_training.md)、
  [research_jieqi_training_scale.md](research_jieqi_training_scale.md)：调研笔记。

> 历史材料中的训练目录、模型、原始测量、私有 Git 提交与本机路径未随仓库分发；
> 日期数字只代表当时环境，不是当前棋力、性能或可复现性的保证。需要复现时，请用
> `tools/` 中的通用入口准备自己的可信模型与数据。

## 其他入口

- [tools/README.md](../tools/README.md)：工具分类与副作用边界。
- [configs/README.md](../configs/README.md)：自建评估池清单格式。
- [examples/README.md](../examples/README.md)：合成棋谱与生成器。
- [tests/](../tests/)：规则、搜索、训练、GUI、Web 与原生扩展测试。
- [pyproject.toml](../pyproject.toml)：pytest 配置（源代码直接运行）。
- [CONTEXT.md](../CONTEXT.md)：领域术语与关键常量。
