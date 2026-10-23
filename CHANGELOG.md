# 版本说明

## 0.1.0 — 2026-10-23

首个公开源码版本，以 [MIT License](LICENSE) 发布。

### 包含

- 独立 Python 规则引擎、RIS-ISMCTS、AlphaZero 风格训练、Tk GUI、本地 Web 与分支复盘。
- 规则版本 5、搜索版本 3、数据版本 3、网络输入 238 通道；棋谱 JSON 使用
  `jieqi-replay` 格式 v2。这些协议版本不是软件发布版本。
- 公开版本排除旧 Git 历史、平台适配、模型、训练数据、抓取缓存和编译文件。
- 纯权重读取显式使用 `weights_only=True`，依赖下限为 PyTorch 2.10；
  数据 Pickle 的可信来源要求见 [SECURITY](SECURITY.md)。
- Web GET/POST 校验可信 Host、实际端口和同源 Origin；使用非本机绑定前请阅读
  [Web 说明](docs/web.md)。
- CI 覆盖标准库/Web、Linux/Windows CPU 全量测试与 Linux 原生后端；
  跳过条件见 [CONTRIBUTING](CONTRIBUTING.md)。
- README 与英文 README 增加网页版和 GUI 界面预览截图。
- 同步冻结历史对手池、Pika 信息态/UCI 桥接、整对 bootstrap 区间和五类配对统计。
- 修复并行评估出错后等待其他对局的问题；及时检查任务错误，清理本次评估所属进程，
  保持成功结果的配对顺序，失败时不生成完成标记。
- 训练默认仅记录评估，不再按 RIS 最高分自动更新 `best.pt`；旧行为需显式
  `--legacy-ris-best`。评估默认双方 1600 模拟/叶批 16，自博弈默认值不变。
  清单、用法和兼容边界见[评估说明](docs/evaluation.md)。

### 说明

历史训练、性能及棋力报告是带日期的实验记录；原始输入未随源码分发，不能将旧数字
作为本版本的复现结果。发布与验证流程见 [docs/public-release-preparation.md](docs/public-release-preparation.md)。
