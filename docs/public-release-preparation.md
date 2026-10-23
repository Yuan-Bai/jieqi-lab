# 发布与验证流程

本页是维护者发布新版本时使用的公开清单。它只描述可公开复现的步骤；本地日志、模型、
训练数据与原始实验记录不进入仓库。

## 发布状态

- 当前版本：`0.1.0`（2026-10-23）
- 许可证：[MIT](../LICENSE)
- 仓库：<https://github.com/Yuan-Bai/jieqi-lab>
- 分发范围：284 个源码/文档文件；二进制只有 7 个由 `gui.py` 生成的 WAV 和 2 张界面
  截图（PNG）；不含模型、训练数据、原始评估记录或外部引擎。
- CI 验证（2026-10-23）：`Tests` 的 `stdlib-and-web`、`python-cpu`（Linux/Windows）
  和 `native-backends` 全绿；CodeQL 的 Python 与 JavaScript/TypeScript 分析通过。

## 发布前检查

- [x] 维护者确认代码与素材权利，采用 MIT License，并加入 [LICENSE](../LICENSE)。
- [x] [依赖与素材来源](../THIRD_PARTY_NOTICES.md)记录 7 个音效与源码生成器逐字节一致。
- [x] 所有 `torch.load` 显式设置 `weights_only=True`；发布依赖下限为 PyTorch 2.10，
  [安全说明](../SECURITY.md)明确 Pickle 数据仅来自可信来源。
- [x] 本地 Web 的 GET/POST 校验可信 Host、实际监听端口和同源 Origin，拒绝重复请求头
  与跨站请求。
- [x] [CI](../.github/workflows/ci.yml)覆盖 Linux/Windows CPU、标准库与 Web、以及
  Linux 原生后端；构建与跳过条件见[贡献说明](../CONTRIBUTING.md)。
- [x] 补 CONTRIBUTING、SECURITY、[版本说明](../CHANGELOG.md)和[可生成的合成棋谱](../examples/README.md)。
- [x] 首页与英文摘要标注历史报告日期、未分发输入及旧 Git 引用的限制；不把历史研究
  包装为当前领先性证据。
- [x] 在 GitHub 仓库 Settings → Security 启用 Private vulnerability reporting，
  并确认 SECURITY.md 中的报告入口可用。
- [x] 首次推送后确认三个 CI job 全绿；远程工作流结果以 Actions 页面为准。

## 本地验证

最近一次 Windows 本地验证（CPython 3.11.8、PyTorch 2.10.0+cpu、NumPy 2.0.2、
pytest 7.4.0、psutil 7.2.2）：

| 检查 | 结果 |
|---|---|
| 完整 Python 套件 | **713 passed / 60 skipped**；另有 13 项需 Windows 命名管道的用例在受限沙箱中失败，应在普通 shell 或 CI 复核 |
| 跳过项 | 29 项可选原生扩展、30 项隔离实验扩展、1 项 CUDA；这些能力未计作通过 |
| 前端 | `node --check web/app.js` 通过；3 个 Node 测试文件共 **10 passed** |
| 标准库运行 | `python -S` 导入规则、RIS、棋谱、搜索适配和 Web 成功；不加载模型 |
| 干净副本 | 仅复制非忽略候选文件后，核心规则/搜索/Web 测试 **146 passed** |
| 文档链接 | 所有本地 Markdown 链接均存在（以发布前最后一次链接检查为准） |
| 合成棋谱 | 生成器一致性、完整导入/导出及 Web 私有吃子过滤通过，包含在全量套件内 |
| 候选清单 | 284 个非忽略文件；二进制为 7 个可追溯 WAV 和 2 张界面截图；模型、数据与构建产物均未进入候选 |

沙箱中的原子写入/硬链接/命名管道失败属于执行环境限制，不应通过削弱保存机制或关闭
断言来“修复”。发布前请在普通 shell 或 CI 重跑，并以实际结果为准。

## GitHub 发布步骤

1. 确认 `git status` 中没有 `data/`、模型、训练数据、缓存或本地发布记录。
2. 首次提交并推送；等待 `Tests` 工作流的 `stdlib-and-web`、`python-cpu`、
   `native-backends` 三个 job 全绿。
3. 在仓库 Settings 中设置 Description 与 Topics（如 `jieqi`、`chinese-chess`、
   `alphazero`、`mcts`、`python`、`tkinter`）。
4. 启用 Private vulnerability reporting、secret scanning 与 branch protection；
   要求 PR 通过 CI。
5. 打 annotated tag `v0.1.0`，以[版本说明](../CHANGELOG.md)为草稿发布 Release。

## 运行入口

纯搜索网页：`python web_server.py`，打开 <http://127.0.0.1:8765>。
GUI：`python gui.py`。神经网络训练需要安装 `requirements.txt` 中的依赖，并自行准备
可信模型/数据。仓库不附预训练模型；首次体验可使用 RIS。完整训练不是发布流程的自动步骤。

历史 ADR、研究结论与带日期的审查报告保留在 `docs/`，其中的数字与私有工作区提交只代表
当时环境，不能作为当前棋力、性能或可复现性的保证。
