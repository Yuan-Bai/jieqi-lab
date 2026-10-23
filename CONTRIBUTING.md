# 参与开发

本项目以 [MIT License](LICENSE) 发布。提交贡献即表示你同意以同一许可证授权，并同意
在提交信息中使用 `Signed-off-by` 声明（Developer Certificate of Origin）。发布与验证
流程见[发布流程](docs/public-release-preparation.md)。

参与本项目即表示同意遵守 [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md)。

## 环境与检查

规则、RIS 搜索和 Web 运行只需要 Python 3.11+ 标准库。完整测试还需要 Tkinter
以及 `requirements.txt`（NumPy、PyTorch、pytest、psutil）。Node 24 用于前端检查，
网页运行不需要 Node 或 npm 安装。建议在独立虚拟环境中安装：

```powershell
python -m venv .venv
# Windows；Linux/macOS 对应 .venv/bin/python
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q -rs
node --check web/app.js
node --test web/preferences.test.cjs web/history.test.cjs web/view-state.test.cjs
```

测试位于 `tests/`，pytest 配置在 `pyproject.toml`；不要新增根目录 `test_*.py`。
先运行修改涉及的测试，再运行完整套件。`-rs` 会显示跳过原因；缺少扩展、CUDA、
Tk 显示环境或 Windows 控制台的跳过，不代表这些功能通过。
Windows 沙箱可能拒绝硬链接、原子替换或子进程操作，应区分权限问题与真实回归，
不要为了测试通过削弱检查点保存机制。

## CI 覆盖范围

[工作流](.github/workflows/ci.yml)配置如下：

| 任务 | 环境与检查 | 未覆盖范围 |
|---|---|---|
| stdlib-and-web | Linux、Python 3.11；禁用 site-packages 的导入检查；规则/Web 单测；Node 24 语法与测试 | 神经网络、真实浏览器交互 |
| python-cpu | Linux/Windows、Python 3.11、PyTorch 2.10.0 CPU；完整 pytest；Linux 使用 Xvfb 运行 Tk 控件测试 | CUDA；未构建的原生后端；Linux 跳过 Windows 控制台集成 |
| native-backends | Linux、Python 3.11、Cython 3.3.0；构建规则、合法着、节点扩展并检查身份；原生专项测试要求零跳过 | Windows 编译工具链、GPU 性能、棋力 |

CI 输出解析后的依赖版本与所有跳过原因。原生扩展是可选运行能力，但其专门任务会
在缺少/过期二进制时失败，不用“全跳过”替代验收。构建方法及回退选项见
[加速配置](docs/performance.md)。普通单元测试包含隔离的小型训练行为测试，
CI 不运行生产长训、棋力评估或性能采样。

## 修改边界

- 规则以 `jieqi.py` 为准；规则、数据、搜索与编码版本必须保持对应关系。
- `ismcts.py`、`azero.py` 的修改应保留行动方信息隔离、机会分支、RNG 与历史语义；
  搜索或性能收益需要单独证据，单元测试通过不能证明棋力。
- 检查点改动必须保留原子保存、完整续训和来源链；不要用真实训练目录做测试。
- `web_server.py` / `web/` 修改需检查会话、私有信息过滤、同源边界和棋谱往返。
- `data/`、模型、研究缓存、平台适配、编译产物与本地发布记录不进入源码提交。
- 生成内容优先修改生成器。示例棋谱来源见 [examples/README](examples/README.md)。

报告问题时附最小复现、操作系统、Python/依赖版本、规则与后端配置、预期/实际结果，
并移除模型、私有棋谱、身份字段和本机路径。安全问题按 [SECURITY](SECURITY.md)处理。
PR 请说明行为变化和实际执行的检查；未运行的环境、跳过项及兼容性限制应单独注明。
