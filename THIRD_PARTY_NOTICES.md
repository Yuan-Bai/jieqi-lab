# 依赖与素材来源

本项目以 [MIT License](LICENSE) 发布；本文件记录本仓库的第三方依赖与素材来源边界，
不替代上游许可。

## 随源码提供的素材

| 文件 | 可核对的来源 |
|---|---|
| `assets/sounds/*.wav`（7 个） | `gui.py` 中 `_tone`、`_noise`、`_build_*`、`_write_wav` 与 `SoundManager.BUILDERS` 合成；无外部音频采样 |
| `web/app.js` 的界面音效 | Web Audio 振荡器与本地噪声合成，不下载音频 |
| `web/favicon.svg` | 内联矩形与“揭”字；使用系统 serif 字体，不附字体文件 |
| `examples/synthetic-branches.json` | `tools/generate_example_record.py` 生成的合成棋谱；无真实玩家或平台数据 |

上述记录提供生成链，不能替代维护者的权利确认。可选素材网站或论文链接仅是引用，
不表示其内容随项目分发或受本项目未来许可证覆盖。

## 外部依赖

依赖通过用户环境或包管理器安装，不作为 vendored 源码/二进制复制进公开快照。
各发行包及其捆绑库适用自己的许可和通知；重新打包时应核对实际安装版本中的许可文件。

| 依赖 | 用途 | 上游 |
|---|---|---|
| Python / Tkinter | 规则、搜索、Web 与桌面 GUI | [CPython](https://github.com/python/cpython)、[Tcl/Tk](https://www.tcl-lang.org/) |
| NumPy / PyTorch | 编码、神经网络、训练与评估 | [NumPy](https://github.com/numpy/numpy)、[PyTorch](https://github.com/pytorch/pytorch) |
| pytest / psutil | 测试、进程诊断与测量工具 | [pytest](https://github.com/pytest-dev/pytest)、[psutil](https://github.com/giampaolo/psutil) |
| Cython / setuptools | 可选原生扩展构建 | [Cython](https://github.com/cython/cython)、[setuptools](https://github.com/pypa/setuptools) |
| Node.js | 前端语法与单元测试 | [Node.js](https://github.com/nodejs/node) |

研究文档中的第三方引擎、论文及规则来源保留引用链接。抓取缓存、第三方全文/源码、
平台适配、截图、训练数据及模型未纳入本次源码候选。历史实验数字的可核验范围见
[发布流程](docs/public-release-preparation.md)，不据此声明领先、首创或官方认证。

## 可选外部评估引擎

`pika_engine.py` 通过独立进程的 UCI 协议连接兼容 `jieqi_old` 的 Pikafish 构建。
本次同步仅包含 Python 桥接与测试，不分发引擎可执行文件、上游源码或网络权重，也不自动下载。
使用者需自行准备有权使用的构建并核对其来源和随附许可；其许可不由本项目的许可证替代。
普通象棋构建不等同于该接口要求的揭棋构建，适配范围见[评估说明](docs/evaluation.md)。
