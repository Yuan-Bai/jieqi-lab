# ADR-0017：Windows 训练启动阶段中断

- 状态：已实现并验证（2026-10-02）
- 实现：`training_runtime.py`、`azero.py`
- 验证：`test_training_interrupt.py`、`tools/probe_interrupt.py`
- 证据：[五轮日志与现场报告](../reviews/2026-10-02-five-iterations-and-interrupt.md)

## 问题

spawn 在 initializer 前导入主模块和 torch，此时 Ctrl+C 可终止 worker。父进程若同步写模型参数，可能阻塞且无法执行 Python 主线程信号回调。新 child 也可能尚未登记到 active_children。正常 executor 关闭、单纯清理已登记进程无法覆盖该窗口。

## 决策

1. torch 导入前调用轻量 `training_runtime.bootstrap(__name__)`；spawn 的 `__mp_main__` 提前忽略 SIGINT/SIGBREAK 和 Windows Ctrl+C。普通 import azero 不改变调用方信号/进程组。
2. 仅 Windows 训练 CLI 创建未命名 Job Object，将自身加入，后代自动属于此组；句柄不继承、设置 KILL_ON_JOB_CLOSE，外部停止父进程时也清理后代。使用 Windows 8+ 嵌套 Job，不影响启动它的终端或其他 Python 任务。
3. 原生 Ctrl+C/Ctrl+Break 回调直接 TerminateJobObject/ExitProcess，退出码130；禁止在回调中使用 logging/stdio、multiprocessing 锁、join、executor shutdown。安装 Python 信号处理器后重新注册原生回调。
4. 无控制台时保留 Job 的外部终止清理。无法建立/加入 Job，在开始训练前明确报错。非 Windows/库调用保留 Python 路径，补当前 spawning Popen、单个终止错误容忍、finally 硬退出。
5. 提交边界仍为原子完整检查点，中断可留下临时文件或未确认 attempt；不删除用户产物，不提交失败轮。Windows 原生退出可直接返回终端，省略旧 Python 中断提示。

仅保护 `python azero.py`，包括其训练内自博弈/评估 worker；独立 GUI、评估工具或库调用不自动启用原生 Job。规则/数据/搜索版本保持3/3/1，原训练参数与优化状态续训有效。

依据：[微软 Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)、[控制台事件](https://learn.microsoft.com/en-us/windows/console/ctrl-c-and-ctrl-break-signals)、[SetConsoleCtrlHandler](https://learn.microsoft.com/en-us/windows/console/setconsolectrlhandler)、[Python 3.11 signal](https://docs.python.org/3.11/library/signal.html)。

## 验证

隔离探针使用专门的隐藏控制台，只向该控制台发事件，输出目录必须是新路径。运行时仅标准库；探针/集成测试用可选 psutil，没有时跳过对应测试。

```powershell
python -m pytest -q tests/test_training_interrupt.py
python tools/probe_interrupt.py --out data/pilots/interrupt_new_test --scenario startup
```

12项测试覆盖轻量/无副作用导入、提前屏蔽、初始化失败/幂等、终止失败和未登记 child、fallback 退出、原生事件筛选/失败关闭，以及真实 Windows 启动阻塞、Ctrl+Break、stderr阻塞、外部停止后代清理。全量 **220 passed（50.71s）**；复现中约0.3s退出且无 worker 残留，不承诺所有系统/终端/库环境固定延迟。
