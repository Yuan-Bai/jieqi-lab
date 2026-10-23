# Cython 规则热点编译

## 范围与实现

在前两轮精确加速基础上，只把 `GameState.is_square_attacked` 接到 Cython 扩展。
原 Python 函数体未修改，保留为 `_is_square_attacked_python`，供回退和独立对照使用。
走法生成、走法顺序、合法性试走/撤销、长将/长捉追踪、MCTS、随机数和网络均不改。

`native/jieqi_rules.pyx` 使用 C 整数和编译后的循环，直接读取现有 Python 棋盘与几何表。
不复制棋盘，不缓存攻击结果，不释放 GIL，不跨进程共享可变局面。
保留索引边界检查；先掩码再转为 C 整数，避免 Python 大整数截断或溢出。
不启用浮点优化、低精度或 `fast-math`。非标准参数/棋盘容器走原 Python 路径。
保留当前攻击判定语义，包括王在射线中的判定，而非借此次加速修改规则。

本机使用 Python 3.11.5、Cython 3.3.0、MSVC 14.42 构建，运行时无需 Cython。
编译链见 [Cython 官方说明](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html)。

## 构建、选择与续训

```powershell
python -m pip install --target data/build_deps Cython==3.3.0
python tools/build_rules.py
python -c "import jieqi; print(jieqi.RULES_BACKEND, jieqi.RULES_BACKEND_ERROR)"
python -m unittest test_native_rules -q
```

编译生成的 `.pyd` 位于项目根目录，C/目标文件在忽略目录 `data/native_build/`。
新进程默认 `JIEQI_RULES_BACKEND=auto`：有效扩展自动启用，缺失或不兼容自动回退。
加载时检查 API、规则版本和 `.pyx` SHA256，过期扩展不继续使用。
没有运行时自动编译、自动下载或全局依赖安装。

```powershell
# 原实现：仅影响随后启动的进程
$env:JIEQI_RULES_BACKEND="python"
# 强制验证启用，缺失/过期扩展则报错
$env:JIEQI_RULES_BACKEND="cython"
# 恢复默认自动选择
Remove-Item Env:JIEQI_RULES_BACKEND
```

规则/搜索/数据版本不变，当前 v5 完整检查点和 buffer 继续使用，不需要重训。
已有训练进程不会热切换；正常结束当前运行后，用原命令重新启动。
新运行记录增加原生源/二进制哈希和实际后端，旧检查点仍可读取。
后续修复来源记录边界：加载时保留实际模块 `__file__` 的绝对路径、SHA256、大小及
API/规则/源版本；manifest 的 `runtime.rules_backend_native` 与 `code.files` 包含该实际文件，
即使来自外部搜索路径或已预加载，也不只枚举根目录二进制。外部文件在 `code.files` 中用绝对路径键。
如果加载后的文件被替换，创建新运行记录会明确拒绝，需重启 Python，避免记录磁盘新文件而执行旧代码。
默认 pytest 配置同时排除 `data`，避免收集本地安装 Cython 的 GDB 调试器测试。
修复验证见 `docs/reviews/2026-10-05-acceleration-code-audit.md` 文末；历史测速结果不覆盖或重写。

## 验证边界

`test_native_rules.py` 的直接攻击对照覆盖 300 个随机棋盘以及逐棋种、双方颜色、
明暗、来源/目标和阻挡组合，合计超过 50 万组；包括 Python 无界整数编码。
另检查边界索引、非法参数错误类型、非 list 容器、12 种种子至多 120 ply 的规则轨迹、
禁着路径、合法着列表顺序、终局、完整规则状态、RNG 与 apply/undo 恢复。
CPU 搜索逐项比较完整树图、节点浮点值、策略字节、着法、叶计数与 RNG，覆盖机会节点
关闭/全量/惰性及叶去重开关。没有把“可成功编译”当作语义验收。

完整 unittest 回归在 Cython 后端下 310 项全部通过（77.775s），
强制 Python 后端下同样 310 项全部通过（76.923s）。新增原生对照 9 项单独先跑通过，
新增来源记录覆盖源/二进制哈希和实际后端，旧检查点兼容测试也通过。
`py_compile` 与 `git diff --check` 通过。
严格翻面对称性工具 `python tools/verify_flip_symmetry.py --games 200` 通过：
200 随机 + 2 固定对局，28651 ply，合法着、终局、状态字段、信息态、value 及输入均零不匹配。
GUI 私有捕获探针通过（3 公开局面 x 2 种子 x 2 私有捕获，共 12 次800模拟搜索），
除耗时外，根价值/访问、着法、网络输入与完整评估轨迹哈希、策略哈希均与上一阶段记录一致。
结果：`data/pilots/perf_20261005/gui_privacy_native.json`；来源 replay/model SHA256 相同且未改动。

```powershell
python tools/probe_gui_replay_privacy.py --replay data/replays/jieqi_20261005_111136.json --model data/train_rules5_20261004/models/latest.pt --sims 800 --seeds 0 1 --out data/pilots/perf_20261005/gui_privacy_native_new.json
```

缺失扩展的原生专用测试会跳过，加载/回退测试仍运行；需编译后执行原生对照。
性能探针将 Python 与 Cython 仅在攻击函数上切换，不对比旧 Git HEAD 或更早的编码实现。
保持 FP32、TF32 设置、模拟数、叶 batch、虚拟损失及噪声策略不变。
模型、完整检查点、状态文件、训练日志、最近 replay 和源/二进制在探针前后验证 SHA256。

## 配对实测

结果文件：`data/pilots/perf_20261005/native_rules.json`。
模型来自 `data/train_rules5_20261004` 第 10 轮检查点，PyTorch 2.6.0+cu118、RTX4060 Laptop。
matmul TF32=false、cuDNN TF32=true，保持原精度，不修改叶去重或批次形状。
基准是包含前两轮编码/输出优化的当前 Python 实现，不与旧 Git HEAD 混比。

从两个固定种子的 64 模拟完整对局（77/61 ply）均匀抽取 16 个真实根局面。
规则微测每组包含 2880 次攻击查询或 16 次合法走法查询，预热后测 80 组，
重复 6 对并交替顺序。搜索每次 64 个根（16 局面 x 4 种子块），其中 8 个用 3200
模拟，其余用 800；8 个持续预热 worker，6 对交替顺序，合计 384 组逐字节对照。

| 工作负载 | Python | Cython | 耗时下降 |
|---|---:|---:|---:|
| 攻击查询组 | 3.916ms | 1.019ms | 74.0%，吞吐 3.84 倍 |
| 合法走法查询组 | 0.995ms | 0.401ms | 59.6%，吞吐 2.48 倍 |
| 8 worker / 64 根搜索组 | 7.335s | 6.957s | 5.15%，吞吐 1.054 倍 |

六个配对的耗时下降分别为 3.96%、5.55%、-1.77%、8.36%、7.04%、7.09%，
平均配对下降 5.04%。有一对变慢，不能承诺每次都加速，更不能把微测 3.84 倍当自博弈倍数。
这里只给小样本描述统计，不宣称生产全局统计显著性，也不与前两阶段收益直接相加。

完整对局是单 worker 的样本等价检查，不是 8 worker 吞吐测量：

| 对局 | Python | Cython | 等价结果 |
|---|---:|---:|---|
| seed20261010 / 普通800 | 5.392s | 5.574s | 黑胜，10 ply，10 样本，7222/7041 总/唯一叶 |
| seed20261011 / strong3200 | 127.411s | 107.845s | 黑胜，50 ply，50 样本，155227/153978 总/唯一叶 |

普通短局略慢，strong 单局下降 15.36%；不能由各一局给出完整对局的平均加速比。
每对全部 CompactSample 字段、样本哈希、最终 RNG 哈希、终局、步数、叶计数均完全一致，
样本哈希也与上一阶段验收文件对应两局一致。
384 组 GPU 搜索的策略字节、温度采样着法、RNG、叶计数和机会节点计数全部一致。
探针 `production_unchanged=true`，未写回任何训练权重或数据。

```powershell
python tools/benchmark_native_rules.py --pairs 6 --blocks 4 --workers 8 --complete-games 2 --output data/pilots/perf_20261005/native_rules_new.json
```

输出采用独占新文件模式，不覆盖已有结果。测速时不要同时运行其他训练/重负载。
本阶段没有调整训练更新代码；验证集中在规则、搜索轨迹和生成样本端，而非重新做学习策略实验。
另核对现有多进程调度：每局种子在提交前生成，返回按任务索引排序，
不会因为某局提前完成就改变训练 buffer 中的对局排列；本次不改这段逻辑。

## 风险与后续

二进制依赖平台和 Python ABI，换版本/平台或修改原生源码后重新构建。
`auto` 回退可能降低速度，可用 `cython` 模式和后端打印明确验收，不能仅凭文件存在判断启用。
规则实现仍是原有近似裁判，本次不新增规则正确性的数学保证。
已验证域内严格等价，不减少样本或模拟预算；有限测试不能证明所有未来代码修改都等价。
尚未做完整 1024 局生产轮次测速，也不以固定根吞吐推断等棋力收敛速度。
下一热点候选是合法性过滤，但本次不将其与攻击判定一并改写。
