# 训练加速代码审查清单（2026-10-05）

审查对象：当前工作区中的实现，包括尚未提交和未跟踪的文件。本轮仅审查与验证，不修改运行代码，不提交，不推进训练，也不写入生产模型或回放。

结论：清单 9 项已逐项完成，发现 2 个 P2 集成问题（测试发现配置、原生后端来源记录）。三批加速在本轮验证的正常使用路径中未发现改变编码、训练更新、规则判断或搜索轨迹的 bug；这不是所有输入和平台的正确性证明。

后续修复见文末“F1/F2 修复记录”；以下问题证据与审查统计保留为修复前快照。

## 项目与审查基线

- 项目为揭棋规则引擎（`jieqi.py`）+ 信念与信息态（`ismcts.py`）+ 神经网络 MCTS / AlphaZero 自博弈训练（`azero.py`）。
- 当前规则/搜索/数据版本为 5/3/3，238 输入通道、2550 动作。训练完整续训涉及 optimizer、scheduler、RNG、增强顺序和 replay 清单。
- 本轮要求核对加速是否保持原来的规则、编码、搜索与训练语义，不以耗时改善或既有测试通过替代代码审查。
- 来源：`2026-10-05-exact-acceleration.md`、`2026-10-05-selfplay-lowrisk-acceleration.md`、`2026-10-05-native-rules-acceleration.md`；`2026-10-05-performance-review.md` 中未实现的候选不当作已实施功能。

## 清单

| 编号 | 实现 / 边界 | 检查重点 | 状态 |
|---|---|---|---|
| A1 | 原位编码与常量平面广播 | 全部通道、float32 舍入、缺失历史、历史容量、独立存储 | 已审查；验证通过 |
| A2 | 训练批次预分配 | 稀疏策略索引与覆盖、旧样本、零策略权重、梯度/BN/optimizer | 已审查；CPU / 真实 CUDA 对照通过 |
| A3 | policy 数组回传 | dtype、只读与生命周期、合法着归一化、Python float 回传 | 已审查；验证通过 |
| A4 | 批内重复历史复用 | 快照时机、去重签名、first_indices 对齐、不完整历史、缓存失效 | 已审查；验证通过 |
| A5 | Cython 攻击判定 | 棋种/颜色/明暗/阻挡、边界、大整数、异常与回退、规则轨迹 | 已审查；原生对照与组合搜索通过 |
| A6 | 构建与后端加载 | API/规则/源码兼容性、auto/强制模式、导入、spawn 与跨平台 | 已审查；本机加载/回退通过；安装路径引发 F1 |
| A7 | 运行来源记录 | 实际后端、源/二进制哈希、旧检查点兼容与子进程一致性 | 已审查；发现 F2；其余相关回归通过 |
| A8 | 性能/等价性探针 | 基线隔离、对照是否独立、覆盖范围、输出防覆盖与生产无副作用 | 已审查；既有结果/源码哈希吻合；未重跑完整测速 |
| A9 | 集成回归 | Python/Cython、搜索配置、训练续训、GUI 信息边界、已有测试覆盖缺口 | 已审查；两后端各 366 项通过；默认 pytest 受 F1 阻断 |

## 问题与证据

### F1 [P2] 本地 Cython 安装使默认 pytest 命令无法收集项目测试

位置：`README.md:24`、`tools/build_rules.py:21` 的 `data/build_deps` 默认安装路径；项目没有排除 `data` 的 pytest 配置。

触发：按 README 执行 `python -m pip install --target data/build_deps Cython==3.3.0` 后，运行 README 提供的全量测试命令 `python -m pytest -q`。

本轮实际结果：pytest 递归进入 Cython 自带的测试目录，收集以下两项时失败：

```text
data/build_deps/Cython/Debugger/Tests/test_libcython_in_gdb.py
data/build_deps/Cython/Debugger/Tests/test_libpython_in_gdb.py
ModuleNotFoundError: No module named 'gdb'
2 errors during collection
```

影响：完成可选加速依赖安装后，常用项目回归命令整体中断，项目自身的测试还没有运行。缺少的是 Cython 调试器测试需要的模块，不是项目运行依赖；给当前训练环境安装 GDB 不能作为此问题的修复。

建议：在项目 pytest 配置中排除 `data` 产物目录，或把构建依赖安装在测试发现范围之外。临时验证命令为 `python -m pytest -q --ignore=data`，本轮已用它验证两种后端。未修改配置或 README。

### F2 [P2] 来源记录可能漏记实际加载的原生二进制

位置：`run_records.py:58` 只枚举仓库根目录的 `_jieqi_rules*.pyd` / `*.so`；`jieqi.py:1349` 通过普通 `import _jieqi_rules` 加载扩展，未把实际模块文件路径交给来源记录。

触发条件：一个有效且源码哈希匹配的扩展已从仓库以外的 Python 搜索路径加载，或已预先进入 `sys.modules`。加载器接受它，`rules_backend` 正确显示 `cython`，但 `create_run()` 使用的代码指纹仍只读取仓库根目录文件。

本轮独立复现：复制本机扩展到隔离目录，仅追加不改变可执行代码的 PE overlay，使两个文件 SHA256 不同；在新进程中优先加载隔离目录。结果：

```text
backend = cython
loaded_sha256 = bfa4b06f8f62c8a3f33251b250418226fe95e39db65662b6bdaac50085a89ac7
loaded_hash_in_record = false
recorded root binary SHA256 = 3e342515466c222227835d9dbaeea96f959f4fb8a818bfdba78bc50f3fd5ad09
```

影响：运行 manifest 可能记录从未加载的二进制，或者完全没有原生二进制哈希，无法按记录还原实际执行来源。本轮没有据此发现规则计算错误；正常从仓库根目录加载的当前扩展不受此复现条件影响。

建议：加载时保留实际扩展的 `__file__`、API/规则/源码版本，并让 manifest 哈希该文件；也可明确限制只接受仓库中的预期扩展。增加“从非根目录加载有效扩展”的来源回归。本轮未修改实现。

复现源码与结果：`data/pilots/acceleration_code_audit_20261005_4c039tra/provenance_probe.py`、`provenance_result.json`。探针只写入新建隔离目录，没有改动仓库根目录二进制。

## 各实现的审查结果

- A1：逐项核对 238 通道。常量仍按黑/红与原通道位置写入；历史查表仍先用 float64 除法再转 float32；重复记录保留顺序、完整标记、当前计数与容量拒绝。分配结果连续且不共享不同批次存储。
- A2：信息态恢复逻辑保持原字段复制和旧数据默认值；策略写入保持 `zip` 截断、整数负索引、重复索引后写覆盖与 Python `float()` 转换。所有样本仍参加前向与 value 监督，零策略权重没有跳过 BN 或改变 loss 分母。CPU 与真实检查点 CUDA AdamW 对照通过。
- A3：完整 softmax 及批次形状保持原样。NumPy 视图持有 CPU tensor 的存储引用；节点展开只读，且先转 Python float 再累加、归一化。value 仍为 Python float，未引入 NumPy float32 回传舍入。
- A4：快照在当前 `pending` 批次内创建，来自独立模拟信息态；签名和编码共用它，唯一样本按同一 `first_indices` 取快照，重复结果按 slot 回映射。`None` 与未提供快照的 sentinel 区分正确，未向可变局面或树节点留下跨批缓存。原来的每叶回传、访问次数和一次展开约束保留。
- A5：原生掩码与原 Python 位运算匹配，明暗士的九宫条件、炮架、马腿、象眼、兵的方向/过河与射线行为均逐段核对。扩展仅读棋盘；原合法着生成、禁着、apply/undo、RNG 不变。异常参数/非 list 容器保留 Python 路径；大整数先掩码再转 C 整数。
- A6/A7：缺失/过期扩展和强制模式已有验证；源码/API/规则版本检查有效，spawn 子进程独立导入后端。构建不在训练启动时自动发生。来源的当前根目录源/二进制覆盖通过，但实际导入来源存在 F2。
- A8：三份 benchmark 均使用新输出文件独占创建，没有保存训练权重、提交回放或推进轮次的调用。编码阶段从 Git 读取旧 `azero.py`，低风险自博弈阶段使用加速前快照，原生阶段只切换攻击函数；基线划分符合各阶段目的。已核对六组交替计时、固定种子、样本与 RNG 比较、生产文件保护及文档统计，未发现支持“整轮加速倍数”的过度推断。
- A9：测试/GUI 隐私探针的 `_evaluate_batch` 包装器均已转发新增关键字；GUI、回放、续训、来源、规则与搜索回归通过。全量 pytest 的发现边界遗漏了第三方构建依赖，见 F1。

## 本轮验证记录

| 验证 | 本轮结果 |
|---|---|
| `python -m unittest test_encoding_acceleration test_selfplay_acceleration test_native_rules test_run_records -q` | 41 项通过，32.421s |
| 默认 `python -m pytest -q` | 失败；收集 Cython GDB 测试，见 F1 |
| 默认 auto 实际为 Cython：`python -m pytest -q --ignore=data` | 366 项通过，81.24s |
| 新进程设置 `JIEQI_RULES_BACKEND=python` 后相同 pytest 命令 | 366 项通过，82.89s |
| 从 Git 加速前源码直接对照 40 个编码状态 | 全部字节一致 |
| iteration=10 真实完整 checkpoint，3 批四种增强后的 1024 样本，CUDA 连续三步 | 输入/标签/布局、loss、梯度、参数、BN、完整 AdamW 状态逐位一致 |
| 同时对照旧编码/列表回传/Python 规则与当前全部加速 | 24 对真实 CUDA 搜索，完整树、策略字节、落子、RNG、叶/机会计数一致，输入局面/信念未修改 |
| 搜索根与配置 | 0/8/16/24 ply；构造轨迹含 4 次暗子私有捕获；机会节点 off/on、k=0/1、去重 off/on，128 模拟、叶 batch64 |
| 本轮探针前后生产文件哈希 | checkpoint、latest、train_state、train.log、最新 replay、核心源文件与原生二进制均未改变 |
| 既有 `native_rules.json` 来源哈希复核 | 所列 11 个生产/源/二进制文件当前均与记录吻合 |
| `git diff --check` / 8 个相关 Python 源文件 AST 解析 | 通过；仅 Git 的 LF/CRLF 提示 |

本轮独立对照的源码、JSON 结果与源文件指纹保存于：

```text
data/pilots/acceleration_code_audit_20261005_4c039tra/audit_probe.py
data/pilots/acceleration_code_audit_20261005_4c039tra/result.json
data/pilots/acceleration_code_audit_20261005_4c039tra/source_fingerprints.json
data/pilots/acceleration_code_audit_20261005_4c039tra/provenance_probe.py
data/pilots/acceleration_code_audit_20261005_4c039tra/provenance_result.json
```

设备/环境：Python 3.11.5、NumPy 2.0.2、PyTorch 2.6.0+cu118、RTX 4060 Laptop；matmul TF32=false，cuDNN TF32=true。独立 CUDA 对照基线为私有历史快照（未随本仓库分发）中的 `azero.py`，规则基线为保留的 `_is_square_attacked_python`。

## 边界与尚未执行的工作

- 两项问题仅记录，尚未修复；本轮新增可审查的文档和隔离探针，没有修改用户已有实现文件或生产数据。
- 本轮没有重新编译/覆盖现有 `.pyd`，没有验证 Linux/macOS、其他 Python ABI 或软件版本的构建与数值行为。
- 没有重新跑完整 384 对 800/3200 模拟性能测试、200 局严格翻面工具、GUI 生产 replay 探针或 1024 局生产轮次；已检查对应工具与历史证据，区分本轮执行结果和既有记录。
- 没有评估 AMP、PCR、集中推理、跨 worker 合批、worker 数扫描等尚未实现方案，也没有据此次审查推断棋力或长期收敛速度。
- 修复 F1 后应确保默认 pytest 命令能完成；修复 F2 后应让外部有效扩展的实际路径/哈希进入 manifest，并保留根目录加载与旧检查点兼容回归。

## F1/F2 修复记录

后续执行修复，未改变规则、搜索、网络更新或数据生成算法，未重编/覆盖现有扩展。

- F1：新增 pytest 配置（现位于 `pyproject.toml`），在保留 pytest 默认递归排除模式的基础上排除 `data`。
  不移动/删除本地依赖或训练数据，不安装 GDB，默认命令仍为 `python -m pytest -q`。
  `tests/test_pytest_discovery.py` 复制实际配置到隔离项目，验证 `data/build_deps` 中的 GDB 测试不收集，
  根目录与 `tools` 的正常测试仍被发现，并保留隐藏目录、build/dist、node_modules、venv 的排除。
- F2：加载时捕获 `RULES_BACKEND_NATIVE`：实际模块文件绝对路径、SHA256、大小、API/规则版本和源哈希。
  新 manifest 写入 `runtime.rules_backend_native`，`code.files` 同时哈希实际载入文件；外部文件用绝对路径键。
  保留正常根目录加载与外部加载，不仅凭目录 glob 猜测执行来源。创建运行记录时再次核对加载指纹，
  如文件已被替换则明确拒绝，要求新进程重新加载，避免磁盘文件与内存中执行代码错配。
  `test_run_records.py` 在新进程中加载隔离复制的有效 PE 扩展，并追加 overlay 使其 SHA256 与根目录不同；
  普通外部导入与预先进入 `sys.modules` 两种路径均验证实际路径/哈希进入 manifest 和代码指纹。
  另外覆盖正常加载、Python 模式没有原生身份、已加载文件变化时拒绝不实记录。

针对性命令 `python -m unittest test_pytest_discovery test_run_records test_native_rules.TestRulesBackendLoading -q`
通过：26 项，8.292s。默认 auto 实际为 Cython，直接执行 `python -m pytest -q`（无 `--ignore=data`）
通过：370 项，90.14s。新进程强制 `JIEQI_RULES_BACKEND=python` 后同一默认 pytest 命令
也通过：370 项，91.25s。`py_compile`、`git diff --check` 通过。

新增来源字段只影响新运行记录，旧 manifest/检查点不追补、不改写，仍兼容读取。
本次没有启动生产训练，生产 checkpoint/latest/train_state/log/replay、`.pyx`/`.pyd` 及
未修改的 `azero.py`/`ismcts.py` 共 9 项哈希仍与原测速记录相同；旧测速 JSON 不覆盖。
