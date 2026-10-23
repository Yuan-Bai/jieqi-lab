# 合法着快路与节点编译代码审查（2026-10-06）

审查对象为当前工作区的 `native/jieqi_legal.pyx`、`native/jieqi_search.pyx`，以及接入、构建、来源记录和对照工具。本轮只审查与验证，不修改运行代码、不提交，也不写回生产模型或 replay。

来源：`2026-10-06-native-legal-experiment.md`、`2026-10-05-native-search-experiment.md`。当前合法着与节点后端均默认 Python；显式节点 Cython 后端仅接入 `expand`，初始化编译仅用于隔离实验。

结论：两项编译的已验证路径未发现改变合法着、完整搜索图或随机轨迹的 bug。发现 1 个 P2 基准隔离/来源记录问题，见 F1；它不直接影响默认生产训练的正确性。当前性能证据仍支持保持两个实验后端默认关闭。

F1 已于同日后续修复，详见末尾“F1 修复记录”；原审查复现和性能证据保持原样。

## 审查清单

| 编号 | 范围 | 重点 | 状态 |
|---|---|---|---|
| C1 | 合法着内部伪合法枚举 | 棋种/颜色/明暗、阻挡、飞将、方向、顺序、整数边界 | 已审查；对照通过 |
| C2 | 合法性过滤与生成器 | 禁着慢路、试走恢复、提前 close/throw、回调异常与非标准输入回退 | 已审查；对照通过 |
| C3 | 节点初始化 | slots 赋值顺序、对象身份、字典隔离、pickle、数值类型 | 已审查；直接及组合对照通过 |
| C4 | 节点扩展 | Python 浮点运算、归一化顺序、机会分支、计数快照、RNG 与部分更新 | 已审查；直接及组合对照通过 |
| C5 | 编译加载与上下文 | 源码/方法指纹、默认开关、失败/回退、克隆模块、CLI/spawn | 已审查；已有专测通过 |
| C6 | 来源与续训 | 实际加载文件哈希、加载后文件变化、后端身份、旧检查点兼容 | 已审查；相关测试通过 |
| C7 | 基准与剖析工具 | 单变量对照、父子进程身份、防覆盖、生产无副作用、结论边界 | 已审查；F1 已后续修复 |
| C8 | 两项联合启用 | 全量回归、CPU/CUDA 完整树与 RNG、单独与组合方法对照 | 已审查；默认/联合配置各 405 项与 CUDA 对照通过 |

## F1 [P2] 节点基准未隔离或记录合法着后端（已修复）

以下描述修复前的发现与复现。

位置：`tools/benchmark_native_search.py:189`、`:192`、`:117`、`:221`。

触发条件：在同一运行环境中同时设置 `JIEQI_SEARCH_BACKEND=cython` 与 `JIEQI_LEGAL_BACKEND=cython`，再运行节点编译 benchmark。

该脚本只要求攻击判定和节点二进制可用，不要求合法着保持 Python；因此会接受并使用编译合法着。源文件保护清单没有 `native/jieqi_legal.pyx` 或实际合法着二进制，最终 JSON 也没有合法着后端/身份字段。worker 初始化只把权重、配置、攻击二进制交给公共初始化器，省略了已经支持的合法着后端身份检查。

本轮在真实当前后端和 iteration=15 检查点下执行预检复现；在开始生成局面前主动停止，没有运行性能计时或生成伪造性能结果：

```text
legal_backend = cython
preflight_accepted = true
loaded_legal_binary_protected = false
legal_source_protected = false
legal_identity_forwarded_to_worker = false
```

影响：可能在不同的合法着配置下取得节点测速结果，输出却不能说明或还原该条件。脚本也无法发现父子进程合法着身份不同，或测速期间该源/二进制文件改变。Python/编译节点在同次运行中仍可比较，因此本轮不据此否定已记录的默认 Python 合法着实验，也不把它描述成生产搜索错误。

建议：如果保持单独节点实验，启动时明确要求 `jq.LEGAL_BACKEND == "python"`；如果允许其他合法着后端，应与合法着 benchmark/公共 profiler 一样，记录后端及实际文件哈希，并将身份传给 worker。增加预检或 worker 回归，确保配置不会静默变化。原审查阶段只记录，后续修复见末尾。

复现文件：`data/pilots/compiled_path_audit_20261006_gccajifi/benchmark_preflight_probe.py`、`benchmark_preflight_result.json`。

## 逐项审查结果

- C1：逐段核对原 `iter_pseudo_moves` 与 `_pseudo`：棋盘扫描、射线及各走法表访问顺序一致，暗士宫内条件、飞将、兵的左右/向前顺序一致。普通整数先掩码再转 C int，整数子类/NumPy 数值仍走 Python 位运算助手。未观察到遗漏或重排着法。
- C2：`_iter_legal_moves` / `_move_scan` 仍先判断长将/长捉禁着，有禁着时保留原 apply/undo 慢路。快路试走顺序和攻击回调不变，恢复棋盘与将位置后才 yield；提前 close/throw 不留下试走。攻击回调异常时保留原部分更新行为，没有将已有异常语义算作本次回归。子类、替换枚举/助手/常量及非标准容器有回退。
- C3：初始化按原 slots 顺序赋值，原 prior/player/move/is_chance 对象不被转换；三个字典独立创建，没有新的共享存储或节点布局。Python、仅初始化、仅扩展、两者编译的直接检查、pickle 和赋值事件顺序通过。
- C4：总和、归一化和 player 运算保留 Python 对象与原操作顺序，没有 C 浮点累计或重排；节点工厂、动作索引和机会采样仍按所属模块动态调用。完整机会分支、双揭示构建、计数快照、懒建支持和 RNG 次数保持一致；输入错误与部分节点更新已有对照。
- C5：加载器首先恢复原方法；默认不导入实验扩展，auto 缺失/过期时回退，cython 明确失败。原生源哈希、原 Python 方法及默认值指纹均检查，代码路径递归规范化但保留行号。原生体使用每个 Python 类保存的所属模块，不共享新扩展的全局模块上下文。CLI、spawn/模块隔离相关测试通过。
- C6：运行记录现在覆盖实际加载的三个扩展文件，外部预加载与加载后文件改变的测试通过。上一轮的默认 pytest 第三方收集问题已由 pytest 配置（现位于 `pyproject.toml`）修复；上一轮实际攻击二进制漏记问题也有修复及回归。新后端字段未改变规则/搜索/数据版本，原完整续训路径测试通过。
- C7：合法着 benchmark 明确拒绝节点后端非 Python，并记录/核对三种后端身份；其保护与独占输出边界符合独立实验要求。节点 benchmark 的对照、完整图、样本/RNG 检查和防覆盖存在，但遗漏新增合法着后端的隔离/身份，见 F1。两个新构建脚本均不会在训练导入时自动编译，构建前关闭各自待替换扩展以避免 Windows DLL 锁定。
- C8：补做报告原先未执行的两项联合启用全量回归，并用当前真实网络比较独立及组合方法。未发现输入局面被修改、节点图共享关系改变或搜索数值/RNG 偏差。

## 本轮验证

| 检查 | 结果 |
|---|---|
| `python -m pytest -q tests/test_native_legal.py tests/test_native_search.py tests/test_run_records.py tests/test_pytest_discovery.py` | 49 项通过，42.35s |
| `JIEQI_LEGAL_BACKEND=cython`、`JIEQI_SEARCH_BACKEND=cython` 后全量 pytest | 405 项通过，113.32s |
| 默认配置全量 `python -m pytest -q` | 405 项通过，113.27s |
| 当前 iteration=15 完整 checkpoint 的 CUDA 组合搜索 | 12 个条件、60 次搜索、48 组相对 Python 基线的对照，全部字节一致 |
| CUDA 比较内容 | 完整 pickle 搜索图（含字段、数值、字典顺序及共享引用）、完整策略、落子、RNG、叶总/唯一计数、机会计数 |
| 组合模式 | Python/Python、合法着 Cython/节点 Python、合法着 Python/节点 expand、两项 Cython、两项 Cython 且初始化也编译 |
| 局面/配置覆盖 | 0/8/16/24 ply，构造轨迹含 7 次私有暗子捕获；chance on/k0/dedup on、on/k1/dedup off、off/dedup on；128 模拟、叶 batch64 |
| 探针前后 SHA256 | checkpoint、latest、train_state、train.log、最新 replay、核心 Python 源、两项新原生源/二进制与攻击二进制均未变 |
| 新问题复现 | 实际后端预检接受编译合法着，但保护清单/worker 身份检查遗漏，见 F1 |
| Python 源解析 / `git diff --check` | 9 个相关源文件 AST 通过；diff 无空白错误，只有已有 LF/CRLF 提示 |

设备：Python 3.11.5、PyTorch 2.6.0+cu118、RTX 4060 Laptop；FP32，matmul TF32=false、cuDNN TF32=true。CUDA 组合探针一直使用同一攻击判定 Cython 后端，只切换本轮两项编译方法。

本轮源码与证据位于：

```text
data/pilots/compiled_path_audit_20261006_gccajifi/cuda_combination_probe.py
data/pilots/compiled_path_audit_20261006_gccajifi/cuda_combination_result.json
data/pilots/compiled_path_audit_20261006_gccajifi/benchmark_preflight_probe.py
data/pilots/compiled_path_audit_20261006_gccajifi/benchmark_preflight_result.json
```

## 验证边界

- 本轮新增文档和隔离探针，未修改用户运行实现或生产数据，没有重新编译/覆盖既有二进制。
- 未重新做 576/1152 组性能配对、完整对局、1024 局生产轮次或训练收敛实验。既有性能报告的每轮不变哈希是其执行时证据；当前训练已由 iteration=10 推进至 15，模型/状态/日志哈希变化不当作本轮修改或历史报告错误。
- 当前合法着报告的源/二进制哈希与现代码吻合；节点旧测量报告中的部分 Python/工具源码在后续接入合法着和默认后端整理时已变化。本轮对照以当前方法体和当前二进制为准，没有直接沿用旧完整验收结论。
- 本机与已测试输入之外、其他 Python ABI/平台、任意运行时改坏走法表的行为没有验收。有限对照不能证明所有未来局面或代码都等价。
- 初始化编译通过等价性检查，但未进入默认/显式生产加载路径；两项实验的微测收益不构成整轮吞吐或棋力收益保证。

## F1 修复记录（2026-10-06）

修复采用单独节点实验的边界，不增加组合性能模式：

- `tools/benchmark_native_search.py` 在读取检查点、生成局面或计时前要求合法着后端为
  Python；否则通过 argparse 明确失败，提示 `JIEQI_LEGAL_BACKEND=python`，不创建结果文件。
- 父进程向节点基准 initializer 传递 `(LEGAL_BACKEND, LEGAL_BACKEND_NATIVE)`，并转交公共
  profiler 的身份检查；worker 后端或二进制身份不同会在加载模型前失败。initializer
  也拒绝调用方要求编译合法着，避免绕过 CLI 后混入组合实验。
- 新结果 JSON 明确记录 `legal_backend`、`legal_backend_error` 与 `native_legal`；独立
  节点实验的正常值为 Python/None/None。额外把 `native/jieqi_legal.pyx` 纳入前后源码
  哈希保护。既然拒绝编译合法着，不会使用或遗漏保护一个已加载的合法着原生文件。
- 新增 `test_benchmark_native_search.py` 的五项回归，覆盖提前拒绝、Python 对照的
  JSON/哈希、initializer 身份转交、worker 不匹配及父进程池参数。
- 节点实验报告的复现命令明确设置合法着为 Python，原有 JSON 与性能数字未改写。

修复后验证：相关测试 `test_benchmark_native_search.py test_profile_selfplay.py` 共
13 项通过，2.50s；使用实际两个扩展的真实 CLI 验证联合启用时退出码为 2，错误
明确提示 Python 合法着，且未生成输出。没有为修复运行新的性能计时、训练或原生构建。
默认配置全量 410 项通过，112.49s；两个实验后端均显式设为 Cython 时，上述相关
13 项也通过，2.44s。修复涉及的两个 Python 文件 `py_compile` 与空白检查通过。
未重跑联合配置的全量套件或 CUDA 组合搜索；原审查的相关证据仍保留在上文。
