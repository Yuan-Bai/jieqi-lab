# 节点扩展与初始化精确编译实验（2026-10-05）

结论：**等价性验收通过，但没有可靠的整体加速收益，不默认投入训练。**
编译初始化在本机微测中较慢；只编译扩展的微测收益很小，8-worker 搜索配对平均
反而慢约 0.6%。因此节点后端默认仍是 Python，已验证的 Cython 攻击判定继续按原策略启用。
保留编译原型、显式开关、对照测试及复现工具，未删除或改写生产训练数据。

这是对多局面热点报告的后续实验，不是“热点占 34%，就能加速 34%”的兑现：
扩展仍要分配大量 Python 节点和三个独立字典、执行 Python 采样助手及对象运算，
单纯把方法编译成扩展不能消除这些成本。具体损耗尚未逐项分解，不能笼统归因为 GPU。

## 实现范围

- 新源文件 `native/jieqi_search.pyx`，只实现初始化与扩展两个方法，不编译整个 MCTS。
- 保留原 `PUCTNode` Python 类、slots、字典、节点共享关系、pickle 和原方法体；
  `_init_python`、`_expand_python` 提供独立基准及回退。
- 数值中间量保持 Python 对象，保留求和/归一化顺序；生成 C 中总和是 `PyObject*`，
  除法仍调用 Python 数值 API。不使用浮点重排、fast-math、精度切换或 GIL 释放。
- `move_to_index`、`sample_reveal_outcomes` 和节点工厂仍调用所属 Python 模块，
  保留完整机会支持、分支创建顺序、RNG 消耗及异常发生前的更新顺序。
- 每个节点类保存自己的模块上下文，避免普通导入、CLI/spawn 或基准克隆模块相互污染。
- 两个编译方法都经过对照；正常选择 Cython 后端时也**只接入 expand**，初始化保留 Python。

Cython 使用 `infer_types=False`、`annotation_typing=False`，保留边界/负索引检查。
编译指令含义见 [Cython 官方文档](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives)。
本机 Python 3.11.5、Cython 3.3.0、MSVC 14.42；原有规则二进制没有重建。
沙箱内构建在 `rc.exe` 链接步骤失败，经允许后用本机已安装 MSVC/SDK 构建成功，
未安装全局依赖，仅产生项目根目录二进制及 `data/native_search_build/` 构建产物。

## 加载与来源记录

`JIEQI_SEARCH_BACKEND=python|auto|cython`，**默认 python**：

- `python`：保持原方法，不尝试加载实验扩展。
- 显式 `auto`：检查成功后编译扩展方法，否则回退到原方法。
- `cython`：要求有效扩展，缺失或过期时明确失败。

加载核对 API=1、SEARCH_VERSION=3、原生源码 SHA256 及两个实际 Python 方法的
字节码/默认值指纹。指纹规范化文件路径以适配 CLI/spawn 和克隆模块，但保留行号等
代码信息，因此方法移动或修改后可能需重建。缺少/过期扩展不在训练启动时自动编译。

新运行记录含 `runtime.search_backend`、错误和实际加载二进制身份；`code.files`
也记录真实文件，即使从外部路径加载或预加载。已加载文件随后改变时拒绝创建记录。
`run_records` 不循环导入 `azero`，通过已完成加载的正规入口模块取得身份。
两个剖析工具也增加节点后端/二进制哈希及 worker 身份检查。

- 二进制：`_jieqi_search.cp311-win_amd64.pyd`，81920 bytes。
- 二进制 SHA256：`5480d2ea07d420800a1684e90e24e22cc8fe27b219c54176a642dc4a0a67fdb5`。
- 原生源 SHA256：`34e798edc880670ad5db910943448699176668b8bb278ef3200f84a543f66583`。
- 方法指纹：`49a1965ee2c2614b5f273a49e2ba7fd1f05008f5fbcb6db24fe507a0aef1b055`。
- 规则二进制 SHA256 仍为 `3e342515466c222227835d9dbaeea96f959f4fb8a818bfdba78bc50f3fd5ad09`。

规则/搜索/数据版本和训练配置不变，当前 v5 检查点及 replay 可继续使用。
已有进程不热切换；默认路径不需要更改原训练命令、清空 buffer 或重新训练。

## 微测

原始结果：`data/pilots/perf_20261005/native_search.json`，输出独占创建，不覆盖旧结果。
基准是当前已优化编码/叶输出且攻击判定仍为 Cython 的版本，不与旧 Git HEAD 比较。

六轮微测交替正序/倒序运行四种模式。构造组创建并保留 60000 个节点；扩展组
包含三个固定根（两个初始局面和一个受控残局），512 次循环、共 1536 次完整扩展。
两项不是纯 GPU 或纯分配计时，也不是完整搜索的收益。

| 模式 | 构造组平均 | 扩展组平均 |
|---|---:|---:|
| 全 Python 节点方法 | 39.949ms | 270.778ms |
| 仅初始化编译 | 43.298ms | 284.978ms |
| 仅扩展编译 | 40.339ms | 266.077ms |
| 初始化与扩展都编译 | 44.188ms | 273.023ms |

初始化编译的构造组约慢 8.4%；只编译扩展约省 1.7%；两项一起编译的扩展组
约慢 0.8%。“仅扩展”构造组仍使用相同 Python 初始化，其微小差异是计时波动，
不能解读为修改了构造行为。

## 真实模型搜索与对局

检查点为 `data/train_rules5_20261004` iteration=10，SHA256：
`aee9016a8b4e0b443feb87dc891ee1d8a642a6ed6953c48b2a9ef7227490cf31`。
PyTorch 2.6.0+cu118、RTX4060 Laptop；FP32，matmul TF32=false、cuDNN TF32=true。
800/3200 simulations、leaf batch64、chance on/k0、dedup、虚拟损失、温度/根噪声和
策略裁剪全部保持原配置，PCR off。

用原方法完成三局 800 模拟真实模型自博弈（47/78/10 ply），抽取 18 根；加入
4 捕获优先轨迹根和 2 受控残局，总共 24 根，覆盖私有捕获、不同材料及历史长度。
8-worker 池预热后每组搜索 24 根 x 4 种子块 = 96 次，12 次 strong3200，其余 800；
strong 根随种子块轮换。六组配对交替顺序，不计进程启动及局面生成。

| 8 worker / 96 搜索 | 平均组墙钟 |
|---|---:|
| Python 节点方法 | 11.365s |
| 仅扩展编译 | 11.435s |

六个配对的耗时节省比例为 -2.69%、-0.64%、-3.41%、+0.11%、-0.04%、+2.83%，
平均 -0.64%。有快有慢，样本很小，不宣称“必然慢 0.6%”，但没有依据默认启用。
未测完整 1024 局生产轮次，不外推等棋力收敛速度。

完整对局另做单 worker 对照，主要用于生成样本验收：

| 种子/预算 | Python | 仅扩展编译 | 结果 |
|---|---:|---:|---|
| 20261010 / 800 | 5.576s | 5.871s | 黑胜，10 ply，7222/7041 总/唯一叶 |
| 20261011 / 3200 | 112.054s | 110.742s | 黑胜，50 ply，155227/153978 总/唯一叶 |

每对终局、步数、所有 CompactSample 字段/顺序、样本哈希及最终 RNG 哈希一致。
普通样本 SHA256 为 `4117f7b58d2625d6cb75b3a243ad18deee56fb235bd6adfd8eec54e1f20302a6`，
strong 为 `46df00a04072d81998441c482c2bbf47e5a4e3c7be9759e4deefcd1b9f0de88d`，
与上一阶段同种子记录一致。不能由各一局的耗时判断平均完整对局收益。

## 等价性与边界

- 新增 13 项测试覆盖：slots/对象身份/新字典、非有限值/负零/极小浮点数、Python
  list、NumPy FP16/32/64、Torch FP32、超大整数、零先验、异常/部分更新、完整机会
  分支及计数快照、不同 chance_k、RNG、Python 回调、pickle、模块隔离、CLI 和来源记录。
- CPU 在有/无私有历史、机会节点关闭/全枚举/惰性及 dedup 开关下，比较四种方法组合的
  完整树图、策略字节、着法、RNG、叶/机会计数和根输入不变。
- CUDA 对 8 根 x 800/3200 = 16 组，逐项对照 Python、仅扩展、两项编译，
  完整 pickle 图 SHA256（包含 slots、浮点位模式、字典顺序和共享关系）全部一致。
- 8-worker 共 576 组配对搜索（1152 次计时搜索），策略字节、着法、RNG、叶总/唯一
  计数和机会计数一致。这是重复固定根，不是 576 个独立局面。
- 两局完整自博弈样本与旧记录逐字节一致；未向 replay 保存任何新样本，未更新网络。

最终回归：默认配置（攻击判定 Cython、节点 Python）390 项通过，93.36s；
显式启用节点 Cython 后端的全量 390 项通过，95.05s；强制两种后端均为 Python
时，节点编译、剖析及来源记录相关的 41 项通过，19.67s。本次没有再跑完全 Python
配置的全量套件。相关源码 `py_compile` 及 `git diff --check` 也通过。

探针前后校验检查点、latest、状态、日志、最近 replay、核心源码和两份原生二进制。
原始结果 `production_unchanged=true`。测量结束后只将默认节点选择改为 Python、补充
剖析工具来源字段和文档；原方法指纹及两份二进制未变。原始 JSON 保留测量当时的
源码哈希，不重写成最终源码哈希。

有限对照不是所有未来代码的数学等价证明；编译体存在维护漂移风险，靠双重指纹、
显式失败/回退及回归约束。节点数据结构重写、惰性字典/分支、对象池、集中推理均
不属于本次小范围实验，也没有为提高数字而减少样本、模拟数或改 batch 形状。

## 复现与建议

```powershell
python tools/build_search.py
$env:JIEQI_SEARCH_BACKEND="cython"
$env:JIEQI_LEGAL_BACKEND="python"
python -m pytest -q test_native_search.py
python tools/benchmark_native_search.py --pairs 6 --blocks 4 --workers 8 --complete-games 2 --output data/pilots/perf_20261005/native_search_new.json
Remove-Item Env:JIEQI_SEARCH_BACKEND
Remove-Item Env:JIEQI_LEGAL_BACKEND
```

输出必须是既有目录中的新文件，测试时不要同时训练或修改核心源文件。
后续合法着扩展接入后，节点 benchmark 明确要求合法着后端保持 Python，并记录其
身份、保护合法着原生源码及检查 worker 一致性；审查与修复见
`2026-10-06-compiled-path-code-audit.md`。历史结果 JSON 保留测量当时的字段与源码哈希。
需要实验初始化时，基准支持 `--optimized both`，但不提供生产初始化编译开关。

当前不建议在主线启用本实验。后续可优先尝试边界更清晰的合法着快路编译；
大幅优化节点分配需要另做数据结构/生命周期实验和更广泛验收，不能当成本次
“把两个方法交给 Cython”已经成功解决的问题。
