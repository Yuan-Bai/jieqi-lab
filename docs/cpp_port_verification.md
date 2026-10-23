# C++ 端口编译验证报告（S1.7）

> 公开副本说明：个人绝对路径已改为相对路径或占位符；标注“未随源码分发”的实验材料仅作为历史记录，不能从本仓库下载。复现命令中的 `<PROJECT_ROOT>` 等占位符需替换为自己的路径。

- 日期：2026-09-11
- 对象：`<LOCAL_WORKSPACE>/jieqi-cpp-prototype.7z（历史归档，未随源码分发）`（2026-02-10/11 归档的 C++ 端口）
- 结论：**端口处于未完成状态，无法在本机跑通搜索，不建议直接作为 Stage 2 后端。** 若要做 C++ 后端，应视为一次正式的移植工程（预计数天），并配套 conformance 测试向量。

## 验证环境

| 项 | 值 |
|---|---|
| 编译器 | Visual Studio 2022 Community（MSVC 14.42） |
| Qt | 仅 Anaconda 自带 qmake（`<HISTORICAL_QMAKE>`）；归档 Makefile 来自 **MSYS2 mingw64 Qt 5.15.16（另一台机器）**，本机无 MSYS2 |
| 验证目录 | 解包至 `<LOCAL_TEMP>/jieqi-cpp-prototype`（未改动归档与主仓库） |
| 测试程序 | 临时编写 `bench_main.cpp`：初始局面 → `init_belief_for_game` → `ISMCTSAgent(10000, 1.4, "heuristic", N)` → `select_move` 计时 |

## 编译结果

- `cl /std:c++17 /O2 /openmp` 直接编译失败：源文件为 **UTF-8 无 BOM**，MSVC 默认按 GBK 解析中文注释产生大量语法错误级联。
- 加 `/utf-8` 后 `jieqi.cpp + ismcts.cpp + bench_main.cpp` 编译链接成功（约 1 分钟）。结论：MSVC 可用，但必须显式 `/utf-8`。
- Qt GUI 未再单独构建：核心库已发现致命问题，GUI 构建无意义；且旧构建依赖的 MSYS2 Qt 环境本机不存在。

## 发现的问题（均在临时副本中修复以继续验证）

| # | 位置 | 问题 | 说明 |
|---|---|---|---|
| 1 | `jieqi.cpp:183`（`initialize_precomputed_tables`） | `RAYS` 声明为 `std::vector<std::vector<int>>[4]`，初始化时未 `resize(BOARD_SIZE)` 就下标访问空 vector | 启动即 access violation（0xC0000005） |
| 2 | 同上 | 函数只填充 `g_precomputed_tables`，而 `is_square_attacked`、`iter_pseudo_moves` 等仍引用**从未填充的裸全局表**（`RAYS`/`KNIGHT_MOVES`/`ELEPHANT_MOVES`/`ADVISOR_MOVES`/`KING_ADJ`） | 重构半途：指针表与裸全局表两套并存，只初始化了一套 |
| 3 | `jieqi.cpp` `GameState::initial()` | 池数组被装成紧凑布局（索引 1..6），而其他所有代码按“类型索引”（`pool[side][type]`，1=车…7=卒）访问 | 导致 `InformationState::determinize` 抛 `pool totals inconsistent`（sum=13 vs total=15） |
| 4 | `ismcts.cpp` 搜索回放 | 修复 1–3 后：单次迭代可完成（1 次完整 rollout ≈ 43.8ms），**10 次迭代即挂起 >70s**，疑似 rollout 无法终止（重复/无吃子/`fast_outcome` 判定与 Python 语义不一致） | 搜索正确性完全未经验证 |

另外：归档中 `jieqi_core.exe` 为 **0 字节**；`release/JieQiGame.exe`（131KB）是旧环境产物，不代表当前源码可构建可运行。

## 性能实测（仅供参考）

| 场景 | 结果 |
|---|---|
| 1 次迭代（单线程，含一次完整 rollout） | 43.8 ms |
| 10 次迭代 | 挂起（不可用） |
| README 声称（4 线程 10000 迭代 ≈1.5s） | **无法复现**（当前连正确完成都做不到） |
| 对照：Python 旧版 1000 迭代 ≈5s（≈5ms/迭代）；S1.5 后 Python 自博弈吞吐 ≈7500 evals/s | 端口当前单迭代耗时反而高于 Python |

## 建议

1. **Stage 2 不采用该端口**；主线的对弈/评估/数据后端继续用 Python（S1.5 已获约 23× 单进程、3–4× 旧生产配置的吞吐提升）。
2. 若未来仍要 C++ 后端：按 ADR-0006 的混合策略**从 Python 语义重新移植规则层**，并先建立 conformance 测试向量（FEN 回放 + 合法着法/perft 对比 + 随机对局回放），否则规则漂移不可控。
3. 归档中的 `README_CPP.md` 性能数字与“已完成”表述不可信，引用时需标注未经验证。
