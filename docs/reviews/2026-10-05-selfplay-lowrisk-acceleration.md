# 两项低风险自博弈加速（2026-10-05）

本轮是在上一轮精确编码优化之上，减少策略列表构建与重复历史整理。默认生效，不修改训练配置，不降低搜索预算，不使用 AMP/PCR，不需要重训或清空 replay。

## 改动

1. `NeuralMCTS._evaluate_batch` 保留原形状的完整网络前向和完整 softmax，将策略输出搬到 CPU 后保留为只读 float32 NumPy 数组，避免构造每批最多 163200 个 Python 概率对象。节点展开仍按原顺序读取合法着并调用 `float()`，累计和归一化运算保持 Python float，不做合法着 masked-softmax 替换。
2. 去重开启时，每个收集的叶快照只整理一次完整重复历史；显式传给签名和编码，去重后按 `first_indices` 对齐快照。完整标记、顺序、物料过滤、当前计数、容量检查均不改变。不向可变 InformationState 或树节点写缓存，不跨批次或跨搜索复用。

value 仍通过 `values.tolist()` 返回 Python float。NumPy float32 与 Python float 混合累加可能降低回传精度，因此没有把 value 一起改成数组。

关闭去重时不预建多余快照；编码仍整理一次历史。测试覆盖了 `None` 表示已知不完整历史与“未提供快照”之间的区别，也覆盖局面修改后的下一批不能使用旧数据。

私有 `_evaluate_batch` 增加可选关键字 `repetition_entries`，原两参数调用仍可使用；测试/GUI 隐私探针中的包装函数同步转发此关键字。结果的内部 policy 类型改为只读数组，最终对外搜索策略与训练样本类型不变。

## 验证与测量

基线不是 Git HEAD，而是本轮修改前的源码快照 `data/pilots/perf_20261005/before_lowrisk_selfplay.py`，包含上一轮编码加速。只读使用 iteration=10 的实际 checkpoint 和回放，设备为 RTX4060 Laptop、PyTorch 2.6.0+cu118。FP32、NCHW、TF32 设置、800/3200 模拟、叶批次 64、chance k=0 与去重配置不变。

固定根测试使用之前相同的两个 64-simulation 完整对局，分别 77 和 61 ply，抽取 16 个开局/中盘/后段根；每组重复四次，使用不同固定种子，共 64 个搜索。其中 8 次采用 3200 模拟，其余采用 800；预热并复用 8-worker 池，六组配对、交替新旧顺序。

| 项目 | 修改前 | 修改后 | 耗时降低 |
|---|---:|---:|---:|
| 单 worker 64 叶评估，含编码/传输/网络/输出回传 | 10.128ms | 7.497ms | 26.0% |
| 8-worker 固定根搜索组平均墙钟时间 | 8.3631s | 7.5970s | 9.2% |

叶评估微测试假设新路径已取得去重签名需要的历史快照；不是把快照准备视为零成本。完整搜索计时包含快照构建，因此更能代表两项优化的净收益。六组搜索耗时降低范围为 7.7% 至 11.0%，搜索吞吐约 1.10 倍。

- [x] 新增回归 7 项：缓存和非缓存输入字节/签名一致、缺失历史不重算、下一批重新读取可变信息态、快照长度检查、policy 字节及 Python value 类型、树统计/RNG、每叶只整理一次历史。
- [x] CPU 完整搜索树对照覆盖去重开/关、机会节点开/关与 chance k=0/1；策略、落子、所有节点统计和 RNG 一致。
- [x] 当前完整 checkpoint、三批增强后的 1024 样本、连续三步 CUDA AdamW 更新：损失、梯度、参数、BN 状态及 optimizer 状态逐位一致。
- [x] 384 对 CUDA 固定根搜索：完整策略哈希、落子、RNG、叶计数及机会链计数一致。
- [x] 普通和强搜索完整对局分别 10 和 50 ply：终局标签、全部 CompactSample 字段/策略/value、叶计数与最终 RNG 完全一致。强搜索局实际前向叶数为 153978，两者一致。
- [x] 数据增强/根噪声审计通过；GUI 隐私探针 6 对搜索通过，且 12 次搜索的输入摘要、完整策略摘要、根价值/访问及落子与此前审计记录完全一致。
- [x] 全量 unittest：300 项通过；`py_compile` 与 `git diff --check` 通过。

两对完整对局中，普通局单 worker 耗时为 5.894s/5.944s，强搜索局为 135.173s/120.965s。普通短局没有显示加速；这里只用于端到端样本等价性验证，不能由两局推导生产总体速度。探针结束时检查生产模型、状态、日志和最新回放文件哈希，全部未改变。

固定根测试不是完整 1024 局生产轮次，不能承诺每轮必然快 9.2%。完整对局测试采用单 worker，仅作样本等价性验收，不以其耗时估计 8-worker 生产吞吐。

## 续训与复现

规则、数据、搜索与 checkpoint 版本不变。当前 `data/train_rules5_20261004` 的模型、完整 optimizer/scheduler/RNG 和 replay 可继续使用，训练命令无需增加开关。新代码在下次正常启动时加载，不对运行中的进程做热替换。

```powershell
python -m unittest test_selfplay_acceleration test_encoding_acceleration test_rule_history_encoding -q
python tools/benchmark_selfplay.py --baseline-source data/pilots/perf_20261005/before_lowrisk_selfplay.py --pairs 6 --blocks 4 --workers 8 --complete-games 2 --output data/pilots/perf_20261005/selfplay_lowrisk_new.json
```

探针拒绝覆盖已有结果，不保存试验权重，不将测试对局加入 replay。详细结果保存在 `data/pilots/perf_20261005/selfplay_lowrisk.json`；结束时检查生产模型、状态、日志及回放文件哈希没有变化。运行性能探针时不要同时启动生产训练。
