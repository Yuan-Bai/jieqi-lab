# 当前训练流程性能检查（2026-10-05）

本文记录优化前的性能检查。随后按“保持训练结果”的要求实施了编码加速，见 `docs/reviews/2026-10-05-exact-acceleration.md`；下文“未修改运行算法”指首轮性能探针。

结论：叶去重减少了约 3.2% 的生产日志叶输入数，但短时配对测试没有证明正的墙钟收益。当前大部分时间用于自博弈，不是梯度更新。加速优先级应为 CPU 搜索热点、编码及 worker/推理组织，其次是 AMP 训练。硬件升级速度仅作规划估计，不能由显卡理论算力直接换算。

## 来源与边界

- 当前目录：`data/train_rules5_20261004`，已完成 10 轮；完整检查点为 iteration=10，训练配置 steps=400、batch=1024、8 worker、800/3200 simulations、strong interval=8、leaf batch=64、chance on/k=0、PCR off。
- 当前设备：i9-13900HX、RTX 4060 Laptop 8GB、约 32GB RAM、Windows；PyTorch 2.6.0+cu118。cuDNN TF32 开启，matmul TF32 关闭。
- 独立探针脚本：`data/pilots/perf_20261005/probe.py`。首次短测为 `result.json`，扩展测量为 `result_large.json`。
- 探针两次检查生产 checkpoint、latest、state、日志及三个核心源文件的哈希，全部未变。不推进主线计数、不保存探针优化后的权重、不将生成的两局数据加入 replay。
- 本次没有修改 `azero.py` 或训练默认配置；没有重新执行全量规则回归，因为没有修改运行算法。

## 生产耗时

最新五轮（索引 5 至 9）平均自博弈 8127.15s，即约 135.45 分钟。自博弈汇总至训练日志的平均间隔 149.15s，包含重放文件保存、固定训练前诊断及其他记账，不是纯优化器计时。

该间隔只占两段合计的 1.80%。即使把它完全消除，也不能大幅加速整轮。以下时间不含第 20 轮的额外对弈评估，不能外推评估轮或更高 train_steps。

## 叶去重净收益

测试从当前模型的两局 64-simulation 对局采集 16 个固定根，覆盖开局、中盘与后段。每轮测试把这些根重复四组、使用不同固定种子，共 64 个搜索，其中 8 个采用 3200，其余采用 800 simulations。复用已预热的 8-worker 池，做六组配对，开关顺序交替，不计进程启动和局面生成。

| 测量项 | 关闭去重 | 开启去重 |
|---|---:|---:|
| 平均墙钟时间 | 8.4774s | 8.5827s |
| 每轮实际叶输入数 | 64038 | 61705 |
| 平均 worker 累计时间 | 59.6062s | 60.2052s |
| 叶批评估累计时间 | 22.6380s | 20.9502s |
| 签名构造累计时间 | 0 | 2.2784s |

开启时收集的叶数为 64013，去掉 2308 条重复输入，减少 3.61%。叶批评估节省约 1.69 worker-seconds，但签名构造新增约 2.28 worker-seconds，另有分组及回映射开销。累计 worker 时间不能直接当作墙钟时间。

墙钟均值开启约慢 1.2%；六组的节省比例范围为 -3.7% 至 +1.8%。以配对为单位的描述性 bootstrap 均值区间约 [-2.60%, +0.29%]，包含零。样本很小，存在持续频率/温度变化，不能把它视为生产全局的严格统计结论，也不能宣布开启必然慢 1.2%。首次更短测试几乎持平。

生产最新五轮日志减少的是 3.16% 至 3.26% 的叶输入数，不是实测节省同等百分比的时间。现阶段没有证据说它每轮净省下几分钟。

数值边界：CUDA 上改变 batch 形状并不保证逐位相同。探针中开关去重的搜索结果不是逐位等价，根策略最大 L1 差约 0.149，64 个温度采样着法中 49 个相同。额外对同一批 22 个实际含重复的叶 batch 检查，所有合并条目的编码张量完全相同；完整批与唯一批前向输出最大 policy 差约 0.000148、value 差约 0.00346。没有据此发现错误合并；但不能宣称切换去重保留固定种子的搜索轨迹。相同精度设置下不同 CUDA kernel 的数值差异是可观测因素，尚未分解其全部来源。

参考：[PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

## 网络训练与推理探针

训练使用最新文件中固定的 1024 条原始样本；每种模式独立加载模型和 AdamW 状态、预热 3 步、测量 12 步。只验证速度和有限值，不验收学习效果。loss 采用 FP32，FP16 使用动态 GradScaler，未保存探针权重。

| 模式 | 每步完整耗时 | CPU 编码 | 传输 | 前后向及更新 |
|---|---:|---:|---:|---:|
| FP32 | 341.1ms | 161.0ms | 8.9ms | 162.0ms |
| AMP FP16 | 243.1ms | 158.0ms | 8.9ms | 69.6ms |
| AMP FP16 + channels_last | 380.4ms | 150.3ms | 9.9ms | 215.6ms |

分项之和小于总计，剩余为计时框架、对象释放及其他 Python 开销。FP32/AMP 两次测量分别约 344/243 和 341/243ms，AMP 完整步吞吐约 1.4 倍、CUDA 分配峰值约从 1.84GB 降至 1.09GB。这不是只测纯 GPU 的 2.3 倍。channels_last 在当前模型/软件栈下更慢，不推荐直接开启。

单 worker 的完整叶评估（batch64，包含编码、传输、网络、softmax 及 2550 维 Python 列表回传，预热后 40 次）另测为：FP32 10.15ms、matmul TF32 9.80ms、AMP FP16 9.97ms。该短测未显示有意义的推理加速，不能把训练 AMP 的收益推广到自博弈。AMP 最大 policy/value 输出差约 0.00511/0.00441，TF32 约 0.000315/0.000537，仅为该固定输入集合的最大误差，不是模型全局误差界。

AMP 是成熟方案，但本项目尚无生产 AMP 参数和 scaler 完整保存恢复。若实现，需要加入开关、检查点 scaler 状态与兼容检查，再用固定诊断和配对对弈验收，不能仅因没有 NaN 就启用主线。参考：[PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)、[channels_last](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)。

## 值得推进的加速清单

- [x] 根据当前生产日志拆分自博弈和训练耗时。
- [x] 8 worker 配对测去重开销及净收益，并记录 CUDA 数值边界。
- [x] 隔离测试 FP32、AMP 和 channels_last，不改变生产状态。
- [ ] 扫描 worker=4/8/12/16；先测吞吐、CPU/GPU 占用及峰值 RAM，不直接假定 worker 越多越快。
- [x] 对编译后版本扩展多局面 profile：24 根、800/3200、串行与 8 worker，并补低扰动扩展边界测量；节点扩展约占累计搜索时间 33%–34%，合法着查询约 11%，下一编译候选调整为扩展/初始化优先。详见 `docs/reviews/2026-10-05-compiled-hotspot-profile.md`；不是整轮加速承诺。
- [x] 在同批内复用已经计算的重复历史表示，减少签名与编码的重复处理；完整输入边界不变。后续两项优化及实测见 `docs/reviews/2026-10-05-selfplay-lowrisk-acceleration.md`。
- [x] 优化训练编码，限制缓存内存。后续已完成预分配批次和精确查表，详见严格等价加速记录；预取尚未启用。把全部 4M 条数据展开为 FP32 输入需要约 343GB，仅输入张量就不可行。
- [x] 给规则计算的纯函数热点做 Cython 实验：攻击判定已编译并保留 Python 回退，对照动作顺序、禁着、终局、apply/undo、完整树图与固定种子自博弈；见 `docs/reviews/2026-10-05-native-rules-acceleration.md`。合法性过滤本身仍保留 Python，未宣称全部规则已编译。
- [x] 节点扩展/初始化的小范围精确编译实验：完整树、策略/RNG、576 组 CUDA 配对搜索及完整对局样本一致，但整体没有可靠加速；默认不启用，仅保留可选实验，见 `docs/reviews/2026-10-05-native-search-experiment.md`。
- [x] 合法着快路精确编译实验：微测耗时下降约 37%–40%，两次独立 8-worker 六组配对平均省 1.82% / 1.11%，波动较大；1152 组配对、完整树及完整对局样本一致，默认仍不启用，见 `docs/reviews/2026-10-06-native-legal-experiment.md`。
- [ ] 评估集中 GPU 推理服务与跨 worker 合批；主要收益可能来自少建 CUDA 上下文、减少小批启动及密集 Python 输出转换，尚无实测加速承诺。
- [ ] AMP 训练按上述检查点和学习验收条件接入。当前 train_steps=400 时整轮收益预计不足 1%，并非第一优先级。

PCR 是另一个可实验方向，但它改变数据生成和策略监督，不属于语义不变的工程加速。普通局 full_prob=0.25、800/100 sims，保留 1/8 的 strong3200 时，等长度且成本线性假设下，平均模拟预算由 1100 降为 640.625，比例约 1.72 倍。这不是整轮实测值，也不保证同棋力收敛更快；当前 PCR fast 档仍记录 value，且其探索配置不完全复现 KataGo。应另做隔离实验，不直接切换主线。

不优先尝试扩大 leaf batch（会改变批量 MCTS 的回传时序）、降低全局模拟数、粗糙缓存信息集、跨根复用 Q 或低比特量化。它们不能仅凭吞吐提升就判为等质量加速。

## Ultra 270K + RTX 5080 的规划估计

将名称理解为桌面 Core Ultra 7 270K Plus + 桌面 RTX 5080。CPU 为 24 核（8P+16E），GPU 为 16GB GDDR7、10752 CUDA cores。当前 i9-13900HX 本身已有 8P+16E，不能由新 CPU 的名称或核心数推断翻倍。

在正常散热、足够 RAM、兼容 PyTorch/CUDA、保持 8 worker/现有搜索和 FP32 配置的条件下，以下是宽范围估计，不是目标硬件实测：

| 阶段 | 相对当前笔记本吞吐 | 对应当前负载时间 |
|---|---:|---:|
| 纯 GPU 前后向/更新部分 | 约 3 至 5 倍 | 不直接等于完整训练步 |
| 完整网络训练阶段，含 CPU 解码 | 约 1.7 至 2.5 倍 | 400 步约 60 至 90 秒 |
| 自博弈及完整普通轮次 | 约 1.5 至 2.5 倍 | 目前约 138 分钟，升级后约 55 至 90 分钟 |

推理、CPU、CUDA 上下文及输出列表转换混合在实测时间中，因此无法由本次 cProfile 精确分解出目标机器速度。上述范围仅用于预算，额外 worker/推理组织优化的收益未计入，不能承诺 4 倍或更高的整轮吞吐。

5080 不能直接沿用当前 PyTorch 2.6+cu118 环境；需在独立环境安装明确支持 Blackwell/sm_120 的构建（PyTorch 2.7 起提供 CUDA12.8 支持），运行项目回归及同探针确认。新软件栈和设备不保证旧训练轨迹逐位一致。建议 64GB 系统内存作为 4M replay 与增加 worker 的余量，不代表单纯加 RAM 一定提速。CPU 的 NPU 和 GPU 游戏帧率不是本流程速度指标。

来源：[Intel 270K Plus](https://www.intel.com/content/www/us/en/products/sku/245692/intel-core-ultra-7-processor-270k-plus-36m-cache-up-to-5-50-ghz/specifications.html)、[NVIDIA RTX5080](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)、[PyTorch 2.7 Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)。

重跑扩展探针时必须使用新的输出文件名，默认拒绝覆盖：

```powershell
python data/pilots/perf_20261005/probe.py --pairs 6 --blocks 4 --output result_new.json
```
