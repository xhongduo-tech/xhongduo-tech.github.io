## 核心结论

TFLOPS 衡量“每秒做了多少浮点计算”，MFU 衡量“理论峰值算力里有多少真正转化成模型训练的有效计算”。

一句话定义如下：

| 指标 | 白话解释 | 主要回答的问题 |
|---|---|---|
| `tokens/s` | 每秒处理多少 token，token 是模型读写文本的最小片段 | 训练吞吐有多高 |
| `TFLOPS` | 每秒完成多少万亿次浮点计算 | 硬件实际算得有多快 |
| `MFU` | Model FLOPs Utilization，模型有效 FLOPs 占理论峰值的比例 | 硬件算力有多少用于有效训练 |

核心公式是：

$$
TFLOPS_{achieved} = \frac{S \times F_{token}}{10^{12}}
$$

$$
MFU = \frac{TFLOPS_{achieved}}{TFLOPS_{peak}} = \frac{S \times F_{token}}{F_{peak}}
$$

其中 `S` 是实际训练吞吐，单位是 tokens/s；`F_token` 是每处理 1 个 token 所需的有效训练 FLOPs；`F_peak` 是硬件理论峰值 FLOPs/s。

新手版例子：两台机器都标称“1000 TFLOPS”。第一台机器大部分时间在等跨卡通信，第二台机器大部分时间在执行矩阵乘法。它们的峰值 TFLOPS 一样，但第一台的 `tokens/s` 更低，MFU 也更低。训练系统的评估不能只看峰值算力，必须同时看 `tokens/s`、实际 TFLOPS 和 MFU。

经验上，优秀的大模型训练系统在 A100 上约能达到 40%-50% MFU，在 H100 上约能达到 45%-55% MFU。这个范围不是硬标准，它依赖模型结构、精度、并行策略、通信网络和 kernel 优化水平。

---

## 问题定义与边界

本文讨论的是训练效率度量，不讨论模型精度、收敛速度，也不讨论推理吞吐。

训练效率度量的核心问题是：给定一个模型、一个训练吞吐和一组硬件，如何判断这套系统是否把硬件用得足够好。

边界必须先说清楚：

| 术语 | 定义 | 注意点 |
|---|---|---|
| `TFLOPS` | Tera FLOPs per second，每秒 $10^{12}$ 次浮点计算 | 可以是实际值，也可以是硬件峰值 |
| `MFU` | 有效模型训练 FLOPs / 理论峰值 FLOPs | 通常不把重算算作有效模型 FLOPs |
| `HFU` | Hardware FLOPs Utilization，实际硬件执行 FLOPs / 理论峰值 FLOPs | 会把 rematerialization/recompute 等额外计算算进去 |
| `tokens/s` | 每秒处理的 token 数 | 最贴近端到端训练吞吐 |

术语首次出现时需要区分：FLOPs 是浮点运算次数，表示做了多少计算；FLOPS 是每秒浮点运算次数，表示计算速度。

本文默认以下前提：

| 前提 | 说明 |
|---|---|
| 模型类型 | dense Transformer，即每个 token 大体经过同样的层和参数 |
| 阶段 | 训练阶段，包括 forward 和 backward |
| 硬件口径 | 使用统一的 GPU 型号、数量、精度峰值 |
| 统计范围 | 单卡就用单卡分子和分母，全集群就用全集群分子和分母 |

变量定义如下：

| 变量 | 含义 |
|---|---|
| `P` | 模型参数量 |
| `L` | Transformer 层数 |
| `H` | 注意力头数 |
| `d` | 每个注意力头的维度 |
| `T` | 序列长度 |
| `S` | 实际吞吐，tokens/s |
| `F_peak` | 理论峰值 FLOPs/s |

玩具例子：一个训练任务日志里写着 `25,000 tokens/s`。这只能说明吞吐，不说明硬件效率。要判断这 25,000 个 tokens 背后用了多少算力，还要估算每个 token 的 FLOPs，再和硬件理论峰值相比。

---

## 核心机制与推导

MFU 的推导从一个问题开始：模型每处理 1 个 token 需要多少有效 FLOPs。

对 dense Transformer，常用近似是：

$$
F_{token} \approx 6P + 12LHTd
$$

这里的 `6P` 来自训练时参数相关计算。粗略理解是：一次训练不只有 forward，还包括 backward 中对激活和参数梯度的计算，所以训练 FLOPs 通常远高于单次前向。`12LHTd` 是注意力随序列长度增长的部分，长上下文训练时不能忽略。

推导链路如下：

```text
tokens/s -> FLOPs/token -> FLOPs/s -> TFLOPS -> MFU
```

分步写就是：

$$
FLOPs/s = S \times F_{token}
$$

$$
TFLOPS_{achieved} = \frac{S \times F_{token}}{10^{12}}
$$

$$
MFU = \frac{S \times F_{token}}{F_{peak}}
$$

为什么要除以峰值？因为单看实际 TFLOPS 无法判断系统是否高效。`1050 TFLOPS` 对 8 张 A100 PCIe 来说可能不错，对 8 张 H100 SXM 来说就偏低。除以同一口径下的理论峰值后，MFU 才能把不同硬件规模拉到同一比例尺上。

新手版数值例子：假设一个 7B dense 模型，先忽略注意力项，取

$$
F_{token} \approx 6P = 6 \times 7 \times 10^9 = 4.2 \times 10^{10}
$$

如果系统吞吐是 `25,000 tokens/s`，则：

$$
FLOPs/s = 25000 \times 4.2 \times 10^{10} = 1.05 \times 10^{15}
$$

也就是：

$$
TFLOPS_{achieved} = 1050
$$

如果硬件是 `8 x A100 80GB PCIe`，按 BF16/FP16 Tensor Core 峰值 `312 TFLOPS/卡` 计算，总峰值是 `2496 TFLOPS`，则：

$$
MFU = \frac{1050}{2496} \approx 42.1\%
$$

真实工程例子：PaLM 540B 论文报告了约 `46.2% MFU` 和 `57.8% HFU`。这两个数不同，原因是 PaLM 使用了 rematerialization，也就是为了节省显存而重新计算部分中间激活。MFU 更关注模型必要训练计算，HFU 会把这些额外硬件计算也统计进去。

---

## 代码实现

下面代码把公式变成一个可复用计算器。它不查询硬件规格，硬件峰值需要调用方按 GPU 型号、数量和精度传入。

```python
def estimate_training_efficiency(P, L, H, d, T, tokens_per_sec, peak_flops):
    """
    P: parameter count
    L: number of Transformer layers
    H: number of attention heads
    d: head dimension
    T: sequence length
    tokens_per_sec: measured training throughput
    peak_flops: hardware peak FLOPs/s for the same device scope
    """
    f_token = 6 * P + 12 * L * H * T * d
    achieved_flops = tokens_per_sec * f_token
    achieved_tflops = achieved_flops / 1e12
    mfu = achieved_flops / peak_flops
    return {
        "F_token": f_token,
        "TFLOPS": achieved_tflops,
        "MFU": mfu,
        "tokens/s": tokens_per_sec,
    }


# Toy example: 7B model on 8 x A100 PCIe, using 312 TFLOPS/card.
result = estimate_training_efficiency(
    P=7_000_000_000,
    L=32,
    H=32,
    d=128,
    T=2048,
    tokens_per_sec=25_000,
    peak_flops=8 * 312e12,
)

assert result["TFLOPS"] > 1050
assert 0.42 < result["MFU"] < 0.45

print(result)
```

示例输出可整理成表：

| 字段 | 示例值 | 含义 |
|---|---:|---|
| `tokens/s` | 25,000 | 实测吞吐 |
| `F_token` | 约 $4.24 \times 10^{10}$ | 每 token 有效训练 FLOPs |
| `TFLOPS` | 约 1060 | 实际训练计算吞吐 |
| `MFU` | 约 42.5% | 相对理论峰值的有效利用率 |

代码里的口径约定很重要。`peak_flops` 必须和 GPU 型号、数量、精度类型一致。例如 A100 PCIe、A100 SXM、H100 PCIe、H100 SXM 的峰值不同；BF16、FP16、FP8 的峰值也不同。集群统计时，分子用全集群 `tokens/s`，分母也必须用全集群峰值。

---

## 工程权衡与常见坑

高 TFLOPS 不等于高 MFU。训练系统的时间会被矩阵乘法、通信、显存访问、kernel launch、CPU 调度、数据加载和同步等待共同瓜分。只有模型有效 FLOPs 占比高，MFU 才会上升。

常见坑如下：

| 坑 | 后果 | 规避建议 |
|---|---|---|
| 只报 TFLOPS，不报 `tokens/s` | 看不出真实训练吞吐 | 同时报 `tokens/s`、TFLOPS、MFU |
| 单卡峰值和全集群峰值混用 | MFU 被放大或缩小 | 分子分母使用同一统计范围 |
| BF16、FP16、FP8 峰值混算 | 不同实验不可比 | 写清 GPU 型号、精度和是否使用稀疏峰值 |
| 把 HFU 当成 MFU | 重算多的实验看起来虚高 | 明确是否把 recompute FLOPs 算进去 |
| 忽略通信开销 | 多卡扩展效率误判 | 观察 all-reduce、all-gather、reduce-scatter 占比 |
| 忽略 kernel 碎片化 | 小算子拖慢 step | 使用融合 kernel、编译优化或更合适的算子实现 |

真实工程中，一个配置可能单卡算力很高，但用了过多 TP/PP，通信变得频繁。另一个配置做了更合理的并行切分，并且使用了 fused attention、fused MLP、FlashAttention 或编译器融合。前者的单个 kernel 看起来快，但端到端 `tokens/s` 可能更低，MFU 也更差。

工程诊断清单：

| 问题 | 观察方向 |
|---|---|
| 通信占比是否过高 | profiler 中 collective 操作时间 |
| 是否存在大量小 kernel | kernel 数量、launch 间隔、GPU idle 时间 |
| 并行策略是否合理 | DP、TP、PP、CP 的切分比例 |
| 是否有足够融合 | attention、MLP、norm、optimizer 是否融合 |
| 重算策略是否过重 | recompute 带来的额外 FLOPs 和节省的显存是否值得 |
| 显存带宽是否成为瓶颈 | 算术强度低的算子是否占比过高 |

优化目标不是追一个单点峰值，而是让更多 step 时间变成有效训练时间。MFU 高通常意味着矩阵乘法占比高、通信隐藏好、kernel 数量少、显存访问模式稳定。

---

## 替代方案与适用边界

MFU 适合比较训练系统的有效硬件利用率，但不是唯一指标。

| 指标 | 适合回答的问题 | 局限 |
|---|---|---|
| `MFU` | 有效模型训练计算用了多少硬件峰值 | 对 MoE、稀疏模型、长上下文公式敏感 |
| `HFU` | 硬件实际执行了多少计算 | 重算多时可能高于 MFU，但不一定代表训练更快 |
| `tokens/s` | 真实训练吞吐是多少 | 不反映模型大小和硬件规模 |
| `samples/s` | 每秒处理多少样本 | NLP 中样本长度不同，可比性弱 |
| `step time` | 每个训练迭代多久 | batch、序列长度不同会影响解释 |
| `cost/token` | 每训练一个 token 花多少钱 | 需要额外知道机器价格、利用率和能耗 |

新手版判断：如果一个实验大量使用重算，MFU 可能不高，但这不一定代表系统差。重算可能是在显存约束下换取更大 batch、更长序列或更稳定的并行策略。此时应该同时看 `tokens/s`、显存占用、step time 和最终训练成本。

MFU 的适用边界：

| 场景 | 是否适合直接用 MFU |
|---|---|
| dense Transformer 预训练 | 适合 |
| 相同模型、相同硬件、不同并行策略 | 很适合 |
| MoE 模型 | 需要修正有效 FLOPs，因为不是每个 token 经过所有专家 |
| 稀疏模型 | 需要说明稀疏计算口径 |
| 非 Transformer 模型 | 需要重新估算 `F_token` |
| 长上下文训练 | 注意力项不能忽略 |
| 强重算训练 | 必须区分 MFU 和 HFU |

何时不用 MFU 作为主指标：当目标是端到端产品吞吐、推理延迟、训练成本或收敛速度时，MFU 只能作为辅助指标。比如一个训练方案 MFU 更高，但需要更小 batch，导致收敛更慢或成本更高，它就不一定是更好的工程方案。

---

## 参考资料

1. [PaLM: Scaling Language Modeling with Pathways](https://research.google/pubs/palm-scaling-language-modeling-with-pathways/)：论文来源，用于理解 MFU/HFU 定义和 PaLM 工程结果。
2. [PaLM Appendix B, ar5iv](https://ar5iv.labs.arxiv.org/html/2204.02311)：公式参考，用于核对训练 FLOPs 与利用率口径。
3. [karpathy/nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)：工程实现参考，包含 `estimate_mfu` 的简化计算方式。
4. [NVIDIA A100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/a100/)：硬件规格参考，用于确认 A100 不同形态和精度的峰值。
5. [NVIDIA H100 Tensor Core GPU](https://www.nvidia.com/en-eu/data-center/h100/)：硬件规格参考，用于确认 H100 的 BF16/FP16/FP8 峰值和带宽。
6. [NVIDIA Megatron-Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)：工程优化参考，用于理解大规模训练的性能诊断方向。
7. [NVIDIA Transformer Engine Performance Considerations](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html)：低精度训练与性能优化参考。
