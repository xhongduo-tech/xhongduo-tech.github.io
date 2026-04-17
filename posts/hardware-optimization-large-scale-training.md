## 核心结论

大规模训练的硬件优化，不是单独比较“哪张卡更快”，而是同时优化计算芯片、芯片间互联、节点内存、网络、存储与监控链路。对训练任务来说，真正决定吞吐的通常不是峰值算力本身，而是整条数据路径能否持续把算力喂满。

GPU 适合通用性强、软件生态成熟的训练场景。这里的“软件生态”是指编译器、通信库、训练框架和调试工具都比较完整。H100 代表更高的 Tensor Core 密度、更快的 HBM3 显存和更强的 NVLink 互联，适合超大模型与更激进的低精度训练；A100 仍然是很多团队的主力，因为它的成本、供货和兼容性更平衡。TPU 更适合在 Google Cloud 内做大规模切片部署，切片就是把许多加速芯片按固定拓扑连成一个训练整体，重点看 ICI 带宽和拓扑形态。

判断硬件是否真的被用好，最实用的指标是 MFU。MFU 的白话解释是“模型训练时，实际跑出来的有效浮点吞吐，占理论峰值的多少”。如果 MFU 明显偏低，优先检查数据加载、网络通信、并行切分和调度，而不是先加卡。

| 硬件 | 关键特性 | 典型场景 |
| --- | --- | --- |
| H100 GPU | Transformer Engine、HBM3、NVLink 高带宽 | 超大语言模型训练、FP8/混合精度 |
| A100 GPU | 80GB HBM2e、生态成熟、成本更稳 | 中型模型训练、预算受限集群 |
| TPU v5p | 3D torus 拓扑、ICI 高带宽、大规模 Pod | Google Cloud 上的大规模切片训练 |
| 专用 AI 芯片 | 矩阵乘加密度高、功耗效率可观 | 特定框架或垂直场景优化 |

玩具例子可以先这样理解：H100 像“更快显存加更宽芯片高速公路”的系统，TPU v5p 像“很多芯片按规则织成大网”的系统。两者都不能只看单芯片峰值，必须看集群整体是否稳定维持高 MFU。

---

## 问题定义与边界

这篇文章讨论的是“训练阶段”的硬件优化，不讨论推理服务、不讨论消费级单机训练，也不讨论芯片设计本身。目标是回答三个问题：

1. 怎么在 GPU、TPU 和其他 AI 加速器之间做选型。
2. 怎么判断瓶颈到底在算力、显存、网络还是存储。
3. 集群规模上去后，怎么避免单点故障和隐性降速把训练拖垮。

边界要先说清楚。大规模训练通常指参数量足够大、数据量足够多，必须依赖多卡或多节点并行。并行的白话解释是“把一份训练工作拆给很多设备同时做”。这时硬件问题不再是“单机快不快”，而是“整条流水线有没有短板”。

一个实用的分层方法如下：

| 层级 | 主要约束 | 典型指标 |
| --- | --- | --- |
| 计算层 | GPU/TPU 峰值 FLOPs、低精度能力 | Tensor Core、FP8/BF16 支持 |
| 显存层 | HBM 容量与带宽 | 单卡可容纳的激活、参数、优化器状态 |
| 数据层 | CPU 核数、CPU 内存、NUMA 绑定 | 预处理速度、dataloader 等待时间 |
| 通信层 | NVLink、PCIe、InfiniBand、ICI | AllReduce 时间、跨节点同步开销 |
| 存储层 | 本地 NVMe、对象存储、分布式文件系统 | 数据预热速度、吞吐抖动 |
| 运维层 | 温度、功耗、ECC、故障检测 | 健康告警、自动隔离、重试能力 |

新手最容易误判的是把“GPU 利用率高”当成训练已经优化好。这个判断不够。GPU 利用率只是芯片某些时刻在忙，不等于整个训练吞吐已经接近最优。比如 dataloader 偶尔堵塞、网络同步很慢、checkpoint 写盘过频，都可能让训练步之间出现大量空转。

真实工程里，经常出现这样的情况：单机测试很快，上了 64 卡之后反而扩展性很差。原因不是卡不够强，而是并行带来的通信复杂度、数据一致性和存储争用开始主导总时间。

---

## 核心机制与推导

大规模训练的核心矛盾可以写成一句话：每一步训练的总耗时，等于“计算时间”和“等待时间”的合成，而等待时间主要来自数据、通信和 I/O。

如果用更形式化的表达，一个训练 step 的时间可以近似写成：

$$
T_{step} \approx T_{compute} + T_{data} + T_{comm} + T_{io} + T_{sync}
$$

其中：

- $T_{compute}$ 是矩阵乘法等真正做计算的时间。
- $T_{data}$ 是数据读取与预处理时间。
- $T_{comm}$ 是梯度或参数在设备间传输的时间。
- $T_{io}$ 是 checkpoint、日志、缓存刷盘等 I/O 时间。
- $T_{sync}$ 是并行训练中的同步等待时间。

训练系统优化，本质上是在压低后四项，让第一项尽可能占主导。

MFU 的定义正好服务这个目标。常见近似写法是：

$$
MFU = \frac{\text{FLOPs per token} \times \text{tokens/s}}{\text{peak FLOPs}}
$$

如果是多卡训练，分母要用整个训练系统的总峰值算力。这个式子的意义很直接：模型每处理一个 token 需要一定 FLOPs，系统每秒能处理多少 token，就得到实际持续输出的 FLOPs；再除以理论峰值，就得到“算力被真正转化为训练吞吐的比例”。

一个玩具例子：

- 模型规模约 7B。
- 经验估算 FLOPs/token 约为 $6 \times 7 \times 10^9 = 4.2 \times 10^{10}$。
- 8 张 A100，总峰值按 $8 \times 312 \times 10^{12}$ FLOPs 估算。
- 实测吞吐是 8000 tokens/s。

则：

$$
\text{sustained FLOPs} = 4.2 \times 10^{10} \times 8000 = 3.36 \times 10^{14}
$$

$$
MFU \approx \frac{3.36 \times 10^{14}}{2.496 \times 10^{15}} \approx 13.5\%
$$

13.5% 说明什么？说明瓶颈大概率不在“再换一批更快 GPU”之前，而在系统没有把现有 GPU 喂满。这个结论对初学者很重要，因为很多人会直接把低吞吐理解成“算力不够”，实际上常常是通信、存储或并行配置有问题。

为什么 H100 往往比 A100 在大模型训练里优势更明显？原因不只是一张卡更快，而是三件事同时成立：

| 机制 | 对训练的影响 | 为什么重要 |
| --- | --- | --- |
| 更高显存带宽 | 降低大矩阵读写等待 | Transformer 训练经常受显存流量影响 |
| 更强低精度支持 | 提高单位时间有效算力 | 在精度可控时能显著提升吞吐 |
| 更强互联 | 降低多卡同步开销 | 大规模并行时通信经常是主瓶颈 |

TPU 的思路则更偏“系统级一体化”。它的优势不是单点上和 GPU 做所有维度逐项对比，而是通过固定拓扑、编译器和切片方式，把大规模通信路径做得更规则。在规模足够大时，规则拓扑比“拼装式集群”更容易获得稳定吞吐。

但这也带来边界：如果你的团队主要依赖 PyTorch 生态、需要灵活地改算子、换并行策略、接各种第三方训练组件，GPU 通常更省工程成本。这里的工程成本是指“把系统跑起来、调通、监控、迁移、排障”的总人力成本，不只是硬件采购价。

---

## 代码实现

下面给出一个最小可运行的 MFU 监控脚本。它不依赖具体训练框架，只要训练日志里能提供吞吐、理论 FLOPs/token 和集群峰值算力，就可以先做第一层告警。

```python
from dataclasses import dataclass


@dataclass
class TrainMetrics:
    flops_per_token: float
    tokens_per_sec: float
    peak_flops: float
    gpu_temp_c: float
    data_wait_ratio: float
    comm_wait_ratio: float


def compute_mfu(flops_per_token: float, tokens_per_sec: float, peak_flops: float) -> float:
    assert flops_per_token > 0
    assert tokens_per_sec >= 0
    assert peak_flops > 0
    sustained_flops = flops_per_token * tokens_per_sec
    return sustained_flops / peak_flops


def diagnose(metrics: TrainMetrics) -> str:
    mfu = compute_mfu(
        metrics.flops_per_token,
        metrics.tokens_per_sec,
        metrics.peak_flops,
    )

    if metrics.gpu_temp_c >= 85:
        return f"MFU={mfu:.2%}，优先检查散热与降频"
    if mfu < 0.25 and metrics.data_wait_ratio > metrics.comm_wait_ratio:
        return f"MFU={mfu:.2%}，疑似数据管线瓶颈"
    if mfu < 0.25 and metrics.comm_wait_ratio >= metrics.data_wait_ratio:
        return f"MFU={mfu:.2%}，疑似通信瓶颈"
    return f"MFU={mfu:.2%}，当前训练效率可接受"


# 玩具例子：8 x A100，7B 模型
toy = TrainMetrics(
    flops_per_token=6 * 7e9,
    tokens_per_sec=8000,
    peak_flops=8 * 312e12,
    gpu_temp_c=72,
    data_wait_ratio=0.18,
    comm_wait_ratio=0.09,
)

toy_mfu = compute_mfu(toy.flops_per_token, toy.tokens_per_sec, toy.peak_flops)
assert round(toy_mfu, 3) == 0.135
assert "数据管线瓶颈" in diagnose(toy)

# 真实工程风格例子：64 x H100，吞吐更高，但通信等待明显
realish = TrainMetrics(
    flops_per_token=6 * 70e9,
    tokens_per_sec=42000,
    peak_flops=64 * 1979e12,   # 示例值：按 H100 低精度峰值近似
    gpu_temp_c=78,
    data_wait_ratio=0.05,
    comm_wait_ratio=0.22,
)

realish_mfu = compute_mfu(realish.flops_per_token, realish.tokens_per_sec, realish.peak_flops)
assert realish_mfu > 0
print(diagnose(realish))
```

这段代码只解决第一层问题：把“低效”定量化。真正工程化时，还要把它接到监控系统里，例如：

- 从训练日志采集 `tokens_per_sec`、step time、dataloader wait。
- 从 DCGM 采集温度、功耗、显存错误、时钟降频。
- 从网络层采集通信耗时、重传、带宽占用。
- 从存储层采集读吞吐、缓存命中率、checkpoint 写入延迟。

一个真实工程例子：假设你在 128 卡集群上训练 70B 模型，发现 GPU 功耗看起来正常，但 MFU 长期只有 22%。这时最容易犯的错是继续调 batch size。更有效的排查顺序通常是：

1. 看 dataloader 等待时间是否明显抬高。
2. 看 AllReduce 或其他同步操作是否占 step 的大头。
3. 看 checkpoint 是否写得过于频繁。
4. 看是否存在个别热点节点温度高、频率掉得快，导致全局同步被最慢节点拖住。

也就是说，MFU 像总报警器，DCGM、网络遥测和存储指标像分诊工具。前者告诉你“系统效率不对”，后者告诉你“究竟是哪一段不对”。

---

## 工程权衡与常见坑

大规模训练里，最昂贵的错误通常不是“买错一张卡”，而是“系统某个短板在千卡规模下被放大”。下面是最常见的一组坑。

| 坑 | 原因 | 技术对策 |
| --- | --- | --- |
| GPU 空转多 | 数据预处理、存储或网络跟不上 | 预热到 NVMe、异步加载、提升网络带宽 |
| 扩卡后不线性加速 | 通信复杂度上升，同步等待变重 | 优化并行策略，缩短跨节点路径 |
| 单卡异常拖垮任务 | 分布式训练常被最慢或故障节点卡住 | 健康检查、自动隔离、失败重试 |
| 温度高后性能抖动 | 降频或硬件不稳定 | 监控温度、功耗、ECC，做节点轮换 |
| checkpoint 成为瓶颈 | 多节点同时写盘，元数据争用严重 | 分层存储、异步保存、降低频率 |

先说硬件选型的典型权衡。

H100 更适合把单机密度做高，特别是在超大模型、激活重计算、低精度训练已经较成熟的团队里。A100 更适合预算受限、已有成熟 CUDA 栈、并且对“绝对最快”没有那么强要求的团队。TPU v5p 更适合能接受云上约束、希望获得大规模规则拓扑和稳定切片部署的场景。

再说一个容易被忽略的点：CPU 和 CPU 内存并不是附属品。训练前的数据解码、分词、打包、shuffle、样本拼接都可能在 CPU 侧完成。如果 CPU 核数不够、NUMA 绑定混乱、内存带宽不稳，就会形成“GPU 很贵，CPU 在拖后腿”的典型失衡。

另一个常见坑是只做“训练成功”监控，不做“训练效率”监控。训练 job 没挂，不代表值得继续跑。一个持续 15% MFU 的任务，即使没有报错，也可能在烧预算。初级工程师经常把“没报错”当成系统正常，这在单机实验里问题不大，在大集群里会直接变成成本问题。

真实工程里，容错策略必须前置设计。多日训练中，总会遇到偶发硬件异常、链路抖动、局部温升、ECC 错误或存储超时。没有自动告警、隔离和 rerun 机制时，单点故障会放大成整批任务失败。训练规模越大，这个问题越严重。

---

## 替代方案与适用边界

不是每个团队都应该追求 H100 或 TPU v5p。选择要从“模型规模、预算、团队栈、运维能力、交付时间”一起判断。

| 方案 | 优点 | 适用边界 |
| --- | --- | --- |
| H100 + NVLink | 吞吐高、适合大模型与低精度 | 预算高，追求极致训练速度 |
| A100 + NVLink | 成本更稳、生态成熟 | 中型模型、已有 GPU 软件栈 |
| TPU v5p slices | 大规模拓扑规则、带宽高 | 接受云平台约束，适合超大训练 |
| 小规模多机 A100 | 采购和维护压力更低 | 团队早期验证、预算敏感 |
| 专用 AI 芯片 | 可能有更好功耗效率 | 软件生态与迁移成本需单独评估 |

一个现实判断原则是：

- 如果你还没有形成稳定的多机监控、自动化部署和失败恢复流程，不要一开始就上过大集群。
- 如果你的模型还处在快速迭代阶段，软件灵活性通常比理论峰值更重要。
- 如果训练任务高度固定、规模很大、云资源可控，TPU 或专用加速器的系统化优势会更明显。
- 如果是中小团队，4 到 8 张 A100 的稳定系统，往往比“偶尔能借到但维护困难”的更大集群更有实际价值。

对零基础到初级工程师来说，最重要的不是背硬件参数表，而是建立一个判断框架：

1. 先看模型需要多少显存和多大并行规模。
2. 再看现有软件栈最适配 GPU 还是 TPU。
3. 用 MFU 判断当前系统是否真的被喂满。
4. 用温度、通信、I/O 和错误指标判断问题出在哪一层。
5. 在吞吐、成本、可维护性之间做可解释的取舍。

这样做的结果是，你不会把“换更贵的卡”当成默认答案，而会先证明系统已经把现有资源用到了合理水平。

---

## 参考资料

- NVIDIA A100 与 H100 的架构、Transformer Engine、HBM 与 NVLink 对比：<https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100>
- Google Cloud TPU v5p 文档，含切片、拓扑与 ICI 说明：<https://docs.cloud.google.com/tpu/docs/v5p>
- MFU 公式、训练 FLOPs 估算与案例说明：<https://debjitpaul.github.io/blog/2025/compute/>
- NVIDIA 关于 DGX Cloud 大规模训练可靠性、遥测与 rerun 的说明：<https://developer.nvidia.com/blog/ensuring-reliable-model-training-on-nvidia-dgx-cloud/>
- NVIDIA 关于数据中心 GPU 监控、DCGM 与集群效率的说明：<https://developer.nvidia.com/blog/making-gpu-clusters-more-efficient-with-nvidia-data-center-monitoring/>
- NVIDIA 关于 NVSentinel 与 Kubernetes AI 集群健康管理的说明：<https://developer.nvidia.com/blog/automate-kubernetes-ai-cluster-health-with-nvsentinel/>
- H100 与 A100 在训练场景中的补充比较：<https://www.gpu.fm/blog/h100-vs-a100-complete-gpu-comparison>
