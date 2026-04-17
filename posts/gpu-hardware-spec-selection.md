## 核心结论

GPU 选型不能只看算力。对训练效率影响最大的，通常是三件事：显存能不能装下模型，卡间通信能不能跟上同步频率，主机到 GPU 的数据通路会不会把卡喂饿。

A100、H100、H200 的代际差异，核心不是“新一代更快”这么简单，而是三组硬件指标的变化：

| GPU | HBM 容量 | HBM 带宽 | NVLink 带宽 | FP8 支持 | 适合场景 |
|---|---:|---:|---:|---|---|
| A100 80GB | 80 GB | 约 2.0 TB/s | 约 600 GB/s | 否 | 中等规模训练、成熟 BF16/FP16 流程 |
| H100 80GB | 80 GB | 约 3.35 TB/s | 约 900 GB/s | 是 | 更高吞吐训练、FP8 推进、更多并行切分 |
| H200 141GB | 141 GB | 约 4.8 TB/s | 约 900 GB/s | 是 | 大模型训练、减少切分、降低跨卡压力 |

这里有三个术语需要先说清楚。

HBM 是高带宽显存，可以理解为 GPU 旁边那一圈“超宽内存总线”，决定参数和激活值能多快被 Tensor Core 取到。  
NVLink 是 GPU 之间的高速互联，可以理解为显卡之间的专用直连通道，不走普通主板总线。  
FP8 是 8 位浮点格式，可以理解为用更小的数据表示数值，在可接受精度损失下换更高吞吐和更低显存占用。

对初学者，一个够用的判断是：

1. 小于等于 30B 级别模型，A100 仍然能做，但多卡同步和显存余量会明显紧张。
2. 32B 到 70B 级别，H100 往往是性价比更高的训练主力，前提是你愿意做 tensor parallel。
3. 70B 以上，H200 的 141GB 显存会直接改变部署方式，因为很多原本必须拆到更多卡上的参数，现在能在更少 GPU 上容纳。

如果一句话总结：A100 的优势是成熟，H100 的优势是“算得更快”，H200 的优势是“更容易装下并且同步得更稳”。

---

## 问题定义与边界

这类选型问题，真正要回答的不是“哪张卡最好”，而是：

给定模型参数规模、训练精度、优化器类型和并行策略，最少需要多少显存、多少卡、什么级别的互联带宽，才能在不频繁 OOM 的前提下把 GPU 利用率提起来。

OOM 是 out of memory，意思是显存不够，程序直接报错或被系统终止。

边界先定清楚，否则结论会漂移：

| 维度 | 典型选项 | 影响 |
|---|---|---|
| 任务类型 | 预训练 / 全参数微调 / LoRA 微调 | 决定参数、梯度、优化器状态是否全量保留 |
| 数值精度 | FP32 / FP16 / BF16 / FP8 | 决定每个参数占多少字节 |
| 并行方式 | 数据并行 DP / 张量并行 TP / 流水并行 PP | 决定通信量和显存切分方式 |
| 节点范围 | 单机 8 卡 / 多机跨节点 | 决定是否必须依赖 NVSwitch 和高速网络 |
| 数据链路 | PCIe Gen4 / Gen5 / GPUDirect | 决定数据加载是否成为瓶颈 |

数据并行是每张卡放一份完整模型，只切分样本。  
张量并行是把一个层内部的矩阵拆到多张卡。  
流水并行是把不同层放到不同卡上，按阶段传递中间结果。

本文边界明确为：讨论大模型训练或全参数微调，不展开推理服务，不展开低成本消费级卡集群，也不展开 ZeRO/FSDP 细节实现。重点是理解“规格如何映射为训练效率”。

一个基准例子：70B BF16 模型。

BF16 是 Brain Floating Point 16-bit，可以理解为 16 位浮点，但保留了更大的指数范围，训练时比 FP16 更稳。  
如果你只有 H100 80GB，那么单卡装不下完整训练状态，几乎一定要靠 TP、PP 或 ZeRO；如果是 H200 141GB，至少显存压力会大幅缓解，卡数和通信压力都更容易控制。

所以这类问题的实质不是“买哪张卡”，而是“你的模型是否会被显存边界强制推向更复杂的并行和通信架构”。

---

## 核心机制与推导

训练时显存不是只装参数。至少要装四类东西：

| 组成 | 含义 | BF16 常见估算 |
|---|---|---:|
| 参数 | 模型权重 | 2 B/参数 |
| 梯度 | 反向传播得到的梯度 | 2 B/参数 |
| 优化器状态 | Adam 的一阶、二阶动量等 | 常按 4 B/参数估算 |
| 激活与缓冲 | 前向中间结果、临时工作区 | 额外预留 10% 到 30% |

对初学者，一个够用的一阶估算公式是：

$$
M_{base} = \Psi \times (b_p + b_g + b_o)
$$

其中：

- $\Psi$ 是参数个数
- $b_p$ 是参数字节数
- $b_g$ 是梯度字节数
- $b_o$ 是优化器状态字节数

如果是 BF16 + Adam，常用近似是：

$$
b_p = 2,\quad b_g = 2,\quad b_o \approx 4
$$

于是：

$$
M_{base} = \Psi \times 8 \text{ bytes}
$$

对于 70B 模型，也就是 $\Psi = 70 \times 10^9$：

$$
M_{base} = 70 \times 10^9 \times 8 = 560 \text{ GB}
$$

这还不是最终值。工程上通常还要乘一个激活和缓冲冗余系数，再加上额外通信/框架开销：

$$
M_{total} \approx M_{base} \times 1.25 + M_{extra}
$$

若取 $M_{extra}=128$ GB，则：

$$
M_{total} \approx 560 \times 1.25 + 128 = 828 \text{ GB}
$$

如果你采用更乐观的口径，把 128GB 理解为已覆盖部分缓冲而不是全部附加，则也会看到约 700GB 的估算。差异来自“激活是否单独算、通信缓存是否单独算、是否启用 checkpointing”。

checkpointing 是激活重计算，可以理解为少存一部分中间结果，反向传播时再算一遍，用算力换显存。

这说明一件关键事实：显存估算本来就不是一个唯一数字，而是一段区间。做硬件规划时宁可高估 15%，也不要按理论下界买卡。

接着看通信。多卡训练慢，通常慢在同步梯度。一个常见近似是：

$$
T_{comm} \approx \frac{2 \times \Psi \times b_p}{B_{interconnect}} \times \log_2 N
$$

其中：

- $B_{interconnect}$ 是互联带宽
- $N$ 是参与同步的 GPU 数量

这个公式不是精确仿真，只是帮助你抓主导项。它表达了三个事实：

1. 模型越大，待同步的数据越多。
2. 带宽越小，同步越慢。
3. 参与卡数越多，树形或分层归约的轮次会上升。

玩具例子先看 8 张卡同步一个小模型。  
假设只有 8B 参数，BF16，通信带宽按 PCIe Gen4 的理想 64 GB/s 来估：

$$
T_{comm} \approx \frac{2 \times 8 \times 10^9 \times 2}{64 \times 10^9} \times \log_2 8
= 0.5 \times 3
= 1.5 \text{ s}
$$

这个数字已经告诉你一个方向：如果梯度真都要走 PCIe，同步会非常贵。真实系统虽然不会每一步都按这么粗暴的全量路径走，但结论不变，PCIe 不是为大规模 all-reduce 设计的主通道。

all-reduce 是多卡把各自梯度做求和再广播回来，可以理解为“先汇总，再让每张卡都拿到同一份结果”。

再看真实工程例子。  
假设一台 HGX H200 8-GPU 服务器，GPU 之间通过 NVLink 4 和 NVSwitch 形成全互联。全互联的意思是任意两张卡之间都能通过交换网络高带宽互通，而不是抢同一根普通总线。这样做的结果是，梯度同步尽可能在 GPU 域内完成，不必频繁绕回 CPU 和 PCIe 根复合体。

NVSwitch 可以理解为 GPU 专用交换芯片，它把“点对点连线”升级成“交换式全互联”。  
这对 TP 尤其重要，因为张量并行会在层内频繁交换切片结果。HBM 带宽决定本卡算得多快，NVLink/NVSwitch 决定多卡协同会不会被拖死。

因此，A100、H100、H200 的差异，不是孤立的参数表差异，而是下面这条链路的整体升级：

$$
\text{HBM 带宽} \rightarrow \text{单卡算子吞吐}
$$

$$
\text{NVLink/NVSwitch} \rightarrow \text{多卡同步效率}
$$

$$
\text{PCIe/NVMe/CPU 通路} \rightarrow \text{数据供给速度}
$$

三者任意一个掉队，TFLOPS 再高也会空转。

---

## 代码实现

下面给一个可运行的 Python 玩具估算器。它不依赖框架，只做三件事：

1. 估算训练总显存需求。
2. 根据单卡显存估算最少卡数。
3. 粗略比较 PCIe、NVLink 下的通信时间量级。

```python
import math

GPU_SPECS = {
    "A100-80GB": {
        "mem_gb": 80,
        "hbm_tbps": 2.0,
        "nvlink_gbps": 600,
        "fp8": False,
    },
    "H100-80GB": {
        "mem_gb": 80,
        "hbm_tbps": 3.35,
        "nvlink_gbps": 900,
        "fp8": True,
    },
    "H200-141GB": {
        "mem_gb": 141,
        "hbm_tbps": 4.8,
        "nvlink_gbps": 900,
        "fp8": True,
    },
}

def estimate_training_memory_gb(params_billion, bp=2, bg=2, bo=4, headroom=1.25, extra_gb=128):
    params = params_billion * 1e9
    base_bytes = params * (bp + bg + bo)
    total_bytes = base_bytes * headroom + extra_gb * 1e9
    return total_bytes / 1e9

def needed_gpus(total_mem_gb, per_gpu_mem_gb):
    return math.ceil(total_mem_gb / per_gpu_mem_gb)

def estimate_comm_seconds(params_billion, bytes_per_param=2, interconnect_gbps=900, num_gpus=8):
    params = params_billion * 1e9
    total_bytes = 2 * params * bytes_per_param
    rounds = math.log2(num_gpus)
    bandwidth_bytes_per_sec = interconnect_gbps * 1e9
    return total_bytes / bandwidth_bytes_per_sec * rounds

# 玩具例子：8B 模型
toy_mem = estimate_training_memory_gb(8, extra_gb=32)
toy_h100 = needed_gpus(toy_mem, GPU_SPECS["H100-80GB"]["mem_gb"])

# 真实工程例子：70B 模型
real_mem = estimate_training_memory_gb(70, extra_gb=128)
real_h100 = needed_gpus(real_mem, GPU_SPECS["H100-80GB"]["mem_gb"])
real_h200 = needed_gpus(real_mem, GPU_SPECS["H200-141GB"]["mem_gb"])

pcie_gen4_time = estimate_comm_seconds(70, interconnect_gbps=64, num_gpus=8)
nvlink4_time = estimate_comm_seconds(70, interconnect_gbps=900, num_gpus=8)

assert toy_mem > 0
assert real_h200 <= real_h100
assert nvlink4_time < pcie_gen4_time

print(f"8B 模型训练显存估算: {toy_mem:.1f} GB, H100 约需 {toy_h100} 张")
print(f"70B 模型训练显存估算: {real_mem:.1f} GB")
print(f"H100 80GB 约需 {real_h100} 张")
print(f"H200 141GB 约需 {real_h200} 张")
print(f"70B 在 PCIe Gen4 上的粗略通信时间: {pcie_gen4_time:.2f} s")
print(f"70B 在 NVLink 4 上的粗略通信时间: {nvlink4_time:.2f} s")
```

这段代码有两个故意保留的“工程现实”：

第一，它把训练显存估算写成参数化函数，而不是硬编码一个神秘常数。这样你可以把 `bo` 改小，模拟更轻量的优化器；也可以把 `headroom` 调高，模拟更保守的估算。  
第二，它把通信估算和显存估算拆开。因为很多人会误以为“显存装得下就能高效训练”，实际经常是显存刚够，但同步太慢，整机吞吐仍然很差。

你可以直接代入两个场景。

玩具例子：8B BF16 模型。  
这类规模在 H100 上通常可以较轻松地训练或微调，重点不是显存够不够，而是数据加载和 batch 设计是否合理。

真实工程例子：70B BF16 模型。  
如果按上面的估算，H100 80GB 需要更多卡数才能容纳完整训练状态，意味着更多 TP/PP 或更重的 ZeRO/FSDP；H200 141GB 则可能把问题从“必须切很多刀”降到“少切几刀就够”，工程复杂度会明显下降。

---

## 工程权衡与常见坑

硬件规格表很容易把人带偏。真正上线时，最常见的坑不是“算力不够”，而是“链路不平衡”。

| 坑点 | 现象 | 根因 | 规避方式 |
|---|---|---|---|
| 只看 TFLOPS | 理论很强，实测吞吐一般 | HBM、互联、IO 任一项落后 | 同时核对显存、HBM、NVLink、PCIe |
| 只算参数不算优化器 | 明明 140GB 参数，却在 8 卡仍 OOM | 忽略梯度和 Adam 状态 | 用完整训练状态估算 |
| 只看单卡显存 | 单卡能放下，训练仍慢 | all-reduce 被互联拖住 | 优先 NVLink/NVSwitch 拓扑 |
| 忽略数据加载 | GPU 利用率忽高忽低 | NVMe 到 GPU 通路过窄 | 更快存储、预取、GPUDirect |
| 盲目追 FP8 | 吞吐上来了，损失不稳定 | 标定、缩放、算子支持不足 | 先确认框架和模型已验证 |
| 误把 PCIe 当卡间主通道 | 多卡越多越慢 | 总线共享严重 | 单机内优先 NVLink 域内通信 |

PCIe Gen4 x16 的理论双向带宽量级远低于 NVLink 域内带宽。更关键的是，PCIe 是共享主机通道，不是“每张卡独占一根永不拥塞的高速专线”。这意味着当你同时做数据读取、checkpoint 保存、卡间同步时，总线竞争会迅速出现。

真实工程里，常见数据路径是：

`NVMe -> CPU 内存 -> PCIe -> GPU HBM`

如果再叠加网络读取、数据解压、随机打乱，GPU 很可能并不是在等计算，而是在等下一批数据。  
这也是为什么很多训练集群在监控上表现为“GPU 利用率只有 40% 到 60%”，但显存已经打满。显存打满不代表训练高效，它只代表你把东西塞进去了。

另一个常见误解是：有 NVLink 就等于没有通信问题。不是。  
NVLink 解决的是 GPU 间带宽问题，不解决不合理并行切分带来的同步频率问题。如果你把一个本不该做细粒度 TP 的模型拆得过细，那么即便是 NVLink 4，也可能被高频同步拖慢。

对初学者，一个很实用的经验是：

1. 先判断能否减少切分，而不是先考虑怎么把切分做复杂。
2. 能用更大显存的卡减少通信，通常比用更多小显存卡硬拼更稳。
3. 当模型规模跨过某个门槛后，H200 的价值主要不在“更快”，而在“更少的工程痛苦”。

---

## 替代方案与适用边界

没有一种 GPU 适合所有团队。选型要跟模型规模、预算、团队工程能力一起看。

| 场景 | 推荐 GPU/互联 | 原因 | 边界 |
|---|---|---|---|
| 7B 到 13B 微调 | A100 或 H100，PCIe/NVLink 均可 | 显存压力有限，流程成熟 | 更看重成本和可获得性 |
| 8B 到 32B 训练 | H100 优先，尽量单机 NVLink | FP8/BF16 吞吐更好 | 需确认框架对 FP8 支持 |
| 30B 到 70B 训练 | H100 多卡或 H200 少卡 | 一个偏性价比，一个偏简化并行 | H100 需要更强并行调度 |
| 70B 以上预训练 | H200 + NVSwitch/HGX | 显存更宽裕，减少跨卡压力 | 成本高，依赖完整机柜方案 |
| 预算受限老集群 | A100 继续用 | 软件生态稳定 | 需要接受更多切分与更低吞吐 |

如果你的团队只是训练 8B 级别模型，优先考虑 H100。原因不是 H200 不好，而是 H200 多出来的显存并没有完全转化为收益，反而可能提高采购门槛。  
如果你要做 70B 以上，并且不希望把大量时间花在 TP、PP、激活重计算、内存碎片和通信调优上，H200 的意义就会非常明确。

还有两类替代方案也要知道，但本文不展开：

第一类是软件层替代，比如 ZeRO、FSDP、激活重计算。它们能把原本装不下的训练状态拆散、重算或分片，但代价是更复杂的通信路径和更强的调试负担。  
第二类是任务层替代，比如 LoRA、QLoRA、参数高效微调。它们并不要求你全量维护优化器和全量梯度，因此硬件门槛会骤降。

所以最终判断标准很简单：

- 如果你追求最低硬件成本，可以接受更复杂的软件并行，A100/H100 仍然有空间。
- 如果你追求更低的工程复杂度和更大的单机可承载规模，H200 更合适。
- 如果你已经进入多机多卡，NVSwitch、NVLink 域和主机 IO 通路必须一起规划，单看显卡型号已经不够。

---

## 参考资料

- E2E Networks, “NVIDIA A100 vs H100 vs H200 GPU Comparison”  
- Best GPUs for AI, “NVIDIA A100 vs H100: INT4, FP8, BF16 & AI Performance Comparison”  
- NVIDIA, “Hopper Architecture In-Depth”  
- NVIDIA, “HGX Platform Specifications”  
- gpu.fm, “How to Size a GPU Cluster for LLM Training”  
- KAD8, “PCIe Gen5 Goes Mainstream in 2026”  
- Medium, “Network Bandwidth — The Hidden Bottleneck in AI Infrastructure”  
- Medium, “GPU Memory Hierarchy — How AI Training Actually Works”  
- Medium, “The Hidden Bottleneck: Why Your GPU is Waiting”
