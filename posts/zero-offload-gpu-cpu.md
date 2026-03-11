## 核心结论

ZeRO-Offload 解决的不是“让 CPU 替 GPU 训练模型”，而是把**长期占用显存、但不必一直留在 GPU 上的训练状态**迁到 CPU 或 NVMe。最典型的是优化器状态，例如 Adam 在更新参数时维护的一阶动量 `m` 和二阶方差 `v`，以及很多实现里保留的 fp32 主参数副本。GPU 仍然负责前向传播、反向传播和大部分高吞吐张量计算，CPU 或 NVMe 负责保存并更新这些长期状态。

它针对的是一个很具体的瓶颈：**在单卡或少卡训练里，模型往往不是“算不动”，而是先“放不下”。** 尤其在使用 Adam、AdamW 这类优化器时，即使模型权重以 fp16 或 bf16 存储，训练过程中通常仍需要额外保存 fp32 主参数、梯度以及优化器状态。结果是参数本体还没到极限，显存已经被多份状态副本吃满。

ZeRO-Offload 可以概括为一句话：

$$
\text{把长期存储放到 CPU/NVMe，把高吞吐计算留给 GPU}
$$

从结果看，它通常带来三类变化：

| 配置 | GPU 上长期保留的内容 | 显存压力 | 吞吐 | 适用场景 |
|---|---|---:|---:|---|
| 普通数据并行 | 参数、梯度、优化器状态、激活 | 最高 | 基准 | 模型较小，显存充足 |
| ZeRO-2 | 参数副本、激活；梯度和优化器状态做分片 | 中等 | 较高 | 多卡训练，显存仍紧 |
| ZeRO-Offload | 计算所需参数、激活；优化器状态和部分梯度迁到 CPU/NVMe | 最低之一 | 低于纯 GPU 最优值，但通常可接受 | 1 到 16 卡、显存受限 |

工程上必须接受一个事实：ZeRO-Offload 不是免费午餐。它是用 PCIe 带宽、CPU 内存带宽、NVMe 异步 IO 和调度复杂度，去换更大的可训练模型规模。因此单步训练时间更接近：

$$
T_{\text{total}} \approx \max(T_{\text{comp}},\ T_{\text{comm}} + T_{\text{cpu\_update}})
$$

其中：

- $T_{\text{comp}}$ 是 GPU 前向与反向计算时间
- $T_{\text{comm}}$ 是 CPU/GPU、GPU/GPU、NVMe/CPU 间的数据搬运时间
- $T_{\text{cpu\_update}}$ 是 CPU 侧优化器更新时间

只有当通信和 CPU 更新能够与 GPU 计算充分重叠时，吞吐才不会明显塌陷。换句话说，ZeRO-Offload 真正的价值不是“更快”，而是**在可接受的速度下，把原本放不下的训练任务做起来。**

---

## 问题定义与边界

先把问题定义清楚。训练一个参数量为 $P$ 的模型时，显存里并不只保存“一份模型参数”。以 Adam 为例，常见的长期或半长期内存组成包括：

1. 模型参数
2. 梯度
3. fp32 主参数
4. 一阶动量
5. 二阶方差
6. 激活

这里的“激活”是前向传播中各层输出的中间结果，反向传播时需要它们来计算梯度。很多初学者第一次接触大模型训练时，会把“激活很占显存”当成唯一瓶颈。但在大参数量模型上，更常见的情况是：**优化器状态先成为瓶颈，激活反而排在后面。**

原因在于，Adam 一类优化器要为每个参数额外维护历史统计量。若参数使用 fp16，每个参数约占 2 字节；而主参数、动量、方差往往使用 fp32，每份约占 4 字节。粗略估算时，一个参数可能对应如下持久状态：

| 项目 | 常见精度 | 每参数字节数 |
|---|---:|---:|
| 参数 | fp16 / bf16 | 2 |
| 梯度 | fp16 / bf16 / fp32 | 2 或 4 |
| fp32 主参数 | fp32 | 4 |
| 一阶动量 `m` | fp32 | 4 |
| 二阶方差 `v` | fp32 | 4 |

如果按最常见的混合精度训练习惯粗略估算，一个参数很容易对应 **14 到 18 字节** 的状态，而不是表面看到的 2 字节。这也是为什么“7B 参数模型的权重看起来只要十几 GB，但训练时远远不止”。

忽略显存碎片、临时缓冲和通信缓存后，数据并行下单卡长期状态显存可近似写成：

$$
M_{\text{gpu}} \approx M_{\text{param}} + M_{\text{grad}} + M_{\text{optim}} + M_{\text{act}}
$$

其中：

- $M_{\text{param}}$：模型参数及其主参数副本
- $M_{\text{grad}}$：梯度
- $M_{\text{optim}}$：优化器状态
- $M_{\text{act}}$：激活和相关反向缓存

ZeRO 的基本思想是“分片”，即**不让每张卡都保存完整状态，而是把状态切开，每张卡只保留其中一部分。** 若数据并行进程数为 $N$，理想化条件下有：

$$
M_{\text{state per gpu}} \approx \frac{M_{\text{param}} + M_{\text{grad}} + M_{\text{optim}}}{N}
$$

ZeRO-Offload 再进一步：不仅分片，还把一部分状态从 GPU 迁到 CPU 或 NVMe。它的能力边界需要说清楚：

| 项目 | ZeRO-Offload 是否主要处理 | 说明 |
|---|---|---|
| 优化器状态 | 是 | 最核心的 offload 对象 |
| 梯度 | 是，视 stage 和配置而定 | 可分片，也可下放到 CPU |
| 参数 | 可以，但不是所有配置都激进 offload 参数 | 经典场景更常见的是 optimizer offload |
| 激活 | 否 | 激活通常靠 activation checkpointing 处理 |
| 前向/后向算子 | 否 | 主要仍在 GPU 执行 |

这一步对新手很重要，因为它避免一个常见误解：**ZeRO-Offload 并不等于“训练都在 CPU 上跑”。** 它只是让 CPU 成为状态仓库和部分更新执行端，GPU 仍然是主要算力来源。

可以用一个玩具场景理解。

假设你有 1 张 32GB GPU，要训练一个很大的 Transformer。模型权重本身如果用 fp16，可能勉强接近能放下；但再加上梯度、fp32 主参数、Adam 的 `m` 和 `v`，显存立刻超限。ZeRO-Offload 的做法不是减少这些状态，而是把它们搬出 GPU：**GPU 像计算工位，CPU 像仓库和账本系统。**

这个类比只用于帮助理解边界，不替代定义。严格地说，它意味着：

- 总内存需求没有消失，只是转移到了 CPU 内存或 NVMe
- GPU 显存下降了，但 CPU 内存、PCIe 带宽和 IO 压力上升了
- 性能瓶颈可能从“显存不够”转成“搬运太慢”

因此它最适合的条件通常是：

- GPU 显存紧张
- GPU 数量不多
- CPU 内存相对充足
- PCIe 和 NVMe 不至于太弱
- 可以接受比纯 GPU 稍低的吞吐

如果这些前提不成立，例如 CPU 内存很小、NVMe 很慢、PCIe 带宽受限，那么 ZeRO-Offload 可能只是把问题从“放不下”变成“跑得极慢”。

---

## 核心机制与推导

要真正理解 ZeRO-Offload，关键不是记配置项，而是看清楚一次训练 step 的数据流。

一个简化的 step 时序如下：

1. GPU 前向传播，读取当前需要的参数分片或已预取参数
2. GPU 反向传播，逐层计算梯度
3. 梯度在数据并行组内执行 `reduce-scatter`
4. 分片后的梯度尽快 offload 到 CPU，或进入 CPU 侧更新流水线
5. CPU 上使用 `DeepSpeedCPUAdam` 之类的优化器，更新 fp32 主参数与动量/方差
6. 更新后的参数按需要转换为 fp16/bf16，回传或预取到 GPU
7. 下一轮训练前，系统继续预取即将使用的参数分片

这里需要先解释两个基础术语。

`all-reduce`：所有卡都把各自梯度求和，最后每张卡都拿到一份完整结果。  
`reduce-scatter`：先求和，再把结果切成片，每张卡只保留其中一片。

对 ZeRO 而言，`reduce-scatter` 更重要，因为它天然和“分片保存状态”一致。每张卡只保留自己负责的那份梯度分片，不再需要完整梯度副本。

若模型一次 step 需要搬运的总数据量为 $M$，有效带宽为 $B$，数据搬运轮数为 $n_{\text{trans}}$，通信时间可先粗略写成：

$$
T_{\text{comm}} \approx \frac{n_{\text{trans}} \cdot M}{B}
$$

如果区分不同路径，例如 GPU-GPU、CPU-GPU、NVMe-CPU，可进一步拆为：

$$
T_{\text{comm}} \approx
\frac{M_{\text{gpu-gpu}}}{B_{\text{gpu-gpu}}}
+
\frac{M_{\text{cpu-gpu}}}{B_{\text{cpu-gpu}}}
+
\frac{M_{\text{nvme-cpu}}}{B_{\text{nvme-cpu}}}
$$

这个拆分更接近真实系统，因为不同链路的带宽差距很大：

| 链路 | 常见带宽级别 | 典型影响 |
|---|---:|---|
| GPU-GPU（NVLink） | 很高 | 多卡通信成本较低 |
| GPU-GPU（PCIe） | 中 | 多卡同步更容易受限 |
| CPU-GPU（PCIe） | 中 | Offload 常见瓶颈之一 |
| NVMe-CPU | 中到较高 | 依赖 SSD 和 AIO 配置 |

计算时间则可以非常粗略地看成与模型规模、序列长度、批大小和总算力有关：

$$
T_{\text{comp}} \propto \frac{M \times \text{seq\_len} \times \text{global\_batch}}{\text{total\_gpu\_throughput}}
$$

这个式子不是精确预测器，但足够说明核心判断：**ZeRO-Offload 划不划算，不取决于“省了多少显存”本身，而取决于额外搬运和 CPU 更新，能不能被 GPU 计算时间掩盖。**

### 参数切分与真实平衡

很多入门文章停留在“显存下降了”这一层，但工程上真正要平衡的是三个维度：

| 维度 | 变大后的收益 | 变大后的代价 |
|---|---|---|
| bucket size | 通信次数减少，固定延迟被摊薄 | 峰值显存上升，尾部等待可能增加 |
| persistence threshold | 小参数常驻 GPU，减少频繁 all-gather | 长驻参数占更多显存 |
| 重计算比例 | 激活显存更低 | 额外前向计算增加 |

其中最容易让新手困惑的是 `stage3_param_persistence_threshold`。它的作用可以白话理解为：

**参数如果小到“搬来搬去不划算”，就让它直接留在 GPU。**

为什么需要这个阈值？因为真实模型里有很多“小层”或“小张量”，它们的总数据量并不大，但如果每次都单独触发传输，系统承担的是大量固定延迟，而不是纯带宽成本。于是会出现一种现象：数据量看起来不大，但 PCIe 调度、内核启动和同步等待很多，step time 仍然很差。

### 玩具例子

假设模型里有 4 组参数，每组 1GB，其中大层只在固定阶段使用，而某些小层在每个 block 中频繁访问。

- 如果全部参数都“现用现取”，通信次数会很多，延迟成本累积
- 如果把频繁访问的小层长期保留在 GPU，PCIe 往返次数会明显下降
- 但代价是多占用一部分显存

这说明一个经常被忽略的事实：**ZeRO-Offload 的瓶颈不只由总传输字节数决定，还由传输颗粒度决定。** 大块传输更容易接近带宽上限，小块频繁传输则更容易被延迟主导。

### 真实工程例子

以 Megatron-LM + DeepSpeed 训练 10B 级 GPT 类模型为例，常见做法通常是：

- 前向和反向计算留在 GPU
- 优化器状态留在 CPU
- 参数分片按 bucket 预取到 GPU
- 用异步 IO 重叠 NVMe 到 CPU、CPU 到 GPU、GPU 计算三条流水线

此时如果只盯着“显存够不够”，会错过真正的性能瓶颈。更常见的瓶颈其实是：

1. CPUAdam 更新是否跟得上 GPU 的 step 节奏
2. 参数预取是否过碎，导致 PCIe 被小包传输拖慢
3. NVMe queue depth 是否足够，能否持续供数
4. 小参数是否因为 persistence threshold 太低而来回抖动

因此，ZeRO-Offload 的工程核心并不是“把东西搬到 CPU”这么简单，而是**把分片、预取、异步 IO 和更新流水线组织成一个可重叠的数据流系统。**

---

## 代码实现

下面先给一个**可直接运行**的玩具脚本，用来模拟“是否值得 offload”。它不是 DeepSpeed 源码实现，也不会精确预测真实训练速度，但能直观展示一个关键条件：**通信和 CPU 更新能否被 GPU 计算重叠。**

```python
from dataclasses import dataclass


@dataclass
class StepEstimate:
    t_comp_ms: float
    t_comm_ms: float
    t_cpu_update_ms: float
    total_ms: float
    throughput_tokens_per_s: float


def estimate_step_time(
    model_gb: float,
    gpu_tflops: float,
    bandwidth_gbps: float,
    transmissions: float,
    cpu_update_ms: float,
    tokens_per_step: int,
    overlap: bool = True,
) -> StepEstimate:
    """
    一个可运行的玩具估算器。

    参数说明：
    - model_gb: 每 step 需要搬运或参与计算的模型状态规模，单位 GB
    - gpu_tflops: GPU 有效算力，单位 TFLOPS
    - bandwidth_gbps: 有效带宽，单位 GB/s。这里故意使用 GB/s，避免和 Gb/s 混淆
    - transmissions: 平均每 step 的等效传输轮数
    - cpu_update_ms: CPU 侧优化器更新时间
    - tokens_per_step: 每 step 处理的 token 数
    - overlap: 是否假设通信和 CPU 更新可与计算重叠
    """
    if model_gb <= 0:
        raise ValueError("model_gb must be > 0")
    if gpu_tflops <= 0:
        raise ValueError("gpu_tflops must be > 0")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be > 0")
    if transmissions < 0:
        raise ValueError("transmissions must be >= 0")
    if cpu_update_ms < 0:
        raise ValueError("cpu_update_ms must be >= 0")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be > 0")

    # 玩具近似：模型规模越大、有效算力越低，则计算时间越长
    t_comp_ms = model_gb / gpu_tflops * 1000.0

    # 通信时间：总传输数据量 / 有效带宽
    t_comm_ms = model_gb * transmissions / bandwidth_gbps * 1000.0

    if overlap:
        total_ms = max(t_comp_ms, t_comm_ms + cpu_update_ms)
    else:
        total_ms = t_comp_ms + t_comm_ms + cpu_update_ms

    throughput_tokens_per_s = tokens_per_step / (total_ms / 1000.0)

    return StepEstimate(
        t_comp_ms=round(t_comp_ms, 2),
        t_comm_ms=round(t_comm_ms, 2),
        t_cpu_update_ms=round(cpu_update_ms, 2),
        total_ms=round(total_ms, 2),
        throughput_tokens_per_s=round(throughput_tokens_per_s, 2),
    )


def main() -> None:
    offload_overlap = estimate_step_time(
        model_gb=8.0,
        gpu_tflops=40.0,
        bandwidth_gbps=24.0,
        transmissions=1.5,
        cpu_update_ms=80.0,
        tokens_per_step=4096,
        overlap=True,
    )
    offload_no_overlap = estimate_step_time(
        model_gb=8.0,
        gpu_tflops=40.0,
        bandwidth_gbps=24.0,
        transmissions=1.5,
        cpu_update_ms=80.0,
        tokens_per_step=4096,
        overlap=False,
    )

    assert offload_overlap.total_ms <= offload_no_overlap.total_ms
    assert offload_overlap.t_comm_ms > 0
    assert offload_overlap.t_comp_ms > 0

    print("overlap=True :", offload_overlap)
    print("overlap=False:", offload_no_overlap)


if __name__ == "__main__":
    main()
```

这个脚本有三个新手最容易忽略的点：

1. `bandwidth_gbps` 这里按 **GB/s** 解释，不是网络里常见的 **Gb/s**
2. `overlap=True` 时，单步总时间取 `max(...)`，因为计算和搬运被视为并行流水
3. `overlap=False` 时，所有开销直接串行相加，吞吐会明显下降

如果你把 `cpu_update_ms` 或 `transmissions` 调大，会看到一个很直观的结果：当

$$
T_{\text{comm}} + T_{\text{cpu\_update}} > T_{\text{comp}}
$$

时，offload 额外链路会成为主导瓶颈。反过来，如果 GPU 计算本身很重，而通信和 CPU 更新能隐藏在里面，那么 offload 的速度损失就可能比较有限。

下面给出一个**能直接作为 JSON 使用**的最小 DeepSpeed 配置示例。这里仍然不是让你死记字段，而是帮助你把字段和硬件路径对应起来。

```python
deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": False,
        "overlap_events": True
    },
    "gradient_clipping": 1.0
}
```

如果需要把参数或优化器状态进一步下放到 NVMe，常见做法是把：

- `offload_optimizer.device` 设为 `"nvme"`
- 或 `offload_param.device` 设为 `"nvme"`

同时补充 NVMe 路径、buffer 数量和 AIO 相关配置。之所以不能只改一个 `device` 字段，是因为 NVMe 和 CPU 内存不同，它需要额外的异步缓冲和队列管理，否则很难真正跑满磁盘吞吐。

常见配置项可以先抓下面这些：

| 配置项 | 作用 | 调参方向 |
|---|---|---|
| `stage` | ZeRO 分片级别 | Offload 常与 Stage 3 搭配 |
| `offload_optimizer.device` | 优化器状态放置位置 | `cpu` 更常见，`nvme` 更省主存 |
| `offload_param.device` | 参数放置位置 | 只有显存极紧时才常激进启用 |
| `reduce_bucket_size` | 梯度归并桶大小 | 太小通信频繁，太大峰值显存高 |
| `stage3_prefetch_bucket_size` | 参数预取粒度 | 太小传输碎，太大占显存 |
| `stage3_param_persistence_threshold` | 小参数常驻阈值 | 太低会抖动，太高会涨显存 |
| `aio.queue_depth` | NVMe 异步 IO 并发深度 | 太低供数不足，太高可能引入额外竞争 |
| `pin_memory` | 是否使用页锁定内存 | 有利于 CPU-GPU 拷贝效率 |

如果是 Megatron-LM 这类真实工程，启动脚本里往往还会同时设置：

- 张量并行
- 流水并行
- 激活检查点
- ZeRO stage
- optimizer offload / param offload

这时最稳妥的顺序不是“一上来把 offload 开到最猛”，而是：

1. 先确认模型能稳定收敛
2. 再确认 step time 没有严重抖动
3. 最后再加大 offload、batch size 或更激进的 NVMe 参数

否则你很难判断问题到底来自优化器、并行策略还是 IO 链路。

---

## 工程权衡与常见坑

ZeRO-Offload 最容易被误解的一点是：**显存节省不等于训练一定更快。** 很多时候，它的真正价值只是让“原本根本跑不起来”的模型变成“能跑，而且速度还能接受”。

下面是常见问题与排查方向。

| 症状 | 可能原因 | 监控信号 | 对策 |
|---|---|---|---|
| step time 抖动很大 | bucket 太小，频繁 all-gather / reduce-scatter | GPU 利用率忽高忽低，PCIe 带宽峰值碎片化 | 增大 bucket，调高 persistence threshold |
| loss 突跳 | CPUAdam 堵塞，更新链路被拖慢 | loss 曲线突然上扬，step time 同时恶化 | 降低 offload 压力，检查 CPU 核数与内存带宽 |
| grad norm 爆炸 | 更新延迟叠加大 batch 或学习率过高 | `grad_norm` 快速增大 | 降学习率，开梯度裁剪，缩小有效 batch |
| GPU 空闲明显 | 异步 IO 没有真正重叠 | GPU util 低，CPU/NVMe util 也不高 | 检查 `overlap_comm`、`pin_memory`、AIO 参数 |
| 吞吐低于预期很多 | NVMe 太慢或 `queue_depth` 太小 | 磁盘 util 高但带宽没跑满 | 调整 `aio.queue_depth`、buffer 数量、块大小 |
| 内存占用异常高 | CPU offload 后主存压力过大 | CPU 内存持续增长，swap 抖动 | 降低 offload 激进程度，确认没有 swap |
| 小层反复抖动 | persistence threshold 太低 | step time 呈周期性尖峰 | 提高 `stage3_param_persistence_threshold` |

一个很实用的判断式是：

$$
\text{若 } T_{\text{comm}} + T_{\text{cpu\_update}} \gg T_{\text{comp}}, \text{则 offload 会明显拖慢训练}
$$

这条式子可以直接指导监控。不要只看 `loss`，至少要同时看三组信号：

1. `loss` 是否持续下降
2. `grad norm` 是否存在异常尖峰
3. `step time`、GPU util、CPU util、磁盘 util 是否稳定

很多表面上的“训练发散”其实并不是优化理论问题，而是工程链路问题。例如：

- `stage3_param_persistence_threshold` 太低，小层参数频繁取回
- `offload_optimizer` 开了，但 CPU 内存不足、NUMA 绑定差，导致访存效率低
- NVMe offload 开了，但 `queue_depth` 太保守，GPU 常常在等数据
- batch size 盲目加大，结果通信和更新没有重叠起来，整体节奏被打乱

这里再补一个对新手更友好的排查框架。把问题按“先活下来，再提速”处理：

| 排查阶段 | 先看什么 | 判断标准 |
|---|---|---|
| 第一阶段：能不能稳 | loss、grad norm、是否 OOM | 能稳定训练多个 step，不发散、不爆显存 |
| 第二阶段：有没有空转 | GPU util、step time 抖动 | GPU 不是大段空闲，step time 波动可控 |
| 第三阶段：链路是否顺 | CPU util、内存带宽、磁盘 util | CPU 和磁盘忙得有意义，不是高占用低吞吐 |
| 第四阶段：是否值得继续压榨 | tokens/s、有效 batch、收敛速度 | 吞吐提升真实带来训练收益 |

经验上，排查顺序通常应该是：

1. 先保证 loss 稳定
2. 再看 step time 是否抖动
3. 再调 bucket 和 threshold
4. 最后再追求更大的 batch 和更激进的 offload

这个顺序的原因很简单：如果训练本身不稳定，再好的吞吐都没有意义。

---

## 替代方案与适用边界

ZeRO-Offload 不是唯一方案。它更像是“在低显存预算下，把训练做成”的工程工具，而不是默认最优解。

先看几种常见方案的对比：

| 方案 | 显存节省 | 通信成本 | 工程复杂度 | 更适合 |
|---|---:|---:|---:|---|
| 普通数据并行 | 低 | 低 | 低 | 小模型、快速实验 |
| ZeRO-2 | 中 | 中 | 中 | 多卡且显存仍紧 |
| ZeRO-3 | 高 | 高 | 中高 | 多卡大模型、纯 GPU 资源较强 |
| ZeRO-Offload | 很高 | 高，且依赖 CPU/NVMe | 高 | 少卡、显存紧张 |
| ZeRO-Infinity | 最高之一 | 更高 | 更高 | 超大模型、极端内存压力 |

它们的差别可以再用一句更直白的话概括：

- 普通数据并行：简单，但每张卡都背全套状态
- ZeRO-2：先切梯度和优化器状态
- ZeRO-3：进一步切参数
- ZeRO-Offload：在分片基础上，把状态迁出 GPU
- ZeRO-Infinity：把 offload 做到更激进、更深层的层级化存储

适用边界可以直接说清楚。

如果你有很多张大显存 GPU，优先考虑纯 GPU 的 ZeRO-2 或 ZeRO-3，因为链路更短，吞吐通常更高，系统更简单。

如果你只有 1 到 4 张 24GB 到 40GB 的卡，但 CPU 内存较大，ZeRO-Offload 往往是把 10B 级实验做起来的关键方案。它不一定最快，但很可能是“唯一能在现有机器上启动训练”的方案。

如果 CPU 内存、PCIe 或 NVMe 很弱，offload 很可能把训练变成“能跑但极慢”。此时更现实的办法往往是：

- 缩小模型规模
- 缩短序列长度
- 减小 micro-batch
- 用更轻的优化器
- 配合 activation checkpointing，而不是盲目继续加大 offload

对于初级工程师，最重要的判断不是“哪种方案最先进”，而是：

**你是在解决显存瓶颈，还是在制造链路瓶颈。**

如果主要矛盾是 GPU 放不下，ZeRO-Offload 很有效。  
如果主要矛盾已经变成通信、调度和 IO，继续加大 offload 往往只会退化。

一个实用决策表如下：

| 你当前的主要问题 | 更优先考虑 |
|---|---|
| 单卡显存不够，CPU 内存很多 | ZeRO-Offload |
| 多卡训练显存仍紧，但 GPU 资源充足 | ZeRO-2 / ZeRO-3 |
| 激活太大，不是优化器状态太大 | Activation Checkpointing |
| 磁盘慢、PCIe 弱、CPU 老旧 | 缩模型或换更轻方案 |
| 极端大模型，单机放不下 | ZeRO-Infinity 或更大规模分布式方案 |

这张表的核心目的，是避免一个常见误区：看到“offload 省显存”，就默认应该开启。实际上，offload 是为了解决特定瓶颈，不是通用加速器。

---

## 参考资料

下面的资料按“先建立整体认识，再看配置，再看细节”的顺序排列。

| 类别 | 资料 | 重点内容 | 推荐顺序 |
|---|---|---|---|
| 架构/原理 | Microsoft Research: DeepSpeed Extreme Scale Model Training for Everyone | ZeRO-Offload 的目标、10 倍模型规模、吞吐结论 | 1 |
| 工程/教程 | DeepSpeed 官方 ZeRO-Offload Tutorial | 配置字段、Megatron-LM 样例、CPUAdam 使用方式 | 2 |
| 工程/文档 | DeepSpeed ZeRO-3 Docs | Stage 3、offload、参数预取与持久化参数说明 | 3 |
| 论文 | ZeRO-Offload: Democratizing Billion-Scale Model Training | 原始设计目标、通信与 CPU 协同机制 | 4 |
| 解读 | Paper Cache 对 ZeRO-Offload 的解读 | 数据流、通信与异步重叠的直观解释 | 5 |
| 补充推导 | Distributed Lexicon 相关文章 | 通信/计算时间的估算视角 | 6 |

为了让参考资料更好用，下面补一列“阅读时重点看什么”。

| 资料 | 建议重点关注 |
|---|---|
| Microsoft Research Blog | 为什么说它能把单 GPU 可训练规模提升到更高量级 |
| DeepSpeed ZeRO-Offload Tutorial | `offload_optimizer`、`CPUAdam`、Megatron 配置范式 |
| DeepSpeed ZeRO-3 Documentation | `prefetch bucket`、`persistence threshold`、stage 3 参数流 |
| ZeRO-Offload 论文 | 设计假设、吞吐评估、CPU 与 GPU 的职责划分 |
| Paper Cache 解读 | 如何把论文里的设计翻译成更直观的数据流 |
| Distributed Lexicon | 如何从带宽、传输轮数和重叠角度估算是否值得 offload |

参考链接如下：

- Microsoft Research Blog: https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/
- DeepSpeed ZeRO-Offload Tutorial: https://www.deepspeed.ai/tutorials/zero-offload/
- DeepSpeed ZeRO-3 Documentation: https://deepspeed.readthedocs.io/en/stable/zero3.html
- ZeRO-Offload 论文: https://arxiv.org/abs/2101.06840
- Paper Cache 解读: https://www.papercache.org/papers/llm/engineering/train/2021/01/01/zero-offload-democratizing-billion-scale-model-training
- Distributed Lexicon: https://distributedlexicon.com/

如果只准备读三份，推荐顺序是：

1. DeepSpeed 官方 Tutorial，先建立配置和执行路径的对应关系
2. ZeRO-3 官方文档，理解 stage 3 下参数分片与预取
3. ZeRO-Offload 原论文或 Microsoft Research Blog，补齐设计动机和评估逻辑
