## 核心结论

ZeRO-Infinity 是 ZeRO-3 的继续扩展。ZeRO-3 的核心是把参数、梯度、优化器状态分片到多张 GPU 上，避免每张卡都保留完整副本；ZeRO-Infinity 再把这些分片进一步分布到 GPU、CPU 内存、NVMe SSD 三层存储中，按执行顺序动态搬运。它解决的不是“模型整体必须放进显存”，而是“当前将要执行的那一小段参数，必须能按时进入显存”。

它的价值不在于把 NVMe 变成高带宽设备，而在于把慢设备组织进一条可重叠的流水线。GPU 只保留当前层计算必需的数据，CPU 内存承担中间缓存和转运，NVMe 负责大容量持久存放。通过异步预取、分块搬运、分层缓存和释放回收，I/O 延迟尽量被隐藏在前向与反向计算时间里，因此模型规模可以远超单机显存上限。

对新手最重要的判断标准只有一句话：ZeRO-Infinity 解决的是容量墙，不是算力墙。显存不够时，它是直接有效的工程方案；GPU 已经算满、瓶颈在矩阵乘法或通信时，它不会把训练速度凭空提上去。

可以用一个简化比喻理解。GPU 像灶台，CPU 内存像备菜台，NVMe 像仓库。做第 $i$ 道菜时，系统已经把第 $i+1$ 到 $i+k$ 道菜的原料从仓库搬到备菜台，并在需要时进一步送到灶台边。关键不是仓库离得近不近，而是搬运能否跟上做菜节奏。

一个最常见的带宽近似式是：

$$
\text{有效带宽}_{\text{per-GPU}} \approx \frac{\text{节点总 NVMe 吞吐}}{\text{数据并行度}}
$$

它表示同一节点上的 NVMe 带宽通常会被多张卡共享。若节点总有效 NVMe 吞吐约为 25 GB/s，那么：

- 8 卡时，单卡理论可分到约 $25/8 \approx 3.1$ GB/s
- 16 卡时，单卡理论可分到约 $25/16 \approx 1.6$ GB/s

这只是上界近似，真实值还会受到 PCIe 拓扑、文件系统、I/O 深度、页缓存、进程争用等因素影响。结论不变：卡越多、模型越大、每步需要搬运的数据越多，预取策略和缓冲策略越重要。

---

## 问题定义与边界

问题定义很明确：在 GPU 显存有限的前提下，训练远大于显存容量上限的模型。这里的“大”可以是几十亿参数，也可以是数百亿、万亿参数。ZeRO-Infinity 的目标不是让每一步都更快，而是让“原本无法启动的训练”变成“可以启动且吞吐仍可接受的训练”。

先把三个存储层级分开看：

| 存储层级 | 典型容量 | 典型带宽/访问特征 | 主要作用 |
| --- | ---: | ---: | --- |
| GPU HBM 显存 | 40-80 GB/卡 | 极高，直接供算子访问 | 当前计算必须驻留的数据 |
| CPU 内存 | 256-1024 GB/节点 | 中等，延迟和带宽均优于 NVMe | 预取缓存、回收缓冲、中转层 |
| NVMe SSD | 4-32 TB/节点 | 数 GB/s 级，且常被多卡共享 | 大规模参数与状态的后备仓库 |

这里的“边界”有两个层面。

第一层是容量边界。ZeRO-Infinity 并不是把所有参数永久丢到 NVMe，而是把当前不会立刻使用的数据下沉到更低层级。所谓“冷数据”，就是当前层计算不需要、但稍后还会访问的数据块。

第二层是时间边界。决定它是否可用的，不只是总容量，还包括：

1. CPU 内存是否足够做中间缓存。
2. NVMe 带宽是否足以覆盖预取窗口。
3. 访问顺序是否足够规律，能否提前知道下一批要取什么。
4. 参数切分粒度是否合理，避免一次搬太多或太少。
5. 计算时间是否足够长，能让 I/O 与计算发生重叠。

一个最小数值例子更直观。假设某层执行前需要 8 GB 参数，NVMe 到 GPU 的等效可用带宽是 3 GB/s，那么纯搬运时间约为：

$$
t_{\text{io}} = \frac{8}{3} \approx 2.67 \text{ s}
$$

如果该层计算本身要 3 s，则理论上可以在计算当前层时把下一层参数搬过来，I/O 有机会被完全隐藏；如果该层只算 1 s，那么仍有约 1.67 s 的搬运无法隐藏，GPU 会在下一步等待数据。

因此，ZeRO-Infinity 的适用边界不是“只要有 NVMe 就能训练超大模型”，而是：

$$
t_{\text{available overlap}} \ge t_{\text{required I/O}}
$$

也就是“预留给预取的计算窗口”必须至少覆盖“下一批数据的搬运时间”。

再给一个更接近工程的判断表：

| 条件 | 状态 | 结论 |
| --- | --- | --- |
| 模型大于总显存，但小于 CPU+NVMe 可承载容量 | 是 | 具备启用前提 |
| 层访问顺序稳定，主要是 Transformer 类顺序堆叠结构 | 是 | 预取效果通常较好 |
| 本地 NVMe 独占且带宽稳定 | 是 | 更适合 |
| 只有网络盘或共享盘 | 是 | 很容易失去意义 |
| 计算时间远短于数据搬运时间 | 是 | 吞吐会明显恶化 |

新手可以把它理解成一句话：ZeRO-Infinity 允许材料放在远处，但要求送货必须准时。只解决“放不下”的问题，不保证“跑得飞快”。

---

## 核心机制与推导

ZeRO-Infinity 的核心是 Infinity Offload Engine。它本质上是一个分层数据调度器：知道模型访问顺序，知道当前层和下一层需要哪些参数块，然后在执行第 $i$ 层时，异步准备第 $i+1$ 到第 $i+k$ 层的参数、梯度或优化器状态。

这个过程可以看成三层流水线：

| 时间点 | GPU 正在做什么 | CPU 正在做什么 | NVMe 正在做什么 |
| --- | --- | --- | --- |
| 执行第 $i$ 层 | 计算第 $i$ 层前向或反向 | 接收第 $i+1$ 层参数、回收第 $i-1$ 层缓存 | 读取第 $i+2$、$i+3$ 层参数块 |
| 执行第 $i+1$ 层 | 计算第 $i+1$ 层 | 转发第 $i+2$ 层，整理空闲缓冲区 | 继续顺序预取更后面的块 |
| 执行第 $i+2$ 层 | 继续计算 | 维护缓存池和页锁定内存 | 滚动推进读取窗口 |

这里最重要的不是“有没有 offload”，而是“offload 是否能和计算重叠”。如果不能重叠，系统就退化成“算一会儿，等一会儿，搬一会儿”，吞吐会很差。

从调度角度看，一次训练 step 可以简化为以下状态机：

1. 当前层参数驻留到 GPU。
2. 当前层计算开始。
3. 调度器根据未来访问顺序预取下一批参数。
4. 当前层完成后，释放不用的数据。
5. 下一层进入执行，重复该过程。

为了让这个过程稳定，系统通常会把参数拆成 chunk，再用 bucket 作为更高一级的预取单位。常见的控制量是 `stage3_prefetch_bucket_size`，它表示一次预取窗口允许拉取多少参数元素。

如果参数精度是 FP16，每个元素约 2 字节，则：

$$
\text{bytes}_{\text{bucket}} \approx \text{bucket\_elements} \times 2
$$

例如：

- `5,000,000` 个元素约为 10 MB
- `50,000,000` 个元素约为 100 MB
- `500,000,000` 个元素约为 1 GB

这个值不能盲目调大。过小会导致 GPU 频繁等数据，过大会让 CPU 缓冲区、NVMe 队列、PCIe 通道同时承压，出现 oversubscription，也就是系统申请的数据量超过稳定承载能力。

还可以把总预取耗时近似写成：

$$
t_{\text{prefetch}} \approx t_{\text{nvme}\rightarrow\text{cpu}} + t_{\text{cpu}\rightarrow\text{gpu}} + t_{\text{scheduling}}
$$

而可隐藏时间是：

$$
t_{\text{compute-window}} \approx \sum_{j=i}^{i+k-1} t_{\text{compute},j}
$$

要让 GPU 尽量不空转，需要满足：

$$
t_{\text{prefetch}} \le t_{\text{compute-window}}
$$

这就是 ZeRO-Infinity 的工程目标。它不是消灭 I/O，而是尽可能把 I/O 塞进已有的计算时间缝隙中。

再看一个更细一点的例子。假设未来两层总共需要 12 GB 参数，当前窗口内的有效带宽分解如下：

- NVMe 到 CPU：4.0 GB/s
- CPU 到 GPU：10.0 GB/s
- 调度与队列管理额外损耗：0.2 s

则：

$$
t_{\text{nvme}\rightarrow\text{cpu}} = \frac{12}{4.0} = 3.0\text{ s}
$$

$$
t_{\text{cpu}\rightarrow\text{gpu}} = \frac{12}{10.0} = 1.2\text{ s}
$$

$$
t_{\text{prefetch}} \approx 3.0 + 1.2 + 0.2 = 4.4\text{ s}
$$

如果当前与下一层累计能提供约 5.0 s 的计算窗口，那么预取大概率能被覆盖；如果窗口只有 3.0 s，则至少会暴露约 1.4 s 的等待时间。

对新手而言，这一节只要记住两个量：

1. 未来要搬多少数据。
2. 当前计算给了多少隐藏时间。

ZeRO-Infinity 的所有参数，本质都围绕这两个量做平衡。

---

## 代码实现

工程上，用户通常不需要重写训练循环，主要工作集中在 DeepSpeed 配置。也就是说，模型定义、优化器定义、学习率调度器、数据加载逻辑可以基本保持不变，真正决定是否启用 ZeRO-Infinity 的是 `zero_optimization` 下的一组字段。

下面给出一个更完整、能直接作为模板修改的配置示例：

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "stage3_prefetch_bucket_size": 5000000,
    "stage3_param_persistence_threshold": 100000,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme/deepspeed_param",
      "buffer_count": 6,
      "buffer_size": 134217728,
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

先解释关键字段：

| 字段 | 含义 | 工程作用 |
| --- | --- | --- |
| `stage` | ZeRO 阶段 | 设为 `3` 表示参数、梯度、优化器状态都参与分片 |
| `stage3_prefetch_bucket_size` | 预取 bucket 大小 | 控制一次提前拉取多少参数元素 |
| `stage3_param_persistence_threshold` | 常驻参数阈值 | 小参数可不频繁换入换出，减少调度开销 |
| `offload_param.device` | 参数 offload 目标设备 | 设为 `nvme` 表示参数可落到 NVMe |
| `offload_optimizer.device` | 优化器状态 offload 设备 | 常见设为 `cpu`，减少 NVMe 压力 |
| `nvme_path` | NVMe 数据目录 | 必须可写，且最好位于本地独占 NVMe |
| `buffer_count` | 异步缓冲区数量 | 决定 I/O 深度和流水线并发度 |
| `buffer_size` | 单个缓冲区大小 | 决定单次 I/O 块大小，需关注对齐 |
| `pin_memory` | 锁页内存 | 减少 CPU 到 GPU 复制开销 |

### 一个最小可运行训练脚本

下面的脚本展示了 DeepSpeed 初始化的基本形态。它不是性能基准脚本，但可以直接作为“结构正确”的入门模板。

```python
import argparse
import json
import os

import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    def __init__(self, n_samples=1024, dim=1024):
        self.x = torch.randn(n_samples, dim)
        self.y = torch.randn(n_samples, dim)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ToyModel(nn.Module):
    def __init__(self, dim=1024, depth=8):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ToyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    model = ToyModel()
    criterion = nn.MSELoss()

    # 这里不手动创建 optimizer，由 DeepSpeed 根据配置接管也可以。
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters
    )

    model_engine.train()
    for step, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(model_engine.local_rank)
        y = y.to(model_engine.local_rank)

        pred = model_engine(x)
        loss = criterion(pred, y)

        model_engine.backward(loss)
        model_engine.step()

        if model_engine.global_rank == 0 and step % 10 == 0:
            print(f"step={step}, loss={loss.item():.6f}")

        if step == 20:
            break


if __name__ == "__main__":
    main()
```

如果要启动，命令通常类似：

```bash
deepspeed --num_gpus=8 train.py --deepspeed_config ds_zero_infinity.json
```

### 一个可运行的参数估算脚本

很多新手的第一个问题不是“怎么写训练代码”，而是“这个配置大概能不能跑起来”。下面这个脚本用来估算分层搬运是否有机会被覆盖，它不依赖 DeepSpeed，直接可运行。

```python
from dataclasses import dataclass


@dataclass
class LayerPlan:
    layer_param_gb: float
    nvme_to_cpu_gbps: float
    cpu_to_gpu_gbps: float
    compute_time_s: float
    scheduling_overhead_s: float = 0.0


def transfer_time(data_gb: float, bandwidth_gbps: float) -> float:
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")
    return data_gb / bandwidth_gbps


def total_prefetch_time(plan: LayerPlan) -> float:
    t1 = transfer_time(plan.layer_param_gb, plan.nvme_to_cpu_gbps)
    t2 = transfer_time(plan.layer_param_gb, plan.cpu_to_gpu_gbps)
    return t1 + t2 + plan.scheduling_overhead_s


def can_hide_io(plan: LayerPlan) -> bool:
    return total_prefetch_time(plan) <= plan.compute_time_s


def per_gpu_nvme_bw(node_total_gbps: float, dp_size: int) -> float:
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    return node_total_gbps / dp_size


if __name__ == "__main__":
    # 例 1：单层 8 GB 参数，3 秒计算窗口
    plan = LayerPlan(
        layer_param_gb=8.0,
        nvme_to_cpu_gbps=4.0,
        cpu_to_gpu_gbps=12.0,
        compute_time_s=3.0,
        scheduling_overhead_s=0.1,
    )
    print("prefetch_time =", round(total_prefetch_time(plan), 3))
    print("can_hide_io =", can_hide_io(plan))

    # 例 2：多卡共享 NVMe 带宽
    node_bw = 25.0
    dp = 16
    eff = per_gpu_nvme_bw(node_bw, dp)
    print("per_gpu_nvme_bw =", round(eff, 3), "GB/s")
```

一组示例输出可以这样解读：

- `prefetch_time <= compute_time`：理论上可以隐藏住主要 I/O
- `prefetch_time > compute_time`：GPU 大概率会等数据
- `per_gpu_nvme_bw` 很低：说明数据并行度已经开始挤占单卡可用 I/O 资源

### `buffer_size` 为什么要关心对齐

实际工程里，`buffer_size` 常被忽略，但它经常直接决定系统是否稳定。可以把它理解成“每个搬运桶的容量”。如果文件中参数块大小与缓冲桶容量不匹配，就可能出现这类错误：

```text
buffer nbytes != file bytes
```

这不是语法问题，而是块大小不一致。一个常见经验是让 `buffer_size` 与内部 chunk 大小保持整数倍关系，例如：

$$
\text{buffer\_size} = n \times \text{chunk\_size}, \quad n \in \mathbb{Z}^{+}
$$

这样做的原因很简单：文件里怎么切块，内存里就尽量按同样的粒度接收，减少尾块碎片和额外重组。

### 一个更贴近真实规模的量级估算

假设训练约 500B 参数模型，采用 FP16 存储参数，仅参数本体就约为：

$$
500 \times 10^9 \times 2\text{ bytes} \approx 1\text{ TB}
$$

若再考虑梯度、优化器状态、分片冗余和缓冲开销，完整训练态的数据量会远高于这个值。即使 ZeRO-3 做了分片，单纯依赖 GPU 显存仍可能不够。这时 ZeRO-Infinity 的价值才真正体现出来：它把问题从“容量根本放不下”转成“吞吐是否还能接受”。

---

## 工程权衡与常见坑

ZeRO-Infinity 的本质权衡是用更多 I/O 和更复杂的调度，换取更大的可训练模型规模。它不是无代价方案。只要 I/O 被成功隐藏，代价是可接受的；一旦隐藏失败，GPU 就会频繁等待，step time 明显上升。

下面按“现象-原因-处理”整理常见问题：

| 常见问题 | 典型现象 | 原因 | 处理策略 |
| --- | --- | --- | --- |
| `buffer_size` 不对齐 | 初始化或训练中报错 | 文件块和缓冲块大小不一致 | 改成 chunk 的整数倍 |
| `buffer_count` 太小 | GPU 利用率抖动，I/O 不连续 | 异步队列深度不足 | 逐步增加缓冲区数量 |
| `stage3_prefetch_bucket_size` 太大 | NVMe 打满、CPU 缓冲吃紧、step 反而变慢 | 同时预取数据过多 | 从默认值附近小步调参 |
| `nvme_path` 不可写 | 初始化失败或运行中断 | 目录权限或磁盘空间有问题 | 提前检查路径和剩余容量 |
| CPU 内存不足 | 主机侧 OOM | 中间缓存和 pinned memory 太大 | 降 bucket、降 buffer、增主机内存 |
| NVMe 性能波动 | step time 不稳定 | SSD 热降频、共享盘争用、文件系统抖动 | 使用本地独占 NVMe，避免混部 |
| 数据并行度过大 | 单卡有效带宽快速下降 | 多卡共享同一存储链路 | 减小 DP 或增加节点本地吞吐 |
| 小层过多 | 调度开销比例变大 | 每层很小但切换频繁 | 通过参数常驻阈值减少反复换入换出 |

### 新手最容易踩的三个误区

第一，把 `stage3_prefetch_bucket_size` 理解成“越大越安全”。这不成立。它更像高速公路上的同时发车数量。车太少，供货不足；车太多，匝道拥堵，整体更慢。

第二，只看 NVMe 容量，不看持续吞吐。训练不是冷数据归档，核心不是“装得下”，而是“训练每一步都能按时读出来”。一个 4 TB 但持续吞吐抖动很大的盘，可能还不如一个 2 TB 但稳定的本地盘。

第三，把 GPU 利用率下降全部归因于算力不足。启用 ZeRO-Infinity 后，GPU 利用率下降往往是 I/O、PCIe、CPU 内存、页锁定内存、缓冲区设置共同导致的。应该同时看 step time、磁盘读吞吐、CPU 内存占用、主机页缓存压力，而不是只看显卡监控。

### 一个实际调参顺序

如果你是第一次部署，调参顺序可以按下面来：

1. 先保证 `nvme_path` 可写且磁盘空间充足。
2. 保持默认 bucket 配置，先确认训练能稳定启动。
3. 观察 step time、GPU 利用率、磁盘吞吐是否有明显等待。
4. 若 GPU 经常空转，先增加 `buffer_count`，再小步调整 `stage3_prefetch_bucket_size`。
5. 若主机内存很紧张，先减 bucket 和 buffer，再评估是否需要减少并发深度。

### 如何判断问题主要出在哪一层

| 观察到的现象 | 更可能的瓶颈层 |
| --- | --- |
| 磁盘读吞吐长期接近上限，GPU 利用率周期性掉到底 | NVMe |
| 主机内存占用持续上升并最终 OOM | CPU 内存缓冲 |
| GPU 利用率不低，但 step time 仍很长 | 计算本身或通信 |
| 只有大 DP 时明显变差 | 共享带宽不足 |
| 小模型正常，大模型崩溃 | 容量与缓冲规划不足 |

可以把它理解成一条链路：NVMe -> CPU -> GPU。链路上任意一段不稳，都会让整个流水线失速。

---

## 替代方案与适用边界

ZeRO-Infinity 不是唯一的显存优化方案，但它解决的是一个很具体的问题：当模型规模已经明显超出 GPU 总显存，而你手里仍有较大的 CPU 内存和本地 NVMe 时，如何在不大改训练代码的前提下把训练跑起来。

先把几种常见方案放在一起：

| 方案 | 参数规模上限 | 对 I/O 依赖 | 显存需求 | 主要代价 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| 纯 GPU ZeRO-3 | 中到大 | 低 | 较高 | GPU 内存仍是硬约束 | 模型还能基本装进多卡总显存 |
| ZeRO-Infinity | 很大到万亿级 | 高 | 最低 | I/O 和调度复杂度高 | 显存明显不足，但有 CPU 内存和 NVMe |
| 激活检查点 | 中到大 | 低 | 中等 | 反向重算，计算量增加 | 算力有余、显存不足一部分 |
| FSDP + CPU offload | 大 | 中 | 低到中 | 配置与行为依实现而异 | PyTorch 原生栈为主的工程 |
| 张量并行/流水并行 | 很大 | 低到中 | 取决于切分方式 | 通信复杂、代码侵入更大 | 多机多卡大规模训练 |

这里最容易混淆的是激活检查点。它节省的是中间激活显存，不直接解决“参数总量太大”的问题。ZeRO-Infinity 则主要处理参数、优化器状态及其存储迁移问题。二者不是互斥关系，常常会一起使用。

适用边界可以概括为：

1. 如果模型仍能放进多卡总显存，优先考虑纯 GPU ZeRO-3。结构更简单，吞吐通常更稳定。
2. 如果模型明确放不下，但节点有大 CPU 内存和稳定本地 NVMe，ZeRO-Infinity 是直接有效的工程方案。
3. 如果没有本地 NVMe，只有共享网络盘或对象存储，通常不建议直接上 ZeRO-Infinity，因为数据供给速度很难达标。
4. 如果训练瓶颈主要在算力、通信或算子效率，而不是容量，则应优先考虑混合精度、FlashAttention、融合算子、并行策略优化。
5. 如果模型特别适合模型并行切分，而且集群通信条件较好，那么张量并行或流水并行可能比重 I/O offload 更合适。

再给一个简明判断表：

| 你的约束 | 更优先考虑 |
| --- | --- |
| 显存不够，但本地 SSD 很强 | ZeRO-Infinity |
| 显存差一点点不够，算力还很充裕 | 激活检查点 |
| 显存够，吞吐最重要 | 纯 GPU ZeRO-3 |
| 多机高速互联很好，愿意做并行切分 | 张量并行/流水并行 |
| 软件栈以 PyTorch 原生为主 | FSDP 或 FSDP + CPU offload |

新手只需要记住：ZeRO-Infinity 不是所有场景的默认优选，它是“容量压力极大，但本地存储条件较好”时最有价值。

---

## 参考资料

下面这组资料按“机制 -> 配置 -> 论文 -> 实战问题”的顺序阅读最有效。

| 资料名称 | 链接 | 主要内容 |
| --- | --- | --- |
| DeepSpeed ZeRO 文档 | [DeepSpeed Docs / ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html) | ZeRO-3、offload、配置项说明，是查字段含义的第一入口 |
| DeepSpeed 配置 JSON 文档 | [DeepSpeed Docs / Configuration JSON](https://www.deepspeed.ai/docs/config-json/) | 汇总配置格式，便于核对 `zero_optimization` 字段 |
| Microsoft Research 博客 | [ZeRO-Infinity: Breaking the GPU Memory Wall](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/) | 用工程视角解释分层内存、预取和 overlap 的设计动机 |
| 论文原文 | [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) | 多级存储设计、带宽模型、实验结果和规模上限 |
| DeepSpeed GitHub 仓库 | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | 示例配置、源码实现、issue 排查入口 |
| DeepSpeed Issues | [microsoft/DeepSpeed Issues](https://github.com/microsoft/DeepSpeed/issues) | `nvme_path`、buffer 对齐、权限和稳定性等真实问题 |
| ZeRO 论文 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 理解 ZeRO-1/2/3 的起点，便于区分 ZeRO-Infinity 解决的新增问题 |

如果只读三份，建议顺序如下：

1. 官方文档。先明确哪些字段真正控制 offload、哪些字段只是辅助优化。
2. Microsoft 博客。理解为什么 CPU 和 NVMe 不是简单“落盘”，而是被组织成流水线。
3. 论文。带着问题去看带宽模型、分层存储设计和实验数据，更容易读懂。

如果你准备实际部署，再补读 GitHub issues。原因很简单：论文告诉你原理，文档告诉你配置，issue 才会暴露真实环境里最容易踩的坑。
