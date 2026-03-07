## 核心结论

ZeRO 的核心作用，是减少数据并行训练中的显存冗余。传统数据并行里，每张 GPU 都保存一整份模型参数、梯度和优化器状态；ZeRO 不改变模型结构，也不改变前向与反向传播公式，只改变这些训练状态在多卡之间的存放方式。

在常见的混合精度训练里，一个参数通常对应约 16 bytes 的常驻状态：

| 组成 | 精度 | 单参数字节数 |
|---|---:|---:|
| 模型参数 | FP16 | 2 |
| 梯度 | FP16 | 2 |
| master weights | FP32 | 4 |
| Adam 一阶矩 | FP32 | 4 |
| Adam 二阶矩 | FP32 | 4 |
| 合计 | - | 16 |

因此，参数量为 $P$ 的模型，传统数据并行下，仅模型状态的显存开销就近似为：

$$
M_{\text{DP}} = 16P
$$

这里的 $P$ 是参数个数，不是层数，也不是字节数。比如 7B 模型，就是 $P = 7 \times 10^9$。

ZeRO 分三阶段逐步消除冗余：

1. ZeRO-1：只分片优化器状态。
2. ZeRO-2：再分片梯度。
3. ZeRO-3：再分片参数。

因此，ZeRO-3 的理想模型状态显存近似为：

$$
\text{显存/卡}=\frac{16\times \text{参数量}}{N_{\text{卡}}}
$$

这个式子的含义不是“训练总显存一定缩小到原来的 $1/N$”，而是“与参数相关的那部分常驻状态，在理想情况下可以按卡数均摊”。真实训练里，还会有激活、通信缓冲区、CUDA runtime、缓存碎片等额外显存。

先看一个玩具例子。假设模型只有 8 个参数，使用 4 张卡训练：

- 传统 DP：每张卡都保存 8 个参数对应的全部状态。
- ZeRO-3：每张卡只常驻其中 2 个参数的状态。

可以把它理解成：原来每个人都保存整本手册，现在把手册拆成 4 份，每个人只长期保管自己负责的那一份；需要某一页时，再临时把相关页拼起来。

再看工程量级的估算。对于 7B 模型，混合精度下模型状态约为：

$$
7\text{B} \times 16\text{B} = 112\text{GB}
$$

如果启用 4 卡 ZeRO-3，则每卡理论上只需承担约：

$$
\frac{112}{4} = 28\text{GB}
$$

这就是 ZeRO 的价值来源。但代价也很明确：前向和反向过程中，需要持续执行 AllGather 与 ReduceScatter，把当前计算所需的完整参数临时拼起来，再把梯度重新汇总并切回分片。ZeRO 省的是显存，付出的主要成本是通信。

---

## 问题定义与边界

ZeRO 解决的问题很具体：传统数据并行容易扩展，但显存利用率极低。随着模型变大，训练失败往往不是因为算力不够，而是因为每张卡都冗余保存了完整状态，显存先耗尽。

先把边界说清楚。ZeRO 解决的是“模型状态如何分布”，不是“模型怎么算”。它不改变以下事项：

| 项目 | 传统 DP | ZeRO |
|---|---|---|
| 模型结构 | 不变 | 不变 |
| 前向计算公式 | 不变 | 不变 |
| 反向传播公式 | 不变 | 不变 |
| 数据并行粒度 | 每卡一个进程 | 仍然每卡一个进程 |
| 状态存储方式 | 每卡完整副本 | 按阶段分片 |

这意味着，ZeRO 不是模型并行。

这里需要区分两个常见概念：

| 概念 | 解决什么问题 | 拆分对象 |
|---|---|---|
| 数据并行（DP） | 提高吞吐 | 不同 batch |
| 模型并行（MP） | 单卡放不下计算本体 | 网络层、张量计算 |
| ZeRO | 减少训练状态冗余 | 参数、梯度、优化器状态 |

因此，ZeRO 仍属于数据并行体系。每张卡依然会对不同样本执行完整的前向和反向逻辑，只是不会长期保留完整的训练状态副本。

用统一视角看，差别如下：

| 方案 | 参数 | 梯度 | 优化器状态 | 冗余程度 |
|---|---|---|---|---|
| 传统 DP | 每卡完整一份 | 每卡完整一份 | 每卡完整一份 | 最高 |
| ZeRO-1 | 完整一份 | 完整一份 | 分片 | 降低一部分 |
| ZeRO-2 | 完整一份 | 分片 | 分片 | 进一步降低 |
| ZeRO-3 | 分片 | 分片 | 分片 | 最低 |

这里“分片”的意思，是把一个大张量按参数维度切开，不同 GPU 分别保存不同切片。比如一个长度为 8 的参数向量，在 4 张卡上可以切成 4 段，每卡只保存其中 2 个元素。

对新手来说，可以这样理解：

- 参数：模型真正参与计算的权重。
- 梯度：损失函数对参数的导数，表示“参数应该往哪个方向更新”。
- 优化器状态：优化器为了更新更稳定、更快而额外维护的历史信息。以 Adam 为例，就是一阶矩和二阶矩。

ZeRO 的适用边界也需要提前说明。它最适合的场景通常满足三点：

1. 模型参数很多，显存是主要瓶颈。
2. 数据并行仍是主要训练方式。
3. GPU 间带宽能承受额外通信。

反过来说，如果模型本身不大、单卡就能轻松放下，或者多卡互联很差，ZeRO-3 未必是最优选择。因为它省下来的显存，可能会被通信延迟抵消掉吞吐收益。

---

## 核心机制与推导

先从“为什么是 16 bytes/参数”开始。

在混合精度训练里，前向和反向常用 FP16 或 BF16，以提高吞吐并减少显存；但优化器更新往往保留 FP32 精度，以保证数值稳定。以 Adam 为例，一个参数通常对应：

- FP16 参数：2 bytes
- FP16 梯度：2 bytes
- FP32 master weights：4 bytes
- Adam 一阶矩：4 bytes
- Adam 二阶矩：4 bytes

总计：

$$
2 + 2 + 4 + 4 + 4 = 16 \text{ bytes}
$$

如果训练器件、优化器类型或精度策略不同，这个数字会变化。比如：

| 配置 | 每参数常驻状态 | 说明 |
|---|---:|---|
| FP16 + Adam | 16 bytes | 最常见教学口径 |
| BF16 + Adam | 常近似 16 bytes | 实现细节可能略有差异 |
| FP16 + SGD | 低于 16 bytes | SGD 状态少，没有 Adam 两个矩 |
| 仅推理 | 明显更低 | 没有梯度和优化器状态 |

本文以下推导，统一沿用“混合精度 + Adam + 16 bytes/参数”的标准教学模型。

### ZeRO-1：分片优化器状态

ZeRO-1 只处理优化器状态。

为什么第一步先切优化器状态？因为 Adam 的状态最重。仅一阶矩和二阶矩就有：

$$
4 + 4 = 8 \text{ bytes/param}
$$

这部分在传统 DP 中每张卡都完整保存，冗余非常明显。ZeRO-1 将它们按卡数 $N$ 分片后，每张卡只保留自己的那一份，于是优化器状态显存变成：

$$
\text{优化器状态显存}=\frac{8P}{N}
$$

而参数、梯度和 master weights 仍是完整副本，所以每卡模型状态显存近似为：

$$
M_{\text{Z1}} = 2P + 2P + 4P + \frac{8P}{N}
$$

把它整理一下，可以写成：

$$
M_{\text{Z1}} = 8P + \frac{8P}{N}
$$

对比传统 DP 的 $16P$，节省比例为：

$$
1 - \frac{8 + 8/N}{16}
= \frac{1}{2} - \frac{1}{2N}
$$

举例来说，4 卡时：

$$
M_{\text{Z1}} = 8P + 2P = 10P
$$

也就是从传统 DP 的 $16P$ 降到 $10P$。对 7B 模型来说，模型状态可从约 112GB 降到约 70GB，每卡约 70GB，而不是 112GB。

这里要补一句工程细节。很多实现中，FP32 master weights 也会被视为优化链路中的状态，并参与更完整的分片策略。为了保持教学上的清晰性，本文先把 ZeRO-1 的核心收益理解为“优先消除 Adam 状态冗余”。

### ZeRO-2：再分片梯度

ZeRO-2 在 ZeRO-1 的基础上，再分片梯度。

梯度是反向传播得到的导数。如果还是让每张卡保留完整梯度，那么即便优化器状态已经切开，显存中仍有一大块冗余。ZeRO-2 的做法，是让每张卡只保留自己负责参数分片对应的梯度片段。

因此，梯度显存从：

$$
2P
$$

下降为：

$$
\frac{2P}{N}
$$

于是每卡模型状态显存近似为：

$$
M_{\text{Z2}} = 2P + 4P + \frac{2P}{N} + \frac{8P}{N}
$$

整理得：

$$
M_{\text{Z2}} = 6P + \frac{10P}{N}
$$

如果将 master weights 也纳入更激进的分片收益估算，那么许多资料会把 ZeRO-2 的教学图画得更省一些；但从理解机制出发，最关键的是：ZeRO-2 已经不再保留完整梯度副本。

这一变化也影响通信方式。传统 DP 的典型做法是 AllReduce 梯度，结果是每张卡最后都得到完整梯度；而 ZeRO-2 更接近于 ReduceScatter：

- Reduce：先把各卡梯度求和或平均。
- Scatter：再把结果切开，每张卡只保留自己那份。

因此，ZeRO-2 不只是“梯度省显存”，也是“梯度同步方式发生了变化”。

### ZeRO-3：再分片参数

ZeRO-3 的关键变化，是连参数本身也不再完整常驻。

这是 ZeRO 三阶段里最激进的一步。参数是前向和反向真正要参与计算的权重，如果把参数也切开，就能继续降低常驻显存，但代价是计算前必须临时把所需完整权重拼出来。

参数显存因此从：

$$
2P
$$

下降为：

$$
\frac{2P}{N}
$$

如果同时按照统一思路，将参数、梯度、master weights、Adam 两个矩都视为可按分片策略均摊，那么 ZeRO-3 的理想化总量可以写成：

- 参数：$ \frac{2P}{N} $
- 梯度：$ \frac{2P}{N} $
- master weights：$ \frac{4P}{N} $
- 优化器状态：$ \frac{8P}{N} $

合计：

$$
M_{\text{Z3}} = \frac{16P}{N}
$$

这就是最常见的 ZeRO-3 结论来源。

不同阶段放在一起看更清楚：

| 阶段 | 每卡模型状态显存近似 |
|---|---|
| 传统 DP | $16P$ |
| ZeRO-1 | $8P + \frac{8P}{N}$ |
| ZeRO-2 | $6P + \frac{10P}{N}$ |
| ZeRO-3 | $\frac{16P}{N}$ |

以 4 卡为例：

| 阶段 | 每卡状态开销 |
|---|---|
| 传统 DP | $16P$ |
| ZeRO-1 | $10P$ |
| ZeRO-2 | $8.5P$ |
| ZeRO-3 | $4P$ |

这说明 ZeRO-3 的节省幅度最大，但它的成立前提也最强：需要更多、更频繁的通信来弥补“参数不再完整常驻”带来的缺口。

### 为什么 ZeRO-3 通信最多

原因很直接：参数已经分散在不同卡上，而前向和反向计算又需要完整权重。

因此，在某一层真正开始计算前，需要先执行 AllGather。AllGather 可以理解为：

- 每张卡拿出自己保存的参数分片；
- 把这些分片发给组内其他卡；
- 最终每张卡都临时拼出当前层所需的完整参数。

一层算完之后，如果这份完整参数不再需要，就可以释放临时缓存，回到“每卡只常驻自己的分片”状态。

反向时同理。为了计算梯度，往往还需要再次拿到该层完整参数或相关上下文；梯度求和之后，再通过 ReduceScatter 把结果重新切成分片。

简化流程可以写成：

1. 前向前：AllGather 当前层参数。
2. 前向后：释放不再需要的完整参数缓存。
3. 反向前：必要时再次 AllGather 参数或上下文。
4. 反向后：ReduceScatter 梯度。
5. 优化器更新：每卡只更新自己持有的参数分片和状态分片。

看一个 8 参数、4 卡的玩具例子：

- 卡 0：持有参数 0, 1
- 卡 1：持有参数 2, 3
- 卡 2：持有参数 4, 5
- 卡 3：持有参数 6, 7

如果某层计算需要全部 8 个参数，那么每张卡必须先把这 8 个参数临时拼全，才能完成矩阵乘法。计算结束后，再释放完整副本，只留下各自负责的 2 个参数。这样做的本质，就是用“按需通信”替代“长期冗余存储”。

ZeRO-3 因此具备两个同时存在的特点：

| 特点 | 含义 |
|---|---|
| 显存最低 | 常驻状态被最大程度分片 |
| 通信最多 | 计算时需要反复把完整参数拼出来 |

所以，ZeRO-3 不是“无代价省显存”，而是“以通信换显存”。

---

## 代码实现

工程里最常见的实现是 DeepSpeed。它把参数分片、梯度分片、通信调度和状态管理封装好了，用户通常只需要准备配置和训练脚本，而不会手写底层的 AllGather/ReduceScatter。

先看一个最小可用的 DeepSpeed 配置：

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

这个配置的含义如下：

| 配置项 | 作用 |
|---|---|
| `train_micro_batch_size_per_gpu` | 每张卡一次前向实际处理的样本数 |
| `gradient_accumulation_steps` | 累积多少次小 batch 再做一次参数更新 |
| `fp16.enabled` | 启用 FP16 混合精度 |
| `zero_optimization.stage` | 选择 ZeRO 阶段，`3` 表示 ZeRO-3 |
| `offload_param` | 将部分参数分片放到 CPU |
| `offload_optimizer` | 将优化器状态放到 CPU |

这里的 `offload`，就是把原本需要驻留在 GPU 的数据转移到 CPU 甚至 NVMe，以进一步压缩 GPU 显存。代价是访问这些数据时要经过更慢的链路，所以通常会牺牲吞吐。

下面给出一个可直接运行的 Python 小程序，用来计算不同 ZeRO 阶段下的理论模型状态显存。它不依赖 DeepSpeed，本质上只是把上面的公式写成程序，方便做数量级判断。

```python
from __future__ import annotations


def zero_memory_gib(params_in_billions: float, num_gpus: int, stage: int) -> float:
    """
    返回每张卡的理论模型状态显存（GiB）。
    口径：
    - FP16 params: 2 bytes
    - FP16 grads: 2 bytes
    - FP32 master weights: 4 bytes
    - Adam moments: 8 bytes
    """
    if params_in_billions <= 0:
        raise ValueError("params_in_billions must be > 0")
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be one of 0, 1, 2, 3")

    total_params = params_in_billions * 1_000_000_000

    if stage == 0:
        bytes_per_param = 2 + 2 + 4 + 8
    elif stage == 1:
        bytes_per_param = 2 + 2 + 4 + (8 / num_gpus)
    elif stage == 2:
        bytes_per_param = 2 + (2 / num_gpus) + 4 + (8 / num_gpus)
    else:  # stage == 3
        bytes_per_param = (2 / num_gpus) + (2 / num_gpus) + (4 / num_gpus) + (8 / num_gpus)

    total_bytes_per_gpu = total_params * bytes_per_param
    return total_bytes_per_gpu / (1024 ** 3)


def pretty_report(params_in_billions: float, num_gpus: int) -> None:
    print(f"model={params_in_billions}B params, gpus={num_gpus}")
    for stage in range(4):
        gib = zero_memory_gib(params_in_billions, num_gpus, stage)
        print(f"stage {stage}: {gib:.2f} GiB / GPU")


if __name__ == "__main__":
    pretty_report(7, 4)

    dp = zero_memory_gib(7, 4, 0)
    z3 = zero_memory_gib(7, 4, 3)

    assert dp > z3
    assert 26 <= z3 <= 29
```

如果运行这个脚本，7B 模型在 4 卡上的理论结果大致会接近：

| 阶段 | 每卡模型状态显存 |
|---|---:|
| Stage 0 / 传统 DP | 104.31 GiB |
| Stage 1 | 65.19 GiB |
| Stage 2 | 55.41 GiB |
| Stage 3 | 26.08 GiB |

这里是 GiB 口径，也就是按 $1024^3$ 换算，因此与前文按 GB 粗算出的 112GB、28GB 会略有差异。两者并不矛盾，只是单位不同：

| 单位 | 换算方式 |
|---|---|
| GB | $10^9$ bytes |
| GiB | $1024^3$ bytes |

为了帮助理解 ZeRO-3 的执行逻辑，下面给一个更接近框架行为的伪代码：

```python
for batch in dataloader:
    hidden = batch.inputs

    # forward
    for layer in model.layers:
        full_weight = all_gather(layer.weight_shard)
        hidden = layer.forward(hidden, full_weight)
        release(full_weight)

    loss = compute_loss(hidden, batch.labels)

    # backward
    grad = loss.backward_seed()
    for layer in reversed(model.layers):
        full_weight = all_gather(layer.weight_shard)
        grad, local_grad_shard = layer.backward(grad, full_weight)
        reduce_scatter_and_accumulate(local_grad_shard)
        release(full_weight)

    # optimizer step on local shards only
    optimizer.step()
    optimizer.zero_grad()
```

这段伪代码要表达的重点不是 API 细节，而是顺序：

1. 当前层计算前，把参数分片拼成完整参数。
2. 当前层算完后，尽快释放完整参数副本。
3. 梯度同步不是简单保留完整结果，而是同步后只保留本地分片。
4. 参数更新也只发生在本地持有的那部分切片上。

如果需要一个更接近真实工程的最小训练脚本，DeepSpeed 里通常是这样的结构：

```python
import deepspeed
import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.net(x)


def main():
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config="ds_config.json",
    )

    for _ in range(10):
        x = torch.randn(8, 128, device=model_engine.device)
        y = torch.randn(8, 128, device=model_engine.device)

        out = model_engine(x)
        loss = ((out - y) ** 2).mean()

        model_engine.backward(loss)
        model_engine.step()


if __name__ == "__main__":
    main()
```

要真正运行它，还需要：

1. 安装 DeepSpeed 与 PyTorch。
2. 准备 `ds_config.json`。
3. 用分布式命令启动，例如 `deepspeed --num_gpus 4 train.py`。

真实工程里，用户通常不会手写底层通信原语，而是让 DeepSpeed 在 `forward`、`backward` 和 `step` 的前后自动插入通信钩子。你真正需要关注的是配置是否合理，特别是：

- `micro batch size`
- `gradient accumulation`
- 是否开启 `offload`
- 机器间拓扑与带宽是否匹配
- 激活显存是否才是真正瓶颈

---

## 工程权衡与常见坑

ZeRO 的核心权衡很简单：显存换通信。因此，Stage 并不是越高越好。

先看 ZeRO-2 与 ZeRO-3 的差别：

| 维度 | ZeRO-2 | ZeRO-3 |
|---|---|---|
| 参数是否分片 | 否 | 是 |
| 梯度是否分片 | 是 | 是 |
| 优化器状态是否分片 | 是 | 是 |
| 前向通信频率 | 较低 | 高 |
| 反向通信频率 | 中 | 高 |
| 对带宽要求 | 较高 | 很高 |
| 吞吐稳定性 | 通常更稳 | 更依赖网络 |

可以把工程上的判断压缩成一句话：ZeRO-2 更像“保守、省显存、吞吐相对稳”的方案；ZeRO-3 更像“最大化压显存，但必须接受更重通信”的方案。

最常见的坑有下面几类。

1. 理论省显存，不等于训练一定更快。  
ZeRO-3 的理论状态显存最低，但 step time 往往更长。因为参数不常驻，前向和反向都要更频繁地通信。如果 GPU 在等网络，算力就浪费了。

2. 网络拓扑的重要性通常被低估。  
“有 8 张卡”这件事本身不说明问题。更关键的是：这 8 张卡怎么连。NVLink、PCIe、跨机以太网、Infiniband，差别非常大。ZeRO-3 在跨机时尤其容易暴露链路短板。

3. offload 不是免费午餐。  
把参数或优化器状态 offload 到 CPU/NVMe，确实能继续压缩 GPU 显存，但访问代价会明显上升。此时瓶颈可能从 GPU 显存变成 PCIe 带宽、主机内存带宽、磁盘延迟。

4. 显存公式只覆盖模型状态，不覆盖全部训练开销。  
训练总显存通常可以分成：

| 部分 | 是否包含在 ZeRO 公式里 |
|---|---|
| 参数 | 是 |
| 梯度 | 是 |
| 优化器状态 | 是 |
| 激活 | 否 |
| CUDA 缓存/碎片 | 否 |
| 通信缓冲区 | 通常不完整包含 |
| 临时工作区 | 否 |

这意味着：即便 ZeRO-3 把模型状态理论上压到了 28GB，也不代表 32GB 显存卡就一定能跑通。激活可能还会额外吃掉很多显存，尤其在长序列、大 batch 或 checkpointing 没开的时候。

5. ZeRO 解决的是状态冗余，不解决算子本体过大。  
如果某一层的单次矩阵乘法本身就大到单卡放不下，单靠 ZeRO 也无能为力。这时需要考虑张量并行、流水并行或算子层面的切分。

下面给一个很常见的误判例子：

| 现象 | 误判 | 实际问题 |
|---|---|---|
| 开 ZeRO-3 后仍 OOM | 以为 ZeRO 失效 | 激活显存或通信缓冲区过大 |
| 吞吐下降 30% | 以为 GPU 性能差 | 互联带宽不足，通信阻塞 |
| `nvidia-smi` 看起来占用很高 | 以为参数分片没生效 | 看到的是总占用，不只是参数状态 |
| CPU 占用激增 | 以为程序有 bug | 开了 offload，数据在 CPU/GPU 间搬运 |

一个实用建议是：在上 ZeRO-3 之前，先用 `ZeRO-2 + offload` 做基线。原因很现实：

- ZeRO-2 通常更稳，调试成本更低。
- 如果 ZeRO-2 已经能把模型塞进目标显存，且吞吐可接受，就没有必要继续上更重的 ZeRO-3。
- 只有当状态显存仍然压不下去时，ZeRO-3 才真正值得尝试。

调试时建议重点观察以下项目：

| 检查项 | 建议 |
|---|---|
| NCCL 通信健康度 | 先跑 `nccl-tests`，确认带宽和延迟 |
| 单步耗时拆分 | 分开看 forward、backward、optimizer、comm |
| 显存来源 | 区分参数状态、激活、缓存、碎片 |
| batch 策略 | 联动调 `micro batch size` 与 `gradient accumulation` |
| 拓扑 | 确认是否走 NVLink、PCIe、IB，避免错误绑卡 |
| offload 效果 | 分别测 `optimizer offload` 与 `param offload` 的开销 |

再看一个工程化的判断例子。假设 8 卡训练时：

- ZeRO-2 的 step time 是 1.0 秒
- ZeRO-3 的 step time 是 1.5 秒

如果两者最终都能把模型跑起来，那么生产环境里并不能简单说 ZeRO-3 更好。因为真正关心的指标往往是单位时间训练 token 数、单位成本完成训练所需时间，而不是“显存省得最多”。

---

## 替代方案与适用边界

方案选择可以看成一个很简单的决策过程：

```text
先看单卡显存是否能放下
-> 能放下：优先传统 DP，简单、稳定、吞吐通常更好
-> 放不下：先看瓶颈是优化器状态、梯度，还是参数本体
-> 若主要卡在优化器状态：先试 ZeRO-1
-> 若梯度也明显占显存：试 ZeRO-2
-> 若参数本身已放不下：再考虑 ZeRO-3
-> 若 GPU 显存仍不足，但 CPU 资源更宽裕：加 offload
-> 若单层计算本身放不下：考虑模型并行，而不只是 ZeRO
```

常见方案的适用性如下：

| 方案 | 适合场景 | 不适合场景 |
|---|---|---|
| 传统 DP | 模型不大、追求简单稳定 | 大模型显存吃紧 |
| ZeRO-1 | 主要卡在 Adam 状态 | 参数和梯度也已成瓶颈 |
| ZeRO-2 | 显存紧张但网络一般可用 | 参数本体已明显放不下 |
| ZeRO-3 | 超大模型、互联带宽较强 | 网络差、跨机通信慢 |
| ZeRO-2 + Offload | GPU 显存紧、可接受更高延迟 | CPU/PCIe 也已紧张 |
| 模型并行 | 单层张量太大、算子必须切开 | 只是状态冗余高，不是算子过大 |

这张表背后的逻辑可以再展开一下：

| 问题类型 | 更优先的思路 |
|---|---|
| 冗余太多 | ZeRO |
| 激活太大 | Activation Checkpointing、减小序列长度、减小 batch |
| 单层算子太大 | Tensor Parallel / Pipeline Parallel |
| GPU 显存不够但 CPU 资源足 | Offload |
| 网络太差 | 降 ZeRO 阶段，优先更少通信的方案 |

看一个 4 卡、每卡 24GB 的简单例子：

- 如果模型状态只要 18GB，传统 DP 往往就够，没必要引入额外复杂度。
- 如果模型状态要 30GB，ZeRO-1 未必够，ZeRO-2 更现实。
- 如果模型状态要 50GB，通常要考虑 ZeRO-3 或 `ZeRO-2 + offload`。
- 如果即使用了 ZeRO-3，某一层的计算仍放不下，那问题已经不是状态冗余，而是模型并行问题。

再给一个更贴近生产的例子。假设是 4 卡训练，链路带宽只有约 10 GB/s：

- 这类环境通常不适合一上来就 ZeRO-3。
- 更稳妥的顺序是：先 ZeRO-2，再尝试 optimizer offload，再评估是否需要 param offload。
- 只有确认 NCCL 拓扑、节点间通信和链路带宽都稳定后，再尝试 ZeRO-3 才更合理。

因此，方案选择的核心不是“哪种技术最先进”，而是“哪种技术在当前硬件条件下总成本最低”。ZeRO 是大模型显存工程的主力方案，但 Stage 的选择必须同时看显存、带宽、拓扑和吞吐目标，不能只看理论压缩比。

---

## 参考资料

| 来源 | 简短说明 |
|---|---|
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | ZeRO 原始论文，定义了分阶段消除参数、梯度、优化器状态冗余的核心思路 |
| DeepSpeed ZeRO 官方文档 | 官方说明 ZeRO-1/2/3 的机制、配置方式与工程限制 |
| DeepSpeed ZeRO-Offload / ZeRO-Infinity 文档 | 说明如何把参数或优化器状态 offload 到 CPU / NVMe，继续扩大可训练规模 |
| DeepSpeed GitHub 示例 | 给出可运行配置与训练脚本，适合理解实际接入方式 |
| NVIDIA NCCL Tests | 用于检查集群通信带宽与延迟，是部署 ZeRO 前的基础排查工具 |
| Microsoft Research 关于 ZeRO 的技术博客 | 用工程视角解释 ZeRO 在大模型训练中的收益与代价 |
| PyTorch Distributed 文档 | 补充理解 AllReduce、AllGather、ReduceScatter 等通信原语的语义 |

如果你需要进一步深挖，建议按下面顺序阅读：

1. 先看 ZeRO 原始论文，明确它到底解决什么冗余。
2. 再看 DeepSpeed 官方文档，理解配置项与阶段行为。
3. 然后看 PyTorch Distributed 文档，把 AllGather、ReduceScatter 的语义补完整。
4. 最后结合 NCCL 测试结果，回到自己的硬件环境里判断 Stage 是否值得升级。
