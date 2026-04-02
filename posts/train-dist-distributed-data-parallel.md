## 核心结论

分布式数据并行 DDP，英文是 `DistributedDataParallel`，意思是“每张卡各自保存一份完整模型，但每轮训练后把梯度同步成同一个结果”。它的核心不是把一整个反向传播做完再统一通信，而是在反向传播过程中给参数梯度注册 hook。某一组梯度一旦“全准备好”，就立刻对这一组做 All-Reduce。

All-Reduce 可以理解成“多张卡把各自的梯度求和或求平均，然后每张卡都拿到同一份结果”。因此，DDP 的 forward 本质上彼此独立，真正发生同步的是 backward 阶段的梯度。

DDP 的性能关键在两个点：

1. 梯度分桶，也就是把很多参数梯度按大小打包成 bucket。
2. 通信计算重叠，也就是某个 bucket 算完就立刻异步通信，不等整张计算图结束。

默认 bucket 大小是 `25 MiB`，对应参数 `bucket_cap_mb=25`。如果模型总梯度体积是 $G$ MiB，那么 bucket 数量近似为：

$$
\text{num\_buckets}=\left\lceil \frac{G}{\text{bucket\_cap\_mb}} \right\rceil
$$

这决定了通信被切成多少段，也决定了 overlap 的颗粒度。颗粒度太粗，通信启动得晚；颗粒度太细，调度开销会上升。

一个新手需要记住的最短结论是：DDP 不是“最后统一同步”，而是“梯度边算边同步”。

---

## 问题定义与边界

DDP 要解决的问题是：在多卡或多节点训练中，如何让多个进程各自处理不同数据，同时又保证模型更新结果一致。

这里的“进程”可以理解成“一张 GPU 对应一个训练工作单元”。典型用法是 8 张卡启动 8 个进程，每个进程：

- 拿到一份模型副本
- 拿到自己那一部分数据
- 独立做 forward
- 独立做 loss
- 独立触发 backward
- 最后用同步后的 `.grad` 执行 `optimizer.step()`

所以 DDP 不负责自动切输入，也不替你写训练循环。它负责的是：在反向传播时，自动把所有副本的梯度同步好。

下面这几个边界必须说清楚：

| 项目 | DDP 是否负责 | 说明 |
|---|---|---|
| 输入数据切分 | 否 | 通常由 `DistributedSampler` 处理 |
| forward 结果同步 | 否 | 每个进程各算各的 |
| backward 梯度同步 | 是 | 这是 DDP 的核心职责 |
| optimizer 状态切分 | 否 | 默认每卡都保留完整 optimizer 状态 |
| 参数副本是否完整 | 是 | 每张卡默认都有完整模型 |

玩具例子：2 张卡训练一个两层 MLP。GPU0 看样本 `[0,1,2,3]`，GPU1 看样本 `[4,5,6,7]`。它们各自 forward、各自算 loss，但 backward 时会把同名参数的梯度同步成一致值，之后再各自执行一步相同的参数更新。这样虽然每张卡看见的数据不同，但模型参数仍然保持一致。

真实工程里，这个边界更重要。比如 8 张 A100 在同一台 DGX 机器上训练，节点内靠 NVLink；如果扩展到多机，就要跨 InfiniBand。DDP 仍然只做“梯度同步”，不会帮你自动解决数据加载瓶颈、checkpoint 设计、失败恢复这些工程问题。

---

## 核心机制与推导

DDP 初始化时会创建 `Reducer`。可以把 `Reducer` 理解成“梯度同步调度器”。它做两件事：

1. 按参数顺序把梯度划进多个 bucket。
2. 给每个参数的梯度累积节点注册 hook。

“hook”可以理解成“某个事件发生时自动执行的回调”。这里的事件就是：某个参数的梯度已经算出来了。

当 backward 运行时，参数梯度会按反向拓扑逐步就绪。每当某个参数梯度 ready，DDP 就给所属 bucket 记一个“已完成”标记；当 bucket 里所有梯度都 ready，就立刻异步发起 All-Reduce。

设第 $i$ 个 bucket 的反向计算耗时为 $C_i$，All-Reduce 耗时为 $R_i$。如果没有 overlap，总时间更接近：

$$
T_{\text{serial}} \approx \sum_i (C_i + R_i)
$$

而有 overlap 时，时间更接近：

$$
T_{\text{overlap}} \approx \sum_i \max(C_i, R_i)
$$

这不是严格公式，但足够表达工程直觉：通信能被后续计算掩盖多少，取决于两者谁更慢。

玩具例子：模型总梯度 800 MiB，`bucket_cap_mb=25`，则 bucket 数大约是：

$$
\left\lceil \frac{800}{25} \right\rceil = 32
$$

假设第 8 个 bucket 的梯度计算要 4 ms，通信要 1 ms。如果先算完所有梯度再通信，这 1 ms 一定额外增加；但如果这个 bucket 一 ready 就通信，那么这 1 ms 可以和后面 bucket 的反向传播并行，额外暴露出来的时间可能很小。粗略看，就是从“4 ms 计算 + 1 ms 通信串行”变成“接近 4 ms 主导”。

为什么 bucket 顺序很重要？因为多进程必须按一致顺序发起 collective 通信，否则就可能互相等待，最后 hang 住。DDP 为了避免这个问题，会按固定规则组织 bucket，并要求不同 rank 上触发的通信顺序保持一致。

这里还有一个常见误解：All-Reduce 不是“把参数同步”，而是“把梯度同步”。参数更新仍然发生在本地 optimizer 里。只是因为所有 rank 在 step 前拿到的 `.grad` 相同，所以 step 后参数也保持一致。

---

## 代码实现

最小可用形态里，DDP 代码和单卡训练几乎一样，变化集中在初始化和模型包装。

```python
import math

def bucket_count(total_grad_mb: float, bucket_cap_mb: float = 25.0) -> int:
    assert total_grad_mb >= 0
    assert bucket_cap_mb > 0
    return math.ceil(total_grad_mb / bucket_cap_mb)

def ring_allreduce_time_ms(bucket_mb: float, bandwidth_gb_s: float) -> float:
    assert bucket_mb >= 0
    assert bandwidth_gb_s > 0
    return (bucket_mb / 1024) / bandwidth_gb_s * 1000

# 玩具例子：800 MiB 梯度、25 MiB bucket
assert bucket_count(800, 25) == 32

# 带宽数量级对比：节点内 NVLink 400 GB/s，跨节点 200 Gbps 约等于 25 GB/s
nvlink_ms = ring_allreduce_time_ms(25, 400)
ib_ms = ring_allreduce_time_ms(25, 25)

assert nvlink_ms < ib_ms
assert round(ib_ms / nvlink_ms) == 16
print(bucket_count(800, 25), round(nvlink_ms, 4), round(ib_ms, 4))
```

这段代码不是 PyTorch 训练代码，而是一个可运行的“通信量级估算器”。它验证了两个事实：

1. 800 MiB 梯度在 25 MiB bucket 下会分成 32 个 bucket。
2. 400 GB/s 与 25 GB/s 的带宽差大约是 16 倍。

对应到 DDP 真实代码，核心结构通常如下：

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    model = nn.Linear(128, 64).cuda(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=25,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    x = torch.randn(32, 128, device=local_rank)
    y = model(x).sum()
    y.backward()

if __name__ == "__main__":
    main()
```

这个例子里最关键的事实有三个：

1. 每个进程只绑定自己的本地 GPU。
2. `model = DDP(model, ...)` 之后，forward 写法基本不变。
3. `loss.backward()` 时，DDP 的 hook 会自动触发 bucket All-Reduce。

真实工程例子：8 张 A100 机器做大模型预训练。每张卡保留完整模型副本，反向传播时通过 NVLink 在节点内高带宽同步梯度；如果是多机训练，则还要经过 200 Gbps InfiniBand 做跨节点同步。节点内典型带宽可到约 `400 GB/s` 量级，跨节点约 `25 GB/s` 量级，差距约 16 倍。这意味着同样一个 bucket，在跨节点环境中更容易让通信成为瓶颈，因此 bucket 设置、batch 设计、梯度累积都会更敏感。

---

## 工程权衡与常见坑

DDP 配置里最常见的四个开关如下：

| 配置 | 作用 | 适合场景 | 代价或风险 |
|---|---|---|---|
| `find_unused_parameters=True` | 查找本轮没参与反向传播的参数 | 动态图、条件分支模型 | 会遍历图，常见约 10% 开销 |
| `static_graph=True` | 告诉 DDP 图结构稳定 | 每轮参与反向传播的参数集合固定 | 配错会导致错误假设 |
| `broadcast_buffers=False` | 关闭 buffer 广播 | 大规模训练、BN 影响可控 | 需确认 buffer 不会引入不一致 |
| `gradient_as_bucket_view=True` | 让 grad 直接作为 bucket 视图 | 想省显存、减少复制 | 不能对 grad 做 `detach_()` |

第一个坑是 unused parameters。它的意思是“某些参数在这一轮 forward 里根本没被用到，因此也不会产生梯度”。如果模型有条件分支，某个分支本轮没走到，那么对应参数可能就没梯度。默认情况下，这容易让 DDP 等待错误的同步点。`find_unused_parameters=True` 能解决，但会增加图遍历开销。

第二个坑是把 `static_graph=True` 当成通用优化。它的前提不是“模型大体不变”，而是“每轮参与 backward 的图结构和参数集合稳定”。如果你有 MoE、随机路由、条件执行，这个前提通常不成立。

第三个坑是 `broadcast_buffers`。buffer 可以理解成“不是参数、但会影响模型行为的状态”，典型例子是 BatchNorm 的 running mean / running var。默认 DDP 会广播这些 buffer，保证副本一致；但在大规模训练中，这也会带来额外通信。如果你本来就不用 BN，或者这些 buffer 同步没有收益，关掉它能减少开销。

第四个坑是 `gradient_as_bucket_view=True`。它通过复用 bucket 内存减少峰值显存，常见能省几个百分点，经验上可接近 5%。但此时 `.grad` 不再是独立张量，而是 bucket 的 view。很多依赖“原地断开梯度”的写法会失效，例如某些自定义优化逻辑里调用 `grad.detach_()` 就会报错。

还有一个常被忽略的点：bucket 太大和太小都不好。太大时，必须等更多梯度 ready 才能启动通信，overlap 变差；太小时，collective 次数增多，调度和启动延迟会抬高。它不是“越小越快”，而是一个要结合模型层次、网络带宽、反向拓扑一起调的参数。

---

## 替代方案与适用边界

DDP 的优点是简单、稳定、和单卡训练代码最接近，但它默认要求“每张卡放得下完整模型和完整 optimizer 状态”。当这个前提不成立，就需要替代方案。

| 方案 | 参数是否完整副本 | 显存占用 | 通信复杂度 | 适用场景 |
|---|---|---|---|---|
| DDP | 是 | 高 | 中 | 模型放得下，追求稳定吞吐 |
| DDP + ZeroRedundancyOptimizer | 参数完整，optimizer 分片 | 中 | 中 | optimizer 状态太大 |
| FSDP | 否，参数/梯度/状态分片 | 低 | 高 | 单卡放不下超大模型 |

`ZeroRedundancyOptimizer` 可以理解成“仍然用 DDP 同步梯度，但 optimizer 状态不再每卡都保存一整份”。这对 Adam 一类优化器尤其有价值，因为它们除了参数本身，还要存一阶、二阶统计量。

FSDP，英文是 `FullyShardedDataParallel`，意思是“把参数、梯度、optimizer 状态都切碎分到不同卡上”。白话说，它不是每卡一整份模型，而是“模型拆砖分放”。这能显著降低单卡显存占用，但会引入更复杂的 gather / shard 通信，也更依赖正确的 wrap 策略和预取策略。

什么时候不该先上 FSDP？当你的模型单卡能放下，而且主要问题是训练速度与工程稳定性时。因为 DDP 的心智负担更低，问题定位也更直接。什么时候 DDP 不够？当你发现激活之外，参数和 optimizer 状态已经把单卡显存压满，这时继续调 bucket、调 batch 往往收益有限，应转向分片方案。

一个实用判断是：

- 模型能放下，网络快，先用 DDP。
- 模型能放下，但 Adam 状态过大，考虑 `DDP + ZeroRedundancyOptimizer`。
- 模型放不下，直接评估 FSDP。

---

## 参考资料

- PyTorch DistributedDataParallel API 文档：https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch DDP Notes：https://docs.pytorch.org/docs/stable/notes/ddp
- PyTorch ZeroRedundancyOptimizer 教程：https://docs.pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html
- PyTorch FSDP 文档：https://docs.pytorch.org/docs/stable/fsdp
- NVIDIA DGX A100 架构介绍：https://developer.nvidia.com/blog/defining-ai-innovation-with-dgx-a100/
