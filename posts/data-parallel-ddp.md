## 核心结论

数据并行（Data Parallelism）的目标，是让多张 GPU 在并行处理不同样本后，仍然执行**同一次参数更新**。这要求各卡在 `optimizer.step()` 之前拥有一致的梯度统计，否则每张卡都会沿着不同方向更新，训练会偏离同步 SGD 的数学定义。

PyTorch 的 DDP（DistributedDataParallel）在典型多 GPU 训练中默认配合 NCCL 使用 `AllReduce` 做梯度同步。它不是把所有梯度先发给一个中心节点统一处理，而是让每张卡都同时参与发送、接收和聚合。这样通信负载分散在集体通信拓扑里，而不是集中到单一参数服务器。

参数服务器（Parameter Server, PS）的问题，不是“不能同步梯度”，而是**扩展时中心节点容易成为热点**。以 8 张卡、每步同步 1 GB 梯度为例，如果 8 个 Worker 同时向中心节点推送梯度，就会形成明显的 `incast`；中心节点完成聚合后，还要把结果再发回去，`outcast` 也会堆在同一处。Ring AllReduce 则避免了这个中心瓶颈。设参与设备数为 $N$，需要同步的梯度总量为 $M$，单卡总通信量为：

$$
V = 2\frac{N-1}{N}\times M
$$

这里的含义是“单卡 send + recv 的总字节数”，不是“集群总流量”。

先纠正常见误解。8 卡同步 1 GB 梯度时，每张卡不是向其余 7 张卡各发 1 GB，因此不是 7 GB。Ring AllReduce 的实际单卡通信量是：

$$
V = 2\times\frac{8-1}{8}\times 1\text{ GB}
= 1.75\text{ GB}
$$

AllReduce 的价值就在这里：它不是全量广播给所有对端，而是把张量切块后沿环传递，在传递过程中完成求和与重组。

DDP 的工程优势不只是“使用了 AllReduce”，而是把 AllReduce 嵌进反向传播。某一批参数梯度一旦就绪，DDP 的 `Reducer` 会立刻把对应 bucket 发起异步通信，而不是等整次 backward 结束后再统一同步。这使得**后面层的梯度计算**可以和**前面 bucket 的通信**重叠，减少纯等待时间。

| 方案 | 8 卡、1 GB 梯度时单卡视角 | 中心瓶颈 | 是否容易与 backward 重叠 |
|---|---:|---|---|
| 参数服务器 | Worker 上传 1 GB、下载 1 GB；中心节点承受 8 路汇聚与 8 路回传 | 高 | 一般较差 |
| Ring AllReduce | 每卡总通信约 1.75 GB | 无单中心 | 好 |
| DDP + Ring AllReduce | 通信量同 Ring；由 Reducer 自动分桶并异步触发 | 无单中心 | 最好，工程上最常用 |

---

## 问题定义与边界

本文只讨论一个具体问题：在多 GPU 同步训练里，如何在保持各卡参数一致的前提下，把梯度同步的通信开销压到可接受范围内。

“参数一致”的精确定义是：在一个训练 step 结束时，所有参与数据并行的进程都要得到相同的参数值。因为每张卡处理的 mini-batch 不同，所以各卡本地先得到的是**局部梯度**；同步阶段的作用，是把这些局部梯度聚合成**全局平均梯度**，然后每张卡再执行相同的优化器更新。用公式写就是：

$$
g^{(i)} = \nabla_\theta \mathcal{L}_i(\theta), \qquad
\bar g = \frac{1}{N}\sum_{i=1}^{N} g^{(i)}
$$

随后每张卡都执行：

$$
\theta_{t+1} = \theta_t - \eta \bar g
$$

其中 $N$ 是参与同步的设备数，$\eta$ 是学习率。

如果不做这一步，全局上并不是“多卡训练”，而是“多份模型各自训练”。短期内 loss 可能仍下降，但这些模型参数会逐步分叉，最终既不等价于大 batch 训练，也无法保证收敛轨迹一致。

本文的边界如下。

| 讨论项 | 本文是否覆盖 | 说明 |
|---|---|---|
| 数据并行 | 是 | 每卡完整持有模型副本 |
| 参数服务器 | 是 | 作为对照架构讨论瓶颈 |
| NCCL Ring AllReduce | 是 | 重点机制 |
| PyTorch DDP Reducer | 是 | 重点工程实现 |
| 模型并行 | 否 | 模型被切到多卡，问题性质不同 |
| ZeRO / FSDP / Sharding | 部分提及 | 只做边界对比 |
| 异步训练 | 否 | 本文只讨论同步 SGD 风格 |

还要区分两个容易混淆的时间点。

第一种做法是“先算完所有梯度，再统一同步”。数学上可行，但会在 step 尾部留下很长一段纯通信时间。

第二种做法是“梯度一部分 ready，就立刻发起该部分同步”。DDP 的高性能实现属于第二种，它依赖 `bucket + backward hook`，把通信尽量前移并与 backward 重叠。

对新手来说，可以把本文理解为回答三个问题：

1. 为什么数据并行一定要同步梯度，而不是各卡各算各的。
2. 为什么参数服务器在规模上去后容易形成中心热点。
3. 为什么 DDP 的性能关键不只是 AllReduce 本身，而是**何时触发 AllReduce**。

---

## 核心机制与推导

### 1. Ring AllReduce 的通信量为什么不是 $(N-1)M$

设总梯度大小为 $M$，设备数为 $N$。Ring AllReduce 一般分两个阶段：

1. `Reduce-Scatter`
   把张量切成 $N$ 块，经过 $N-1$ 轮传递与累加后，每张卡持有其中一块的全局和。
2. `All-Gather`
   再经过 $N-1$ 轮，把这些已求和的块重新分发给所有卡，使每张卡重新得到完整结果。

每一轮每张卡只发送一块、接收一块，每块大小约为 $M/N$。因此单卡总通信量为：

$$
V = 2(N-1)\times \frac{M}{N}
= 2\frac{N-1}{N}\times M
$$

这个公式给出的是**单卡总 send+recv**。如果看全集群总流量，则还要再乘以 $N$：

$$
V_{\text{cluster}} = N \cdot V = 2(N-1)M
$$

所以要始终分清“单卡视角”和“集群总量”两个视角。

### 2. 8 卡、1 GB 梯度为什么是 1.75 GB

把数字代入上式：

- $N = 8$
- $M = 1\text{ GB}$

得到：

$$
V = 2\frac{7}{8}\times1\text{ GB}=1.75\text{ GB}
$$

更细一点看：

- 每轮传输块大小：$1/8$ GB = 128 MB
- 总轮数：`Reduce-Scatter` 7 轮 + `All-Gather` 7 轮 = 14 轮
- 单卡总量：$14 \times 128\text{ MB} = 1792\text{ MB} \approx 1.75\text{ GB}$

整理成表：

| 项目 | 数值 |
|---|---:|
| GPU 数 $N$ | 8 |
| 梯度总量 $M$ | 1 GB |
| 每轮块大小 $M/N$ | 128 MB |
| 总轮数 $2(N-1)$ | 14 |
| 单卡总通信量 | 1792 MB ≈ 1.75 GB |
| 集群总流量 | 14 GB |

因此，“8 卡同步 1 GB 梯度，每卡要发 7 GB”这个说法是把 AllReduce 错看成了“一对多复制广播”。

### 3. 参数服务器为什么容易堵在中心

参数服务器的一个简化流程是：

1. 每个 Worker 计算本地梯度。
2. 全部 Worker 把梯度推送到中心节点。
3. 中心节点做求和或平均。
4. 中心节点再把结果发回 Worker。

如果仍以 8 卡、1 GB 梯度为例：

- 每个 Worker 上传 1 GB
- 中心节点短时间收到 8 GB
- 聚合后还要向 8 个 Worker 再发送总计 8 GB

从单个 Worker 看，好像只是“上传 1 GB、下载 1 GB”；但从中心节点看，压力集中到了同一个 NIC、同一组内存复制、同一台机器的调度上。也就是说，PS 的主要问题不在单个 Worker 的通信量，而在**链路和处理能力都集中在中心节点**。

### 4. DDP 的关键不只是 AllReduce，而是分桶和触发时机

PyTorch DDP 的核心组件是 `Reducer`。它会把参数梯度按顺序组织成多个 `bucket`，每个 bucket 是一批梯度张量的连续缓冲区。这样做有两个目的：

1. 把大量小梯度合并，减少通信调用次数。
2. 某个 bucket 一旦就绪，就立刻异步发起 AllReduce，从而和后续 backward 重叠。

流程可以抽象成：

1. backward 从最后一层开始计算梯度。
2. 某些参数的梯度先完成。
3. 这些梯度被填入对应 bucket。
4. 某个 bucket 内全部梯度都 ready。
5. Reducer 立刻为该 bucket 发起异步 AllReduce。
6. backward 继续计算更前面的层。
7. 通信与计算并行推进。

这依赖 backward hook。每个参数在梯度生成时都会触发一个内部回调，Reducer 用这个回调追踪“哪些梯度已经就绪”。所以 DDP 的梯度同步不是训练尾部的一个独立大步骤，而是被拆散插入到 backward 流程中。

### 5. 为什么大模型必须依赖 overlap

以 175B 参数模型为例，假设梯度使用 FP16，每个参数的梯度占 2 字节，那么单步梯度总量约为：

$$
175\times10^9 \times 2
= 350\times10^9 \text{ bytes}
\approx 350\text{ GB}
$$

如果梯度以 FP32 存储，则约为：

$$
175\times10^9 \times 4
\approx 700\text{ GB}
$$

哪怕只看梯度，这已经是单步数百 GB 的通信级别。此时如果把同步完全放在 backward 结束后统一进行，step 尾部会出现长时间阻塞。工程上必须依赖：

- 分桶
- 异步触发
- 通信与反向重叠
- 更高带宽互联（如 NVLink、InfiniBand）
- 必要时进一步采用 ZeRO/FSDP 等分片方案

### 6. 一个最小数值例子

假设只有两张卡，参数向量为三维，本地梯度分别是：

$$
g^{(1)} = [2,4,6], \qquad g^{(2)} = [1,3,5]
$$

同步平均梯度为：

$$
\bar g = \frac{g^{(1)} + g^{(2)}}{2}
= \frac{[3,7,11]}{2}
= [1.5, 3.5, 5.5]
$$

然后两张卡都用这个相同的 $\bar g$ 更新参数。这个例子虽然小，但表达了数据并行最本质的要求：**每张卡计算局部梯度，同步阶段构造全局平均梯度，更新必须一致**。

---

## 代码实现

下面给两段代码。

第一段是纯 Python 脚本，用来验证 Ring AllReduce 的通信量公式和大模型梯度体量估算。它不依赖 GPU，可以直接运行。

```python
def ring_allreduce_volume_bytes(num_gpus: int, tensor_bytes: int) -> float:
    assert num_gpus >= 2
    assert tensor_bytes >= 0
    return 2 * (num_gpus - 1) / num_gpus * tensor_bytes


def format_gib(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def main() -> None:
    gb = 1024 ** 3

    # 8 卡同步 1 GiB 梯度
    per_gpu = ring_allreduce_volume_bytes(8, gb)
    assert abs(format_gib(per_gpu) - 1.75) < 1e-12

    # 175B 参数，FP16 梯度约 350 GB（十进制）
    params = 175 * 10**9
    fp16_grad_bytes = params * 2
    assert fp16_grad_bytes == 350_000_000_000

    print(f"8 GPUs, 1 GiB gradients -> per GPU traffic = {format_gib(per_gpu):.2f} GiB")
    print(f"175B params, FP16 gradients -> total gradient bytes = {fp16_grad_bytes:,}")


if __name__ == "__main__":
    main()
```

预期输出类似：

```text
8 GPUs, 1 GiB gradients -> per GPU traffic = 1.75 GiB
175B params, FP16 gradients -> total gradient bytes = 350,000,000,000
```

第二段是一个**可运行的最小 DDP 示例**。它使用 CPU + `gloo` 后端，因此即使没有多张 GPU，也能在本机用两个进程验证“各进程本地梯度不同，但同步后参数一致”这一点。

运行方式：

```bash
torchrun --standalone --nproc_per_node=2 ddp_demo.py
```

代码如下：

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def setup() -> tuple[int, int]:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def main() -> None:
    rank, world_size = setup()
    torch.manual_seed(0)

    model = TinyModel()
    ddp_model = DDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    # 两个 rank 使用不同样本，模拟数据并行
    x = torch.tensor(
        [[1.0, 0.0, 0.0, float(rank + 1)],
         [0.0, 1.0, 0.0, float(rank + 1)]]
    )
    y = torch.tensor([[1.0], [2.0]])

    optimizer.zero_grad(set_to_none=True)
    pred = ddp_model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()

    # backward 之后，各 rank 的梯度已经被 DDP 自动同步并平均
    grad = ddp_model.module.linear.weight.grad.detach().clone()
    print(f"rank={rank}, synced_grad={grad.tolist()}")

    optimizer.step()

    # 验证更新后参数一致
    weight = ddp_model.module.linear.weight.detach().clone()
    gather_list = [torch.zeros_like(weight) for _ in range(world_size)]
    dist.all_gather(gather_list, weight)

    if rank == 0:
        for i, w in enumerate(gather_list):
            print(f"rank={i}, updated_weight={w.tolist()}")
        assert torch.allclose(gather_list[0], gather_list[1])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

这段代码里要抓住三点：

1. `DDP(model)` 之后，不需要你手写 `all_reduce`。
2. 真正触发梯度同步的是 `loss.backward()`，不是 `optimizer.step()`。
3. 每个 rank 输入样本不同，但 backward 完成后梯度已经被自动聚合，因此更新后的权重一致。

如果把它迁移到多 GPU，只需要把：

- 后端换成 `nccl`
- 为每个进程设置 `local_rank`
- 把模型和数据移动到对应 GPU

典型 GPU 版本骨架如下：

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = TinyModel().cuda(local_rank)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=25,
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

    x = torch.randn(32, 1024, device=local_rank)
    y = torch.randint(0, 10, (32,), device=local_rank)

    optimizer.zero_grad(set_to_none=True)
    logits = ddp_model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

这里 `bucket_cap_mb` 控制每个梯度桶的上限大小。它不改变数学结果，但会直接影响：

- 通信调用有多碎
- bucket 能多早 ready
- backward 与通信的重叠程度

---

## 工程权衡与常见坑

DDP 的性能瓶颈通常不在“有没有同步”，而在“同步以什么粒度、在什么时机发生”。

### 1. `bucket_cap_mb` 太小

如果 bucket 太小，会出现大量很小的 AllReduce。每次集体通信都有固定启动成本，调用次数过多会导致：

- 通信库调度开销上升
- 小消息比例过高
- GPU 难以形成稳定的计算/通信流水

典型现象是 profiler 里出现大量很短的 NCCL 调用，吞吐下降，但单次调用并不显著慢。

### 2. `bucket_cap_mb` 太大

bucket 太大时，虽然调用次数减少，但 bucket 需要等待内部所有梯度都 ready 才能发起通信。若一个 bucket 覆盖太多层，就会把本应更早开始的通信延后，表现为：

- backward 末尾通信尾巴很长
- overlap 不足
- step time 对通信更敏感

因此 bucket 不是越大越好，而是在“减少启动开销”和“尽早发起通信”之间取平衡。

### 3. 参数顺序改变会影响 bucket 形成

DDP 通常按参数注册顺序组织 bucket。模型结构改动、模块重排、条件分支或某些参数长期不参与反向，都会改变 bucket 的 ready 顺序。结果可能是理论上相同的模型，实际 overlap 却不同。

对新手来说，一个实用判断是：如果你做了结构改动，训练吞吐突然下降，但 FLOPs 基本没变，就要怀疑 bucket 顺序和 overlap 被破坏了。

### 4. 自定义梯度逻辑可能破坏标准同步路径

常见风险包括：

- 手工改写 `.grad`
- 在 backward 外额外做梯度聚合
- 使用某些非常规 autograd 路径
- 让部分参数在某些 step 不参与梯度计算

这些操作未必让训练报错，但可能让 Reducer 无法按预期追踪 bucket 的 ready 状态，导致同步时机退化。

### 5. `find_unused_parameters=True` 不是免费选项

当模型存在条件分支时，用户常打开 `find_unused_parameters=True` 以避免部分参数在某步没有梯度而导致死锁。它能解决正确性问题，但会增加额外跟踪开销。若模型实际上没有未使用参数，长期保留该选项可能带来不必要性能损失。

### 6. 理论通信量正确，不等于实测一定快

同样的 Ring AllReduce 公式，在不同硬件上表现会差很多，因为真实瓶颈还取决于：

| 硬件路径 | 影响 |
|---|---|
| PCIe | 带宽相对有限，跨卡通信更容易吃亏 |
| NVLink / NVSwitch | 设备间带宽更高，更适合高频集体通信 |
| InfiniBand / RoCE | 决定跨节点 AllReduce 的上限 |
| CPU / NUMA 拓扑 | 影响进程绑核、数据搬运和启动开销 |

因此，理论公式负责回答“量级是否合理”，profiling 才负责回答“真实慢在哪里”。

### 7. 该监控什么

最少应观察三类指标：

| 监控项 | 看什么 | 典型工具 |
|---|---|---|
| overlap | backward 和 AllReduce 是否交叠 | PyTorch Profiler, Nsight Systems |
| bucket 分布 | bucket 数量、大小、ready 时机 | PyTorch profiler trace |
| 通信尾巴 | step 末尾是否残留长时间 NCCL 通信 | Nsight Systems, NCCL trace |

如果只看“GPU 利用率”或“step time”，通常无法定位到底是算力、链路还是 bucket 设置出了问题。

---

## 替代方案与适用边界

DDP 不是唯一方案，但在“每卡持有完整模型副本”的同步数据并行场景里，它通常是最稳健、最通用的默认选择。

### 1. 参数服务器

参数服务器适合以下场景：

- 训练规模不大
- 系统历史上已经围绕中心节点构建
- 需要更灵活的参数管理或异步训练语义

它的优点是实现概念直观，调度模型清晰；缺点是扩展性差，尤其在卡数和梯度体量一起上升时，中心节点容易成为瓶颈。

### 2. ZeRO / FSDP

ZeRO 与 FSDP 解决的是另一类问题：当模型参数、梯度、优化器状态过大时，完整副本模式本身已经不可接受。它们通过分片让单卡不再长期持有完整状态，因此在超大模型下比朴素 DDP 更可扩展。

但代价也很明确：通信模式更复杂，不再只是“做一次完整梯度 AllReduce”，而是会涉及：

- `Reduce-Scatter`
- `All-Gather`
- 参数重建
- 状态分片管理

所以它们不是“DDP 的简单加速版”，而是“为了突破显存上限而接受更复杂通信模式”的方案。

### 3. 混合并行

在百亿到千亿参数规模时，真实训练通常是多种并行方式叠加：

- 数据并行
- 张量并行
- 流水并行
- ZeRO/FSDP

原因很直接：仅靠数据并行无法同时解决显存、通信和吞吐三类问题。以 175B 为例，单步梯度就已是数百 GB 量级，若再加上参数本体和优化器状态，单纯依赖完整副本 DDP 往往不现实。

对比表如下：

| 方案 | 每卡是否完整持有模型 | 主要通信模式 | 扩展性 | 主要用途 |
|---|---|---|---|---|
| DDP | 是 | AllReduce 梯度同步 | 高 | 中大规模同步数据并行 |
| 参数服务器 | 通常是 | Worker 与中心节点推拉 | 中到低 | 小规模或历史系统 |
| ZeRO / FSDP | 否，状态或参数分片 | Reduce-Scatter / All-Gather / 重建 | 很高 | 超大模型与显存受限场景 |
| 混合并行 | 否 | 多种通信并存 | 最高，也最复杂 | 超大规模训练 |

可以把适用边界压缩成一句话：

- 模型放得下、希望实现简单、互联较好：优先 DDP。
- 显存已经成为主瓶颈：转向 ZeRO/FSDP。
- 系统必须围绕中心节点设计，或强调异步：才考虑参数服务器。
- 模型规模进入超大区间：通常要采用混合并行，而不是依赖单一方案。

---

## 参考资料

| 资料 | 核心贡献 | 建议阅读方式 |
|---|---|---|
| PyTorch Docs. “Distributed Data Parallel.” https://docs.pytorch.org/docs/stable/notes/ddp.html | 官方解释 DDP 的 Reducer、bucket、autograd hook 与执行流程 | 先读概念图，再对照源码中的 reducer 行为理解 |
| NVIDIA NCCL Documentation. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ | 说明 NCCL 支持的集体通信原语、拓扑与环境变量 | 用于理解 `AllReduce`、`ReduceScatter`、`AllGather` 的底层语义 |
| Sergeev, Alexander, and Mike Del Balso. “Horovod: fast and easy distributed deep learning in TensorFlow.” arXiv, 2018. https://arxiv.org/abs/1802.05799 | 展示 Ring AllReduce 在深度学习训练中的工程价值 | 重点看为什么去中心化同步比参数服务器更适合规模扩展 |
| Li, Mu, et al. “Scaling Distributed Machine Learning with the Parameter Server.” OSDI, 2014. https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu | 参数服务器经典论文，定义了中心化参数管理的设计空间 | 重点看其优势与为何需要分片、层次化扩展 |
| Rajbhandari, Samyam, et al. “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” SC20. https://arxiv.org/abs/1910.02054 | 说明分片优化器状态、梯度和参数如何降低冗余 | 用于建立 DDP 与 ZeRO/FSDP 的边界感 |
| Shoeybi, Mohammad, et al. “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.” arXiv, 2019. https://arxiv.org/abs/1909.08053 | 展示超大模型训练为何需要模型并行与混合并行 | 阅读其并行策略部分即可，不必先陷入实现细节 |
| Narayanan, Deepak, et al. “Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.” SC21. https://arxiv.org/abs/2104.04473 | 系统化讨论大模型训练中的并行组合与通信代价 | 用于把 DDP 放回更大的分布式训练体系中理解 |
| GPT-3 资源与系统课程讲义. https://llmsystem.github.io/llmsystem2026spring/assets/files/llmsys-17-zero-20eb6c8d8c1e7092e1b922abf03d8cdd.pdf | 汇总大模型训练中的参数、梯度、优化器状态与通信账本 | 适合做数量级估算，快速判断何时必须转向分片式方案 |
