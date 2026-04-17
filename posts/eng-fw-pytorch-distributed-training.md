## 核心结论

PyTorch 的分布式训练，本质是把“单卡上的一次训练步骤”拆到多个进程和多个设备上并行执行，再用通信原语把这些进程重新拉回同一个参数状态。这里的“通信原语”可以理解为一组底层协作操作，比如求和、广播、聚合，它们负责让多张卡看到一致的训练结果。

对初学者最重要的结论有两条。

第一，主流方案不是 `DataParallel`，而是 `torch.distributed` 配合 `DistributedDataParallel`，简称 DDP。DDP 的基本单位是“单进程单 GPU”，每个进程各自跑一份模型、处理一部分数据，然后在反向传播时同步梯度。梯度就是“损失函数告诉参数该往哪个方向改”的数值信号。

第二，分布式训练并没有改变模型数学本质，改变的是“谁来算、怎么算完后怎样同步”。如果同步做得正确，那么 $N$ 个进程的同步 SGD，可以近似看成把 batch 扩大后的单机训练。其核心公式是：

$$
\bar{g} = \frac{1}{N}\sum_{i=1}^{N} g_i
$$

也就是每个进程先算本地梯度 $g_i$，再通过 `AllReduce` 求和并平均，得到全局梯度 $\bar{g}$，最后每个进程都用同一个 $\bar{g}$ 更新参数，所以模型副本始终一致。

一个最直观的玩具例子是：有 2 张 GPU，GPU0 算出梯度 `[2, 4]`，GPU1 算出梯度 `[4, 6]`。同步后得到平均梯度 `[3, 5]`，两张卡都按 `[3, 5]` 更新。这样两张卡虽然各算各的，但参数始终同步，看起来像一台“虚拟的大显卡”。

单机和 DDP 的主要差别可以先看这张表：

| 场景 | 主要计算瓶颈 | 主要通信瓶颈 | 是否有单点聚合问题 |
| --- | --- | --- | --- |
| 单机单卡 | GPU 计算 | 几乎没有 | 否 |
| 单机多卡 `DataParallel` | 主卡和 CPU 调度 | 卡间复制 | 有 |
| 单机/多机 DDP | 每卡本地计算 | 梯度 AllReduce | 通常没有中心单点 |

---

## 问题定义与边界

分布式训练要解决的核心问题不是“如何把代码跑起来”，而是“多个执行单元怎样在每一步都保持状态一致”。这里的“执行单元”通常是一个进程绑定一张 GPU。状态一致主要包含三件事：模型参数一致、梯度语义一致、数据切分策略一致。

如果只说一句定义，可以这样概括：

PyTorch 分布式训练是在多个进程上执行同一训练程序，通过可控通信保持参数同步，从而把训练吞吐扩展到多卡或多机。

它的边界也要说清楚，否则很容易把问题想混。

| 训练方式 | 通信频率 | 结果一致性 | 故障恢复复杂度 | 本文是否覆盖 |
| --- | --- | --- | --- | --- |
| 非分布式 | 无 | 最高 | 最低 | 是，对照组 |
| 同步分布式训练 | 每步或固定步数同步 | 高 | 中等 | 是，本文重点 |
| 异步训练 | 不固定 | 较弱 | 高 | 否，超出本文边界 |

本文只讨论 PyTorch 中最常见的同步训练路径：`torchrun` 启动，`torch.distributed` 建立进程组，DDP 做梯度同步。这里不展开参数服务器式异步训练，也不讨论推理服务的分布式调度。

一个新手容易混淆的点是：分布式训练不等于模型并行。模型并行是“一个模型拆到多张卡上”；数据并行是“每张卡都有完整模型，只处理不同数据”。DDP 主要解决的是数据并行。FSDP 虽然也属于 PyTorch 分布式体系，但它进一步把模型参数、梯度、优化器状态分片，适合更大模型。

真实工程里，一个典型目标是：2 台机器，每台 4 张卡，总共 8 卡，希望训练行为接近单机 8 卡。要做到这一点，必须保证每张卡在同一步上参与同一轮梯度同步，否则就会出现参数漂移。参数漂移就是不同副本开始更新到不同位置，后续训练不再等价。

---

## 核心机制与推导

DDP 为什么成立，可以从一次训练步骤拆开看。

1. 每个进程读取自己那份 mini-batch。
2. 每个进程执行前向传播，得到损失。
3. 每个进程执行反向传播，得到本地梯度 $g_i$。
4. DDP 在梯度 ready 后触发 `AllReduce`。
5. 所有进程拿到同一个平均梯度 $\bar{g}$。
6. 每个进程各自执行 `optimizer.step()`，参数更新结果一致。

文本图示如下：

`每卡本地前向/反向 -> 梯度进入 bucket -> NCCL AllReduce 求和 -> 按 world_size 平均 -> 每卡本地 optimizer.step()`

这里的 bucket 可以理解为“为了减少通信次数，把多个参数梯度打包成块再同步”的缓存结构。它不是新的数学机制，而是性能优化手段。

设总进程数为 `world_size = N`，第 $i$ 个进程上的局部梯度为 $g_i$，那么同步后：

$$
\bar{g} = \frac{1}{N}\text{AllReduceSum}(g_1, g_2, \dots, g_N)
$$

如果优化器是最基本的 SGD，学习率为 $\eta$，参数更新写成：

$$
\theta_{t+1} = \theta_t - \eta \bar{g}
$$

因为每个进程拿到的 $\bar{g}$ 完全一样，初始参数也一样，所以更新后参数仍然一样。这就是 DDP 能保持副本一致的根本原因。

看一个玩具例子。设 `world_size=2`：

- rank0 的梯度：$g_0 = [2,4]$
- rank1 的梯度：$g_1 = [4,6]$

求和后：

$$
g_0 + g_1 = [6,10]
$$

平均后：

$$
\bar{g} = [3,5]
$$

两个进程都用 `[3,5]` 更新，因此不会分叉。

下面这个可运行的 Python 片段，不依赖 GPU，只模拟 DDP 梯度平均逻辑：

```python
def allreduce_mean(grads):
    world_size = len(grads)
    summed = [sum(values) for values in zip(*grads)]
    return [v / world_size for v in summed]

g0 = [2, 4]
g1 = [4, 6]

g_bar = allreduce_mean([g0, g1])

assert g_bar == [3.0, 5.0]

theta = [10.0, 20.0]
lr = 0.1
new_theta_rank0 = [p - lr * g for p, g in zip(theta, g_bar)]
new_theta_rank1 = [p - lr * g for p, g in zip(theta, g_bar)]

assert new_theta_rank0 == new_theta_rank1
assert new_theta_rank0 == [9.7, 19.5]
print("ddp gradient sync toy example passed")
```

在 PyTorch 真正实现里，`AllReduce` 通常交给 NCCL。NCCL 是 NVIDIA 的集体通信库，可以理解为“专门给多 GPU 做高效同步的底层通道”。它负责 ring、tree 等通信算法的执行细节，PyTorch 上层只需要声明“把这些梯度做 all-reduce”。

`torchrun` 则负责进程编排。它做的不是训练本身，而是把多个训练进程正确拉起来，并注入诸如 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量。没有这层编排，单个 Python 脚本不知道自己是第几个进程，也不知道该和谁建立通信。

---

## 代码实现

一个最小可理解的 DDP 训练脚本，关键步骤只有四步：初始化进程组、绑定设备、包装 DDP、使用 `DistributedSampler`。

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(0, 100, dtype=torch.float32).unsqueeze(1)
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = nn.Linear(1, 1).to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)  # 保证各进程每轮 shuffle 一致但切分不同

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()   # 这里会触发 DDP 的梯度同步
            optimizer.step()

        if dist.get_rank() == 0:
            print(f"epoch={epoch}, loss={loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

启动命令通常是：

```bash
torchrun --nnodes=1 --nproc_per_node=4 train.py
```

如果是两台机器、每台 4 卡，则类似：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  train.py
```

另一台机器把 `--node_rank` 改成 `1`，其他通信参数保持一致。

这里每一步的作用要明确：

| 组件 | 作用 | 为什么不能省 |
| --- | --- | --- |
| `init_process_group` | 建立通信组 | 没有它就没有 rank 间同步 |
| `LOCAL_RANK` | 决定当前进程绑定哪张 GPU | 否则多个进程会抢同一张卡 |
| `DDP(model)` | 注册梯度同步钩子 | 不包裹就只是多进程单卡训练 |
| `DistributedSampler` | 给每个进程分配不同数据切片 | 否则会重复读样本，等效 batch 错乱 |
| `sampler.set_epoch(epoch)` | 保证每轮 shuffle 的随机性在各进程间对齐 | 否则数据顺序可能失配 |

真实工程例子可以看图像分类或 LLM 预训练。比如 2 台机器、每台 4 张 A100，用 `torchrun` 拉起 8 个进程。每个进程读取全局 batch 的一个分片，反向传播后用 NCCL 做梯度 AllReduce。训练中每隔若干步保存 checkpoint。如果其中一台机器临时宕机，作业调度系统重启后可从最近 checkpoint 继续，而不是从头训练几十小时。这时候，分布式训练已经不仅是“多卡提速”，而是“吞吐、恢复、资源编排”三个系统问题一起处理。

常见环境变量如下：

| 环境变量 | 含义 |
| --- | --- |
| `RANK` | 当前进程在全局中的编号 |
| `LOCAL_RANK` | 当前进程在本机中的编号 |
| `WORLD_SIZE` | 总进程数 |
| `MASTER_ADDR` | 主节点地址 |
| `MASTER_PORT` | 主节点端口 |

---

## 工程权衡与常见坑

分布式训练最容易让初学者误判的一点是：代码能启动，不等于训练是对的。很多错误不会直接报错，而是表现为卡死、吞吐异常、loss 不收敛、偶发性超时。

先看一张高频问题表：

| 问题 | 典型症状 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 没配好 `MASTER_ADDR/PORT` | 进程一直等待初始化 | 进程组无法互相发现 | 所有节点统一配置 |
| 多进程绑到同一张卡 | CUDA OOM 或性能极差 | `LOCAL_RANK` 使用错误 | 每进程显式 `set_device` |
| 没用 `DistributedSampler` | 收敛异常、重复样本 | 每个进程都读完整数据 | 必须按 rank 切分数据 |
| 忘了 `sampler.set_epoch(epoch)` | 每轮 shuffle 不合理 | 各进程随机序列长期固定 | 每轮显式设置 |
| NCCL 挂死 | 反向传播卡住 | 通信拓扑或环境变量异常 | 开启 debug，检查网卡与后端 |
| checkpoint 太稀疏 | 故障恢复回退过多 | 恢复点间隔太大 | 按恢复成本设计保存频率 |

`DistributedSampler` 的正确用法通常长这样：

```python
from torch.utils.data import DistributedSampler, DataLoader

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

这里有一个容易忽略的细节：`shuffle=True` 不是让每个进程各自随便打乱，而是让所有进程在同一个 epoch 种子下得到“整体一致、局部分片不同”的打乱结果。否则 rank0 和 rank1 可能拿到不对齐的数据流，虽然不一定立刻报错，但会让整体统计性质发生偏移。

再看 checkpoint 策略。checkpoint 就是“把训练状态落盘的快照”，通常包括模型参数、优化器状态、学习率调度器状态、当前 epoch/step。保存太频繁会拖慢训练，因为磁盘和网络写入都要成本；保存太稀疏则意味着宕机后回退太多步。这个权衡没有统一答案，通常按“恢复可接受损失的时间窗口”设计，例如允许最多回退 10 分钟训练，就按这个窗口定保存频率。

NCCL 相关问题是工程里最难查的一类。因为很多时候不是代码错，而是环境错，比如网卡选择不对、容器权限问题、节点间时钟或路由异常。排查时常见做法是开启：

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

它会输出更细的分布式调试日志，帮助判断卡在初始化、建连、bucket 同步，还是某个 rank 先退出导致其他 rank 一直等。

还有一个常见误区是把 global batch size、local batch size、world size 混为一谈。关系通常是：

$$
\text{global batch size} = \text{local batch size} \times \text{world size} \times \text{gradient accumulation steps}
$$

如果你从单卡迁移到 8 卡，但仍想保持等效训练行为，就要重新审视学习率和 batch 的关系，而不是只把卡数乘上去。

---

## 替代方案与适用边界

不是所有多卡训练都需要 DDP。方案选择要看模型大小、机器规模、通信预算、容错需求。

先看对比表：

| 方案 | 适用规模 | 通信开销 | 跨机能力 | 容错与扩展性 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| `DataParallel` | 小到中等 | 较高，主卡聚合明显 | 弱 | 弱 | 单机试验、快速验证 |
| DDP | 中到大 | 中等，梯度同步 | 强 | 强 | 多卡主流训练 |
| FSDP | 大到超大 | 更复杂，但显存更省 | 强 | 强 | 大模型、显存紧张 |

`DataParallel` 的问题不在于不能用，而在于扩展性差。它采用单进程多线程，主设备容易成为瓶颈。对于单机 2 到 4 卡的小实验，它仍然可以作为低心智成本方案；但一旦进入跨机、多节点、长时间训练，DDP 通常是更稳的默认选择。

FSDP 是在 DDP 基础上进一步走向“大模型”的方案。它把参数、梯度、优化器状态做分片，不要求每张卡完整持有整个模型。直观理解就是：DDP 是“每卡一整份模型”，FSDP 是“每卡只拿模型的一部分，需要时再聚合”。代价是实现和调试都更复杂。

FSDP 的封装形式通常类似：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyLargeModel().to(device)
model = FSDP(model)
```

什么时候该从 DDP 升级到 FSDP？一个实用判断是：如果模型在单卡上已经接近或超过显存上限，而你又不想大幅缩小 batch 或裁模型结构，就该认真评估 FSDP 或 ZeRO 类方案。ZeRO 可以理解为“把训练状态拆散存到不同设备上”，核心目标也是省显存。

另一个替代策略是 gradient accumulation，也就是梯度累积。它不是分布式并行方案，而是“减少同步频率”的折中办法。比如每 4 个 local step 才做一次优化器更新，就能在通信受限时降低同步成本。但它的边界也很明确：不能代替多机扩容，只是用时间换显存和通信压力。

对初学者，一个判断框架就够了：

- 单机 1 到 2 卡、只是验证代码：先用普通训练或 `DataParallel`。
- 单机多卡、正式训练：优先 DDP。
- 多机多卡、需要稳定扩展：用 DDP + `torchrun`。
- 大模型显存不够：考虑 FSDP 或 ZeRO。
- 网络差、通信贵：考虑减小同步频率或做梯度累积，但要重新评估收敛行为。

---

## 参考资料

| 资料 | URL | 用途 | 推荐阅读部分 |
| --- | --- | --- | --- |
| PyTorch Distributed Overview | https://docs.pytorch.org/tutorials/beginner/dist_overview.html | 建立整体认知 | 架构概览、并行方式比较 |
| `torch.distributed` 官方文档 | https://docs.pytorch.org/docs/stable/distributed.html | 查 API 与通信语义 | process group、backend、collective operations |
| DDP Fault Tolerance 教程 | https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html | 理解 checkpoint 与恢复 | snapshot、restart、容错流程 |
| DDP 系列教程 | https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html | 学最小可运行范式 | init、sampler、训练循环 |
| FSDP 文档 | https://docs.pytorch.org/docs/stable/fsdp.html | 了解大模型分片训练 | sharding、state dict、内存权衡 |

如果是第一次系统学习，建议顺序是：

1. 先读 `Distributed Overview`，理解 DDP 在整个分布式版图里的位置。
2. 再读 `torch.distributed` 文档，明确 `rank`、`world_size`、backend、collective 的语义。
3. 然后看 DDP 教程，自己跑通最小脚本。
4. 最后再看 fault tolerance 和 FSDP，进入工程级问题。
