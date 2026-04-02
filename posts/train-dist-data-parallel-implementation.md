## 核心结论

数据并行（Data Parallelism，简称 DP，白话讲就是“多张卡各自训练同一个模型，只是喂不同的数据”）的核心做法很直接：$N$ 个 worker 各持有一份完整模型副本，每个 worker 处理不同的 mini-batch，先各自完成前向和反向，再把梯度做一次全局同步，最后所有副本用同一份平均梯度更新参数。

这件事真正难的部分不在“复制模型”，而在“同步梯度”。如果同步方式选错，训练会被通信拖死。朴素的 Parameter Server（参数服务器，白话讲就是“所有卡都把梯度发到中心节点，再由中心节点汇总回发”）因为流量集中到一个点，容易形成带宽瓶颈。现代实现更常用 Ring All-Reduce（环形全归约，白话讲就是“每张卡只和邻居通信，边传边累加”），标准过程拆成 Reduce-Scatter 和 All-Gather 两阶段，通信负载均匀分摊到每张卡。

对工程实践来说，可以记住三个结论：

| 结论 | 含义 | 直接影响 |
|---|---|---|
| 每张 GPU 持有完整模型 | DP 不切模型本体 | 模型太大时显存先爆 |
| 每步都要同步梯度 | 同步点在反向之后、更新之前 | 通信性能决定扩展效率 |
| 常配合线性学习率缩放与 warmup | batch 变大时同步调整优化超参 | 不调会导致前期震荡 |

通信时间常用下面的近似模型描述：

$$
T=\alpha\cdot(N-1)+\beta\cdot\frac{2P(N-1)}{N}
$$

其中，$N$ 是 worker 数量，$P$ 是参数量，$\alpha$ 表示每次通信启动延迟，$\beta$ 表示单位数据传输成本。这个式子表达的不是“绝对精确时间”，而是“随着卡数和参数量增长，通信为什么会成为主导因素”。

一个最小玩具例子是 4 张 GPU 训练同一个小模型。每张卡都算出自己的梯度后，不是发给中央服务器，而是把梯度切块，沿着环路传 3 轮。每轮只和相邻节点交换一部分数据，最后每张卡都拿到全局平均梯度，再一起更新。这样单点不会堵车，带宽利用率可以接近 $\frac{N-1}{N}$。

---

## 问题定义与边界

先把问题说清楚。数据并行解决的是“单个模型可以放进每张卡，但一张卡算得太慢，于是希望多张卡同时处理更多数据”的问题。它不解决“模型本身大到单卡放不下”的问题；那已经进入 ZeRO、模型并行、流水并行等范畴。

因此，DP 的边界有两条。

第一条边界是显存边界。因为每张 GPU 都有完整模型副本，所以模型参数、梯度、优化器状态都会在每张卡上占一份空间。只看参数本体时，7B 参数模型在 FP16 下大约要：

$$
7\times 10^9 \times 2\text{ bytes} \approx 14\text{ GB}
$$

这还没算梯度、优化器状态和激活值。也就是说，7B 只是“参数本体约 14GB”，不等于“14GB 显存就能训练”。这就是为什么 DP 更适合 10B 以内或能被单卡轻松容纳的模型，更大模型通常要和 ZeRO 结合。

第二条边界是通信边界。DP 的训练步骤可以概括为：

| 变量/步骤 | 含义 |
|---|---|
| $P$ | 模型参数总量 |
| $N$ | worker 或 GPU 数 |
| local batch | 每张卡自己处理的样本数 |
| global batch | 全部卡的 batch 之和，约为 $N \times$ local batch |
| forward | 前向计算损失 |
| backward | 反向计算本地梯度 |
| all-reduce | 汇总所有卡的梯度 |
| optimizer step | 用全局平均梯度更新参数 |

关键点在于：更新前必须同步。因为每张卡看的数据不同，所以它们算出的梯度也不同。如果不做同步，每张卡就会朝不同方向更新，模型副本很快分叉，训练等价于失败。

对零基础读者来说，可以把它理解成一句话：数据并行不是“多卡各算各的”，而是“多卡先各算各的，再强制对齐结果，再一起走下一步”。

真实工程例子更能看出边界。假设你有 8 张 A100，想训练一个 7B 模型做指令微调。模型本体约 14GB，梯度与优化器状态还会进一步吃显存。如果使用纯 DP，每张卡都得背完整模型和对应训练状态。这时模型虽然“能放进去”，但可能已经把余量压得很紧，batch 稍大就 OOM。这个场景说明：DP 是否可行，不只看“参数能否装下”，还要看训练态显存是否能承受。

---

## 核心机制与推导

DP 的核心机制分成两层：计算层和通信层。

计算层很简单。每张卡拿到同构模型副本和不同数据子集。设第 $k$ 张卡的 mini-batch 是 $B_k$，损失函数是 $L_k(\theta)$，则本地梯度为：

$$
g_k=\nabla_\theta L_k(\theta)
$$

全局目标相当于把所有本地 batch 合在一起，因此理想更新方向是平均梯度：

$$
g=\frac{1}{N}\sum_{k=1}^{N} g_k
$$

这就是 all-reduce 的数学目标。它不是“把别人梯度看一眼”，而是“把所有本地梯度求和再平均，让所有副本得到同一个 $g$”。

通信层才是实现重点。Ring All-Reduce 的标准实现分两阶段。

| 阶段 | 做什么 | 结果 |
|---|---|---|
| Reduce-Scatter | 梯度先分片，再沿环传递并累加 | 每张卡得到某一片的全局和 |
| All-Gather | 已累加完成的分片继续沿环广播 | 每张卡拿回完整全局梯度 |

为什么这比参数服务器更好？因为 Parameter Server 把所有卡的流量打到中心节点，中心节点入带宽和出带宽都会成为瓶颈；而 Ring All-Reduce 中每张卡都只和左右邻居通信，网络压力平均摊开，没有单一热点。

看一个 4 卡玩具例子。把梯度向量切成 4 片：$[c_0,c_1,c_2,c_3]$。

1. Reduce-Scatter 第 1 轮，每张卡把自己的一片发给右邻居，同时从左邻居读一片并相加。
2. 第 2 轮，继续转发已经部分累加的片段，再接收新的片段继续加。
3. 第 3 轮结束后，每张卡恰好持有一个“已经汇总好所有卡贡献”的片段。
4. 然后进入 All-Gather，再传 3 轮，把这 4 个完整片段广播回来。
5. 最终每张卡都得到完整的全局梯度。

因此，4 张卡总共需要 3 轮 Reduce-Scatter 和 3 轮 All-Gather。单卡的总收发量不是“每轮都传整个模型”，而是总计约：

$$
2P\cdot\frac{N-1}{N}
$$

当 $N=4$ 时，就是：

$$
2P\cdot\frac{3}{4}=1.5P
$$

这就是常见说法“Ring All-Reduce 的单卡通信量约为 $2P\frac{N-1}{N}$”。如果有人口头说“每步通信量是 $2P$”，通常是在描述整个标准流程由两部分组成，而不是说单卡每次都真的发送完整两倍参数量。

把这个量代入时间模型，可以得到前文公式：

$$
T=\alpha\cdot(N-1)+\beta\cdot\frac{2P(N-1)}{N}
$$

它传达两个工程事实。

第一，卡数增加不会无限线性加速。因为虽然算力增加了，但同步次数和通信量也在增长，尤其当模型大、互联慢时，通信项会吞掉收益。

第二，参数量越大，DP 越依赖高性能互联。NVLink、InfiniBand、NCCL 优化，本质上都是在压 $\beta$ 这一项。

接着看学习率。全局 batch 从 $B$ 提高到 $kB$ 时，梯度方差通常下降，常见经验规则是线性缩放：

$$
\eta' = k\eta
$$

这里 $\eta$ 是学习率，白话讲就是“每次更新跨出去多大一步”。但线性缩放不是无条件成立。batch 很大时，训练前几步的统计特性不稳定，直接把学习率拉到目标值，容易出现 loss 抖动甚至发散。所以工程上常加 warmup（预热，白话讲就是“前几步先用较小学习率，逐渐升到目标值”）。

例如基线是单卡 batch 512、学习率 $1e{-4}$、warmup 1000 步。扩展到 4 卡，global batch 变成 2048，常见起手式是学习率改成 $4e{-4}$，同时把 warmup 适当拉长，比如 4000 步，避免一开始梯度更新过猛。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，用数组模拟 4 个 worker 的梯度平均、线性学习率缩放和 warmup。它不依赖 GPU，也不依赖 NCCL，但逻辑和真实 DP 是一致的。

```python
from typing import List

def average_gradients(grads: List[List[float]]) -> List[float]:
    n = len(grads)
    m = len(grads[0])
    out = [0.0] * m
    for g in grads:
        assert len(g) == m
        for i, v in enumerate(g):
            out[i] += v
    return [v / n for v in out]

def warmup_lr(base_lr: float, world_size: int, step: int, warmup_steps: int) -> float:
    target_lr = base_lr * world_size  # linear scaling rule
    if warmup_steps <= 0:
        return target_lr
    factor = min((step + 1) / warmup_steps, 1.0)
    return target_lr * factor

# 4 个 worker 的本地梯度
local_grads = [
    [1.0, 2.0, 3.0],
    [3.0, 2.0, 1.0],
    [2.0, 2.0, 2.0],
    [4.0, 0.0, 2.0],
]

global_grad = average_gradients(local_grads)
assert global_grad == [2.5, 1.5, 2.0]

base_lr = 1e-4
world_size = 4

lr_step_0 = warmup_lr(base_lr, world_size, step=0, warmup_steps=1000)
lr_step_999 = warmup_lr(base_lr, world_size, step=999, warmup_steps=1000)

assert abs(lr_step_0 - 4e-7) < 1e-12
assert abs(lr_step_999 - 4e-4) < 1e-12

# 用平均梯度做一步 SGD 更新
params = [10.0, 20.0, 30.0]
updated = [p - lr_step_999 * g for p, g in zip(params, global_grad)]

assert updated[0] < params[0]
assert updated[1] < params[1]
assert updated[2] < params[2]

print("global_grad =", global_grad)
print("lr@step999 =", lr_step_999)
print("updated_params =", updated)
```

上面的 `average_gradients` 就是在模拟 all-reduce 的最终语义：所有 worker 都拿到同一份平均梯度。`warmup_lr` 体现了两个规则：先按 `world_size` 做线性缩放，再在 warmup 阶段渐进拉高。

如果写成更接近真实框架的伪代码，典型 DP 流程如下：

```python
# pseudo code
init_process_group(backend="nccl")
model = build_model().to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

optimizer = AdamW(model.parameters(), lr=base_lr * world_size)

for step, batch in enumerate(loader):
    x, y = batch
    x = x.to(local_rank)
    y = y.to(local_rank)

    lr = warmup_lr(base_lr, world_size, step, warmup_steps)
    set_lr(optimizer, lr)

    optimizer.zero_grad()
    loss = model(x, y)
    loss.backward()

    # DDP 内部会在这里对梯度做 all-reduce / average
    optimizer.step()
```

在 PyTorch DDP 中，开发者通常不会手写 `ncclReduceScatter` 和 `ncclAllGather`。框架已经把这些同步细节封装好了。但理解底层顺序仍然重要：

1. 每张卡各自做 forward/backward。
2. 梯度 ready 后触发通信。
3. 通信本质是 Reduce-Scatter + All-Gather。
4. 所有卡拿到一致梯度。
5. optimizer.step() 在每张卡本地执行，但因为梯度相同、参数初值相同，所以更新结果也相同。

真实工程例子可以看大模型微调。比如 8 张 GPU 对 7B 模型做监督微调，每卡 local batch 很小，例如 1 到 4。此时会同时使用：
- DP 或 DDP 做多卡梯度同步。
- 梯度累积扩大等效 batch。
- warmup 让大 batch 下前期更稳。
- 混合精度降低显存和带宽压力。

这说明“代码实现”不是只会写一个 `all_reduce`，而是把同步、优化器、学习率调度和显存控制组合成完整训练系统。

---

## 工程权衡与常见坑

DP 最大的优点是概念简单、实现成熟、和现有训练框架兼容性最好。只要模型能放进单卡，先上 DDP 往往是成本最低的方案。但它的限制也很直接：显存复制和同步开销都是真成本。

下面是最常见的坑。

| 坑 | 现象 | 规避策略 |
|---|---|---|
| 使用 Parameter Server | 中心节点带宽打满，扩展性差 | 优先用 NCCL Ring All-Reduce |
| 不做 warmup | 大 batch 训练前几步 loss 抖动 | 至少 1000 步 warmup，必要时更长 |
| 只看参数显存，不看训练态显存 | 模型能加载但训练时 OOM | 把梯度、优化器状态、激活值一起算 |
| batch 放大但学习率不调 | 收敛速度变慢 | 先尝试线性缩放 |
| batch 很大仍机械线性缩放 | 发散或泛化变差 | 改更长 warmup、sqrt 缩放或梯度累积 |
| 网络拓扑差 | 多机训练效率异常低 | 优先保证高速互联和 NCCL 配置正确 |

先说 Parameter Server。它的问题不在“能不能跑”，而在“跑大了会不会堵”。可以把它理解成所有人都在往一个收费站汇合，收费站再把车放出去。Ring All-Reduce 则更像环岛通行，每个人只跟相邻车辆交互，流量天然被摊平。训练规模一大，后者通常明显优于前者。

再说 warmup。很多新手知道“batch 变大，学习率也变大”，但忽略了“变大的学习率不能在第 1 步硬上”。这会造成前几步权重更新过大，参数迅速偏离稳定区间。特别是 global batch 超过 4096 时，这个问题经常放大。实践中常见处理是：
- 线性缩放学习率。
- warmup 1000 步起。
- 如果仍不稳，增加 warmup、降低峰值学习率，或改成梯度累积。

显存问题更容易被低估。只看“7B FP16 约 14GB”是远远不够的。训练还会存：
- 梯度。
- 优化器状态，例如 Adam 的一阶、二阶动量。
- 中间激活值。

因此“参数能装下”不等于“训练能跑起来”。很多工程事故就是把推理显存估算误当成训练显存估算。

---

## 替代方案与适用边界

当模型规模继续增大，纯 DP 很快碰到天花板。这时常见替代方案有三类：DP+ZeRO、模型并行、流水并行。

ZeRO（零冗余优化器，白话讲就是“把原本每张卡都重复保存的训练状态拆开分放”）最接近 DP 的思路，因为它不要求你彻底改掉“每卡跑相同计算图”的基本范式，只是把冗余状态逐步切分掉。

| 方案 | 参数是否分片 | 梯度是否分片 | 优化器状态是否分片 | 适用边界 |
|---|---|---|---|---|
| 纯 DP | 否 | 否 | 否 | 模型能轻松放入单卡 |
| DP + ZeRO-1 | 否 | 否 | 是 | 优化器状态太大 |
| DP + ZeRO-2 | 否 | 是 | 是 | 梯度和优化器状态都吃紧 |
| DP + ZeRO-3 | 是 | 是 | 是 | 连参数本体也放不下 |
| 模型并行 | 是 | 部分 | 视实现而定 | 单层或单模块已超单卡容量 |

ZeRO-1、ZeRO-2、ZeRO-3 的区别可以理解成“分片对象越来越多”。先分优化器状态，再分梯度，最后连参数本体也分。对于 10B 以上模型，这通常比纯 DP 更现实。

模型并行则是另一条路。它不是“每卡一份完整模型”，而是“不同卡负责模型的不同部分”。比如一张卡放前几层，另一张卡放后几层。这样能突破单卡显存限制，但代价是前向和反向过程中会更频繁地跨卡传激活和梯度，编程复杂度和调试成本都更高。

所以适用边界可以简化为：

1. 模型能装进单卡，而且你最关心实现稳定和开发效率，优先纯 DP/DDP。
2. 模型接近单卡极限，但还希望保留 DP 风格训练，优先 DP + ZeRO。
3. 模型结构已经明显超单卡承载，才考虑模型并行或流水并行。

换句话说，DP 不是“万能方案”，但它通常是第一方案。只有在显存边界被击穿时，才值得引入更复杂的并行策略。

---

## 参考资料

1. Ilyee, 《Distributed Training》, 2024-03-06. 介绍数据并行、Ring All-Reduce、Reduce-Scatter 与 All-Gather 的基本机制。  
2. ADG CSDN, All-Reduce 时间与通信量公式解析。用于理解 $\frac{N-1}{N}$ 带宽利用率和通信模型。  
3. Liz in Tech, ZeRO 与分布式并行介绍。用于说明 ZeRO-1/2/3 的分片边界。  
4. M. Brenndoerfer, Learning Rate Warmup and Linear Scaling in Large-Batch Training. 用于说明大 batch 下线性缩放与 warmup 的工程实践。  
5. NVIDIA GPU 文档。用于估算不同 GPU 上大模型训练的显存约束与适配边界。
