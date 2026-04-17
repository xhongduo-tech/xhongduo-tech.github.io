## 核心结论

Tensor Parallel，简称 TP，就是把一个大张量按列或按行切到多张 GPU 上分别算。它真正的难点不在“切”，而在“切完之后怎么把各卡结果重新对齐”。这个对齐过程依赖集合通信，也就是多张卡一起参与的一次通信操作。

在单机多卡、固定张量切分、环形拓扑近似成立的前提下，TP 最核心的通信原语可以统一理解为 `AllReduce`，而工程实现里通常进一步拆成 `Reduce-Scatter + AllGather`。`Reduce-Scatter` 的意思是“先把同位置的数据做规约，再把结果分片留在不同卡上”；`AllGather` 的意思是“把各卡手里的分片重新拼回完整张量并发给所有卡”。NCCL 文档明确说明：`ReduceScatter` 后接 `AllGather`，等价于一次 `AllReduce`。

对于 Ring AllReduce，每张 GPU 的总通信量是：

$$
Comm = 2 \times \frac{t-1}{t} \times M
$$

其中，$t$ 是 GPU 数，$M$ 是参与同步的张量总字节数。前半段是 `Reduce-Scatter`，每卡发送/接收约 $\frac{t-1}{t}M$；后半段是 `AllGather`，再发送/接收约 $\frac{t-1}{t}M$。所以总量自然变成两倍。

一个必须记住的数值例子是 4 卡、1 GB 张量。此时每卡通信量为：

$$
2 \times \frac{4-1}{4} \times 1\text{GB}=1.5\text{GB}
$$

也就是每卡先在 `Reduce-Scatter` 阶段传 0.75 GB，再在 `AllGather` 阶段传 0.75 GB。这个量不随具体是哪一层线性层变化，本质上只取决于“这次要同步的张量有多大”和“TP 组有几张卡”。

在高速互联上，TP 才真正划算。以 NVIDIA 官方规格为例，H100 NVL 的 GPU 间 NVLink 带宽是 600 GB/s，HGX H100 8-GPU NVSwitch 互联可到 900 GB/s GPU-to-GPU 带宽。带宽足够高时，TP 的通信可以接近被计算掩盖，多卡训练效率常能维持在 90% 以上；带宽不足时，TP 会迅速从“提升吞吐”变成“卡在同步”。

---

## 问题定义与边界

问题可以表述成一句话：在 TP 里，不同 GPU 各自只持有张量的一部分，怎样在前向和反向的关键边界上，把这些局部结果规约并重新分发，保证每张卡接下来看到的数据是一致的。

这里的“一致”，不是说每张卡永远都要持有完整张量，而是说在需要完整语义的阶段，大家必须得到同一个全局结果。比如某些层按列切分后，前向阶段每卡算出部分输出，就需要做聚合；某些层按行切分后，反向阶段每卡算出部分梯度，也需要做聚合。如果不通信，后续层读取到的就只是局部结果，数学上已经不是原模型。

对零基础读者，可以用“4 个班级传作业总分”的玩具例子理解。每个班级先只算自己那一列题目的分数，这相当于每张 GPU 只算自己负责的张量切片。最后要得到全班总分，就不能只看自己班，必须把各班同学的分数加起来，再把总分发回去。Ring AllReduce 就像班级之间按顺序传纸条：不是所有班都把纸条交给一个老师，而是相邻班级轮流传，边传边累加，最后每个班都拿到完整结果。

本文边界刻意收窄，只讨论这些场景：

| 操作阶段 | 需要同步的张量 | 通信方式 |
|---|---|---|
| 前向 | 需要恢复完整语义的激活或线性层输出分片 | `Reduce-Scatter + AllGather` 或等价 `AllReduce` |
| 反向 | 需要跨卡求和的输入梯度或权重梯度分片 | `Reduce-Scatter + AllGather` 或等价 `AllReduce` |
| 优化器前 | 已切分参数对应的梯度分片 | 常见为 `Reduce-Scatter` 后本地更新，或先聚合再更新 |

本文不讨论 ZeRO、Pipeline Parallel、跨节点 InfiniBand 分层通信，也不讨论 MoE 的 `AllToAll`。原因很简单：这些机制一旦混进来，通信图就不再是“单一 TP 环”，分析会混到多个维度，初学者很容易把问题看散。

---

## 核心机制与推导

先从最小机制开始。设有 $t$ 张 GPU，一次要同步的张量总大小为 $M$。Ring AllReduce 之所以适合 TP，是因为它没有“中心节点”。没有一张卡要负责收完所有数据再发回去，所以不会出现单点瓶颈。

### 1. 为什么可以拆成 `Reduce-Scatter + AllGather`

`Reduce-Scatter` 做两件事：

1. 把各卡同位置的数据做规约，比如求和。
2. 把规约后的结果分片散到各卡。

这一步结束后，每张卡只拿到“完整结果中的一部分”，但那一部分已经是全局求和后的正确值。

`AllGather` 再做一件事：

1. 把每张卡那一部分结果收集起来，拼回完整张量，并让所有卡都拿到。

所以两步连起来，等价于“全局求和后，每张卡都得到完整结果”，这就是 `AllReduce` 的语义。

### 2. 通信量为什么是 $\frac{t-1}{t}M$

把张量均匀切成 $t$ 片，每片大小为 $\frac{M}{t}$。

在 `Reduce-Scatter` 中，环一共跑 $t-1$ 轮。每一轮，每张卡只发送一片大小的数据，也就是 $\frac{M}{t}$。因此每卡总发送量是：

$$
\frac{M}{t} \times (t-1)=\frac{t-1}{t}M
$$

`AllGather` 同理，也是 $t-1$ 轮，每轮传一片：

$$
\frac{M}{t} \times (t-1)=\frac{t-1}{t}M
$$

两段相加：

$$
Comm = \frac{t-1}{t}M+\frac{t-1}{t}M
=2\times\frac{t-1}{t}M
$$

这就是 Ring AllReduce 每卡总通信量。

### 3. 3 卡玩具例子：数据怎么在环上走

设有 3 张卡：G0、G1、G2。每张卡上的向量都分成 3 段：`a0 a1 a2`、`b0 b1 b2`、`c0 c1 c2`。目标是每一段都做求和，得到：

- 第 0 段：`a0+b0+c0`
- 第 1 段：`a1+b1+c1`
- 第 2 段：`a2+b2+c2`

环可以画成这样：

`G0 -> G1 -> G2 -> G0`

在 `Reduce-Scatter` 阶段：

- 第 1 轮：G0 把一段发给 G1，G1 把一段发给 G2，G2 把一段发给 G0。收到的人立即做加法。
- 第 2 轮：继续把已经部分累加过的段往前传，再累加一次。

两轮结束后：
- G0 手里保留其中一段的最终和
- G1 手里保留另一段的最终和
- G2 手里保留最后一段的最终和

接着进入 `AllGather`：

- 第 1 轮：每张卡把自己已经算好的那一段传给下一个邻居
- 第 2 轮：继续轮转

最后三张卡都拥有三段完整结果。

这个过程有一个很重要的工程含义：每张卡始终只在发“一个分片”，而不是发完整张量，所以链路可以持续流动，带宽利用率高。

### 4. 4 卡、1 GB 的数值例子

现在看常用的 4 卡例子。总张量大小 $M=1$ GB，卡数 $t=4$。

每卡通信量：

$$
2\times\frac{4-1}{4}\times 1=1.5\text{ GB}
$$

拆开看：

- `Reduce-Scatter`：每卡 0.75 GB
- `AllGather`：每卡 0.75 GB

如果这 1 GB 是某层前向输出，4 张卡各自只算了 1/4 的局部贡献，那么 `Reduce-Scatter` 先把局部贡献求和并分片保留，`AllGather` 再重组成完整输出，让下一层的每张卡都能继续按正确语义计算。

这里要强调一个准确性边界：在真实 TP 实现中，不是每一层前向和反向都一定显式调用完整 `AllReduce`。很多框架会根据层的切分方向，把同步点落成 `all-gather`、`reduce-scatter` 或延迟到后续边界再做。但从“这一段必须完成全局规约并让后续看到一致结果”的抽象上看，用 `AllReduce = Reduce-Scatter + AllGather` 来分析通信量是成立的。

---

## 代码实现

代码层面要抓住两件事：一是通信顺序，二是张量尺寸。对初学者来说，可以先写一个能跑的“环形规约模拟器”，再看 NCCL 风格伪代码。

先给出一个可运行的 Python 玩具实现。它不依赖 GPU，只是在 CPU 上模拟 `Reduce-Scatter + AllGather` 后的结果与直接求和是否一致。

```python
from typing import List

def chunk(xs: List[float], parts: int) -> List[List[float]]:
    assert len(xs) % parts == 0
    size = len(xs) // parts
    return [xs[i * size:(i + 1) * size] for i in range(parts)]

def add_vec(a: List[float], b: List[float]) -> List[float]:
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]

def ring_allreduce_sim(inputs: List[List[float]]) -> List[List[float]]:
    # 输入：每个 rank 一份完整向量，长度可被 rank 数整除
    ranks = len(inputs)
    n = len(inputs[0])
    assert ranks > 1
    assert all(len(x) == n for x in inputs)
    assert n % ranks == 0

    # 直接算“正确答案”
    total = [0.0] * n
    for vec in inputs:
        total = add_vec(total, vec)

    # 用 Reduce-Scatter 的最终语义模拟：
    # 第 r 张卡保留 total 的第 r 个分片
    total_chunks = chunk(total, ranks)
    scattered = [total_chunks[r] for r in range(ranks)]

    # 再用 AllGather 的最终语义模拟：
    # 每张卡拿回全部分片
    gathered = []
    for _ in range(ranks):
        full = []
        for part in total_chunks:
            full.extend(part)
        gathered.append(full)

    return gathered

inputs = [
    [1, 2, 3, 4, 5, 6],
    [10, 20, 30, 40, 50, 60],
    [100, 200, 300, 400, 500, 600],
]

outputs = ring_allreduce_sim(inputs)
expected = [[111, 222, 333, 444, 555, 666]] * 3

assert outputs == expected
print(outputs[0])
```

这段代码没有真的逐轮传递数据，但它验证了一个核心事实：`Reduce-Scatter` 的最终保留结果加上 `AllGather` 的最终重组结果，与直接全量求和完全一致。

再看更接近训练框架的 NCCL 风格伪代码。这里“组调用”的意思是把多个通信操作打包提交，减少调度开销。

```python
# 假设：
# grad_full: 本层逻辑上的完整梯度，大小为 M
# grad_shard: 当前 rank 负责的梯度分片，大小为 M / tp_world_size
# full_grad_buffer: 重组后的完整梯度缓冲区，大小为 M

with nccl_group():
    # 1. 先把各卡的局部梯度做规约，并把结果按 rank 分片留下
    nccl_reduce_scatter(
        sendbuf=grad_full,
        recvbuf=grad_shard,
        op="sum"
    )

    # 2. 再把每张卡手里的正确分片收集起来，恢复完整梯度
    nccl_all_gather(
        sendbuf=grad_shard,
        recvbuf=full_grad_buffer
    )

optimizer.step()
```

这段伪代码表达的是“完整 AllReduce 语义”。在真实 TP 实现里，常见写法会更细：

- 列并行线性层前向后，若下游需要完整激活，则做聚合或 gather。
- 行并行线性层反向后，若梯度需要全局求和，则做 reduce 或 reduce-scatter。
- 若优化器支持分片状态，则可能在 `reduce-scatter` 后直接本地更新，不再立即 `all-gather`。

真实工程例子可以看大模型训练中的张量并行线性层。假设隐藏维度 16384，TP=8，那么一次激活或梯度同步的数据量很容易达到 GB 级。如果每个训练 step 有多次这样的同步，通信路径是否落在同一 NVLink/NVSwitch 域，就会直接决定吞吐。

---

## 工程权衡与常见坑

TP 的理论公式很干净，但落到机器上，真正决定效果的是拓扑，也就是 GPU 之间到底怎么连。

先给出一个工程判断表：

| 拓扑类型 | 实际带宽量级 | 通信惩罚 | 建议措施 |
|---|---:|---|---|
| 同一 NVSwitch 域的 HGX H100 8 卡 | GPU-to-GPU 可到 900 GB/s | 最低，适合高频 TP 通信 | 优先把 TP 组固定在同一 NVSwitch 域 |
| H100 NVL 直连对 | 单对 GPU 间 NVLink 600 GB/s | 低，但跨对后可能退化 | 让强耦合 TP rank 落在直连或同域路径 |
| 纯 PCIe Gen5 | 约 128 GB/s 理论，常见有效带宽更低 | 明显变慢，容易拖累 step time | 降低 TP 度，增加 DP/PP 占比 |
| 跨 NUMA/跨交换芯片绕行 | 取决于路径，波动大 | 延迟抖动明显 | 绑定进程、检查 rank 映射与 NUMA 拓扑 |

第一个坑是“只看 GPU 算力，不看 GPU 互联”。TP 不是算完再偶尔同步一次，而是层间高频同步。链路一慢，问题不是多花一点时间，而是整个训练流水线被通信截断。

第二个坑是“TP rank 映射错位”。同样是 8 卡，若 rank 顺序没有贴合物理拓扑，逻辑上的相邻 rank 可能对应物理上的远路径。Ring AllReduce 会沿着这个 rank 顺序传数据，于是本来应该走高速直连的流量，被迫走慢路径。

第三个坑是“通信与计算抢 SM 资源”。SM 就是 GPU 上真正执行计算线程的核心资源。NCCL 的集合通信虽然已经尽量低占用，但在高频 TP 中，通信 kernel 仍会与矩阵乘法争抢调度。NVIDIA 的 NCCL 技术博客提到，其实现目标之一就是用较低 occupancy 跑满带宽，把更多 SM 留给计算。实际工程里，用户缓冲区注册、合并多个 collective、减少碎片化小包，都能降低这类冲突。

第四个坑是“误把小张量经验套到大模型”。小张量下延迟占主导，大张量下带宽占主导。TP 同步通常是后者，所以要重点看 `GB/s`，不是只看单次调用的毫秒数。

用一个真实工程例子说明。假设在 H100 NVL 8 卡节点上，某次前向或反向要同步 1.5 GB 数据，若理想链路带宽按 600 GB/s 粗算，则纯带宽时间约为：

$$
1.5 / 600 \approx 0.0025\text{ s}
$$

也就是约 2.5 ms。这个量级在大型 GEMM 前后通常还能接受，尤其当通信与部分计算重叠时，整体扩展效率仍可能保持在 90% 以上。反过来，如果同样的数据落到更慢的 PCIe 路径上，通信时间会成倍上升，TP 的收益就会被迅速吞掉。

所以工程上常见结论是：TP 不是“能开就开”，而是“互联够强才开到高并行度”。

---

## 替代方案与适用边界

Ring AllReduce 不是唯一方案，只是在单机同构多卡里最常见、最稳定。

可以把几种方案用一个表对比：

| 方案 | 通信流程 | 数据大小变化 | 适用边界 |
|---|---|---|---|
| Ring AllReduce | 邻居之间轮转传递，先 `Reduce-Scatter` 再 `AllGather` | 每轮传一片，总量约 `2(t-1)/t×M` | 单机多卡、带宽高、rank 同构 |
| Hierarchical AllReduce | 先组内汇总，再组间汇总，最后组内广播 | 组内和组间分层，跨慢链路数据更少 | 多交换域、跨节点、分层拓扑明显 |
| Parameter Server | 所有 worker 向中心节点上传，再由中心下发 | 中心节点承压，容易成瓶颈 | 异构环境、容错优先、带宽集中管理 |
| Gradient Compression | 先压缩梯度，再通信，再解压 | 数据量下降，但有额外误差和算子成本 | 带宽极差、允许近似训练 |

对新手可以用一句话区分：

- Ring：轮流传纸条。
- Hierarchical：先小组长汇总，再年级组长汇总。
- Parameter Server：所有人都交给一个老师统一处理。

为什么 TP 通常偏爱 Ring？因为 TP 的同步很频繁，而且张量往往大、结构规则，最怕中心瓶颈。Ring 正好把流量均摊到所有卡上。为什么有时要换成 Hierarchical？因为物理拓扑本来就分层，比如 8 卡里 4 卡一组更快、组间更慢，这时先组内汇总能减少慢链路压力。

TP 的适用边界也要说清楚。它最适合：

- 单层参数量非常大，单卡放不下或单卡 GEMM 吞吐太低。
- GPU 间有高带宽互联，比如 NVLink、NVSwitch。
- 模型结构规则，张量切分后通信模式稳定。

它不适合：

- 链路慢，通信明显压过计算。
- 模型层很碎，小算子很多，通信调度开销占比高。
- 需要跨大量节点且拓扑复杂，此时往往应降低 TP 度，转向更高比例的数据并行、流水线并行，或采用 2D/3D 混合并行。

一个常见误区是“TP 越大越好”。实际上，TP 度增加时，每卡计算量下降，但同步次数和链路压力不会同步按比例下降。到某个点以后，继续加卡只会把系统推向“通信受限”。这就是为什么很多训练系统会把 TP 限制在单节点内，再通过 DP 或 PP 扩展到更多节点。

---

## 参考资料

| 来源 | 关注点 | 用途 |
|---|---|---|
| [NCCL Collective Operations 文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2287/user-guide/docs/usage/collectives.html) | `AllReduce`、`ReduceScatter`、`AllGather` 的定义与等价关系 | 验证“`ReduceScatter + AllGather = AllReduce`”这一核心机制 |
| [NVIDIA Technical Blog: Fast Multi-GPU collectives with NCCL](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/) | NCCL 如何通过环形通信和低占用 kernel 提高带宽利用率 | 理解为什么 Ring 在单机多卡里常见且有效 |
| [NVIDIA H100 官方规格页](https://www.nvidia.com/en-us/data-center/h100/) | H100 NVL 的 600 GB/s NVLink、H100 SXM 的 900 GB/s 互联规格 | 核对文中带宽数字与拓扑差异 |
| [NVIDIA HGX Platform 官方页](https://www.nvidia.com/en-us/data-center/hgx) | HGX H100 8-GPU NVSwitch GPU-to-GPU 带宽 | 说明单节点高带宽互联为何适合 TP |
| [NCCL 开发者文档归档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_242/nccl-developer-guide/docs/usage/operations.html) | `ReduceScatter` 与 `AllGather` 的原始语义说明 | 交叉核对概念，避免混淆“规约”和“广播” |
| [多 GPU 通信课程资料](https://gpu.mulf.net/course/11-multi-gpu.html?utm_source=openai) | Ring AllReduce 通信量公式与示意 | 辅助理解每卡通信量 `2(t-1)/t×M` 的推导 |
| [TP/集合通信讲解文章](https://www.cnblogs.com/t-bar/p/19652224?utm_source=openai) | 面向初学者的 TP 与集合通信直观解释 | 补充“为什么前后向都需要同步”的叙述角度 |
