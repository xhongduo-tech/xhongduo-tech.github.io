## 核心结论

虚拟流水线并行的交错调度，本质是把一张 GPU 上原来连续的一大段模型层，再切成 $v$ 个更小的“虚拟 stage”。这里的“虚拟 stage”可以理解成一张卡内部再拆出来的多个执行片段，它们不要求对应连续层号，而是按调度顺序交错执行。

在经典 1F1B 流水线中，若有 $p$ 个物理 stage、$m$ 个 micro-batch，则常见的简化 bubble 比例是：

$$
\text{bubble fraction} \approx \frac{p-1}{m}
$$

引入每卡 $v$ 个虚拟 stage 后，调度粒度更细，简化后可写成：

$$
\text{bubble fraction}_{\text{interleaved}} \approx \frac{p-1}{v \cdot m}
$$

结论很直接：在计算均衡、通信能较好重叠的前提下，$v$ 越大，流水线空转越少，吞吐越高。

对新手可以这样理解：传统流水线让一块 GPU 连续跑一大段层，前向和后向都得等前后卡配合；交错调度把任务拆成多个小 chunk，一张卡完成一个 chunk 就能立刻收发数据，白等时间更少。

但这个结论有边界。交错调度不是“免费加速”，因为它会把层间收发次数放大，通信压力通常接近放大到原来的 $v$ 倍。只有在 NVLink、InfiniBand 这类高带宽互联，且框架能做 `overlap_p2p_comm` 这类通信重叠时，bubble 减少带来的收益才通常大于通信增加的损失。

---

## 问题定义与边界

流水线并行，是把模型层按顺序切到多张卡上，让不同 micro-batch 像工厂流水线一样并行流动。这里的 “micro-batch” 就是把一个大 batch 再切成更小的计算单元，目的是让多张卡同时有活干。

问题在于，标准流水线并行即使使用 1F1B，也仍然有 fill 和 flush 阶段。所谓 bubble，就是某些 GPU 在这些阶段没有可执行任务，只能等待。等待不是小问题，因为总训练时间里这部分是纯损失。

我们关心的不是“能不能并行”，而是“并行以后空转有多少”。交错调度要解决的正是这个问题：在每卡总计算量不变的大前提下，把调度粒度做细，降低空闲比例。

下面这个表先给出最核心的比较。这里的 bubble fraction 是训练步内空转占比的简化估计，适用于 stage 负载较均匀、每个 micro-batch 形状一致、通信未成为主导瓶颈的场景。

| 方案 | 物理 stage 数 $p$ | micro-batch 数 $m$ | 虚拟 stage 数 $v$ | 简化 bubble fraction |
| --- | --- | --- | --- | --- |
| 传统 pipeline | $p$ | $m$ | 1 | $\frac{p-1}{m}$ |
| 交错 pipeline | $p$ | $m$ | $v$ | $\frac{p-1}{v \cdot m}$ |

玩具例子：设 $p=4, m=8$。  
传统流水线的 bubble fraction 是 $3/8=0.375$。  
如果每张卡再切成 $v=2$ 个虚拟 stage，那么变成：

$$
\frac{3}{2 \cdot 8}=0.1875
$$

这表示空转比例近似减半。注意，这里减半的是“等待比例”，不是总训练时间一定减半，因为总时间还受通信、算子效率、显存约束影响。

边界也要说清楚：

| 条件 | 含义 | 不满足时会怎样 |
| --- | --- | --- |
| stage 负载大致均衡 | 每段层的计算时间别差太多 | 某些卡长期拖后腿，bubble 公式失真 |
| micro-batch 足够多 | 流水线要能被填满 | $m$ 太小会导致交错收益很弱 |
| 互联带宽高 | 卡间收发不能太慢 | 通信可能反过来变成主瓶颈 |
| 框架支持重叠 | 计算和收发尽量同时发生 | 多出来的通信直接暴露成额外延迟 |

---

## 核心机制与推导

先看传统流水线。设一个物理 stage 上，前向时间是 $t_f$，后向时间是 $t_b$。在理想均衡情况下，一个 micro-batch 完整经过一个 stage 的主要计算时间近似为 $t_f+t_b$。对于 $p$ 个 stage，流水线填充和排空引入的额外等待大致是：

$$
T_{\text{bubble}} \approx (p-1)(t_f+t_b)
$$

而处理 $m$ 个 micro-batch 的有效主体时间近似是：

$$
T_{\text{work}} \approx m(t_f+t_b)
$$

于是得到常见的近似比例：

$$
\text{bubble fraction} \approx \frac{(p-1)(t_f+t_b)}{m(t_f+t_b)}=\frac{p-1}{m}
$$

交错调度做了什么？它没有改变一张卡负责的总层数，而是把它们拆成 $v$ 个更小 chunk。于是单个 chunk 的前向和后向时间，近似变成：

$$
t_f' \approx \frac{t_f}{v}, \quad t_b' \approx \frac{t_b}{v}
$$

对应的 bubble 时间就近似缩小成：

$$
T_{\text{bubble}}' \approx (p-1)\left(\frac{t_f+t_b}{v}\right)
$$

再除以总主体计算时间，得到：

$$
\text{bubble fraction}_{\text{interleaved}} \approx \frac{p-1}{v \cdot m}
$$

这就是公式来源。它不是拍脑袋，而是“把同一张卡上的工作切得更细，因此每次等待的时间片更短”。

文字示意图如下：

| 时刻 | 标准流水线 | 交错流水线 |
| --- | --- | --- |
| 早期 fill | 后面 stage 大量空闲 | 仍有空闲，但每次空闲片段更短 |
| 中间 steady state | 1F1B 稳态运行 | 更细粒度的 1F1B 交错运行 |
| 后期 flush | 前面 stage 再次空闲 | 仍需 flush，但暴露等待更少 |

新手版理解：原来一张卡必须把“整大段”算完，下一张卡才能继续；现在它只要把“其中一小段”算完就能传给别人，同时自己还能切回另一个 chunk 处理别的 micro-batch，所以等待被打散了。

但代价也同步出现。因为 chunk 变多，边界变多，边界一多，跨卡传输次数就更多。于是：

- 前向激活传输次数变多
- 后向梯度传输次数变多
- 调度表更复杂
- 激活缓存与元数据管理更复杂

因此工程上真正要优化的不是“只开 `v>1`”，而是“让多出来的 P2P 通信尽量和计算重叠”。

真实工程例子：在大模型训练里，Megatron-LM 论文报告交错流水线可带来 10% 以上吞吐提升；但这个提升依赖大规模 GPU 集群和高质量互联。换句话说，同一个调度，在单机 PCIe 环境和多机 InfiniBand 环境，结果可能完全不同。

---

## 代码实现

Megatron-Core 里，流水线并行的核心入口是 `pipeline_parallel.schedules`。官方文档明确给出了两类调度函数：

- `forward_backward_pipelining_without_interleaving`
- `forward_backward_pipelining_with_interleaving`

同时，配置里有一个关键参数：

| 配置项 | 作用 | 典型值 |
| --- | --- | --- |
| `pipeline_model_parallel_size` | 物理 pipeline stage 数 | 2, 4, 8 |
| `virtual_pipeline_model_parallel_size` | 每个物理 stage 内的虚拟 stage 数 $v$ | 2, 4 |
| `overlap_p2p_comm` | 是否把 P2P 收发和计算重叠 | `True` |
| `batch_p2p_comm` | 使用批量 isend/irecv 还是分开发送 | 依实现而定 |
| `microbatch_group_size_per_vp_stage` | 每个虚拟 stage 一次处理的 micro-batch 组大小 | 常按默认或调优设置 |

其核心思想不是“修改模型数学”，而是“修改执行顺序”。

一个新手能看懂的伪代码如下：

```python
# 每个 physical stage 内部有多个 virtual chunks
for micro_batch in micro_batches:
    for virtual_chunk in local_virtual_chunks:
        x = recv_forward_if_needed(micro_batch, virtual_chunk)
        y = forward_chunk(virtual_chunk, x)
        send_forward_if_needed(y, micro_batch, virtual_chunk)

for micro_batch in reversed(micro_batches):
    for virtual_chunk in reversed(local_virtual_chunks):
        dy = recv_backward_if_needed(micro_batch, virtual_chunk)
        dx = backward_chunk(virtual_chunk, dy)
        send_backward_if_needed(dx, micro_batch, virtual_chunk)
```

真正的 Megatron 实现会更复杂，因为它不是“先全前向再全后向”，而是在 warmup 之后进入交错的 1F1B 稳态，并显式调用类似下面这些 P2P 原语：

- `recv_forward`
- `send_forward`
- `recv_backward`
- `send_backward`
- `send_forward_recv_backward`
- `send_backward_recv_forward`

还要跟踪“当前是哪个 virtual pipeline rank 在工作”。这类状态切换的目的，是让同一张 GPU 上不同 chunk 轮流执行，而不是一次把一个 chunk 从头跑到尾。

下面给一个可运行的 Python 玩具代码，只演示 bubble 公式与最基本的收益判断：

```python
def bubble_fraction(p: int, m: int, v: int = 1) -> float:
    assert p >= 1
    assert m >= 1
    assert v >= 1
    return (p - 1) / (v * m)

def should_interleave(p: int, m: int, v: int, comm_penalty_ratio: float) -> bool:
    """
    comm_penalty_ratio: 额外通信开销占一次训练步主体时间的比例估计
    这里只做非常粗糙的工程判断：
    如果 bubble 减少量 > 额外通信比例，就认为值得开交错。
    """
    base = bubble_fraction(p, m, 1)
    interleaved = bubble_fraction(p, m, v)
    saved = base - interleaved
    return saved > comm_penalty_ratio

# 题目中的玩具例子
assert abs(bubble_fraction(4, 8, 1) - 0.375) < 1e-9
assert abs(bubble_fraction(4, 8, 2) - 0.1875) < 1e-9

# 若额外通信只占 10%，则值得交错
assert should_interleave(4, 8, 2, 0.10) is True

# 若网络很差，额外通信占到 25%，则收益可能被吃掉
assert should_interleave(4, 8, 2, 0.25) is False
```

这个代码故意很简化，但它表达了真实工程里的核心判断：交错调度是否有效，不只由公式决定，还由通信代价决定。

真实工程例子：如果一个 48 层 Transformer，采用 `pipeline_model_parallel_size=4`，每卡本来持有 12 层。启用 `virtual_pipeline_model_parallel_size=2` 后，每卡可近似视为两个 6 层 chunk，调度器会在这两个 chunk 之间交替推进不同 micro-batch，并借助 P2P 收发保持流水线连续。

---

## 工程权衡与常见坑

交错调度的理论收益很漂亮，但落地时最常见的问题都在“约束条件没满足”。

| 常见坑 | 现象 | 规避方式 |
| --- | --- | --- |
| `m` 太小 | 流水线填不满，交错收益很弱 | 增大 global batch 或梯度累积步数，提升 micro-batch 数 |
| `m` 与调度不匹配 | 某些 rank 提前闲置或尾部拖长 | 优先让 `m` 满足框架对 PP/VPP 调度的约束 |
| 带宽不足 | 开了交错反而更慢 | 先测 P2P 带宽，低带宽环境保守设置 `v=1` |
| 未启用通信重叠 | 额外收发完全暴露 | 检查 `overlap_p2p_comm`、kernel timeline、NCCL 配置 |
| 层切分不均 | 个别 chunk 明显更重 | 重新分层或使用更灵活的 pipeline layout |
| 激活/缓存管理复杂 | 显存峰值异常、调试困难 | 结合 activation checkpointing 和 profile 工具定位 |

新手常见误区之一，是把“micro-batch 必须可整除 `p`”理解成数学硬约束。更准确地说，很多实现和经验配置都偏好让 `m` 与 pipeline 调度周期对齐，否则吞吐和尾部效率会变差，某些框架下还可能直接不支持你想要的 schedule。工程上最稳妥的做法，是按框架文档和实际 schedule table 去选 `m`，而不是只看总 batch 大小。

例如，`p=3, m=5` 时，理论上不是绝对不能算，但调度通常不规整，steady state 很短，尾部占比偏大。把 `m` 调到 6 或 9，往往更容易得到稳定吞吐。

另一个坑是“只看平均算力，不看互联”。交错调度的收益本质上是在拿“更多通信”换“更少等待”。如果机器之间只有较慢的网络，或者跨节点通信远慢于单节点 NVLink，那么这个交换可能不划算。

---

## 替代方案与适用边界

交错调度不是默认最优，而是特定条件下的优选项。

| 方案 | 适用场景 | 优点 | 限制 |
| --- | --- | --- | --- |
| Interleaved Pipeline | 模型很大、PP 较深、互联快、支持 overlap | bubble 更低，吞吐通常更高 | 通信更频繁，调度更复杂 |
| Traditional Pipeline | 已需要 PP，但网络一般，想先稳妥上线 | 实现成熟，行为更可预测 | bubble 更大 |
| Pure Data Parallel | 单卡能装下模型，或模型不够大 | 实现最简单，扩展直接 | 显存压力大，模型规模受限 |
| Tensor Parallel + 少量 PP | 单节点多卡、高速互联 | 单层内并行强，适合大矩阵 | 跨节点后通信成本可能高 |

可以把选择逻辑概括成三句话：

1. 模型还没大到必须深 PP，优先数据并行或少量张量并行。  
2. 已经必须做深 PP，但网络一般，先用传统 pipeline，`v=1`。  
3. 已经是大规模训练，且具备 NVLink/InfiniBand 和通信重叠能力，再考虑把 `v` 提到 2 或 4。

对新手来说，一个很实用的经验是：先把标准 pipeline 跑稳，再逐步增加 `v`。因为交错调度排查问题更难，性能问题也更依赖 profile。不要一开始就把所有高级开关一起打开，否则你分不清瓶颈到底来自算子、调度还是网络。

---

## 参考资料

- Megatron Core 开发文档，`pipeline_parallel` API：查看交错与非交错调度入口，以及 `recv_forward`、`send_backward` 等 P2P 接口  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html

- Megatron Core `ModelParallelConfig`：查看 `virtual_pipeline_model_parallel_size`、`overlap_p2p_comm`、`microbatch_group_size_per_vp_stage` 等配置项  
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.model_parallel_config.html

- Megatron-LM / Megatron-Core 论文《Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM》：交错流水线的原始论文来源，给出 10%+ 吞吐提升的实验结论  
  https://arxiv.org/abs/2104.04473

- Microsoft Research 论文页面：论文摘要与发表信息，便于快速确认论文背景  
  https://www.microsoft.com/en-us/research/publication/efficient-large-scale-language-model-training-on-gpu-clusters/

- MindSpore Pipeline Parallel 文档：给出 interleaved pipeline 的概念说明，适合理解“非连续层块”的实现思路  
  https://www.mindspore.cn/docs/en/r2.4.0/model_train/parallel/pipeline_parallel.html

- NVIDIA NeMo / Megatron Bridge 通信重叠文档：解释为何虚拟流水线增大后，P2P overlap 变得更重要  
  https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html

- 腾讯云中文解读：适合作为中文辅助材料，帮助对照理解 bubble 与交错调度的直观含义  
  https://cloud.tencent.com/developer/article/2315034
