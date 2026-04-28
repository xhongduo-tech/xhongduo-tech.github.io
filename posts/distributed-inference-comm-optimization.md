## 核心结论

张量并行推理的核心矛盾不是单卡算力不够，而是多卡把同一层拆开计算后，必须在层内同步出一致结果。这里的“同步”可以白话理解为：每张卡只算了部分答案，但下一步计算需要完整答案，所以要通过集体通信把分散结果合并回来。对长上下文推理，真正拉高尾延迟的常常不是矩阵乘法，而是 `All-Reduce` 这类通信。

对推理优化，最直接的目标只有两类：

1. 减少通信启动次数，也就是少发几次 collective。
2. 把通信时间尽量藏进计算时间里，也就是 overlap compute with comm。

如果用统一符号表示，`B` 是 batch size，`S` 是序列长度，`d` 是 hidden size，`b` 是每个元素的字节数，那么单次需要同步的有效载荷近似是：

$$
payload = B \times S \times d \times b
$$

对 ring `All-Reduce`，单卡通信量常用近似是：

$$
C_{ar} \approx 2 \times \frac{p-1}{p} \times payload
$$

其中 `p` 是张量并行的卡数。Transformer 一层里，attention 和 FFN 通常各有一次同步点，所以单层通信量可粗估为：

$$
C_{layer} \approx 2 \times C_{ar}
$$

再把通信量除以有效带宽 `BW_eff`，就得到时间下限：

$$
T \approx \frac{C}{BW_{eff}}
$$

一个新手最容易建立直觉的说法是：8 张 GPU 分头算一层，算完以后要“拼回一个完整结果”，而且这一层通常不止拼一次。如果拼接速度跟不上 GEMM，GPU 就会长时间空等。

| 场景 | 计算是否容易扩展 | 通信是否容易放大 | 谁更可能成为瓶颈 |
|---|---|---:|---|
| 短序列、较小模型 | 容易 | 不明显 | 计算或 kernel launch |
| 长序列 prefill | 容易 | 很明显 | 通信 |
| decode 单 token | 计算本身小 | 通信 payload 也小 | 更容易受调度和小消息开销影响 |
| TP 卡数继续增大 | 单卡算更少 | 同步点更多、更敏感 | 通信 |

这里要特别指出一个常见误解：如果有人声称在 `B=1, S=32768, d=8192, p=8, b=2` 且仅有 400Gbps 级网络时，单层两次全量 `All-Reduce` 只要约 `1ms/层`，那通常不是这个公式直接算出来的结果，而是以下情况之一：实际同步张量更小、用了分片结果而非全量结果、链路层级更复杂、做了强 overlap，或者测的是不同阶段而不是完整 prefill 路径。

---

## 问题定义与边界

张量并行，英文通常叫 Tensor Parallel，意思是把同一个线性算子的权重或输出维度切到多张卡上并行计算。白话解释是：不是一张卡算一整层，而是多张卡各算这层的一部分。这样做的直接后果是，层内一定会出现同步点。

本文讨论的边界很明确：

1. 只讨论推理，不讨论训练中的反向传播。
2. 只讨论张量并行下的层内通信，不展开数据并行和专家并行。
3. 重点讨论 Transformer 层里的 attention 和 FFN 同步。
4. 重点关注长上下文推理，尤其是 prefill。

必须先区分 `prefill` 和 `decode`。

`prefill` 是把整段 prompt 一次性送进模型，目的是建立 KV cache。白话解释是：先把整篇上下文“读完并记住”。这时 `S` 很大，通信张量通常是大块数据。

`decode` 是每一步只生成一个或少量新 token。白话解释是：模型已经记住前文，现在每次只往后续写一个字。此时 `S` 在算子输入上往往表现为 1 或很小，通信量和 prefill 不是一个数量级。

| 阶段 | 典型 `B` | 典型 `S` | payload 特征 | 延迟特征 |
|---|---:|---:|---|---|
| prefill | 1 到几十 | 几千到几万 | 大消息 | 通信容易主导 |
| decode | 1 到几十 | 1 或很小 | 小消息 | launch 和同步更敏感 |

玩具例子可以这样看。假设只有一层线性层，输出形状是 `[B, S, d]`。如果 `B=1, S=4, d=8`，两张卡各算一半输出，那么每张卡拿到的是 `[1, 4, 4]` 的局部结果。下一步如果需要完整的 `[1, 4, 8]`，就必须通信。这就是“算子被切分，层内必须同步”的最小直觉。

真实工程例子则是 8 卡单机或多机长上下文推理。比如 `TP=8`，模型 hidden size 为 `8192`，prefill 序列长度是 `32768`。这时每层里 attention 和 FFN 的输出同步都会接近百 MiB 到近 GiB 级别的通信动作。此时“网络是否跟得上”直接决定尾延迟，而不是“Tensor Core 是否满载”。

从执行路径看，典型流程可以简化成：

```text
输入张量切分
-> 每卡做本地 attention/FFN 子计算
-> collective 同步局部结果
-> 拿到下一步需要的张量形状
-> 进入下一层
```

所以问题本质不是“有没有通信”，而是“通信次数、通信大小、通信是否能被隐藏”。

---

## 核心机制与推导

先把公式落到数字上。定义：

- `B`：batch size
- `S`：sequence length
- `d`：hidden size
- `b`：每元素字节数，比如 BF16 常取 2
- `p`：张量并行卡数
- `BW_eff`：有效带宽，也就是真正能被应用层吃到的吞吐，不是链路线速

单次同步张量载荷是：

$$
payload = B \times S \times d \times b
$$

对 ring `All-Reduce`，单卡通信量近似：

$$
C_{ar} \approx 2 \times \frac{p-1}{p} \times payload
$$

为什么前面有一个 2？因为 ring `All-Reduce` 通常可分成 reduce-scatter 和 all-gather 两阶段。白话解释是：先把每个人手里的部分数据做分段归并，再把归并后的完整结果重新分发回来。

如果一层里 attention 和 FFN 各有一次同步，则：

$$
C_{layer} \approx 2 \times C_{ar}
$$

时间粗估为：

$$
T \approx \frac{C}{BW_{eff}}
$$

代入题目给出的典型长上下文例子：

- `B=1`
- `S=32768`
- `d=8192`
- `b=2`
- `p=8`
- `BW=400Gbps \approx 50GB/s`

先算 payload：

$$
payload = 1 \times 32768 \times 8192 \times 2 = 536{,}870{,}912\ bytes \approx 512MiB
$$

单次 `All-Reduce`：

$$
C_{ar} \approx 2 \times \frac{7}{8} \times 512MiB \approx 896MiB \approx 0.94GB
$$

若直接按 `50GB/s` 粗估时间下限：

$$
T_{ar} \approx 0.94 / 50 \approx 18.8ms
$$

单层两次同步：

$$
T_{layer} \approx 37.6ms
$$

这说明一个重要事实：在这个参数组合下，如果真的做两次接近全量的 ring `All-Reduce`，单靠 400Gbps 线速很难把单层压到 `1ms` 量级。工程里之所以常见更低数字，通常是因为测量对象不同，或者通信和计算发生了重叠，或者根本没有在每个点都传完整结果。

下面这个表能看出 `p` 增大时的变化。注意，固定 `payload` 时，`2(p-1)/p` 会逐渐逼近 2，也就是单次 `All-Reduce` 通信量不会因为卡更多而线性下降。

| `p` | 系数 $2(p-1)/p$ | 单次 `All-Reduce` 通信量（相对 `payload`） |
|---:|---:|---:|
| 2 | 1.0 | 1.0 倍 |
| 4 | 1.5 | 1.5 倍 |
| 8 | 1.75 | 1.75 倍 |
| 16 | 1.875 | 1.875 倍 |

这也是为什么“继续堆 TP 卡数”不一定带来更低延迟。单卡本地 GEMM 会变小，但层内同步的敏感度会更高。

再给一个玩具例子。设 `B=1, S=8, d=16, b=2, p=4`：

- `payload = 1 × 8 × 16 × 2 = 256 bytes`
- `C_ar ≈ 2 × 3/4 × 256 = 384 bytes`

这时绝对值看起来很小，但如果你每层反复发很多小 collective，真正卡住你的可能不是传 384 bytes 本身，而是每次启动 collective 的固定开销。这就是为什么“小消息场景更怕 launch latency”。

真实工程里，长上下文 prefill 和单 token decode 的瓶颈结构不同：

- prefill 更像“大块数据搬运”，关注总字节量和带宽。
- decode 更像“小包高频请求”，关注启动开销、同步点和流水线。

---

## 代码实现

代码实现的重点不是“有没有 `all_reduce`”，而是“能不能把多个 collective 合并发起，以及能不能把通信放到后台执行”。

先给一个可运行的 Python 小程序，它不依赖 GPU，只是把上面的公式固化成一个检查器，方便做容量估算。

```python
def ring_allreduce_bytes(batch, seq_len, hidden, bytes_per_elem, tp):
    payload = batch * seq_len * hidden * bytes_per_elem
    comm = 2 * (tp - 1) / tp * payload
    return payload, comm

def layer_comm_time_ms(batch, seq_len, hidden, bytes_per_elem, tp, bw_gb_s, collectives_per_layer=2):
    payload, comm = ring_allreduce_bytes(batch, seq_len, hidden, bytes_per_elem, tp)
    total_bytes = comm * collectives_per_layer
    time_ms = total_bytes / (bw_gb_s * 1e9) * 1000
    return payload, comm, total_bytes, time_ms

payload, comm, total_bytes, time_ms = layer_comm_time_ms(
    batch=1, seq_len=32768, hidden=8192, bytes_per_elem=2, tp=8, bw_gb_s=50
)

assert payload == 536_870_912
assert round(comm) == 939_524_096
assert 37.0 < time_ms < 38.5

print(payload, comm, total_bytes, round(time_ms, 2))
```

如果进入实际框架，第一步通常是异步发起通信。`async_op=True` 的白话意思是：先把通信任务交给后台，不要让当前线程立刻卡死等待。

```python
import torch
import torch.distributed as dist

def overlapped_block(x, comm_stream, comp_stream):
    with torch.cuda.stream(comm_stream):
        work = dist.all_reduce(x, async_op=True)

    with torch.cuda.stream(comp_stream):
        y = torch.nn.functional.gelu(x)
        z = y * 2.0

    work.wait()
    return z
```

这段代码表达的不是最佳实现，而是最小思想：通信和计算放到不同 CUDA stream，让它们有机会并发。

更接近工程的伪代码是 group launch。`ncclGroupStart/End` 的白话意思是：把多次 collective 放进一组，一次性交给 NCCL 调度，减少每次启动带来的固定开销。

```text
ncclGroupStart()
all_reduce(tensor_a, stream=comm_stream)
all_reduce(tensor_b, stream=comm_stream)
ncclGroupEnd()

launch_gemm(stream=comp_stream)
launch_activation(stream=comp_stream)
wait_if_next_op_needs_full_result()
```

如果框架支持 `reduce-scatter` 保留分片结果，也常见这种写法：

```text
局部 GEMM
-> reduce-scatter 得到已归约但仍分片的输出
-> 如果下游算子能直接消费分片，就继续本地算
-> 只有在确实需要全量结果时再 all-gather
```

这类做法的本质，是尽量推迟“拿全量张量”的时间点，因为一旦过早追求全量结果，通信压力会立刻放大。

| 做法 | 作用 | 代价 |
|---|---|---|
| `async_op=True` | 让通信异步执行，争取 overlap | 需要自己处理 wait 时机 |
| 独立 CUDA stream | 给通信和计算并发机会 | 有隐式同步时可能失效 |
| `ncclGroupStart/End` | 减少多次 collective 的启动开销 | 不减少 payload |
| `reduce-scatter` 代替直接 `all-reduce` | 保留分片结果，减少不必要的全量物化 | 下游必须能消费分片布局 |
| 算子融合 | 减少中间张量落地和额外同步 | 实现复杂，调试难 |

真实工程例子里，TensorRT-LLM、Megatron 一类系统常见做法就是：通信尽早发起，后续 GEMM、激活、归一化在别的 stream 上推进；能 group 的 collective 尽量 group；能继续使用分片输出的路径，不急着做全量聚合。

---

## 工程权衡与常见坑

第一类坑是把“减少 collective 次数”和“减少字节数”混为一谈。

`group launch` 主要减少的是 launch latency，而不是 payload。也就是说，原来要传 `X + Y` 字节，现在还是传 `X + Y` 字节，只是少了多次启动和调度的碎片开销。对小消息很有效，对超大 payload 不能指望它单独扭转瓶颈。

第二类坑是默认 overlap 一定生效。

不是所有 `async_op=True` 都真的重叠成功。如果后面马上访问通信结果，或者中间有隐式同步，比如某些框架操作默认同步当前流，那么通信仍然会退化成“异步发起，马上等待”，效果接近没有优化。

第三类坑是混淆 prefill 和 decode。

32K prompt 的 prefill 可能每层都在搬大块张量，但 decode 每步只生成一个 token，通信形状完全不同。把 prefill 的大消息开销直接套到 decode，会得到错误结论；反过来用 decode 的轻量路径去估 prefill，也会严重低估尾延迟。

第四类坑是盲目替换 collective 类型。

有人会把 `All-Reduce` 改成 `Reduce-Scatter`，然后宣称通信优化了。问题在于，如果下游仍然必须使用完整张量，那最终还要补一个 `All-Gather`。这时并不是“凭空少传了”，只是把通信重新拆分到别的位置。

| 常见坑 | 错误理解 | 正确理解 | 规避方式 |
|---|---|---|---|
| 混淆 prefill/decode | 两者通信成本差不多 | 张量形状完全不同 | 分阶段单独建模 |
| 误以为少 collective 就等于少字节 | 次数少了，流量也同比下降 | launch 变少不代表 payload 变少 | 同时看次数和字节 |
| 以为 async 一定 overlap | 发起异步就自动隐藏延迟 | 隐式同步会打断重叠 | 用 profiler 看实际时间线 |
| 盲目改 `Reduce-Scatter` | collective 名字变了就是优化 | 若后面仍需全量，可能总量没变 | 先确认下游是否吃分片 |

错误写法和正确写法的对照可以简化成下面这样：

```text
错误写法：
all_reduce(x, async_op=True)
wait()
ffn(x)

正确写法：
all_reduce(x, async_op=True)   # 尽早发起
在独立 stream 上做不依赖 x 完整结果的计算
只在真正需要时 wait()
```

判断优先级时，可以用一个很朴素的标准：如果通信等待时间已经高于主要算子的执行时间，那么先优化通信；否则先优化计算 kernel、内存布局或 batch 策略更划算。

---

## 替代方案与适用边界

不是所有模型都应该优先做 TP 通信优化。是否值得投入，取决于模型大小、上下文长度、并行方式和硬件拓扑。

一个简单判断式是：

$$
\text{若 } T_{comm} > T_{compute} \text{，优先优化通信；否则优先优化计算或调度。}
$$

如果模型本身不大，`S` 也不长，那么通信可能不是主瓶颈。此时更值得做的往往是算子融合、batch 策略调整、KV cache 布局优化。反过来，如果主要业务是长上下文 prefill，TP 通信优化通常最直接。

| 方案 | 解决的主要问题 | 适用场景 | 不适用或收益有限的场景 |
|---|---|---|---|
| 张量并行通信优化 | 层内同步延迟 | 长上下文、TP 较大 | 单卡就能放下模型 |
| Pipeline 并行 | 模型放不下单机或单卡 | 超大模型、跨层切分自然 | 微批太小导致气泡严重 |
| 算子融合 | kernel launch、多次读写内存 | 中短序列、算子链固定 | 通信已绝对主导时 |
| KV cache 优化 | decode 的访存与缓存压力 | 长会话生成 | prefill 主导场景收益有限 |
| 调整切分策略 | 改变局部计算与同步形状 | 模型结构稳定、可重构 | 强依赖现成框架默认实现 |

玩具例子可以这样理解：如果你的模型很小，单卡都能轻松跑，硬要上 `TP=8`，那你可能是在主动引入通信成本。此时最优解通常不是“再继续调 NCCL”，而是直接减少 TP，或者干脆单卡跑。

真实工程例子则更典型：面向企业知识库问答，用户一次输入几万 token 的上下文，系统主要耗时在 prefill。这类场景里，TP 通信优化、算子融合和 KV cache 优化三者并不平权，优先级通常是“先看 TP 通信，再看算子路径”，因为 decode 还没成为主战场。

可以把决策过程简化成伪代码：

```text
if 模型单卡放得下 and S 不长:
    优先减少 TP 或做算子融合
elif 业务以长上下文 prefill 为主:
    优先做 TP 通信次数优化和 overlap
elif 业务以长对话 decode 为主:
    优先看 KV cache、调度、小消息开销
else:
    先 profile，再决定是通信还是计算主导
```

所以，“减少通信次数”和“通信与计算重叠”不是永远正确的第一选择，而是在“层内同步已成为主要等待项”这个边界内，最直接、最稳定的两种优化方向。

---

## 参考资料

| 来源 | 能支持的论点 | 适合放在正文哪一章 |
|---|---|---|
| MegatronLM | 张量并行的基本思想与工程路径 | 问题定义与边界、核心机制与推导 |
| NCCL Group Calls | group launch 减少多次 collective 启动开销 | 代码实现、工程权衡与常见坑 |
| NCCL Tests Performance | 理解带宽、算法和性能测量方式 | 核心机制与推导 |
| PyTorch distributed all_reduce | `all_reduce(async_op=True)` 接口行为 | 代码实现 |
| TensorRT DistCollective | 推理框架中的分布式 collective 语义 | 代码实现、替代方案与适用边界 |
| Megatron Bridge Communication Overlap | overlap 的具体工程实践 | 代码实现、工程权衡与常见坑 |

1. [MegatronLM - NVIDIA ADLR](https://research.nvidia.com/labs/adlr/MegatronLM/)
2. [NCCL Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
3. [NCCL Tests Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
4. [torch.distributed 文档](https://docs.pytorch.org/docs/stable/distributed)
5. [TensorRT DistCollective](https://docs.nvidia.com/deeplearning/tensorrt/10.16.1/_static/operators/DistCollective.html)
6. [Megatron Bridge Communication Overlap](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/communication-overlap.html)
