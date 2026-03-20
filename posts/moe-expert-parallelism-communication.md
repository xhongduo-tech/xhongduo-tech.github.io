## 核心结论

MoE 的专家并行，本质是“权重分片，激活搬运”。专家并行指不同 GPU 只保存部分专家参数，而不是每张卡都保存全部专家。这样能把模型容量做大，但代价是 token 必须在路由前后跨设备移动。

对一个 MoE 层来说，前向至少包含两次 All-to-All。All-to-All 是一种“每个设备都向其他所有设备发送不同数据”的集体通信。第一次把 token 发到目标专家所在设备，通常叫 dispatch；第二次把专家输出送回 token 原始设备，通常叫 gather 或 combine。反向传播还会再发生对应通信，因此吞吐率常常先被网络而不是算力卡住。

通信量的第一近似可以写成：

$$
V_{\text{layer}} = 2 \times B \times S \times K \times D
$$

其中 $B$ 是 batch size，$S$ 是序列长度，$K$ 是 top-k，表示每个 token 激活多少个专家，$D$ 是隐层维度。这个式子说明：batch 更大、上下文更长、top-k 更高、hidden size 更宽，都会线性放大专家并行的通信成本。

如果一个专家并行组有 $P$ 张设备，并且路由足够均匀，那么单设备平均收发量大约是总量按 $P$ 分摊，即数量级为 $O(B \cdot S \cdot K \cdot D / P)$。这也是“专家数能扩展，但网络也必须一起扩展”的核心原因。

| 通信/计算环节 | 时间占比近似 | 说明 |
| --- | --- | --- |
| All-to-All（dispatch/gather） | 30%–40% | 前向和反向都会发生，常成为 MoE 迭代瓶颈 |
| 其他计算（FFN/GEMM） | 60%–70% | 可通过 Grouped GEMM、FP8、融合核提高利用率 |

---

## 问题定义与边界

先把问题说清楚。这里讨论的是“专家并行”，不是“数据并行”。数据并行是每张卡各算一份不同样本，最后同步梯度；专家并行是每张卡只存一部分专家，token 需要去找对应专家。两者可以同时存在，而且互不替代。

MoE 中的 router，也叫路由器，可以理解成“给 token 分配专家的打分器”。它为每个 token 选出 top-k 个专家。只要这些专家不在当前设备，token 就必须发出去。于是通信不是训练结束时才发生，而是进入每个 MoE 层时就发生。

边界主要有三类：

| 变量 | 变大后直接影响什么 | 为什么 |
| --- | --- | --- |
| $B$ | 通信总量增大 | token 总数更多 |
| $S$ | 通信总量增大 | 每个样本的 token 更多 |
| $K$ | 通信和专家计算都增大 | 一个 token 要去更多专家 |
| $D$ | 单 token 载荷变大 | 每次搬运的向量更宽 |
| $P$ | 单卡负载可能下降，但协调更复杂 | 卡更多后跨设备路径更长、同步更难 |

玩具例子：假设 4 张 GPU、8 个专家、每张卡放 2 个专家，top-2 路由。一个 token 如果被分到专家 1 和专家 6，那么它会同时去 GPU0 和 GPU3。专家算完后，两份结果再回到原始 GPU 做加权合并。这就是“发出去一次，再收回来一次”的两趟通信。

真实工程例子：8 张 GPU 组成一个专家并行组，每张卡放 16 个专家。一个 batch 里有 $32 \times 2048=65536$ 个 token，若 top-k=2，则会产生 131072 个 token-expert 对。即使路由均匀，所有设备也必须同时处理自己发出的流量和别人发来的流量；只要其中一张卡接收过载，整个 step 都要等它。

---

## 核心机制与推导

MoE 专家并行可以拆成四步：

1. 本地路由：router 为本地 token 选择 top-k 专家。
2. 第一次 All-to-All：把 token 发到目标专家所在设备。
3. 本地专家计算：每个设备只对自己持有的专家做 FFN。
4. 第二次 All-to-All：把结果送回 token 原始设备并合并。

为什么通信量是上面的公式？因为每个 token 会被复制到 $K$ 个专家，每份拷贝都是长度为 $D$ 的向量。本层前向需要发出去一次、拉回来一次，所以是 2 倍：

$$
V_{\text{layer}} = 2 \times (B \times S) \times K \times D
$$

这个式子按“元素个数”计。如果用 BF16，每个元素通常占 2 字节，则字节量约为：

$$
\text{Bytes}_{\text{layer}} \approx 2 \times B \times S \times K \times D \times 2
$$

带入常见数字：

| 变量 | 数值 | 含义 |
| --- | --- | --- |
| $B$ | 32 | batch size |
| $S$ | 2048 | 序列长度 |
| $K$ | 2 | top-k |
| $D$ | 4096 | 隐层维度 |
| $V_{\text{layer}}$ | $5.37 \times 10^8$ 元素 | 单层两次 All-to-All 的总元素数 |

如果按 BF16 估算，单层前向通信字节数约为：

$$
5.37 \times 10^8 \times 2 \approx 1.07 \text{ GB}
$$

若一个模型在前向中有 16 个 MoE 层，总前向通信量就接近 17 GB；把反向也算上，训练一步可到 30 GB 量级。这也是工程上经常看到 All-to-All 吃掉大量迭代时间的原因。

再看单设备负载。若专家并行组有 $P$ 张卡，且路由近似均匀，则每张卡平均收发量大致是：

$$
V_{\text{per-device}} \approx O(B \cdot S \cdot K \cdot D / P)
$$

这里的“均匀”是理想条件。实际路由会偏斜，某些专家更热门，于是单卡流量会高于平均值，形成 straggler。straggler 可以理解成“最慢的那张卡拖住整个同步点”。

---

## 代码实现

下面的代码不依赖分布式库，只做通信量估算和简单路由模拟。它能直接运行，并用 `assert` 保证结果符合推导。

```python
from collections import defaultdict

def moe_comm_elements(batch_size: int, seq_len: int, top_k: int, hidden_dim: int) -> int:
    # 两次 All-to-All：dispatch + gather
    return 2 * batch_size * seq_len * top_k * hidden_dim

def moe_comm_bytes_bf16(batch_size: int, seq_len: int, top_k: int, hidden_dim: int) -> int:
    return moe_comm_elements(batch_size, seq_len, top_k, hidden_dim) * 2  # bf16 = 2 bytes

def route_to_devices(expert_ids, experts_per_device):
    device_buckets = defaultdict(list)
    for token_id, experts in enumerate(expert_ids):
        for expert_id in experts:
            device_id = expert_id // experts_per_device
            device_buckets[device_id].append((token_id, expert_id))
    return dict(device_buckets)

# 公式校验
elements = moe_comm_elements(batch_size=32, seq_len=2048, top_k=2, hidden_dim=4096)
assert elements == 536870912

bytes_bf16 = moe_comm_bytes_bf16(batch_size=32, seq_len=2048, top_k=2, hidden_dim=4096)
assert bytes_bf16 == 1073741824  # 约 1 GiB / MoE layer / forward

# 玩具路由例子：4 卡 8 专家，每卡 2 专家
expert_ids = [
    [1, 6],  # token 0 -> device 0 and 3
    [2, 3],  # token 1 -> device 1 and 1
    [0, 7],  # token 2 -> device 0 and 3
]
buckets = route_to_devices(expert_ids, experts_per_device=2)

assert set(buckets.keys()) == {0, 1, 3}
assert len(buckets[0]) == 2
assert len(buckets[1]) == 2
assert len(buckets[3]) == 2

print("per-layer forward bytes (BF16):", bytes_bf16)
print("dispatch buckets:", buckets)
```

真正工程代码的顺序通常就是下面这样：

```python
tokens, routing = router(hidden_states)          # 选 top-k 专家
dispatched = all_to_all_dispatch(tokens, routing)
expert_out = grouped_gemm(dispatched)            # 本地专家计算
combined = all_to_all_gather(expert_out, routing)
```

这里 `grouped_gemm` 的意思是“把多个小矩阵乘法合并成更少的大调用”，白话讲就是减少零碎 kernel，提升 GPU 利用率。Megatron Core 里常见的优化开关包括 `--overlap-moe-expert-parallel-comm`、`--moe-grouped-gemm`、`--moe-router-fusion`、`--moe-permute-fusion`。它们的共同目标不是减少理论通信量，而是减少等待和调度开销。

---

## 工程权衡与常见坑

第一个坑是把“平均流量”当成“真实流量”。公式给的是平均量级，但系统真正卡住的往往是最忙的那台设备。只要热门专家集中在少数 GPU，其他设备即使空闲，也不能让 step 提前结束。

第二个坑是只看带宽，不看同步。All-to-All 不只是搬数据，还要求所有参与设备在正确时间准备好发送和接收缓冲区。于是小批次、碎片化消息、频繁 kernel launch，都会让链路利用率远低于理论峰值。

第三个坑是忽视专家并行和其他并行维度的组合关系。数据并行的梯度同步通常是 All-Reduce 或 Reduce-Scatter；专家并行的 token 交换通常是 All-to-All。两者正交，但不会自动最优。如果 TP、PP、EP 分组和硬件拓扑不匹配，就会把本来该走 NVLink 的流量打到更慢的跨节点网络上。

真实工程例子：在 Megatron Core 的大规模 MoE 训练中，EP All-to-All 常被直接列为通信瓶颈，官方建议对它做 overlap，并配合延迟权重梯度计算、shared expert overlap、router/permute fusion 一起使用。这说明问题不在“有没有通信”，而在“能不能把通信藏在计算后面”。

| 优化 | 作用 | 配置示例 |
| --- | --- | --- |
| EP A2A Overlap | 让通信和计算重叠 | `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` |
| Shared Expert Overlap | 共享专家与 token 交换并行 | `--moe-shared-expert-overlap` |
| Grouped GEMM | 合并小矩阵乘法 | `--moe-grouped-gemm` |
| Router/Permute Fusion | 减少小核与同步点 | `--moe-router-fusion`、`--moe-permute-fusion` |

---

## 替代方案与适用边界

如果集群内通信很强，例如单机多卡 NVLink 充足，标准专家并行通常已经够用。此时重点是把 dispatch/gather 与专家计算重叠，而不是重写路由策略。

如果跨节点网络更慢，问题就变成“尽量少做远程 All-to-All”。一种思路是分层通信，也就是先做机内，再做机间。白话讲，就是先把本地能合并的 token 合并好，再把真正需要远程发送的部分发出去。

另一类方法是 Occult 这类协作通信优化。它的核心思想不是改变 MoE 定义，而是提高“共同被激活的专家”在同设备上的概率，从而减少跨设备传输。适合大规模集群、top-2 或更高路由、并且专家协同模式比较稳定的场景。

还有一些方法会限制路由目标空间，例如 node-limited routing 或 collaboration-constrained routing。它们本质上是在模型自由度和通信成本之间做交换：路由不再完全自由，但网络压力更可控。

| 策略 | 优势 | 适用情况 |
| --- | --- | --- |
| 标准 EP + overlap | 实现成熟，兼容主流框架 | 单机或高速互联集群 |
| 分层 All-to-All | 减少跨节点远程流量 | 多机训练、机内带宽高于机间 |
| Occult/协作通信 | 降低跨设备协作成本 | 大规模 top-k 路由、热点明显 |
| 受限路由 | 稳定通信边界 | 网络受限、延迟敏感训练 |

所以适用边界可以一句话总结：当网络不是主要瓶颈时，标准专家并行最直接；当 All-to-All 已经主导 step 时间时，下一步不是继续堆专家，而是重做拓扑映射、通信分层和路由约束。

---

## 参考资料

1. NVIDIA Megatron Core MoE 指南，重点看专家并行、通信重叠、Grouped GEMM、Router/Permute Fusion：<https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html>
2. NVIDIA Megatron Core MoE API 文档，重点看 EP 与 DP/TP/PP/CP 的组合，以及 shared expert overlap：<https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html>
3. Michael Brenndoerfer 对 Expert Parallelism 的推导，重点看两次 All-to-All 和公式 $V_{\text{layer}} = 2BSKD$：<https://mbrenndoerfer.com/writing/expert-parallelism-distributed-moe-training>
4. Occult 论文，重点看 collaborative communication、超过 40% 运行时间的通信瓶颈、以及降低跨设备协作成本的方法：<https://proceedings.mlr.press/v267/luo25f.html>
5. Emergent Mind 关于专家并行的综述，可作为补充阅读，帮助理解拓扑与通信代价之间的关系：<https://www.emergentmind.com/topics/expert-parallelism-ep>
