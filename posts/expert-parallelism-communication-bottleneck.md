## 核心结论

Expert Parallelism 的本质不是“把专家分散到不同 GPU 上”本身，而是“为了让 token 找到对应专家，系统必须为每个 MoE 层付出两次全局交换的代价”。MoE 的路由器负责给 token 选择专家。它的作用很直接：输入一个 token 的隐藏状态，输出这个 token 应该交给哪些专家处理。只要这些专家不在同一张卡上，通信就不可避免。

在标准 Expert Parallelism 中，一个 MoE 层通常至少包含两次 all-to-all：

1. `dispatch`：把 token 发送到被选中专家所在的设备。
2. `combine`：把专家输出再送回 token 原来的顺序位置。

因此，Expert Parallelism 的扩展上限，首先由互连网络决定，而不是由 GPU 算力单独决定。对一个 MoE 层，若总 batch 为 $B$，序列长度为 $S$，隐藏维度为 $D$，每个 token 选择 top-$K$ 个专家，则一层前向的核心通信体量可近似写成：

$$
V_{\text{layer}} \approx 2 \cdot B \cdot S \cdot K \cdot D
$$

这里的 $V_{\text{layer}}$ 是“需要跨设备搬运的数据元素数”。如果要换成字节数，只需再乘上每个元素的字节数 $q$：

$$
V_{\text{bytes}} \approx 2 \cdot B \cdot S \cdot K \cdot D \cdot q
$$

若采用 BF16，则 $q = 2$ 字节。  
若这些 token 在 $P$ 个设备上近似均匀分布，则单设备平均负担约为：

$$
V_{\text{per-device}} \approx \frac{2 \cdot B \cdot S \cdot K \cdot D \cdot q}{P}
$$

这只是平均口径，不代表真实耗时一定均匀。因为 all-to-all 的瓶颈常常由最忙的链路、最热的专家和最慢的一张卡决定。

一个足够直观的数值例子是：取 $B=32,\ S=2048,\ D=4096,\ K=2$。则一层前向两次 all-to-all 的元素搬运量约为：

$$
2 \cdot 32 \cdot 2048 \cdot 2 \cdot 4096 = 1{,}073{,}741{,}824
$$

若按 BF16 计算，则一层前向理论通信量约为：

$$
1{,}073{,}741{,}824 \times 2 \approx 2{,}147{,}483{,}648\ \text{bytes} \approx 2.0\ \text{GiB}
$$

这个数只覆盖“理想化 token 向量搬运”，没有把 padding、路由索引、容量控制、分块流水、反向传播和负载不均衡算进去。工程上真正观测到的等效流量通常更高。

通信时间的第一近似可写成：

$$
t_{\text{comm}} \approx \frac{V}{B_w}
$$

其中 $B_w$ 是有效带宽，不是理论峰值带宽。所谓有效带宽，就是程序真实运行时，系统在当前拓扑、当前负载、当前消息大小下，能够持续拿到的实际吞吐。

这也是为什么 Wide-EP 在大 NVLink 域内通常能扩得更远，而一旦跨节点进入 InfiniBand，MoE 的可扩展性就会明显收紧。前者提供高带宽、低延迟的设备内或机架级互连；后者带宽量级更低，拥塞和尾延迟也更难控制。结论可以收敛成一句话：

**MoE 不是先把专家切开，再去想网络怎么补救；而是先看网络能承受多重的 token 交换，再决定专家怎么切。**

---

## 问题定义与边界

Expert Parallelism 讨论的是：在 MoE 层中，把不同专家放到不同设备上，让多个 GPU 分担专家参数存储和专家计算负载。这里的“专家”，通常指被稀疏激活的前馈网络分支。一个 token 并不会经过所有专家，而只会经过少数几个被路由器选中的专家。

它解决的核心问题主要有两个。

第一，单卡放不下足够多的专家参数。  
当专家数从几十增长到几百时，单张 GPU 的显存会先成为约束。把专家拆到多卡上，等价于把参数容量扩展到了整个集群。

第二，单卡无法同时高效处理大量 token 对多个专家的访问。  
即使显存还能放下，单卡也未必能在一个 step 内高效完成所有专家的权重读取、矩阵计算和批处理组织。把专家分布出去后，理论上可以提高整体吞吐。

但 Expert Parallelism 引入了新的边界条件：**token 不是静态地待在原地等待计算，而是必须跟着路由结果跨设备流动。** 这和张量并行不同。张量并行主要在固定算子内部切矩阵；Expert Parallelism 则是先做路由，再做动态重排和跨设备派发。动态路由意味着两件事：

1. 通信是重的，因为要搬 token 向量。
2. 通信不是完全规则的，因为每轮路由分布都可能不同。

可以用一个玩具例子说明。假设有 4 个 token：`t0, t1, t2, t3`，有 4 个专家：`e0, e1, e2, e3`，分别放在 4 张 GPU 上。若路由结果是：

| token | 选中的专家 |
| --- | --- |
| `t0` | `e0`, `e3` |
| `t1` | `e1`, `e2` |
| `t2` | `e0`, `e2` |
| `t3` | `e1`, `e3` |

那么每张卡都不能只处理“自己本地输入的 token”。例如 `t0` 既要送到 `e0` 所在卡，也要送到 `e3` 所在卡；专家计算结束后，两个结果还要按路由权重做聚合，再放回 `t0` 原来的序列位置。这就是 dispatch 和 combine 两次 all-to-all 的来源。

如果把这个流程拆开，新手更容易理解：

| 阶段 | 做什么 | 为什么必须通信 |
| --- | --- | --- |
| 路由 | 给每个 token 选 top-$K$ 专家 | 路由结果可能指向远端 GPU |
| dispatch | 把 token 发送到专家所在设备 | 专家参数不在本地 |
| expert compute | 专家各自做前馈计算 | 这是算力阶段，不是主要瓶颈 |
| combine | 把多个专家输出送回并聚合 | 下一层仍需按原 token 顺序继续 |

这个问题的边界也需要明确，否则讨论会混杂：

| 边界项 | 是否讨论 | 说明 |
| --- | --- | --- |
| 前向 MoE 层通信 | 是 | 本文核心对象 |
| 反向传播中的 dispatch/combine | 部分提及 | 会放大总通信，但不是主推导对象 |
| 路由负载均衡 | 是 | 直接影响热点专家和尾延迟 |
| 专家内部 GEMM 优化 | 否 | 本文重点不在算子实现，而在跨设备搬运 |
| 单机 NVLink / NVSwitch 域 | 是 | 用于说明高带宽域为何重要 |
| 多机 InfiniBand / RDMA | 是 | 用于说明跨节点扩展为何迅速变难 |
| 专家精度、量化、蒸馏 | 否 | 不改变本文的通信主结论 |

所以，本文讨论的是：**在 MoE 的 Expert Parallelism 中，通信为什么会成为主瓶颈，这个瓶颈如何估算，以及工程上通常如何减轻它。**

---

## 核心机制与推导

先看最基本的量纲。对一个 token，如果路由器选择 top-$K$ 个专家，那么这个 token 的隐藏状态向量通常要被发送到 $K$ 个目的地。隐藏状态向量长度为 $D$，也就是这个 token 在当前层的特征表示维度。

因此，若一个 MoE 层总共有 $B \cdot S$ 个 token，那么在理想无额外开销情况下，dispatch 阶段搬运的元素量近似为：

$$
V_{\text{dispatch}} \approx B \cdot S \cdot K \cdot D
$$

combine 阶段还要再来一次同量级的数据回传：

$$
V_{\text{combine}} \approx B \cdot S \cdot K \cdot D
$$

于是总通信元素量约为：

$$
V_{\text{layer}} \approx 2 \cdot B \cdot S \cdot K \cdot D
$$

若转成字节数，则有：

$$
V_{\text{bytes}} \approx 2 \cdot B \cdot S \cdot K \cdot D \cdot q
$$

其中 $q$ 是每个元素占用的字节数。常见情形下：

| 数据类型 | 每元素字节数 |
| --- | --- |
| FP32 | 4 |
| FP16 / BF16 | 2 |
| FP8 | 1 |

如果平均分布到 $P$ 张卡，则每张卡平均看到的流量约为：

$$
V_{\text{per-device}} \approx \frac{2 \cdot B \cdot S \cdot K \cdot D \cdot q}{P}
$$

但这个公式只是“平均值”，真实系统通常更差。原因至少有四类。

| 偏离理想值的来源 | 会带来什么问题 |
| --- | --- |
| 路由不均衡 | 某些专家更热，部分 GPU 和链路先拥塞 |
| capacity factor / padding | 为了让专家批处理形状规则，发送量可能超过理论最小值 |
| 元数据 | 路由索引、偏移量、目标 rank 等信息也要组织和传递 |
| 分块流水与同步 | 数据不是一次性干净搬完，存在多轮发送和同步等待 |

一个更工程化的写法，是把实际通信量表示成理论量乘上放大因子 $\phi$：

$$
V_{\text{real}} \approx \phi \cdot 2 \cdot B \cdot S \cdot K \cdot D \cdot q,\quad \phi \ge 1
$$

这里的 $\phi$ 可以粗略理解为“现实世界的损耗系数”。当路由很均匀、分块合理、拓扑友好时，$\phi$ 接近 1；当热点专家明显、padding 多、跨节点多跳严重时，$\phi$ 会明显上升。

通信时间的最简单估计是：

$$
t_{\text{comm}} \approx \frac{V}{B_w}
$$

若把固定启动延迟和拓扑跳数也考虑进去，可以写成：

$$
t_{\text{comm}} \approx \alpha \cdot H + \frac{V}{B_w}
$$

其中：

| 符号 | 含义 | 直白解释 |
| --- | --- | --- |
| $\alpha$ | 单跳固定延迟 | 每次发消息都要付出的启动成本 |
| $H$ | hop 数 | 数据在网络里要经过多少次转发 |
| $V$ | 实际通信量 | 真正搬了多少字节 |
| $B_w$ | 有效带宽 | 程序运行时真实可用带宽 |

这套公式能解释一个常见现象：**相同的理论通信量，在节点内和跨节点的耗时差距会很大。** 因为节点内高带宽互连通常同时拥有更低的 $\alpha$、更小的 $H$ 和更高的 $B_w$。

下面给出一组参数化估算。设：

| 项目 | 取值 | 说明 |
| --- | --- | --- |
| $B$ | 32 | batch size |
| $S$ | 2048 | 序列长度 |
| $D$ | 4096 | 隐藏维度 |
| $K$ | 2 | 每个 token 选 2 个专家 |
| MoE 层数 | 16 | 仅统计 MoE 层 |
| $q$ | 2 bytes | BF16 |

则单层通信字节数为：

$$
V_{\text{bytes, layer}} = 2 \cdot 32 \cdot 2048 \cdot 2 \cdot 4096 \cdot 2
$$

约为：

$$
2{,}147{,}483{,}648\ \text{bytes} \approx 2.0\ \text{GiB}
$$

于是 16 层前向约为：

$$
16 \times 2.0\ \text{GiB} \approx 32.0\ \text{GiB}
$$

如果再考虑训练中的反向传播，通信量会继续放大。很多工程实现中，真正的训练步通信代价远不止前向的两次 all-to-all。

再把这个数字代入带宽模型，感受一下数量级。假设某高带宽域的有效带宽约为 900 GB/s，而跨节点链路的有效带宽约为 50 GB/s，则同样是 2 GiB 数据：

$$
t_{\text{high-bw}} \approx \frac{2}{900}\ \text{s} \approx 2.2\ \text{ms}
$$

$$
t_{\text{inter-node}} \approx \frac{2}{50}\ \text{s} \approx 40\ \text{ms}
$$

这只是理想带宽除法，还没把多路竞争、同步等待和尾延迟算进去。即便如此，数量级差异已经足够说明问题：**一旦主要路径落在较慢的跨节点互连上，通信时间很容易反超专家计算时间。**

下面这个表可以帮助新手建立直觉：

| 参数变化 | 对通信量的影响 |
| --- | --- |
| batch size $B$ 增大 | 线性增大 |
| 序列长度 $S$ 增大 | 线性增大 |
| 隐藏维度 $D$ 增大 | 线性增大 |
| top-$K$ 增大 | 线性增大 |
| MoE 层数增多 | 总通信近似线性叠加 |
| 数据精度从 BF16 变 FP8 | 字节数下降，但不改变 all-to-all 结构 |

最后看互连差异。简化比较如下：

| 互连 | 主要域 | 带宽特征 | 延迟特征 | 对 MoE EP 的意义 |
| --- | --- | --- | --- | --- |
| NVLink / NVSwitch | 节点内或大规模 NVLink 域内 | 高 | 低 | 更适合承接重 token 交换 |
| InfiniBand / RDMA | 跨节点 | 较低 | 更高 | 容易成为扩展上限 |

这张表的关键不在于背规格，而在于理解：**MoE 的 all-to-all 同时怕带宽不够，也怕延迟累积。**  
因此，Wide-EP 能成立的前提并不是“专家铺得更开”，而是“专家虽然铺得更开，但大部分交换仍留在高带宽域内”。

---

## 代码实现

下面先给一个可运行的 Python 示例。它不依赖 NCCL、PyTorch 或 RDMA，只用标准库完成三件事：

1. 计算单层与多层的理论通信量。
2. 估算不同有效带宽下的通信时间。
3. 用一个简单路由模拟展示负载不均衡如何放大最慢设备的压力。

```python
from dataclasses import dataclass
from collections import Counter
from typing import List, Sequence
import math
import random


@dataclass(frozen=True)
class MoEConfig:
    batch_size: int
    seq_len: int
    hidden_dim: int
    top_k: int
    num_layers: int
    bytes_per_elem: int = 2   # BF16
    num_experts: int = 16
    num_devices: int = 8


def num_tokens(cfg: MoEConfig) -> int:
    return cfg.batch_size * cfg.seq_len


def layer_comm_elements(cfg: MoEConfig) -> int:
    # dispatch + combine
    return 2 * num_tokens(cfg) * cfg.top_k * cfg.hidden_dim


def layer_comm_bytes(cfg: MoEConfig) -> int:
    return layer_comm_elements(cfg) * cfg.bytes_per_elem


def total_forward_comm_bytes(cfg: MoEConfig) -> int:
    return layer_comm_bytes(cfg) * cfg.num_layers


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def comm_time_seconds(num_bytes: int, bandwidth_gbps: float) -> float:
    # 这里把输入带宽当作 GB/s，而不是 Gb/s
    return num_bytes / (bandwidth_gbps * (10 ** 9))


def expert_to_device_map(num_experts: int, num_devices: int) -> List[int]:
    return [expert_id % num_devices for expert_id in range(num_experts)]


def simulate_routing(
    total_tokens: int,
    num_experts: int,
    top_k: int,
    hot_expert: int | None = None,
    hot_prob: float = 0.0,
    seed: int = 0,
) -> List[List[int]]:
    """
    返回每个 token 选中的 top-k 专家列表。
    若设置 hot_expert 和 hot_prob，则以一定概率强行把该专家加入路由结果，
    用于模拟热点专家。
    """
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")

    rng = random.Random(seed)
    routing = []

    for _ in range(total_tokens):
        experts = list(range(num_experts))
        chosen = []

        if hot_expert is not None and rng.random() < hot_prob:
            chosen.append(hot_expert)
            experts.remove(hot_expert)

        need = top_k - len(chosen)
        chosen.extend(rng.sample(experts, need))
        routing.append(chosen)

    return routing


def per_device_dispatch_load(
    routing: Sequence[Sequence[int]],
    hidden_dim: int,
    bytes_per_elem: int,
    expert_device: Sequence[int],
) -> Counter:
    """
    统计 dispatch 阶段每个设备接收了多少字节。
    combine 阶段量级通常近似相同，因此总量可再乘 2。
    """
    loads = Counter()

    for experts in routing:
        for expert_id in experts:
            device = expert_device[expert_id]
            loads[device] += hidden_dim * bytes_per_elem

    return loads


def summarize_device_load(loads: Counter) -> str:
    if not loads:
        return "no load"
    total = sum(loads.values())
    peak = max(loads.values())
    avg = total / len(loads)
    imbalance = peak / avg if avg > 0 else math.inf
    return (
        f"total_dispatch={bytes_to_gib(total):.2f} GiB, "
        f"avg_per_device={bytes_to_gib(avg):.2f} GiB, "
        f"peak_device={bytes_to_gib(peak):.2f} GiB, "
        f"peak/avg={imbalance:.2f}x"
    )


def main() -> None:
    cfg = MoEConfig(
        batch_size=32,
        seq_len=2048,
        hidden_dim=4096,
        top_k=2,
        num_layers=16,
        bytes_per_elem=2,
        num_experts=16,
        num_devices=8,
    )

    per_layer = layer_comm_bytes(cfg)
    total_fwd = total_forward_comm_bytes(cfg)

    assert layer_comm_elements(cfg) == 1_073_741_824
    assert per_layer == 2_147_483_648
    assert total_fwd == 34_359_738_368

    print("=== Theory ===")
    print(f"tokens={num_tokens(cfg)}")
    print(f"per_layer={bytes_to_gib(per_layer):.2f} GiB")
    print(f"total_forward={bytes_to_gib(total_fwd):.2f} GiB")

    for bw in (900, 200, 50):
        t_ms = comm_time_seconds(per_layer, bandwidth_gbps=bw) * 1000
        print(f"effective_bandwidth={bw:>4} GB/s -> per_layer_comm={t_ms:.2f} ms")

    print("\n=== Routing Load Simulation ===")
    expert_device = expert_to_device_map(cfg.num_experts, cfg.num_devices)

    uniform_routing = simulate_routing(
        total_tokens=num_tokens(cfg),
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        seed=42,
    )
    uniform_load = per_device_dispatch_load(
        routing=uniform_routing,
        hidden_dim=cfg.hidden_dim,
        bytes_per_elem=cfg.bytes_per_elem,
        expert_device=expert_device,
    )
    print("uniform routing :", summarize_device_load(uniform_load))

    hot_routing = simulate_routing(
        total_tokens=num_tokens(cfg),
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hot_expert=0,
        hot_prob=0.50,
        seed=42,
    )
    hot_load = per_device_dispatch_load(
        routing=hot_routing,
        hidden_dim=cfg.hidden_dim,
        bytes_per_elem=cfg.bytes_per_elem,
        expert_device=expert_device,
    )
    print("hot expert      :", summarize_device_load(hot_load))


if __name__ == "__main__":
    main()
```

这个脚本可以直接运行，输出会稳定体现三件事：

1. 通信量随 $B,S,D,K$ 线性增长。
2. 同样的数据量，在不同有效带宽下耗时差异很大。
3. 路由一旦出现热点，最忙设备的负载会显著高于平均值，而系统步时通常由这个“最慢点”决定。

如果要把上面的数字和公式对应起来，可以看下面这张表：

| 代码中的量 | 公式中的量 | 含义 |
| --- | --- | --- |
| `num_tokens(cfg)` | $B \cdot S$ | 总 token 数 |
| `layer_comm_elements(cfg)` | $2BSDK$ | 单层前向元素数 |
| `layer_comm_bytes(cfg)` | $2BSDKq$ | 单层前向字节数 |
| `comm_time_seconds(...)` | $V/B_w$ | 一阶通信时间估算 |
| `peak/avg` | 负载不均衡程度 | 越高说明热点越严重 |

再看更接近真实实现流程的伪代码。它不是某个具体框架的源码，而是把多节点 Expert Parallelism 的数据路径抽象出来：

```python
def moe_layer_forward(token_chunks, router, experts, intra_node_net, inter_node_net):
    outputs = []

    for chunk in token_chunks:
        routing_map = router(chunk)  # 为每个 token 选择 top-k 专家

        # 1. 先在节点内打包和重排，减少跨节点零散流量
        local_packets, remote_packets = intra_node_net.pack(chunk, routing_map)
        staged = intra_node_net.dispatch(local_packets, remote_packets)

        # 2. 只有真正需要跨节点的部分，才发到远端
        remote_ready = inter_node_net.transfer(staged.remote_part)

        # 3. 本地专家和远端到达的专家输入一起执行
        expert_input = staged.local_part + remote_ready
        expert_output = experts.run(expert_input)

        # 4. 输出回收：先跨节点，再节点内恢复 token 原顺序
        returned = inter_node_net.gather(expert_output.remote_part)
        combined = intra_node_net.combine(
            local_output=expert_output.local_part,
            remote_output=returned,
            routing_map=routing_map,
        )

        outputs.append(combined)

    return outputs
```

这段伪代码里有两个工程重点。

第一，**分层通信**。  
先在节点内聚合，再把必须跨节点的数据发出去。原因很简单：节点内互连通常更快，能先在本地做掉的重排，不应该直接扔给跨节点网络。

第二，**通信与计算重叠**。  
把 token 切成 chunk 后，一部分 chunk 在传输时，另一部分 chunk 已经在专家侧计算。这样做的目标不是减少总字节数，而是减少“纯等待时间”。

如果再往底层走，像 Hybrid-EP 这类实现会把一个内核进一步拆成多个数据通道和 warp group：有的负责 dispatch，有的负责 combine，有的负责节点内搬运，有的负责 RDMA 发送。这个设计的目的只有一个：**尽量把 all-to-all 的等待切碎，并塞进计算空档里。**

---

## 工程权衡与常见坑

工程上最常见的误判是：只盯着 FLOPs，不盯 all-to-all。  
FLOPs 描述的是浮点计算量，它对稠密模型通常很有解释力；但对 MoE，很多系统在 FLOPs 还没打满前，网络已经先撞墙了。原因不复杂：稀疏激活减少了“每个 token 要算多少专家”，却没有消除“每个 token 可能要去远端找专家”的事实。

第一个坑，是把理论峰值带宽当成有效带宽。  
链路规格写得再高，也不代表 all-to-all 一定能接近峰值。all-to-all 不是单向大流，而是多源多目的并发交换。它容易受到拓扑、消息大小、并发流数、软件栈、队列深度和拥塞控制影响。因此，真正有意义的不是“卡面峰值”，而是“当前负载下测得的有效带宽”。

第二个坑，是低估热点专家的破坏力。  
负载均衡不只是“平均吞吐更漂亮”，它直接决定尾延迟。若路由器偏向少数专家，那么这些专家所在 GPU 会同时面临三种压力：

| 压力来源 | 结果 |
| --- | --- |
| 输入 token 变多 | 专家侧批次变大，排队加剧 |
| 对应链路流量上升 | 网络入口更容易拥塞 |
| buffer 占用上升 | 更容易出现等待和回收延迟 |

此时即使全局平均通信量没变，整层时间也会被最慢路径拖长。

第三个坑，是 chunk 大小选错。  
chunk 太小，固定启动开销占比高，包太碎，网卡和软件栈更难高效工作。chunk 太大，则通信和计算容易串行化，丢掉 overlap 的收益。经验上，合理 chunk 往往不是一个全局常数，而是要随拓扑分层：节点内可更大，跨节点更谨慎。

第四个坑，是 capacity factor 和 padding 被忽略。  
很多初学者会把 $2BSDKq$ 当成“真实字节数”。实际上，专家通常希望拿到更规则的批处理形状，这会引入 padding；为了防止某专家被路由过多 token，系统又常设置 capacity 上限，这会进一步改变真实发送量、丢弃策略或回退逻辑。也就是说，理论式子给出的是下界附近的估算，不是完整账单。

第五个坑，是 RDMA buffer 管理。  
RDMA 的优势是低 CPU 介入、低开销传输，但前提是缓冲区准备得当。若每轮都重新注册大块显存、频繁分配回收，控制路径就会吃掉本该用于数据传输的收益。实际系统通常会预分配、预注册并复用大 buffer，用最坏情况做容量规划。

第六个坑，是只看平均 step time，不看尾延迟。  
MoE 很容易出现“平均还行，但 P99 很差”的现象。原因是动态路由带来的负载和流量不规则，会让少数 step 触发更严重的热点与排队。对于训练，这会拉低整体吞吐；对于在线推理，这会直接恶化 SLA。

下面给一个简化权衡表：

| 维度 | 节点内高带宽域 | 跨节点 Hybrid-EP |
| --- | --- | --- |
| 带宽 | 更高 | 更低 |
| 延迟 | 更低 | 更高 |
| token 重排成本 | 相对可控 | 更易成为瓶颈 |
| 优化重点 | 提高局部并发和专家利用率 | 减少跨节点字节数与等待时间 |
| 主要风险 | 域大小有限 | NIC 饱和、热点、buffer 管理、尾延迟 |

真实工程里，经常出现一种链式阻塞：GPU 在等网络，网络在等缓冲，缓冲在等上一次发送完成。这时再增加 GPU 数量，通常不会线性提升性能，因为瓶颈根本不在算力侧。更有效的手段通常是：

1. 优先把专家尽量留在高带宽域内。
2. 对跨节点流量做节点内预聚合。
3. 预注册并复用通信 buffer。
4. 用 micro-batch 和 chunk pipeline 做 overlap。
5. 通过负载均衡损失、capacity 策略或 shared experts 降低热点。

---

## 替代方案与适用边界

如果当前集群的跨节点网络偏弱，最直接的替代思路不是盲目扩大 Expert Parallelism，而是重新定义“哪些通信必须发生，哪些通信必须跨节点发生”。

第一类方案是 Wide-EP。  
它适用于存在大规模 NVLink 域的系统，例如大 NVLink / NVSwitch 域内的部署。它并没有改变 MoE 的原理，而是改变了通信发生的地理位置：尽量把最重的 token 交换留在高带宽域内。这样做有两个直接收益：

1. 单卡承载的专家数减少，权重驻留压力下降。
2. dispatch / combine 虽然仍存在，但主路径落在更快的互连上。

它适合“高速互连域足够大”的系统，不适合“主要依赖普通跨节点网络”的集群。

第二类方案是 Hybrid-EP 或其他层次化通信方案。  
它适用于专家必须跨节点部署的情形。核心做法不是幻想消灭 all-to-all，而是把 all-to-all 拆成层次结构：

1. 节点内先聚合。
2. 节点间只发送真正需要远程处理的 chunk。
3. 尽可能把传输和专家计算重叠起来。

这个方案的价值在于缩短最差路径，而不是让跨节点通信 magically 变便宜。

第三类方案，是直接降低稀疏层的通信强度。  
这类方法的共同点是：接受一点模型表达或结构自由度上的折中，换取更稳的系统效率。例如：

| 手段 | 作用 | 代价 |
| --- | --- | --- |
| 降低 top-$K$ | 线性减少 token 复制次数 | 可能影响模型质量 |
| 减少部分 MoE 层的隐藏维度 | 降低每次搬运向量长度 | 会影响层容量 |
| shared experts | 让高频模式留在本地 | 结构更复杂 |
| 更强的负载均衡约束 | 降低热点专家概率 | 路由自由度下降 |
| 调整 micro-batch | 缓解尾延迟和瞬时拥塞 | 可能影响吞吐 |

第四类方案，是回退到更保守的并行边界。  
如果集群规模不大、网络一般、任务又偏低延迟，那么与其上大规模 Expert Parallelism，不如减少专家数，甚至回退到更密集的结构。原因不是 MoE 理论不好，而是系统条件不支持它兑现理论收益。

下面给一个适用边界表：

| 方案 | 适合场景 | 不适合场景 |
| --- | --- | --- |
| Wide-EP | 有大规模高带宽互连域，追求高吞吐训练或推理 | 主要依赖普通跨节点网络的集群 |
| Hybrid-EP | 必须跨节点放专家，且软件栈支持分层通信与 overlap | 网络弱、buffer 管理差、无法做细粒度流水 |
| 降低 top-$K$ / shared experts | 网络已成为主瓶颈，需要先稳住系统效率 | 追求最强稀疏表达能力 |
| 更少专家或更密集结构 | 小集群、调试环境、低延迟在线业务 | 目标是极大容量稀疏扩展 |

一个可操作的判断准则是：

$$
t_{\text{comm}} \gtrsim t_{\text{expert\_compute}}
\Rightarrow \text{继续增加专家数的收益会快速变差}
$$

也就是说，如果跨节点链路的有效带宽已经低到无法支撑 token 交换速度，那么再增加专家规模，收益很可能被通信抵消。反过来，如果大部分 token 交换都能留在高带宽域内，Wide-EP 才更有机会把吞吐优势兑现出来。

所以，Expert Parallelism 的适用边界不是“专家越多越好”，而是：

**你的网络是否允许 token 以足够低的代价完成两次全局搬运。**

这才是 MoE 扩展能否成立的第一性约束。

---

## 参考资料

- Noam Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*：<https://arxiv.org/abs/1701.06538>
- Dmitry Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*：<https://arxiv.org/abs/2006.16668>
- William Fedus, Barret Zoph, Noam Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*：<https://arxiv.org/abs/2101.03961>
- NVIDIA Technical Blog, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack Scale Systems*：<https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/>
- NVIDIA Technical Blog, *Optimizing Communication for Mixture-of-Experts Training with Hybrid Expert Parallel*：<https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/>
- NVIDIA Megatron Core 文档与仓库，用于了解训练侧并行组合与 MoE 支持：<https://github.com/NVIDIA/Megatron-LM>
- MBrenndoerfer, Expert Parallelism 的通信量分析与推导：<https://mbrenndoerfer.com/writing/expert-parallelism-distributed-moe-training>
