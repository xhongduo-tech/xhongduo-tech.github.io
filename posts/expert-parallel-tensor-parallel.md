## 核心结论

Expert Parallel，简称 EP，可以理解为“按专家分机器”：在 MoE（Mixture of Experts，混合专家模型，意思是一次前向只激活少数专家）里，把不同专家完整地放到不同 GPU 上。Tensor Parallel，简称 TP，可以理解为“把一个大专家拆开”：单个专家内部的矩阵权重沿某个维度切分到多张 GPU 上共同计算。

两者解决的是不同瓶颈。EP 主要解决“专家很多，单卡没必要全放，也放不下”的问题；TP 主要解决“单个专家本身已经大到单卡装不下”的问题。因此它们可以组合，而不是互斥。

在大规模 MoE 里，EP 和 TP 的通信模式也不同。EP 的核心通信是 All-to-All，也就是“把 token 发到对应专家所在卡，再把结果取回”；TP 的核心通信通常是 AllReduce 或 ReduceScatter/AllGather，也就是“多个 GPU 分别算一部分，再把部分结果同步成完整结果”。因为通信语义不同，这两类并行通常要放在独立进程组里。

一个直接结论是：如果 GPU 资源足够，优先让 `EP = 专家数`，尽量避免 TP；如果 GPU 数不够，或者单个专家太大，就用 `EP + TP` 混合。以 Mixtral 8x7B 为例，8 个专家可以做成 `EP=8, TP=1`，即每卡一个专家；也可以做成 `EP=4, TP=2`，即每个专家跨 2 卡切分。第二种能降低单卡显存压力，但会引入 TP 额外同步成本。

对初学者，一个可用的直观图景是邮局系统：

- EP 像“把包裹送到正确站点”：每个 token 先被路由到对应专家所在 GPU。
- TP 像“站点内部拆包协作”：一个大专家的矩阵计算拆给两张或多张卡共同完成。
- Sequence Parallel，简称 SP，可以理解为“按序列维度切输入并配合 TP 同步”的控制室：它不替代 EP 或 TP，而是保证 TP/DP 的同步和 EP 路由后的张量布局能接上。

简化流程图如下：

```text
输入 token
   |
Router 选 Top-K 专家
   |
EP group (epsilon)
dispatch: All-to-All
   |
到达目标专家所在 rank
   |
TP group (tau)
专家内部线性层切分计算
   |
TP 同步: AllReduce / ReduceScatter
   |
EP group (epsilon)
combine: All-to-All
   |
输出 token
```

---

## 问题定义与边界

先定义讨论范围。本文只讨论训练或大批量推理中的稠密 Transformer + MoE FFN 结构，不讨论纯稠密模型，也不讨论专家并行以外的流水并行细节。

MoE 的基本事实是：总参数量很大，但每个 token 只走少数专家，例如 Top-2 路由就是每个 token 只进入 2 个专家。这里的 Top-K 指“每个 token 被送到 K 个专家”。这意味着总参数可以很大，但单次激活的计算量不一定线性增长。

EP 与 TP 的边界可以用一句话区分：

- EP 处理“专家放在哪些卡，以及 token 怎么送过去”。
- TP 处理“一个专家内部太大，单卡算不动，怎么拆开一起算”。

这两个边界决定了它们的进程组不能混用。EP 的组负责 token 路由，TP 的组负责矩阵同步；如果把两类通信混在一个组里，张量语义会错位，结果不是变慢，而是直接算错。

下面用 Mixtral 8x7B 做边界说明。Mixtral 8x7B 可以粗略理解为“有 8 个专家分支的 MoE 模型，每次 token 激活其中少数专家”。若有 8 张卡，可做 `EP=8, TP=1`；若只有 4 张卡但单卡显存不足，可以改做 `EP=4, TP=2`，也就是每个逻辑专家由 2 张卡共同承载。

| 场景 | GPU 数 | 推荐组合 | 主要通信 | 主要瓶颈 |
|---|---:|---|---|---|
| 资源充足，专家数与卡数接近 | 8 | EP=8, TP=1 | EP All-to-All | 路由与负载均衡 |
| 卡数不足但带宽较好 | 8 | EP=4, TP=2 | EP All-to-All + TP AllReduce | 通信叠加 |
| 更少 GPU，单卡显存紧张 | 4 | EP=2, TP=2 或 EP=1, TP=4 | TP AllReduce 更重 | 单专家切分成本 |
| 小模型或无 MoE | 2-8 | Pure DP / Pure TP | DP/TP 同步 | 不需要 EP |

参数组合并不任意。若总 world size 记为 $W$，通常需要满足：

$$
W = DP \times EP \times TP \times PP
$$

其中 DP 是 Data Parallel，数据并行，意思是不同卡跑不同样本再同步梯度；PP 是 Pipeline Parallel，流水并行，意思是把网络层切成多段。

再看进程组数量。若采用 `EP x TP` 组合，至少需要：

| 组符号 | 作用 | 典型操作 |
|---|---|---|
| $\epsilon$ | EP 路由组 | All-to-All |
| $\tau$ | TP 计算组 | AllReduce / ReduceScatter / AllGather |
| $\delta$ | DP 梯度组 | AllReduce / ReduceScatter |
| SP 相关组 | Sequence Parallel 输入输出同步 | ReduceScatter / AllGather |

这里的核心边界是：EP 解决“横向分专家”，TP 解决“纵向拆专家内部矩阵”。二者互补，但各自要求张量布局不同，所以必须独立组织。

---

## 核心机制与推导

先看 EP。设：

- $B$ 是 batch size，即一次并行处理多少样本。
- $S$ 是序列长度，即每个样本多少 token。
- $K$ 是 Top-K，即每个 token 路由到几个专家。
- $h$ 是 hidden size，即每个 token 的隐藏向量维度。
- $EP$ 是 expert parallel size，即专家分布到多少个 rank。

总共有 $B \cdot S$ 个 token，每个 token 被复制或分发到 $K$ 个专家，所以总路由 token 份数是：

$$
B \cdot S \cdot K
$$

若专家均匀分布在 $EP$ 个 rank 上，平均只有 $\frac{1}{EP}$ 的 token 份数留在本地，其余 $\frac{EP-1}{EP}$ 需要发给其他 rank。因此每个 rank 的 dispatch 或 combine token 数近似为：

$$
B \cdot S \cdot K \cdot \frac{EP-1}{EP}
$$

这就是常见的 EP 路由通信量估计公式。

每个 token 份数需要发送一个长度为 $h$ 的向量。若使用 FP16，每个元素 2 字节，则单向字节量近似为：

$$
2 \cdot B \cdot S \cdot K \cdot h \cdot \frac{EP-1}{EP}
$$

当 $EP$ 较大时，$\frac{EP-1}{EP}$ 接近 1，所以常简写为单向约：

$$
2BShK
$$

往返一次 dispatch + combine，总字节量约：

$$
4BShK
$$

这只是激活通信，不含门控分数、索引、padding 和对齐损失。

### 玩具例子

假设只有 4 个 token、2 个专家、2 张卡、Top-1 路由。

- GPU0 放专家 E0
- GPU1 放专家 E1
- token 路由结果是 `[E0, E1, E1, E0]`

那么 GPU0 本地原始持有 4 个 token，但真正需要给 E1 处理的 token 要发到 GPU1；GPU1 处理完后还要把结果送回原 token 所在位置。这就是 EP 的 dispatch/combine 两次 All-to-All。

如果 E1 本身太大，GPU1 放不下，就把 E1 再切成两半。例如第一层线性矩阵前 50% 列放在 GPU1a，后 50% 列放在 GPU1b。此时 token 到达 E1 后，专家内部再做 TP 协作计算。这就是“先 EP 路由，再 TP 计算”的最小图景。

### 数值验算

按题设取：

$$
B=1,\ S=512,\ K=2,\ h=8192,\ EP=8
$$

每个 rank 的单向 dispatch token 份数：

$$
1 \times 512 \times 2 \times \frac{7}{8}=896
$$

单向元素数：

$$
896 \times 8192 = 7,340,032
$$

若 FP16，每元素 2 字节，则单向字节量约：

$$
7,340,032 \times 2 = 14,680,064\ \text{bytes} \approx 14.0\text{-}14.7\ \text{MB}
$$

往返 dispatch + combine 约 29 MB 量级。这个数不大到离谱，但它发生在每层、每个 microbatch，而且要与计算严格编排，所以很容易成为瓶颈。

若改成 `EP=4, TP=2`，EP 的单向通信量变成：

$$
1 \times 512 \times 2 \times \frac{3}{4}=768 \text{ token 份数}
$$

表面上 EP 的远端比例下降了，但每个专家内部又多出 TP 的同步。也就是说，EP=4+TP=2 不是简单“更省通信”，而是把一部分跨专家通信换成了专家内部同步。

下面用表格对比：

| 并行方式 | 张量移动对象 | 通信操作 | 通信方向 | 典型代价来源 |
|---|---|---|---|---|
| EP | token 激活 | All-to-All | 多对多 | 路由、负载不均、packing |
| TP | 层内部分和/部分激活 | AllReduce / RS / AG | 组内同步 | 矩阵切分后的结果合并 |
| DP | 梯度 | AllReduce / RS | 副本间同步 | 参数量大时梯度同步 |
| SP | 序列分片后的激活 | ReduceScatter / AllGather | 与 TP 协同 | 避免激活冗余 |

这里必须强调 SP。Sequence Parallel 不是“又一种可选优化”，而是在很多 TP+MoE 组合里用于保证张量布局一致的必要补充。白话说，TP 把一个层拆开以后，输入输出常常不能继续让每张卡都保留完整副本；SP 负责按序列维度拆开激活，让 TP 的同步和 EP 路由后的数据排列能对上。

真实工程里，经常采用下面这种顺序：

```text
token -> router
     -> EP dispatch (All-to-All, epsilon)
     -> expert local compute
     -> TP sync inside expert (tau)
     -> EP combine (All-to-All, epsilon)
     -> next layer with SP-compatible layout
```

---

## 代码实现

下面给一个最小可运行的 Python 玩具实现。它不依赖 `torch.distributed`，只在单机里模拟 EP 的 token 路由量和通信字节估算，用来验证公式。

```python
from dataclasses import dataclass

@dataclass
class MoEConfig:
    batch_size: int
    seq_len: int
    top_k: int
    hidden_size: int
    ep_size: int
    bytes_per_elem: int = 2  # FP16

def ep_dispatch_tokens(cfg: MoEConfig) -> int:
    return cfg.batch_size * cfg.seq_len * cfg.top_k * (cfg.ep_size - 1) // cfg.ep_size

def ep_dispatch_tokens_float(cfg: MoEConfig) -> float:
    return cfg.batch_size * cfg.seq_len * cfg.top_k * (cfg.ep_size - 1) / cfg.ep_size

def ep_bytes_one_way(cfg: MoEConfig) -> float:
    tokens = ep_dispatch_tokens_float(cfg)
    return tokens * cfg.hidden_size * cfg.bytes_per_elem

def ep_bytes_round_trip(cfg: MoEConfig) -> float:
    return 2 * ep_bytes_one_way(cfg)

cfg = MoEConfig(
    batch_size=1,
    seq_len=512,
    top_k=2,
    hidden_size=8192,
    ep_size=8,
)

tokens = ep_dispatch_tokens_float(cfg)
one_way = ep_bytes_one_way(cfg)
round_trip = ep_bytes_round_trip(cfg)

assert abs(tokens - 896.0) < 1e-9
assert abs(one_way - 14680064.0) < 1e-6
assert abs(round_trip - 29360128.0) < 1e-6

print("dispatch tokens:", tokens)
print("one-way MB:", one_way / (1024 * 1024))
print("round-trip MB:", round_trip / (1024 * 1024))
```

如果进入真正的分布式实现，关键不是公式，而是进程组要先划清。下面是简化的 PyTorch 风格伪代码：

```python
import torch.distributed as dist

def build_groups(world_size, dp_size, ep_size, tp_size):
    assert world_size % (dp_size * ep_size * tp_size) == 0

    ep_groups = []
    tp_groups = []
    dp_groups = []

    # 这里只示意，不展开 rank 映射细节
    for ep_ranks in compute_ep_rank_sets(world_size, dp_size, ep_size, tp_size):
        ep_groups.append(dist.new_group(ranks=ep_ranks))   # epsilon

    for tp_ranks in compute_tp_rank_sets(world_size, dp_size, ep_size, tp_size):
        tp_groups.append(dist.new_group(ranks=tp_ranks))   # tau

    for dp_ranks in compute_dp_rank_sets(world_size, dp_size, ep_size, tp_size):
        dp_groups.append(dist.new_group(ranks=dp_ranks))   # delta

    return ep_groups, tp_groups, dp_groups

def moe_forward(x, router, expert, ep_group, tp_group, sp_enabled=True):
    # 1. 路由：决定每个 token 去哪些专家
    route_info = router(x)

    # 2. EP: 按专家所在 rank 重排并发送 token
    dispatched = all_to_all_dispatch(x, route_info, group=ep_group)

    # 3. 专家内部计算
    if sp_enabled:
        dispatched = sequence_parallel_prepare(dispatched, tp_group)

    # 4. TP: 专家内部线性层切分计算并同步
    partial = expert.local_shard_forward(dispatched)
    partial = tp_allreduce_or_rs_ag(partial, group=tp_group)

    # 5. EP: 将专家输出送回原 token 所在 rank
    output = all_to_all_combine(partial, route_info, group=ep_group)
    return output
```

这个伪代码里有三个要点。

第一，EP 的所有 dispatch/combine 都必须在 $\epsilon$ 组里完成，不能偷懒复用 TP 组。

第二，TP 的同步只对“专家内部切分出来的部分结果”负责，不关心 token 原始来自哪个样本。

第三，SP 要放在 TP 兼容的位置上。白话说，EP 已经把 token 打散重排过一次，TP 再切专家内部矩阵时，输入布局必须先整理成 TP 能接受的形状，否则后续 ReduceScatter 或 AllGather 对不上。

可以把通信组记成下表：

| 组名 | 对应符号 | 建议 API | 负责内容 |
|---|---|---|---|
| ExpertParallelGroup | $\epsilon$ | `dist.new_group(ep_ranks)` | token dispatch/combine |
| TensorParallelGroup | $\tau$ | `dist.new_group(tp_ranks)` | 专家内部结果同步 |
| DataParallelGroup | $\delta$ | `dist.new_group(dp_ranks)` | 梯度同步 |
| SequenceParallelGroup | 与 $\tau$ 协同 | 常复用 TP 拓扑 | 序列维度激活切分 |

### 真实工程例子

以 Mixtral 8x7B 为例：

- 在 8 张 A100/H100 上，若单专家能装下，优先 `EP=8, TP=1`。优点是专家内部没有 TP 同步，路径最短。
- 若单卡显存不够，改成 `EP=4, TP=2`。此时每个逻辑专家跨 2 卡，路由后还要做专家内部同步。
- 若继续缩到 4 张卡，可能需要 `EP=2, TP=2`，甚至让更多模块共享显存预算，此时设计重点已经从“能否跑通”转为“通信是否压垮吞吐”。

---

## 工程权衡与常见坑

最常见的误解是“EP 已经把专家分出去了，所以通信应该不大”。这不准确。EP 的问题不在于参数同步，而在于激活路由。每层都要 dispatch 和 combine，一旦 token 数大、hidden 大、Top-K 大，All-to-All 很容易吃掉 30% 到 60% 的层时延。

下面把瓶颈拆开看：

| 瓶颈类型 | 典型来源 | 表现 | 常见缓解方式 |
|---|---|---|---|
| EP 通信 | All-to-All 路由 | GPU 等数据 | token packing、overlap、减小 Top-K |
| TP 同步 | AllReduce / RS-AG | 带宽占满 | 降低 TP 度、启用 SP |
| 热专家 | 路由不均 | 个别 rank 变慢 | load balancing、capacity factor |
| 同步错位 | 组配置错误 | 结果错或 hang 住 | 独立进程组、严格 rank 映射 |

“热专家”是最典型的真实问题。白话说，就是某个专家特别受欢迎，像只有一个柜台前排长队。即使平均通信量没问题，最忙的 rank 也会拖慢整批训练，形成 straggler，也就是拖后腿的慢 worker。

解决思路通常有几类：

- 负载均衡损失，让 router 不要总偏向少数专家。
- capacity factor，限制单个专家可接收的 token 容量。
- overlap 多个 microbatch，把通信和计算尽量重叠。
- 分层调度，例如 DeepEP 或类似方案，把跨节点和节点内路径分开优化。
- Hybrid-EP，即混合专家并行，在拓扑上进一步考虑节点边界与专家放置。

对初学者，一个可视化顺序如下：

```text
microbatch 1: EP dispatch ---- compute ---- EP combine
microbatch 2:        EP dispatch ---- compute ---- EP combine
                     ^^^^^^^^^^^ 与前一个 batch 的计算重叠
```

真正的坑经常不是“理论没懂”，而是“系统细节没对齐”。常见坑如下：

| 风险 | 为什么会发生 | 结果 | 规避方式 |
|---|---|---|---|
| 只配 EP，不配 SP | TP 需要兼容的激活布局 | TP 输出错位或额外复制 | 按框架要求启用 SP |
| 复用同一进程组做 EP/TP | 通信语义不同 | 死锁或错误同步 | 分离 $\epsilon$ 与 $\tau$ |
| Top-K 设太大 | token 复制次数增加 | All-to-All 放大 | 先验证 K=1/2 的收益 |
| 忽略专家不均衡 | router 自然偏斜 | 热专家拖慢吞吐 | 加平衡损失与容量控制 |
| 只看平均带宽 | P99 延迟被忽略 | 尾延迟拉高 step time | 看最慢 rank 而非平均值 |

一个工程上的判断标准是：如果你把 `EP=8, TP=1` 改成 `EP=4, TP=2` 后，显存压力下降了，但 step time 反而变差，这通常不是“TP 一定不好”，而是因为 TP 同步、SP 布局变换、EP 打包开销一起叠上来了。需要分阶段 profile，而不是凭感觉调参。

---

## 替代方案与适用边界

选并行策略时，不要先问“哪个最先进”，而要先看三个量：GPU 数、单卡显存、互联带宽。

可用下面的决策表快速判断：

| GPU 条件 | 推荐策略 | 适用原因 | 代价 |
|---|---|---|---|
| 卡多且单卡显存足 | Pure EP 或 EP=专家数 | 通信路径最简单 | 需要较多 GPU |
| 卡数一般，单专家偏大 | EP + TP + SP | 同时解决专家分布和单专家显存 | 组管理复杂，通信叠加 |
| 小模型或专家数少 | Pure TP / Pure DP | 不值得引入 EP 路由 | 无法利用 MoE 专家稀疏性 |
| 路由极不均衡 | Hybrid-EP | 更关注拓扑与负载均衡 | 实现复杂度更高 |

可以把 8 卡、4 卡、2 卡的典型方案理解为：

```text
8卡:
[EP=8, TP=1]  优先
[EP=4, TP=2]  备选

4卡:
[EP=2, TP=2]  常见折中
[EP=1, TP=4]  专家很大时才考虑

2卡:
[EP=1, TP=2] 或纯TP/DP
MoE收益通常开始被通信和复杂度侵蚀
```

几种替代方案的边界如下。

Pure DP 适合小模型。优点是实现最简单；缺点是参数全量复制，MoE 专家再多也不节省显存。

Pure TP 适合专家数少、但单层特别大的情况。优点是每层都能切；缺点是每层都同步，且没有利用“专家稀疏激活”的分布式放置优势。

Pure EP 适合专家多且每个专家单卡能放下的情况。优点是最符合 MoE 直觉；缺点是遇到热专家或单专家过大时会卡住。

Hybrid-EP 适合跨节点大集群。它不是“再造一种并行”，而是在 EP 框架内更精细地安排专家放置、负载均衡和跨层级通信。

因此，对新手最实用的一条经验是：

- 先看单专家能否单卡放下。
- 能放下，优先把专家一一映射到卡，即优先 EP。
- 放不下，再考虑 TP。
- 只要用了 TP，就要同时检查 SP 和对应进程组是否正确。
- 最后再谈 overlap、负载均衡和通信优化。

---

## 参考资料

1. NVIDIA Megatron Core, *Mixture of Experts User Guide*  
引用路径：<https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/moe.html>  
备注：用于“核心结论”“代码实现”“替代方案与适用边界”中关于 EP/TP/SP 组合与进程组关系的描述。

2. 腾讯大模型面试题解析，*TP/EP 通信量推导*  
引用路径：<https://blog.51cto.com/u_16163452/12995416>  
备注：用于“核心机制与推导”中 $B\cdot S\cdot K\cdot \frac{EP-1}{EP}$ 与通信字节量估算。

3. PARASMOE 论文，*Independent Process Groups for EP/TP/DP*  
引用路径：<https://openreview.net/pdf/bb4af9ea1fe7360b9b6af13c5a458474e2ea58c2.pdf>  
备注：用于“问题定义与边界”“代码实现”中关于独立进程组的工程必要性。
