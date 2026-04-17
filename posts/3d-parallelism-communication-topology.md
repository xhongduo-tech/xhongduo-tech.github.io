## 核心结论

3D 并行是把训练用到的 GPU 组织成一个 $TP \times PP \times DP$ 的三维网格。这里的“网格”可以理解为：同一批 GPU 既按张量切分、又按层切分、还按数据副本切分，每个维度负责一种不同的扩展目标。

最重要的结论不是“把模型拆开”本身，而是“把不同频率的通信放到合适的链路上”。通信频率从高到低通常是：

$$
\text{TP} \gg \text{PP} \gg \text{DP}
$$

因此带宽映射的基本原则是：

| 并行维度 | 主要通信 | 通信频率 | 推荐链路 | 典型带宽量级 | 原因 |
| --- | --- | --- | --- | --- | --- |
| TP | AllReduce / AllGather / ReduceScatter | 每层都发生，最高 | NVLink / NVSwitch | 约 600 GB/s | 高频通信必须留在机内高速互联 |
| PP | 激活值与梯度在阶段间传递 | 每个微批都会发生 | InfiniBand | 200 Gb/s 量级 | 频率中等，适合跨节点串联 |
| DP | 梯度或参数同步 | 每步一次或少量几次 | 剩余集群网络 | 依集群而定 | 频率最低，容忍较慢链路 |

如果把 TP 错误地跨节点映射到 InfiniBand，上层框架虽然还能跑，但吞吐量会显著下降。原因很直接：最密集的通信走了最慢的一层链路，算力被通信阻塞。工程上，错误映射带来 40% 以上的吞吐损失并不罕见。

---

## 问题定义与边界

3D 并行解决的是三个不同问题同时出现时的训练扩展问题：

1. 模型太大，单卡显存放不下，需要 TP。
2. 模型太深，单纯做 TP 后仍然难以高效扩展，需要 PP。
3. 想提高总吞吐，还要复制多个完整训练副本，需要 DP。

这里先解释三个术语。

“张量并行（Tensor Parallelism, TP）”是把同一层里的矩阵运算拆到多张卡上，每张卡只算一部分，白话讲就是“同一层大家一起分工做”。

“流水线并行（Pipeline Parallelism, PP）”是把不同层分给不同 GPU 组，前一组算完激活值再交给后一组，白话讲就是“不同阶段接力跑”。

“数据并行（Data Parallelism, DP）”是复制多个完整模型副本，各自处理不同样本，最后同步梯度，白话讲就是“多人各做一份同样的题，最后统一答案更新”。

3D 并行的硬约束可以先写成一个公式：

$$
\text{Total GPUs} = TP \times PP \times DP
$$

这个公式不是装饰，它决定了你能否把硬件整齐地铺满。例如 512 张 GPU，如果选 $TP=8, PP=8, DP=8$，那么：

$$
512 = 8 \times 8 \times 8
$$

这意味着：
- 每 8 张 GPU 构成一个 TP 组；
- 8 个 TP 组串成一个完整流水线；
- 再复制 8 份这样的流水线作为 DP 副本。

玩具例子可以先看一个非常小的配置。假设你只有 16 张 GPU，每节点 8 卡，节点内有 NVLink，节点间走 InfiniBand。一个合理配置可能是：

- $TP=8$
- $PP=2$
- $DP=1$

此时一个完整模型正好占满两个节点。为什么不设成 $TP=16$？因为那会把 TP 跨到节点间，让每层都走慢链路。为什么不设成 $PP=8$？因为阶段过多会增加流水线气泡，实际效率未必高。

边界也要说清楚。3D 并行不是所有训练任务的默认答案。小模型、单节点训练、或者显存已经足够时，直接用 DP 或 FSDP 往往更简单。只有当“单卡放不下 + 单节点不够 + 吞吐还要继续扩展”同时成立时，3D 并行才真正必要。

---

## 核心机制与推导

理解 3D 并行的关键，不是记住缩写，而是理解三类通信的“频率”和“大小”。

先看 TP。TP 把同一层拆开，因此一层算完往往就要做一次集合通信。对 Transformer 来说，这类通信会在前向和反向中反复出现。也就是说，只要层数很多，TP 的通信就会非常密集。

再看 PP。PP 的通信发生在阶段边界，传的是激活值和对应反向梯度。它不是“每层都通信”，而是“每个微批在阶段之间通信”。

最后看 DP。DP 一般是在若干层都算完之后，对梯度做同步，频率最低。

所以三者并不是“平行的三种拆法”，而是“频率完全不同的三种通信模式”。这就是为什么它们必须和硬件拓扑对齐。

可以写一个简化的带宽预算思路。设：

- TP 每层通信量为 $C_{tp}$
- PP 每个微批阶段间通信量为 $C_{pp}$
- DP 每步同步量为 $C_{dp}$

如果一轮训练中有 $L$ 层、$M$ 个微批，则总通信时间的粗略上界可写成：

$$
T_{comm} \approx \frac{L \cdot C_{tp}}{B_{tp}} + \frac{M \cdot C_{pp}}{B_{pp}} + \frac{C_{dp}}{B_{dp}}
$$

这里的 $B_{tp}, B_{pp}, B_{dp}$ 分别是三个维度实际拿到的有效带宽。

这个公式的重点不是精确预测秒数，而是告诉你：因为 $L$ 很大，TP 项通常最敏感，所以必须优先给最高带宽；PP 次之；DP 最后。

再看一个更具体的玩具例子。假设一个模型总参数与优化器状态折算后约占 350 GB，单卡显存 80 GB。若选 $TP=8$，只看张量切分后的平均承载量：

$$
\text{Model per GPU} \approx 350 / 8 = 43.75 \text{ GB}
$$

43.75 GB 已经进入单卡可承载范围，剩余显存还能留给激活值、KV cache、梯度和临时张量。然后如果模型有 96 层，再选 $PP=8$，就变成每个流水线阶段平均负责 12 层。这个拆法同时解决了“层太宽放不下”和“层太深跑不动”两个问题。

真实工程例子更能说明问题。100B 以上的大模型训练里，常见做法是把 8 卡节点当成一个天然 TP 单元，也就是 $TP=8$ 且严格锁在单节点内部。随后再用多个节点串起 PP，最后在更外层复制 DP 副本。这样映射的本质是：

- 把最频繁的层内通信留在 NVLink / NVSwitch；
- 把中频的阶段通信放到 InfiniBand；
- 把最低频的同步交给剩余网络窗口。

如果反过来做，比如 PP 放在机内而 TP 跨节点，那么理论上“也满足 $TP \times PP \times DP$ 的乘积关系”，但实际吞吐通常会塌。原因不是公式错，而是公式没有表达链路层级。3D 并行先是一个“通信拓扑问题”，其次才是一个“整数分解问题”。

---

## 代码实现

工程实现的第一步不是写训练脚本，而是先确认硬件拓扑。`nvidia-smi topo -m` 的作用是输出 GPU 之间的连接关系，白话讲就是“先看哪几张卡彼此最近、带宽最高”。

下面给一个可运行的 Python 玩具实现。它不依赖真实 GPU，只模拟如何按“节点内优先做 TP，节点间再做 PP，最外层再复制 DP”的规则生成分组。

```python
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GPU:
    global_rank: int
    node_id: int
    local_rank: int


def build_cluster(num_nodes: int, gpus_per_node: int) -> List[GPU]:
    gpus = []
    rank = 0
    for node in range(num_nodes):
        for local_rank in range(gpus_per_node):
            gpus.append(GPU(rank, node, local_rank))
            rank += 1
    return gpus


def build_3d_groups(num_nodes: int, gpus_per_node: int, tp: int, pp: int, dp: int):
    gpus = build_cluster(num_nodes, gpus_per_node)
    total = num_nodes * gpus_per_node
    assert total == tp * pp * dp, "GPU 总数必须等于 TP*PP*DP"
    assert tp <= gpus_per_node, "TP 必须能放进单节点，否则会跨节点"
    assert pp % num_nodes == 0 or num_nodes % pp == 0 or pp > 0

    nodes = {node: [g for g in gpus if g.node_id == node] for node in range(num_nodes)}

    # 1. 先在每个节点内切 TP 组
    tp_groups = []
    node_tp_units = []
    for node in range(num_nodes):
        local = nodes[node]
        assert len(local) % tp == 0
        units = []
        for i in range(0, len(local), tp):
            group = local[i:i + tp]
            tp_groups.append(group)
            units.append(group)
        node_tp_units.append(units)

    # 每个 TP 单元视为一个“流水线位置候选”
    flat_units = [unit for units in node_tp_units for unit in units]
    assert len(flat_units) == pp * dp

    # 2. 再按 DP 副本切 PP 链
    pp_groups = []
    for d in range(dp):
        start = d * pp
        chain = flat_units[start:start + pp]
        pp_groups.append(chain)

    # 3. 反向得到 DP 组：同一 stage index 上来自不同副本的单元构成一个 DP 组
    dp_groups = []
    for stage_idx in range(pp):
        replicas = []
        for d in range(dp):
            replicas.append(pp_groups[d][stage_idx])
        dp_groups.append(replicas)

    return tp_groups, pp_groups, dp_groups


if __name__ == "__main__":
    tp_groups, pp_groups, dp_groups = build_3d_groups(
        num_nodes=8,
        gpus_per_node=8,
        tp=8,
        pp=8,
        dp=8,
    )

    assert len(tp_groups) == 8 * 8  # 64 个 TP 单元
    assert len(pp_groups) == 8       # 8 个 DP 副本，每个副本一条 PP 链
    assert len(pp_groups[0]) == 8    # 每条链 8 个阶段
    assert len(dp_groups) == 8       # 8 个 stage index 对应 8 个 DP 组

    # TP 组必须都在单节点内
    for group in tp_groups:
        assert len({gpu.node_id for gpu in group}) == 1

    print("3D grouping sanity checks passed.")
```

这个示例省略了 NCCL 通信域、torch.distributed 初始化和真实拓扑探测，但核心顺序是对的：

1. 先按节点切 TP。
2. 再把 TP 单元串成 PP。
3. 最后把多条 PP 链复制成 DP。

真实工程里，你通常还需要做三件事。

第一，读取拓扑信息，而不是假设 local rank 连续就一定互联最佳。有些机器即使是 8 卡，也可能存在 NUMA 或不同交换芯片分区。

第二，显式生成三个 communicator。也就是 TP 组内用一套通信域，PP 相邻阶段用点对点通信或专门 group，DP 再用一套同步域。

第三，把 rank 映射固定下来。因为只要 rank 到 GPU 的映射漂移，同一份配置文件也可能跑出完全不同的通信效果。

---

## 工程权衡与常见坑

3D 并行的难点不在“能不能跑”，而在“跑出来的吞吐是否接近预期”。

最常见的坑，是把 TP 做跨节点。很多新手以为只要总 GPU 数够，`TP=16` 比 `TP=8` 更能分摊显存。这个推理只看到了显存，没有看到通信。TP 是层内高频通信，一旦跨节点，就等于每层都在慢链路上等待。结果通常不是“稍微慢一点”，而是 GPU 大量空转。

第二个坑，是 PP 分段不均衡。PP 的每个 stage 最好计算量接近，否则会出现“快阶段等慢阶段”。这叫流水线气泡，白话讲就是“传送带上有空档，后面的机器没活干”。即使通信拓扑完全正确，只要切层不均，吞吐还是会掉。

第三个坑，是把 DP 同步看得过于便宜。虽然 DP 频率最低，但当参数量非常大、优化器状态很多、或者 gradient accumulation 设置不合理时，DP 仍然可能在步末形成明显尾部延迟。

下面用表格总结常见问题。

| 坑 | 直接后果 | 典型表现 | 规避策略 |
| --- | --- | --- | --- |
| TP 跨节点 | 高频 AllReduce 被慢链路阻塞 | GPU 利用率低，step time 大幅上升 | TP 严格限制在单节点 NVLink 域内 |
| PP 切分不均 | 流水线气泡增大 | 某些 stage 长时间忙，其他 stage 等待 | 按层计算量和激活大小重新切 stage |
| 微批过少 | PP 无法填满流水线 | bubble 比例高，吞吐接近串行 | 增加 micro-batch 或虚拟 pipeline stage |
| DP 同步过重 | 步末尾延迟变长 | backward 结束后仍长时间卡住 | 梯度累积、分片优化器、压缩同步窗口 |
| rank 映射漂移 | 同配置不同效果 | 同一脚本在不同节点顺序下性能波动 | 固定 hostfile、rank 表、拓扑感知分配 |

一个非常实用的判断标准是：如果你发现训练日志里每一步的计算时间不低，但总 step time 仍很高，且网络流量峰值集中在层内同步阶段，那么大概率是 TP 映射出了问题。

真实工程例子里，这类错误经常出现在“先按框架默认值起任务，后面再看性能”的流程。比如一套 64 卡训练作业，理论设计是 8 节点、每节点 8 卡、$TP=8, PP=4, DP=2$。但调度器把 rank 打散后，某个 TP 组实际落在两个节点上，于是每层集合通信都穿过 InfiniBand。结果是 loss 正常下降，脚本也没有报错，但 tokens/s 明显偏低。这类问题最危险，因为它不是功能错误，而是性能错误。

---

## 替代方案与适用边界

不是所有模型都应该直接上 3D 并行。一个更稳妥的策略是按问题来源逐步升级。

如果模型还放得进单卡或单节点，优先用 DP 或 FSDP。FSDP 可以理解为“把参数、梯度、优化器状态进一步分片”，白话讲就是“不是复制整份，而是每人只保管一部分”。

如果模型的单层特别宽，矩阵乘法已经放不下，就加 TP。TP 针对的是“宽”，不是“深”。

如果模型层数很多，单纯 TP 后仍然难以扩展到更多节点，再加 PP。PP 针对的是“深”，不是“单层太大”。

如果你面对的是超长上下文，还可能引入 CP。CP 是 Context Parallelism，即上下文并行，白话讲就是“把长序列按 token 维度分开处理”。如果是 MoE 模型，还可能加入 EP。EP 是 Expert Parallelism，即专家并行，白话讲就是“不同专家分布在不同设备上，路由时再做交换”。

下面给出一个决策表。

| 场景 | 优先策略 | 主要解决的问题 | 何时升级 |
| --- | --- | --- | --- |
| 小模型、单节点 | DP | 提高吞吐 | 单卡显存开始吃紧时 |
| 参数很多、单层很宽 | DP + TP | 单层矩阵放不下 | TP 已放满单节点仍不够时 |
| 模型很深、需要多节点 | TP + PP + DP | 深层模型跨节点扩展 | stage 不均或链路变差时重新切分 |
| 超长上下文 | 在原方案上加 CP | 序列维度显存与通信压力 | 上下文长度继续增加时 |
| MoE 稀疏模型 | 在原方案上加 EP | 专家路由与容量扩展 | 专家数、路由流量进一步增大时 |

适用边界也要强调。3D 并行默认假设你有比较清晰的机内和机间带宽层级，例如“节点内明显快于节点间”。如果硬件本身没有这种强层级，或者单节点 GPU 很少、拓扑很不规则，那么标准的“TP 放机内、PP 跨节点、DP 最外层”仍然是基线，但不一定是全局最优，必须实际压测。

---

## 参考资料

- NVIDIA Megatron Core Parallelism Guide：并行维度定义、$TP \times PP \times DP$ 的基本约束，以及多种并行策略的组合方式。
- System Overflow《3D Parallelism and Topology Aware Mapping in Production》：3D 并行与拓扑感知映射的核心原则，包含大模型训练中的带宽分配思路与生产环境案例。
- Simulations4All《LLM Parallelism Strategies Explained》：TP、PP、DP 的通信模式差异，以及错误拓扑映射对利用率和吞吐的影响。
- Medium《Distributed GPU Training》：分布式训练中的基本数值例子，包括 512 GPU 配置下 TP、PP、DP 的切分直觉。
