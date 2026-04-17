## 核心结论

MoE，Mixture of Experts，中文常译为“专家混合”，可以理解为“很多个前馈网络里，只挑少数几个真正参与当前 token 计算”。容量因子 $C$ 的作用，是给每个专家分配一个最大接收上限，防止某几个热门专家在一次前向里被塞满。

容量因子的核心定义是：

$$
C_{\text{per\_expert}}=\left\lfloor C\cdot \frac{N}{E}\right\rfloor
$$

其中，$N$ 是当前批次 token 数，$E$ 是专家数。白话说，先假设 token 被平均分到每个专家，再乘一个缓冲倍数 $C$。这个缓冲区太小，会让热门专家装不下，出现 token 溢出；太大，又会让显存、padding 和跨卡通信成本一起上升。

一个最常见的经验值是 $C=1.25$。比如 8 个专家处理 1024 个 token，平均每个专家是 128 个；乘上 $1.25$ 后，每个专家最多收 160 个。如果某个专家被路由到 200 个 token，多出来的 40 个就必须按溢出策略处理，而不是继续塞进去。

| 容量因子 $C$ | 直接效果 | 优点 | 代价 |
|---|---|---|---|
| 小于 1.0 或接近 1.0 | 每个专家缓冲很紧 | 显存省、延迟稳 | 热门专家更容易溢出 |
| 约 1.1 到 1.25 | 留出适度冗余 | 通常是训练期较稳的折中 | 仍需监控溢出率 |
| 明显大于 1.25 | 缓冲很宽松 | 更少 token 被丢弃 | padding、通信、显存浪费明显 |

结论可以压缩成一句话：容量因子不是“越大越好”，而是“在可接受显存成本下，把溢出率压到足够低”。这也是为什么工程上会同时调 $C$、路由策略和负载均衡损失，而不是只调一个超参数。

---

## 问题定义与边界

这个问题讨论的是“路由已经做完之后，专家装不下怎么办”。也就是说，先有 gate，门控网络，负责给每个 token 选专家；再有 capacity check，容量检查，负责判断这个专家是否还有空位。

设第 $i$ 个专家实际收到的 token 数是 $T_i$，那么它的溢出量是：

$$
O_i=\max(0,\ T_i-C_{\text{per\_expert}})
$$

如果把所有专家的溢出合在一起，总体溢出率可以写成：

$$
r=\frac{1}{N}\sum_{i=1}^{E}\max(0,\ T_i-C_{\text{per\_expert}})
$$

有些文章也写成按比例形式：

$$
r=\frac{1}{N}\sum_i \max\left(0,\ f_i-\frac{C}{E}\right)
$$

其中 $f_i$ 表示第 $i$ 个专家实际接收到的 token 占比。两种写法本质上都在描述同一件事：热门专家超额了多少。

玩具例子先看最小版。假设：

- $N=16$
- $E=4$
- $C=1.25$

那么每个专家容量是：

$$
\left\lfloor 1.25\times \frac{16}{4}\right\rfloor=5
$$

如果路由结果是 `[6, 5, 3, 2]`，说明第 1 个专家多收了 1 个 token，其他专家没满。此时系统要决定：这 1 个 token 直接丢弃、改派给别人，还是送去一个兜底专家。

这里要明确边界：

| 问题 | 属于本文范围吗 | 说明 |
|---|---|---|
| gate 如何计算 top-1 或 top-2 | 部分涉及 | 只讲到它影响负载分布，不展开推导 |
| 专家内部 FFN 结构 | 不属于 | 这和容量因子无直接关系 |
| 多机多卡 all-to-all 细节 | 部分涉及 | 只讨论它为何放大容量设置的成本 |
| token 溢出后的处理 | 属于 | 这是本文核心 |

真实工程里，容量问题通常出现在“路由分布不均”而不是“平均容量不够”。例如一个 64 专家的大模型，理论上每个专家应收 1/64 的 token，但实际会出现几个专家长期偏热，其余专家长期偏冷。于是平均值看起来合理，局部却频繁溢出。这就是容量因子必须和负载均衡一起讨论的原因。

---

## 核心机制与推导

MoE 的执行流程可以压缩成：

$$
\text{token} \rightarrow \text{routing} \rightarrow \text{capacity check} \rightarrow \text{overflow strategy}
$$

更直白地说，就是“先分配，再验收，最后处理塞不进去的 token”。

### 1. 为什么会溢出

如果 gate 对一批 token 的打分高度集中，很多 token 会同时选择同一个专家。即使平均上每个专家只该拿 $\frac{N}{E}$ 个，实际也可能变成“少数专家爆满，多数专家空闲”。

这时容量因子相当于给每个专家一个固定长度的队列。超过长度的部分，不是“排队等待”，而是必须立刻走别的分支，因为训练和推理一般都要求张量形状固定、并行调度稳定。

### 2. 三种主流处理策略

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| 丢弃 | 超出的 token 不进专家，只走残差 | 最简单，延迟最稳 | token 没有得到专家变换，质量可能下降 |
| 随机或打分重路由 | 把溢出 token 送到还有余量的其他专家 | 利用空闲容量，提高吞吐 | 逻辑更复杂，可能引入不稳定 |
| 辅助专家兜底 | 主专家满后送次选专家或专门备用专家 | 延迟和质量折中较好 | 需要额外设计路由与容量规则 |

丢弃策略最容易理解。某个 token 原本该进专家 A，但 A 已满，那么这个 token 不再经过专家 FFN，只保留残差路径，相当于“这一层稀疏专家没有帮到它”。这并不等于 token 消失，而是“失去本层专家计算机会”。

重路由策略的关键是：不要把所有溢出 token 重新乱分一遍，否则代价很高。更常见的办法是给每个 token 保留候选专家列表，例如 top-2 分数；主专家满了，就尝试次选专家。如果次选也满了，再决定丢弃还是继续找第三个候选。

辅助专家兜底则更接近“预留应急车道”。系统可以规定每个 token 除主专家外还保留一个备用槽，主槽满了就进备用槽。这样实现比全局重路由更简单，也更容易做延迟控制。

### 3. 为什么还要 balancing loss

balancing loss，负载均衡损失，白话解释是“额外加一个约束，惩罚路由长期偏向少数专家”。否则会出现 expert collapse，专家塌缩，也就是只有极少数专家被频繁调用，其余专家几乎拿不到梯度。

一种常见形式是：

$$
L_{\text{balance}} = N\sum_{i=1}^{E} f_i p_i
$$

其中：

- $f_i$ 是第 $i$ 个专家实际接收的 token 比例
- $p_i$ 是 gate 给第 $i$ 个专家的平均路由概率

这个损失的目标不是让每个 token 都均匀分配，而是让“长期统计上”不要过度偏科。因为只要分布稍微均匀一些，容量因子的有效性就会大幅提升。

### 4. 一个完整的玩具例子

假设 8 个专家、1024 个 token、$C=1.25$。每个专家容量为：

$$
C_{\text{per\_expert}}=1.25\times \frac{1024}{8}=160
$$

现在某一步路由后，专家 3 收到 220 个 token。则：

$$
O_3 = 220-160 = 60
$$

这 60 个 token 的命运有三种：

- 直接丢弃：60 个 token 本层只走残差
- 重路由：尝试把这 60 个 token 按分数高低送到其他未满专家
- 备用专家：直接进入其 top-2 或专门 fallback expert

这三种做法都能运行，但它们优化的目标不同。第一种优先保证实现简单和稳定延迟，第二种优先减少 token 浪费，第三种优先做中间折中。

### 5. 一个真实工程例子

在大规模训练中，专家往往分布在多张 GPU 上。token 一旦被路由到不同卡，系统就要做 all-to-all，把 token 送到对应设备。此时如果 $C$ 过大，不仅每个专家要留更多缓冲区，还会让跨卡传输张量变大、padding 变多。结果是虽然溢出率下降了，但显存和通信开销迅速上升，吞吐反而变差。

所以真实系统的目标通常不是“零溢出”，而是“把溢出率压到很低，比如低于 1%，同时让 GPU 利用率和通信开销可接受”。这比单纯追求理论最优更符合工程现实。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只模拟“按路由结果计算容量、保留 token、标记溢出、尝试次选重路由”的过程。

```python
from math import floor
from collections import defaultdict

def route_with_capacity(assignments, second_choices, num_experts, capacity_factor):
    """
    assignments: 每个 token 的主专家编号
    second_choices: 每个 token 的次选专家编号
    """
    n = len(assignments)
    capacity = floor(capacity_factor * n / num_experts)

    expert_tokens = defaultdict(list)
    overflow_tokens = []

    # 第一轮：按主专家装填
    for token_id, expert_id in enumerate(assignments):
        if len(expert_tokens[expert_id]) < capacity:
            expert_tokens[expert_id].append(token_id)
        else:
            overflow_tokens.append(token_id)

    dropped = []
    rerouted = []

    # 第二轮：尝试重路由到次选专家
    for token_id in overflow_tokens:
        backup_expert = second_choices[token_id]
        if len(expert_tokens[backup_expert]) < capacity:
            expert_tokens[backup_expert].append(token_id)
            rerouted.append(token_id)
        else:
            dropped.append(token_id)

    total_kept = sum(len(v) for v in expert_tokens.values())

    return {
        "capacity": capacity,
        "expert_tokens": dict(expert_tokens),
        "rerouted": rerouted,
        "dropped": dropped,
        "kept": total_kept,
        "overflow_rate": len(dropped) / n,
    }


# 玩具数据：16 个 token，4 个专家
assignments =     [0,0,0,0,0,0, 1,1,1,1,1, 2,2,2, 3,3]
second_choices =  [1,1,1,2,2,3, 0,2,2,3,3, 1,3,3, 0,1]

result = route_with_capacity(
    assignments=assignments,
    second_choices=second_choices,
    num_experts=4,
    capacity_factor=1.25,
)

assert result["capacity"] == 5
assert result["kept"] + len(result["dropped"]) == 16
assert all(len(tokens) <= result["capacity"] for tokens in result["expert_tokens"].values())
assert result["overflow_rate"] >= 0.0

print(result)
```

这段代码体现的是最基础的“先装满，再处理剩下”。几个核心变量如下：

| 变量 | 含义 | 工程含义 |
|---|---|---|
| `capacity` | 每个专家最大可收 token 数 | 由 $C \cdot N / E$ 决定 |
| `expert_tokens` | 每个专家最终接到的 token | 进入专家计算的主集合 |
| `overflow_tokens` | 第一轮装不下的 token | 需要后处理 |
| `rerouted` | 成功重路由的 token | 质量通常优于直接丢弃 |
| `dropped` | 最终未进入任何专家的 token | 只能走残差或默认路径 |

如果把它改成真实训练代码，通常还会加两类信息：

- 每个 token 的 gate score，用于按分数而不是按出现顺序决定谁保留
- 每个专家所在设备的信息，用于估算通信代价

一个更接近工程实践的简化伪流程是：

1. 对每个 token 计算 top-k 专家分数  
2. 按主专家对 token 分桶  
3. 每个桶按 score 排序，只保留前 `capacity` 个  
4. 其余 token 尝试次选专家  
5. 次选也失败时，执行丢弃或 fallback expert 逻辑  

这比“来一个塞一个”的朴素贪心更稳，因为它至少保证分数高的 token 优先得到专家资源。

---

## 工程权衡与常见坑

容量因子最容易踩的坑，不是公式算错，而是把它当成一个孤立超参数。实际上它总是和路由偏置、padding 比例、通信拓扑、负载均衡损失绑在一起。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 容量太低 | 溢出率高，模型质量波动 | 提高 $C$，或优化路由分布 |
| 容量太高 | 显存和通信成本飙升 | 监控 padding ratio，不要只看丢弃率 |
| 路由偏向少数专家 | expert collapse，少数专家过热 | 加 balancing loss，检查门控温度和初始化 |
| 重路由纯贪心 | batch 打乱后结果不稳定 | 用 score-based reroute，必要时固定随机种子 |
| 只监控平均值 | 平均负载正常但局部专家爆满 | 监控每个专家的负载分布与 P99 |

这里最重要的工程权衡有两个。

第一，容量过低时，损失的不只是少量 token。因为溢出往往集中在“热门模式”上，也就是模型更常见、更关键的输入区域。看起来只丢了不到 1% 的 token，实际上可能主要丢在高频结构上，导致效果下降比数字看上去更明显。

第二，容量过高时，浪费的不只是显存。很多实现为了并行效率，会给每个专家预留固定大小张量。于是专家哪怕只收到很少 token，也要按上限做 padding。这个 padding 在单机上浪费算力，在多机上还会放大通信负担。

真实工程例子是：某个训练任务里，64 个专家中长期只有 2 到 4 个专家非常活跃，其余专家接近空闲。此时如果只把 $C$ 从 1.0 提到 1.5，表面上溢出率下降了，但由于活跃专家仍然过热，根本问题并没解决，反而让所有专家都预留了更大的缓冲区，吞吐下降。这种情况下，优先级应是先修路由分布，再考虑提高容量。

监控指标建议至少包括：

- `overflow_rate`：最终没进专家的 token 比例
- `reroute_rate`：被次选专家接收的比例
- `expert_load_histogram`：各专家实际负载分布
- `padding_ratio`：预留容量中实际未使用的比例
- `top_hot_expert_share`：最热几个专家占了多少 token

---

## 替代方案与适用边界

容量因子不是唯一方案，它只是“固定上限”的一种工程化表达。不同场景下，最合适的溢出处理策略并不相同。

| 方案 | 适合场景 | 优点 | 边界 |
|---|---|---|---|
| 直接丢弃并走残差 | 低延迟推理、小显存设备 | 最简单、最稳 | token 质量损失最直接 |
| 次选专家重路由 | 训练期、希望少丢 token | 更充分利用空闲专家 | 逻辑和调试成本更高 |
| 备用专家兜底 | 延迟敏感但不能大量丢 token | 折中较好 | 需要额外专家规划 |
| 提高 top-k | 希望提升路由弹性 | 更容易分流 | 计算和通信成本增加 |
| 强化负载均衡损失 | 长期路由偏斜明显 | 从源头减少溢出 | 不能替代容量控制 |

如果机器资源很紧，比如只有 8GB 显存的推理卡，通常不应该默认照搬训练时的 $C=1.25$。更现实的做法往往是：

- 把 $C$ 控制在 1.0 或接近 1.0
- 溢出 token 直接走残差
- 优先保证延迟稳定和显存不爆

反过来，如果是训练大模型，目标是整体收敛质量，那么更合理的选择可能是：

- 用适中的 $C$，例如 1.1 到 1.25
- 配合 balancing loss
- 为溢出 token 提供次选重路由
- 长期监控专家冷热分布

需要强调的适用边界是：如果模型规模很小、专家数很少，或者根本没有明显的专家拥塞问题，那么复杂的 overflow 处理未必值得。因为你增加的控制逻辑、调试成本和系统复杂度，可能比节省下来的 token 浪费更贵。

---

## 参考资料

| 资料 | 主要贡献 |
|---|---|
| Aman.ai, *Mixture-of-Experts Primer* | 系统整理了容量因子定义、典型取值和 Switch Transformer 的工程经验 |
| Switch Transformer 相关设计说明 | 展示了 top-1 路由、容量限制与 token 丢弃在大规模训练中的折中 |
| mbrenndoerfer, *Expert Networks / MoE Architecture / FFN Implementation* | 总结了溢出 token 的常见处理策略，包括丢弃、重路由和备用专家 |
| mbrenndoerfer, *Auxiliary Balancing Loss in MoE* | 解释了负载均衡损失如何缓解 expert collapse |
| *Maximum Score Routing for Mixture-of-Experts*（ACL 2025） | 给出了基于分数的重路由思路，适合理解“谁应优先占用有限专家容量” |

可直接查阅的来源：

- Aman.ai: https://aman.ai/primers/ai/mixture-of-experts/
- mbrenndoerfer: https://mbrenndoerfer.com/writing/expert-networks-moe-architecture-ffn-implementation
- mbrenndoerfer: https://mbrenndoerfer.com/writing/auxiliary-balancing-loss-mixture-of-experts-moe
