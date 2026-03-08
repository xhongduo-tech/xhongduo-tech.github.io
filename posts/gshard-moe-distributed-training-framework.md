## 核心结论

GShard 是把 MoE 真正推进到超大规模分布式训练中的关键工程框架。MoE 的基本思想是：一层里不再只有一个统一的前馈网络，而是有很多个专家前馈网络；每个 token 只激活其中少数几个。GShard 的核心贡献不是发明专家，而是把“稀疏路由 + 跨设备分发 + 自动分片”连成一条能在大规模 TPU 集群上稳定运行的训练链路。

它的基本做法可以概括为两步：

1. 用门控网络 `gate` 为每个 token 选出 Top-2 专家。
2. 用 All-to-All 通信把 token 发送到持有目标专家的设备，本地完成专家计算，再把结果发回原位置并加权合并。

因此，GShard 实现的是 Expert Parallelism，也就是专家并行。它把“总参数量”与“单 token 实际计算量”拆开处理：总参数量可以很大，因为专家分散在不同设备上；而每个 token 的计算量仍然只和少数被激活的专家有关。

论文中的代表性结果是：约 600B 参数的多语种翻译模型，训练 FLOP 仍可控制在接近 12B 稠密 Transformer 的水平，但翻译质量明显更高。这个结果说明，稀疏模型最重要的价值不是“参数更省”，而是“总参数很多，但单次只激活很少一部分参数”。

先用一张表把三类方案区分清楚：

| 方案 | 总参数量 | 每个 token 激活参数 | 主要通信 | 扩展性瓶颈 |
|---|---:|---:|---|---|
| 稠密 Transformer | 中等 | 全部激活 | 低到中 | 显存和单设备算力 |
| 传统 MoE | 很大 | 少数专家激活 | 中到高 | 路由、负载均衡、实现复杂度 |
| GShard | 很大 | Top-2 专家激活 | 高，核心是 All-to-All | 互联带宽、容量管理、路由稳定性 |

如果把 2048 个 TPU 看成一个大集群，那么 GShard 做的事情可以直说成一句话：`gate` 决定每个 token 该去哪些专家，系统负责把 token 送到对应设备、算完、再送回。它解决的是“专家已经分散在很多设备上之后，稀疏计算怎么真的跑起来”。

---

## 问题定义与边界

GShard 要解决的问题可以精确定义为：

> 在专家分布于大量设备的前提下，如何让每个 token 只访问少量专家，同时保持训练可扩展、通信可控、负载不过度倾斜，并且不让专家路由把系统吞吐拖垮。

这个定义里有四个边界条件。

### 1. 专家数通常大于单机可承载范围

如果总共有 $E$ 个专家，而单台设备只能容纳其中一部分，那么专家参数必须分散在多个设备上。此时 token 不可能总在本地完成计算，而必须跨设备访问目标专家。

设每台设备最多容纳 $E_{local}$ 个专家，若

$$
E > E_{local}
$$

则专家层天然要求跨设备分片。这和普通数据并行不同。数据并行是“模型复制多份”；专家并行是“不同专家本来就分散在不同设备”。

### 2. 每个 token 只激活少量专家

GShard 使用 Top-2 gating，即每个 token 最多进入两个专家。设隐藏表示为 $x \in \mathbb{R}^d$，专家集合大小为 $E$，则门控网络先输出一个长度为 $E$ 的打分向量：

$$
g = W_g x
$$

再经过 softmax 得到路由概率：

$$
p_i = \frac{\exp(g_i)}{\sum_{j=1}^{E}\exp(g_j)}
$$

最后只保留概率最高的两个专家。于是，每个 token 的专家计算复杂度从“全部 $E$ 个专家都算”降成“至多 2 个专家参与”。这就是稀疏激活成立的前提。

### 3. 每个专家有容量上限

容量的含义是：某一轮中，一个专家最多接收多少个 token。因为张量通信和本地计算通常需要固定形状，如果不预先给每个专家设定容量，热门专家会导致动态形状膨胀，系统实现会非常困难。

设组内 token 数为 $T$，每个 token 选 $k$ 个专家，专家总数为 $E$，则平均到每个专家的理论负载近似为：

$$
\frac{T \cdot k}{E}
$$

工程实现通常再乘一个放大系数 `capacity_factor`，得到实际容量：

$$
C = \left\lceil \frac{T \cdot k}{E} \cdot \text{capacity\_factor} \right\rceil
$$

这个公式不是为了给出“真实自然规律”，而是为了给系统分配一个固定大小的槽位张量。若某个专家收到的 token 数超过 $C$，超出的 token 就会溢出。

### 4. 负载均衡不是可选项

如果 gate 长期把大量 token 都发给少数专家，那么即使总专家数很多，实际也会退化成“只有少数专家在工作”。因此，GShard 会配合辅助负载均衡损失，让路由概率和实际分配都尽量均匀。

常见记号是：

- $m_e$：专家 $e$ 实际接收的 token 占比
- $P_e$：专家 $e$ 的平均门控概率

GShard/Switch 系列常用的辅助思想可以写成：

$$
L_{aux} \propto E \sum_{e=1}^{E} m_e P_e
$$

直观解释是：如果某个专家拿到了很高的概率质量，也确实接收了很多 token，那么这个乘积会变大；训练会被推动去压平这种过度集中。

下面这张表把边界整理成工程视角：

| 维度 | GShard 的边界 | 为什么重要 |
|---|---|---|
| 专家数 vs 设备数 | 专家通常分散在多设备上 | 决定是否需要跨设备路由 |
| 路由方式 | 每个 token 只选 Top-2 | 决定稀疏计算能否控制 FLOP |
| 通信方式 | 依赖 All-to-All 重分片 | 决定系统能否把 token 送到目标专家 |
| 容量控制 | 每个专家每轮最多接收 $C$ 个 token | 决定固定形状张量能否成立 |
| 溢出处理 | 超容量 token 可能被丢弃或只走残差 | 决定训练有效样本是否受损 |
| 训练目标 | 大参数量下仍保持可训练和可扩展 | 决定 GShard 是否有工程价值 |

对新手来说，可以把这一节记成一句话：**GShard 不是“让所有专家一起讨论”，而是“每个 token 只找很少几个专家，并且系统必须保证这些 token 真能被送过去、算完、再送回来”。**

---

## 核心机制与推导

GShard 的实际链路可以拆成五步：`Gate -> 分桶 -> All-to-All -> 本地专家计算 -> All-to-All 回收与合并`。下面按这个顺序展开。

### 1. Top-2 gating

给定 token 表示 $x$，门控网络计算全部专家的打分：

$$
g = W_g x
$$

做 softmax 后得到专家概率：

$$
p_i = \frac{\exp(g_i)}{\sum_j \exp(g_j)}
$$

第一专家一般取概率最大的那个：

$$
e_1 = \arg\max_i p_i
$$

GShard 的关键细节在于第二专家。第二专家并不是简单固定选“第二名然后永远激活”，而是会带有随机路由的成分。直观写法可以表示成：

$$
e_2 = \text{second\_best}(p), \qquad
z_2 \sim \operatorname{Bernoulli}(r_2)
$$

其中 $r_2$ 与第二专家对应的门控概率相关，只有当 $z_2 = 1$ 时，第二专家才真正参与本轮计算。于是最终激活集合是：

$$
\mathcal{K}(x) = \{e_1\} \cup \left(\{e_2\} \text{ if } z_2 = 1 \text{ else } \varnothing \right)
$$

这个随机化有两个工程作用：

1. 它不是纯贪心路由，能给次优专家持续分到样本。
2. 当主专家过热时，第二专家有机会分担一部分流量。

如果第二专家概率很低，那么它大概率不会触发，行为就退化到接近 Top-1。也就是说，Top-2 不是“强行双专家”，而是“主专家必选，第二专家按概率参与”。

### 2. 一个完整玩具例子

假设有 4 台设备，总共 32 个专家，每台设备放 8 个专家。一个 batch 内有 64 个 token，每个 token 尝试选择 Top-2 专家。

平均每个专家应接收的 token 数近似为：

$$
\frac{64 \times 2}{32} = 4
$$

若 `capacity_factor = 1.25`，则每个专家容量约为：

$$
C = \left\lceil 4 \times 1.25 \right\rceil = 5
$$

现在看某个 token 的路由概率：

| 专家编号 | 概率 |
|---|---:|
| 7 | 0.62 |
| 19 | 0.30 |
| 3 | 0.05 |
| 其他 | 很小 |

这个 token 一定会选专家 7 作为第一专家，第二专家候选是 19。若第二专家被触发，那么这个 token 会被复制成两份路由记录：

| token | 目标专家 | 权重 |
|---|---:|---:|
| `t` | 7 | 0.62 |
| `t` | 19 | 0.30 |

这时关键问题出现了：专家 7 和专家 19 很可能在不同设备上。于是，原先存放在设备 A 上的 token 表示，必须被发往设备 B 和设备 C。专家算完后，结果还要回传到原始 token 所在位置并按权重合并。

新手最容易忽略的一点是：**MoE 的难点不是挑专家，而是挑完之后数据怎么跨设备移动。**

### 3. Dispatch 与 All-to-All 的张量重排

从系统视角看，路由后的 token 需要先按专家分桶，再在设备之间交换。论文里通常把这一过程写成张量维度重排。

设：

- $G$：group，设备组
- $S$：组内 token 数
- $E$：专家数
- $C$：每个专家容量
- $M$：隐藏维度

在 dispatch 之前，张量更接近“按 token 视角组织”，可抽象为：

$$
[G, S, M]
$$

在建立路由索引和容量槽位后，可构造成“按专家装箱”的中间形式：

$$
[G, E, C, M]
$$

然后通过 All-to-All，把每个设备上属于其他设备专家的 token 分片交换出去，使数据从“原 token 所在设备视角”变成“目标专家所在设备视角”。可以把这个过程记成：

$$
\text{dispatch}: [G, S, M] \rightarrow [G, E, C, M]
$$

$$
\text{all-to-all}: [G, E, C, M] \rightarrow [E_{local}, G, C, M]
$$

白话解释是：原本每台设备拿着“自己这一批 token”；通信之后，每台设备拿到的是“属于自己本地专家的那部分 token”。

这一步之所以是 All-to-All，而不是普通 gather/scatter，是因为每台设备都既要发送给很多设备，也要从很多设备接收回来。所有设备彼此交换各自的一部分数据，正是 All-to-All 的定义。

### 4. 本地专家计算与回收

当 token 已经被送到目标专家所在设备后，本地计算就简单了。每个专家本质上就是一段普通 FFN：

$$
E_e(x) = W_{2,e} \, \sigma(W_{1,e}x)
$$

如果某个 token 进入了两个专家，那么会得到两个专家输出：

$$
y_{e_1} = E_{e_1}(x), \qquad y_{e_2} = E_{e_2}(x)
$$

最后按门控权重加权合并：

$$
y = \alpha_1 y_{e_1} + \alpha_2 y_{e_2}
$$

其中

$$
\alpha_i = \frac{p_{e_i}}{\sum_{j \in \mathcal{K}(x)} p_j}
$$

如果第二专家没有触发，那么集合 $\mathcal{K}(x)$ 中只有一个专家，这个式子就退化为单专家输出。

完整流程可以压缩成下表：

| 阶段 | 输入视角 | 做什么 | 输出视角 |
|---|---|---|---|
| Gate | token | 计算专家概率并选 Top-2 | token 带专家标签 |
| Dispatch | token | 按专家分桶并放入容量槽位 | 待发送分片 |
| All-to-All | 设备间 | 把 token 发往目标专家设备 | 专家聚合后的 token |
| Local compute | 专家 | 本地执行 FFN | 专家输出 |
| Combine | token | 结果回传并加权合并 | 恢复原 token 顺序 |

### 5. 为什么 GShard 能扩到 2048 个 TPU

如果仍然只用纯数据并行，那么每台设备都要保留同一份完整模型。模型一旦做到数百亿或数千亿参数，很快就会先撞上显存限制，再撞上参数同步与优化器状态开销。

GShard 的思路是：

1. 把稠密 FFN 换成很多个专家 FFN。
2. 把这些专家参数分散到大量设备上。
3. 每个 token 只激活两个专家，因此单步计算量不随总专家数线性上涨。

所以，GShard 的本质不是“把模型做小”，而是把下面这两个量拆开：

$$
\text{总参数量} \quad \text{vs.} \quad \text{单 token 激活计算量}
$$

这是 GShard 与普通大模型并行最大的区别。普通并行通常只是在想“一个大稠密模型怎么切开”；GShard 则是在想“有没有必要让每个 token 都看到全部参数”。它的回答是：没有必要，只让 token 访问少数相关专家即可。

---

## 代码实现

下面给出一个可直接运行的 Python 玩具实现，演示四件事：

1. Top-2 路由
2. 第二专家按概率触发
3. 容量限制与 token 溢出
4. 专家输出的回收与加权合并

它不是 TensorFlow/XLA 的真实 GShard 实现，也没有真正执行跨设备 All-to-All，但把 GShard 的核心数据流缩成了单机可运行的模拟器。直接用标准库即可运行。

```python
import math
import random
from typing import Dict, List, Tuple


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def compute_capacity(num_tokens: int, top_k: int, num_experts: int, capacity_factor: float) -> int:
    return math.ceil(num_tokens * top_k / num_experts * capacity_factor)


def select_top2_with_random_second(logits: List[float], rng: random.Random) -> Tuple[List[int], List[float]]:
    probs = softmax(logits)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    first, second = order[0], order[1]

    selected = [first]
    # 用第二名对应概率决定是否激活第二专家，模拟 GShard 的 random routing 思路。
    if rng.random() < probs[second]:
        selected.append(second)
    return selected, probs


def renorm_weights(selected: List[int], probs: List[float]) -> Dict[int, float]:
    total = sum(probs[i] for i in selected)
    return {i: probs[i] / total for i in selected}


def expert_ffn(expert_id: int, token_value: float) -> float:
    # 用可解释的标量映射代替真实两层 MLP，不依赖外部库。
    scale = 1.0 + 0.1 * expert_id
    bias = 0.25 * expert_id
    hidden = max(0.0, token_value * scale + bias)
    return 0.5 * hidden + expert_id


def simulate_moe_layer(
    token_values: List[float],
    batch_logits: List[List[float]],
    num_experts: int,
    capacity_factor: float,
    seed: int = 0,
) -> Tuple[List[float], Dict[int, List[Tuple[int, float]]], List[int], int]:
    rng = random.Random(seed)
    capacity = compute_capacity(
        num_tokens=len(token_values),
        top_k=2,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    expert_slots: Dict[int, List[Tuple[int, float]]] = {e: [] for e in range(num_experts)}
    token_assignments: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(token_values))}
    dropped_tokens: List[int] = []

    # Dispatch: token -> expert slots
    for token_id, logits in enumerate(batch_logits):
        selected, probs = select_top2_with_random_second(logits, rng)
        weights = renorm_weights(selected, probs)

        accepted = 0
        for expert_id in selected:
            if len(expert_slots[expert_id]) < capacity:
                weight = weights[expert_id]
                expert_slots[expert_id].append((token_id, weight))
                token_assignments[token_id].append((expert_id, weight))
                accepted += 1

        if accepted == 0:
            dropped_tokens.append(token_id)

    # Local expert compute + combine
    outputs: List[float] = []
    for token_id, token_value in enumerate(token_values):
        routes = token_assignments[token_id]
        if not routes:
            # 溢出后走残差旁路：玩具实现里直接返回原值。
            outputs.append(token_value)
            continue

        merged = 0.0
        for expert_id, weight in routes:
            merged += weight * expert_ffn(expert_id, token_value)
        outputs.append(merged)

    return outputs, expert_slots, dropped_tokens, capacity


def main() -> None:
    token_values = [0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
    batch_logits = [
        [3.2, 2.4, 0.1, 0.1],
        [3.0, 2.5, 0.1, 0.1],
        [2.8, 2.6, 0.2, 0.1],
        [2.7, 2.4, 0.5, 0.1],
        [0.2, 3.1, 2.8, 0.1],
        [0.1, 3.0, 2.7, 0.2],
        [0.1, 0.4, 3.2, 2.5],
        [0.1, 0.3, 3.1, 2.7],
    ]

    outputs, expert_slots, dropped_tokens, capacity = simulate_moe_layer(
        token_values=token_values,
        batch_logits=batch_logits,
        num_experts=4,
        capacity_factor=1.0,
        seed=42,
    )

    loads = {expert_id: len(items) for expert_id, items in expert_slots.items()}

    assert capacity == 4
    assert all(load <= capacity for load in loads.values())
    assert len(outputs) == len(token_values)
    assert sum(loads.values()) >= len(token_values) - len(dropped_tokens)

    print("capacity =", capacity)
    print("expert loads =", loads)
    print("dropped tokens =", dropped_tokens)
    print("outputs =", [round(x, 4) for x in outputs])


if __name__ == "__main__":
    main()
```

在本地运行这段代码，会得到一组稳定输出，能观察到：

- 每个 token 至多进入两个专家。
- 每个专家的接收数不会超过容量。
- 容量较小或路由过于集中时，会出现 `dropped tokens`。
- 被丢弃的 token 在这个玩具实现里走“残差旁路”，直接保留原值。

这段代码和真实 GShard 的关系，可以用下表对应：

| 玩具实现组件 | 对应 GShard 概念 | 真实系统里的复杂点 |
|---|---|---|
| `select_top2_with_random_second` | Top-2 gating + random routing | 真正实现要可并行、可微分、可批量化 |
| `expert_slots` | 专家容量槽位 | 通常要映射成固定形状张量 |
| `simulate_moe_layer` 中的 dispatch | token 分桶 | 实际会跨设备发送 |
| `expert_ffn` | 本地专家计算 | 真实是两层 MLP 或更复杂 FFN |
| `merged += weight * ...` | combine 阶段 | 真实实现还涉及张量还原与反向传播 |

如果写成更接近论文流程的伪代码，可以压成下面几行：

```python
def moe_layer(inputs):
    logits = gate(inputs)
    top2, probs = select_top2_with_random_second(logits)
    dispatch_tensor = build_dispatch_tensor(top2, capacity)
    expert_inputs = all_to_all(dispatch_tensor)
    expert_outputs = local_expert_ffn(expert_inputs)
    gathered_outputs = all_to_all(expert_outputs)
    return combine(gathered_outputs, probs)
```

关键超参数的工程含义如下：

| 参数 | 含义 | 调大后的直接效果 | 主要风险 |
|---|---|---|---|
| `num_experts` | 专家总数 | 总参数容量增大 | 通信更重，专家更难均衡 |
| `top_k` | 每个 token 激活专家数 | 表达能力增强 | 通信、合并、显存占用增加 |
| `capacity_factor` | 容量放大倍数 | token 溢出减少 | padding 增多，内存和通信更浪费 |
| `group_size` | 一次路由的设备组规模 | 更大组通常带来更强专家共享 | All-to-All 压力更大 |
| `expert_parallelism` | 专家跨设备分片程度 | 支持更大总参数量 | 更依赖高速互联和编译器分片能力 |

对新手来说，这一节最重要的收获不是“记住每一行代码”，而是记住数据流顺序：**先选专家，再装箱分桶，再跨设备交换，再本地算专家，再回收合并。**

---

## 工程权衡与常见坑

GShard 在论文中看上去很整齐，但工程里真正棘手的地方几乎都集中在三个词上：**容量、负载、通信**。下面按最常见的问题展开。

### 1. 容量太小会直接伤害有效训练

仍以上面的例子为例，若有 64 个 token、32 个专家、Top-2 路由，则理论平均负载是：

$$
\frac{64 \times 2}{32} = 4
$$

如果 `capacity_factor = 1.0`，那么每个专家容量约为 4。只要 gate 对某几个专家稍微偏好一点，这些专家就会迅速打满。超出的 token 只能被丢弃，或者走残差旁路，相当于“这一层没有真正参与稀疏学习”。

这会带来两个后果：

1. 有效训练信号减少，因为部分 token 没有进入专家。
2. 热门专家对输入分布的学习被截断，梯度统计不稳定。

### 2. 容量太大也不是免费午餐

把 `capacity_factor` 从 1.0 提高到 1.25、1.5 甚至 2.0，确实能减少溢出，但代价也很直接：

1. 每个专家都要预留更大的槽位张量，显存占用上升。
2. 很多容量槽位实际上没有填满，形成 padding 浪费。
3. All-to-All 交换的数据量变大，通信效率下降。

所以容量的本质是一个折中问题：

$$
\text{小容量} \Rightarrow \text{更省资源，但更容易溢出}
$$

$$
\text{大容量} \Rightarrow \text{更少丢 token，但更浪费内存和带宽}
$$

### 3. 负载均衡辅助 loss 不是装饰

如果不加负载均衡约束，最常见的退化就是“专家塌缩”：少数专家一直很忙，其他专家长期没什么 token。这样会出现明显的正反馈：

1. 某些专家先因为随机初始化拿到更多 token。
2. 它们更新更频繁，表现变得更强。
3. gate 更愿意把 token 发给这些专家。
4. 其他专家越来越拿不到样本。

所以辅助 loss 不是论文里的点缀，而是防止路由系统退化的必要部件。它的目标不是强行让每个专家完全一样，而是避免分配极端倾斜。

### 4. All-to-All 往往才是真瓶颈

很多人第一次看 MoE 会把注意力放在“FLOP 变少了”，但真实系统里经常更先撞上的，是跨设备交换带来的延迟和带宽瓶颈。

原因很简单：

- token 本来是按原始 batch 分布在各个设备上；
- 路由以后，它们必须被重新按专家聚合；
- 算完专家后，还要再交换回来。

这意味着一层 MoE 往往至少包含两次大的跨设备数据重排。若互联带宽不够，或者设备拓扑不适合大规模全交换，那么系统会从“算力受限”变成“通信受限”。

### 5. 自动分片降低了门槛，但没有消掉复杂度

GShard 名字里的另一个重点是 automatic sharding。它通过编译器与分片注解，降低了开发者手写分布式切分逻辑的负担。但这不等于“分布式复杂度消失了”。你仍然需要面对：

- 哪一维按数据并行切
- 哪一维按专家并行切
- 哪些张量在 All-to-All 前后形状变化
- 负载不均时如何观察和诊断

也就是说，GShard 把实现方式从“手工搬运所有张量”提升到“让编译器帮你完成大部分分片”，但代价模型依然要理解。

下面这张表总结最常见的坑：

| 问题 | 表现 | 根因 | 常见解决手段 |
|---|---|---|---|
| 容量太小 | token 溢出、训练不稳定 | 热门专家迅速打满 | 提高 `capacity_factor`，观察溢出率 |
| 负载不均 | 少数专家很忙，其余专家空闲 | gate 过度偏置 | 辅助 loss、random routing、调初始化 |
| 通信过重 | 吞吐下降、设备空等 | All-to-All 成本高 | 分组路由、优化拓扑、提高互联带宽 |
| 专家塌缩 | 少量专家长期垄断 token | 路由正反馈失衡 | 加强均衡约束，监控专家利用率 |
| padding 过多 | 显存涨、有效吞吐下降 | 容量预留过大 | 调低容量或改进分组粒度 |

可以把工程调参的核心指标整理成一张运维表：

| 指标 | 说明 | 异常信号 |
|---|---|---|
| expert load variance | 专家负载方差 | 方差持续过大说明负载不均 |
| overflow rate | token 溢出比例 | 比例过高说明容量过小或路由过偏 |
| all-to-all latency | 一次全交换耗时 | 耗时过高说明通信成为主瓶颈 |
| tokens per expert | 每轮每专家样本数 | 长期接近 0 的专家可能在塌缩 |
| padding ratio | 容量槽位中空位比例 | 比例过高说明容量配置浪费 |

对第二专家概率，一个直观约束是：

$$
P(\text{route to } e_2) \approx p_{e_2}
$$

这表示第二专家不应被完全忽略，也不应被无条件强行加入。它应该大体服从门控分数，从而在“主专家稳定”与“次优专家获得样本”之间保持平衡。

---

## 替代方案与适用边界

GShard 不是默认答案。它成立的前提是：你确实需要很大的总参数量，同时你的硬件和互联也确实能承受稀疏路由的系统代价。否则，它可能比稠密模型更复杂，却不一定更划算。

### 什么时候不该优先用 GShard

如果你的训练规模大致落在下面这些条件里，GShard 往往不是第一选择：

- 设备数量不多，例如只有几台 GPU。
- 模型参数量还在普通张量并行或流水并行可覆盖的范围内。
- 训练重点是快速迭代、简单调试，而不是极限扩展。
- 部署场景对端到端延迟非常敏感，不愿承担复杂路由与通信。

这时引入 GShard，往往会把问题从“模型怎么训练”变成“系统怎么维护”。复杂度增加得很快，但收益未必同步增加。

### 什么时候 GShard 合适

GShard 更适合下面这类场景：

- 目标模型容量非常大，稠密模型已经明显撞到显存或算力墙。
- 希望总参数量继续扩大，但单 token FLOP 不要同步爆炸。
- 训练环境具备高带宽互联，能承受大规模 All-to-All。
- 团队能接受 MoE 特有的监控、调参与故障诊断成本。

尤其在 TPU 这类互联较强、编译器分片能力较成熟的环境里，GShard 的工程收益更容易兑现。

### 和其他方案的比较

| 方案 | 是否稀疏激活 | 是否依赖 All-to-All | 是否有容量控制 | 适用场景 |
|---|---|---|---|---|
| 稠密 Transformer | 否 | 否 | 否 | 中小规模训练，系统最简单 |
| 纯数据并行 | 否 | 否 | 否 | 模型能完整放下，追求实现简单 |
| 张量并行 / 流水并行 | 否 | 有，但不是专家路由式全交换 | 否 | 稠密大模型扩展 |
| 传统 MoE | 是 | 可能有 | 不一定完善 | 中等规模稀疏模型探索 |
| GShard | 是，Top-2 | 是，属于核心机制 | 是 | 超大规模专家并行训练 |
| Switch Transformer | 是，Top-1 | 是，但路由更简化 | 是 | 想降低 Top-2 的复杂度与通信开销 |

可以把适用边界直接记成三条：

1. 设备规模小、互联一般：优先稠密模型或普通并行。
2. 模型总参数很大，但想控制单 token FLOP：MoE 才有意义。
3. 互联足够强、能承受 All-to-All：GShard 才真正成立。

一句话总结这个边界就是：**如果只是训练普通中型模型，GShard 往往太重；如果要训练数百亿到数千亿参数、还希望单步计算维持在较低水平，GShard 这类稀疏专家并行才值得引入。**

---

## 参考资料

| 来源名称 | 类型 | 关键贡献简述 |
|---|---|---|
| GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | 论文 | 原始方案，给出 Top-2 路由、自动分片和大规模 TPU 上的专家并行训练 |
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | 论文 | 用 Top-1 路由简化 GShard 的部分复杂度，适合理解 GShard 之后的工程折中 |
| Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | 论文 | 更早期的稀疏 MoE 代表工作，适合理解“专家 + 门控”的基础起点 |
| 工程类解读文章与博客 | 二手资料 | 适合补足 All-to-All 数据流、容量控制和负载均衡的直观理解 |

1. Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, ICLR 2021 / arXiv:2006.16668。重点看论文中的 Top-2 gating、专家并行与 2048 TPU 扩展结果。
2. Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022 / arXiv:2101.03961。适合理解为什么后来很多系统从 Top-2 走向 Top-1。
3. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017 / arXiv:1701.06538。适合理解 MoE 早期门控与专家思想的源头。
4. 若你已经理解普通 Transformer 并行，但还不清楚 GShard 的系统代价，优先补“容量限制、专家负载均衡、All-to-All 通信”这三部分，因为它们比公式本身更决定工程成败。

{"summary":"GShard 用 Top-2 路由与 All-to-All，把 MoE 专家并行扩展到超大规模分布式训练。"}
