## 核心结论

Expert Choice 路由的核心变化只有一句话：**不再让 token 选择专家，而是让专家选择 token**。这里的 token 可以理解为“序列中的一个位置向量”，例如一句话里的一个词经过嵌入后得到的表示；专家可以理解为“一个专门处理某类模式的小前馈网络”。

这一步反转看起来只是路由方向变化，实质上改变了 MoE 的资源分配方式。传统 token→expert 路由里，负载均衡通常要靠额外损失、容量截断或 token 丢弃来补救；Expert Choice 直接先给每个专家固定名额，再让专家从全局 token 池里挑自己最擅长处理的 top-k，于是每个专家天然满载，负载几乎完美均衡。

关键公式是：

$$
k=\frac{n\times c}{e}
$$

其中，$n$ 是本批次 token 数，$e$ 是专家数，$c$ 是 capacity factor。这里的 capacity factor 可以直白理解为“平均每个 token 预算上想分配多少次专家计算”。于是每个专家固定处理 $k$ 个 token，全局总共承诺 $n\times c$ 个专家槽位。

这带来两个直接后果：

1. **负载均衡天然成立**：每个专家都处理恰好 $k$ 个 token。
2. **计算分配变成异质的**：同一个 token 可能被 0 个、1 个、2 个甚至更多专家选中。

第二点比第一点更重要。它意味着模型不再强迫“每个 token 平均分一口计算”，而是允许困难 token 获得更多专家处理，简单 token 少占资源，甚至完全走残差路径继续前传。

可以用一个新手版玩具比喻理解：16 个老师各自有 128 个名额，他们不是等学生报名，而是自己从全年级里挑最适合自己课程的学生。热门学生可能被多个老师同时看中，冷门学生可能一个老师都没选上，只能继续上公共课。**Expert Choice 的价值，不是让所有学生都被服务得一样，而是让资源自然流向更“值得投入”的 token。**

一个简化对比如下：

| token 类型 | 与专家匹配分数 | 被多少专家选中 | 实际计算量 | 结果 |
|---|---:|---:|---:|---|
| 热门 token | 高 | 2 到 4+ | 高 | 表达更细，适合复杂模式 |
| 普通 token | 中 | 1 到 2 | 中 | 获得常规专家增强 |
| 冷门 token | 低 | 0 | 低 | 仅走残差/主干路径 |

从公开结果看，Google 的 8B/64E Expert Choice 模型在与 Switch Transformer、GShard 等相近预算下训练时，收敛速度和下游 GLUE/SuperGLUE 表现都有优势。结论很明确：**Expert Choice 真正提供的不只是均衡，而是在固定预算下可学习的异质计算分配。**

---

## 问题定义与边界

先定义问题。MoE，Mixture of Experts，中文可理解为“专家混合模型”，本质是在一个层里放多个专家子网络，但每个 token 只激活其中一部分，而不是把所有专家都算一遍。目标是：**参数量变大，但单次计算量尽量可控**。

传统 MoE 的难点不在“有没有专家”，而在“token 怎么分给专家”。如果路由不稳，会出现三类常见问题：

| 问题 | 直白解释 | 后果 |
|---|---|---|
| 专家拥塞 | 很多 token 挤向少数专家 | 有些专家爆满，部分 token 被丢弃 |
| 专家闲置 | 一些专家几乎没人用 | 参数浪费，训练不充分 |
| 计算同质化 | 每个 token 获得差不多的专家预算 | 难以把算力集中到困难样本 |

Expert Choice 针对的是这组问题，但它也有边界。它解决的是**稀疏 FFN 层里的专家选择问题**，不是整个 Transformer 的所有计算调度问题。也就是说：

- 它通常作用在 MoE FFN 层，而不是替换自注意力机制。
- 没被专家选中的 token 仍然会通过残差路径向前传播，不等于“信息丢失”。
- 它保证的是专家侧负载均衡，不保证每个 token 都一定得到至少一个专家处理。
- 它控制的是**每个专家处理多少 token**，而不是直接控制**每个 token 处理多少次**。
- 它改善的是路由分配效率，不自动解决通信带宽、显存布局、并行策略等系统问题。

把变量统一一下：

| 符号 | 含义 | 示例 |
|---|---|---:|
| $n$ | 当前批次 token 数 | 1024 |
| $e$ | 专家数 | 16 |
| $c$ | capacity factor，平均专家预算 | 2 |
| $k$ | 每个专家固定选择的 token 数 | 128 |
| $n\times c$ | 全局专家槽总数 | 2048 |

代入公式：

$$
k=\frac{1024\times 2}{16}=128
$$

于是整个层的约束非常清楚：16 个专家，每个专家固定吃 128 个 token，总共 2048 个专家槽。**这里的“平均每 token 2 次”只是全局平均，不是逐 token 的硬约束。**

很多新手第一次读到这里会误解成“那是不是每个 token 平均会被两个专家处理，所以总体上差不多还是均匀的？”答案是否定的。平均数只约束总量，不约束个体分布。下面这个简单反例就足够说明问题：

| token | 被选中的专家数 |
|---|---:|
| t1 | 4 |
| t2 | 3 |
| t3 | 2 |
| t4 | 1 |
| t5 | 1 |
| t6 | 1 |
| t7 | 0 |
| t8 | 0 |

上表总和仍然可以满足平均值要求，但资源已经明显向少数 token 集中。**Expert Choice 利用的就是这种“总量固定、个体可变”的自由度。**

一个简化流程可以写成：

`批次 token -> 计算 token-专家 affinity -> 每个专家独立选 top-k -> 被选中的 token 进入对应专家 -> 未被选中的 token 走残差路径`

这就定义了问题边界：预算先固定，再让路由在预算内自由决定“谁多算、谁少算、谁不算”。

---

## 核心机制与推导

先看机制。设输入是一个批次的 token 表示矩阵 $X\in \mathbb{R}^{n\times d}$，其中 $d$ 是隐藏维度。路由器会为每个 token 和每个专家计算一个亲和分数，也常写成 affinity score：

$$
A\in \mathbb{R}^{n\times e}
$$

其中 $A_{t,j}$ 表示第 $t$ 个 token 与第 $j$ 个专家的匹配程度。这里的“亲和分数”可以白话理解为“这个专家觉得自己有多适合处理这个 token”。

一种常见写法是先用线性层得到路由 logits：

$$
G = XW_g \in \mathbb{R}^{n\times e}
$$

再经过可选的归一化或温度缩放得到 affinity：

$$
A_{t,j}=\frac{G_{t,j}}{\tau}
$$

其中 $\tau$ 是温度参数。$\tau$ 越小，分数排序越尖锐；$\tau$ 越大，分数越平滑。严格来说，Expert Choice 选 top-k 只依赖排序，因此不一定必须做 softmax；但在训练实现中，往往仍会配合归一化、噪声或正则来稳定路由分布。

传统路由一般是：对每个 token，在专家维度上选 top-1 或 top-2。Expert Choice 反过来：**对每个专家，在 token 维度上选 top-k**。

因此第 $j$ 个专家处理的 token 集合是：

$$
S_j=\operatorname{TopK}_{t}(A_{t,j}, k)
$$

而且

$$
|S_j|=k=\frac{n\times c}{e}
$$

这一步直接保证了每个专家处理的 token 数完全一致。于是专家负载方差理论上接近 0：

$$
\operatorname{Std}\big(|S_1|,\dots,|S_e|\big)\approx 0
$$

如果实现中没有额外裁剪、补位或异常过滤，那么更严格地说：

$$
|S_1|=|S_2|=\cdots=|S_e|=k
$$

因此：

$$
\operatorname{Var}\big(|S_1|,\dots,|S_e|\big)=0
$$

但对 token 来说，情况不同。一个 token $t$ 可能出现在多个专家集合中，其被选中的次数为：

$$
m_t=\sum_{j=1}^{e}\mathbf{1}[t\in S_j]
$$

这里的 $m_t$ 就是 token 的“专家占用数”。它可以是 0、1、2、3…… 这正是 Expert Choice 的异质性来源。全局上有：

$$
\sum_{t=1}^{n} m_t = \sum_{j=1}^{e}|S_j| = n\times c
$$

这条式子很关键。它说明总预算固定，但分配方式不固定。于是会出现：

- 有些 token：$m_t=0$
- 多数 token：$m_t=1$ 或 $2$
- 少数热点 token：$m_t\ge 3$

这不是 bug，而是设计目标。因为模型在学习中会逐渐把更多专家槽给“难处理、信息密度高、歧义强”的 token。

为了把这件事说得更完整，可以再看输出聚合。若 token $t$ 被专家集合 $\mathcal{E}(t)$ 选中，则该 token 的稀疏专家输出可以写成：

$$
y_t=\sum_{j\in \mathcal{E}(t)} \alpha_{t,j}\,E_j(x_t)
$$

其中：

- $E_j(\cdot)$ 表示第 $j$ 个专家网络
- $\alpha_{t,j}$ 表示聚合权重
- $\mathcal{E}(t)=\{j\mid t\in S_j\}$

若 $\mathcal{E}(t)=\varnothing$，则通常退化为 fallback 路径，例如：

$$
y_t = x_t
$$

或者：

$$
y_t = \operatorname{DenseFFN}(x_t)
$$

具体走哪条路径取决于实现。论文和工程实现里更常见的是保留主干残差，使未命中 token 仍然稳定前传。

### 玩具例子

假设有 8 个 token、4 个专家，$c=1.5$，则：

$$
k=\frac{8\times 1.5}{4}=3
$$

每个专家选 3 个 token，总共 12 个专家槽。假设最终结果是：

| token | 被多少专家选中 |
|---|---:|
| t1 | 3 |
| t2 | 2 |
| t3 | 2 |
| t4 | 1 |
| t5 | 1 |
| t6 | 1 |
| t7 | 1 |
| t8 | 1 |

总和是 12，满足预算。但资源明显不平均。t1、t2、t3 获得更多专家处理，说明这些 token 更“复杂”或更符合多个专家的模式。

如果把视角换到专家一侧，同一个例子也可以写成：

| 专家 | 选中的 token |
|---|---|
| e1 | t1, t2, t4 |
| e2 | t1, t3, t5 |
| e3 | t1, t2, t6 |
| e4 | t3, t7, t8 |

这个表说明两件事：

1. 每个专家都恰好处理 3 个 token，负载完全一致。
2. token t1 被多个专家同时看中，因此得到更多计算。

### 真实工程例子

论文和辅助材料中给出的一个代表性统计是：在某层里，约 70% 的 token 被 1 到 2 个专家处理，约 23% 被 3 到 4 个专家处理，约 3% 超过 4 个专家，还有一小部分为 0。这个分布说明 Expert Choice 并不追求“每个 token 两个专家刚刚好”，而是在固定预算下把额外算力集中给热点 token。

如果把一个长文档输入看成真实工程场景，就容易理解这种分配。比如：

- 标点、停用词、模板化 token，语义简单，可能不值得占用专家预算。
- 命名实体、跨句指代、数学符号、代码片段，往往更复杂，容易被多个专家同时选中。
- 主干自注意力仍然保底，因此冷门 token 即使 0 专家，也不会直接消失。

下面给出一个经验分布表：

| token 被选次数区间 | 占比含义 | 工程解释 |
|---|---|---|
| 0 | 极少数 | 走 fallback，表示专家都不认为值得专门处理 |
| 1 到 2 | 主体 | 常规 token，获得标准级增强 |
| 3 到 4 | 少数热点 | 难 token，值得更多计算 |
| >4 | 极少数高热点 | 高歧义或高信息密度 token |

伪代码框架如下：

```text
for each expert j:
    scores = affinity[:, j]
    top_tokens = topk(scores, k)
    assign expert j to top_tokens

for each token t:
    experts = assigned_experts[t]
    if experts is empty:
        output = residual_path(x_t)
    else:
        output = aggregate(expert_j(x_t) for j in experts)
```

如果想把它和传统 top-2 token routing 做一个一眼能懂的对比，可以看下面这张表：

| 维度 | token 选专家 | Expert Choice |
|---|---|---|
| 选择方向 | 对每个 token 选 top-k 专家 | 对每个专家选 top-k token |
| 负载控制对象 | token 激活数固定 | 专家容量固定 |
| 专家负载 | 可能高度不均 | 天然一致 |
| token 计算量 | 更接近固定 | 允许高度可变 |
| 典型补救手段 | 辅助均衡损失、丢弃、截断 | 主要调容量与路由尖锐度 |

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是生产级高性能代码，但足以准确说明 Expert Choice 的核心逻辑：先算 affinity，再按专家选 top-k，最后统计每个 token 被多少专家选中，并对未命中的 token 走 fallback。

```python
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def validate_affinity(affinity: Sequence[Sequence[float]]) -> None:
    if not affinity:
        raise ValueError("affinity must not be empty")

    num_experts = len(affinity[0])
    if num_experts == 0:
        raise ValueError("affinity must contain at least one expert column")

    for row_id, row in enumerate(affinity):
        if len(row) != num_experts:
            raise ValueError(
                f"row {row_id} has {len(row)} columns, expected {num_experts}"
            )


def compute_k(num_tokens: int, num_experts: int, capacity_factor: float) -> int:
    raw_k = num_tokens * capacity_factor / num_experts
    k = int(raw_k)

    if k <= 0:
        raise ValueError(
            f"k must be positive, got {k}. "
            f"Increase capacity_factor or batch size."
        )
    if k > num_tokens:
        raise ValueError(
            f"k must be <= num_tokens, got k={k}, num_tokens={num_tokens}"
        )
    return k


def expert_choice_routing(
    affinity: Sequence[Sequence[float]],
    capacity_factor: float,
) -> Dict[str, object]:
    """
    affinity: shape [num_tokens, num_experts]
    capacity_factor: average number of expert assignments per token
    """
    validate_affinity(affinity)

    num_tokens = len(affinity)
    num_experts = len(affinity[0])
    k = compute_k(num_tokens, num_experts, capacity_factor)

    token_assignment: List[List[int]] = [[] for _ in range(num_tokens)]
    expert_assignment: List[List[int]] = [[] for _ in range(num_experts)]

    for expert_id in range(num_experts):
        scored_tokens = [
            (float(affinity[token_id][expert_id]), token_id)
            for token_id in range(num_tokens)
        ]

        # 先按分数降序，再按 token_id 升序，避免并列时结果不稳定
        scored_tokens.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored_tokens[:k]

        for score, token_id in chosen:
            token_assignment[token_id].append(expert_id)
            expert_assignment[expert_id].append(token_id)

    dropped_tokens = [
        token_id
        for token_id, experts in enumerate(token_assignment)
        if len(experts) == 0
    ]
    experts_per_token = [len(experts) for experts in token_assignment]
    total_slots = sum(experts_per_token)

    assert total_slots == k * num_experts
    assert all(len(tokens) == k for tokens in expert_assignment)

    return {
        "k": k,
        "token_assignment": token_assignment,
        "expert_assignment": expert_assignment,
        "experts_per_token": experts_per_token,
        "dropped_tokens": dropped_tokens,
        "total_slots": total_slots,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
    }


def log_routing_stats(
    token_assignment: Sequence[Sequence[int]],
    num_experts: int,
    step: int,
) -> Dict[str, float]:
    loads = [0] * num_experts
    dropped = 0
    max_experts_for_token = 0

    for experts in token_assignment:
        if not experts:
            dropped += 1
        max_experts_for_token = max(max_experts_for_token, len(experts))
        for expert_id in experts:
            loads[expert_id] += 1

    num_tokens = len(token_assignment)
    experts_per_token = [len(experts) for experts in token_assignment]
    drop_rate = dropped / num_tokens
    mean_load = sum(loads) / num_experts
    load_var = sum((x - mean_load) ** 2 for x in loads) / num_experts
    load_std = math.sqrt(load_var)
    mean_experts_per_token = sum(experts_per_token) / num_tokens

    stats = {
        "step": float(step),
        "token_drop_rate": round(drop_rate, 4),
        "fallback_usage_rate": round(drop_rate, 4),
        "per_expert_load_std": round(load_std, 4),
        "mean_experts_per_token": round(mean_experts_per_token, 4),
        "max_experts_for_one_token": float(max_experts_for_token),
    }
    return stats


def main() -> None:
    affinity = [
        [0.90, 0.80, 0.10, 0.20],  # token 0: hot
        [0.80, 0.70, 0.30, 0.10],  # token 1
        [0.20, 0.90, 0.80, 0.20],  # token 2: hot
        [0.10, 0.20, 0.70, 0.80],  # token 3
        [0.30, 0.10, 0.20, 0.90],  # token 4
        [0.50, 0.40, 0.20, 0.30],  # token 5
        [0.10, 0.10, 0.10, 0.10],  # token 6: cold
        [0.60, 0.30, 0.40, 0.20],  # token 7
    ]

    result = expert_choice_routing(affinity, capacity_factor=1.5)

    assert result["k"] == 3
    assert result["total_slots"] == 12
    assert len(result["token_assignment"]) == 8
    assert len(result["expert_assignment"]) == 4
    assert sum(len(v) for v in result["token_assignment"]) == 12
    assert max(result["experts_per_token"]) >= 2

    stats = log_routing_stats(result["token_assignment"], num_experts=4, step=10)

    print("routing_result =", result)
    print("routing_stats  =", stats)


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出的重点应当满足下面几条：

| 检查项 | 期望结果 | 含义 |
|---|---|---|
| `k == 3` | 成立 | 每个专家固定处理 3 个 token |
| `total_slots == 12` | 成立 | 全局专家槽总数等于 $n \times c$ |
| `per_expert_load_std == 0.0` | 成立 | 每个专家负载完全一致 |
| `max(experts_per_token) >= 2` | 成立 | 至少有 token 被多个专家选中 |
| `token_drop_rate` 可能大于 0 | 允许 | 存在 fallback token 是正常现象 |

这段代码里的关键变量可以对应到真实实现：

| 张量或变量 | 形状 | 含义 |
|---|---|---|
| `affinity` | `[T, E]` | token 与专家的亲和分数 |
| `expert_assignment` | `E` 个列表 | 每个专家最终选中了哪些 token |
| `token_assignment` | `T` 个列表 | 每个 token 被哪些专家选中 |
| `experts_per_token` | `[T]` | 每个 token 实际获得的专家数 |
| `dropped_tokens` | `[D]` | 没有任何专家选择的 token |
| `total_slots` | 标量 | 全局已使用专家槽数 |

如果把这段玩具实现映射到真实训练框架，通常还需要补上下面几个组件：

| 组件 | 作用 | 新手容易忽略的问题 |
|---|---|---|
| 路由线性层 | 产生 $A_{t,j}$ | 只看结构，不看数值尺度，容易导致分数过尖 |
| `topk` | 为每个专家选 token | 并列分数的稳定性会影响可复现性 |
| gather/scatter | 把 token 发到专家、再写回原位置 | 多机通信开销可能成为瓶颈 |
| 聚合权重 | 合并多专家输出 | 需要明确是求和、平均还是加权和 |
| fallback 路径 | 处理 0 专家 token | 没有 fallback 会导致语义断裂 |

在工程实现里，通常还要处理下面三个部分：

1. `capacity_factor`  
   直白解释：平均给每个 token 预留多少次专家预算。值越大，token drop 越少，但算力和通信开销越高。

2. `temperature`  
   直白解释：控制 affinity 分布尖锐程度。温度过低，分数会过于极端，导致少数 token 被反复争抢，未命中 token 增多。

3. `fallback`  
   直白解释：没有被任何专家选中的 token 去哪里。常见做法是直接走残差路径，或保持主干 dense FFN/attention 继续前传。

训练时的监控代码可以写成：

```python
def pretty_print_stats(stats: dict) -> None:
    for key, value in stats.items():
        print(f"{key}: {value}")


result = expert_choice_routing(
    affinity=[
        [0.90, 0.80, 0.10, 0.20],
        [0.80, 0.70, 0.30, 0.10],
        [0.20, 0.90, 0.80, 0.20],
        [0.10, 0.20, 0.70, 0.80],
        [0.30, 0.10, 0.20, 0.90],
        [0.50, 0.40, 0.20, 0.30],
        [0.10, 0.10, 0.10, 0.10],
        [0.60, 0.30, 0.40, 0.20],
    ],
    capacity_factor=1.5,
)

stats = log_routing_stats(result["token_assignment"], num_experts=4, step=10)
pretty_print_stats(stats)
```

如果 `per_expert_load_std` 不是接近 0，往往说明你实现的不是严格 Expert Choice，或者 top-k 之后又做了额外裁剪。相反，如果 `per_expert_load_std` 很低，但 `token_drop_rate` 异常高，那就说明你的问题不在“专家均衡”，而在“预算或路由分布设置不合理”。

---

## 工程权衡与常见坑

Expert Choice 的优点很明显，但真正上线或训练时，工程问题也很明确。

首先，**不要把“平均每 token c 个专家”误写成“每个 token 固定 c 个专家”**。这两者差别非常大。前者保留异质性，后者会把路由重新拉回均匀分配。公开材料中，像 EC-CAP2 这类“限制每个 token 最多两个专家”的变体，在微调分数上会比原始 Expert Choice 更差，说明模型收益主要来自异质性，而不是单纯的均衡。

其次，**token drop 不是偶发现象，而是机制允许的自然结果**。但“允许存在”和“比例过高”是两回事。如果大量 token 都没有专家处理，说明路由过于尖锐或容量过低。

再往下看，Expert Choice 的难点并不只在算法，还在系统实现。因为一个 token 可能被多个专家同时选中，所以它可能需要被复制到多个专家设备上，这会直接放大通信和聚合成本。负载均衡问题虽然缓解了，但**路由复制和结果回写的成本**会更突出。

常见坑可以整理成表：

| 常见坑 | 现象 | 根因 | 应对策略 |
|---|---|---|---|
| 强行限制每 token 专家数 | 精度下降 | 异质性被削弱 | 保留可变专家数分配 |
| token drop 过高 | 很多 token 只走 fallback | `capacity_factor` 太低或温度过低 | 提高容量，调温度，做正则 |
| 路由过于尖锐 | 少数 token 吃掉大量专家槽 | affinity 分布塌缩 | 加温度、熵正则或分数裁剪 |
| 路由统计不稳定 | 不同 batch 波动很大 | 数据分布变化大 | 监控滑动平均与分层统计 |
| 通信开销升高 | 多机专家交换变重 | 热点 token 被多个专家重复发送 | 控制 capacity，优化 all-to-all |
| 聚合不一致 | 同一 token 多专家输出尺度失衡 | 权重归一化不清楚 | 明确使用 sum、mean 或 softmax 加权 |
| 实现结果不可复现 | 同分 token 选取顺序变动 | `topk` 并列处理不稳定 | 固定 tie-break 规则和随机种子 |

建议长期监控的指标至少包括：

- `token_drop_rate`
- `fallback_usage_rate`
- `per_expert_load_std`
- `mean_experts_per_token`
- `high_hot_token_ratio`，例如被 4 个以上专家选中的 token 比例
- `router_entropy`，观察路由分布是否过尖
- `all_to_all_bytes`，衡量通信是否正在成为瓶颈

这些指标之间最好一起看，而不是单独看。例如：

| 指标组合 | 可能含义 |
|---|---|
| `per_expert_load_std` 低，`drop_rate` 低 | 理想状态，既均衡又没有明显预算浪费 |
| `per_expert_load_std` 低，`drop_rate` 高 | 专家负载虽均衡，但容量不足或路由过尖 |
| `per_expert_load_std` 高，`drop_rate` 低 | 实现可能偏离严格 EC，或者后处理破坏了容量一致性 |
| `high_hot_token_ratio` 高，通信高 | 热点 token 被过度复制，系统成本在上升 |

一个真实工程例子是大模型预训练中的长上下文批次。序列里包含普通自然语言、代码、公式、URL、表格边界符等不同模式。此时 Expert Choice 往往会让代码标识符、数学符号、稀有词片段获得更多专家预算。如果温度设得太低，这些热点 token 会过度集中，导致普通 token 的 drop rate 上升。结果不是“热点处理更好了”，而是“整体路由失衡，fallback 压力过大”。

因此工程上要接受一个事实：**Expert Choice 优化的不是 token 公平性，而是预算利用率与表达力。** 如果你的业务场景要求每个 token 都必须获得同等专家处理，它反而未必合适。

还可以把几个关键超参数的影响概括成一张表：

| 参数 | 调大后的常见效果 | 风险 |
|---|---|---|
| `capacity_factor` | 平均专家预算增加，drop 下降 | 计算量和通信量上升 |
| `temperature` | 路由更平滑，热点争抢减弱 | 专家选择区分度可能下降 |
| 专家数 `e` | 参数容量更大，模式更细分 | 调度更复杂，单专家样本可能变少 |
| batch token 数 `n` | top-k 统计更稳定 | 显存和并行同步压力上升 |

---

## 替代方案与适用边界

最直接的替代方案是 Switch Transformer 和 GShard。

- Switch Transformer：每个 token 选 top-1 专家，实现简单，通信和聚合都轻。
- GShard：每个 token 选 top-2 专家，比 top-1 表达力更强，但仍是 token 主动选专家。
- Expert Choice：专家反向选 token，负载均衡最自然，但实现和调度更复杂。

对比如下：

| 方案 | 负载均衡 | 表达力 | 路由复杂度 | 部署成本 | 适用场景 |
|---|---|---|---|---|---|
| Switch top-1 | 依赖辅助损失 | 较低 | 低 | 低 | 延迟敏感、先跑通系统 |
| GShard top-2 | 中等 | 中等 | 中 | 中 | 想要更强表达力但保持稳定 |
| Expert Choice | 很强 | 高 | 高 | 高 | 大模型训练、希望利用 token 异质性 |

如果只看理论，Expert Choice 更优；但在工程上不是总该优先选。可以用一个简短决策流判断：

| 条件 | 更适合的方案 |
|---|---|
| 预算紧、先求简单稳定 | Switch |
| 想要比 top-1 更强表达力，但不想改太多调度 | GShard |
| 有足够训练预算，重视负载均衡和异质计算收益 | Expert Choice |

它特别适合下面这类任务：

- 大规模预训练
- 长上下文建模
- 输入复杂度差异很大的多模态或代码任务
- 希望把额外算力集中到“困难 token”的场景

它不那么适合下面这类约束：

- 在线推理强延迟敏感
- 分布式通信带宽很紧
- 团队暂时没有能力监控复杂路由统计
- 场景要求每个 token 都必须有确定、固定的专家服务次数

把三类方案再放到同一个判断框架里，会更容易做工程决策：

| 关注点 | Switch | GShard | Expert Choice |
|---|---|---|---|
| 首要目标 | 最小化实现复杂度 | 平衡复杂度与表达力 | 最大化预算利用率 |
| token 计算次数 | 基本固定 | 基本固定 | 可变 |
| 专家拥塞风险 | 高 | 中 | 低 |
| 是否依赖辅助均衡损失 | 强依赖 | 常依赖 | 依赖更弱 |
| 通信复制风险 | 低 | 中 | 中到高 |
| 对监控体系要求 | 低 | 中 | 高 |

因此更准确的判断不是“Expert Choice 一定更先进”，而是：**当你愿意为更复杂的路由系统付出实现和通信成本，并且任务确实存在明显 token 难度差异时，Expert Choice 才能把它的异质性优势转化为实际收益。**

最后用一句更工程化的话收束这部分：如果你的主要痛点是“专家总是挤爆或闲置”，Expert Choice 值得考虑；如果你的主要痛点是“推理延迟必须极稳、系统必须尽量简单”，那它往往不是第一选择。

---

## 参考资料

| 类型 | 资料 | 侧重点 |
|---|---|---|
| 论文 | Mixture-of-Experts with Expert Choice Routing | 公式、机制、负载均衡定义 |
| 博客 | Google Research: Mixture-of-Experts with Expert Choice Routing | 工程直觉、训练速度、对比实验 |
| 辅助材料 | Heterogeneity Matters 相关整理材料 | token 被多专家处理的经验分布、限制异质性的退化现象 |

1. Zhou 等，*Mixture-of-Experts with Expert Choice Routing*  
   价值：NeurIPS 2022 正式论文，给出 Expert Choice 的核心公式、路由定义和实验结果。  
   适合重点看：第 2 节路由定义、第 3 节实验对比、附录里的实现细节。  
   URL: https://papers.neurips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf

2. Google Research Blog, *Mixture-of-Experts with Expert Choice Routing*  
   价值：更适合工程视角阅读，解释为什么“专家选 token”能带来更好的负载均衡和更快收敛。  
   适合重点看：和 Switch、GShard 的对比图，以及 8B/64E 的训练效率说明。  
   URL: https://research.google/blog/mixture-of-experts-with-expert-choice-routing/

3. Heterogeneity Matters 相关辅助材料整理  
   价值：帮助理解 token 被 0、1、2、3... 个专家处理的经验分布，以及限制 token 专家数为何会伤害性能。  
   阅读提醒：这类材料通常更偏二手整理，适合辅助理解，不宜替代原论文。  
   URL: https://www.scribd.com/document/818239276/choice-routing

4. Fedus 等，*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*  
   价值：理解 token→expert 路由为什么需要额外负载均衡手段，便于和 Expert Choice 做对照。  
   URL: https://jmlr.org/papers/volume23/21-0998/21-0998.pdf

5. Lepikhin 等，*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*  
   价值：理解 top-2 token routing、多专家激活和分布式 MoE 系统实现的早期代表方案。  
   URL: https://arxiv.org/pdf/2006.16668.pdf

如果只保留最小阅读集，建议顺序是：

1. 先读 Google Research Blog，建立“专家选 token”的直观图景。
2. 再读 EC 原论文第 2 节和第 3 节，确认公式、预算约束和实验结果。
3. 最后对照 Switch 和 GShard，理解 Expert Choice 改变的到底是哪一层假设。
