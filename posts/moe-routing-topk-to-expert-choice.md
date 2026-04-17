## 核心结论

MoE（Mixture of Experts）中的路由问题，核心不是“为每个 token 找到最强专家”，而是“在有限容量约束下，让 token 分配既准确又稳定，还尽量不浪费专家吞吐”。

在稀疏 MoE FFN 层里，Router 会给每个 token 和每个 expert 打一个分。真正困难的地方不在打分，而在分数出来之后怎样分配容量。因为专家容量有限，而 token 偏好往往高度集中，所以路由策略一旦设计不当，就会出现两个直接后果：

1. 热门专家爆满，冷门专家闲置。
2. 一部分 token 因为容量冲突被丢弃，或者只能走残差旁路。

Token-Choice 的规则是“每个 token 自己选前 $K$ 个专家”。它的优点是局部、简单、天然因果：位置 $t$ 的 token 只根据当前位置的表示做决策，不需要看未来 token。但它的结构性弱点也很明显：路由约束落在 token 侧，而不是专家侧，所以全局负载是否均衡，只能靠辅助损失和容量冗余去缓解，不能由路由规则本身保证。

Expert-Choice 把选择方向反过来：不是 token 选 expert，而是每个 expert 从整个 batch 中选自己最想处理的前 $C$ 个 token。这样做的直接结果是：

1. 每个 expert 的容量都能被稳定用满。
2. 负载均衡由结构保证，而不只是由 loss“鼓励”。
3. token drop 会显著减少，或者转化成“少数 token 未被任何 expert 选中”的可监控覆盖问题。

先看最重要的对比：

| 维度 | Token-Choice | Expert-Choice |
|---|---|---|
| 路由方向 | 每个 token 选 expert | 每个 expert 选 token |
| 约束位置 | 限制每个 token 选几个 expert | 限制每个 expert 接几个 token |
| 单 token 激活 expert 数 | 通常固定为 $K$ | 可变，可能是 0、1、2... |
| 单 expert 容量利用 | 常不均衡，热点会溢出 | 固定容量，利用率更稳定 |
| token drop 风险 | 高，热点专家溢出就会发生 | 低，但可能出现 coverage 不足 |
| 因果性 | 适合严格自回归 | 直接用于严格自回归会有未来泄露风险 |
| 工程特征 | 简单、易部署、推理友好 | 训练吞吐更稳，多机并行更友好 |

如果只保留一句话，结论是：

**Expert-Choice 解决的是容量利用和负载均衡问题，代价是它通常依赖对整批 token 的全局视图，因此不能直接照搬到严格自回归推理。**

为了帮助新手建立直觉，可以先用“食堂窗口”的图像理解，但要明确它只是类比，不是定义：

- Token-Choice：每个人自己选窗口，大家可能都挤到同一个窗口。
- Expert-Choice：每个窗口主动叫固定人数，窗口负载天然更均匀，但前提是窗口先看到了整队人。

所以本文的核心不是比较“哪个更先进”，而是明确一件事：**两者优化的目标不同。Token-Choice 优先满足因果和在线决策；Expert-Choice 优先满足批量均衡和系统吞吐。**

---

## 问题定义与边界

先把问题形式化。

在 Transformer 的 MoE FFN 层中，Router 对每个 token 输出一个对各 expert 的评分矩阵：

$$
S \in \mathbb{R}^{T \times E}
$$

其中：

- $T$ 表示当前 batch 内参与路由的 token 数。
- $E$ 表示 expert 数量。
- $S_{t,e}$ 表示 token $t$ 分配给 expert $e$ 的匹配分数或路由 logit。

如果把第 $t$ 个 token 的隐藏状态记为 $h_t \in \mathbb{R}^d$，最常见的 router 形式就是一个线性层：

$$
S_{t,:} = h_t W_r + b
$$

其中 $W_r \in \mathbb{R}^{d \times E}$。如果再做 softmax，可得到门控概率：

$$
p_{t,e} = \frac{\exp(S_{t,e})}{\sum_{j=1}^{E}\exp(S_{t,j})}
$$

但是否 softmax，不改变本文讨论的核心矛盾。真正决定系统行为的是：**在容量约束下，谁拥有最终选择权。**

### 容量约束是什么

每个 expert 的容量记为 $C$，表示该轮最多能接收多少 token。常见设法是：

$$
C = \left\lceil \frac{T \cdot K}{E} \cdot \text{capacity\_factor} \right\rceil
$$

这里需要注意一个细节：

- 如果是 top-1 路由，平均总派发数接近 $T$，则常写成 $C \approx \lceil \frac{T}{E} \cdot \text{capacity\_factor} \rceil$。
- 如果是 top-$K$ 路由，每个 token 最多派到 $K$ 个 expert，总派发槽位平均是 $T \cdot K$，容量公式更准确的写法应包含 $K$。

其中 `capacity_factor` 是容量冗余系数。它不是“越大越好”，而是用来在均衡性和资源开销之间折中。

### 为什么 Token-Choice 会天然遇到容量冲突

Token-Choice 的决策是逐 token 独立完成的。每个 token 都会问：“我最想去哪几个 expert？”但没有任何一步会先问：“全局来看，这些 expert 会不会同时被太多 token 挤爆？”

因此，即使理论平均每个 expert 应处理 $\frac{T \cdot K}{E}$ 个 token，实际分配也可能严重偏斜。只要有一批 token 同时偏好某几个 expert，就会出现：

- 热点 expert 爆满。
- 冷门 expert 闲置。
- 多出来的 token 被 drop，或者被迫走旁路。

这正是 Switch Transformer 一类方法为什么要引入 load balancing loss 和 capacity factor 的原因。它们并不是“可有可无的小修饰”，而是在补 Token-Choice 的结构缺口。

### 一个最小例子

设：

- $T=4$
- $E=2$
- top-1 路由
- `capacity_factor = 1`

则平均每个 expert 容量为：

$$
C = \left\lceil \frac{4}{2} \right\rceil = 2
$$

假设四个 token 都更偏好 expert 0，则 Token-Choice 的结果可能是：

| token | 对 expert 0 分数 | 对 expert 1 分数 | 选择结果 |
|---|---:|---:|---|
| 0 | 0.99 | 0.20 | 选 expert 0 |
| 1 | 0.95 | 0.40 | 选 expert 0 |
| 2 | 0.92 | 0.70 | 选 expert 0 |
| 3 | 0.90 | 0.80 | 选 expert 0 |

由于 expert 0 最多只能接 2 个 token，所以最终会变成：

- expert 0 接收 token 0、1
- token 2、3 溢出
- expert 1 空闲

如果改成 Expert-Choice，则规则变成：

- expert 0 从全部 4 个 token 中选自己最喜欢的 2 个
- expert 1 也从全部 4 个 token 中选自己最喜欢的 2 个

这样即使 expert 1 不是任何 token 的第一选择，它仍然可以挑选自己评分较高的 token，整体覆盖会明显改善。

### 本文的讨论边界

Expert-Choice 不是“全面替代 Token-Choice”，它只是把约束换了位置，因此也换来新的边界：

| 维度 | 优势 | 代价 |
|---|---|---|
| 负载均衡 | 每个 expert 容量固定，利用更稳定 | 需要全局比较 token 分数 |
| token 覆盖 | 更少出现热点爆满导致的大量丢弃 | 仍可能存在 token 完全未被选中 |
| 系统吞吐 | dispatch shape 更稳定，多机更友好 | 实现复杂度更高 |
| 因果性 | 适合训练和非因果并行场景 | 严格自回归下会触发未来信息依赖 |

因此，本文讨论的是以下边界内的问题：

- 训练阶段，batch 大，专家多，希望提高设备利用率。
- 非严格在线决策，可以接受按整批 token 做路由。
- 重点关注负载均衡、通信效率和 token 覆盖，而不是只关注单 token 的局部最优。

---

## 核心机制与推导

### 1. Token-Choice 的数学形式

Token-Choice 的核心操作是对评分矩阵的每一行做 top-$K$。

给定：

$$
S \in \mathbb{R}^{T \times E}
$$

第 $t$ 个 token 的评分向量为：

$$
S_{t,:} \in \mathbb{R}^{E}
$$

Token-Choice 的定义是：

$$
\mathcal{E}_t = \operatorname{TopK}(S_{t,:})
$$

其中 $\mathcal{E}_t$ 表示 token $t$ 选择的 expert 集合。

如果采用带权门控，则常见组合形式为：

$$
y_t = \sum_{e \in \mathcal{E}_t} g_{t,e} \cdot f_e(h_t)
$$

其中：

- $f_e(\cdot)$ 是第 $e$ 个 expert 的 FFN。
- $g_{t,e}$ 是对应 gate 权重，可能来自 softmax 后再归一化。
- $y_t$ 是该 MoE 层对 token $t$ 的输出。

从表达式上看，这个形式很自然。但它隐含了一个约束顺序：

1. 先让每个 token 独立选 expert。
2. 再检查每个 expert 有没有超容量。
3. 如果超了，再决定哪些 token 保留，哪些 token 丢弃。

也就是说，容量冲突发生在路由决策之后，而不是决策之中。于是局部最优选择很容易堆成全局拥塞。

### 2. Expert-Choice 的数学形式

Expert-Choice 的关键不是换了一个 top-k，而是把矩阵视角反过来。

先把评分矩阵按 expert 视角理解。对于某个 expert $e$，它看到的是所有 token 对自己的匹配分数：

$$
S_{:,e} \in \mathbb{R}^{T}
$$

然后每个 expert 选自己最想处理的前 $C$ 个 token：

$$
\mathcal{T}_e = \operatorname{TopC}(S_{:,e})
$$

其中 $\mathcal{T}_e$ 表示 expert $e$ 接收的 token 集合。

于是输出可写为：

$$
y_t = \sum_{e:\, t \in \mathcal{T}_e} \tilde{g}_{t,e} \cdot f_e(h_t)
$$

注意这里与 Token-Choice 有一个根本差别：

- Token-Choice 中，先确定“token 选哪些 expert”。
- Expert-Choice 中，先确定“expert 接哪些 token”。

如果把两者放在一起看：

$$
\text{Token-Choice: } \forall t,\ \mathcal{E}_t = \operatorname{TopK}(S_{t,:})
$$

$$
\text{Expert-Choice: } \forall e,\ \mathcal{T}_e = \operatorname{TopC}(S_{:,e})
$$

两者最大的结构差异，是约束施加的位置不同：

| 方法 | 固定的是什么 | 可变的是什么 |
|---|---|---|
| Token-Choice | 每个 token 的激活 expert 数 | 每个 expert 接收多少 token |
| Expert-Choice | 每个 expert 的接收 token 数 | 每个 token 被几个 expert 接收 |

这会直接改变训练统计量、通信形状和负载曲线。

### 3. 为什么 Expert-Choice 更均衡

设每个 expert 的容量都固定为 $C$，则总槽位数为：

$$
\text{TotalSlots} = E \cdot C
$$

只要路由分数不是完全退化的，expert 通常都能填满自己的前 $C$ 个 token。于是有：

$$
|\mathcal{T}_e| = C,\quad \forall e
$$

这意味着：

- 每个 expert 的接收量在结构上被钉住。
- 不会出现一个 expert 忙到溢出、另一个 expert 完全没活干的极端情况。
- dispatch tensor 的 shape 更稳定，padding 和通信也更可控。

这里常说的“完美均衡”，严格讲并不是指每个 token 都被同样多的 expert 处理，而是指：

$$
\text{expert utilization variance} \approx 0
$$

也就是 expert 维度上的负载方差非常小。它是**按 expert 看均衡**，不是按 token 看均衡。

为了让这个区别更直观，可以看下面这张表：

| 统计对象 | Token-Choice 更稳定 | Expert-Choice 更稳定 |
|---|---|---|
| 每个 token 激活几个 expert | 是 | 否 |
| 每个 expert 接多少 token | 否 | 是 |
| 通信桶大小 | 否 | 是 |
| token 是否会因为热点冲突被丢弃 | 否 | 更少 |

### 4. 变量专家数是副产品，不是附赠功能

Token-Choice 里，每个 token 通常固定激活 $K$ 个 expert。无论这个 token 是简单词元、格式标记，还是真正困难的逻辑片段，它消耗的 expert 数基本一致。

Expert-Choice 不是这样。因为每个 expert 独立做 top-$C$，同一个 token 可能：

- 没被任何 expert 选中；
- 被 1 个 expert 选中；
- 被多个 expert 同时选中。

如果定义 token 的覆盖数为：

$$
m_t = \sum_{e=1}^{E} \mathbf{1}[t \in \mathcal{T}_e]
$$

那么在 Expert-Choice 下，$m_t$ 是可变的。这意味着模型可以把更多计算预算分给“多个 expert 都认为值得处理”的 token，把更少预算给简单 token。这个性质常被称为 heterogeneous routing，本质上是**计算量随 token 难度自适应**。

但这也是一把双刃剑。因为一旦没有额外约束，少数高分 token 可能反复被多个 expert 抢到，导致：

- 计算预算过度集中；
- 长尾 token 覆盖不足；
- 模型对“高路由分 token”偏置过强。

所以“变量 expert 数”不是白送的收益，它要求额外监控和正则。

### 5. 因果性问题从哪里来

这部分必须说清楚，否则容易误把 Expert-Choice 当成“训练更先进的 Token-Choice”。

在自回归语言模型中，位置 $t$ 的计算必须满足：

$$
h_t = F(x_{\le t})
$$

也就是说，位置 $t$ 的隐藏状态只能依赖当前和过去的信息。

但如果在某一层使用 Expert-Choice，则 expert $e$ 在选择 top-$C$ token 时，要比较整个 batch 中所有 token 的分数：

$$
\mathcal{T}_e = \operatorname{TopC}(S_{:,e})
$$

如果这个 batch 包含序列中的未来位置，那么 token $t$ 是否能被 expert $e$ 接收，不仅取决于 $S_{t,e}$，还取决于未来 token 的分数是否更高。于是就出现了：

$$
\mathbf{1}[t \in \mathcal{T}_e]
\ \text{依赖于}\ 
S_{t',e},\ t' > t
$$

这正是未来信息泄露。

换句话说，Expert-Choice 的强均衡能力不是“免费获得”的。它来自一种更强的全局视角，而在严格自回归场景中，这种全局视角正好违背因果约束。

可以用一句更直接的话概括：

**Expert-Choice 之所以均衡，是因为它允许 expert 在全局竞争中挑 token；而自回归要求当前位置不能因为未来 token 的存在与否而改变路由结果。**

---

## 代码实现

下面给出一个可直接运行的 Python 示例，对比 Token-Choice 和 Expert-Choice。代码只依赖标准库，包含：

- 可复现的 top-k 选择；
- Token-Choice 路由；
- Expert-Choice 路由；
- 覆盖率、利用率和重复选择统计；
- 一个最小示例输出。

```python
from math import ceil
from typing import List, Dict, Tuple


def topk_indices(values: List[float], k: int) -> List[int]:
    """Return top-k indices by value, tie-broken by smaller index."""
    if k <= 0:
        return []
    ranked = sorted(enumerate(values), key=lambda x: (-x[1], x[0]))
    return [idx for idx, _ in ranked[:k]]


def compute_capacity(num_tokens: int, num_experts: int, k: int = 1, capacity_factor: float = 1.0) -> int:
    """Typical MoE capacity formula."""
    return ceil((num_tokens * k / num_experts) * capacity_factor)


def token_choice_route(
    scores: List[List[float]],
    k: int,
    capacity: int,
) -> Tuple[Dict[int, List[int]], List[int], Dict[int, List[int]]]:
    """
    Token-choice routing.

    Returns:
        accepted_by_expert: expert -> accepted token list
        dropped_tokens: tokens that could not be placed into any chosen expert
        chosen_experts_by_token: token -> original top-k expert choices
    """
    num_tokens = len(scores)
    num_experts = len(scores[0])
    accepted_by_expert = {e: [] for e in range(num_experts)}
    chosen_experts_by_token = {}

    dropped_tokens = []

    for t in range(num_tokens):
        chosen = topk_indices(scores[t], k)
        chosen_experts_by_token[t] = chosen

        placed = False
        for e in chosen:
            if len(accepted_by_expert[e]) < capacity:
                accepted_by_expert[e].append(t)
                placed = True
                break

        if not placed:
            dropped_tokens.append(t)

    return accepted_by_expert, dropped_tokens, chosen_experts_by_token


def expert_choice_route(
    scores: List[List[float]],
    capacity: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], List[int]]:
    """
    Expert-choice routing.

    Returns:
        accepted_by_expert: expert -> selected token list
        experts_by_token: token -> experts that selected it
        uncovered_tokens: tokens not selected by any expert
    """
    num_tokens = len(scores)
    num_experts = len(scores[0])

    accepted_by_expert = {e: [] for e in range(num_experts)}
    experts_by_token = {t: [] for t in range(num_tokens)}

    for e in range(num_experts):
        expert_view = [scores[t][e] for t in range(num_tokens)]
        selected_tokens = topk_indices(expert_view, capacity)
        accepted_by_expert[e] = selected_tokens
        for t in selected_tokens:
            experts_by_token[t].append(e)

    uncovered_tokens = [t for t, experts in experts_by_token.items() if not experts]
    return accepted_by_expert, experts_by_token, uncovered_tokens


def summarize_utilization(accepted_by_expert: Dict[int, List[int]], capacity: int) -> Dict[int, float]:
    """expert -> utilization ratio"""
    return {e: len(tokens) / capacity for e, tokens in accepted_by_expert.items()}


def summarize_coverage(experts_by_token: Dict[int, List[int]]) -> Dict[int, int]:
    """token -> number of experts covering it"""
    return {t: len(experts) for t, experts in experts_by_token.items()}


def invert_expert_assignments(accepted_by_expert: Dict[int, List[int]], num_tokens: int) -> Dict[int, List[int]]:
    """Convert expert -> tokens into token -> experts."""
    token_to_experts = {t: [] for t in range(num_tokens)}
    for e, tokens in accepted_by_expert.items():
        for t in tokens:
            token_to_experts[t].append(e)
    return token_to_experts


def main() -> None:
    # 4 tokens, 2 experts
    # All tokens prefer expert 0, but expert 1 still has some affinity.
    scores = [
        [0.99, 0.20],  # token 0
        [0.95, 0.40],  # token 1
        [0.92, 0.70],  # token 2
        [0.90, 0.80],  # token 3
    ]

    num_tokens = len(scores)
    num_experts = len(scores[0])

    k = 1
    capacity_factor = 1.0
    capacity = compute_capacity(num_tokens, num_experts, k=k, capacity_factor=capacity_factor)

    tc_accept, tc_drop, tc_choices = token_choice_route(scores, k=k, capacity=capacity)
    tc_token_to_experts = invert_expert_assignments(tc_accept, num_tokens)

    ec_accept, ec_token_to_experts, ec_uncovered = expert_choice_route(scores, capacity=capacity)

    tc_util = summarize_utilization(tc_accept, capacity)
    ec_util = summarize_utilization(ec_accept, capacity)

    tc_cov = summarize_coverage(tc_token_to_experts)
    ec_cov = summarize_coverage(ec_token_to_experts)

    # Deterministic checks
    assert capacity == 2

    # Token-choice: everyone first picks expert 0, capacity overflows.
    assert tc_choices == {
        0: [0],
        1: [0],
        2: [0],
        3: [0],
    }
    assert tc_accept == {
        0: [0, 1],
        1: [],
    }
    assert tc_drop == [2, 3]

    # Expert-choice: each expert selects its own top-2 tokens.
    assert ec_accept == {
        0: [0, 1],
        1: [3, 2],
    }
    assert ec_uncovered == []

    print("=== Settings ===")
    print(f"num_tokens={num_tokens}, num_experts={num_experts}, k={k}, capacity={capacity}")

    print("\n=== Token-Choice ===")
    print("accepted_by_expert:", tc_accept)
    print("dropped_tokens:", tc_drop)
    print("utilization:", tc_util)
    print("coverage_per_token:", tc_cov)

    print("\n=== Expert-Choice ===")
    print("accepted_by_expert:", ec_accept)
    print("uncovered_tokens:", ec_uncovered)
    print("utilization:", ec_util)
    print("coverage_per_token:", ec_cov)


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出结果会稳定体现下面的差异：

| 指标 | Token-Choice | Expert-Choice |
|---|---|---|
| expert 0 利用率 | 100% | 100% |
| expert 1 利用率 | 0% | 100% |
| 被丢弃 / 未覆盖 token 数 | 2 | 0 |
| 每个 token 覆盖 expert 数 | 固定至多 1，但有 token 为 0 | 可变，但该例中都为 1 |

### 为什么这个例子足够说明问题

这不是因为它模拟了完整 MoE，而是因为它保留了最关键的两个约束：

1. expert 容量有限；
2. token 偏好集中在少数 expert 上。

一旦这两个条件同时成立，Token-Choice 的容量冲突和 Expert-Choice 的均衡优势就会立即显现。真实工程只是在此基础上多了：

- 门控权重归一化；
- all-to-all 通信；
- 多卡 expert-parallel；
- combine 策略；
- 辅助损失与监控指标。

### 对应真实工程时需要再补上的部件

真实实现通常还需要处理下面这些问题：

| 模块 | 需要做什么 | 为什么不能省 |
|---|---|---|
| dispatch index | 记录 token 到 expert 的映射 | 决定如何把 token 搬到各专家设备 |
| combine rule | 多个 expert 命中同一 token 时如何合并 | 否则无法回写到 token 顺序 |
| fallback path | 未被任何 expert 选中的 token 如何处理 | 避免表示直接断路 |
| load monitor | 记录专家利用率、覆盖率、重复率 | 不然训练会“看起来没错，实际很偏” |
| aux loss | 对路由分布做正则 | 防止少数 expert 或少数 token 长期垄断 |

一个非常常见的 combine 形式是：

$$
y_t = \sum_{e:\, t \in \mathcal{T}_e} \alpha_{t,e} \cdot f_e(h_t)
$$

其中 $\alpha_{t,e}$ 可能来自：

- 原始 router 分数再归一化；
- softmax over selected experts；
- 简化成平均值；
- 或者只保留最高分 expert 的输出。

如果某些 token 没被任何 expert 选中，则常见 fallback 是：

$$
y_t = h_t
\quad \text{或} \quad
y_t = h_t + f_{\text{shared}}(h_t)
$$

这本质上是在说：即使稀疏专家没处理这个 token，主干路径也不能让表示断掉。

### 为什么固定容量对系统层尤其重要

在 expert-parallel 场景里，expert 往往分布在不同 GPU 或不同机器上。此时路由不仅是算法问题，也是通信问题。固定容量带来的收益通常有三类：

1. 每个 expert 的输入 batch 大小更稳定，kernel shape 更稳定。
2. all-to-all 的发送量更接近定长，通信尾延迟更小。
3. padding 和临时 buffer 更可预测，显存规划更容易。

可以把它总结成一句工程话：

**Token-Choice 更像“先做决策，再收拾拥塞”；Expert-Choice 更像“先按桶分好容量，再填桶”。前者对因果友好，后者对系统友好。**

---

## 工程权衡与常见坑

Expert-Choice 的优势很明确，但它真正难的地方不在“能不能写出来”，而在“训练目标、系统目标和部署目标是否一致”。

先看常见问题总表：

| 常见坑 | 现象 | 根因 | 常见规避策略 |
|---|---|---|---|
| 因果泄露 | 训练效果好，严格自回归推理掉点 | 路由依赖未来 token 的相对分数 | 不直接用于严格 AR 推理；改用 causal 近似方案 |
| 覆盖不足 | 几乎没有 drop，但部分 token 从不被 expert 选中 | 容量固定过紧，top-C 竞争过强 | 提高 `capacity_factor`，监控 uncovered ratio |
| 重复覆盖过高 | 少数 token 被多个 expert 重复处理 | 高分 token 被多个 expert 同时偏好 | 加 coverage regularization 或限制最大重复数 |
| 容量过大 | 吞吐变差、显存上涨、padding 增加 | 为了追求“零遗漏”开太多冗余 | 结合 batch 和 expert 数做 sweep，不要只看 loss |
| 训练推理不一致 | 训练路由和线上行为差异大 | 训练用 EC，推理却只能用 causal token-choice | 做 router distillation、KL 对齐或双路由训练 |
| 监控指标选错 | 表面无异常，但模型长期退化 | 只看 expert utilization，不看 token coverage | 同时监控 utilization、coverage、duplication |

下面把最关键的几类问题展开说。

### 1. 因果泄露不是“小偏差”，而是约束冲突

在自回归模型里，训练和推理必须满足同一因果图。若训练阶段用的是“整批 top-C 选 token”，那就等于允许未来 token 参与当前位置的 expert 竞争排序。

这会导致一种典型的 train-inference mismatch：

- 训练时，router 学会利用未来 token 的竞争结构来分配容量。
- 推理时，未来 token 不存在，原先的竞争格局消失。
- 结果就是 expert 分配模式漂移，模型行为和训练时不一致。

这类问题往往不是简单的 perplexity 轻微变差，而是长序列稳定性、风格一致性、工具调用边界之类的行为性问题先出问题。

### 2. Expert-Choice 下“没有 drop”不等于“没有损失”

Token-Choice 里，问题通常以“drop”形式暴露，很容易看见。Expert-Choice 下则更容易变成 coverage 问题：

- 某些 token 被 2 到 3 个 expert 重复处理；
- 另一些 token 一个 expert 都没选中。

所以在 EC 中，最该监控的不是单一的 drop rate，而是这几个指标：

$$
\text{Utilization}_e = \frac{|\mathcal{T}_e|}{C}
$$

$$
\text{Coverage}_t = \sum_{e=1}^{E}\mathbf{1}[t \in \mathcal{T}_e]
$$

$$
\text{UncoveredRatio} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\text{Coverage}_t=0]
$$

$$
\text{DuplicationRatio} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\text{Coverage}_t>1]
$$

其中：

- `Utilization` 反映 expert 是否满载；
- `UncoveredRatio` 反映是否有 token 完全没拿到专家计算；
- `DuplicationRatio` 反映是否有过多计算集中在少数 token 上。

对新手来说，一个很重要的认知是：**EC 的问题更像“覆盖结构偏了”，而不只是“桶装没装满”。**

### 3. 容量调参不能只盯着 utilization

很多人看到 Expert-Choice 负载更均衡，就会自然以为容量可以压得更小。这个判断经常出错。

容量太小的后果通常有三层：

1. 专家虽然都满载，但覆盖不到足够多的 token。
2. token 竞争过强，只有高分 token 能反复进入 expert。
3. 模型丧失“把额外算力分给困难 token”的弹性，因为总槽位太少。

因此容量 sweep 时至少要联合看四组量：

| 指标 | 说明 |
|---|---|
| 平均 expert utilization | 专家是否被有效使用 |
| uncovered token ratio | 是否有太多 token 完全没有专家路径 |
| duplication ratio | 是否有太多重复计算集中在少数 token |
| step time / 显存 / all-to-all 波动 | 系统代价是否可接受 |

如果只看利用率，很容易得出“容量越小越好”的错误结论，因为专家永远都能看起来很忙，但 token 学不到足够多的专家表示。

### 4. null expert 为什么常被提到

对于严格因果场景，一个常见的折中思路不是硬搬 Expert-Choice，而是在 causal Token-Choice 框架里增加一个 null expert 或 shared fallback expert。

它的直觉是：

- 保持 token 自主选择，因此保留因果性；
- 允许一部分 token 选择“少算”甚至“不算”；
- 用这种方式近似实现 data sparsity，而不是让 expert 全局反向挑 token。

null expert 可以理解为一个合法但几乎不做计算的出口。若把它记作 expert 0，则某些 token 的输出可以近似写成：

$$
y_t =
\begin{cases}
f_e(h_t), & \text{if routed to real expert } e \\
h_t \text{ or } 0, & \text{if routed to null expert}
\end{cases}
$$

它解决的不是“让 EC 因果化”，而是“在因果条件下，保留一部分变计算量能力”。

---

## 替代方案与适用边界

如果场景是严格自回归推理，Token-Choice 依然非常有竞争力，而且很多时候是更稳的选择。原因不是它更“高级”，而是它满足了部署最关键的约束：局部、在线、因果。

下面这张表可以把适用边界压缩得更清楚：

| 场景 | 更适合的方法 | 主要原因 |
|---|---|---|
| 大 batch 训练，expert 数多 | Expert-Choice | 负载更稳，专家更满载，吞吐更高 |
| 严格自回归推理 | Token-Choice | 路由不依赖未来 token，部署简单 |
| 模型规模中等，热点不明显 | Token-Choice | 简单方案足够，额外复杂度不值 |
| 超大规模多机训练 | Expert-Choice | 固定容量更利于 all-to-all 与 expert-parallel |
| 希望 token 计算量自适应 | Expert-Choice 或 causal 稀疏变体 | token 可变 expert 数更自然 |
| 非常重视训练推理一致性 | Token-Choice 或统一双路由方案 | 避免 train-inference mismatch |

更现实的工程路线通常不是二选一，而是分阶段使用：

| 阶段 | 常见做法 | 目标 |
|---|---|---|
| 训练期 | Expert-Choice | 追求吞吐、均衡和收敛效率 |
| 蒸馏或对齐期 | 增加 router 对齐损失 | 缩小训练路由和推理路由差异 |
| 推理期 | Token-Choice 或 causal 近似路由 | 保证因果性和实现稳定 |

这类混合路线的好处是把各自优势放到最适合的阶段，但代价也很明确：

1. 需要两套路由逻辑。
2. 需要额外的 router 对齐目标。
3. 评估指标不能只看 loss，还要看路由统计是否漂移。

如果你的系统满足下面任意一条，继续用 Token-Choice 往往更合理：

- 在线推理是主场景；
- 模型规模还没大到通信成为主要瓶颈；
- expert 数有限，负载不均尚未严重影响训练；
- 团队更在乎实现稳定性，而不是极限吞吐。

相反，如果你已经遇到下面这些信号，Expert-Choice 才更值得引入：

- 热点 expert 经常溢出；
- `capacity_factor` 已经开得很高，但仍然 drop 明显；
- 多机 all-to-all 抖动严重；
- expert-utilization 方差过大，导致有些卡长期忙、有些卡长期闲。

所以最实际的判断标准不是“论文里谁更先进”，而是：

**你的当前瓶颈到底在因果部署，还是在训练期负载均衡。**

---

## 参考资料

1. Google Research Blog, *Mixture-of-Experts with Expert Choice Routing*  
   作用：最适合先建立直觉，解释了为什么把约束放到 expert 侧后，容量利用和训练吞吐会更稳定。对“固定容量”和“异构 token 计算量”给出了直观说明。  
   链接：https://research.google/blog/mixture-of-experts-with-expert-choice-routing/

2. Zhou et al., NeurIPS 2022, *Mixture-of-Experts with Expert Choice Routing*  
   作用：正式论文版本，适合查定义、公式、容量设置和实验结果。本文关于 per-expert top-$C$、变量 token 覆盖数和均衡性的正式描述，主要应以该文为准。  
   链接：https://papers.nips.cc/paper_files/paper/2022/file/2f00ecd787b432c1d36f3de9800728eb-Paper-Conference.pdf

3. Fedus et al., JMLR 2022, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*  
   作用：理解 Token-Choice、top-1 路由、capacity factor、token drop 和辅助负载均衡损失的经典起点。若不先看这篇，很容易低估 Token-Choice 为什么必须配套 load balancing。  
   链接：https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf

4. Kilian et al., 2026, *Improving MoE Compute Efficiency by Composing Weight and Data Sparsity*  
   作用：帮助理解为什么 expert-choice 式的全局 token 选择会和严格自回归因果性冲突，以及 null expert 一类思路为什么会被重新讨论。  
   链接：https://arxiv.org/abs/2601.15370  
   可浏览摘要页：https://huggingface.co/papers/2601.15370

5. Scribd, *Expert Choice Routing* 讨论文档  
   作用：适合新手快速建立“矩阵转视角后按 expert 选 token”的图像化理解。但它属于二手材料，适合作为入门辅助，不应替代原论文。  
   链接：https://www.scribd.com/document/818239276/choice-routing

6. lonepatient.top 对 2026 新论文的中文总结  
   作用：适合把“EC 的非因果性”和“causal 稀疏近似方案”快速串起来，方便建立阅读路径。正式结论仍应回到论文原文确认。  
   链接：https://lonepatient.top/2026/01/23/arxiv_papers_2026-01-23.html
