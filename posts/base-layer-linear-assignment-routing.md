## 核心结论

Base Layer 的核心不是“把路由分数换一种写法”，而是把 MoE 中的 token-to-expert 分配直接定义成一个**线性分配问题**。线性分配的直白解释是：在一组候选匹配里，找出**总收益最高**、同时满足**每个位置名额固定**的那组分配。

传统 Top-k 路由通常先让每个 token 独立选择分数最高的专家，再用 `capacity factor` 和 `auxiliary loss` 去补救负载不均。Base Layer 反过来做：先把“每个专家必须接收多少 token”写成硬约束，再在这个约束下求全局最优分配。结果是每个专家在一个批次内**精确接收相同数量的 token**，负载均衡不再依赖软惩罚项，也不再依赖“希望辅助损失把流量慢慢拉回来”。

如果一个批次有 $T$ 个 token、$E$ 个专家，并且 $E \mid T$，Base Layer 的目标可以写成：

$$
\max_{a_1,\dots,a_T}\sum_{t=1}^T h_t^\top w_{a_t}
$$

约束为：

$$
\forall e:\sum_{t=1}^T \mathbf{1}[a_t=e]=T/E
$$

其中：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $h_t$ | 第 $t$ 个 token 的表示向量 | 当前 token 携带的语义特征 |
| $w_e$ | 第 $e$ 个专家的路由向量 | 这个专家更偏好的 token 方向 |
| $h_t^\top w_e$ | token 与专家的匹配分数 | 分数越高，越适合把该 token 发给该专家 |
| $a_t$ | token $t$ 的目标专家编号 | 最终路由决策 |

一个玩具例子最直观。假设有 8 个学生和 4 张桌子，每张桌子只能坐 2 个人。Top-k 的效果像是让学生自由冲向自己最喜欢的桌子，最后可能出现 `3-3-1-1`；Base Layer 则是一次性排座位，强制得到 `2-2-2-2`，同时尽量让每个学生坐到最适合的位置。

这件事的关键不在“分数怎么打”，而在“**分配是局部独立决策，还是全局受约束优化**”。Base Layer 选的是后者。

---

## 问题定义与边界

MoE，Mixture of Experts，意思是“很多专家子网络并列放着，每个 token 只经过其中少数几个专家”。这样做的目的不是减少参数量，而是让模型在**每个 token 的实际计算量可控**的前提下，拥有更大的总参数规模。

路由器的职责就是回答一个问题：**这个 token 应该发给哪个专家？**

传统路由的问题通常不是“不会选”，而是“选出来之后系统层面不稳定”。如果每个 token 都独立做 Top-k 选择，会出现两个直接后果：

1. 某些专家被大量 token 挤满，某些专家几乎空闲。
2. 分布式训练里会出现 `straggler`，也就是少数慢设备拖住整步训练。

容量限制可以截断超载专家，但截断本身会带来副作用：

| 现象 | 结果 |
|---|---|
| 热门专家超出容量 | 一部分 token 被迫改投次优专家 |
| 超载后直接丢弃 token | 路由信息损失，训练信号变差 |
| 专家负载长期偏斜 | 一些专家学得过多，一些专家几乎学不到 |

辅助负载损失可以鼓励均衡，但它本质上仍然是惩罚项，不保证**每一步**都均衡。换句话说，Top-k + aux loss 解决的是“平均意义上尽量均衡”，Base Layer 解决的是“这一步必须均衡”。

Base Layer 讨论的是一个更窄、但更硬的目标：在**单个批次内部**，让每个专家接收到完全相同数量的 token，同时最大化总匹配分数。它不是在概率空间里“希望更均匀”，而是在组合优化空间里“必须均匀”。

下面用 8 个 token、4 个专家的最小例子看差异。每个专家都应接收 2 个 token。

| 专家 | 传统 Top-k 可能分布 | Base Layer 分布 |
|---|---:|---:|
| Expert 1 | 3 | 2 |
| Expert 2 | 3 | 2 |
| Expert 3 | 1 | 2 |
| Expert 4 | 1 | 2 |

这个边界很重要：

| 条件 | Base Layer 的要求 | 含义 |
|---|---|---|
| $T$ 能被 $E$ 整除 | 最理想 | 每个专家名额固定为 $T/E$ |
| 单批 token 数不能过大 | 很重要 | 否则线性分配求解成本会变重 |
| 训练更关注吞吐稳定性 | 适合 | 精确负载均衡的收益明显 |
| 追求超大 batch 下极致便宜的路由 | 未必适合 | Top-k 通常更省算力 |
| 训练容易出现专家塌缩 | 更适合 | 硬配额能直接约束流量偏斜 |

这里顺带解释两个常见术语：

| 术语 | 定义 | 为什么重要 |
|---|---|---|
| `capacity factor` | 每个专家可接收 token 的上限倍数 | 防止专家无限超载 |
| `expert collapse` | 少数专家长期吸走大部分 token | 其他专家学不到有用参数 |

所以，Base Layer 不是“永远优于 Top-k”，而是“在 token 数可控、负载稳定性重要的前提下，用确定性全局分配替代概率式局部抢占”。

---

## 核心机制与推导

Base Layer 的第一步是构造 `affinity matrix`，也叫亲和度矩阵。它本质上是一张“每个 token 对每个专家有多合适”的打分表。

设有 $T$ 个 token、$E$ 个专家。对每个 token 表示 $h_t$ 和专家向量 $w_e$ 计算：

$$
s_{t,e}=h_t^\top w_e
$$

于是得到一个 $T \times E$ 的分数矩阵：

$$
S=
\begin{bmatrix}
s_{1,1} & s_{1,2} & \cdots & s_{1,E}\\
s_{2,1} & s_{2,2} & \cdots & s_{2,E}\\
\vdots & \vdots & \ddots & \vdots\\
s_{T,1} & s_{T,2} & \cdots & s_{T,E}
\end{bmatrix}
$$

如果直接做 Top-1，就是每一行各自取最大值：

$$
a_t=\arg\max_e s_{t,e}
$$

但 Base Layer 不允许每一行独立决策，因为列上还有名额约束。也就是说，不能只问“某个 token 最喜欢谁”，还必须问“这个专家是不是已经满员”。

### 从配额约束到标准分配问题

为把它变成标准线性分配问题，常见做法是把每个专家复制成若干个“槽位”。若 $T=8,E=4$，则每个专家有 $T/E=2$ 个槽位，总共 8 个槽位。问题就变成：

- 8 个 token
- 8 个槽位
- 每个 token 必须占用且只占用 1 个槽位
- 每个槽位必须分给且只分给 1 个 token

令专家 $e$ 的第 $j$ 个槽位记作 $(e,j)$，则优化目标可以写成：

$$
\max_{x_{t,e,j}} \sum_{t=1}^T \sum_{e=1}^E \sum_{j=1}^{T/E} s_{t,e}x_{t,e,j}
$$

约束为：

$$
\sum_{e=1}^E \sum_{j=1}^{T/E} x_{t,e,j}=1,\quad \forall t
$$

$$
\sum_{t=1}^T x_{t,e,j}=1,\quad \forall e,j
$$

$$
x_{t,e,j}\in\{0,1\}
$$

这里的 $x_{t,e,j}$ 表示：token $t$ 是否被分配到专家 $e$ 的第 $j$ 个槽位。

这三个约束分别表示：

| 约束 | 数学含义 | 直白解释 |
|---|---|---|
| 每个 token 恰好被分一次 | 第一条约束 | 一个 token 不能去两个专家，也不能没人接收 |
| 每个槽位恰好装一个 token | 第二条约束 | 每个专家名额都必须填满 |
| 决策是离散的 | 第三条约束 | 不是“分 0.3 个 token”，而是“去或不去” |

### 为什么这是“全局最优”

看下面这个玩具矩阵。8 个 token，4 个专家，每个专家 2 个名额：

| Token | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
| T1 | 9 | 2 | 1 | 0 |
| T2 | 8 | 3 | 2 | 1 |
| T3 | 7 | 6 | 1 | 0 |
| T4 | 1 | 9 | 4 | 2 |
| T5 | 0 | 8 | 5 | 3 |
| T6 | 2 | 1 | 9 | 6 |
| T7 | 1 | 2 | 8 | 7 |
| T8 | 0 | 1 | 3 | 10 |

如果按每行独立取最大值：

- T1 -> E1
- T2 -> E1
- T3 -> E1
- T4 -> E2
- T5 -> E2
- T6 -> E3
- T7 -> E3
- T8 -> E4

得到分布：

| 专家 | 获得 token 数 |
|---|---:|
| E1 | 3 |
| E2 | 2 |
| E3 | 2 |
| E4 | 1 |

这不满足精确均衡。Base Layer 会做的是：在总分尽量高的前提下，把某些 token 从“个人第一志愿”挪到“全局更合理的位置”。例如把 T3 从 E1 挪给 E2，总代价只减少 $7-6=1$，但全局约束被满足了。

一种满足 `2-2-2-2` 的可行分配是：

| Token | 分配专家 | 得分 |
|---|---|---:|
| T1 | E1 | 9 |
| T2 | E1 | 8 |
| T3 | E2 | 6 |
| T4 | E2 | 9 |
| T5 | E3 | 5 |
| T6 | E3 | 9 |
| T7 | E4 | 7 |
| T8 | E4 | 10 |

总分为：

$$
9+8+6+9+5+9+7+10=63
$$

这个例子说明了 Base Layer 的核心逻辑：

- 它不保证每个 token 都拿到自己的第一名专家。
- 它保证整个批次在硬约束下的**总分最优或近似最优**。
- 它优化的是**批次级目标**，不是 token 级贪心目标。

### 求解算法怎么选

把问题变成标准分配问题后，就可以使用经典算法：

| 算法 | 特点 | 适合场景 |
|---|---|---|
| 匈牙利算法 | 经典、结果精确 | 中小规模标准分配 |
| 拍卖算法 | 常用于并行或近似高效求解 | 更偏工程实现 |
| 分块求解 | 先切 chunk 再局部最优化 | 大规模训练更常见 |

匈牙利算法常被拿来解释 Base Layer，因为它是最容易讲清楚的“精确一对一匹配”算法。标准复杂度通常写为 $O(n^3)$，其中 $n$ 是方阵边长。若把问题展开成“token 对槽位”的方阵，$n$ 大致等于 token 数或总槽位数，因此 token 很大时计算会迅速变重。

这也是为什么论文和工程实现里都强调：**Base Layer 的重点不是数学上能不能解，而是怎么把求解规模压到硬件可接受的范围内。**

---

## 代码实现

下面先给一个**可直接运行**的 Python 示例，用动态规划求解小规模 Base Layer。它比暴力枚举更实用，依然只适合教学，不适合大规模训练，但输入、输出、约束检查都是真实可运行的。

```python
from functools import lru_cache


def base_layer_route(scores):
    """
    求解精确均衡的 Base Layer 路由。

    参数:
        scores: List[List[float]], shape = [T, E]
            scores[t][e] 表示第 t 个 token 分配给第 e 个专家的分数

    返回:
        assignment: List[int], 长度为 T
            assignment[t] = e，表示 token t 被分给专家 e
        total_score: float
            在精确配额约束下的最优总分

    约束:
        1. T 必须能被 E 整除
        2. 每个 token 恰好分给一个专家
        3. 每个专家恰好接收 T / E 个 token
    """
    T = len(scores)
    if T == 0:
        raise ValueError("scores must not be empty")

    E = len(scores[0])
    if E == 0:
        raise ValueError("scores must have at least one expert")

    if any(len(row) != E for row in scores):
        raise ValueError("all rows in scores must have the same length")

    if T % E != 0:
        raise ValueError(f"T={T} must be divisible by E={E}")

    quota = T // E

    @lru_cache(maxsize=None)
    def dp(t, counts):
        """
        t: 当前处理到第几个 token
        counts: 一个长度为 E 的元组，表示每个专家已接收的 token 数
        """
        if t == T:
            if all(c == quota for c in counts):
                return 0.0, ()
            return float("-inf"), ()

        best_score = float("-inf")
        best_suffix = None

        for e in range(E):
            if counts[e] >= quota:
                continue

            next_counts = list(counts)
            next_counts[e] += 1

            suffix_score, suffix_assignment = dp(t + 1, tuple(next_counts))
            if suffix_score == float("-inf"):
                continue

            current_score = scores[t][e] + suffix_score
            if current_score > best_score:
                best_score = current_score
                best_suffix = (e,) + suffix_assignment

        return best_score, best_suffix

    total_score, assignment = dp(0, tuple([0] * E))
    if assignment is None:
        raise RuntimeError("no feasible assignment found")

    return list(assignment), total_score


def top1_route(scores):
    """
    每个 token 独立做 Top-1，用于和 Base Layer 对比。
    """
    assignment = []
    total_score = 0.0

    for row in scores:
        best_e = max(range(len(row)), key=lambda e: row[e])
        assignment.append(best_e)
        total_score += row[best_e]

    return assignment, total_score


def count_by_expert(assignment, num_experts):
    counts = [0] * num_experts
    for e in assignment:
        counts[e] += 1
    return counts


if __name__ == "__main__":
    scores = [
        [9, 2, 1, 0],   # T1
        [8, 3, 2, 1],   # T2
        [7, 6, 1, 0],   # T3
        [1, 9, 4, 2],   # T4
        [0, 8, 5, 3],   # T5
        [2, 1, 9, 6],   # T6
        [1, 2, 8, 7],   # T7
        [0, 1, 3, 10],  # T8
    ]

    T = len(scores)
    E = len(scores[0])

    top1_assignment, top1_total = top1_route(scores)
    base_assignment, base_total = base_layer_route(scores)

    print("Top-1 assignment:", top1_assignment)
    print("Top-1 counts:", count_by_expert(top1_assignment, E))
    print("Top-1 total:", top1_total)
    print()

    print("Base Layer assignment:", base_assignment)
    print("Base Layer counts:", count_by_expert(base_assignment, E))
    print("Base Layer total:", base_total)

    assert len(base_assignment) == T
    assert count_by_expert(base_assignment, E) == [T // E] * E
```

这段代码有几个值得注意的地方。

### 1. 它是真的可运行

运行后你会看到两种结果：

- `Top-1` 总分通常更高或相近，但可能不均衡
- `Base Layer` 严格满足 `2-2-2-2` 的配额约束

在上面的示例矩阵里，Top-1 的结果通常是：

```text
Top-1 assignment: [0, 0, 0, 1, 1, 2, 2, 3]
Top-1 counts: [3, 2, 2, 1]
Top-1 total: 69.0
```

而 Base Layer 会输出满足精确均衡的分配，例如：

```text
Base Layer assignment: [0, 0, 1, 1, 2, 2, 3, 3]
Base Layer counts: [2, 2, 2, 2]
Base Layer total: 63.0
```

这里可以看出一个非常重要的工程事实：

$$
\text{Base Layer 总分} \le \text{无约束 Top-1 总分}
$$

原因不是 Base Layer “更差”，而是它多了一个硬约束。无约束问题的最优值天然不低于有约束问题。

### 2. 它清楚展示了“全局优化”的含义

动态规划里的状态是：

$$
(t, c_1, c_2, \dots, c_E)
$$

表示“已经处理到第 $t$ 个 token，且每个专家已接收的 token 数分别为 $c_1,\dots,c_E$”。这说明 Base Layer 的决策不是只看当前 token，而是要同时考虑：

- 当前 token 给谁分数更高
- 哪些专家还剩名额
- 未来 token 是否更适合占用剩余名额

这正是“全局分配”与“局部贪心”之间的本质差异。

### 3. 它不是工程实现

这段代码复杂度依然很高，只能用于教学。真实系统不会这样求解，而是使用更高效的算法或近似策略。教学代码的目标是把下面三件事讲清楚：

1. 先算 affinity，也就是 `scores[t][e]`
2. 再解带配额约束的全局分配
3. 最后按分配结果把 token 发到对应专家

如果写成更接近工程实现的伪代码，流程通常是：

```python
def route_tokens(token_batch, router_weights):
    """
    token_batch: [T, d]
    router_weights: [E, d]
    """
    scores = token_batch @ router_weights.T      # [T, E]
    assignment = solve_linear_assignment(scores) # [T], 每个元素是 expert id

    expert_buckets = [[] for _ in range(len(router_weights))]
    for token_idx, expert_idx in enumerate(assignment):
        expert_buckets[expert_idx].append(token_idx)

    return expert_buckets
```

如果再往工程侧走一步，通常还要补上两个现实问题。

### 4. `T % E != 0` 怎么办

真实训练里 token 数不一定总能被专家数整除，因此需要处理余数。常见方法如下：

| 方法 | 做法 | 代价 |
|---|---|---|
| Padding | 补一些虚拟 token 到最近的可整除数 | 会引入额外无效计算 |
| Uneven quota | 允许少数专家多接收 1 个 token | 失去完全对称的精确均衡 |
| Drop tail | 丢弃末尾少量 token | 实现简单，但会损失样本 |

若令：

$$
q = \left\lfloor \frac{T}{E} \right\rfloor,\quad r = T \bmod E
$$

那么一种常见做法是让前 $r$ 个专家接收 $q+1$ 个 token，其余专家接收 $q$ 个 token。此时约束改写为：

$$
\sum_{t=1}^T \mathbf{1}[a_t=e] \in \{q, q+1\}
$$

并且满足恰好有 $r$ 个专家取到 $q+1$。

### 5. 一个更贴近系统的例子

假设某层当前 worker 上有 2048 个 token、32 个专家，则每个专家理想情况下接收：

$$
2048 / 32 = 64
$$

个 token。路由流程可以写成：

1. 先在本 worker 内计算一个 $2048 \times 32$ 的 affinity 矩阵
2. 再按 chunk 做线性分配，例如切成多个 256 或 512 token 的小块
3. 每个 chunk 内保证精确均衡
4. 最后把 token 按专家重排并发往相应设备

这样做的收益是：在通信开始前，token 已经按均衡结果被整理好，后续 all-to-all 更稳定，不容易出现某些专家设备特别忙、某些设备几乎闲着的情况。

---

## 工程权衡与常见坑

Base Layer 的最大优点和最大代价其实是同一件事：它把负载均衡做成了**硬约束**。优点是训练阶段几乎不会出现专家长期闲置或过载；代价是你必须真的支付组合优化求解的成本。

如果把问题展开成“token 对专家槽位”的方阵，标准精确求解复杂度常写为：

$$
O(n^3)
$$

其中 $n$ 近似等于 token 数或槽位总数。这也是为什么 Base Layer 不能简单理解为“把 Top-k 的 argmax 换成另一个函数调用”。
它改变的不只是路由结果，也改变了路由器本身的计算形态。

下面这张表可以直接说明工程差异：

| 方案 | 计算复杂度 | 调参需求 | 负载均衡 | 通信稳定性 |
|---|---|---|---|---|
| Base Layer | 高，常见为 $O(n^3)$ 级求解或近似求解 | 低 | 强，硬约束 | 高 |
| Top-k + aux loss | 低 | 高，需要调 `capacity` 和 loss 权重 | 中，软约束 | 中到低 |
| Top-k + capacity only | 低 | 中 | 弱，容易挤压或丢弃 token | 低 |

常见坑主要有四类。

### 第一，直接在超大 token 数上做全局求解

这是最常见的误用。理论上“全局一次性最优”很好，工程上却可能让路由器自己成为瓶颈。更合理的做法通常是：

- 按 worker 切
- 按 microbatch 切
- 按 chunk 切
- 在每个 chunk 内做确定性分配

这样做本质上是在交换两种成本：

| 选择 | 收益 | 代价 |
|---|---|---|
| 全量求解 | 更接近全局最优 | 求解太重 |
| 分块求解 | 成本可控 | 只能做到块内最优 |

### 第二，误以为“均衡”就一定“质量更高”

Base Layer 优化的是：

$$
\max \sum_{t=1}^T h_t^\top w_{a_t}
\quad
\text{s.t.}\quad
\sum_{t=1}^T \mathbf{1}[a_t=e]=T/E
$$

它不是在最大化“每个 token 都拿到自己最喜欢的专家”，而是在最大化“**整个批次在均衡约束下的总收益**”。因此某些 token 被分给次优专家是正常现象，不是 bug。

这件事对新手尤其容易误解。可以用一句话记住：

- Top-k 追求的是“单个 token 的局部满意度”
- Base Layer 追求的是“整个批次的全局资源最优”

### 第三，忽略 $T \bmod E \neq 0$ 的处理

真实系统里 token 数并不总是刚好整除专家数。很多文章只写最理想的对称情况，代码里却必须把边界条件写清楚，否则会在以下地方出错：

- quota 计算错误
- 断言失败
- 少量 token 无法分配
- 某些专家名额不一致却没有被路由器显式建模

这个问题不难，但不能省。

### 第四，把 Base Layer 当成所有层的默认路由

这也不对。Base Layer 更适合：

- token 数量受控的层
- 通信代价敏感的层
- 专家利用率重要的层
- 容易出现负载抖动的训练场景

如果某一层的 token 极多，而你又非常依赖快速、廉价、硬件友好的路由，那么 Top-k 仍然更现实。

### 为什么它常被认为能让训练更稳定

Base Layer 带来的收益，很多时候并不来自“更强的表达能力”，而来自更稳定的系统行为。可以把训练吞吐近似理解成：

$$
\text{step time} \approx \max_e(\text{expert } e \text{ 的处理时间}) + \text{通信开销}
$$

如果某些专家长期超载，那么 `max` 项会被少数热点专家拉高。即使平均负载不错，整步训练仍会被尾部延迟拖慢。Base Layer 通过精确配额把这个波动压小，因此更容易得到：

- 更稳定的每步耗时
- 更可预测的 all-to-all 通信
- 更均匀的专家训练机会

但要注意，若分块策略不合理、求解器过慢，收益会被抵消。也就是说，**Base Layer 解决的是负载不均问题，不自动解决路由计算开销问题。**

---

## 替代方案与适用边界

Base Layer 最主要的替代方案仍然是 Top-k 路由配合负载均衡损失。它的优势很直接：

- 便宜
- 简单
- 硬件友好
- 易于扩展到大 token 数

它的缺点也同样直接：均衡只是倾向，不是保证。

如果你的场景是每层每步 token 非常多，例如单层路由前就有 8192 个 token、32 个专家，那么精确线性分配通常不该直接在全量上做。更实际的判断标准如下：

| 场景 | 更合适的方案 | 原因 |
|---|---|---|
| token 数较少，且能稳定切 chunk | Base Layer | 能承受求解成本，换来确定性均衡 |
| token 数极多，路由必须极快 | Top-k | 先保证吞吐 |
| 某些层通信代价很高 | Base Layer 或混合策略 | 稳定负载能减少尾部等待 |
| 训练初期容易专家塌缩 | Base Layer | 精确配额能直接抑制流量坍缩 |
| 后期模型已稳定，追求更便宜训练 | Top-k 或混合策略 | 降低路由成本 |

这里的 `expert collapse`，就是“少数专家长期吸走大部分 token，其他专家几乎学不到东西”。

一个更实用的教程式判断流程可以写得非常明确：

1. 先看某层每 batch 有多少 token、多少专家。
2. 如果 token 数已经大到路由求解会成为瓶颈，不要默认上 Base Layer。
3. 再问是否能切成 2048、1024、甚至更小的 chunk。
4. 如果能切，并且切完后每个 chunk 仍有足够 token 让专家分配有意义，Base Layer 可行。
5. 如果不能切，或者切完会显著破坏并行流水，优先退回 Top-k。
6. 如果训练早期专家塌缩严重，可以考虑前期用 Base Layer、后期切回更便宜的路由。

混合策略在工程上很常见。例如：

| 阶段或层 | 路由策略 | 目的 |
|---|---|---|
| 前几层、token 很多 | Top-k | 保吞吐 |
| 中后层、token 已被裁剪 | Base Layer | 保负载均衡 |
| 训练前期 | Base Layer | 先稳住专家利用率 |
| 训练后期 | Top-k 或近似路由 | 降低成本 |

所以更准确的结论是：Base Layer 不是要取代所有 MoE 路由，而是提供一种在**可控规模**下非常强的、确定性的负载均衡手段。它适合的是“愿意用更高路由成本换更稳定训练行为”的场景，而不是所有场景。

---

## 参考资料

1. Lewis et al., *BASE Layers: Simplifying Training of Large, Sparse Models*. ICML 2021. 重点是把 MoE 路由明确写成线性分配问题，并报告大规模稀疏训练结果。  
2. PMLR 论文页面与 PDF：<https://proceedings.mlr.press/v139/lewis21a.html>、<https://proceedings.mlr.press/v139/lewis21a/lewis21a.pdf>。重点是正式目标函数、算法描述、实验设置与训练表现。  
3. 线性分配问题的经典背景资料，例如匈牙利算法相关教程。可用于理解“为什么一对一匹配能精确表达固定配额路由”，以及 $O(n^3)$ 级复杂度从何而来。  
4. MoE 负载均衡与 expert collapse 的工程讨论资料。重点是理解负载偏斜为什么会导致通信抖动、训练变慢、部分专家学习不足。  
5. 任何介绍稀疏专家模型的综述性材料。阅读时可以重点比较三类路由：`Top-k`、`Top-k + aux loss`、`Base Layer`，不要只看最终指标，也要看吞吐稳定性和实现代价。
