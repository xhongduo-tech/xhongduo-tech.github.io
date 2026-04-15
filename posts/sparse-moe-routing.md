## 核心结论

Sparse MoE routing 的核心不是“把网络做得更复杂”，而是把计算路径变成按输入内容动态选择。MoE 是 Mixture of Experts，直译为“专家混合”，可以理解成一层里放了多个并行子网络；routing 是“路由”，就是决定当前 token 该送到哪些子网络。稀疏的意思是每个 token 只激活很少几个专家，而不是所有专家都算一遍。

这件事解决的是大模型扩展时最现实的问题：参数可以继续加，但每次前向和反向的实际计算不能等比例爆炸。普通稠密 FFN 层里，所有 token 都走同一套参数；Sparse MoE 里，不同 token 会被分流到不同专家，因此模型总参数可以很大，但单个 token 的计算只落在 `top-k` 个专家上。于是“参数容量”和“每步 FLOPs”被部分拆开。

最常见的计算形式是：

$$
s_{t,e} = x_t^\top w_e,\quad
p_{t,e} = \text{softmax}(s_t)_e
$$

$$
S_t = \text{TopK}(p_t, k)
$$

$$
y_t = \sum_{e \in S_t} \tilde p_{t,e} f_e(x_t)
$$

这里 $x_t$ 是第 $t$ 个 token 的表示，$w_e$ 是路由器对专家 $e$ 的打分参数，$f_e$ 是专家网络，$\tilde p_{t,e}$ 是在被选中的专家集合上重新归一化后的权重。

下面这张表先把直观差异压缩出来：

| 方案 | 总参数 | 单 token 实际计算 | 路径是否固定 | 典型问题 |
|---|---:|---:|---|---|
| 稠密 FFN | 中 | 中 | 是 | 扩参数就增计算 |
| Dense MoE | 很高 | 很高 | 否，但全专家都算 | 几乎没有稀疏收益 |
| Sparse MoE | 很高 | 近似随 `k` 增长 | 否，只算 `top-k` | 路由稳定性、负载均衡、通信 |

一个玩具例子足够说明价值。假设有 4 个专家，每个专家都是一个小 FFN，`k=2`。某个 token 经过 router 后，只选中专家 1 和 2，专家 3、4 不参与计算。这样总共保留了 4 套参数，但每个 token 只付出 2 套专家的计算成本。参数规模变大了，单 token 计算没有按 4 倍增长。

---

## 问题定义与边界

Sparse MoE routing 要解决的核心问题不是“模型完全学不会”，而是“稠密层扩展到更大参数时，训练和推理成本很快不可接受”。尤其在 Transformer 里，FFN 往往占据大量参数。如果把 FFN 宽度直接做大，所有 token 都得吃下这笔成本；如果把 FFN 换成专家集合，再让 token 只访问其中少数专家，就能在相似算力下容纳更多参数。

先定义几个术语边界：

| 术语 | 含义 | 白话解释 |
|---|---|---|
| router | 路由器 | 一个小网络，负责决定 token 去哪个专家 |
| expert | 专家 | 一组子网络，通常是 FFN 的替身 |
| top-k | 选前 `k` 个 | 只保留分数最高的几个专家 |
| capacity | 容量上限 | 每个专家一批里最多接收多少 token |
| load balancing | 负载均衡 | 避免少数专家特别忙、其余专家闲置 |

这里要明确边界。Sparse MoE routing 只讨论“token 如何选专家”这一层机制，它不是整个 MoE 系统的全部。完整工程里还至少包含：

| 组成部分 | 是否属于 routing 本体 | 说明 |
|---|---|---|
| 打分与 top-k 选择 | 是 | 决定 token 去向 |
| 专家内部结构 | 否 | 专家可以是 FFN，也可以是别的子网络 |
| capacity 与 token 丢弃策略 | 部分相关 | 属于路由后处理，但影响很大 |
| 负载均衡损失 | 部分相关 | 训练时约束路由分布 |
| 跨设备 all-to-all 通信 | 否，但工程上关键 | 专家常分布在不同设备上 |

对初学者，一个有效对照是：普通 FFN 可以看成“所有 token 走同一条流水线”；Sparse MoE 是“token 先分诊，再去不同窗口处理”。这个“分诊”不是手工规则，而是模型学出来的打分函数。

因此，Sparse MoE 适合回答的问题是：“如何在一层里做条件计算，使不同 token 走不同专家？”它不直接回答“专家应该长什么样”“专家怎么分布到机器上”“多机通信怎么优化”。这些都很重要，但它们是相邻问题，不是同一个问题。

---

## 核心机制与推导

完整链路可以拆成 6 步：输入表示、路由打分、概率化、`top-k` 选择、权重归一化、专家输出聚合。先看公式，再看数值。

设一批 token 中第 $t$ 个 token 的隐藏状态为 $x_t \in \mathbb{R}^d$，有 $E$ 个专家。router 对每个专家给一个分数：

$$
s_{t,e} = x_t^\top w_e
$$

这里内积的含义可以理解为“相似度打分”，也就是 token 当前特征和专家偏好的匹配程度。把所有专家分数做 softmax，得到概率：

$$
p_{t,e} = \text{softmax}(s_t)_e
$$

然后只取前 `k` 个：

$$
S_t = \text{TopK}(p_t, k)
$$

如果直接把原概率截断后相加，和通常不等于 1，所以要在选中的专家上重新归一化：

$$
\tilde p_{t,e} =
\frac{p_{t,e}}{\sum_{j \in S_t} p_{t,j}}, \quad e \in S_t
$$

最后把专家输出按权重求和：

$$
y_t = \sum_{e \in S_t} \tilde p_{t,e} f_e(x_t)
$$

### 玩具例子

设 `E=4, k=2`，某个 token 的路由概率为：

$$
p_t = [0.40, 0.35, 0.15, 0.10]
$$

那么被选中的集合是：

$$
S_t = \{1, 2\}
$$

重新归一化后：

$$
\tilde p_t = \left[\frac{0.40}{0.75}, \frac{0.35}{0.75}\right]
= [0.533, 0.467]
$$

若专家 1 输出 `10`，专家 2 输出 `4`，则最终输出为：

$$
y_t = 0.533 \times 10 + 0.467 \times 4 \approx 7.2
$$

这个例子里，专家 3 和 4 完全没有参与计算。稀疏性就在这里。

### 为什么会出现负载不均衡

如果只按任务损失训练 router，模型很容易学出“某几个专家总是最安全”，于是大量 token 都挤到少数专家上。这叫专家塌缩，意思是专家的使用分布塌到少数几个点上。结果有三层问题：

1. 热门专家 capacity 爆掉，部分 token 被丢弃或降级。
2. 冷门专家几乎没有梯度，长期学不好。
3. 多机训练时，通信和算力利用率都变差。

因此常在训练目标里加负载均衡项。一个常见写法是：

$$
L_{lb} = \lambda E^2 \cdot \text{mean}_e(\rho_e \pi_e)
$$

其中：
- $\rho_e$ 是实际被分配到专家 $e$ 的 token 比例，白话说就是“这个专家真正接了多少活”。
- $\pi_e$ 是 router 给专家 $e$ 的平均概率，白话说就是“路由器平均有多偏爱它”。

这个损失的目标不是让每个 token 平均分，而是让批级别的专家使用更均匀。

### token 路由流程表

| 步骤 | 输入 | 输出 | 关键风险 |
|---|---|---|---|
| 1. 打分 | `x_t`, `w_e` | `s_{t,e}` | 分数尺度不稳定 |
| 2. softmax | `s_t` | `p_t` | 数值精度敏感 |
| 3. `top-k` 选择 | `p_t` | `S_t` | 非连续选择带来训练难点 |
| 4. 归一化 | `p_t, S_t` | `\tilde p_t` | 权重和必须为 1 |
| 5. 专家计算 | `x_t, S_t` | `f_e(x_t)` | capacity 溢出、通信开销 |
| 6. 聚合与反传 | `\tilde p_t, f_e(x_t)` | `y_t` | 热门专家梯度过多，冷门专家梯度不足 |

### 真实工程例子

在大语言模型里，常见做法是把 Transformer block 里的 FFN 层替换成 MoE 层。注意，通常不是替换注意力层，而是替换 FFN，因为 FFN 参数多且结构规则，最适合做专家化。比如 Switch Transformer 选择 `top-1`，也就是每个 token 只去一个专家，这样可以把路由和通信简化到最低；GShard 则把稀疏专家扩展到超大规模多机训练，证明这种条件计算能支撑非常大的参数量。

---

## 代码实现

实现 Sparse MoE，真正难的不是写几个专家，而是把路由、`top-k`、mask、容量限制和聚合路径写对。下面给一个能运行的最小 Python 版本，忽略自动求导和多机通信，但保留核心逻辑。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def topk_indices(values, k):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]

def expert_0(x): return 2.0 * x + 1.0
def expert_1(x): return -1.0 * x + 6.0
def expert_2(x): return 0.5 * x + 2.0
def expert_3(x): return 1.5 * x - 1.0

experts = [expert_0, expert_1, expert_2, expert_3]

def route_token(x, router_w, k=2, capacity=None, expert_load=None):
    scores = [x * w for w in router_w]
    probs = softmax(scores)
    idx = topk_indices(probs, k)

    if capacity is not None and expert_load is not None:
        kept = []
        for i in idx:
            if expert_load[i] < capacity:
                kept.append(i)
                expert_load[i] += 1
        idx = kept

    if not idx:
        return x, probs, []  # 全部溢出时，退回残差路径

    selected_probs = [probs[i] for i in idx]
    z = sum(selected_probs)
    norm_probs = [p / z for p in selected_probs]

    y = sum(norm_probs[j] * experts[idx[j]](x) for j in range(len(idx)))
    return y, probs, idx

router_w = [0.8, 0.6, 0.1, -0.2]
y, probs, idx = route_token(x=5.0, router_w=router_w, k=2)

assert len(probs) == 4
assert idx == [0, 1]
assert abs(sum(probs) - 1.0) < 1e-9
assert y > 0

loads = [0, 0, 0, 0]
y2, probs2, idx2 = route_token(x=5.0, router_w=router_w, k=2, capacity=0, expert_load=loads)
assert idx2 == []
assert y2 == 5.0
```

上面代码对应的最小流程就是：

```python
scores = x @ router_w
probs = softmax(scores)
topk_idx = topk(probs, k)
topk_prob = probs[topk_idx]
topk_prob = topk_prob / topk_prob.sum()
y = sum(topk_prob[i] * experts[topk_idx[i]](x) for i in range(k))
loss = task_loss + lb_loss
```

真实工程中还要处理一批 token，而不是单个 token。通常步骤是：
1. 计算所有 token 的 router logits。
2. 做 softmax 并取 `top-k`。
3. 构造 token-to-expert 的 dispatch mask。
4. 按专家把 token 重新分组。
5. 各专家并行执行。
6. 把结果按原 token 顺序 gather 回来并加权求和。

训练和推理经常不完全一样：

| 项目 | 训练 | 推理 |
|---|---|---|
| router 噪声 | 常见，会加少量噪声促进探索 | 通常关闭 |
| capacity | 常启用，防止单专家过载 | 视延迟预算决定 |
| dropped token | 可能允许，或走残差旁路 | 一般希望严格可控 |
| 精度 | router 常保留 `float32` 更稳 | 可混合精度，但需验证一致性 |
| 负载均衡损失 | 必须有或大多需要 | 不参与 |

如果把这套逻辑放进 Transformer，MoE 层一般替代 FFN 的位置，即 `Attention -> MoE FFN -> Residual`。这样做的原因不是理论必须，而是这里参数密度最高，稀疏化收益最大。

---

## 工程权衡与常见坑

Sparse MoE 最常见的失败，不是论文公式写错，而是工程约束没顶住。真正要看的不只是 loss，还包括利用率、丢弃率、通信量和延迟尾部。

先看问题与规避：

| 问题 | 现象 | 后果 | 常见规避 |
|---|---|---|---|
| 专家塌缩 | 少数专家吃掉大多数 token | 学习不均、capacity 爆掉 | 负载均衡损失、router 噪声、bias 调整 |
| 容量溢出 | 热门专家接收 token 超上限 | dropped token 上升，质量波动 | 调大 `capacity_factor`，改进均衡 |
| 路由不稳定 | 小扰动导致 `top-k` 切换 | 训练振荡，复现差 | router 用高精度，控制初始化和温度 |
| `k` 过大 | 更多专家被同时激活 | 计算和通信接近稠密模型 | 从 `top-1` 或 `top-2` 起步 |
| 训练推理不一致 | 训练有噪声，推理无噪声 | 离线指标和线上表现脱节 | 单独验证推理路径 |

capacity 可以理解为“每个专家这一批最多接待多少 token”。常见近似是：

$$
\text{capacity} = \left\lceil \frac{T \cdot k}{E} \cdot \text{capacity\_factor} \right\rceil
$$

其中 $T$ 是 batch 中 token 数。这个公式的含义很直接：如果理想均匀分配，每个专家平均应接收 $\frac{T \cdot k}{E}$ 个 token，再乘一个冗余系数做缓冲。

一个真实工程坑是：某个专家长期收到 40% 以上 token，而平均值本该只有 12.5%（例如 8 个专家时）。这会产生连锁反应：该专家所在设备显存和计算先成为瓶颈；其他设备空闲等待；通信 all-to-all 出现长尾；即使整体 FLOPs 看起来不高，实际 step time 也会被拖慢。解决时不能只调 loss 权重，通常还要同时检查 router 初始化、温度、容量系数和 batch 规模。

建议监控以下指标：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| `ρ_e` | 专家实际分配比例 | 直接看是否塌缩 |
| `π_e` | router 平均概率 | 看偏好是否早已失衡 |
| dropped-token ratio | 被丢弃 token 比例 | 直接影响质量 |
| expert entropy | 专家分布熵 | 熵太低说明选择过于集中 |
| token throughput | 每秒处理 token 数 | 稀疏收益是否真的兑现 |
| p99 latency | 尾延迟 | 热门专家可能拖慢整批 |

初学者容易忽略的一点是：Sparse MoE 省下的是“被激活专家的局部计算”，但它引入了更复杂的路由和分发。如果专家跨设备放置，token 需要被重新打包并发送到不同机器，通信可能抵消一部分计算收益。所以论文里的参数规模优势，不会自动变成线上吞吐优势。

---

## 替代方案与适用边界

Sparse MoE 不是默认最优解。它在“大参数、可分布式、训练预算足、希望提高容量”这些条件下非常强；在“小模型、低延迟、单机部署、通信很贵”的场景下，未必划算。

下面直接比较：

| 方案 | 计算量 | 通信量 | 实现复杂度 | 质量上限 | 适用场景 |
|---|---|---:|---:|---:|---|
| 稠密 FFN | 稳定且固定 | 低 | 低 | 中到高 | 小中型模型、低延迟部署 |
| Sparse MoE top-1 | 低 | 中 | 中 | 高 | 超大规模训练、追求吞吐 |
| Sparse MoE top-2 | 中 | 更高 | 更高 | 通常更稳 | 大规模训练、希望质量更优 |
| Soft MoE / 稠密门控 | 高 | 中 | 中 | 中到高 | 想保留可微分平滑路由 |
| 共享专家 + 少量专属专家 | 中 | 低到中 | 中 | 中到高 | 想降低塌缩风险 |
| 固定分组专家 | 中 | 低 | 中 | 受限 | 路由逻辑需简单可控 |

两个场景对比最能说明边界。

真实工程例子一：边缘设备推理。设备算力小、内存紧、通信几乎没有多机空间，且请求延迟要求高。这种情况下，Sparse MoE 的动态路由和额外分发逻辑可能不划算，稠密 FFN 或轻量门控层往往更稳。

真实工程例子二：超大规模训练。比如数百亿到万亿参数的语言模型，训练在大集群上进行，目标是在固定 FLOPs 下提升参数容量。这里 Sparse MoE 的优势明显，因为可以把专家分散到多设备，单 token 只激活少数专家，让总参数继续上升而不让单步计算线性爆炸。

一个实用选择建议：

| 情况 | 建议 |
|---|---|
| 先求系统简单、吞吐优先 | 从 `top-1` 开始 |
| 质量更重要，且可承受更高通信 | 考虑 `top-2` |
| 模型不大，或单机场景为主 | 先不要上 MoE |
| 路由长期不稳、丢弃率高 | 先把稠密基线做扎实 |
| 专家并行基础设施成熟 | 再考虑扩大专家数 |

和代表性工作相比，可以这样理解边界：
- Shazeer 2017 提出了稀疏门控专家层的基本框架，重点是“条件计算可以把模型做得很大”。
- GShard 强调“如何把这种稀疏专家放到大规模分布式系统里”。
- Switch Transformer 则进一步简化为 `top-1` 路由，核心取舍是“少一点表达灵活性，换更简单稳定的工程实现”。

所以，是否使用 Sparse MoE，关键不是“论文里它强不强”，而是你的目标函数是什么。如果目标是把容量做大并维持训练算力可控，它很合适；如果目标是单机、低延迟、强稳定性，稠密结构常常更省心。

---

## 参考资料

| 资料 | 年份 | 作用 | 链接 |
|---|---:|---|---|
| Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | 2017 | 稀疏门控 MoE 的基础论文，给出经典 routing 与负载均衡思路 | https://arxiv.org/abs/1701.06538 |
| GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | 2020 | 说明稀疏专家如何进入大规模分布式训练系统 | https://arxiv.org/abs/2006.16668 |
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | 2021 | 用 `top-1` 路由显著简化 MoE 工程实现 | https://arxiv.org/abs/2101.03961 |
| Google Research GShard 页面 | 2020 | 补充 GShard 的工程背景与应用语境 | https://research.google/pubs/gshard-scaling-giant-models-with-conditional-computation-and-automatic-sharding/ |

推荐阅读顺序：
1. 先读 Shazeer 2017，建立“稀疏专家 + 路由 + 负载均衡”的基本概念。
2. 再读 GShard，理解为什么分布式系统会成为 Sparse MoE 成败关键。
3. 最后读 Switch Transformer，理解为什么工业实现经常宁可用更简单的 `top-1`。

正文里的核心公式来源主要对应 Shazeer 2017 的稀疏门控专家层，以及后续 GShard、Switch Transformer 对路由和工程实现的扩展讨论。
