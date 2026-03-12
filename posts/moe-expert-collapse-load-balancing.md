## 核心结论

MoE，Mixture of Experts，中文常译为“专家混合”，指一层里放很多个前馈子网络，但每个 token 只激活其中少数几个。它的核心收益是“参数很多，但每次只算一小部分”。

专家崩塌，指路由器长期把大多数 token 都发给少数几个专家，其他专家几乎闲置。对白话解释就是：本来想把工作分给很多人，最后却变成 1 到 2 个人一直加班，其余人没活干。

这不是小问题。因为 MoE 的速度优势依赖“负载分散”。一旦 token 集中到少数专家，训练会同时出现三类退化：

1. 稀疏计算退化成局部密集计算，吞吐下降。
2. 闲置专家几乎收不到梯度，能力越来越弱。
3. 路由器形成正反馈，越偏向头部专家越难纠正。

最常见也最有效的工程组合，不是只靠单一技巧，而是同时使用：

| 机制 | 主要作用 | 解决什么问题 |
|---|---|---|
| `L_balance` 辅助均衡损失 | 惩罚负载不均 | 防止少数专家长期吃满 |
| Noisy Top-K | 给路由加噪声探索 | 避免早期过早锁死 |
| Capacity Factor | 给每个专家设容量上限 | 阻止单个专家爆满 |
| Router Z-loss | 压制 logit 过大 | 提高 softmax 数值稳定性 |
| Bias-based balancing | 在选 top-k 前修正偏置 | 不增加额外梯度干扰 |

一个新手版直觉例子：top-1 路由下，某个专家在训练早期只是“偶尔表现更好”，softmax 就会把这点优势指数放大，更多 token 会流向它；更多 token 又让它继续得到更多梯度；这个循环持续后，它会吸走几乎全部 token，其他专家等于被废弃。

下面这张表能直观看到崩塌前后差异：

| 状态 | 8 个专家的平均 token 数 | 最忙专家 token 数 | 最闲专家 token 数 |
|---|---:|---:|---:|
| 负载均衡 | 256 | 280 左右 | 230 左右 |
| 专家崩塌 | 256 | 512 甚至更多 | 0 到 40 |

---

## 问题定义与边界

先定义问题。设：

- $E$：专家数。
- $k$：每个 token 选择的专家数，top-1 就是 $k=1$，top-2 就是 $k=2$。
- $B \times T$：一个 batch 的 token 总数。
- $f_i$：第 $i$ 个专家实际接收的 token 比例。
- $P_i$：路由器给第 $i$ 个专家的平均概率。
- $z_{t,i}$：第 $t$ 个 token 对第 $i$ 个专家的路由 logit。
- $CF$：capacity factor，容量因子，用来放宽或收紧单专家容量上限。

专家崩塌的正式定义，不是“有点不均衡”，而是负载和路由概率都明显集中到少数专家，并持续多个 step。常见观察指标有三类：

| 指标 | 含义 | 理想状态 |
|---|---|---|
| expert token share | 每个专家拿到的 token 占比 | 接近均匀 |
| router entropy | 路由分布熵，衡量分散程度 | 不应过低 |
| load imbalance factor | 负载不均衡倍率 | 越接近 1 越好 |

一个常用的不均衡倍率可以写成：

$$
\text{Imbalance}_i = E \cdot f_i
$$

如果完全均匀，$f_i=\frac{1}{E}$，那么 $\text{Imbalance}_i=1$。如果某个专家分到双倍平均 token，那么它的倍率就是 2。

专家容量常写为：

$$
\text{capacity} = CF \cdot \frac{B \cdot T \cdot k}{E}
$$

它表示单个专家在当前 batch 最多处理多少 token。超过容量的 token，要么被丢弃，要么重路由，要么留给后备逻辑处理。

玩具例子：8 个专家、2048 个 token、top-1 路由。理想情况下每个专家应处理：

$$
\frac{2048}{8}=256
$$

如果其中一个专家拿到 512 个 token，那么：

$$
f_i = \frac{512}{2048}=0.25,\quad E \cdot f_i = 8 \cdot 0.25 = 2
$$

这说明它承担了两倍于理想平均值的负载。对白话解释就是：这个专家已经满到溢出，而别的专家还空着。计算资源没有真正并行展开，稀疏结构的意义被破坏了。

如果此时设置 $CF=1.25$，则单专家容量是：

$$
1.25 \cdot 256 = 320
$$

也就是说，512 个 token 里至少有 192 个无法按原计划进入该专家。它们必须被重定向，或者直接丢弃。这就是容量机制为什么既能保护系统，又可能伤害质量。

边界也要说清楚：

1. 专家崩塌主要发生在训练阶段，不是推理阶段才出现的问题。因为路由器偏置和专家能力是在训练中一起形成的。
2. top-1 比 top-2 更容易崩塌。因为 token 没有第二候选，选择更“硬”。
3. batch 较小、数据分布单一、专家数过多时，更容易看到短时不均衡。
4. 负载不均衡不等于一定崩塌。短期波动很正常，关键是是否持续、是否导致熵下降和闲置专家长期无梯度。

---

## 核心机制与推导

### 1. 为什么 softmax 会强化崩塌

softmax 的作用是把 logit 变成概率。对白话解释就是：把“偏好分数”转成“分配比例”。

$$
p_{t,i}=\frac{e^{z_{t,i}}}{\sum_{j=1}^{E} e^{z_{t,j}}}
$$

指数函数会放大小差异。比如两个专家 logit 只差 1，概率差就可能已经很明显；如果差 3，几乎就锁死了。这样一来，某专家只要早期略占优势，就会吸到更多 token，得到更多梯度，再继续领先。这就是正反馈。

### 2. 辅助均衡损失 $L_{balance}$

经典形式之一是：

$$
L_{balance}=E\cdot\sum_{i=1}^{E} f_iP_i
$$

这里：

- $f_i$ 是“实际拿到了多少 token”。
- $P_i$ 是“路由器平均上有多偏爱它”。

如果系统完美均衡，那么每个专家都有 $f_i=P_i=\frac{1}{E}$，代入得到：

$$
L_{balance}=E\cdot E\cdot\frac{1}{E}\cdot\frac{1}{E}=1
$$

所以它的最小值接近 1。这个式子的直觉是：如果某些专家既被偏爱，又真的拿走大量 token，那么乘积会变大，损失会上升。训练就会被迫把流量往其他专家分散。

可以把它理解成“实际流量”和“路由倾向”的联合惩罚，而不是只看其中一项。

### 3. Noisy Top-K

Noisy Top-K 的思路是在选 top-k 前先加噪声。对白话解释就是：别让路由器太早认定“只有这几个专家值得选”，先保留一点探索。

简化写法：

$$
\tilde{z}_{t,i}=z_{t,i}+\epsilon_{t,i},\quad \epsilon_{t,i}\sim \mathcal{N}(0,\sigma_i^2)
$$

然后用 $\tilde{z}_{t,i}$ 选 top-k，再在入选专家上做 softmax。

它的作用不是直接“拉平均”，而是让边缘专家在训练早期也有机会被选中，从而拿到梯度。如果没有这一步，top-1 很容易在前期就把弱势专家永久边缘化。

### 4. Capacity Factor

容量机制不是让路由更公平，而是给不公平加硬约束。定义仍是：

$$
\text{capacity}=CF\cdot\frac{B\cdot T\cdot k}{E}
$$

如果某专家超过 capacity，多余 token 不能继续进入。这相当于告诉系统：“你可以偏心，但不能无限偏心。”

它能立刻阻止最坏情况，但副作用也明确：

1. 如果重路由策略弱，token 可能被丢弃。
2. 如果 $CF$ 太小，很多 batch 都会发生截断。
3. 如果 $CF$ 太大，又失去约束效果。

### 5. Router Z-loss

Z-loss 常写成：

$$
L_z=\frac{1}{B}\sum_{t=1}^{B}\left(\log\sum_{i=1}^{E} e^{z_{t,i}}\right)^2
$$

这里的 $\log \sum e^z$ 也叫 log-sum-exp，它近似代表 logit 的整体尺度。Z-loss 惩罚的不是“哪位专家更好”，而是“整体分数是不是越来越大”。

直觉上可以这样理解：如果所有 logit 都越拉越大，softmax 会越来越尖锐，小差距也会被放大成近乎 0/1 的决策。Z-loss 就是在给这种“过尖”加刹车。它不是直接均衡负载，但会减缓路由器进入极端状态。

### 6. Bias-based balancing

近年的一类方法是 loss-free balancing，即不额外加辅助损失，而是在 top-k 前给各专家加一个动态 bias：

$$
\hat{z}_{t,i}=z_{t,i}+b_i
$$

其中 $b_i$ 根据历史负载更新。负载高的专家，bias 往下调；负载低的专家，bias 往上调。它的特点是直接作用于分配决策，而不是通过额外梯度慢慢“说服”路由器。

真实工程例子：大规模语言模型训练里，最慢卡往往由“最忙专家”所在设备决定。即使平均 FLOPs 没变，只要少数专家持续爆满，多机并行就会被拖慢。此时 bias-based balancing 的优势不是“理论更优雅”，而是它常能在不明显伤害主损失的情况下更快拉平各卡负载。

---

## 代码实现

下面给出一个可运行的玩具实现，展示一条典型路由链路：

`clean logits -> bias -> noise -> top-k -> softmax -> assignment -> capacity clamp -> aux loss`

```python
import math
import random

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def route_tokens(token_logits, k=1, capacity_factor=1.25, bias=None, noise_std=0.0):
    """
    token_logits: List[List[float]], shape = [num_tokens, num_experts]
    """
    num_tokens = len(token_logits)
    num_experts = len(token_logits[0])
    bias = bias or [0.0] * num_experts

    capacity = math.ceil(capacity_factor * (num_tokens * k) / num_experts)
    assigned = [[] for _ in range(num_experts)]
    probs_mean = [0.0] * num_experts

    # 先统计 clean+bias 后的平均概率，用于 balance loss
    for row in token_logits:
        adjusted = [x + b for x, b in zip(row, bias)]
        p = softmax(adjusted)
        for i in range(num_experts):
            probs_mean[i] += p[i]
    probs_mean = [x / num_tokens for x in probs_mean]

    dropped = 0
    for token_id, row in enumerate(token_logits):
        adjusted = []
        for i, x in enumerate(row):
            noisy = x + bias[i] + random.gauss(0.0, noise_std)
            adjusted.append((i, noisy))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        topk = adjusted[:k]

        # top-k 内再做 softmax，得到门控权重
        topk_scores = softmax([x[1] for x in topk])

        # 这里演示 top-1 / top-k 的主分配逻辑
        chosen = []
        for (expert_id, _), gate in zip(topk, topk_scores):
            chosen.append((expert_id, gate))

        for expert_id, gate in chosen:
            if len(assigned[expert_id]) < capacity:
                assigned[expert_id].append((token_id, gate))
            else:
                dropped += 1

    token_share = [len(x) / num_tokens for x in assigned]
    balance_loss = num_experts * sum(f * p for f, p in zip(token_share, probs_mean))

    return {
        "capacity": capacity,
        "assigned_counts": [len(x) for x in assigned],
        "token_share": token_share,
        "probs_mean": probs_mean,
        "balance_loss": balance_loss,
        "dropped": dropped,
    }

# 玩具例子：4 个专家，16 个 token，专家 0 初始更强
random.seed(0)
logits = []
for _ in range(16):
    logits.append([2.0, 1.0, 0.8, 0.7])

result_bad = route_tokens(logits, k=1, capacity_factor=4.0, bias=[0, 0, 0, 0], noise_std=0.0)
result_better = route_tokens(logits, k=1, capacity_factor=1.25, bias=[-0.8, 0.2, 0.2, 0.2], noise_std=0.3)

assert sum(result_bad["assigned_counts"]) + result_bad["dropped"] == 16
assert sum(result_better["assigned_counts"]) + result_better["dropped"] == 16
assert result_bad["assigned_counts"][0] >= result_better["assigned_counts"][0]
assert result_better["capacity"] == math.ceil(1.25 * 16 / 4)

print(result_bad)
print(result_better)
```

这段代码不是生产实现，但足够说明四个关键点：

| 代码段 | 作用 | 可调参数 |
|---|---|---|
| `bias` | 对高负载专家降温，对低负载专家升温 | `bias[i]` |
| `random.gauss` | 给路由探索空间 | `noise_std` |
| `capacity` | 防止单专家爆满 | `capacity_factor` |
| `balance_loss` | 把不均衡反馈到训练目标 | `lambda_balance` |

如果把它放进真实模型，一般流程是：

1. 用一个 router 线性层生成 `clean logits`。
2. 加上可学习或动态更新的 `bias`。
3. 训练早期加入噪声，后期可衰减。
4. 选 top-k。
5. 在选中的专家上归一化 gate。
6. 根据 capacity 截断、重路由或记录丢弃。
7. 统计 `f_i`、`P_i`、entropy、dropped tokens。
8. 把 `L_main + \lambda L_balance + \beta L_z` 一起回传。

真实工程例子：如果你在 64 卡上做 expert parallel，专家不均衡不仅是“数学问题”，还会直接表现为 step time 抖动。一台卡上专家过载，其他卡先算完也得等它，最终训练墙钟时间被最慢专家决定。

---

## 工程权衡与常见坑

真正难的不是“知道这些机制存在”，而是知道参数怎么一起调。

| 常见坑 | 现象 | 规避手段 |
|---|---|---|
| `λ_balance` 太大 | 主任务 loss 下降变慢，模型学会“平均分”但不学内容 | 从很小值起扫参，用验证集看主指标 |
| `λ_balance` 太小 | token 仍集中到少数专家 | 联合观察 token share 与 entropy |
| `noise_std` 太大 | 路由像随机抽签，专家难收敛 | 训练前期大一些，后期衰减 |
| `noise_std` 太小 | 探索不足，早期锁死 | 尤其在 top-1 下要谨慎 |
| `CF` 太小 | token 经常被截断或丢弃 | 统计 dropped token ratio |
| `CF` 太大 | 容量约束形同虚设 | busiest expert 长期过载时下调 |
| 只看平均 loss | 忽略局部专家崩塌 | 必须记录每个专家的 token 占比 |
| 只做单 batch 判断 | 误把正常波动当崩塌 | 用滑动窗口看趋势 |

建议长期监控这些日志名：

| 指标名 | 含义 | 经验信号 |
|---|---|---|
| `expert_token_share[i]` | 第 i 个专家 token 占比 | 头部专家长期超均值 2 倍要警惕 |
| `router_entropy` | 路由分布熵 | 持续下降说明越来越尖锐 |
| `balance_loss` | 均衡损失 | 长期偏高说明分配失衡 |
| `dropped_token_ratio` | 被容量截断的 token 比例 | 上升说明 capacity 过紧 |
| `max_expert_load / mean_load` | 最忙专家负载倍率 | 接近或超过 2 通常不健康 |

对新手最重要的一条经验是：不要只看“模型最终能不能跑通”，要看“负载改进是否真的换来了整体收益”。有些配置会让图表更好看，但下游任务准确率反而下降。

识别方法也很直接：

1. 先做小模型 ablation，对比 `baseline / +balance / +noise / +bias / 组合方案`。
2. 同时记录验证集指标和负载指标。
3. 如果负载更平但主指标更差，先减小 `λ_balance` 或减小噪声，而不是继续堆机制。
4. 如果吞吐差但验证指标更好，要检查是不是 capacity 太松，导致局部专家过载拖慢并行。

---

## 替代方案与适用边界

不同方法解决的是不同层面的问题，不能混为一谈。

| 方案 | 优点 | 局限 | 更适合什么场景 |
|---|---|---|---|
| 辅助均衡损失 | 直接优化均衡目标，简单常用 | 会和主损失争夺梯度 | 预训练、大模型、已有成熟调参流程 |
| Noisy Top-K | 增强探索，缓解早期锁死 | 噪声过强会伤收敛 | top-1 路由、训练早期 |
| Capacity Factor | 硬限制最坏负载 | 可能丢 token | 多机训练、吞吐敏感场景 |
| Z-loss | 稳定 logit 尺度 | 不直接保证均衡 | 路由 logit 经常过尖的系统 |
| Bias-only / Loss-Free | 不额外引入辅助梯度 | 依赖偏置更新策略 | 微调、延迟敏感、想减少训练干扰的场景 |

一个实用判断：

- 如果你在做大规模预训练，优先考虑 `辅助 loss + Noisy Top-K + 合理 CF + Z-loss`。这是更稳的组合。
- 如果你在做微调，尤其下游任务对主损失很敏感，可以优先试 `bias-only + 轻量 capacity`，因为它对主目标的干扰更小。
- 如果资源有限，先别一次开全套。最小可复现流程通常是：先开 `CF`，再看是否仍崩塌；若仍崩塌，再加小权重 `L_balance`；若训练早期锁死明显，再加入轻量噪声。

新手可复现的简化流程：

1. 先用 top-1 MoE 跑一个小模型 baseline。
2. 记录 `expert_token_share` 和 `router_entropy`。
3. 加入 `CF=1.25`。
4. 若头部专家仍长期超过均值 2 倍，再加很小的 `λ_balance`。
5. 若前几个 epoch 熵快速掉到底，再加 Noisy Top-K，并在后期衰减噪声。
6. 若主任务指标明显受损，尝试改成 bias-based balancing。

适用边界也必须明确：

1. 小模型和小数据集里，专家崩塌未必是主要瓶颈，收益可能不如直接用稠密 FFN。
2. 如果任务强依赖少数模式，过度均衡可能反而抹平有效专家分工。
3. 推理场景如果路由已冻结，训练期的均衡技巧不会自动转化为更低延迟，还要看部署时专家并行和批处理方式。

---

## 参考资料

- Switch Transformer. 提出大规模 top-1 路由实践，说明稀疏 MoE 如何在超大参数规模上保持可训练性。重要性在于它把“能不能训练”变成了真实工程问题。
- Shazeer 等人的 Sparsely-Gated MoE 系列工作。早期系统化定义了稀疏门控与负载均衡问题，是后续很多 balance loss 设计的源头。
- Michael Brenndoerfer, *MoE Load Balancing: Token Distribution & Expert Collapse*. 对专家崩塌、token 分布、负载监控有面向工程实现的清晰解释，适合先建立整体直觉。
- Michael Brenndoerfer, *Auxiliary Balancing Loss in MoE*. 重点解释 $L_{balance}$ 中 $f_i$ 与 $P_i$ 的含义，适合定位公式为什么这样写。
- Michael Brenndoerfer, *Router Z-Loss: Numerical Stability*. 重点说明 Z-loss 如何压制路由 logit 规模，适合理解“为什么不均衡有时来自数值过尖”。
- Loss-Free Balancing for Mixture of Experts. 代表一类 bias-based balancing 思路，核心贡献是尽量不通过额外辅助损失来修正负载。
- 关于 Capacity Factor 的工程论文和实现说明。重点看 capacity 设置、token 截断、重路由对吞吐和质量的影响。

阅读这些资料时，优先定位三类信息：

1. 公式里到底惩罚了什么变量。
2. 实验里监控了哪些负载指标。
3. 结论是在预训练、微调还是部署场景下成立。

如果要持续追踪最新进展，重点关注三类更新：大型开源 MoE 项目的路由实现、训练日志里公开的 expert load 指标、以及新论文是否减少了“均衡改进但主任务退化”的副作用。
