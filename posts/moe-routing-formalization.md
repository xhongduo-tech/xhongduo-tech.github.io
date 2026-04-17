## 核心结论

MoE，Mixture of Experts，中文通常叫“专家混合模型”，本质是把一个大前馈层拆成多个专家网络，再让一个 router，也就是“分发器”，决定每个 token 应该交给哪些专家处理。它的数学核心可以写成：

$$
G(x)=\operatorname{TopK}\left(\operatorname{softmax}\left(\frac{Wx+b}{\tau}\right)\right)
$$

其中 $x$ 是当前 token 的表示，$W,b$ 是 router 参数，$\tau$ 是 softmax 温度，Top-K 表示“只保留分数最高的 K 个专家”。

结论有三点。

1. MoE 的收益来自“条件计算”。条件计算的意思是：参数很多，但每次只用一小部分参数。这样模型容量大，但单个 token 的计算量不会线性增长到所有专家之和。
2. $\tau$ 直接控制专家分配是否集中。$\tau$ 小，概率分布更尖锐，少数专家会被频繁选中；$\tau$ 大，分布更平滑，负载更均衡，但稀疏性变弱。
3. Top-K 截断会改变梯度流。梯度流就是“误差如何反向传回参数”。被选中的专家和对应 router 路径能得到主要梯度，未选中的专家梯度很弱甚至为零，所以训练稳定性必须依赖额外机制。

一个玩具例子先建立直觉。假设有 4 个专家，router 算出分数后只选前 2 个。token 不会同时经过 4 个专家，而是只经过最合适的 2 个，最后再按权重把它们的输出加权求和。这就是稀疏门控。

| 温度 $\tau$ | softmax 分布形状 | 专家利用率 | 常见结果 |
|---|---|---|---|
| 低 | 尖锐 | 集中到少数专家 | 易过载，稀疏性强 |
| 高 | 平滑 | 更均匀 | 更稳，但计算更接近稠密 |
| 过低 | 接近 one-hot | 极端不均 | 容易出现专家“睡死” |
| 过高 | 接近均匀 | 很平均 | Top-K 区分度下降 |

---

## 问题定义与边界

问题可以表述为：在有 $E$ 个专家的前提下，如何让每个 token 只激活其中 $K$ 个专家，同时尽量不损失模型表达能力。

设输入 token 表示为 $x\in\mathbb{R}^d$，专家数为 $E$，则 router 输出一个长度为 $E$ 的分数向量。这个向量经过 softmax 后可以看作“每个专家被选中的相对概率”。再经过 Top-K 截断，只保留最大的 $K$ 个值。

边界同样要说清楚。

1. 这里讨论的是 token-level routing，也就是“每个 token 单独选专家”，不是句子级、层级或任务级路由。
2. 这里讨论的是 Top-K sparse routing，不是 dense routing。dense routing 指所有专家都参与，只是权重不同。
3. 这里主要分析训练阶段的梯度与容量问题，不展开分布式通信实现细节，比如 all-to-all 的具体优化。

可以把流程压缩成一张表：

| 步骤 | 数学对象 | 白话解释 | 主要风险 |
|---|---|---|---|
| 输入映射 | $z=Wx+b$ | 给每个专家打分 | 初始分数偏置 |
| 概率化 | $\pi=\operatorname{softmax}(z/\tau)$ | 把分数变成可比较权重 | 温度过低或过高 |
| 稀疏选择 | $\operatorname{TopK}(\pi)$ | 只保留前 K 个专家 | 梯度稀疏 |
| 归一化 | $\hat{\pi}$ | 让保留的权重和为 1 | 权重过度集中 |
| 容量约束 | capacity mask | 超载专家丢弃或重路由 token | 负载失衡 |

新手版直觉可以用一句话概括：像打电话求助时你只能同时找 2 个人，router 会先估计谁最适合，然后把任务交给他们；没被选中的人这次不参与，也通常拿不到这次的主要训练信号。

---

## 核心机制与推导

先写完整计算式。令专家集合为 $\{E_1,\dots,E_E\}$，router 先计算：

$$
z(x)=Wx+b,\qquad \pi_i(x)=\frac{\exp(z_i(x)/\tau)}{\sum_{j=1}^{E}\exp(z_j(x)/\tau)}
$$

再取前 $K$ 个专家集合 $S(x)$：

$$
S(x)=\operatorname{TopK}(\pi(x))
$$

然后只对被选中的专家重新规范化：

$$
\hat{\pi}_i(x)=
\begin{cases}
\frac{\pi_i(x)}{\sum_{j\in S(x)} \pi_j(x)}, & i\in S(x) \\
0, & i\notin S(x)
\end{cases}
$$

最终输出为：

$$
y(x)=\sum_{i=1}^{E}\hat{\pi}_i(x)E_i(x)=\sum_{i\in S(x)}\hat{\pi}_i(x)E_i(x)
$$

这就是“先软分配，再硬截断，再局部归一化”。

看一个必须掌握的玩具例子。设有 4 个专家，$K=2$，$\tau=1$，且：

$$
Wx+b=[2.2,1.0,0.5,-0.3]
$$

softmax 后近似得到：

$$
\pi=[0.58,0.24,0.10,0.08]
$$

Top-2 选择第 1、2 个专家，重新归一化后：

$$
\hat{\pi}=[0.71,0.29,0,0]
$$

于是输出是：

$$
y=0.71E_1(x)+0.29E_2(x)
$$

如果把温度改成 $\tau=0.5$，实际上是把 logits 放大 2 倍，softmax 会更尖锐，第 1 个专家权重进一步上升，第 2 个下降。这说明低温度会提高“赢家通吃”倾向。

为什么温度影响专家利用率，可以从 softmax 的形式直接看出来。softmax 比较的是相对差值 $\frac{z_i-z_j}{\tau}$。当 $\tau$ 变小，同样的分数差会被放大，所以排名靠前的专家概率会快速抬升。若很多 token 都出现同样现象，少数专家就会收到过多 token，形成 capacity overflow，也就是“专家容量溢出”。

容量可以用期望值近似理解。若一个 batch 有 $B$ 个 token，每个 token 选 $K$ 个专家，则总分配次数约为 $BK$。如果第 $i$ 个专家平均被选中的概率是 $p_i$，那么它的期望负载约为：

$$
\mathbb{E}[N_i]\approx BKp_i
$$

若所有专家完全均匀，则 $p_i\approx \frac{1}{E}$。此时期望负载约为：

$$
\mathbb{E}[N_i]\approx \frac{BK}{E}
$$

这就是 capacity factor 设计的基础。工程上常设专家容量为：

$$
C=\text{capacity\_factor}\times \frac{BK}{E}
$$

如果某些 $p_i$ 明显大于 $\frac{1}{E}$，这些专家就更容易超载，超出的 token 要么被丢弃，要么退回残差路径，要么重分配到次优专家。

真实工程例子是大模型中的 Switch Transformer 或其他稀疏 LLM 层。一个 batch 的 token 会跨设备分发到多个专家，router 不仅决定“语义上谁最合适”，还间接决定“通信负载是否均匀”。所以 MoE 路由不是单纯的数学选择问题，而是精度、吞吐和分布式负载的联合优化问题。

梯度方面要特别小心。Top-K 之后，未选专家的 $\hat{\pi}_i=0$，它们通常拿不到主要梯度。这意味着：
- 被频繁选中的专家越学越强。
- 很少被选中的专家可能长期学不到东西。
- router 也会强化已有偏好，形成正反馈。

因此很多实现会加 auxiliary load loss 或 importance loss，目标是让各专家的使用率不要偏离均匀分布太远。

---

## 代码实现

下面给出一个可运行的 Python 版玩具实现，只保留 router 的核心逻辑：线性投影、softmax、Top-K、重新归一化，以及容量估计。

```python
import math

def softmax(xs, tau=1.0):
    scaled = [x / tau for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]

def topk_route(logits, k=2, tau=1.0):
    probs = softmax(logits, tau=tau)
    top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    top_sum = sum(probs[i] for i in top_idx)
    gated = [0.0] * len(probs)
    for i in top_idx:
        gated[i] = probs[i] / top_sum
    return probs, top_idx, gated

def expected_capacity(batch_size, k, num_experts, capacity_factor=1.25):
    return capacity_factor * batch_size * k / num_experts

# 玩具例子
logits = [2.2, 1.0, 0.5, -0.3]
probs, top_idx, gated = topk_route(logits, k=2, tau=1.0)

assert top_idx == [0, 1]
assert abs(sum(probs) - 1.0) < 1e-9
assert abs(sum(gated) - 1.0) < 1e-9
assert gated[0] > gated[1] > 0
assert gated[2] == 0.0 and gated[3] == 0.0

# 温度更低时，头部专家更集中
probs_tau1, _, _ = topk_route(logits, k=2, tau=1.0)
probs_tau05, _, _ = topk_route(logits, k=2, tau=0.5)
assert probs_tau05[0] > probs_tau1[0]

# 容量估计
cap = expected_capacity(batch_size=128, k=2, num_experts=8, capacity_factor=1.25)
assert abs(cap - 40.0) < 1e-9
```

这段代码里最重要的是两件事。

1. `topk_route` 先用 softmax 算全局概率，再只保留前 `k` 个，并做二次归一化。
2. `expected_capacity` 用 `capacity_factor * batch_size * k / num_experts` 估计单专家容量上限。这不是严格概率上界，而是工程上的配额线。

如果写成伪代码，流程更直接：

```python
logits = linear(x)
probs = softmax(logits / tau)
topk_idx = select_topk(probs, k)
weights = renormalize(probs[topk_idx])
y = sum(weights[i] * experts[topk_idx[i]](x) for i in range(k))
```

真实工程里还会多几步：
- 记录每个专家接收到多少 token。
- 对超出容量的 token 做 mask。
- 统计 load loss。
- 训练初期使用 soft gating 或 noisy top-k，避免路由过早塌缩。

---

## 工程权衡与常见坑

MoE 路由最常见的问题，不是公式不会写，而是训练时很容易失衡。

| 问题 | 现象 | 根因 | 常见缓解策略 |
|---|---|---|---|
| 梯度稀疏 | 少数专家越学越强 | Top-K 截断导致未选专家缺梯度 | soft warm-up、EMA 默认输出、noisy top-k |
| 负载失衡 | 某些专家超载 | router 偏向头部专家 | load loss、importance loss、调高 $\tau$ |
| 专家塌缩 | 大量专家长期不工作 | 早期偏置被放大 | 温度退火、路由噪声、均衡初始化 |
| 容量溢出 | token 被丢弃或回退 | 单专家接收 token 过多 | 提高 capacity factor、增加专家数 |
| 稀疏性不足 | 速度收益不明显 | 温度过高或 K 过大 | 降低 $\tau$、减小 K |

一个典型坑是把 $\tau$ 设得太低。这样 router 很快变成“永远选那几个专家”。训练前期这看起来 loss 降得快，但长期会导致两个后果：一是专家利用率极低，二是热门专家频繁 overflow。另一种相反错误是把 $\tau$ 设得太高，结果所有专家分数都差不多，Top-K 虽然还在做，但选择几乎接近随机，模型很难形成明确专长。

一个实用策略是温度调度。调度就是“按训练阶段改变参数”。比如前期用较高温度鼓励探索，中后期逐步降低温度，让专家分工变清晰。

另一个常见坑是只看平均 loss，不看专家分布。如果 16 个专家里始终只有 3 个专家高负载，说明 router 已经局部失效，即使主任务指标暂时还行，后面也会出现吞吐波动和可扩展性问题。

真实工程中，大规模 LLM 的 MoE 层通常会同时监控：
- 每个专家的 token 数分布。
- token 丢弃率或回退率。
- load loss 和主任务 loss 的比例。
- 不同温度和 capacity factor 下的吞吐变化。

---

## 替代方案与适用边界

Top-K MoE 不是唯一方案。它适合“参数规模大、希望单 token 计算量受控”的场景，但并不适合所有模型。

| 方案 | 核心公式 | 适用场景 | 主要缺点 |
|---|---|---|---|
| Top-K MoE | $y=\sum_{i\in TopK}\hat{\pi}_iE_i(x)$ | 大模型、希望条件计算 | 路由复杂、负载不稳 |
| Dense soft routing | $y=\sum_i \pi_iE_i(x)$ | 小中型模型、实现简单 | 所有专家都算，省不了算力 |
| Hard gating | $y=E_{i^\*}(x),\ i^\*=\arg\max_i \pi_i$ | 推理极致稀疏 | 训练不可导，常需 STE |
| Soft-to-sparse 过渡 | 先 dense 后 Top-K | 训练稳定性优先 | 调参更复杂 |

dense soft routing 的好处是训练平滑，因为所有专家都能收到梯度；坏处也很直接，所有专家都得算，失去 MoE 最关键的条件计算收益。

hard gating 更极端，只选 1 个专家。这对推理延迟很友好，但训练时离散选择不可导，通常要靠 straight-through estimator，意思是“前向按硬选择走，反向用近似梯度顶上”。这类方法实现复杂，数值稳定性也更敏感。

对低资源模型，一个常见折中是 soft-to-sparse。也就是先让所有专家都参与一点，等 router 学到初步分工，再逐渐切换到 Top-K。白话解释就是：先让每个专家都试着做一点，再慢慢只保留最擅长的专家。

所以适用边界可以概括为：
- 如果模型不大，dense routing 往往更省心。
- 如果追求极致吞吐，Top-K 或 hard gating 更值得投入。
- 如果训练资源紧张、调参预算有限，先用 soft routing 建立基线通常更稳。

---

## 参考资料

- Next Electronics: 动态路由与稀疏 MoE 介绍，适合先建立 softmax + Top-K 的整体直觉。https://next.gr/ai/deep-learning-theory/sparse-mixture-of-experts-routing?utm_source=openai
- Next Electronics: Dynamic Token Routing in MoE Transformers，补充 token 级路由在 Transformer 里的位置。https://www.next.gr/ai/object-detection/dynamic-token-routing-in-moe-transformers?utm_source=openai
- Neptune.ai: Mixture of Experts in LLMs，偏工程视角，讨论负载均衡、容量和温度控制。https://neptune.ai/blog/mixture-of-experts-llms?utm_source=openai
- Michael Brenndoerfer: MoE Gating Networks and Router Architecture Design，适合理解 router 的公式化表达与数值例子。https://mbrenndoerfer.com/writing/moe-gating-networks-router-architecture-design?utm_source=openai
- Emergent Mind: Sparse Top-K Mixture-of-Experts，聚焦 sparse top-k 的梯度、塌缩和训练问题。https://www.emergentmind.com/topics/sparse-top-k-mixture-of-experts-moe?utm_source=openai

建议阅读路径：
1. 先看 Next Electronics，建立“先打分、再 Top-K、再加权”的主流程。
2. 再看 Neptune.ai，把容量、负载均衡和工程实现连起来。
3. 最后看 Emergent Mind 与 router 设计分析，理解为什么 Top-K 会带来梯度稀疏和专家塌缩。
