## 核心结论

MoE，Mixture of Experts，白话讲就是“让一个路由器只挑少数几个专家网络来处理当前输入”。它的训练难点不在专家本身，而在路由。核心路由通常写成：

$$
G(x)=\operatorname{top\text{-}k}(\operatorname{softmax}(W_r x))
$$

其中 $W_r x$ 是路由 logits，意思是“每个专家的原始打分”；softmax 把打分变成概率；top-k 再把概率里最大的 $k$ 个留下，其余直接置零。问题就出在最后一步。top-k 是离散选择，白话讲就是“要么选中，要么不选中”，中间没有连续过渡，所以梯度会在大部分位置断掉。

这意味着：如果前向传播严格使用 hard top-k，反向传播就很难把“应该更偏向哪个专家”的信息传回路由器。工程上通常有三类补救方法：

| 方法 | 核心思路 | 偏差 | 方差 | 典型用途 |
| --- | --- | --- | --- | --- |
| STE | 前向硬选，反向假装它是连续的 | 有偏 | 低 | 默认工程实现 |
| Gumbel-Softmax | 用可导的连续采样逼近离散选择 | 可控 | 中等 | 需要更平滑训练时 |
| REINFORCE | 把路由看成策略采样，直接估计期望梯度 | 无偏 | 高 | 需要严格随机优化或带奖励信号时 |

结论可以压缩成一句话：MoE 的 top-k 让“真实梯度”不可直接用，所以训练不是在找完美梯度，而是在“偏差更小”和“方差更小”之间选一个能跑得动的估计器。

---

## 问题定义与边界

先把问题说窄。本文讨论的是“稀疏路由”场景，也就是一次只激活少量专家，例如 top-1 或 top-2。这里不讨论所有专家都参与加权求和的 dense MoE，因为 dense MoE 没有 top-k 这一步，自然也没有本文的核心断点。

假设有两个专家，softmax 输出为 $[0.6, 0.4]$，top-1 后得到 $[1, 0]$。如果把第一个概率从 $0.6$ 改成 $0.61$，第二个从 $0.4$ 改成 $0.39$，top-1 结果还是 $[1,0]$，前向结果完全没变。对白话理解来说，这就像“路由器心里更偏向专家 1 了一点点，但行为上没有任何变化”。只要行为没变，损失函数对 logits 的局部导数就几乎看不到这点变化。

这就是离散函数的阶梯性。除了切换边界附近，其余区域都像平台。平台上的梯度要么为零，要么不可稳定使用。对链式法则来说，前一层会收到“没信号”。

可以用一个对照表看得更清楚：

| 对象 | 数学形式 | 对输入变化的反应 | 梯度状态 |
| --- | --- | --- | --- |
| softmax 输出 | 连续概率向量 | 小扰动会带来小变化 | 连续、通常非零 |
| top-k 输出 | 稀疏 one-hot 或 k-hot | 只有越过阈值才变化 | 阶梯型、几乎处处为 0 |
| top-k 边界点 | 排名恰好切换 | 极小扰动导致离散翻转 | 不连续，难稳定训练 |

这件事为什么对 MoE 特别严重？因为专家参数只会从被选中的路径收到梯度。未被选中的专家不仅前向没参与，路由器也可能几乎收不到“下次试试它”的有效信号。久而久之，就会出现两个典型问题：

| 问题 | 白话解释 | 后果 |
| --- | --- | --- |
| 专家塌缩 | 总是少数专家被选中 | 负载不均衡，容量浪费 |
| 路由迟钝 | 未选专家几乎没有修正机会 | 收敛慢，容易陷入次优分配 |

所以本文的边界很明确：我们关心的不是“top-k 能不能算”，而是“top-k 后怎么给路由器构造一个可训练的梯度估计”。

---

## 核心机制与推导

先看玩具例子。两个专家，logits 为 $[2,1]$，$k=1$。

softmax 概率为：

$$
p=\operatorname{softmax}([2,1])\approx [0.73,0.27]
$$

hard top-1 结果是 $[1,0]$。前向里只走专家 1。问题是，真实反向若严格穿过 top-1，这一步几乎不给 logits 梯度。

### 1. STE：前向硬，反向软

STE，Straight-Through Estimator，白话讲就是“前向时认真做离散决策，反向时假装这一步是连续的”。写成近似就是：

$$
\frac{\partial G}{\partial W_r}\approx \frac{\partial \operatorname{softmax}(W_r x)}{\partial W_r}
$$

也就是说，真实前向还是 top-k，但反向直接借用 softmax 的导数。对上面的例子，虽然 top-1 输出是 $[1,0]$，但反向不会把梯度彻底截断，而是让梯度按 $[0.73,0.27]$ 的软分布流动。

优点是简单、低成本、兼容现有自动求导。缺点是有偏。所谓有偏，白话讲就是“你传回去的不是这条计算图真正的梯度，而是一个你希望它像梯度的替代品”。它能优化，但理论上并不等于原目标的真实导数。

### 2. Gumbel-Softmax：把离散采样改成连续近似

Gumbel-Softmax 的思路是：不要直接做 argmax 或 top-k，而是在 logits 上加随机噪声，再用温度 $\tau$ 控制软化程度。公式是：

$$
y_i=\frac{\exp((l_i+g_i)/\tau)}{\sum_j \exp((l_j+g_j)/\tau)}
$$

其中 $g_i$ 是 Gumbel 噪声，白话讲就是“专门配合离散采样的一种随机扰动”。当 $\tau$ 较大时，输出更平滑；当 $\tau \to 0$ 时，输出更接近 one-hot。

它的路径导数是：

$$
\frac{\partial y_i}{\partial l_j}=\frac{1}{\tau}(\delta_{ij}-y_j)y_i
$$

这里 $\delta_{ij}$ 是 Kronecker delta，白话讲就是“相同下标取 1，不同取 0”。这个式子说明两点：

1. 只要 $\tau$ 不是 0，导数就存在。
2. $\tau$ 越小，前面的 $1/\tau$ 越大，梯度波动可能更激烈。

还是用 $[2,1]$ 的玩具例子。若采到一组噪声后变成 $[2.2, 0.7]$，取 $\tau=0.5$，softmax 后可能得到近似 $[0.95, 0.05]$。它已经非常接近 one-hot，但仍然保留了可导路径。这就是“连续松弛”。所谓松弛，白话讲就是“先把硬约束放松成软版本，等训练稳定后再慢慢变硬”。

### 3. REINFORCE：把路由当策略

REINFORCE 不再尝试穿过离散操作，而是直接承认“这就是一次采样”，然后对采样分布求期望梯度：

$$
\nabla_\theta \mathbb{E}[f(z)] = \mathbb{E}[f(z)\nabla_\theta \log p_\theta(z)]
$$

这里 $f(z)$ 是样本对应的收益，白话讲就是“选了这个专家后，结果好不好”；$\log p_\theta(z)$ 是该动作的对数概率。其含义是：如果某个被采样动作带来更高收益，就提高它的概率；如果收益差，就降低它的概率。

在两专家例子里，若采到专家 1，奖励是 $r$，则单样本梯度近似为：

$$
r \cdot \nabla_\theta \log p_\theta(z=1)
$$

这类估计对原始期望目标是无偏的。无偏，白话讲就是“平均起来不歪”。但单次样本波动极大，所以方差高。工程上通常会引入 baseline，即基线值，改写成：

$$
(r-b)\nabla_\theta \log p_\theta(z)
$$

其中 $b$ 是平均奖励或可学习估计器。它不改变期望，但能降低波动。

### 三者的偏差与方差为什么不同

可以用一句话记忆：

| 方法 | 为什么有偏或高方差 |
| --- | --- |
| STE | 反向替换了真实梯度，所以有偏 |
| Gumbel-Softmax | 优化的是松弛目标，$\tau$ 不为 0 时有偏 |
| REINFORCE | 直接估计原目标梯度，无偏，但采样噪声带来高方差 |

因此，三者不是“谁更先进”的关系，而是“你愿意为更低偏差付出多少训练噪声成本”的关系。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，只依赖 `numpy`。它不训练真实大模型，只演示三种梯度估计器在两专家路由上的核心行为。

```python
import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exp_x = np.exp(logits)
    return exp_x / exp_x.sum()

def hard_topk(probs, k=1):
    out = np.zeros_like(probs)
    idx = np.argsort(probs)[-k:]
    out[idx] = 1.0
    return out

def ste_surrogate_grad(probs, upstream_grad):
    # STE 的核心：前向硬选择，反向直接复用 softmax 路径
    p = probs.reshape(-1, 1)
    jacobian = np.diagflat(probs) - p @ p.T
    return jacobian @ upstream_grad

def sample_gumbel(shape, rng):
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=shape)
    return -np.log(-np.log(u))

def gumbel_softmax_sample(logits, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = sample_gumbel(len(logits), rng)
    y = softmax((np.asarray(logits) + g) / tau)
    return y

def reinforce_grad(probs, action, reward, baseline=0.0):
    # 对 logits 的单样本梯度： (r-b) * (one_hot(action) - probs)
    one_hot = np.zeros_like(probs)
    one_hot[action] = 1.0
    return (reward - baseline) * (one_hot - probs)

# 玩具例子：两个专家，logits=[2,1]
logits = np.array([2.0, 1.0])
probs = softmax(logits)
hard = hard_topk(probs, k=1)

# 检查 softmax 数值
assert np.allclose(probs.sum(), 1.0)
assert probs[0] > probs[1]
assert np.allclose(np.round(probs, 2), np.array([0.73, 0.27]))

# STE: 给定一个上游梯度，能得到非零近似梯度
upstream = np.array([1.0, -1.0])
ste_grad = ste_surrogate_grad(probs, upstream)
assert ste_grad.shape == (2,)
assert not np.allclose(ste_grad, np.zeros(2))

# Gumbel-Softmax: tau=0.5 时输出接近 one-hot，但仍连续
y = gumbel_softmax_sample(logits, tau=0.5, seed=42)
assert np.allclose(y.sum(), 1.0)
assert np.all(y > 0.0)

# REINFORCE: 假设采样到专家0，奖励高于基线
rf_grad = reinforce_grad(probs, action=0, reward=1.2, baseline=0.7)
assert rf_grad[0] > 0
assert rf_grad[1] < 0

print("softmax:", np.round(probs, 4))
print("hard top-1:", hard)
print("STE surrogate grad:", np.round(ste_grad, 4))
print("gumbel-softmax sample:", np.round(y, 4))
print("REINFORCE grad:", np.round(rf_grad, 4))
```

如果把这个思路迁移到深度学习框架，常见写法如下：

```python
import torch
import torch.nn.functional as F

def route(logits, mode="ste", tau=1.0, reward=None, baseline=None):
    if mode == "ste":
        probs = F.softmax(logits, dim=-1)
        hard_idx = probs.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(-1, hard_idx, 1.0)
        # 前向用 hard，反向梯度走 probs
        return hard + probs - probs.detach()

    if mode == "gumbel":
        u = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
        g = -torch.log(-torch.log(u))
        y = F.softmax((logits + g) / tau, dim=-1)
        return y

    if mode == "reinforce":
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        adv = reward - baseline
        loss = -(adv.detach() * log_prob).mean()
        return action, loss

    raise ValueError("unknown mode")
```

这段伪实现有三个关键点：

| 位置 | 作用 |
| --- | --- |
| `tau` | 控制 Gumbel-Softmax 的软硬程度 |
| `reward` | 给 REINFORCE 提供样本收益 |
| `baseline` | 给 REINFORCE 降低方差 |

真实工程例子可以看大规模语言模型中的 top-1 或 top-2 路由。比如一个 batch 中每个 token 都要被分配到少量专家。如果全部采用纯 hard 路由，训练前期路由器很容易过早偏向少数专家，导致负载极不均衡。工程上常见做法是：主路径先用 STE 保持吞吐量，训练早期若需要更稳定的探索，可以在辅助实验或小规模验证里引入 Gumbel 温度退火，再配合负载均衡损失。

---

## 工程权衡与常见坑

这部分最重要，因为很多实现“数学上能写，系统里跑不稳”。

| 方法 | 常见坑 | 原因 | 常见规避手段 |
| --- | --- | --- | --- |
| STE | 路由跳跃、专家切换过于频繁 | 反向和前向不一致，近边界时很敏感 | logits 正则、z-loss、负载均衡损失、噪声平滑 |
| Gumbel-Softmax | 低温不稳定，高温又不够稀疏 | $\tau$ 同时影响偏差与方差 | warmup 后逐步 anneal，监控熵与负载分布 |
| REINFORCE | 收敛慢，样本效率差 | 单样本梯度噪声大 | baseline、优势函数、批量平均、控制变量 |

### 1. STE 的坑不是“梯度断了”，而是“梯度方向可能不对”

很多初学者觉得 STE 最大优点是“有梯度”。这不完整。真正风险是：它给的是替代梯度，不一定和真实目标一致。如果 logits 很接近边界，前向一会儿选专家 1，一会儿选专家 2，反向却一直按 softmax 平滑分布更新，就可能出现路由抖动。

常见补救：

1. 给 logits 加轻微噪声或平滑。
2. 配合 auxiliary load balancing loss，强制不同专家都被看到。
3. 在训练前期保持更高熵，避免过早 one-hot 化。

### 2. Gumbel-Softmax 的关键不在“加噪声”，而在“温度调度”

若固定 $\tau=5$，输出很软，梯度稳定，但学到的不是你最终想要的 hard 路由。若固定 $\tau=0.1$，输出接近 one-hot，但训练初期极不稳定。典型策略是：

| 训练阶段 | 温度策略 | 目标 |
| --- | --- | --- |
| 初期 | 较高 $\tau$ | 稳定梯度、鼓励探索 |
| 中期 | 缓慢下降 | 逐步逼近离散选择 |
| 后期 | 低但非零 $\tau$ 或切换硬路由 | 提高稀疏性和部署一致性 |

### 3. REINFORCE 的坑在系统规模上会被放大

在真实 MoE 里，一个 batch 可能有成千上万个 token 路由决策。若每个决策都用采样策略梯度，方差会非常大，且很难和吞吐量目标兼容。所以 REINFORCE 更适合作为补充，而不是默认主干。常见位置包括：

1. 路由里带显式任务奖励，而不是只看交叉熵损失。
2. 某些搜索式路由或层级决策，需要优化不可导指标。
3. 小规模研究实验，用来验证无偏估计是否值得额外成本。

---

## 替代方案与适用边界

如果目标不是“严格三选一”，还有一些替代思路。

### 1. 用 dense gradient 补足未选专家

一种工程改法是：即使前向只执行 top-k 专家，反向也给未选专家构造近似梯度。例如对未激活专家维护 EMA，Exponential Moving Average，白话讲就是“用历史平均输出当作一个稳定替身”。这样总输出可写成“选中专家真实输出 + 未选专家的默认向量近似”。它的作用不是提升前向精度，而是让

$$
\frac{\partial y}{\partial \pi_i}
$$

对所有专家都不为零，从而路由器参数能持续更新。这在大模型预训练里很有价值，因为未选专家长期没梯度，往往比单次估计偏差更致命。

### 2. 更高级的连续松弛

除了基本 Gumbel-Softmax，还有改进连续松弛、可逆高斯重参数化、Structured Softmax Trick 等变体。它们的共同点是：试图在更复杂的离散结构上保留更低偏差的路径导数。代价是实现复杂、超参数更多、系统兼容性更差。对零基础到初级工程师来说，默认不该优先上这些方案，除非你已经证明基础方法在目标任务上不够用。

### 3. 适用边界对照

| 方案 | 偏差-方差特征 | 适用场景 | 不适合的场景 |
| --- | --- | --- | --- |
| STE | 偏差较大、方差低 | 大规模训练主路径、吞吐量优先 | 需要严格无偏估计 |
| Gumbel-Softmax | 偏差可调、方差中等 | 需要可导采样、希望逐步逼近离散 | 对温度非常敏感且难调参的系统 |
| REINFORCE | 无偏、方差高 | 带奖励优化、不可导目标、研究验证 | 超大规模常规 MoE 主训练 |
| Dense gradient / EMA 补全 | 引入结构性近似，覆盖所有专家 | 预训练中缓解专家饥饿 | 资源极紧、实现必须最简 |
| 高级 relaxations | 可能更低偏差，但更复杂 | 高维结构化离散变量 | 工程团队经验不足时 |

所以实际选择通常不是“只用一个”。更合理的工程组合是：

1. 主干路由用 STE 保持效率。
2. 训练早期或实验分支用 Gumbel-Softmax 帮助探索。
3. 需要额外奖励或不可导指标时局部引入 REINFORCE。
4. 若专家长期饥饿，再考虑 dense gradient 补偿或 EMA 类方法。

---

## 参考资料

- Sparse Mixture-of-Experts Routing Gradient Challenges: https://www.next.gr/ai/deep-learning-theory/sparse-mixture-of-experts-routing
- Gumbel-Softmax relaxation: https://www.emergentmind.com/topics/gumbel-softmax-relaxation-9e7d074a-4229-4c58-b6d7-d5652f6827f5
- Gumbel-Softmax: Differentiable Discrete Sampling: https://www.emergentmind.com/topics/gumbel-softmax-relaxation
- Sparse Top-K Mixture-of-Experts: https://www.emergentmind.com/topics/sparse-top-k-mixture-of-experts-moe
- Policy gradient method: https://en.wikipedia.org/wiki/Policy_gradient_method
- Fabian Fuchs, Gumbel Softmax: https://fabianfuchsml.github.io/gumbel
