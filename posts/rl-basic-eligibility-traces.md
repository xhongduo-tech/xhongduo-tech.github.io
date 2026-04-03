## 核心结论

资格迹（eligibility trace）可以理解为“一条会随时间衰减的记忆线索”：某个状态或某组特征刚刚被访问过，它就更“有资格”接收后续 TD 误差；访问得越早，这个资格就越弱。它解决的是信用分配问题，也就是“当前奖励到底应该归功于哪些过去状态”。

单步 TD(0) 每次只更新当前状态，更新快，但回溯范围只有 1 步。Monte Carlo 会等整条轨迹结束后再更新，回溯完整，但延迟高、方差大。TD($\lambda$) 介于两者之间：它在每一步在线更新，同时把当前误差按衰减权重分给过去若干步访问过的状态或特征。

前向视角把目标写成 $\lambda$-return：

$$
G_t^{(\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

这里的 $G_t^{(n)}$ 是 $n$ 步回报，意思是“先看未来 $n$ 步奖励，再在第 $n$ 步处 bootstrap（自举）”。后向视角则不显式算所有 $n$ 步目标，而是维护一条迹向量 $e_t$，每一步用 TD 误差 $\delta_t$ 直接更新参数。在线性函数逼近下，这两种写法是等价的。

一个最直接的理解方式是比较“单步”和“多步”：

| 方法 | 更新时机 | 回溯跨度 | 偏差/方差特点 | 是否必须等回合结束 |
|---|---|---:|---|---|
| TD(0) | 每一步 | 1 步 | 偏差较高，方差较低 | 否 |
| TD($\lambda$) | 每一步 | 近似多步，受 $\lambda$ 控制 | 在偏差和方差之间折中 | 否 |
| Monte Carlo | 回合结束后 | 整段轨迹 | 偏差低，方差高 | 是 |

玩具例子：在一条长度为 3 的链上，状态依次是 $A \to B \to C \to$ 奖励。TD(0) 在走到 $B$ 时只能更新 $B$；TD($\lambda$) 会把当前误差一部分分给 $B$，另一部分按衰减权重分给之前的 $A$。这就是“误差沿时间反向传播”，但它不是深度学习里的反向传播，而是通过资格迹做时序上的反向分配。

---

## 问题定义与边界

资格迹要解决的核心问题是：单步 TD 更新太“短视”。它知道下一步的预测差了多少，却不能立刻把这个误差信号传给更早访问的状态。对长期依赖任务，这会导致学习很慢。

更形式化地说，若我们估计状态价值 $v(s)$，单步 TD 的误差为：

$$
\delta_t = r_{t+1} + \gamma v(s_{t+1}) - v(s_t)
$$

其中 $\gamma$ 是折扣因子，白话上就是“未来奖励还值多少钱”。TD(0) 只用这个 $\delta_t$ 更新 $s_t$ 自己，不能直接更新 $s_{t-1}, s_{t-2}$。资格迹引入一个额外记忆量：

$$
e_t = \gamma \lambda e_{t-1} + \phi(s_t)
$$

$\phi(s_t)$ 是状态特征，白话上就是“把状态变成可计算向量的表示”。这个递推式说明两件事：

1. 过去的访问记录会按 $\gamma\lambda$ 衰减。
2. 当前访问的状态会给自己的迹增加一份新权重。

因此，资格迹的边界也很清楚。它不是免费提升性能，而是用额外状态换更快的信用分配。

| $\lambda$ 取值 | 回溯长度 | 偏差/方差 | 内存与实现成本 | 典型含义 |
|---|---|---|---|---|
| $\lambda=0$ | 1 步 | 偏差高、方差低 | 最低 | 退化为 TD(0) |
| $0<\lambda<1$ | 有限但可长 | 折中 | 中等到较高 | 常用工作区间 |
| $\lambda=1$ | 接近整条轨迹 | 偏差低、方差高 | 较高 | 接近 Monte Carlo |

真实工程例子：机器人巡航任务通常是持续任务，没有天然的“这一局结束”。如果你坚持用 Monte Carlo，就要等很久才能把奖励反馈到早期动作；如果只用 TD(0)，又只能做一步信用分配，收敛会慢。TD($\lambda$) 的价值就在这里：每走一步都更新，而且当前误差会沿着迹线回传给最近访问过的状态。

但它也有边界。第一，必须多维护一个 trace 向量，表格法是每个状态一个值，函数逼近则往往是每个特征或每个参数一个值。第二，$\lambda$ 不是越大越好，过大可能把噪声也回传得太远。第三，在非线性函数逼近尤其是大规模神经网络里，资格迹的稳定实现明显比 TD(0) 复杂。

---

## 核心机制与推导

后向视角最适合工程实现。它不去显式展开所有多步目标，而是在每一步维护迹，再把单步 TD 误差乘到迹上：

$$
e_t=\gamma\lambda e_{t-1}+\phi(s_t)
$$

$$
\delta_t=r_{t+1}+\gamma v(s_{t+1})-v(s_t)
$$

$$
\theta_{t+1}=\theta_t+\alpha\delta_t e_t
$$

这里 $\theta$ 是参数，白话上就是“价值函数内部需要被训练的数字”；$\alpha$ 是学习率，表示“每次改多大”。

为什么这能把误差分给过去？因为 $e_t$ 本身就保存了“最近访问过哪些状态/特征，以及它们离现在有多远”。如果一个特征刚出现过，它在 $e_t$ 里的值就大；如果已经过去很多步，它会被 $\gamma\lambda$ 连续衰减。

看一个最小数值例子。设：

- $\gamma = 0.9$
- $\lambda = 0.8$
- 上一步迹 $e_{t-1}=0.2$
- 当前特征 $\phi(s_t)=1$
- 当前 TD 误差 $\delta_t=0.5$
- 学习率 $\alpha=0.1$

则：

$$
e_t = 0.9 \times 0.8 \times 0.2 + 1 = 1.144
$$

$$
\Delta \theta = \alpha \delta_t e_t = 0.1 \times 0.5 \times 1.144 = 0.0572
$$

这说明当前状态贡献最大，但上一步的痕迹还保留在里面。最近状态得分最高，越早的访问权重越小。

再把递推过程写成表格：

| 时刻 | 当前特征贡献 | 衰减项 $\gamma\lambda e_{t-1}$ | 新迹 $e_t$ | 若 $\delta_t=0.5,\alpha=0.1$ 的参数增量 |
|---|---:|---:|---:|---:|
| $t-1$ | - | - | 0.2 | - |
| $t$ | 1.0 | 0.144 | 1.144 | 0.0572 |

前向视角则从目标函数角度解释同一件事。它把所有 $n$ 步回报做加权平均，权重按 $\lambda^{n-1}$ 递减。意思是：不仅看 1 步，也看 2 步、3 步，但步数越长，权重越小。前向写法直观，便于理解“到底在逼近什么目标”；后向写法适合在线实现，因为你不必等未来很多步都发生完。

两种视角在线性函数逼近下等价，原因是：前向写法中的所有多步目标，经过代数展开后，可以重写成每一步 TD 误差乘以一个随时间衰减的系数和；而后向视角中的迹，恰好就是这个系数和的在线递推形式。直观上，前向视角是在“先看未来，再算今天怎么改”，后向视角是在“今天先记账，未来误差到了再按账本分配”。

---

## 代码实现

最常见的是线性函数逼近版 TD($\lambda$)。它的主循环很短，但正确顺序很重要：先算 TD 误差，再更新迹，再按迹更新参数。

```python
from typing import List, Tuple

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def td_lambda(
    transitions: List[Tuple[List[float], float, List[float]]],
    theta: List[float],
    alpha: float = 0.1,
    gamma: float = 0.9,
    lam: float = 0.8,
) -> List[float]:
    """
    transitions: [(phi_t, reward_{t+1}, phi_{t+1}), ...]
    线性价值函数 v(s)=theta·phi(s)
    """
    e = [0.0 for _ in theta]

    for phi_t, r_tp1, phi_tp1 in transitions:
        v_t = dot(theta, phi_t)
        v_tp1 = dot(theta, phi_tp1)
        delta = r_tp1 + gamma * v_tp1 - v_t

        # 累加迹 accumulating trace
        e = [gamma * lam * ei + xi for ei, xi in zip(e, phi_t)]

        # 参数更新
        theta = [w + alpha * delta * ei for w, ei in zip(theta, e)]

    return theta

# 玩具序列：s0 -> s1 -> terminal
theta0 = [0.0, 0.0]
transitions = [
    ([1.0, 0.0], 0.0, [0.0, 1.0]),
    ([0.0, 1.0], 1.0, [0.0, 0.0]),
]
theta1 = td_lambda(transitions, theta0, alpha=0.1, gamma=1.0, lam=0.8)

assert len(theta1) == 2
assert theta1[1] > 0.0   # 当前状态直接收到奖励误差
assert theta1[0] > 0.0   # 前一状态通过资格迹也被更新
```

这段代码可以直接运行。最后两个 `assert` 验证了资格迹的关键现象：奖励在第二步出现，但第一步状态也得到了正向更新。

如果是表格法，可以把 `phi(s_t)` 换成 one-hot 状态访问；如果是稀疏特征，可以只维护非零特征对应的 trace，减少计算。主循环的骨架通常是：

```python
for each step:
    delta = r + gamma * v(next_state) - v(state)
    e = gamma * lambda * e + phi(state)
    theta += alpha * delta * e
```

常见的两种 trace 策略：

| 策略 | 更新规则 | 直观含义 | 常见场景 |
|---|---|---|---|
| 累加迹 accumulating trace | 访问到就加上特征值 | 多次访问会叠加资格 | 线性特征、理论推导常见 |
| 替代迹 replacing trace | 某维访问后设为 1 或取更大值 | 避免同一维被反复累加过大 | 二值特征、表格法常见 |

真实工程例子：在一个持续运行的推荐系统 bandit-like 强化学习模块里，用户点击反馈不是立刻归因到唯一一次曝光，而是与最近一串曝光和排序状态都有关系。若只做 TD(0)，系统只能更新最后一个状态；若维护轻量级资格迹，就能把点击误差按衰减分配给最近几次相关特征，提高学习速度，同时不必等到整段会话结束。

---

## 工程权衡与常见坑

资格迹的主要收益是更快的信用分配，主要代价是更多状态与更高实现复杂度。表格法时代价还可控；进入高维稀疏特征或神经网络后，trace 的维护可能成为工程负担。

一个核心量是 $\gamma\lambda$。它决定了迹衰减有多慢。若 $\gamma=0.99,\lambda=0.9$，则单步衰减因子是：

$$
\gamma\lambda = 0.891
$$

这意味着迹不会很快消失。经过 $k$ 步后，早先贡献大约按 $(0.891)^k$ 缩小。比如 $k=10$ 时仍有约 $0.315$，说明十步前的访问还保留相当影响。对长轨迹任务，这可能有用；对噪声大的任务，也可能把错误信用传太远。

| 策略 | 内存开销 | 计算开销 | 优点 | 风险 |
|---|---|---|---|---|
| TD(0) | 低 | 低 | 实现最简单、稳定 | 信用分配短 |
| TD($\lambda$) 累加迹 | 中到高 | 中到高 | 多步信息充分 | 某些特征会累加过大 |
| TD($\lambda$) 替代迹 | 中到高 | 中到高 | 更稳，避免重复爆涨 | 理论形式没累加迹统一 |
| Monte Carlo | 低到中 | 低到中 | 目标直接 | 延迟大、方差高 |

常见坑有五类。

第一，忘记在 episode 结束时清零 trace。终止状态之后如果还沿用旧迹，下一条轨迹会错误继承上一条的信用记录。

第二，更新顺序写错。很多初学者会先衰减 trace 再更新参数，或者先更新参数再算 $\delta_t$。严格来说，要保持与你采用的算法定义一致，否则结果会偏。

第三，$\lambda$ 调太大。比如机器人巡航任务里，如果 $\lambda \approx 1$，早期动作可能长期带着较大资格，局部噪声会被传得很远；如果 $\lambda=0.1$，又基本退回单步 TD。工程上常把 $0.6 \sim 0.9$ 当作候选区间，再结合任务时长调参，例如先试 $\lambda=0.7$ 作为折中。

第四，特征尺度不一致。因为参数更新是 $\alpha\delta_t e_t$，而 $e_t$ 又由特征递推得到，所以特征值很大时，迹也会放大，导致更新爆炸。需要做特征归一化或更小的学习率。

第五，在非线性网络中直接套用传统资格迹。理论上最清楚的等价关系建立在线性函数逼近上；神经网络场景下也可定义对参数的 trace，但内存和稳定性都更难处理，很多工程系统最终转向 n-step return、GAE 或 replay-based 方法，而不是完整参数级 eligibility trace。

---

## 替代方案与适用边界

资格迹不是唯一的多步信用分配方法。它最大的特点是“在线、连续、多步、无需等回合结束”，但代价是要维护 trace。若资源紧张或系统结构不适合维护 trace，可以用替代方案。

$n$-step TD 是最直接的替代。它固定看未来 $n$ 步：

$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n v(s_{t+n})
$$

它不像 $\lambda$-return 那样混合所有步长，而是只选一个固定窗口。Monte Carlo 则把窗口拉到回合末尾，不做 bootstrap。可以把三者理解成同一条光谱上的不同点。

| 方法 | 更新频率 | 偏差/方差 | 实现复杂度 | 适用边界 |
|---|---|---|---|---|
| $n$-step TD | 较高 | 比 TD(0) 低偏差、比 MC 低方差 | 中等 | 固定回溯长度可接受 |
| $\lambda$-return / TD($\lambda$) | 高 | 自适应折中 | 较高 | 想在线近似多步回报 |
| Monte Carlo | 低 | 偏差低、方差高 | 低 | 回合短且终止明确 |

玩具例子：在简单网格世界里，状态数不大、特征共享简单，用 TD($\lambda$) 往往很合适。代理每走一步都能把误差传给最近一串格子，学到“接近终点的路线更有价值”会比 TD(0) 快。

真实工程例子：在 Atari 一类长轨迹任务中，很多实现更偏向 $n$-step return，而不是显式资格迹。原因不是资格迹思想无效，而是深度网络下为每个参数维护 trace 成本太高，实现和调试都更难。固定的 $n$-step 目标更容易与 minibatch、replay buffer、GPU 训练流程对齐。

因此，适用边界可以总结为：

1. 持续控制、在线更新、需要快速信用分配时，TD($\lambda$) 很有价值。
2. 轨迹极长、特征极高维或神经网络参数很多时，完整资格迹的工程收益会下降。
3. 若你只需要“比 TD(0) 多看几步”，$n$-step 往往更便宜。
4. 若回合天然很短且终止明确，Monte Carlo 可能已经足够。

---

## 参考资料

1. [EmergentMind: Eligibility Traces in RL](https://www.emergentmind.com/topics/eligibility-traces?utm_source=openai)  
用途：给出 $\lambda$-return、后向资格迹、线性函数逼近下前向/后向等价等核心公式。

2. [lcalem: Sutton & Barto 第12章 TD-Lambda](https://lcalem.github.io/blog/2019/02/25/sutton-chap12?utm_source=openai)  
用途：强调持续任务中的在线更新视角，适合理解为什么不能总等回合结束。

3. [Data Science Stack Exchange: Purpose of trace-decay parameter in eligibility traces](https://datascience.stackexchange.com/questions/70007/purpose-of-trace-decay-parameter-in-eligibility-traces?utm_source=openai)  
用途：帮助理解 $\lambda$ 控制“回溯强度”，以及 $\lambda=0$ 到 $\lambda=1$ 的含义。

4. [Oboe: Eligibility Traces and TD(lambda)](https://oboe.com/learn/advanced-monte-carlo-and-temporal-difference-methods-dpywpn/eligibility-traces-and-tdlambda-1?utm_source=openai)  
用途：面向初学者解释前向/后向视角，以及“最近状态权重更高”的直观图景。
