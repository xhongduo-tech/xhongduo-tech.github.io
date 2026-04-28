## 核心结论

`n` 步回报（n-step return，意思是“先往前看 `n` 步再构造学习目标”）的本质，是把“真实观测到的奖励”和“模型当前的估计值”拼接起来：前 `n` 步用真实奖励，后面更远的未来只做一次 bootstrap。bootstrap 的白话解释是“先用自己当前的估计补上还没真正看到的未来”。

它连接了两端：

- `n = 1` 时，就是 `1` 步 TD，也就是 TD(0)
- `n` 足够大并一直延伸到回合结束时，就接近 Monte Carlo

最常用的状态值形式是：

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n})
$$

这里 $\gamma$ 是折扣因子，白话解释是“未来奖励值多少钱”；越接近 1，说明越重视长期收益。

一句话对比三者：

- `1` 步 TD：只看下一步就更新，目标短，更新快
- Monte Carlo：等整局结束再更新，无需估计尾部
- `n` 步回报：先看几步真实结果，再用一次估计补尾巴

核心不是“谁绝对更好”，而是“偏差和方差怎么平衡”。偏差的白话解释是“系统性偏离真值”；方差的白话解释是“同一方法在不同样本上波动有多大”。

| 方法 | 目标长度 | 偏差 | 方差 | 更新速度 | 典型场景 |
|---|---:|---|---|---|---|
| `1` 步 TD | 1 步 | 通常更大 | 通常更小 | 快 | 奖励密集、任务较短 |
| `n` 步回报 | 可调 | 中等 | 中等 | 中等 | 稀疏奖励、需要更快传播 |
| Monte Carlo | 到回合结束 | 无偏或更接近无偏 | 很大 | 慢 | 回合短、可完整采样 |

要记住的不是某个固定公式，而是一条连续光谱：`1` 步 TD 和 Monte Carlo 不是两个孤立算法，`n` 步回报是两者之间的中间层。

---

## 问题定义与边界

它解决的问题很具体：值函数学习里，目标到底应该看多远。

如果目标只看一步，那么更新很早就能做，但大量依赖当前值函数自己的判断；如果目标一直等到整局结束，得到的是更完整的真实回报，但训练会更慢，噪声也更大。`n` 步回报就是把“看多远”做成一个可调参数。

边界最重要：

- `n = 1`：退化成 TD(0)
- `1 < n < T-t`：既有真实奖励，又有一次 bootstrap
- `n \ge T-t`：如果从时刻 $t$ 到回合结束总共不足 `n` 步，就不再 bootstrap，直接变成截断的 Monte Carlo 回报

Q 值版本也是同样思路：

$$
G_t^{(n,Q)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n \max_a Q(s_{t+n},a)
$$

这里 $Q(s,a)$ 的白话解释是“在状态 `s` 做动作 `a` 之后，未来总收益的估计”。

为什么需要这个长度可调的机制？看一个玩具例子。

假设第 10 步才出现唯一一次奖励 `+1`，前 9 步奖励全是 0。  
如果用 `1` 步 TD，第 9 步先学到“后面不错”，然后第 8 步再慢慢通过第 9 步传回去。奖励要一层层倒着传播。  
如果用 `5` 步回报，那么第 5 步附近的状态可以更快接收到更远处奖励的信息，传播速度明显更快。

这就是多步学习的第一性原理：不是奖励变多了，而是信用分配路径缩短了。信用分配的白话解释是“到底该把回报记到前面哪些动作和状态头上”。

| 情况 | 目标构成 | 是否 bootstrap | 行为特征 |
|---|---|---|---|
| `n=1` | `r_{t+1} + \gamma V(s_{t+1})` | 是 | 最短目标，最依赖当前估计 |
| 中等 `n` | 前几步真实奖励 + 一次尾部估计 | 是 | 传播更快，波动也更大 |
| 很大 `n` | 大量真实奖励 + 很晚才估计 | 可能是 | 更接近 Monte Carlo |
| 回合已结束 | 只累积实际奖励 | 否 | 完全不依赖尾部估计 |

这里有个常见误解需要直接澄清：  
不是“`1` 步 TD 偏差小方差大，Monte Carlo 无偏方差极大”，而是相反的典型结论更常见：`1` 步 TD 往往偏差更大、方差更小；Monte Carlo 更接近无偏，但方差更大。多步方法的价值，正是在这两端之间找更合适的平衡点。

---

## 核心机制与推导

先从状态值版本开始。目标是估计：

$$
V^\pi(s_t)=\mathbb{E}_\pi\left[\sum_{i=0}^{\infty}\gamma^i r_{t+i+1}\right]
$$

TD 方法不直接等整段和式都采完，而是拆成“短前缀 + 估计尾巴”。

`n` 步回报写成：

$$
G_t^{(n)}=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^n V(s_{t+n})
$$

它的含义可以按两段理解：

1. 前 `n` 步奖励是真实采样值
2. 第 `n` 步之后的未来，不再继续展开，而是交给当前的 `V` 估计

所以它不是纯采样，也不是纯 bootstrap，而是混合目标。

看一个最小数值玩具例子。设：

- $\gamma=0.9$
- $r_{t+1}=1$
- $r_{t+2}=0$
- $r_{t+3}=2$
- $V(s_{t+3})=5$

则

$$
G_t^{(3)} = 1 + 0.9\cdot 0 + 0.9^2\cdot 2 + 0.9^3\cdot 5
$$

$$
=1+0+1.62+3.645=6.265
$$

这说明一件事：前三步已经“真实结算”，只有第 3 步之后的更远未来，才由估计值补上。

如果做控制而不是状态值预测，常见版本会替换成：

$$
G_t^{(n,Q)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n \max_a Q(s_{t+n},a)
$$

这就是 `n` 步 Q-learning 的核心目标形式。`max_a` 的意思是“假设之后都选当前看来最好的动作”。

再往前一步，就会遇到 `\lambda-return`。它不是固定用某一个 `n`，而是把所有长度的 `n` 步回报做几何加权平均：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

其中 $\lambda \in [0,1]$。  
白话上看：

- $\lambda=0$：只保留 `1` 步 TD
- $\lambda$ 越大：越重视更长的回报
- $\lambda \to 1$：越来越接近 Monte Carlo 风格

这可以理解为：不再问“到底选几步”，而是“把不同步长按权重混起来”。

接着区分 forward-view 和 backward-view。

forward-view 的白话解释是“先把目标写出来，再根据完整目标更新”；  
backward-view 的白话解释是“每走一步，就把这一步的 TD 误差信号往前分摊给最近访问过的状态”。

两者关系可以概括成这条链：

`1步 TD -> n步回报 -> λ-return -> eligibility traces`

eligibility trace 的白话解释是“最近访问过的状态保留一个还没完全消失的记忆权重”。访问得越近、越频繁，这个权重通常越大。

经典结论是：

- 离线 TD(λ) 下，forward-view 的 `\lambda-return` 与 backward-view 的 eligibility traces 严格等价
- 在线情形下，经典 TD(λ) 只近似这个等价
- `true online TD(λ)` 才把在线等价关系处理得更精确

真实工程例子可以看 DQN 到 Rainbow 这一条线。Atari 任务里奖励常常稀疏，单步 TD 传播慢，于是 multi-step target 成为重要组件。它不是因为公式更“高级”，而是因为在长时延奖励问题上，固定 `n` 的多步回报常常能显著加快有效信号传播。

---

## 代码实现

工程实现的重点不在公式本身，而在样本切片和边界处理。

先给一个最小可运行 Python 实现，演示如何计算 `n` 步回报。这里用状态值尾部做 bootstrap，并显式处理终止状态。

```python
from math import isclose

def n_step_return(rewards, gamma, n, bootstrap_value=0.0, terminated=False):
    """
    rewards: 从 t+1 开始的奖励列表，例如 [r_{t+1}, r_{t+2}, ...]
    n: 希望最多看多少步
    bootstrap_value: V(s_{t+n}) 或 max_a Q(s_{t+n}, a)
    terminated: 第 n 步时是否已到终止状态
    """
    G = 0.0
    steps = min(n, len(rewards))
    for k in range(steps):
        G += (gamma ** k) * rewards[k]

    # 只有没终止且确实凑够 n 步时，才接 bootstrap
    if (not terminated) and steps == n:
        G += (gamma ** n) * bootstrap_value
    return G

# 玩具例子
g = n_step_return(
    rewards=[1, 0, 2],
    gamma=0.9,
    n=3,
    bootstrap_value=5.0,
    terminated=False,
)
assert isclose(g, 6.265, rel_tol=1e-9)

# 如果回合在第三步前就结束，不应继续 bootstrap
g_terminal = n_step_return(
    rewards=[1, 0, 2],
    gamma=0.9,
    n=5,
    bootstrap_value=999.0,
    terminated=True,
)
assert isclose(g_terminal, 1 + 0.9 * 0 + 0.9**2 * 2, rel_tol=1e-9)
```

训练时更常见的是从轨迹或 replay buffer 中构造 `n` 步样本。伪代码可以写成：

```python
def build_n_step_target(trajectory, t, n, gamma, bootstrap_fn):
    G = 0.0
    for k in range(n):
        idx = t + k
        if idx >= len(trajectory):
            break

        _, _, reward, next_state, done = trajectory[idx]
        G += (gamma ** k) * reward

        if done:
            return G

    # 走到这里说明前 n 步都没终止
    state_n = trajectory[t + n - 1][3]
    G += (gamma ** n) * bootstrap_fn(state_n)
    return G
```

如果是 DQN 风格的真实工程实现，还要注意 replay buffer 中常常不直接存单步样本，而是存已经聚合好的 `n` 步转移：

- 起点状态 `s_t`
- 起点动作 `a_t`
- 聚合后的 `n` 步折扣奖励
- 第 `n` 步后的状态 `s_{t+n}`
- 中间是否提前终止

这样做的好处是训练时目标构造更快，但代价是 buffer 写入时逻辑更复杂。

一个 Atari/Rainbow 风格的真实工程例子是：

1. 环境连续产生单步转移
2. 用一个长度为 `n` 的小队列缓存最近转移
3. 队列凑满后，把前 `n` 步奖励折叠成一个样本写入 replay buffer
4. 训练时目标写成  
   $$
   y=r_t^{(n)}+\gamma^n(1-d_t^{(n)})\max_a Q_{\text{target}}(s_{t+n},a)
   $$
5. 其中 $d_t^{(n)}$ 表示这 `n` 步内是否已经终止

这里 `(1-d)` 的作用很关键：终止后不能再 bootstrap，否则会把不存在的未来价值加进去。

---

## 工程权衡与常见坑

`n` 的效果可以直接理解成三件事同时变化：

- 奖励传播距离变长
- 目标噪声变大
- 更新时延增加

所以 `n` 越大不是越先进，只是更偏向“远视”。对稀疏奖励任务，中等 `n` 常常有效；对短回合、密集奖励任务，单步 TD 往往已经足够稳定。

最常见的坑不是理论问题，而是实现细节。

| 错误类型 | 表现 | 原因 | 修正方法 |
|---|---|---|---|
| 奖励下标错一位 | 学习到错误时序关系 | 把 $r_{t+k+1}$ 写成 $r_{t+k}$ | 统一定义“第一个奖励就是 `r_{t+1}`” |
| 终止后仍 bootstrap | 回报偏大，值函数漂移 | 忘记检查 `done` | 终止后直接截断，不再接尾值 |
| 把 `n` 步回报和 `λ-return` 混淆 | 调参逻辑混乱 | 一个是固定长度，一个是加权混合 | 先区分“单一 `n`”和“所有 `n` 的组合” |
| 离策略直接套多步 | 训练不稳定甚至发散 | 行为策略和目标策略不一致 | 需要重要性采样或专门校正方法 |
| `n` 取过大 | 波动明显、收敛慢 | 方差和更新延迟上升 | 从中等 `n` 开始扫参 |
| replay 样本对齐错误 | loss 异常、回报无意义 | `s_t` 与 `s_{t+n}` 配错 | 写单元测试检查时间对齐 |

“下标错一位”是最隐蔽也最常见的问题。  
例如你本来要算：

$$
G_t^{(2)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})
$$

结果写成：

$$
\tilde{G}_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})
$$

这会把整个目标向前错开一拍。表面上代码还能跑，loss 也可能下降，但语义已经错了：状态 `s_t` 被拿去对齐了不属于它的奖励。

还有一个工程上经常被忽略的点：on-policy 和 off-policy 要分清。  
on-policy 的白话解释是“用什么策略采样，就评估什么策略”；  
off-policy 的白话解释是“采样时用一个策略，学习目标却是另一个策略”。

多步方法在 off-policy 下更敏感，因为你累积的是一串由行为策略生成的奖励和状态转移，步数越长，策略不一致带来的误差越可能放大。DQN 类方法之所以还能用 `n` 步，很大程度上依赖目标网络、经验回放以及特定目标定义共同稳定训练。

经验规则可以很直接：

- 稀疏奖励任务：先试中等 `n`，如 3 或 5
- 密集奖励且训练不稳：先退回 `1` 步
- 如果你发现值函数波动显著增大，不要只怀疑学习率，也要怀疑 `n` 太长

---

## 替代方案与适用边界

不是所有任务都需要多步回报。

如果任务很短、奖励很快出现、目标又是稳定第一，那么 `1` 步 TD 的简单性和低方差常常更有优势。相反，如果任务长、奖励稀疏、信用分配困难，那么固定 `n` 的多步方法通常是很自然的第一选择。

如果还想更细粒度地混合不同时间尺度，可以考虑 `\lambda-return` 或 TD(λ)。如果你在意在线更新时 forward-view 与 backward-view 的精确一致性，可以进一步看 `true online TD(λ)`。

| 场景 | 推荐方法 | 原因 | 风险 |
|---|---|---|---|
| 短任务、奖励密集 | `1` 步 TD | 稳定、实现简单 | 奖励传播不远 |
| 长任务、稀疏奖励 | 中等 `n` 步回报 | 更快传播远处奖励 | 方差上升 |
| 想混合多种步长 | `λ-return` / TD(λ) | 不必死选单个 `n` | 实现与调参更复杂 |
| 在线连续学习 | true online TD(λ) | 更精确的在线等价 | 理解门槛更高 |
| 强离策略控制 | 校正后的多步方法 | 兼顾多步收益与策略不一致 | 实现复杂、易不稳 |

再给一个方法层面的对比：

| 方法 | 核心思想 | 优势 | 适用边界 |
|---|---|---|---|
| `n` 步回报 | 固定看 `n` 步 | 概念清晰、好实现 | 需要手工选 `n` |
| `λ-return` | 对所有 `n` 做几何加权 | 时间尺度更平滑 | 通常先从预测任务理解 |
| TD(λ) | backward-view 在线实现 | 在线分配信用 | trace 实现细节多 |
| true online TD(λ) | 修正在线近似误差 | 理论更严谨 | 对新手不够直接 |

如果只按场景做最务实的选择：

- “短任务、奖励立刻出现” -> `1` 步 TD 往往已经够用
- “长任务、奖励拖很久才来” -> 中等 `n` 更常见
- “不想把时间尺度锁死在一个 `n`” -> 考虑 `λ-return`
- “在线学习且很在意理论一致性” -> 再看 true online TD(λ)

如果只想记住一个结论，就记住：`n` 步回报不是一个孤立技巧，它是把 TD 和 Monte Carlo 连成一条连续光谱的方法。

---

## 参考资料

1. [Sutton & Barto, Reinforcement Learning: An Introduction, Chapter 7 n-step Bootstrapping](https://incompleteideas.net/book/bookdraft2018mar21.pdf)
2. [van Seijen & Sutton, True Online TD(λ)](https://proceedings.mlr.press/v32/seijen14.html)
3. [Hessel et al., Rainbow: Combining Improvements in Deep Reinforcement Learning](https://aaai.org/papers/11796-rainbow-combining-improvements-in-deep-reinforcement-learning/)
4. [Hernandez-Garcia & Sutton, Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target](https://incompleteideas.net/publications.html)
