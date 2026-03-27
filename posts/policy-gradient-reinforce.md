## 核心结论

REINFORCE 是最基础的策略梯度方法。策略梯度的白话解释是：不先学“这个状态值多少钱”，而是直接学“这个状态下每个动作该给多大概率”。它直接优化策略参数 $\theta$，目标是最大化期望回报

$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} r_t\right]
$$

其中 $\tau$ 表示一条轨迹，也就是一次完整交互序列 $(s_0,a_0,r_0,\dots)$。

REINFORCE 的核心更新式是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
$$

这里 $G_t$ 是从时刻 $t$ 往后的折扣回报，白话解释是“这个动作之后实际拿到了多少总收益”。如果某个动作后续回报高，就增大它的概率；如果后续回报低，就减小它的概率。

它的优点是概念直、推导干净、对离散动作和连续动作都能用。它的缺点也很明确：方差大，样本效率低，而且必须等整条 episode 结束才能更新。

最常见的改进是加 baseline。baseline 的白话解释是“先减掉一个正常水平，再看这次到底高了还是低了”。常见写法是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot (G_t-b(s_t))\right]
$$

如果 $b(s_t)$ 只依赖状态，不依赖动作，那么这个估计仍然无偏，只是方差更小。工程上最常用的 baseline 是状态值函数 $V^\pi(s)$)。

| 项目 | 数学形式 | 直观含义 | 主要问题 |
|---|---|---|---|
| 目标函数 | $J(\theta)=\mathbb{E}[\sum_t r_t]$ | 让策略拿到更高总奖励 | 只能通过采样估计 |
| REINFORCE | $\nabla \log \pi(a_t|s_t)\cdot G_t$ | 好回报强化产生它的动作 | 方差大 |
| 带 baseline 的 REINFORCE | $\nabla \log \pi(a_t|s_t)\cdot (G_t-b(s_t))$ | 只强化“高于正常水平”的动作 | 需要额外估计 baseline |

---

## 问题定义与边界

REINFORCE 解决的问题是：环境不可微，奖励可能延迟出现，但策略 $\pi_\theta(a|s)$ 本身可微，这时能否直接对策略参数做梯度上升。答案是可以。

它有几个边界条件必须先说清楚。

第一，它是 on-policy 方法。on-policy 的白话解释是“只能用当前策略自己采出来的数据更新自己”。这意味着策略一变，旧数据就不再严格匹配当前分布，不能像很多离策略方法那样反复重用经验池。

第二，它是 Monte Carlo 方法。Monte Carlo 的白话解释是“用完整采样结果近似期望”。因此 REINFORCE 往往要等 episode 结束，拿到完整回报后，才能算每个时间步的 $G_t$。如果任务没有天然终点，通常需要人为截断。

第三，它不做 bootstrapping。bootstrapping 的白话解释是“用自己当前的估计去更新未来的估计”。REINFORCE 不这么做，它直接用真实采样到的完整回报，所以无偏，但噪声很大。

它的基本流程可以写成：

`当前策略采样一整条轨迹 -> 反向计算每个时刻的 G_t -> 计算 log π 的梯度项 -> 累加后更新参数 -> 用新策略再采样下一条轨迹`

这也解释了它的使用边界：

| 维度 | REINFORCE 的边界 |
|---|---|
| 数据来源 | 必须来自当前策略 |
| 更新时机 | 通常 episode 结束后 |
| 样本效率 | 较低，旧样本难复用 |
| 方差 | 高，轨迹越长越明显 |
| 适合任务 | 可模拟、可反复采样、奖励定义清晰 |

玩具例子最容易说明它在做什么。假设一个两步任务，$\gamma=0.9$，奖励序列为 $r_1=1,r_2=0$。那么

$$
G_1=r_1+\gamma r_2=1+0.9\times 0=1,\quad G_2=r_2=0
$$

如果在 $s_1$ 有两个动作 $A,B$，当前概率都为 $0.5$，实际采样到 $A$，并最终得到 $G_1=1$，那么更新方向会包含

$$
\nabla_\theta \log \pi_\theta(A|s_1)\cdot 1
$$

也就是提升动作 $A$ 的概率。第二步因为 $G_2=0$，这一时刻对更新没有贡献。这个例子把“好动作涨概率”变成了可计算的规则。

---

## 核心机制与推导

核心推导从目标函数开始。设一条轨迹的概率为

$$
p_\theta(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

轨迹回报为 $R(\tau)$，则

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)]
=\sum_\tau p_\theta(\tau)R(\tau)
$$

对参数求梯度：

$$
\nabla_\theta J(\theta)=\sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)
$$

用似然比技巧。似然比的白话解释是“把概率的梯度改写成概率乘以对数概率的梯度”：

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
$$

代入得

$$
\nabla_\theta J(\theta)=\sum_\tau p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)R(\tau)
=\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)R(\tau)]
$$

再展开 $\log p_\theta(\tau)$。环境转移概率 $p(s_{t+1}|s_t,a_t)$ 与策略参数无关，所以只有策略项留下：

$$
\log p_\theta(\tau)=\log p(s_0)+\sum_t \log \pi_\theta(a_t|s_t)+\sum_t \log p(s_{t+1}|s_t,a_t)
$$

因此

$$
\nabla_\theta \log p_\theta(\tau)=\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

于是

$$
\nabla_\theta J(\theta)=
\mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot R(\tau)\right]
$$

进一步可以把整条轨迹总回报 $R(\tau)$ 换成从时刻 $t$ 开始的回报 $G_t$，因为时刻 $t$ 之前的奖励和当前动作无关，不影响该动作的期望梯度：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

所以得到常用形式：

$$
\nabla_\theta J(\theta)=
\mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
$$

为什么减 baseline 不会引入偏差？因为对任意只依赖状态的 $b(s_t)$，

$$
\mathbb{E}_{a_t\sim \pi_\theta(\cdot|s_t)}
[\nabla_\theta \log \pi_\theta(a_t|s_t)\, b(s_t)]
=
b(s_t)\sum_{a_t}\pi_\theta(a_t|s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

而

$$
\pi_\theta(a|s)\nabla_\theta \log \pi_\theta(a|s)=\nabla_\theta \pi_\theta(a|s)
$$

所以

$$
b(s_t)\sum_{a_t}\nabla_\theta \pi_\theta(a_t|s_t)
=
b(s_t)\nabla_\theta \sum_{a_t}\pi_\theta(a_t|s_t)
=
b(s_t)\nabla_\theta 1
=0
$$

因此

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
=
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot (G_t-b(s_t))\right]
$$

这说明 baseline 改变的是方差，不是期望。

从直觉上看，如果某状态下正常回报是 5，这次采样得到 $G_t=8$，那么优势 $A_t=G_t-b(s_t)=3$，说明动作比平均好，应强化；如果这次只有 2，则 $A_t=-3$，说明动作比平均差，应抑制。这里的优势 advantage，就是“比基准线高多少”。

---

## 代码实现

下面给一个可运行的最小 Python 版本。它不依赖深度学习框架，只演示 REINFORCE 的核心更新逻辑。策略是单状态二动作 softmax，方便看清数学关系。

```python
import math

def softmax(theta):
    m = max(theta)
    exps = [math.exp(x - m) for x in theta]
    s = sum(exps)
    return [x / s for x in exps]

def discounted_returns(rewards, gamma):
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out

def grad_log_softmax(theta, action):
    p = softmax(theta)
    grad = [-x for x in p]
    grad[action] += 1.0
    return grad

def reinforce_update(theta, actions, rewards, gamma=0.9, lr=0.1, baseline=None):
    returns = discounted_returns(rewards, gamma)
    if baseline is None:
        baseline = [0.0] * len(returns)

    grad_sum = [0.0 for _ in theta]
    for a, G, b in zip(actions, returns, baseline):
        adv = G - b
        g = grad_log_softmax(theta, a)
        for i in range(len(theta)):
            grad_sum[i] += g[i] * adv

    new_theta = [t + lr * g for t, g in zip(theta, grad_sum)]
    return new_theta, returns, grad_sum

# 玩具例子：两步轨迹，第一步奖励 1，第二步奖励 0
theta = [0.0, 0.0]  # 两个动作初始概率都为 0.5
actions = [0, 1]    # 第一步选 A，第二步选 B
rewards = [1.0, 0.0]

new_theta, returns, grad_sum = reinforce_update(theta, actions, rewards, gamma=0.9, lr=0.1)

assert returns == [1.0, 0.0]
assert softmax(theta) == [0.5, 0.5]
assert new_theta[0] > new_theta[1]  # 动作 A 的参数应上升更多
assert abs(sum(grad_sum)) < 1e-9    # softmax 的梯度和为 0

print("returns =", returns)
print("old_probs =", softmax(theta))
print("new_probs =", softmax(new_theta))
```

这段代码体现了三个关键步骤：

1. 先从后往前算 $G_t$。
2. 再算 $\nabla \log \pi(a_t|s_t)$。
3. 最后用 $G_t-b(s_t)$ 加权梯度并做梯度上升。

如果换成神经网络版本，结构通常是：

`state -> policy network -> action distribution -> sample action -> collect rewards -> compute returns/advantages -> backward policy loss`

真实工程例子可以看公平分类。那类任务里，准确率和公平性指标往往不可微或难直接优化，于是可以把“分类决策”看成策略，把“准确率与公平性的组合目标”看成奖励，再用 REINFORCE 直接优化。论文中的做法不是裸用 $G_t$，而是增加一个 baseline 网络来降低梯度噪声，否则训练非常不稳定。这个例子说明：REINFORCE 不只用于游戏控制，也能用于非微分目标优化。

---

## 工程权衡与常见坑

REINFORCE 的主要问题不是“不会收敛”，而是“更新太吵”。噪声大的白话解释是：同样一个状态下，偶然碰到高奖励轨迹时，整条轨迹上的动作都会被一起放大，即使其中很多动作只是陪跑。

常见坑有五类。

第一，直接用原始 $G_t$，方差极大。轨迹长、奖励稀疏、动作空间大时尤其严重。解决办法通常是加状态 baseline，或者进一步用 advantage 标准化。

第二，把 baseline 设计成依赖动作。普通 baseline 若依赖动作，通常会破坏无偏性；只有满足特殊推导条件的 action-dependent baseline 才能安全使用。对初学者而言，先坚持 $b(s)$ 最稳妥。

第三，忘了这是 on-policy。旧策略采样的数据不能随便混进新策略更新，否则梯度方向会偏。很多训练发散不是公式错，而是数据分布错。

第四，episode 太长。因为每个动作都要等未来完整回报，时间跨度越长，信用分配越难。信用分配的白话解释是“最终奖励到底该记到前面哪个动作头上”。

第五，学习率过大。REINFORCE 更新本身噪声就大，如果再配高学习率，概率分布会迅速塌缩到单一动作，探索直接消失。

下面这张表能概括工程上的取舍：

| 方法 | 方差 | 样本效率 | 实现成本 | 适用情况 |
|---|---|---|---|---|
| 原始 REINFORCE | 高 | 低 | 低 | 教学、玩具任务、验证公式 |
| REINFORCE + baseline | 中 | 低到中 | 中 | 大多数可复现实验 |
| 更强方差缩减方法 | 低 | 中 | 高 | 长时序、高维动作、昂贵采样 |

一个实际判断标准是：如果你发现同一套超参下，不同随机种子结果差异巨大，平均回报曲线忽上忽下，通常不是网络不够大，而是方差控制不够。

---

## 替代方案与适用边界

REINFORCE 值得学，但不一定值得长期直接用在生产级强化学习任务里。

如果任务很简单，episode 短，环境可以便宜地反复采样，那么 vanilla REINFORCE 足够。它的优势是实现极简，推导透明，调试时容易定位“公式错”还是“代码错”。

如果任务较复杂，最常见替代方案是 Actor-Critic。critic 的白话解释是“专门估计状态价值或优势的辅助网络”。它用估计值替代纯 Monte Carlo 回报，虽然会引入一些偏差，但通常显著降低方差、提升样本效率。A2C、A3C、PPO 本质上都在这条路线上。

如果任务是高维连续控制，比如机械臂抓取、灵巧手操作、长时间稳定控制，那么纯 REINFORCE 往往太浪费样本。每条轨迹都很长，单次采样昂贵，回报延迟又强，这时更适合 Actor-Critic 或带更强控制变量的策略梯度方法。

可以用一个简单选择表判断：

| 场景 | 推荐方法 |
|---|---|
| 教学、推导、极小实验 | vanilla REINFORCE |
| 想保留无偏梯度但降低波动 | REINFORCE + state baseline |
| 长轨迹、昂贵采样、需要更快收敛 | Actor-Critic / PPO |
| 高维动作且方差特别大 | 更强的方差缩减策略梯度 |

所以 REINFORCE 的定位很清楚：它不是终点，而是理解策略梯度家族的起点。把它学明白，后面看 advantage、critic、GAE、PPO 时就不会只记结论，不懂来路。

---

## 参考资料

| 标题 | 年份 | 用途 | 链接 |
|---|---:|---|---|
| Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | 1992 | REINFORCE 原始论文 | DOI: 10.1007/BF00992696 |
| REINFORCE - Monte Carlo Policy Gradient | 2025 更新 | 复现策略梯度与 baseline 推导 | https://parasdahal.com/notes/reinforce%2B-%2Bmonte%2Bcarlo%2Bpolicy%2Bgradient |
| REINFORCE Policy Gradient Agent | 持续更新 | 工程实现与 on-policy/Monte Carlo 边界说明 | https://www.mathworks.com/help/reinforcement-learning/ug/reinforce-policy-gradient-agents.html |
| Fair classification via Monte Carlo policy gradient method | 2021 | 真实工程例子：非可微公平约束优化 | https://doi.org/10.1016/j.engappai.2021.104398 |
| Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines | 2018 | 进阶方差缩减与高维动作场景 | https://openai.com/index/variance-reduction-for-policy-gradient-with-action-dependent-factorized-baselines/ |
