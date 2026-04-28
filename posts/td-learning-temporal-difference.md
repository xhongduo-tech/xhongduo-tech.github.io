## 核心结论

TD Learning，中文通常叫“时序差分学习”，本质是**用一步真实反馈修正当前价值估计**。这里的“价值”可以先理解成“一个状态未来大概还能拿到多少收益的分数”。

它介于 Monte Carlo 和 Dynamic Programming 之间。

| 方法 | 学习目标来自哪里 | 何时更新 | 是否依赖环境模型 | 主要特点 |
| --- | --- | --- | --- | --- |
| Monte Carlo | 整条轨迹的真实回报 | 回合结束后 | 否 | 无偏，但方差高 |
| TD | 一步奖励 + 下一状态估计值 | 每一步 | 否 | 有偏，但方差低、更新快 |
| Dynamic Programming | Bellman 方程全期望 | 可全局反复更新 | 是 | 需要已知转移模型 |

TD 的核心更新式是：

$$
V(s_t) \leftarrow V(s_t) + \alpha \Big(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\Big)
$$

其中：

- $\alpha$ 是学习率，白话解释是“这次修正听多少”。
- $\gamma$ 是折扣因子，白话解释是“未来收益现在值多少钱”。
- $V(s)$ 是状态价值，白话解释是“站在这个状态往后看，平均能有多大收益”。

这条式子里最关键的不是更新，而是目标：

$$
\text{TD target}=r_{t+1}+\gamma V(s_{t+1})
$$

它一半来自真实世界的一步奖励 $r_{t+1}$，一半来自模型当前对未来的估计 $V(s_{t+1})$。这就是 TD 的核心思想：**不等整局结束，走一步就学一步**。

TD(0) 是最基本的单步版本。再往上推广，会得到 $n$ 步 TD、TD($\lambda$)、资格迹。这一条线最终通向 Q-learning、SARSA、Actor-Critic 等更完整的强化学习算法。换句话说，TD 不是边角概念，而是现代价值学习的基础部件。

---

## 问题定义与边界

TD 学习主要解决的是**策略评估**问题。所谓“策略评估”，就是在一个固定策略 $\pi$ 下，估计每个状态的价值 $V^\pi(s)$，或者每个状态动作对的价值 $Q^\pi(s,a)$。这里的“策略”可以先理解成“在每个状态下如何行动的规则”。

先固定最常见的状态价值版本。记号如下：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $s_t$ | 时刻 $t$ 的状态 | 当前所处位置或局面 |
| $r_{t+1}$ | 从 $s_t$ 转移后得到的奖励 | 这一步立刻拿到的反馈 |
| $s_{t+1}$ | 下一状态 | 走一步之后到了哪里 |
| $\gamma$ | 折扣因子 | 未来奖励在当前的重要程度 |
| $V(s)$ | 状态价值函数 | 这个状态长期值不值 |

TD 适用的典型边界是：

1. 你能与环境交互，拿到 $(s_t, r_{t+1}, s_{t+1})$ 这样的样本。
2. 你不一定知道环境完整模型，也就是不知道精确的状态转移概率和奖励函数。
3. 你希望在线更新，而不是等所有数据收集完再统一训练。
4. 回合可能很长，甚至没有明确终点，不能指望总是等到最终回报。

一个玩具例子是走廊机器人。机器人在一条走廊里向前移动，左边是起点，右边是目标。每走一步，系统只告诉它“这一步奖励是多少”和“你现在到了哪个格子”，但不会告诉它“从这里出发最终一定能不能成功”。TD 正适合这种设定，因为它只需要一步反馈。

但 TD 也有明确边界。它不是直接求最优策略的完整答案。最基础的 TD 是“评估”，不是“控制”。如果你想一边评估一边优化动作选择，通常会进一步扩展成 SARSA、Q-learning 或 Actor-Critic。它们仍然使用 TD 误差，但任务目标已经从“估值”变成了“边估值边改策略”。

---

## 核心机制与推导

TD 的出发点来自 Bellman 方程。Bellman 方程的意思是：**一个状态的价值，等于一步奖励加上下一个状态价值的折扣期望**。写成公式是：

$$
V^\pi(s)=\mathbb{E}_\pi[r_{t+1}+\gamma V^\pi(s_{t+1})\mid s_t=s]
$$

这里的 $\mathbb{E}$ 是期望，白话解释是“平均来看会怎样”。

如果环境模型已知，可以直接对这个期望做精确计算，这就是 Dynamic Programming 的路径。但很多真实问题里没有模型，于是只能采样。采样以后，把期望替换成单次观测，就得到 TD 的目标：

$$
r_{t+1}+\gamma V(s_{t+1})
$$

然后定义 TD error，也叫时序差分误差：

$$
\delta_t = r_{t+1}+\gamma V(s_{t+1})-V(s_t)
$$

这里的 $\delta_t$ 可以理解成“当前估计错了多少”。如果它为正，说明当前状态被低估了；如果为负，说明被高估了。

于是 TD(0) 更新就变成：

$$
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
$$

如果是参数化形式，比如用神经网络或线性函数逼近 $V(s;w)$，就写成：

$$
\delta_t = r_{t+1}+\gamma V(s_{t+1};w_t)-V(s_t;w_t)
$$

$$
w_{t+1}=w_t+\alpha \delta_t \nabla_w V(s_t;w_t)
$$

这里的 $\nabla_w V(s_t;w_t)$ 是梯度，白话解释是“参数改一点，当前状态价值会朝哪个方向变”。

### 一个最小数值例子

设：

- $V(s_t)=0.50$
- $V(s_{t+1})=0.60$
- $r_{t+1}=1$
- $\gamma=0.9$
- $\alpha=0.1$

则：

$$
\delta_t=1+0.9\times0.60-0.50=1.04
$$

更新后：

$$
V(s_t)\leftarrow 0.50+0.1\times1.04=0.604
$$

这一步没有等回合结束，只基于一步转移就把 $V(s_t)$ 从 `0.50` 推到了 `0.604`。这就是“在线自举更新”。“自举”这个词的白话解释是：**用自己当前的估计去帮助生成新的学习目标**。

### 从 TD(0) 到 TD($\lambda$)

TD(0) 只看一步。更一般地，可以看 $n$ 步回报：

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n V(s_{t+n};w_t)
$$

它的含义是：前 $n$ 步用真实奖励，后面没观察到的未来部分仍用估计值补上。

再进一步，把不同 $n$ 步回报按权重混合，就得到 $\lambda$-return：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

其中 $\lambda\in[0,1]$ 控制“更信短期还是更信长期”。直观上：

| 参数位置 | 更接近什么 | 含义 |
| --- | --- | --- |
| $\lambda=0$ | TD(0) | 完全单步自举 |
| $\lambda\to1$ | Monte Carlo | 更依赖长回报 |
| 中间值 | 折中 | 在偏差和方差之间找平衡 |

TD($\lambda$) 常用资格迹实现。资格迹可以理解成“最近访问过哪些状态，它们现在还应不应该继续分到本次误差的责任”。其公式是：

$$
z_t=\gamma\lambda z_{t-1}+\nabla_w V(s_t;w_t)
$$

$$
w_{t+1}=w_t+\alpha \delta_t z_t
$$

它的工程意义是：一次 TD 误差不只更新当前状态，也会沿着近期访问路径向后传播。这样学得更快，尤其在延迟奖励场景里更明显。

### 偏差与方差为什么会变

Monte Carlo 的目标是真实完整回报，因此无偏，但一条轨迹可能很长，波动很大，所以方差高。TD 用了估计值补未来，因此引入偏差，但因为不需要等整条轨迹、目标波动更小，所以方差更低。

这个权衡可以粗略理解成：

$$
\text{误差} \approx \text{偏差} + \text{方差}
$$

TD 不是“更真实”，而是“更稳定、更快更新”。在工程里，很多时候这是更重要的性质。

### 一个真实工程例子

在 Actor-Critic 里，Actor 是策略网络，白话解释是“负责决定动作”；Critic 是价值网络，白话解释是“负责判断当前决策大概值不值”。

Critic 常用 TD 学习状态价值：

$$
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)
$$

然后 Actor 用这个误差或 Advantage 去更新策略。如果 Critic 学得太慢，Actor 的梯度就会很噪；如果 Critic 学得更稳定，策略优化也更容易收敛。所以很多连续控制、广告投放、推荐排序在线训练任务里，真正稳定训练流程的关键常常不是策略公式本身，而是 TD 这层价值估计是否稳。

---

## 代码实现

先给一个最小可运行的 TD(0) 表格版实现。它不依赖第三方库，适合先把更新逻辑看清楚。

```python
from collections import defaultdict

def td0_update(V, s, r, s_next, alpha=0.1, gamma=0.9, terminal=False):
    next_v = 0.0 if terminal else V[s_next]
    target = r + gamma * next_v
    delta = target - V[s]
    V[s] += alpha * delta
    return delta

def run_toy_episode():
    V = defaultdict(float)

    # 手工初始化，便于验证
    V["A"] = 0.50
    V["B"] = 0.60

    delta = td0_update(V, s="A", r=1.0, s_next="B", alpha=0.1, gamma=0.9, terminal=False)

    assert abs(delta - 1.04) < 1e-9
    assert abs(V["A"] - 0.604) < 1e-9
    assert abs(V["B"] - 0.60) < 1e-9

    # 终止状态处理：下一状态价值必须视为 0
    V["C"] = 2.0  # 即使表里有值，terminal=True 时也不能拿来用
    old_b = V["B"]
    delta2 = td0_update(V, s="B", r=0.0, s_next="C", alpha=0.5, gamma=0.9, terminal=True)

    expected_delta2 = 0.0 - old_b
    expected_b = old_b + 0.5 * expected_delta2

    assert abs(delta2 - expected_delta2) < 1e-9
    assert abs(V["B"] - expected_b) < 1e-9

    return V

if __name__ == "__main__":
    values = run_toy_episode()
    print(dict(values))
```

这段代码里最重要的不是语法，而是两件事：

1. `target = r + gamma * next_v`
2. 终止状态时 `next_v = 0.0`

如果第二条错了，整条价值传播都会被污染。

下面是实现中的关键变量。

| 变量 | 作用 | 常见错误 |
| --- | --- | --- |
| `alpha` | 学习率，控制更新幅度 | 设太大导致震荡 |
| `gamma` | 折扣因子，控制未来权重 | 和 `lambda` 混淆 |
| `target` | TD 学习目标 | 终止状态仍错误引用 `V(s_next)` |
| `delta` | TD 误差 | 号写反，导致朝反方向更新 |
| `terminal` | 是否终止 | 忽略后会持续错误 bootstrap |

如果扩展到 TD($\lambda$)，伪代码结构会变成：

```python
initialize V(s), eligibility trace z(s)=0
for each episode:
    reset z
    s = start_state
    while s is not terminal:
        observe r, s_next
        delta = r + gamma * V(s_next) - V(s)
        z(s) += 1
        for each state x:
            V(x) += alpha * delta * z(x)
            z(x) *= gamma * lambda
        s = s_next
```

表格版足够讲清机制，但真实工程里通常会用函数逼近。线性模型时，`V(s;w)=w^\top x(s)`；神经网络时，`V(s;w)` 由网络输出。无论外层模型多复杂，TD 的目标和误差定义都没变，变的只是“怎么表示 $V$”。

一个真实工程例子是推荐系统的长时价值预估。假设状态是“用户最近行为序列”，动作是“展示哪个内容”，奖励是“点击、停留、转化”。短期点击容易观察，长期收益要延迟体现。此时可用 Critic 预测状态价值，用 TD 把短期反馈和未来估计连接起来。工程里不一定直接叫 `V(s)`，也可能叫“长期收益预测”“value head”或“bootstrap target”，但底层逻辑仍然是 TD。

---

## 工程权衡与常见坑

TD 的优点很明确：

- 每一步都能更新，样本利用更及时。
- 不要求等完整轨迹结束，适合长回合或持续任务。
- 相比 Monte Carlo，训练波动通常更小。

但它的代价同样明确：**目标里包含当前估计，所以会有偏差，而且可能把自己的错误继续传播下去。**

下面是常见坑。

| 现象 | 原因 | 处理方式 |
| --- | --- | --- |
| 价值发散或剧烈震荡 | 学习率过大，函数逼近不稳 | 降低 `alpha`，做梯度裁剪 |
| 回报越学越奇怪 | 终止状态未置零 | terminal 时强制 `V(next)=0` |
| 学习很慢 | 奖励过于稀疏 | 奖励塑形或更长回报传播 |
| TD 误差极端抖动 | 奖励尺度过大 | reward normalization |
| 离策略训练不稳定 | bootstrapping + off-policy + function approximation 组合风险 | target network、experience replay、Gradient-TD |

最常见的新手错误有三个。

第一，把 $\gamma$ 和 $\lambda$ 混为一谈。$\gamma$ 解决的是“未来值多少钱”，$\lambda$ 解决的是“误差沿多长历史传播”。它们控制的不是同一个维度。

第二，终止状态处理错。正确规则是：如果 `s_next` 是终止状态，那么目标应是

$$
\text{target}=r_{t+1}
$$

而不是 $r_{t+1}+\gamma V(s_{t+1})$。终点之后没有未来价值。

第三，以为“公式没问题，训练就该稳定”。实际工程里，TD 的不稳定经常不是推导错，而是动态系统问题。特别是离策略训练加函数逼近时，经典的“deadly triad”风险会出现：bootstrapping、自举；off-policy，离策略；function approximation，函数逼近。这三者叠加可能直接导致发散。

真实工程里常见的稳定化手段包括：

- `reward normalization`：把奖励缩放到更稳定区间。
- `gradient clipping`：防止 TD 误差过大时梯度爆炸。
- `target network`：用延迟参数提供更平稳的 bootstrap 目标。
- `experience replay`：打散样本相关性，提高数据复用。
- `Gradient-TD / Emphatic TD`：在特定离策略设定下提供更稳的收敛性质。

如果任务是简单表格环境，这些问题可能不明显。但一旦进入深度强化学习，很多所谓“算法差异”最后都体现在“TD 目标怎么构造、怎么稳定”上。

---

## 替代方案与适用边界

TD 不是唯一方法，也不是所有情况下最优的方法。选法取决于你能拿到什么信息，以及你更在意偏差还是方差。

| 方法 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| Monte Carlo | 回合短，终局回报清晰 | 无偏，概念直接 | 必须等结束，方差高 |
| TD | 在线学习，回合长，持续任务 | 更新快，方差低 | 有偏，可能不稳定 |
| Dynamic Programming | 已知环境模型 | 可直接做全局备份 | 现实中常拿不到模型 |
| SARSA | on-policy 控制 | 行为与学习一致 | 探索时较保守 |
| Q-learning | off-policy 控制 | 可直接逼近最优动作价值 | 稳定性要求更高 |
| Actor-Critic | 大规模控制、连续动作 | 策略优化与价值估计结合 | 工程复杂度更高 |

如果只是做**预测问题**，例如评估固定策略下不同状态值不值，TD 学 $V(s)$ 往往就是直接答案。

如果是**控制问题**，也就是不仅要估值，还要学会更好的动作选择，那么 TD 通常成为更大算法的一部分：

- SARSA：用 TD 学 $Q(s,a)$，并按照当前策略行动。
- Q-learning：用 TD 学最优动作价值。
- Actor-Critic：Critic 用 TD 估值，Actor 用它来改策略。

再看选择边界。

一个玩具例子里，如果一局只有 5 步，终点回报清晰，而且你完全可以等回合结束，那么 Monte Carlo 其实很自然，甚至更容易讲明白。

一个真实工程例子里，如果你在做机器人控制或广告竞价，决策链很长，在线反馈连续到来，不可能总等最终收益完全实现再更新，那么 TD 通常比纯 Monte Carlo 更实用。

可以把它概括成一句话：**只要你需要“边交互边学习”，并且愿意接受一点偏差换更稳定的更新，TD 通常是默认起点。**

---

## 参考资料

1. [Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences](https://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
2. [Sutton & Barto. Reinforcement Learning: An Introduction, Chapter 6 and Chapter 12](https://incompleteideas.net/book/RLbook2020.pdf)
3. [Watkins, C. J. C. H. & Dayan, P. (1992). Q-Learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html)
4. [Konda, V. R. & Tsitsiklis, J. N. (1999). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms)
5. [van Seijen, H. & Sutton, R. S. (2014). True Online TD(lambda)](https://proceedings.mlr.press/v32/seijen14.html)
