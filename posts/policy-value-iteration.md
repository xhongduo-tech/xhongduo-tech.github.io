## 核心结论

策略迭代和价值迭代解决的是同一个问题：在**有限状态、有限动作、已知环境模型**的马尔可夫决策过程（MDP，意思是“当前怎么选动作，只和当前状态有关”）里，找到最优策略和最优价值函数。它们都属于广义策略迭代（GPI, Generalized Policy Iteration，意思是“评估当前做法，再朝更优做法推进”的总框架）。

两者的差别不在结果，而在每轮更新的深度。

| 维度 | 策略迭代 | 价值迭代 |
|---|---|---|
| 目标 | 求最优策略 $\pi^*$ | 求最优价值 $V^*$，再提取 $\pi^*$ |
| 每轮做什么 | 先完整评估当前策略，再改进 | 直接做最优贝尔曼备份 |
| 更新核心 | $V^\pi \rightarrow \text{greedy}(V^\pi)$ | $V_{k+1}(s)=\max_a Q_{V_k}(s,a)$ |
| 单轮成本 | 高 | 低 |
| 迭代轮数 | 通常少 | 通常多 |
| 输出形式 | 直接得到稳定策略 | 先得到价值，再导出策略 |
| 直观理解 | 先把当前方案算准，再换方案 | 每轮只往最优方向推一步 |

新手可以先抓住一句话：**策略迭代是“深算后再换”，值迭代是“边算边逼近最优”。**

---

## 问题定义与边界

本文讨论的是标准动态规划场景，也就是模型已知的 MDP。模型已知，意思是你知道执行动作后会转移到哪里、概率是多少、奖励是多少。

一个 MDP 常写成五元组：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $S$ | 状态集合 | 系统当前所处的位置 |
| $A$ | 动作集合 | 当前能做的选择 |
| $p(s',r\mid s,a)$ | 转移与奖励模型 | 在状态 $s$ 做动作 $a$ 后，以什么概率到 $s'$ 并得到奖励 $r$ |
| $\gamma \in [0,1)$ | 折扣因子 | 未来奖励的重要程度 |
| $\pi$ | 策略 | 在每个状态决定做什么 |

这里有两个最容易混的术语：

| 对象 | 定义 | 作用 |
|---|---|---|
| 策略 $\pi(s)$ | 在状态 $s$ 该选哪个动作 | 决定“做什么” |
| 价值函数 $V(s)$ | 从状态 $s$ 出发未来总回报的估计 | 衡量“这样做有多好” |

如果策略固定为 $\pi$，它对应的状态价值函数满足贝尔曼期望方程：

$$
V^\pi(s)=\sum_{s',r} p(s',r\mid s,\pi(s))\left[r+\gamma V^\pi(s')\right]
$$

这条式子的含义很直接：当前状态的价值，等于执行当前策略后，所有可能后继结果的“奖励 + 折扣后的未来价值”的期望。

边界也要先说清楚。本文不讨论以下情况：

1. 模型未知，只能靠和环境交互采样。
2. 状态空间极大，无法显式存储整个 $V(s)$ 表。
3. 深度强化学习里用神经网络近似价值函数或策略。

一旦模型未知，问题就不再是纯动态规划，而会转向 TD、SARSA、Q-learning、Actor-Critic 这些近似 GPI 方法。

---

## 核心机制与推导

统一看这两个算法，最好先引入一个中间量：

$$
Q_V(s,a)=\sum_{s',r} p(s',r\mid s,a)\left[r+\gamma V(s')\right]
$$

这里的 $Q_V(s,a)$ 可以理解成：**假设后续状态的价值由当前的 $V$ 给出，那么现在在 $s$ 先做动作 $a$，这一步的好坏是多少。**

有了它以后，“改进策略”就很好写：

$$
\pi'(s)=\arg\max_a Q_V(s,a)
$$

这叫贪心改进。贪心，意思是“基于当前价值估计，局部选择最优动作”。

### 策略迭代的逻辑

策略迭代分两步反复执行：

1. **策略评估**：给定当前策略 $\pi$，求解或迭代逼近它的真实价值 $V^\pi$。
2. **策略改进**：用 $V^\pi$ 计算每个动作的好坏，再令新策略对每个状态都选最优动作。

流程可以写成：

`π -> 评估 -> V^π -> 改进 -> π'`

为什么它会收敛？核心原因是策略改进定理：如果对每个状态都按 $Q_{V^\pi}(s,a)$ 选更优动作，那么新策略不会比旧策略差，通常还会更好。有限 MDP 中可行策略数有限，所以不断改进后最终会停在某个无法继续提升的策略上，也就是最优策略。

### 值迭代的逻辑

值迭代直接把“评估”和“改进”压缩成一步：

$$
V_{k+1}(s)=\max_a \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma V_k(s')\right]
$$

也就是：

$$
V_{k+1}(s)=\max_a Q_{V_k}(s,a)
$$

这就是贝尔曼最优备份。备份，意思是“用后继状态的估计值反推当前状态的新值”。

流程可以写成：

`V_k -> 最优备份 -> V_{k+1} -> ... -> 收敛后提取策略`

它不显式维护“当前策略”，而是每轮都直接朝最优价值推进。等 $V_k$ 收敛后，再提取：

$$
\pi^*(s)=\arg\max_a Q_{V^*}(s,a)
$$

### 玩具例子

设只有一个非终止状态 $s$，折扣因子 $\gamma=0.9$。

- 动作 $a$：立即终止，奖励 $5$
- 动作 $b$：留在 $s$，每步奖励 $1$

如果当前策略总选 $a$，那么：

$$
V^\pi(s)=5
$$

但按这个价值去看动作 $b$：

$$
Q_{V^\pi}(s,b)=1+0.9\times 5=5.5
$$

所以策略改进后会改成选 $b$。而如果总选 $b$，价值满足：

$$
V(s)=1+0.9V(s)
$$

解得：

$$
V^*(s)=\frac{1}{1-0.9}=10
$$

这个例子说明了两件事：

1. 当前策略的价值是“对当前做法的评估”，不是全局最优。
2. 只要改进步骤发现更优动作，策略就会继续变化。

如果用值迭代，从 $V_0(s)=0$ 开始：

- $V_1=\max(5,1)=5$
- $V_2=\max(5,1+0.9\times 5)=5.5$
- $V_3=\max(5,1+0.9\times 5.5)=5.95$

后面会逐步逼近 $10$。这就是“每轮向最优方向推一点”。

### GPI 视角

GPI 的关键不在某个具体公式，而在两股力量同时存在：

1. **评估**让价值函数更接近当前策略的真实回报。
2. **改进**让策略更接近当前价值函数下的最优动作。

这两条线不一定要完全分开。策略迭代是“长评估 + 一次改进”，值迭代是“极短评估 + 立即改进”。因此，值迭代可以看成策略迭代的一个极端版本。

更一般地，完整评估并不是必须的。工程中常见三种变体：

| 变体 | 做法 | 意义 |
|---|---|---|
| 截断评估 | 评估几轮就改进 | 降低单轮成本 |
| 异步更新 | 每次只更新部分状态 | 更适合大状态空间 |
| 部分策略改进 | 只在部分状态上改进 | 便于分块优化 |

这些都仍然属于 GPI，因为“评估 + 改进”的骨架没有变。

---

## 代码实现

下面给一个最小可运行实现。环境仍然使用上面的单状态玩具例子。重点不在语法，而在三个函数的分工：

1. `policy_evaluation(policy)`
2. `policy_improvement(V)`
3. `value_iteration()`

```python
from math import isclose

gamma = 0.9
states = ["s"]
actions = ["a", "b"]

# MDP: transitions[(state, action)] = list of (prob, next_state, reward, done)
transitions = {
    ("s", "a"): [(1.0, None, 5.0, True)],   # terminate with reward 5
    ("s", "b"): [(1.0, "s", 1.0, False)],   # stay with reward 1
}

def q_from_v(state, action, V):
    total = 0.0
    for prob, next_state, reward, done in transitions[(state, action)]:
        future = 0.0 if done else V[next_state]
        total += prob * (reward + gamma * future)
    return total

def policy_evaluation(policy, theta=1e-10):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            old_v = V[s]
            V[s] = q_from_v(s, policy[s], V)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            return V

def policy_improvement(V):
    policy = {}
    stable = True
    for s in states:
        best_action = max(actions, key=lambda a: q_from_v(s, a, V))
        policy[s] = best_action
    return policy, stable

def policy_iteration():
    policy = {"s": "a"}
    while True:
        V = policy_evaluation(policy)
        new_policy = {}
        stable = True
        for s in states:
            old_action = policy[s]
            new_action = max(actions, key=lambda a: q_from_v(s, a, V))
            new_policy[s] = new_action
            if new_action != old_action:
                stable = False
        policy = new_policy
        if stable:
            return policy, V

def value_iteration(theta=1e-10):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            old_v = V[s]
            V[s] = max(q_from_v(s, a, V) for a in actions)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    policy = {s: max(actions, key=lambda a: q_from_v(s, a, V)) for s in states}
    return policy, V

pi_star_pi, V_pi = policy_iteration()
pi_star_vi, V_vi = value_iteration()

assert pi_star_pi["s"] == "b"
assert pi_star_vi["s"] == "b"
assert isclose(V_pi["s"], 10.0, rel_tol=1e-8, abs_tol=1e-8)
assert isclose(V_vi["s"], 10.0, rel_tol=1e-8, abs_tol=1e-8)

print("policy iteration:", pi_star_pi, V_pi)
print("value iteration:", pi_star_vi, V_vi)
```

这段代码体现了两个实现差异：

| 函数 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `policy_evaluation` | 当前策略 $\pi$ | $V^\pi$ | 评估当前策略到底值多少 |
| `policy_improvement` | 当前价值 $V$ | 新策略 | 根据价值选更优动作 |
| `value_iteration` | 模型与初值 | 近似 $V^*$ 和贪心策略 | 直接做最优备份 |

如果写成伪代码，结构更清楚。

```python
# policy evaluation
repeat:
    delta = 0
    for s in states:
        v = V[s]
        V[s] = sum p(s',r|s, pi[s]) * (r + gamma * V[s'])
        delta = max(delta, abs(v - V[s]))
until delta < theta
```

```python
# policy improvement
policy_stable = True
for s in states:
    old = pi[s]
    pi[s] = argmax_a sum p(s',r|s,a) * (r + gamma * V[s'])
    if old != pi[s]:
        policy_stable = False
```

```python
# value iteration
repeat:
    delta = 0
    for s in states:
        v = V[s]
        V[s] = max_a sum p(s',r|s,a) * (r + gamma * V[s'])
        delta = max(delta, abs(v - V[s]))
until delta < theta

# extract greedy policy
pi[s] = argmax_a Q_V(s, a)
```

### 真实工程例子

仓储机器人路径规划是一个典型例子。状态可以定义为“机器人位置 + 朝向 + 是否载货”，动作是“前进、左转、右转、等待”。如果地图、障碍概率、碰撞代价、电量消耗都已知，那么就可以先建立 MDP。

- 如果场景规模中等，且离线规划时间允许，策略迭代常用于求高质量稳定策略。
- 如果状态很多，比如还要叠加时间窗、拥堵级别、货架状态，值迭代或异步值迭代更常见，因为每轮更新更轻，便于分块并行。

这里的工程重点不是“哪种算法更高级”，而是**模型规模、算力预算、是否需要快速得到一个可用策略**。

---

## 工程权衡与常见坑

工程上比较这两种方法，核心不是数学优雅，而是成本结构。

| 问题 | 策略迭代 | 价值迭代 |
|---|---|---|
| 单轮是否重 | 重 | 轻 |
| 是否需要完整评估 | 需要或近似需要 | 不需要 |
| 是否容易实现 | 中等 | 简单 |
| 是否能早期得到可用近似 | 一般 | 更容易 |
| 大状态空间下是否灵活 | 一般 | 更灵活 |

最常见的误区有以下几个。

| 常见坑 | 错误表现 | 正确做法 |
|---|---|---|
| 把 $V(s)$ 当成 $\pi(s)$ | 以为价值大就等于动作最优 | 先算 $Q_V(s,a)$，再取 $\arg\max$ |
| 值迭代中途就提策略并认为已最优 | 第 1、2 轮策略变化很大 | 收敛后再提取，或显式监控策略稳定 |
| 忽略模型已知前提 | 只有采样轨迹却硬套 DP | 模型未知时用 TD/Q-learning 等 |
| 终止条件混用 | 价值变化小就认为策略一定稳定 | 分开看“价值阈值”和“策略稳定” |
| 原地更新与同步更新没分清 | 结果和推导不一致 | 明确使用 in-place 还是 batch 更新 |

这里尤其要强调第二点。值迭代的中间结果经常会让人误判。因为 $V_k$ 可能已经比前一轮好很多，但基于它提取的贪心策略还没稳定。价值变好，不等于策略已经最优。

一个实用检查清单如下：

1. 转移模型 $p(s',r\mid s,a)$ 是否和环境定义一致。
2. 折扣因子 $\gamma$ 是否满足 $[0,1)$。
3. 策略迭代里，评估是否做到了足够收敛。
4. 值迭代里，是否在收敛后再提策略。
5. 终止状态的价值和后继处理是否单独处理正确。

---

## 替代方案与适用边界

当模型已知且状态不大时，策略迭代和值迭代就是标准解法。但一旦条件变化，方法也要变。

| 场景 | 更合适的方法 | 原因 |
|---|---|---|
| 模型已知，状态不大 | 策略迭代 | 迭代轮数少，最终策略稳定 |
| 模型已知，状态较大 | 值迭代、异步值迭代 | 单轮更轻，便于局部更新 |
| 模型未知，但可交互采样 | TD、SARSA、Q-learning | 不能直接写出转移模型 |
| 状态极大，需函数近似 | 近似动态规划、DQN、Actor-Critic | 无法存完整表格 |

这些替代方案并不是和 GPI 无关。恰恰相反，它们大多可以看成**近似 GPI**。

- TD：用采样估计价值，相当于“近似评估”。
- Q-learning：直接逼近最优动作价值，相当于“采样版最优备份”。
- Actor-Critic：Critic 负责评估，Actor 负责改进，是 GPI 分工最清楚的现代形式之一。

所以策略迭代和值迭代的重要性，不只是它们本身可用，而是它们提供了一个总骨架：**任何强化学习算法，几乎都可以问一句，它在怎样做评估，又在怎样做改进。**

适用边界也因此很明确：

1. 如果你有完整环境模型，优先考虑动态规划。
2. 如果模型没有，但可以采样，就转向采样式方法。
3. 如果状态空间太大，必须接受近似，不能再要求精确 $V^\pi$ 或精确贪心改进。

---

## 参考资料

1. [Reinforcement Learning: An Introduction 官方书页](https://incompleteideas.net/book/the-book.html)
2. [Sutton & Barto, Reinforcement Learning: An Introduction, Chapter 4 Draft PDF](https://incompleteideas.net/book/bookdraft2018mar21.pdf)
3. [Sutton 官方 GPI 章节 4.6 Generalized Policy Iteration](https://www.incompleteideas.net/book/4/node7.html)
4. [Sutton & Barto 早期书稿 PDF](https://incompleteideas.net/book/bookdraft2016sep.pdf)
