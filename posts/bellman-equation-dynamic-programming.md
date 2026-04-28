## 核心结论

Bellman 方程的核心作用，是把“一个长期决策问题”拆成“当前一步收益”和“未来剩余价值”两部分。白话说，它回答的是：现在这个状态到底值多少钱，不只看眼前这一步，还要把后面能拿到的回报一起算进去。

在给定策略 $\pi$ 时，状态价值函数满足 Bellman 期望方程：

$$
V^\pi(s)=\mathbb{E}[r+\gamma V^\pi(s')]
$$

在追求最优策略时，最优状态价值函数满足 Bellman 最优方程：

$$
V^*(s)=\max_a \mathbb{E}[r+\gamma V^*(s')]
$$

这两条式子分别是策略评估、策略迭代、值迭代、Q-learning 和 DQN 的数学基础。它们不是“经验公式”，而是不动点方程。所谓不动点，就是反复应用更新规则后会稳定下来的解。只要满足马尔可夫性并且 $\gamma<1$，对应的 Bellman 算子通常是压缩映射，迭代就会收敛到唯一解。

先看一句最直白的话：今天的总价值 = 今天拿到的奖励 + 明天之后还能拿到的价值。Bellman 方程只是把这句话写成了数学形式。

一个最小玩具例子能直接说明“未来价值会影响当前决策”。假设在状态 $s_1$ 有两个动作：

| 状态 | 动作 | 立即奖励 | 下一状态 |
|---|---|---:|---|
| $s_1$ | A | 2 | 终止 |
| $s_1$ | B | 0 | $s_2$ |

如果 $s_2$ 的未来价值是 10，且 $\gamma=0.9$，那么动作 B 的总价值是 $0+0.9\times 10=9$，虽然眼前奖励更小，但长期更优。这就是 Bellman 思想最重要的地方：不能只看一步。

---

## 问题定义与边界

Bellman 方程解决的问题是：在序列决策里，如何把未来收益系统地折算进当前决策。这里的“序列决策”指一连串相互影响的选择，比如走迷宫、下棋、广告投放、机器人控制。当前动作不只决定当前奖励，还会改变之后能到达的状态。

它通常建立在马尔可夫决策过程（MDP）上。马尔可夫性的白话解释是：只要当前状态描述完整，未来就只依赖当前状态和动作，不需要再额外记更久远的历史。

核心符号如下：

| 符号 | 名称 | 白话解释 |
|---|---|---|
| $s$ | 状态 | 当前局面 |
| $a$ | 动作 | 当前能做的选择 |
| $r$ | 奖励 | 这一步立刻拿到的反馈 |
| $P(s' \mid s,a)$ | 转移概率 | 做完动作后到下一个状态的概率 |
| $\gamma$ | 折扣因子 | 未来奖励打几折，范围通常是 $[0,1)$ |
| $V(s)$ | 状态价值 | 站在某个状态往后看，长期能拿多少回报 |
| $Q(s,a)$ | 动作价值 | 在某个状态先做某个动作，长期能拿多少回报 |

走迷宫就是一个典型例子。当前位置是状态 $s$，向上走是动作 $a$，撞墙或前进后的格子是下一状态 $s'$，到终点得到正奖励，掉进陷阱得到负奖励。Bellman 方程不是只算“下一步去哪”，而是算“走这一步会把你带进怎样的未来”。

它也有明确边界。

1. 状态必须足够完整。如果状态漏掉关键信息，Bellman 递推会基于错误前提。
2. 环境不能强非平稳。若奖励规则和转移规律一直变化，固定 Bellman 方程就不成立。
3. 终止状态要单独处理。终止状态之后没有未来回报，因此通常取 $V(\text{terminal})=0$。
4. 如果任务不是标准序列决策，比如一次性静态分类问题，Bellman 方程就不是核心工具。

---

## 核心机制与推导

先看给定策略下的价值评估。若策略 $\pi(a\mid s)$ 已知，那么状态价值函数定义为“从状态 $s$ 出发，之后一直按策略 $\pi$ 行动时的期望累计回报”。它满足：

$$
V^\pi(s)=\sum_a \pi(a|s)\sum_{s',r}P(s',r|s,a)\left[r+\gamma V^\pi(s')\right]
$$

这不是循环定义，而是递归拆解。所谓递归，是把大问题拆成同类更小问题。这里的大问题是“从现在到结束的总回报”，小问题是“下一状态开始的总回报”。

同理，动作价值函数 $Q^\pi(s,a)$ 表示“在状态 $s$ 先执行动作 $a$，之后再按策略 $\pi$ 行动时”的长期价值：

$$
Q^\pi(s,a)=\sum_{s',r}P(s',r|s,a)\left[r+\gamma \sum_{a'}\pi(a'|s')Q^\pi(s',a')\right]
$$

当目标从“评估一个已知策略”变成“寻找最优策略”时，就把对策略的平均换成对动作的最大化。于是得到最优状态价值函数：

$$
V^*(s)=\max_a \sum_{s',r}P(s',r|s,a)\left[r+\gamma V^*(s')\right]
$$

以及最优动作价值函数：

$$
Q^*(s,a)=\sum_{s',r}P(s',r|s,a)\left[r+\gamma \max_{a'}Q^*(s',a')\right]
$$

这里的“自举”值得单独说明。自举的白话解释是：当前估计值，借助下一时刻的估计值来更新自己。它让学习高效，因为不必等到整条轨迹结束，但也引入了误差传播和训练不稳定问题。

看一个两步链式玩具例子。设有 $s_1 \to s_2 \to \text{terminal}$，在给定策略下：

- 从 $s_2$ 到终止，奖励为 3
- 从 $s_1$ 到 $s_2$，奖励为 1
- 折扣因子 $\gamma=0.9$

则：

$$
V(s_2)=3+0.9\times 0=3
$$

$$
V(s_1)=1+0.9\times 3=3.7
$$

这个计算非常重要，因为它说明当前状态的价值不等于当前奖励，而等于“当前奖励 + 折扣后的未来奖励”。

再看 $V$ 和 $Q$ 的区别：

| 对象 | 回答的问题 | 是否包含动作选择 |
|---|---|---|
| $V(s)$ | “站在这个状态，往后总共值多少？” | 不直接指定动作 |
| $Q(s,a)$ | “在这个状态先做这个动作，往后总共值多少？” | 显式包含动作 |

因此，若已知 $Q^*(s,a)$，最优动作就是 $\arg\max_a Q^*(s,a)$；若只知道 $V^*(s)$，还需要额外比较每个动作通向的下一状态价值。

从数学上，值迭代之所以能收敛，是因为在 $\gamma<1$ 时，Bellman 最优算子满足压缩性质：

$$
\|T V_1 - T V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty
$$

压缩的意思是：每做一次更新，两个候选价值函数之间的最坏差距最多缩小到原来的 $\gamma$ 倍。因为 $\gamma<1$，差距会越缩越小，最终收敛到唯一不动点。

---

## 代码实现

先看值迭代的伪代码。它适用于“已知环境模型”的表格型问题。

```text
Initialize V(s)=0 for all states
repeat:
    delta = 0
    for each state s:
        old_v = V(s)
        V(s) = max_a sum_{s',r} P(s',r|s,a) * [r + gamma * V(s')]
        delta = max(delta, |old_v - V(s)|)
until delta < tolerance
```

下面给出一个可运行的 Python 玩具实现。这个例子里，`s1` 有两个动作：`direct` 立即拿 2 分结束，`detour` 先到 `s2` 再拿 3 分结束。正确答案应该偏向 `detour`，因为 $0+0.9\times 3=2.7 > 2$。

```python
def value_iteration(gamma=0.9, tol=1e-10, max_iter=1000):
    states = ["s1", "s2", "terminal"]
    actions = {
        "s1": ["direct", "detour"],
        "s2": ["finish"],
        "terminal": []
    }

    transitions = {
        ("s1", "direct"): [(1.0, "terminal", 2.0)],
        ("s1", "detour"): [(1.0, "s2", 0.0)],
        ("s2", "finish"): [(1.0, "terminal", 3.0)],
    }

    V = {s: 0.0 for s in states}

    for _ in range(max_iter):
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if s == "terminal":
                new_V[s] = 0.0
                continue

            q_values = []
            for a in actions[s]:
                q = 0.0
                for p, s_next, r in transitions[(s, a)]:
                    q += p * (r + gamma * V[s_next])
                q_values.append(q)

            new_V[s] = max(q_values)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < tol:
            break

    return V

V = value_iteration()
assert abs(V["s2"] - 3.0) < 1e-8
assert abs(V["s1"] - 2.7) < 1e-8
assert V["s1"] > 2.0
print(V)
```

当环境模型未知时，常见做法是用采样近似 Bellman 更新。Q-learning 的单步目标值是：

$$
y = r + \gamma \max_{a'}Q(s',a')
$$

若 $s'$ 是终止状态，则未来价值为 0，因此：

$$
y = r
$$

对应的更新为：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left(y-Q(s,a)\right)
$$

其中 $\alpha$ 是学习率，白话解释是“这次新信息对旧估计覆盖多少”。

```python
def q_learning_update(q, s, a, r, s_next, done, alpha=0.1, gamma=0.9):
    next_max = 0.0 if done else max(q.get((s_next, a2), 0.0) for a2 in ["left", "right"])
    y = r + gamma * next_max
    old = q.get((s, a), 0.0)
    q[(s, a)] = old + alpha * (y - old)
    return q[(s, a)]

q = {}
new_q = q_learning_update(q, "A", "right", r=1.0, s_next="B", done=True)
assert abs(new_q - 0.1) < 1e-8
```

真实工程例子是推荐系统或广告排序。一次曝光不只影响当前点击，还会影响用户后续停留、继续浏览和长期留存。如果只优化单步点击率，系统可能学会“骗点击”；如果使用 Bellman 视角建模长期价值，就能把后续收益一起纳入当前排序决策。这也是强化学习在推荐和在线决策中的一个重要出发点。

---

## 工程权衡与常见坑

Bellman 方程在理论上很干净，但工程上并不天然稳定。原因是它常和函数逼近、采样估计、自举更新一起出现，这三者叠加后容易放大误差。

最常见的问题如下：

| 常见坑 | 典型表现 | 修正手段 |
|---|---|---|
| `V` 和 `Q` 混用 | 动作选择逻辑混乱，更新目标写错 | 明确 `V` 不含动作，`Q` 含动作 |
| 终止状态未归零 | 回报被重复累计，价值虚高 | `done=True` 时令下一步价值为 0 |
| $\gamma$ 过大或取值不当 | 收敛很慢，噪声长期累积 | 根据任务时域选择，一般先从 0.9 或 0.99 开始 |
| 只看单次采样不看期望 | 更新方差大，训练曲线抖动 | 用更多样本、经验回放或批量估计 |
| 自举目标导致训练发散 | 目标值一直漂移，越学越不稳 | 固定 target network，降低更新耦合 |

为什么 DQN 里要用 target network？因为 Bellman 目标里的 $\max_{a'}Q(s',a')$ 也来自模型自己。如果在线网络一边预测当前值，一边又立刻给自己制造新目标，目标会不断漂移。target network 的作用，就是先冻结一份较慢更新的目标网络，让“被学习的目标”在短时间内相对稳定。这不是在修改 Bellman 方程，而是在修正采样近似 Bellman 更新时的工程不稳定性。

还要注意“单样本不是期望”。公式里写的是 $\mathbb{E}$，但代码里经常只拿到一个样本转移。如果把单个样本误当成精确期望，就会高估当前更新的可靠性。表格型动态规划和深度强化学习的核心差异之一，就在于前者通常直接访问完整模型，后者只能通过样本逼近。

---

## 替代方案与适用边界

Bellman 体系不是唯一解法。它适合能定义价值递推的问题，但并不覆盖所有场景。

| 方法 | 是否需要环境模型 | 是否自举 | 方差/偏差 | 适合场景 |
|---|---|---|---|---|
| Bellman / Dynamic Programming | 需要 | 是 | 低方差、低偏差，但要求模型已知 | 小规模、模型明确的 MDP |
| Monte Carlo | 不需要 | 否 | 高方差、低偏差 | 轨迹完整、可等待回合结束 |
| n-step TD | 不需要 | 部分自举 | 偏差和方差折中 | 想兼顾稳定性与效率 |
| Policy Gradient | 不需要 | 通常不靠值函数主导 | 方差较高 | 连续动作、高维策略优化 |
| Model-based Planning | 需要或学习模型 | 可选 | 依赖模型误差 | 规划、控制、样本昂贵任务 |

一个关键对比是 value-based 和 policy-based。

1. Bellman 类 value-based 方法：先学“这个状态或动作值多少”，再据此选动作。
2. Policy Gradient：直接优化“在这个状态做哪个动作的概率更大”。

前者优点是样本效率通常更高，缺点是容易受自举误差影响；后者优点是可以直接优化策略，特别适合连续动作，缺点是方差往往更大。

当环境强部分可观测、状态维度极高、长期信用分配非常困难时，纯 Bellman 方法会越来越难。比如复杂对话系统、长链工具调用、多智能体博弈，往往需要把 Bellman 思想和记忆机制、模型规划、策略优化结合使用，而不是只靠一个表格型价值递推。

---

## 参考资料

1. [Reinforcement Learning: An Introduction, Sutton and Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
2. [Dynamic Programming and Optimal Control, Dimitri P. Bertsekas](https://web.mit.edu/dimitrib/www/dpchapter.html)
3. [Bellman Equation, David Silver RL Course Lecture Notes](https://www.davidsilver.uk/teaching/)
4. [Q-Learning, Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)
5. [Playing Atari with Deep Reinforcement Learning, Mnih et al.](https://arxiv.org/abs/1312.5602)
