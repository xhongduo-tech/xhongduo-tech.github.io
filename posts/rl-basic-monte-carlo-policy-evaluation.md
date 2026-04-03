## 核心结论

蒙特卡洛策略评估，简称 MCPE，可以直接理解成“按固定策略把一局完整玩完，再用这局真实拿到的总回报给状态打分”。这里的“回报”是从某个时刻开始，后续奖励按折扣系数加权后的总和。它要估计的是状态价值函数 $V^\pi(s)$，也就是“当智能体遵循策略 $\pi$ 时，处在状态 $s$ 未来大概能拿到多少累计收益”。

它的核心公式很简单：

$$
V^\pi(s) \approx \frac{1}{N}\sum_{i=1}^{N} G_t^{(i)}
$$

其中 $G_t^{(i)}$ 是第 $i$ 次在状态 $s$ 上观察到的完整轨迹回报，$N$ 是样本次数。直白说，就是“同一个状态多看几次，把后续真实结果取平均”。

它有三个最重要的性质：

| 性质 | 含义 |
| --- | --- |
| 不需要环境模型 | 不用提前知道状态转移概率和奖励函数，直接靠采样 |
| 不做 bootstrap | 不用拿别的估计值去近似后续价值，只用真实整局结果 |
| 无偏但高方差 | 长期平均是对的，但单次样本波动可能很大 |

玩具例子可以用一个最小迷宫来理解。假设策略 $\pi$ 很简单：在迷宫里一直优先向右走，走不通再向下。某一局从起点出发后，奖励序列是：

| episode | 从起点算起的回报 $G$ |
| --- | --- |
| 第1局 | 0, 0, 0, 1，故 $G=1$ |
| 第2局 | 0, -1, 0, 1，故 $G=0$ |
| 第3局 | 0, 0, -1, 1，故 $G=0$ |

如果折扣因子 $\gamma=1$，那么起点状态的估计值就是这三局平均值 $\frac{1+0+0}{3}=\frac13$。这就是 MCPE 的基本逻辑：不是提前推演，而是多次“玩完再总结”。

---

## 问题定义与边界

MCPE 解决的问题是：给定一个固定策略 $\pi$，估计每个状态的价值

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
$$

这里的“策略”可以理解成“遇到某个状态时如何选动作的规则”，而“价值”就是“从这个状态出发，后面平均还能拿多少分”。

回报 $G_t$ 的定义是：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1}R_T
$$

其中：

- $R_{t+1}$ 是下一步奖励
- $\gamma \in [0,1]$ 是折扣因子，白话说就是“未来奖励要不要打折”
- $T$ 是回合结束时刻

边界要说清楚。MCPE 主要适用于 episodic task，也就是“回合式任务”。白话说，这类任务有明确开始和结束，比如一局迷宫、一盘棋、一轮游戏。因为只有回合结束后，才能准确知道这条轨迹从某状态开始到底拿了多少总回报。

一个两状态玩具例子：

- 状态 `Start`
- 状态 `Goal`
- 到达 `Goal` 后回合结束，奖励为 `+1`

如果智能体从 `Start` 出发，中间还没到终点时，你并不知道这条轨迹最终会不会绕路、撞墙、拿惩罚，所以无法立刻得到完整 $G_t$。这就是为什么 MCPE 必须等 episode 终止。

适用性可以直接对比：

| 任务类型 | 能否直接用 MCPE | 原因 |
| --- | --- | --- |
| 迷宫到终点 | 可以 | 有明确终止状态 |
| 棋类对局 | 可以 | 一盘棋天然结束 |
| 持续运行的广告推荐 | 不适合直接用 | 没有天然 episode 终点 |
| 机器人长期控制 | 通常不适合直接用 | 回合太长，更新延迟大 |

所以，MCPE 的前提不是“环境简单”，而是“能拿到完整轨迹”。

---

## 核心机制与推导

MCPE 的机制可以拆成三步：

1. 按策略 $\pi$ 采样完整 episode。
2. 对 episode 中每个时刻向后回溯，算出 $G_t$。
3. 把同一状态对应的多个 $G_t$ 做平均。

它成立的原因并不神秘。因为定义上就有：

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
$$

也就是说，状态价值本来就是“条件期望回报”。既然目标是期望，最自然的估计办法就是样本均值。随着样本数 $N \to \infty$，根据大数定律，样本均值会收敛到真实期望：

$$
\frac{1}{N}\sum_{i=1}^{N}G^{(i)} \to \mathbb{E}[G]
$$

这里有两个经典版本。

| 方法 | 统计规则 | 特点 |
| --- | --- | --- |
| First-Visit MC | 每个 episode 中，某状态只在第一次出现时记一次回报 | 样本更“干净” |
| Every-Visit MC | 每个 episode 中，某状态每次出现都记回报 | 样本更多，但相关性更强 |

看一个玩具例子。某一局中状态 `A` 出现了两次，后续奖励如下：

- 第一次到达 `A` 后，未来奖励序列为 `[0, 2, 1]`
- 第二次到达 `A` 后，未来奖励序列为 `[2, 1]`
- 设 $\gamma=1$

那么：

- 第一次出现的回报：$G^{(1)} = 0+2+1=3$
- 第二次出现的回报：$G^{(2)} = 2+1=3$

如果换个例子，第一次出现后奖励是 `[0, 2, -1]`，第二次出现后奖励是 `[2, -1]`，则：

- 第一次出现回报 $G^{(1)}=1$
- 第二次出现回报 $G^{(2)}=1$

First-Visit 只记第一次，Every-Visit 两次都记。直白说，First-Visit 认为“一局里重复访问同一状态，不要重复算样本”；Every-Visit 则认为“既然看到了，就都利用起来”。

需要注意，Every-Visit 在同一条轨迹里得到的多个回报通常不是独立样本，因为它们共享后续路径。但在常见条件下，随着样本足够多，它依然是一致的，也就是长期会收敛到正确值。

真实工程例子是棋类模拟器。假设你先固定一个下棋策略，不做策略改进，只想知道“某个局面大概值多少分”。你可以让程序按这个策略自我对弈很多盘，每盘结束后再把中间出现过的局面都打上最终折扣回报。这样得到的就是基于真实对局结果的价值估计。优点是直观，缺点是如果一盘棋特别长，更新会很慢。

---

## 代码实现

下面给出一个可运行的最小 Python 实现。它展示了两件事：

- 如何从一条轨迹倒序计算每个状态的回报
- 如何用在线均值更新，而不是把所有回报都存下来

```python
from collections import defaultdict

def compute_returns(trajectory, gamma=1.0):
    """
    trajectory: [(state, reward_after_state), ...]
    约定 reward_after_state 是从该状态执行动作后立刻得到的奖励。
    返回与 trajectory 等长的 G_t 列表。
    """
    G = 0.0
    returns = [0.0] * len(trajectory)
    for t in reversed(range(len(trajectory))):
        _, reward = trajectory[t]
        G = reward + gamma * G
        returns[t] = G
    return returns

def mc_policy_evaluation(episodes, gamma=1.0, first_visit=True):
    value = defaultdict(float)
    count = defaultdict(int)

    for episode in episodes:
        returns = compute_returns(episode, gamma)
        visited = set()

        for t, ((state, _), G) in enumerate(zip(episode, returns)):
            if first_visit:
                if state in visited:
                    continue
                visited.add(state)

            count[state] += 1
            # 在线均值更新，避免保存所有历史回报
            value[state] += (G - value[state]) / count[state]

    return dict(value), dict(count)

# 玩具 episode
episodes = [
    [("A", 0), ("B", 0), ("T", 1)],   # A 的回报 = 1
    [("A", -1), ("C", 0), ("T", 1)],  # A 的回报 = 0
    [("A", 0), ("A", 2), ("T", 0)],   # A 第一次回报 = 2, 第二次回报 = 2
]

returns0 = compute_returns(episodes[0], gamma=1.0)
assert returns0 == [1.0, 1.0, 1.0]

v_first, c_first = mc_policy_evaluation(episodes, gamma=1.0, first_visit=True)
v_every, c_every = mc_policy_evaluation(episodes, gamma=1.0, first_visit=False)

assert c_first["A"] == 3      # 每局最多记一次
assert c_every["A"] == 4      # 第三局里 A 出现两次
assert abs(v_first["A"] - 1.0) < 1e-9   # (1 + 0 + 2) / 3 = 1
assert abs(v_every["A"] - 1.25) < 1e-9  # (1 + 0 + 2 + 2) / 4 = 1.25
```

这段代码里有两个实现细节值得记住。

第一，`compute_returns` 用倒序累加。因为：

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

这意味着你不需要每次都从当前位置重新把后续奖励加一遍，只要从末尾往前推即可。

第二，均值更新用了：

$$
\text{new\_mean}=\text{old\_mean}+\frac{G-\text{old\_mean}}{n}
$$

这叫在线更新，白话说就是“每来一个新样本，就把均值修正一点”。这样只需要维护 `value[state]` 和 `count[state]`，不需要 `Returns[state]` 列表，内存更省。

如果把它翻译成更贴近教材的伪代码，就是：

```python
for episode in range(N):
    trajectory = sample(policy=pi)
    returns = backward_compute_G(trajectory)

    visited = set()
    for t, state in enumerate(trajectory.states):
        if first_visit and state in visited:
            continue
        visited.add(state)

        count[state] += 1
        V[state] += (returns[t] - V[state]) / count[state]
```

---

## 工程权衡与常见坑

MCPE 的问题不在“会不会错”，而在“收敛慢不慢、能不能等得起”。

先看主要权衡：

| 问题 | 表现 | 常见处理 |
| --- | --- | --- |
| 方差高 | 同一状态不同 episode 的回报波动大 | 增加样本量，控制 episode 长度 |
| 更新延迟 | 必须等整局结束才能更新 | 改用 TD 或 n-step |
| 长回合不友好 | 一局几千步时，反馈非常慢 | 截断任务或重定义 episode |
| Every-Visit 样本相关 | 同一局多次访问共享后缀路径 | 不误用独立样本假设 |

“方差高”可以从回报公式直接看出来：

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}
$$

如果 episode 很长，或者奖励分布波动很大，那么 $G_t$ 的方差也会很大。尤其当 $\gamma$ 接近 1 时，远期奖励几乎不打折，回报受长尾路径影响更明显，单次样本就会更不稳定。

真实工程例子是棋类模拟器或长流程游戏。假设一局平均 3000 步，只有结局胜负给出 $\pm1$ 奖励。那 MCPE 的问题会很直接：

- 前 2999 步都没有价值更新
- 同一个开局局面要靠很多整局结果才能估稳
- 如果策略本身很弱，大量 episode 可能都在随机游走

这会导致训练或评估吞吐量很差。工程上常见的做法不是“硬扛更多样本”，而是重新设计任务边界，比如缩短对局、分段评估，或者干脆改用 TD 方法。

几个新手常踩的坑：

1. 把 MCPE 用在 continuing task 上。
持续任务没有自然终点，完整 $G_t$ 不好定义或非常长，直接套公式通常不合适。

2. 误以为 Every-Visit 一定更好。
它样本更多，但同一局里的样本相关性更强，不代表单位样本信息量一定更高。

3. 忽略探索覆盖。
如果某个状态在策略 $\pi$ 下几乎不会访问到，那么你根本拿不到足够样本，估计也不可靠。

4. 在实现里重复正向求和。
每个时刻都重新遍历后缀会让时间复杂度变差，倒序递推更合理。

5. 把“无偏”理解成“少量样本就准确”。
无偏只说明长期平均对，不说明前几十局就稳定。

---

## 替代方案与适用边界

如果 MCPE 的主要问题是“必须等整局结束”，那么最直接的替代方案就是 TD。TD，直白说，就是“不等最终结果，先拿当前一步和下一个状态的估计值做近似更新”。

可以用“走楼梯”理解：

- MC：等你走到顶楼，再回头评价第一层台阶值不值
- TD(0)：每上一层，就先根据“下一层看起来还不错”来更新当前层价值

三种常见方法对比如下：

| 方法 | 是否要完整 episode | 是否依赖 bootstrap | 适用场景 | 主要问题 |
| --- | --- | --- | --- | --- |
| MCPE | 是 | 否 | 回合式模拟、策略调试 | 方差高，更新慢 |
| TD(0) | 否 | 是 | 持续任务、长回合任务 | 有偏，但通常更高效 |
| Q-learning | 否 | 是 | 控制问题，不只评估策略 | 目标是学最优动作价值，不是单纯评估固定策略 |

如果环境模型已知，还有另一条路：直接用贝尔曼期望方程求解。模型，白话说，就是“你知道在每个状态执行动作后会转移到哪里、概率是多少、奖励是多少”。这时可以不靠采样，直接算期望，往往比 MC 更省样本。

所以适用边界可以总结成：

- 只想评估一个固定策略，而且有模拟器、任务会结束：MCPE 合适
- 任务很长，甚至不会结束：优先考虑 TD 或 n-step
- 已知环境动态：优先考虑基于模型的动态规划
- 想学最优行为，而不是只评估当前策略：进入控制方法，如 SARSA、Q-learning

MCPE 最适合的阶段通常是强化学习入门、教学演示、简单 simulator 验证。因为它把“价值就是平均未来回报”这件事讲得最直接，不藏中间近似。

---

## 参考资料

1. GeeksforGeeks，《Monte Carlo Policy Evaluation》。
重点：给出 MCPE 的直观定义、episode 采样流程，以及“不依赖模型、不做 bootstrap”的核心特征。适合先读，用来建立整体图景。

2. Goodboychan，《Monte Carlo Policy Evaluation》。
重点：公式更完整，明确写出 $G_t$ 与 $V^\pi(s)=\mathbb{E}_\pi[G_t|s_t=s]$ 的关系，也解释了 First-Visit 和 Every-Visit 的区别。适合第二步阅读。

3. Upgrad，《Monte Carlo Reinforcement Learning》。
重点：更偏工程视角，强调高方差、更新延迟、长回合任务不友好等实际问题。适合在实现前看，避免方法选型过于理想化。

4. Stats StackExchange 上关于 First-Visit MC 收敛性的讨论。
重点：帮助理解为什么 First-Visit 与 Every-Visit 在合适条件下都能收敛，以及“样本独立性”在分析里到底扮演什么角色。适合已经理解基本流程后再读。

5. Sutton and Barto，《Reinforcement Learning: An Introduction》。
重点：这是更系统的标准教材，MC prediction、TD learning、n-step 方法的关系在书里讲得最清楚。如果要继续学 TD 和控制方法，应直接读相关章节原文。
