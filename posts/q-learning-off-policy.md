## 核心结论

Q-Learning 是一种**off-policy** 的动作值学习算法。off-policy 可以先用一句白话理解：**“我执行动作时可以带点随机试错，但我学习时始终朝着‘如果以后都选最优动作会怎样’这个目标去更新。”**  
它维护一个 Q 表，表中每个元素 $Q(s,a)$ 表示“在状态 $s$ 下执行动作 $a$，未来总回报大概有多少”。

它的核心更新规则是：

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\Big]
$$

其中，当前动作可能来自 $\varepsilon$-greedy 行为策略，但更新目标永远取下一状态的最大 Q 值，所以它学习的目标始终是贪婪策略，也就是最优策略方向。

一个最直观的玩具例子是迷宫。把每个格子看成状态，把“上、下、左、右”看成动作。Q 表记录“在这个格子往某个方向走值不值”。训练时，智能体偶尔随机乱走，避免一直卡在局部路径；但每次更新时，它都假设下一步会选当前看来最好的动作，因此最终会把最短或最高回报路径学出来。

| 项目 | 行为策略 | 学习目标 |
|---|---|---|
| Q-Learning | 常用 $\varepsilon$-greedy | 始终使用 greedy 的 $\max Q$ |
| 直观含义 | 执行时允许试错 | 学习时瞄准最优 |

这就是它和 on-policy 方法最关键的区别：**行为可以不贪婪，但目标必须贪婪。**

---

## 问题定义与边界

Q-Learning 要解决的问题是：**在不知道环境转移概率的情况下，直接估计最优动作价值函数 $Q^*(s,a)$。**  
“动作价值函数”这几个字的白话意思是：**在某个状态下做某个动作，长期来看值多少分。**

它不先学环境模型，而是通过和环境交互得到样本 $(s,a,r,s')$，不断修正 Q 表。只要每个重要的状态动作对都能被访问到，Q 表就会逐渐逼近最优值。

但它有明确边界：

| 维度 | Q-Learning 的要求 | 影响 |
|---|---|---|
| 状态空间 | 最好离散且规模不大 | 状态太多时 Q 表存不下 |
| 动作空间 | 需要离散动作 | 连续动作无法直接枚举 $\max_{a'}$ |
| 采样覆盖 | 行为策略要覆盖足够多的 $(s,a)$ | 否则某些值永远学不准 |
| 环境反馈 | 依赖反复交互 | 样本太少时收敛慢 |

以 FrozenLake-v1 的 4×4 网格为例，每个格子是一个状态，动作是上下左右。目标是走到终点，掉进洞里会失败。Q-Learning 不需要提前知道地图概率，只需要不断尝试，就能估计“在第 6 个格子向右走值不值”。

这里有一个容易忽略的边界：**off-policy 不等于可以完全不探索。**  
如果行为策略只在少数动作上打转，很多 $(s,a)$ 从未访问过，那么这些位置的 Q 值只能停留在初始值，最终 greedy 策略也会被错误估值带偏。  
所以常说“行为策略必须覆盖状态动作空间”，白话就是：**该试过的路，至少得试过几次。**

---

## 核心机制与推导

Q-Learning 的更新来自 Bellman 最优方程。Bellman 方程可以白话理解为：**一个动作现在值多少，等于眼前奖励加上未来最优收益。**

最优动作价值满足：

$$
Q^*(s,a)=\mathbb{E}\left[r+\gamma \max_{a'}Q^*(s',a') \mid s,a\right]
$$

Q-Learning 用一次采样去逼近这个期望，于是得到：

$$
Q_{t+1}(s,a)=Q_t(s,a)+\alpha\Big[r+\gamma \max_{a'}Q_t(s',a')-Q_t(s,a)\Big]
$$

其中中括号里的量叫 **TD error**，时序差分误差。白话解释是：**“新观测告诉我的目标值”和“我旧估计”之间差了多少。**

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $\alpha$ | 学习率 | 新信息占多大权重 |
| $\gamma$ | 折扣因子 | 多看重未来回报 |
| $r$ | 即时奖励 | 这一步立刻拿到多少分 |
| $\max Q(s',a')$ | 下一状态最优估计 | 如果接下来选最好动作，能有多少收益 |

看一个数值例子。假设当前有：

- $Q(s,\text{右}) = 5.2$
- 即时奖励 $r=-1$
- 下一状态的最大 Q 值为 $12.5$
- $\alpha=0.1$
- $\gamma=0.9$

代入：

$$
Q \leftarrow 5.2 + 0.1\Big[(-1)+0.9\times 12.5 - 5.2\Big]
$$

$$
= 5.2 + 0.1(11.25-1-5.2)=5.2+0.505=5.705
$$

可近似写成 5.7。  
这说明虽然当前这一步奖励是负的，但因为它把你带到了一个“未来很好”的状态，所以这个动作整体价值被上调。

玩具例子仍然可以用迷宫来理解。假设从格子 A 向右走会扣 1 分，但右边那个格子离终点很近。那 Q-Learning 不会只盯着眼前这 -1，而会把“下一格以后最好路径的收益”折回来，因此学到“这个动作虽然短期亏一点，长期却值得”。

这正是它和“只看眼前奖励”的方法不同的地方。

---

## 代码实现

最小实现一般包含四部分：

1. Q 表
2. $\varepsilon$-greedy 行为策略
3. 环境交互
4. 基于 $\max Q$ 的更新

下面给一个可运行的 Python 玩具实现。环境是一个长度为 5 的一维走廊，起点在最左边，终点在最右边。每步奖励 -1，到终点奖励 10。动作只有左和右。

```python
import random

class LineWorld:
    def __init__(self, n=5):
        self.n = n
        self.start = 0
        self.goal = n - 1
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # 0: left, 1: right
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.n - 1, self.state + 1)

        done = self.state == self.goal
        reward = 10 if done else -1
        return self.state, reward, done

def choose_action(q_row, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    return 0 if q_row[0] >= q_row[1] else 1

def train_q_learning(episodes=300, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.99):
    env = LineWorld(n=5)
    q_table = [[0.0, 0.0] for _ in range(env.n)]

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = choose_action(q_table[state], epsilon)
            next_state, reward, done = env.step(action)
            best_next = max(q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            state = next_state

        epsilon = max(0.05, epsilon * decay)

    return q_table

q = train_q_learning()
# 在起点更应该向右
assert q[0][1] > q[0][0], q[0]
# 靠近终点时，向右的价值一般仍更高
assert q[3][1] > q[3][0], q[3]
print(q)
```

这段代码里有两个关键点。

第一，`choose_action` 实现的是 $\varepsilon$-greedy。白话解释是：**大多数时候选当前最优动作，少数时候故意随机。**  
这样做是为了兼顾探索和利用。探索是“多试没走过的路”，利用是“优先走当前最优路”。

第二，更新时用的是：

```python
best_next = max(q_table[next_state])
```

注意这里不是“下一步实际选了什么动作”，而是“下一状态中最好的动作值是多少”。这就是 off-policy 的核心。  
如果把它改成“按当前行为策略选出来的那个动作值”，那算法就更接近 SARSA 了。

真实工程里，比如 FrozenLake-v1，主循环也是同样结构：

```python
for episode in episodes:
    state = env.reset()
    while not done:
        action = choose_action(q_table[state], epsilon)
        next_state, reward, done = env.step(action)
        best_next = max(q_table[next_state])
        q_table[state][action] += alpha * (
            reward + gamma * best_next - q_table[state][action]
        )
        state = next_state
    epsilon *= decay
```

这类代码很短，但它已经包含了强化学习里最核心的三件事：采样、估值、改进。

---

## 工程权衡与常见坑

Q-Learning 在教材里很干净，但在工程里经常卡在三个问题：探索不足、参数震荡、状态空间爆炸。

| 常见问题 | 现象 | 常见原因 | 建议措施 |
|---|---|---|---|
| 探索不足 | 某些动作 Q 长期接近 0 | $\varepsilon$ 衰减太快 | 延长高 $\varepsilon$ 阶段，统计访问次数 |
| 学习率震荡 | Q 值来回波动 | $\alpha$ 太大 | 先减小 $\alpha$，观察 max Q 趋势 |
| 过度看远期 | 学到绕路策略 | $\gamma$ 太高且奖励设计不清晰 | 重新设计奖励，适当调低 $\gamma$ |
| 未访问状态 | greedy 后策略很差 | 行为策略覆盖不足 | 保证每个 $(s,a)$ 能被访问 |
| Q 表过大 | 内存和训练时间快速上升 | 状态维度太高 | 改用函数近似，如 DQN |

一个真实工程例子是 FrozenLake。假设你把 $\varepsilon$ 从 0.9 很快衰减到 0.1，训练前几百轮里大量格子还没被真正尝试过，尤其是靠边角的位置。结果是某些方向的 Q 值一直维持初始化附近，greedy 策略会误以为“这条路没价值”，从而始终不走，形成自我强化的偏差。

调参时通常有一个实用顺序：

| 参数 | 先看什么 | 常见经验 |
|---|---|---|
| $\alpha$ | Q 值是否振荡 | 不稳定时先减小 |
| $\gamma$ | 是否过度追求远期回报 | 导航类任务常设较高，但别盲目接近 1 |
| $\varepsilon$ | 访问覆盖是否足够 | 前期大，后期小 |
| 衰减方式 | 收敛速度与覆盖率 | 指数衰减实现简单，线性衰减更可控 |

另一个常见坑是奖励设计。Q-Learning 本身没有问题，但如果奖励函数只在终点给分，中间全是 0，很多环境会非常稀疏，训练会慢得多。  
“奖励稀疏”可以白话理解为：**大多数时候模型根本不知道自己刚才那一步到底算不算进步。**

所以在工程里，Q-Learning 是否好用，往往不只取决于公式，还取决于状态设计、奖励设计和探索策略是否合理。

---

## 替代方案与适用边界

Q-Learning 很经典，但不是所有场景都该直接用它。

最常见的替代方案是 SARSA。SARSA 是 **on-policy** 方法，白话解释是：**“我怎么做，就按我真正要做的方式来学习。”**  
它的更新目标不是 $\max_{a'}Q(s',a')$，而是下一步实际按当前策略选出的动作 $a'$ 对应的 Q 值：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma Q(s',a')-Q(s,a)\Big]
$$

代码上只差一行，但含义完全不同：

```python
# Q-Learning
best_next = max(q_table[next_state])
target = reward + gamma * best_next

# SARSA
next_action = choose_action(q_table[next_state], epsilon)
target = reward + gamma * q_table[next_state][next_action]
```

在 FrozenLake 这种带随机风险的环境里，SARSA 往往更保守。原因是它学习的是“当前带探索噪声的策略真实会发生什么”，而不是“如果以后都选最优动作会怎样”。  
如果系统更关注安全，比如机器人不能频繁冒险靠近危险区，SARSA 有时反而更合适。

当状态维度很高时，比如图像输入、连续传感器输入，Q 表就不现实了。此时通常转向 DQN。  
DQN 可以理解为：**用神经网络代替表格去近似 Q 值。**  
它仍保留 off-policy 思路，但工程上必须引入经验回放和目标网络来稳定训练，否则很容易发散。

| 方法 | 更新目标 | 策略属性 | 适用场景 |
|---|---|---|---|
| Q-Learning | $\max_{a'}Q(s',a')$ | off-policy | 离散、小规模、要逼近最优策略 |
| SARSA | $Q(s',a')$ | on-policy | 风险敏感、希望策略更保守 |
| DQN | 网络预测的 $\max Q$ | off-policy | 高维状态、无法维护 Q 表 |

所以边界可以总结为：

- 状态和动作都小而离散，优先考虑 Q-Learning
- 希望策略更贴近当前行为风险，考虑 SARSA
- 状态维度太大，Q 表存不下，转向 DQN 一类近似方法

---

## 参考资料

- Danyel Koca, “Q-Learning: Interactive Reinforcement Learning Foundation”  
  用途：适合理解更新公式和数值推导，尤其适合验证 $Q \leftarrow Q + \alpha[\cdots]$ 的计算过程。  
  链接主题：Q-Learning 公式与交互示例

- Hugging Face Deep RL Course, “Introducing Q-Learning”  
  用途：适合建立直观认识，清楚解释了 Q 表、$\varepsilon$-greedy 行为策略和 off-policy 学习目标之间的关系。  
  链接主题：Q-Learning 入门与 FrozenLake 教学

- Simplilearn, “Q-Learning Explained”  
  用途：偏工程实践，适合快速查看 Gym/FrozenLake 类型环境中的训练流程和常见调参问题。  
  链接主题：Q-Learning 工程实现与常见坑
