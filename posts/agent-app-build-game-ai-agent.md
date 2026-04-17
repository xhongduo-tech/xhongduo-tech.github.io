## 核心结论

构建游戏 AI Agent，核心不是把“会动的脚本”做复杂，而是把**规划**和**学习**接到同一个闭环里。规划负责“先算几步再行动”，也就是基于规则模型或搜索模型做短期推演；学习负责“从成败里修正策略”，也就是根据反馈更新参数。游戏环境一旦同时存在实时性、部分可观测、对手变化和资源预算，单靠固定脚本通常会失效。

游戏里的长期目标通常写成累计回报：

$$
R_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
$$

这里的 $\gamma$ 是折扣率，表示“未来奖励值不值得今天为它付出代价”。$\gamma$ 越接近 1，智能体越重视长期布局；越小，越偏向眼前收益。

一个对初学者最有用的直觉是：游戏智能体更像会反复试错的棋手，而不是背固定连招的播放器。它会观察局势，记录哪些状态下赢率更高，再在下一轮调整行为。这个闭环通常是：

1. 感知环境
2. 评估当前动作带来的奖励
3. 更新策略
4. 再次部署并继续试错

固定脚本和“规划+学习”的差异可以先用下面这张表看清：

| 维度 | 固定脚本 | 规划+学习 |
|---|---|---|
| 反馈使用 | 基本不用历史反馈，只按条件触发 | 显式使用奖励、胜率、轨迹数据 |
| 适应性 | 敌人一变就容易失效 | 能逐步适应新地图、新对手、新节奏 |
| 资源依赖 | 线上便宜，开发期靠人工写规则 | 训练和调参更贵，线上还要做推理预算 |
| 可解释性 | 高，规则直接可读 | 中等，学习部分通常黑盒 |
| 上限 | 通常受人工规则覆盖率限制 | 在复杂博弈里上限更高 |

如果把它压缩成一句工程判断，就是：**游戏 AI Agent 的关键不是“有没有模型”，而是能否在有限预算下，把感知、规划、执行、反馈、更新连成稳定闭环。**

---

## 问题定义与边界

在形式化层面，游戏 AI Agent 常被建模为**马尔可夫博弈**。马尔可夫博弈可以理解为“多个决策者在同一个会变化的环境里同时或交替行动”的模型，记作：

$$
MG=(N,S,\{A_i\},P,\{r_i\},\gamma)
$$

各部分在游戏里的对应关系如下：

| 符号 | 含义 | 游戏中的实际对应 |
|---|---|---|
| $N$ | 智能体集合 | 多个英雄、多个单位、玩家与敌人 |
| $S$ | 状态空间 | 地图、血量、金币、迷雾、冷却、兵线 |
| $\{A_i\}$ | 每个智能体的动作空间 | 移动、攻击、释放技能、采集、撤退 |
| $P$ | 状态转移概率 | 当前动作执行后，下一帧会变成什么局面 |
| $\{r_i\}$ | 奖励函数 | 击杀、占点、存活、资源收益、胜负 |
| $\gamma$ | 折扣率 | 更看重眼前还是长线收益 |

如果每一步奖励写成：

$$
r_i^t = r_i(s,a)
$$

意思是第 $i$ 个智能体在时刻 $t$ 得到的奖励，由当前状态 $s$ 和联合动作 $a$ 决定。这里的“联合动作”不是某一个单位单独做了什么，而是**这一时刻所有参与者一起做了什么**。在多人游戏里，同一个动作在不同队友、不同敌方响应下，后果可能完全不同。

还要区分零和与非零和：

$$
\sum_i r_i = 0
$$

若成立，就是零和博弈。一个人多拿一分，另一个人就少一分，典型如棋类和严格对抗竞技。若不成立，就是非零和，典型如合作副本、塔防协作、多人资源建设。

### 为什么“边界”比“算法名”更重要

初学者常见误区是先问“该不该用强化学习”，但真正应该先问的是：

| 先问什么 | 典型选项 | 对方案选择的影响 |
|---|---|---|
| 能看到多少信息 | 完全观测 / 部分可观测 | 决定是否需要记忆、通信或 belief state |
| 决策节奏多快 | 回合制 / 实时制 | 决定是否能做搜索、搜索能做多深 |
| 有几个智能体 | 单体 / 多体协作 / 多体对抗 | 决定是否存在非平稳和协调问题 |
| 奖励是否明确 | 稠密奖励 / 稀疏奖励 | 决定训练是否稳定，是否需要奖励塑形 |
| 线上预算多紧 | 每秒数十毫秒 / 数百毫秒 | 决定能否跑树搜索、大模型或多次重规划 |

一个玩具例子：两个机器人玩塔防。机器人 A 能看到全图，机器人 B 只能看到附近三格；每回合每个机器人最多执行 2 个动作；目标是守住基地 20 回合。这里的边界立刻就清楚了：

| 约束类型 | 例子 |
|---|---|
| 信息结构 | A 完全观测，B 部分可观测 |
| 资源约束 | 每回合只能行动 2 次 |
| 时序约束 | 敌人和我方动作异步到达 |
| 目标关系 | 两者合作，不是零和 |

再把它翻成工程语言，就是下面这组输入输出：

| 工程层 | 你需要关心什么 |
|---|---|
| 感知层 | 能从游戏引擎读到哪些状态，是否有延迟或噪声 |
| 决策层 | 动作是离散还是连续，能否拆成高层与低层 |
| 学习层 | 奖励从哪里来，数据如何回放，更新多频繁 |
| 部署层 | 每帧多少毫秒，掉帧后如何降级，错误如何回滚 |

所以，“构建游戏 AI Agent”不是泛指任何自动角色，而是专指：在有限时间、算力、通信和观测约束下，能够自主感知、决策、执行、根据反馈修正行为的系统。离开这些边界谈“智能”，通常会把问题说得太宽，最后落不到实现上。

---

## 核心机制与推导

### 1. 为什么要把游戏做成马尔可夫博弈

单智能体强化学习默认环境相对稳定，但多智能体场景里，对手也在更新策略，于是环境对某个 agent 来说会变得**非平稳**。非平稳的白话解释是：你以为规则没变，其实别人的行为方式在变，所以你观测到的数据分布一直在漂移。

这也是多智能体强化学习（MARL）困难的根源。每个 agent 都想最大化自己的长期回报：

$$
R_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
$$

若零和对抗中每步奖励恒为 1，且 $\gamma=0.9$，那么：

$$
R_0 = 1+0.9+0.81+\cdots = \frac{1}{1-0.9}=10
$$

这个例子说明，哪怕每一步奖励形式不复杂，折扣率也会改变策略偏好。一个重视长期回报的 agent，会接受短期撤退去换未来优势；一个只看即时奖励的 agent，更容易“见人就冲”。

再往前推一步，价值函数通常写成：

$$
Q^\pi(s,a)=\mathbb{E}_\pi \left[\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}\mid s_t=s,a_t=a\right]
$$

它的含义很直接：**在状态 $s$ 下先做动作 $a$，然后继续按策略 $\pi$ 行动，长期来看能拿多少收益**。如果你把一个动作的即时收益看成“眼前分数”，把 $Q$ 看成“眼前分数加后续连锁反应”，为什么游戏 AI 不能只盯当前帧，就会很清楚。

### 2. 规划和学习怎么分工

规划通常负责短期决策。典型方法是 MCTS，即**蒙特卡洛树搜索**，白话说就是“从当前局面往后模拟很多种走法，估计哪条分支更值”。学习通常负责长期策略，用神经网络把高维输入映射成动作分布或价值估计。

可以把二者理解成两层：

| 模块 | 作用 | 适合解决的问题 |
|---|---|---|
| 规划 | 基于当前局面做前向搜索 | 短期战术、局部避险、分支评估 |
| 学习 | 从大规模经验中拟合策略 | 长期资源运营、复杂模式识别 |
| 反馈闭环 | 把执行结果再喂回模型 | 持续改进、适应对手变化 |

一个初学者版例子：在即时战略游戏里，微操模块负责“当前 3 秒怎么打”，战略模块负责“这一局要走速攻还是运营”。前者更像规划，后者更像学习。如果只做规划，远期目标会不稳定；如果只做学习，局部操作可能太粗糙。

更实用的理解方式是按时间尺度拆分：

| 时间尺度 | 典型问题 | 更适合谁负责 |
|---|---|---|
| 1 到 10 帧 | 躲技能、补刀、集火顺序 | 规划或规则 |
| 10 到 100 帧 | 小规模团战、路线选择 | 规划 + 学习共同作用 |
| 100 帧以上 | 资源运营、阵容成型、地图控制 | 学习主导 |

这一拆分很重要，因为许多失败案例都不是算法本身错，而是把不适合搜索的问题硬交给搜索，把不适合学习的问题硬交给学习。

### 3. 同步更新和异步更新的差异

若所有 agent 在同一时刻一起更新策略，可写成：

$$
\theta_i^{(t+1)} = \theta_i^{(t)} + \alpha \nabla J_i(\theta_i^{(t)}, \theta_{-i}^{(t)})
$$

这里 $\theta_i$ 是第 $i$ 个 agent 的参数，$\alpha$ 是学习率，$\theta_{-i}$ 表示其他 agent 的参数。

如果是异步更新，则可能变成：

$$
\theta_i^{(t+1)} = \theta_i^{(t)} + \alpha \nabla J_i(\theta_i^{(t)}, \theta_{-i}^{(\tau)})
$$

其中 $\tau < t$，表示它看到的可能是其他 agent 的旧策略。白话说，你在和“昨天的对手模型”打，但线上面对的是“今天的新对手”。这会带来训练震荡、信用分配错误和策略过拟合。

可以把同步和异步差异理解成下面这张表：

| 更新方式 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 同步更新 | 数据分布更整齐，调试更容易 | 等待成本高，吞吐低 | 小规模训练、实验验证 |
| 异步更新 | 吞吐高，工程上更易扩展 | 参数陈旧，训练更不稳定 | 大规模并行采样、在线更新 |

如果要再往前一步理解“为什么异步会抖”，可以看 TD 目标：

$$
y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

当对手策略、队友策略和自己的目标网络都在变时，$y_t$ 本身就在漂移。目标值不稳，学习过程自然容易抖。

### 4. 常见 MARL 路线

| 机制 | 核心思路 | 适用信息条件 | 资源条件 | 风险 |
|---|---|---|---|---|
| 独立学习 | 每个 agent 当环境稳定，单独学 | 观测可分、耦合不强 | 实现最便宜 | 非平稳最严重 |
| 集中式训练、分散执行 | 训练时看全局，执行时只看本地 | 训练可拿全局状态 | 训练成本中等到高 | 训练部署不一致 |
| 价值分解 | 团队奖励拆到个体上 | 合作任务明显 | 适合多单位协同 | 奖励拆分可能失真 |
| 通信机制 | agent 之间显式传消息 | 部分可观测强 | 需通信带宽和协议设计 | 延迟、欺骗、冗余 |

对于新手，一个实用判断是：

| 你遇到的问题 | 更优先考虑什么 |
|---|---|
| 对手变化太快 | 集中式训练或对手池训练 |
| 多单位经常互相抢目标 | 价值分解或任务分配机制 |
| 单位视野太小 | 记忆模块或通信机制 |
| 局部战斗经常打得很差 | 搜索、规则兜底或 imitation 预训练 |

真实工程例子是 StarCraft II。它的问题同时具备高维状态、部分可观测、长时序信用分配和多单位协同。实践里常见做法不是只靠一个大模型，而是把“局部操作策略”“高层战术规划”“资源与生产调度”拆开，再通过奖励和中间目标对齐。这类系统能成立，不是因为某个单点算法神奇，而是因为它把不同时间尺度的决策拆开处理。

---

## 代码实现

下面给出一个**可直接运行**的极简玩具实现。它不是完整 MARL 框架，但把核心流程串起来了：`observe -> infer -> plan -> act -> store -> update`。

这个例子模拟一个很小的战斗场景。状态只有三个字段：

| 状态字段 | 含义 | 取值示例 |
|---|---|---|
| `hp` | 当前生命值 | `1` 到 `3` |
| `enemy_near` | 敌人是否贴近 | `True` / `False` |
| `ammo` | 可用弹药 | `0` 到 `2` |

动作为三个离散选项：

| 动作 | 含义 | 直觉 |
|---|---|---|
| `attack` | 进攻 | 有敌人且有弹药时更合理 |
| `defend` | 防守 | 血量低或敌人逼近时更合理 |
| `farm` | 采集/补给 | 安全时补资源，为后续行动做准备 |

示例代码里，规划器先给出一个保守建议，学习器再根据历史经验决定是否覆盖这个建议。也就是说，**规划负责兜底，学习负责修正**。

```python
from collections import defaultdict, deque
import random


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Planner:
    def propose(self, state):
        # 规则规划器：血量过低先防守；能打且有弹药时进攻；否则补给
        if state["hp"] <= 1:
            return "defend"
        if state["enemy_near"] and state["ammo"] > 0:
            return "attack"
        return "farm"


class QPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.q = defaultdict(float)

    def encode(self, state, action):
        return (state["hp"], state["enemy_near"], state["ammo"], action)

    def score(self, state, action):
        return self.q[self.encode(state, action)]

    def best_action(self, state):
        return max(self.actions, key=lambda a: self.score(state, a))

    def update(self, batch, lr=0.1, gamma=0.9):
        for state, action, reward, next_state, done in batch:
            key = self.encode(state, action)
            next_best = 0.0 if done else max(
                self.score(next_state, a) for a in self.actions
            )
            target = reward + gamma * next_best
            self.q[key] += lr * (target - self.q[key])


class Agent:
    def __init__(self, actions, epsilon=0.2):
        self.actions = actions
        self.policy = QPolicy(actions)
        self.planner = Planner()
        self.epsilon = epsilon

    def act(self, state):
        plan_action = self.planner.propose(state)
        learned_action = self.policy.best_action(state)

        # 探索：少量随机动作，避免一直卡在旧策略
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        plan_score = self.policy.score(state, plan_action)
        learned_score = self.policy.score(state, learned_action)

        # 学习价值明显更高时，覆盖规划器建议
        if learned_score > plan_score + 0.3:
            return learned_action
        return plan_action


def random_state():
    return {
        "hp": random.randint(1, 3),
        "enemy_near": random.choice([True, False]),
        "ammo": random.randint(0, 2),
    }


def step_env(state, action):
    hp = state["hp"]
    enemy_near = state["enemy_near"]
    ammo = state["ammo"]

    if action == "attack":
        if enemy_near and ammo > 0:
            reward = 3
            next_state = {
                "hp": max(1, hp - random.choice([0, 1])),
                "enemy_near": random.choice([True, False]),
                "ammo": ammo - 1,
            }
        else:
            reward = -2
            next_state = {
                "hp": max(0, hp - 1),
                "enemy_near": random.choice([True, False]),
                "ammo": max(0, ammo - 1),
            }
    elif action == "defend":
        reward = 1 if enemy_near else 0
        next_state = {
            "hp": min(3, hp + 1),
            "enemy_near": random.choice([True, False]),
            "ammo": ammo,
        }
    else:  # farm
        reward = 2 if not enemy_near else -1
        next_state = {
            "hp": hp,
            "enemy_near": random.choice([True, False]),
            "ammo": min(2, ammo + 1),
        }

    done = next_state["hp"] <= 0
    return next_state, reward, done


def train(episodes=200, steps_per_episode=20):
    random.seed(7)
    actions = ["attack", "defend", "farm"]
    agent = Agent(actions)
    buffer = ReplayBuffer()
    returns = []

    for _ in range(episodes):
        state = random_state()
        total_reward = 0

        for _ in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done = step_env(state, action)
            buffer.add((state, action, reward, next_state, done))

            if len(buffer) >= 8:
                batch = buffer.sample(8)
                agent.policy.update(batch)

            total_reward += reward
            state = random_state() if done else next_state

        returns.append(total_reward)

    return agent, returns


if __name__ == "__main__":
    agent, returns = train()

    print("training finished")
    print("last_5_avg_return =", sum(returns[-5:]) / 5)

    test_state_1 = {"hp": 1, "enemy_near": True, "ammo": 1}
    test_state_2 = {"hp": 3, "enemy_near": False, "ammo": 0}

    print("state_1 ->", agent.act(test_state_1))
    print("state_2 ->", agent.act(test_state_2))
```

这段代码里有五个关键模块：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 状态采样 `random_state` | 无 | 初始状态 | 模拟环境起点 |
| 规划 `Planner.propose` | 当前状态 | 候选动作 | 给出规则层短期建议 |
| 决策 `Agent.act` | 状态、规划建议、学习价值 | 最终动作 | 在规划与学习结果之间做融合 |
| 环境推进 `step_env` | 状态、动作 | 下一状态、奖励 | 产生反馈 |
| 学习 `QPolicy.update` | 经验批次 | 更新后的价值表 | 从历史结果中修正动作价值 |

如果你是第一次接触这类代码，最重要的是看清数据流，而不是背类名：

$$
(s_t, a_t, r_t, s_{t+1}) \rightarrow \text{Replay Buffer} \rightarrow \text{Update}
$$

它表示：智能体先在状态 $s_t$ 下做动作 $a_t$，收到奖励 $r_t$，再到达新状态 $s_{t+1}$；然后把这段经验存起来，后面反复抽样训练。

如果要把 MCTS 接进来，伪代码通常长这样：

```python
def act(state):
    prior = policy_network.predict(state)   # 学习模块给先验
    plan = mcts.search(state, prior)        # 规划模块做局部搜索
    action = merge(prior, plan)             # 融合分数后输出动作
    return action
```

这个结构很常见。学习模块负责给搜索“一个较好的初始方向”，搜索模块负责在关键局面做更细的局部推演。对初学者来说，重点不是先把算法名背全，而是先理解模块边界：**网络给偏好，搜索给校正，回放给更新。**

---

## 工程权衡与常见坑

真正把游戏 AI Agent 做上线，难点通常不在“能不能跑”，而在“能不能稳定跑”。典型失败模式如下：

| 失败类型 | 现象 | 原因 | 常见防御 |
|---|---|---|---|
| 推理漂移 | 目标越打越偏，明明该守塔却跑去追残血 | 中间状态解释错、奖励设计偏 | 验证回路、目标重评估、关键状态规则兜底 |
| 执行资源失败 | 帧超时、卡顿、搜索爆炸 | 每帧预算失控、调用链过深 | 沙箱、超时、降级策略、缓存 |
| 协调安全失败 | 多个 agent 挤在同一点不动、重复操作同一目标 | 通信延迟、共享状态不同步 | checkpoint、冲突检测、重新规划 |

一个常见的初学者版坑：多名智能体收到“到 A 点集合”的指令，但因为通信延迟，每个体都认为别人还没到，于是全部持续等待，最后卡死。这本质上是分布式系统里的**死锁**，白话说就是“大家都在等别人先动”。处理方法往往不是更复杂的模型，而是简单但硬的工程规则，例如最大等待时间、冲突重试和超时后改道。

### 预算不是附属品，而是系统边界的一部分

可以用一个非常实用的预算模型约束线上资源：

$$
B_i = T_i + \lambda C_i + \mu M_i \leq B_{\max}
$$

其中 $T_i$ 表示第 $i$ 个 agent 单回合推理时间，$C_i$ 表示通信开销，$M_i$ 表示额外模型调用或搜索节点数，$\lambda,\mu$ 是把不同资源折算到统一预算的权重。白话说，不是只管“准确率”，还要给每回合决策设硬预算。

例如规定：

$$
\text{每个 agent 每回合最多 } N=3 \text{ 次规划调用}
$$

超过就强制退化到默认策略或轻量规则。这个约束很土，但工程上非常有效，因为线上系统首先要活着，其次才是变聪明。

### 奖励设计常见错法

很多项目不是模型不行，而是奖励写坏了。常见错法如下：

| 错法 | 会发生什么 | 典型修正 |
|---|---|---|
| 只奖励击杀 | agent 过度冒进，不守目标 | 增加生存、占点、团队目标奖励 |
| 只奖励最终胜利 | 训练太稀疏，几乎学不到东西 | 加入阶段性奖励或中间目标 |
| 奖励项过多且权重随意 | 目标冲突，策略摇摆 | 先保留主目标，再逐项加权验证 |
| 训练奖励和线上 KPI 不一致 | 离线高分，线上表现差 | 统一评估口径，加入真实对局指标 |

一个简单判断标准是：**奖励函数有没有把你真正想要的行为写进去，而不是只写了“看起来相关”的代理指标。**

### 协作问题为什么总在上线后爆发

真实工程例子：如果一个 MOBA 里的打野、边路和辅助都由 agent 控制，它们可能同时判断“小龙优先级高”，结果三个人都脱线去抢资源，导致防御塔丢失。这不是单个策略错误，而是团队级目标没有做信用分配。解决办法一般包括：

1. 设置团队级任务锁，避免重复领取同一战略目标
2. 用集中式价值评估训练高层协作
3. 在线执行时保留本地规则兜底，防止全体同时偏航
4. 关键节点写 checkpoint，出现异常时回滚到保守策略

如果把这部分再压缩成一句话，就是：**游戏 Agent 上线失败，很多时候不是“不会决策”，而是“多人同时决策后没人兜底”。**

---

## 替代方案与适用边界

不是所有游戏都值得上“规划+学习”的复杂系统。方案选择要看状态空间、可观测性、对手变化速度和训练预算。

| 方案 | 适用条件 | 优点 | 缺点 |
|---|---|---|---|
| 规则脚本 / 状态机 | 状态少、敌人固定、地图有限 | 开发快、可解释、线上成本低 | 覆盖外局能力弱，容易被针对 |
| 深度强化学习（DRL） | 高维状态、反馈明确、可拿训练数据 | 适合复杂模式和长时序目标 | 训练贵，稳定性和可解释性弱 |
| 博弈论推导 | 完全信息、小规模动作空间 | 理论清晰，可得均衡分析 | 状态爆炸，难直接落地大游戏 |
| 规划+学习 | 局部推演重要且长期目标复杂 | 综合性能高，适应复杂对抗 | 系统复杂，调参与预算压力大 |

初学者可以先记一个切换逻辑：

- 如果敌人行为固定、关卡数量有限，用状态机或规则脚本。
- 如果敌人会变化、状态维度很高，用 DRL。
- 如果局部每一步都很关键，而且还能模拟未来，用规划+学习。
- 如果是完全信息、状态空间不大、强调理论最优，可考虑博弈论分析。

为什么纯博弈论方法很难直接覆盖大型游戏？因为复杂度通常会随着状态和动作空间迅速膨胀，粗略看可写成：

$$
O(|S| \times |A|)
$$

若考虑多智能体联合动作，实际还会更高，常见写法接近：

$$
O\left(|S| \times \prod_{i=1}^{N}|A_i|\right)
$$

白话说，单个智能体动作数不大时问题还可算；一旦多个单位同时行动，联合动作空间会迅速爆炸。

一个简单玩具例子：敌人行为模式只有“追击”和“撤退”两种，地图只有 10 个状态，这时状态机非常合适。一个真实工程例子：开放世界战斗、资源收集、组队协同、迷雾信息同时存在时，规则脚本会迅速失控，因为你要手写的条件组合会爆炸。这时更合理的是高层用学习，局部冲突用规划，底层再留一层规则兜底。

所以“替代方案”不是谁落后谁先进的问题，而是是否匹配问题边界。能用简单方法稳定解决的场景，不应该硬上复杂系统；必须处理部分可观测、非平稳对手和长链条决策时，才需要把规划与学习接起来。

---

## 参考资料

| 资料 | 重点 | 应用场景 |
|---|---|---|
| *A Survey of Planning and Learning in Games*（MDPI, 2020） | 系统说明规划与学习为什么互补，适合建立“短期搜索 + 长期学习”的总体框架 | 想先建立全景认识时阅读 |
| *Intelligent games meeting with multi-agent deep reinforcement learning*（Springer, 2025） | 总结 MADRL 在部分可观测、非平稳、多智能体协作中的核心方法和难点 | 想深入多智能体强化学习与复杂游戏落地时阅读 |
| *AI Agent Failure Modes*（Randeep Bhatia, updated 2026-01-05） | 从工程角度整理推理漂移、资源失败、协调冲突等典型故障 | 做线上 agent 系统，需要补齐稳定性与防御设计时参考 |
| *Mastering the game of Go without human knowledge*（Nature, 2017） | 展示策略网络、价值网络与搜索结合的代表性范式 | 想理解“学习给先验，搜索做校正”时阅读 |
| *StarCraft II: A New Challenge for Reinforcement Learning*（DeepMind / Blizzard, 2017） | 解释为什么 RTS 是部分可观测、长时序、多单位协作的高难环境 | 想理解真实复杂游戏环境为什么难时阅读 |

对初学者来说，可以这样理解这几类资料的分工：MDPI 综述负责解释“规划+学习为什么要组合”；Springer 综述负责解释“多智能体强化学习在复杂游戏里怎么落地”；AlphaGo/AlphaZero 一类工作负责解释“搜索与学习如何耦合”；StarCraft II 环境论文负责解释“问题为什么这么难”；工程故障资料负责解释“系统明明能跑，为什么上线后仍会翻车”。

可直接查阅的链接如下：

- MDPI: https://www.mdpi.com/2076-3417/10/11/3920
- Springer: https://link.springer.com/article/10.1007/s10462-025-11133-8
- Randeep Bhatia: https://randeepbhatia.com/reference/agent-failure-modes
- Nature AlphaGo Zero: https://www.nature.com/articles/nature24270
- StarCraft II Environment: https://arxiv.org/abs/1708.04782
