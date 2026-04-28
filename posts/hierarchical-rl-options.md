## 核心结论

分层强化学习（Hierarchical Reinforcement Learning, HRL）把一个长任务拆成不同时间尺度上的决策层。白话说，就是“不让一个策略同时管战略和手脚动作”，而是让高层先决定“接下来做哪类子任务”，低层再决定“这一小段具体怎么做”。

它的核心价值不在于“模型更大”，而在于结构更合理。普通强化学习常见的问题不是动作空间不够，而是奖励来得太晚，导致早期动作几乎拿不到有效学习信号。HRL 通过引入子目标、option 或技能，把原本只在终点出现的外部奖励，变成多段更短、更可学习的过程。

可以先看总览：

| 组件 | 作用 | 白话解释 |
|---|---|---|
| 高层策略 | 选择子目标或 option | 决定下一阶段要做什么 |
| 低层策略 | 在若干步内执行 | 决定每一步具体怎么做 |
| 终止机制 | 决定何时切换子任务 | 当前阶段什么时候算结束 |
| 内在奖励 | 评估是否接近子目标 | 不等最终成功，也能学到过程 |

玩具例子是迷宫找出口。单层策略可能要走很多步，最后到出口才得到奖励；HRL 可以先学“走到走廊口”“走到楼梯边”“走到出口附近”这些中间结果。

真实工程里，仓储机器人拣货更典型。高层决定“去哪个货架”“是否先对准托盘”“何时转向出口”，低层负责避障、转向、速度控制、机械臂微调。这样的分工，本质上是在把长时序信用分配问题拆短。

---

## 问题定义与边界

先定义普通强化学习的困难。强化学习里的“信用分配”指的是最终结果应该归因给前面哪些动作。白话说，就是“到底是哪几步真正导致了成功或失败”。当任务很长、奖励很稀疏时，这个归因非常难做。

设环境状态为 $s_t$，动作是 $a_t$，策略是 $\pi(a_t|s_t)$，目标通常是最大化累计回报：

$$
J(\pi)=\mathbb{E}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
$$

问题在于，如果只有最后一步才有奖励，那么前面上百步动作的梯度信号会很弱。即使算法理论上可解，训练上也会非常慢。

HRL 适合的边界很明确：任务长、奖励稀疏、可以分解为若干阶段，并且这些阶段在不同样本之间可复用。如果任务本身很短，比如机械臂只需一步抓取；或者奖励本身很密，比如每靠近目标就加分；再或者任务没有明显层次结构，那么分层带来的额外复杂度可能不值。

对比可以直接看表：

| 场景 | 普通 RL | HRL |
|---|---|---|
| 短任务、密奖励 | 通常足够 | 可能过度设计 |
| 长任务、稀疏奖励 | 学习困难 | 更有优势 |
| 子任务可分解 | 难直接利用结构 | 适合分层建模 |
| 子策略可复用 | 利用有限 | 可以复用“到门口”“绕障”等技能 |

迷宫是一个典型边界案例。如果迷宫很小、出口奖励不算稀疏，PPO 或 DQN 可能就够了；如果迷宫大、路径长、关键节点明显，HRL 的收益才开始显现。也就是说，HRL 不是“默认更高级的 RL”，而是“当任务有层次结构时更合适的 RL”。

---

## 核心机制与推导

### 1. Options：把“一段行为”当成动作

option 可以理解为“持续执行的一段策略”，不是单步动作。它定义为：

$$
o=(I_o,\pi_o,\beta_o)
$$

其中：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| $I_o$ | 启动条件 | 什么状态下可以启用这个 option |
| $\pi_o$ | option 内策略 | 启用后每一步怎么选动作 |
| $\beta_o$ | 终止条件 | 什么时候停止这个 option |

高层策略 $\mu(o|s)$ 不是直接选原子动作，而是先选一个 option：

$$
o_t \sim \mu(o|s_t), \quad a_k \sim \pi_o(a|s_k)
$$

然后这个 option 会持续执行，直到终止条件触发：

$$
\beta_o(s_{k+1})=1
$$

这件事解决的是“高层每步都改主意”的问题。比如“走到门口”应该持续好几步，而不是每一帧都重新决定是不是还去门口。

### 2. Manager-Worker：高层给方向，低层管执行

在 manager-worker 架构里，高层不一定直接选离散 option，也可以输出一个子目标向量 $g_t$。子目标向量就是“希望状态朝哪个方向变化”的表示。白话说，它不是命令“左转 10 度”，而是命令“朝出口附近移动”。

典型形式是：

$$
g_t = \text{Manager}(h_t)
$$

$$
a_t \sim \pi_{\text{worker}}(a|s_t,g_t)
$$

这里 $h_t$ 是高层可见的表示，可能是状态编码、历史摘要或潜在空间特征。FeUdal Networks 的关键思想是，低层不只看环境状态，还要看高层目标，然后通过内在奖励学习“尽量把状态变化方向对齐目标方向”：

$$
r_t^{int} = \cos(z_{t+1}-z_t, g_t)
$$

其中 $z_t$ 是状态嵌入，$\cos$ 是余弦相似度。它衡量的是“状态变化方向”和“高层目标方向”是否一致。

这套机制解决的是：高层负责抽象方向，低层负责局部控制。比如仓储机器人里，高层发出“靠近货架 A”，低层自己决定如何避障、减速、修正角度。

### 3. HIRO 与 HAC：层间非平稳怎么处理

“非平稳”指训练过程中数据分布在变。白话说，就是高层今天面对的低层，和明天面对的低层，已经不是同一个能力水平了。这样高层存下来的旧经验会失真。

HIRO 的核心做法是 hindsight relabeling，也就是“事后重标记”。高层当时可能发出目标 $g_t$，但低层实际做出来的是另一段行为。那训练高层时，不要死信当时的目标，而是回头寻找“什么目标最能解释这段低层动作”：

$$
g_t^*=\arg\max_g \prod_{i=t}^{t+c-1}\pi_{low}(a_i|s_i,g)
$$

白话解释：不是问“当时想干什么”，而是问“从结果看，这段动作更像是在追什么目标”。

HAC 的思路更直接。如果子目标没完成，就用实际到达的状态来改写目标。这样低层学到的是可达目标，而不是总被不现实目标惩罚。它缓解的是“高层太理想化、低层永远做不到”的问题。

可以把整条信息流记成：

```text
高层决策 -> 子目标/option -> 低层执行 -> 终止或失败
       -> 经验回放 -> hindsight 重标记 -> 更新高层与低层
```

### 4. 一个玩具例子

一维位置任务，起点 $x=2$。高层每 $c=3$ 步给一次子目标，设目标是“向右移动 3 格”，即 $g=+3$。低层实际执行的状态变化是：

$$
2 \rightarrow 3 \rightarrow 4 \rightarrow 4
$$

也就是动作效果相当于 $(+1,+1,0)$。那么真实达到的位移是：

$$
g^* = x_{t+3}-x_t = 4-2 = 2
$$

于是这段经验更适合被标成“追求 $+2$ 的目标”。如果不重标记，低层会一直因为没达到 $+3$ 被惩罚；如果做了重标记，高层和低层都能从真实可达行为中学习。

---

## 代码实现

工程上最重要的不是把论文公式抄全，而是看清数据怎么流。一次高层决策，通常会展开成多步低层动作，然后再把这一段轨迹汇总回高层样本。

最小训练流程可以写成：

```text
for each episode:
    s = env.reset()

    while not done:
        g = high_level_policy(s)
        segment = []

        for t in range(c):
            a = low_level_policy(s, g)
            s_next, r_env, done = env.step(a)
            r_int = intrinsic_reward(s, s_next, g)

            store_low(s, g, a, r_int, s_next, done)
            segment.append((s, a, s_next, r_env))
            s = s_next

            if done or subgoal_reached(s, g):
                break

        g_relabel = relabel_goal(segment)
        store_high(segment[0].state, g_relabel, sum_env_reward(segment), s, done)

        update_low_level()
        update_high_level()
```

经验字段可以分开记：

| 字段 | 作用 |
|---|---|
| `state` | 当前环境状态 |
| `subgoal` | 高层下发的子目标 |
| `action` | 低层原子动作 |
| `env_reward` | 环境外部奖励 |
| `intrinsic_reward` | 子目标内在奖励 |
| `terminated` | 段落是否结束 |
| `relabeled_goal` | hindsight 重标记后的目标 |

下面给一个可运行的 Python 玩具实现，只演示“高层给目标、低层执行、事后重标记”的数据流，不依赖深度学习框架：

```python
from dataclasses import dataclass

@dataclass
class Transition:
    state: int
    goal: int
    action: int
    next_state: int
    done: bool

def low_level_policy(state: int, goal: int) -> int:
    # 简化规则：目标为正就尽量向右走，目标为负就向左走
    if goal > 0:
        return 1
    if goal < 0:
        return -1
    return 0

def env_step(state: int, action: int, max_state: int = 4) -> tuple[int, bool]:
    next_state = max(0, min(max_state, state + action))
    done = next_state == max_state
    return next_state, done

def rollout_segment(start_state: int, high_goal: int, c: int = 3):
    state = start_state
    traj = []
    for _ in range(c):
        action = low_level_policy(state, high_goal)
        next_state, done = env_step(state, action)
        traj.append(Transition(state, high_goal, action, next_state, done))
        state = next_state
        if done:
            break
    return traj

def relabel_goal(traj):
    return traj[-1].next_state - traj[0].state

traj = rollout_segment(start_state=2, high_goal=3, c=3)
states = [t.state for t in traj] + [traj[-1].next_state]
actions = [t.action for t in traj]
g_star = relabel_goal(traj)

assert states == [2, 3, 4]
assert actions == [1, 1]
assert g_star == 2

print("states:", states)
print("actions:", actions)
print("relabeled_goal:", g_star)
```

这个例子里，高层原本想让低层走 `+3`，但环境上界是 4，所以从 2 最多走到 4，真实可达位移只有 `+2`。`assert g_star == 2` 就是在验证 hindsight 重标记逻辑。

真实工程例子可以看自动驾驶园区小车。高层每 2 秒更新一次“前往路口 A”“进入泊车区”“绕开临时障碍”这类中间目标，低层每 50 毫秒控制一次转角和油门。如果只用单层策略，它既要管路线阶段切换，又要管高频稳定控制，样本效率和稳定性都会很差。

---

## 工程权衡与常见坑

HRL 的主要难点不在“是否能写出两层网络”，而在两层之间的接口设计。接口设计错了，训练会非常不稳定。

最常见的问题如下：

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 子目标尺度不对 | 低层反复失败 | 先固定 `c` 和目标范围，再调粒度 |
| 层间非平稳 | 高层训练波动大 | 用 hindsight relabeling 或 off-policy correction |
| 奖励设计过强 | 两层互相打架 | 分开记录环境奖励和内在奖励 |
| 层数过多 | 复杂但收益不明显 | 先做两层，再决定是否加深 |
| 终止条件不稳 | option 时长忽长忽短 | 监控平均持续步数和失败率 |

几个坑需要单独说清：

第一，子目标太大。比如高层总要求“跨整个房间”，但低层还不会稳定绕障，这种目标就是不可达指令。结果是高层学不到什么目标可行，低层也一直挨罚。

第二，子目标太小。比如高层每次只要求“前进 0.1 米”，那它和原子动作几乎没区别，分层优势就消失了。HRL 有意义的前提是，高层决策必须比低层动作更粗粒度。

第三，终止条件不合理。终止太短，option 退化成单步动作；终止太长，低层会在明显错误方向上持续执行，像“死锁”。工程上通常会同时设置“达到目标即终止”和“超过最大步数强制终止”。

第四，评估方式不对。只看最终回报不够，还要看高层目标达成率、低层内在奖励、平均 option 持续步数、重标记比例。否则你不知道问题出在任务分解，还是出在局部控制。

---

## 替代方案与适用边界

HRL 不是解决长任务的唯一方法。很多任务先试更直接的方法，通常更划算。

| 方法 | 适合场景 | 主要优点 | 主要缺点 |
|---|---|---|---|
| Flat RL | 任务短、奖励密 | 简单直接 | 长任务难学 |
| Reward Shaping | 目标清晰 | 训练快 | 容易引入偏差 |
| Curriculum Learning | 可逐步加难 | 降低初期难度 | 依赖课程设计 |
| Skill Discovery | 子技能可复用 | 更通用 | 训练复杂 |
| HRL | 长时序、可分解任务 | 分工明确 | 工程复杂 |

Reward shaping 是给中间行为加人工奖励。白话说，就是“人为把稀疏奖励改密”。它比 HRL 简单，但风险是把模型引向错误捷径。比如你奖励“离目标更近”，模型可能学会原地抖动骗奖励。

Curriculum learning 是课程学习，也就是“先做简单版本，再逐步加难”。它适合初始探索极难的任务，但它不直接解决结构复用问题。

Skill discovery 是技能发现，目标是自动学出可复用技能，而不是人工指定子目标。它更通用，但稳定性通常更差，实现复杂度也高。

因此，选择 HRL 的判断标准可以很实用：

1. 任务是否真的长到需要跨时间尺度决策。
2. 是否存在清晰、可复用的子任务。
3. 是否愿意为更强结构先验付出更高工程复杂度。

短任务如“机械臂走到目标点”，普通 PPO 往往够用。组合任务如“跨房间取物、避障、开门、放置”，HRL 的收益更明显，因为这些中间技能确实能跨任务复用。

---

## 参考资料

| 论文 | 作用 | 适合放在文章中的位置 |
|---|---|---|
| Sutton, Precup, Singh, *Between MDPs and Semi-MDPs* | options 原始定义 | 核心机制起点 |
| Vezhnevets et al., *FeUdal Networks* | manager-worker 思路 | 分层直觉与公式 |
| Nachum et al., *HIRO* | 目标重标记 | 层间非平稳处理 |
| Levy et al., *HAC* | 失败回放与子目标学习 | 工程策略与坑位分析 |
| Hutsebaut-Buysse et al., *Survey* | 总体综述 | 结尾扩展阅读 |

建议阅读顺序是：先读 options 理论，理解“时间抽象”是什么；再读 FeUdal Networks，建立 manager-worker 直觉；之后读 HIRO 和 HAC，理解工程上为什么必须处理层间非平稳；最后看综述，把方法放回大图景里。

1. [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning](https://doi.org/10.1016/S0004-3702(99)00052-1)
2. [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161)
3. [Data-Efficient Hierarchical Reinforcement Learning (HIRO)](https://arxiv.org/abs/1805.08296)
4. [Hierarchical Actor-Critic (HAC)](https://arxiv.org/abs/1712.00948)
5. [Hierarchical Reinforcement Learning: A Survey and Open Research Challenges](https://www.mdpi.com/2504-4990/4/1/9)
