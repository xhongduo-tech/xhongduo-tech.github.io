## 核心结论

REINFORCE 是最直接的策略梯度方法。策略梯度的意思是：不去学“这一步值多少钱”，而是直接调策略参数，让期望回报 $J(\theta)$ 变大。它的基本估计式是

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\,G_t\right]
$$

这里的 $G_t$ 是从时刻 $t$ 开始的累计回报，可以理解成“这一步之后整局最终拿了多少分”。

问题在于：$G_t$ 带着整条轨迹的随机性。环境抖动、探索动作、长回合误差累积，都会被一起乘到 $\nabla \log \pi_\theta(a_t|s_t)$ 上。结果不是“方向错了”，而是“方向噪声太大”，更新忽左忽右，样本效率很低。

解决思路是引入 baseline。baseline 是基线分数，白话讲就是“在这个状态下，正常水平大概能拿多少”。把梯度改写为

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\,\big(G_t-b(s_t)\big)\right]
$$

只要 $b(s_t)$ 与动作 $a_t$ 无关，这个改动不改变梯度期望，也就是仍然无偏；但它会把更新从“看总分”变成“看超过平均线多少”，从而显著降低方差。

玩具例子可以这样理解：你在玩投币游戏，策略决定“保守下注”还是“激进下注”。如果某一局因为运气好突然大赢，原始 REINFORCE 会把这一局里几乎所有动作都往上推。减去 baseline 后，只会奖励那些“比平均表现更好”的动作，不会让一次偶然好运把整条行为链都抬高。

---

## 问题定义与边界

我们先把问题说清楚。REINFORCE 讨论的是 on-policy 策略优化，也就是“当前策略采样，当前策略更新”。它不解决离线数据复用，也不直接处理价值函数最优性，而是研究如何无偏地估计

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}[R(\tau)]
$$

其中 $\tau$ 是轨迹，$R(\tau)$ 是一条轨迹的总回报。

它的核心边界有三条：

| 维度 | REINFORCE 的特点 | 为什么会带来方差问题 |
|---|---|---|
| 数据来源 | 必须由当前策略采样 | 样本之间相关，不能像监督学习那样稳定复用 |
| 回报定义 | 常用整段轨迹回报或 reward-to-go | 每一步都受到未来随机结果影响 |
| 更新目标 | 直接优化期望回报 | 不引入偏差，但用样本近似时波动很大 |

方差问题最明显出现在三类环境里：

| 场景 | 特征 | 影响 |
|---|---|---|
| 稀疏奖励 | 很久才得到一次正反馈 | 大部分样本梯度接近 0，少量正样本极端放大 |
| 长回合任务 | $T$ 很大 | 一个早期动作会被很远未来的随机结果污染 |
| 高噪声环境 | 同一状态下奖励波动大 | 相同动作多次采样得到的 $G_t$ 差异很大 |

一个新手容易忽略的点是：这里说“方差大”，不是说梯度平均值错了，而是说你用有限样本估出来的梯度很不稳定。比如两条长度相同的游戏录像，一条偶然得 100 分，一条只得 10 分。如果直接把这两个回报分别乘到对应的 log 概率梯度上，那么一次好运和一次普通表现就会在参数空间里拉出很不均衡的更新，训练过程容易震荡。

再看一个简单分布对比。假设同一个状态下，策略两次采样得到的回报分别是 10 和 4：

| 方法 | 梯度权重样本 | 样本均值 | 样本围绕哪个中心波动 |
|---|---|---|---|
| 无 baseline | 10, 4 | 7 | 围绕 7 |
| baseline = 7 | 3, -3 | 0 | 围绕 0 |

数值上，这两个集合的离散程度都与 9 相关，但工程上差别很大：去中心化后，优化器接收到的是“正负偏离”，而不是“全是正的大数”。这会让梯度累计、学习率缩放、批次平均更稳定。尤其在神经网络训练里，中心位置本身就会影响更新尺度和数值稳定性。

---

## 核心机制与推导

策略梯度定理的关键技巧是 log-derivative trick，也就是

$$
\nabla_\theta \pi_\theta(a|s)=\pi_\theta(a|s)\nabla_\theta \log \pi_\theta(a|s)
$$

它把“对概率分布求导”改写成“概率乘 log 概率梯度”，这样就能把导数放进期望里。最终得到 REINFORCE 形式：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)G_t\right]
$$

为什么可以减 baseline 而不改期望？看下面这一步：

$$
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)b(s_t)\right]
=
\mathbb{E}_{s_t}\left[b(s_t)\sum_{a_t}\pi_\theta(a_t|s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)\right]
$$

又因为

$$
\sum_{a}\pi_\theta(a|s)\nabla_\theta \log \pi_\theta(a|s)
=
\sum_a \nabla_\theta \pi_\theta(a|s)
=
\nabla_\theta \sum_a \pi_\theta(a|s)
=
\nabla_\theta 1
=
0
$$

所以

$$
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)b(s_t)\right]=0
$$

于是有

$$
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)G_t\right]
=
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\big(G_t-b(s_t)\big)\right]
$$

这就是“期望不变、方差下降”的数学基础。

这里的 $(G_t-b(s_t))$ 通常叫 advantage，也就是优势。优势的白话解释是：“这次结果比这个状态下的平均预期好多少或差多少。”如果 $b(s_t)\approx V^\pi(s_t)$，也就是 baseline 近似该状态的值函数，那么：

- $A_t>0$：这一步后的结果比平均好，应提高该动作概率。
- $A_t<0$：这一步后的结果比平均差，应降低该动作概率。
- $A_t\approx 0$：这一步表现接近预期，更新可以很小。

玩具例子最容易看出这个机制。假设某个状态下两次采样的回报分别为 10 和 4，baseline 取 7。则两次梯度贡献从

$$
g_1=\nabla \log \pi \cdot 10,\quad g_2=\nabla \log \pi \cdot 4
$$

变成

$$
g_1'=\nabla \log \pi \cdot 3,\quad g_2'=\nabla \log \pi \cdot (-3)
$$

现在信息变成了“一个比平均高 3，一个比平均低 3”。这比“一个是 10，一个是 4”更接近策略真正关心的量，因为策略更新不应该只看绝对总分，而应看“该动作相对预期的增益”。

再往前一步，就得到 Actor-Critic 的起点。Actor 是策略网络，白话讲就是“负责选动作的模型”；Critic 是值函数网络，白话讲就是“负责估计当前状态平均能拿多少分的模型”。当 Critic 学到 $V^\pi(s)$ 后，Actor 就不必等待整条轨迹结束才能拿到训练信号，而可以用更低噪声的优势或 TD 误差更新。

真实工程例子是机械臂抓取。一个抓取任务通常包含接近、对齐、闭合、抬起等多个阶段。若整局成功才给 1 分，失败给 0 分，那么原始 REINFORCE 会把“最终成功”这个稀疏结果乘到前面几十步甚至上百步动作上，早期动作被严重噪声污染。加入值函数 baseline 后，策略学习的是“当前这一步是否让成功概率高于平均水平”，训练会稳定得多。

---

## 代码实现

实现时最重要的顺序不是“先更新网络”，而是“先把轨迹上的统计量算清楚”。常见处理链路是：

1. 用当前策略采样一批轨迹。
2. 对每条轨迹从后往前计算 $G_t$。
3. 用状态 baseline 或 Critic 估计 $b(s_t)$。
4. 计算优势 $A_t = G_t - b(s_t)$。
5. 用 $\log \pi_\theta(a_t|s_t)\cdot A_t$ 构造策略损失。
6. 再对 Critic 做回归，让它逼近 $G_t$ 或 TD 目标。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示“reward-to-go + baseline + advantage”的数值过程。

```python
import math

def reward_to_go(rewards, gamma=1.0):
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    return list(reversed(returns))

def advantages(returns, baselines):
    assert len(returns) == len(baselines)
    return [g - b for g, b in zip(returns, baselines)]

def normalize(xs):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in xs]

# 玩具轨迹：三步后结束
rewards = [0.0, 0.0, 1.0]
returns = reward_to_go(rewards, gamma=0.9)
# 假设 critic 估计每个状态的 baseline
baselines = [0.6, 0.7, 0.8]
advs = advantages(returns, baselines)
norm_advs = normalize(advs)

# 检查 reward-to-go
assert returns == [0.81, 0.9, 1.0]
# 检查 advantage 方向
assert advs[0] > 0 and advs[1] > 0 and advs[2] > 0
# 归一化后均值接近 0
assert abs(sum(norm_advs) / len(norm_advs)) < 1e-9

print("returns:", returns)
print("advantages:", advs)
print("normalized advantages:", norm_advs)
```

如果写成训练伪代码，顺序通常如下：

```python
for each iteration:
    trajectories = collect_with_current_policy()

    all_states = []
    all_actions = []
    all_returns = []

    for traj in trajectories:
        G = compute_reward_to_go(traj.rewards, gamma)
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
        all_returns.extend(G)

    baselines = critic(all_states)          # b(s)
    advantages = all_returns - baselines    # A = G - b(s)
    advantages = normalize(advantages)      # 可选：再降方差

    policy_loss = -mean(logpi(all_states, all_actions) * advantages)
    update_actor(policy_loss)

    critic_loss = mean_square_error(critic(all_states), all_returns)
    update_critic(critic_loss)
```

可以把它看成一个很短的流程图：

| 阶段 | 输入 | 输出 | 目的 |
|---|---|---|---|
| 采样轨迹 | 当前策略 $\pi_\theta$ | states, actions, rewards | 收集 on-policy 数据 |
| 计算回报 | rewards | $G_t$ | 给每一步分配未来收益 |
| 估计 baseline | states | $b(s_t)$ | 提供平均表现参考线 |
| 计算优势 | $G_t, b(s_t)$ | $A_t$ | 去中心化，降低方差 |
| 更新 Actor | $\log\pi, A_t$ | 新策略 | 提高高优势动作概率 |
| 更新 Critic | states, $G_t$ | 新值函数 | 让 baseline 更准 |

真实工程里，常见增强包括：

- reward-to-go：不用整局总回报，而用“从当前时刻往后的回报”，减少无关过去噪声。
- 折扣因子 $\gamma$：让更远未来的奖励权重下降，符合因果，也降低方差。
- 批量更新：攒一批轨迹后再求平均，避免单条轨迹极端值主导更新。
- advantage 归一化：把一个 batch 内的优势减均值除标准差，控制数值尺度。

---

## 工程权衡与常见坑

REINFORCE 加 baseline 不难写，难的是不把“无偏”偷偷破坏掉。

| 坑 | 原因 | 规避 |
|---|---|---|
| baseline 依赖动作 | 证明里要求 $b$ 与 $a$ 无关，否则额外项不为 0 | baseline 用 $V(s)$，不要直接把动作相关项当作纯 baseline |
| 用整局总回报训练每一步 | 早期动作被很远未来噪声污染 | 改用 reward-to-go |
| $\gamma$ 设得过高 | 有效时间跨度太长，方差变大 | 根据任务回合长度调 $\gamma$，不要默认 0.999 |
| 单条轨迹直接更新 | 高方差样本主导方向 | 采用 batch 平均 |
| Critic 太弱 | baseline 不准，优势噪声仍大 | 先保证值函数拟合质量，再追求复杂策略 |
| advantage 不做尺度控制 | 不同 batch 数值范围变化大 | 归一化或裁剪 |
| 稀疏奖励下只等最终成功 | 学习信号太弱 | 设计中间奖励或用 TD 类方法 |

最典型的误区是把动作价值 $Q(s,a)$ 直接当 baseline 使用。原因不是 $Q$ 不能用，而是“纯 baseline”这件事要求它与动作无关。如果把一个依赖当前动作的量直接拿来减，就会引入偏差，梯度不再对应原目标函数的无偏估计。正确做法一般是使用：

$$
A(s,a)=Q(s,a)-V(s)
$$

这里 $V(s)$ 才是 baseline，$Q(s,a)$ 是动作相关项，二者组合成 advantage 后是合理的。

再看一个真实工程例子。在线驾驶模拟中，策略网络输出转向和油门。如果只用整局是否撞车作为最终奖励，那么一次路面扰动、传感器噪声或随机探索，都可能把前面数十步动作的梯度一并污染。工程上通常会：

- 用 reward-to-go 而不是 episode total return。
- 加一个 Critic 估计车当前姿态和速度下的 $V(s)$。
- 用 batch 收集多条并行环境轨迹。
- 对 advantage 做标准化。
- 控制 $\gamma$，避免“把 10 秒后的结果全部甩给当前一步”。

这样做的核心不是让算法“更高级”，而是让更新信号更接近因果链条。

---

## 替代方案与适用边界

如果只看概念层次，可以把几种方法理解成“同一个目标，不同噪声控制力度”。

| 方法 | 用什么训练 Actor | 方差 | 偏差 | 适用边界 |
|---|---|---|---|---|
| 原始 REINFORCE | $G_t$ | 高 | 低 | 教学、极小任务、验证公式 |
| REINFORCE + baseline | $G_t-b(s_t)$ | 中 | 低 | 简单 on-policy 任务 |
| REINFORCE + reward-to-go | 从 $t$ 开始的回报 | 中偏低 | 低 | 长回合任务 |
| Actor-Critic | TD advantage 或 $\delta_t$ | 更低 | 有少量函数逼近偏差 | 工程主流方案 |

Actor-Critic 常用的 TD 误差是

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

它的白话解释是：“这一步实际拿到的结果，比上一步预测的好还是差。”相比整条轨迹回报 $G_t$，它只看一步或几步的时序差分，噪声更局部，所以方差更小。

可以把它画成一个简短流程：

| 方法 | 更新信号路径 |
|---|---|
| REINFORCE | 采样整条轨迹 -> 算 $G_t$ -> 更新策略 |
| baseline 版 | 采样整条轨迹 -> 算 $G_t$ 与 $V(s)$ -> 算优势 -> 更新策略 |
| Actor-Critic | 每步交互 -> Critic 给 $V(s)$ 和 $V(s')$ -> 算 $\delta_t$ -> 即时更新 |

玩具例子里，如果投币游戏只有 3 步，原始 REINFORCE 还能工作；但如果变成机械臂抓取、连续控制、在线导航这类长时序问题，Actor-Critic 通常更合适，因为它不需要把“整条轨迹最终结果”全部压到单步更新上。

不过 REINFORCE 仍有明确边界价值：

- 它是理解策略梯度无偏估计的最短路径。
- baseline 的数学性质在这里最容易看清。
- 很多复杂算法，本质上都在做“更好的 advantage 估计”和“更稳定的方差控制”。

所以实践上可以这样判断：

| 任务条件 | 更合适的方法 |
|---|---|
| 教学、推导、最小实验 | 原始 REINFORCE |
| 小型离散动作任务 | REINFORCE + baseline / reward-to-go |
| 长回合、稀疏奖励、连续控制 | Actor-Critic |
| 对样本效率要求高 | Actor-Critic 及其改进版 |

---

## 参考资料

| 资料 | 简要描述 | 关注点 |
|---|---|---|
| Hugging Face Deep RL Course: The Problem of Variance in Reinforce | 面向初学者的直观解释 | 为什么 $G_t$ 导致高方差，baseline 如何去中心化 |
| Wikipedia: Policy Gradient Method | 概念总览与基本公式 | 策略梯度定理、REINFORCE 与 baseline 的标准表达 |
| Harvard: Policy Gradient Methods | 更偏数学推导与控制视角 | 从目标函数到策略梯度、Actor-Critic 的理论连接 |
| Tildes / Alice Maz: Policy Gradient & Actor-Critic | 工程实践导向的说明 | reward-to-go、TD、Actor-Critic 的直觉与实现差异 |

推荐阅读顺序：

1. 先看 Hugging Face，建立“方差来自哪里”的直觉。
2. 再看 Wikipedia，把公式和术语对齐。
3. 最后看 Harvard 版本，补齐推导与更严格的理论背景。
