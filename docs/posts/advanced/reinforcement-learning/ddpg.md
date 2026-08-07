---
title: DDPG：深度确定性策略梯度
date: 2026-08-07
---

# DDPG：深度确定性策略梯度

<div class="epigraph">
<p>当动作连续到无法枚举，就让 Q 的梯度直接「指向」更好的动作。</p>
<footer>—— 改编自蒂莫西 · 利利克拉普（Timothy Lillicrap）等，2016</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Lillicrap et al. 2016 ｜ 2026-08-07</p>
</div>

## 为什么连续动作需要「Q 的梯度」

DQN 的成功建立在「离散动作 + $\max_a$」上——动作一连续，$\max_a Q(s,a)$ 变成对连续函数的优化，难做。第13章的策略梯度靠「采样」绕开了这个坑，但代价是 on-policy、样本效率低。**DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）** 走第三条路：**让演员（actor）直接输出确定性的动作 $\mu(s)$，用评论家（critic）的 $Q$ 关于动作的梯度来指导 actor 更新**——「动作空间里往 $Q$ 上升的方向走」。它把 DQN 的离线技巧（回放、目标网络）与「确定性策略梯度」结合，在连续控制上同时拿到「离策略的高样本效率」与「直接优化动作」的能力。<span class="marginnote">DDPG 是「深度 Q 学习 × 连续动作」的桥：DQN 用 $\max$ 选最优动作，DDPG 用一个「可微的 actor」代替 $\max$——动作由网络输出、优化通过 Q 的梯度。这也是「Actor-Critic」架构里 actor 的又一种训练方式（与 REINFORCE 的采样梯度并列）。</span>

## 1 确定性策略梯度：Q 的梯度指路

DDPG 的 actor 是确定性策略 $\mu(s; \boldsymbol{\theta}^\mu)$：给定状态，直接输出一个动作。它的更新用 **确定性策略梯度（deterministic policy gradient）**：

$$
\nabla_{\boldsymbol{\theta}^\mu} J \;\approx\; \mathbb{E}_{s \sim \text{Replay}}\Big[\, \nabla_a Q(s, a; \boldsymbol{\theta}^Q)\big|_{a = \mu(s)}\, \nabla_{\boldsymbol{\theta}^\mu} \mu(s; \boldsymbol{\theta}^\mu) \,\Big]
$$

读法：**「Q 关于动作的梯度」在 $a=\mu(s)$ 处评估，再乘「actor 关于参数的梯度」**——链式法则：先问「动作往哪个方向挪，$Q$ 会变大」（$\nabla_a Q$），再问「参数怎么改，动作往那个方向走」（$\nabla_\theta \mu$）。**actor 被训练成「让 $Q$ 最大的动作生成器」**。<span class="marginnote">对比随机策略梯度（第13章）：随机版用「$\ln\pi$ 的梯度 × 优势」做期望；确定性版用「$Q$ 的梯度 × $\mu$ 的梯度」——没有对动作空间的积分、没有采样方差，这是它能高效处理连续动作的原因。Silver 等 2014 年证明了它的收敛性。</span>

## 2 DQN 的装备全搬：回放、目标网络、软更新

DDPG 把 DQN 的两大稳定技巧全部继承，并做了「双网络」适配：

- **经验回放**：actor 与 critic 都从回放缓冲区采样训练——**离策略**（actor 的探索噪声让行为策略 ≠ 目标策略），数据复用、样本高效。
- **目标网络（软更新）**：actor 与 critic 各配一份目标网络 $\mu'$, $Q'$，用**软更新（soft update）**缓慢追踪：

$$
\boldsymbol{\theta}' \;\leftarrow\; \tau\,\boldsymbol{\theta} + (1-\tau)\,\boldsymbol{\theta}', \qquad \tau \ll 1 \ (\text{如 } 0.005)
$$

**软更新让目标网络「永不冻结、缓慢跟随」**——与 DQN 的「每 C 步硬复制」不同，它每步都朝在线参数挪一小点，目标稳定且平滑变化。<span class="marginnote">critic 的目标是 DQN 式：$y = r + \gamma Q'(s', \mu'(s');\boldsymbol{\theta}^{Q'})$——<strong>actor 的目标网络给出「下一动作」，critic 目标网络评估它</strong>。这套「双目标网络」让自举目标完全脱离在线参数，稳定性比 DQN 更进一步。</span>

## 3 探索：确定性策略怎么「乱试」

确定性 actor 本身不会探索（它总是输出同一个动作）。DDPG 的探索靠**在动作上加噪声**：

$$
a_t \;=\; \mu(s_t) + \mathcal{N}_t
$$

$\mathcal{N}_t$ 可以是**高斯噪声**或 **OU 噪声（Ornstein-Uhlenbeck）**——后者是**时间相关的**噪声（有「惯性」），适合物理控制任务（让动作平滑地随机游走，而不是每步独立抖动）。**行为策略是「确定性动作 + 噪声」，目标策略是纯确定性 $\mu$**——这正是离策略框架的意义：探索与学习分离。<span class="marginnote">OU 噪声的历史意义大于实际作用：DDPG 论文用它，但后来的实践（TD3、SAC）大多用简单的高斯噪声就够。真正要紧的是「<strong>探索噪声加在动作上、学的是无噪的确定性策略</strong>」这个「探索-学习分离」的设计。</span>

## 4 公式解析：actor 更新的链式法则

$$
\underbrace{\nabla_a Q(s,a)|_{a=\mu(s)}}_{\text{① Q 沿动作的梯度（方向）}} \;\cdot\; \underbrace{\nabla_{\boldsymbol{\theta}^\mu} \mu(s)}_{\text{② actor 沿参数的梯度（执行）}}
$$

- **第一步，认方向**：$\nabla_a Q$ 是「在 $s$ 选动作 $a$ 时，$Q$ 随 $a$ 的变化率」。它指向「动作往哪挪、$Q$ 涨得快」——**这就是「更好的动作在哪」的信号**。
- **第二步，认执行**：$\nabla_{\boldsymbol{\theta}^\mu}\mu(s)$ 是「参数 $\boldsymbol{\theta}^\mu$ 怎么改，动作 $\mu(s)$ 往那个方向走」。它把「动作该往哪挪」翻译成「参数该怎么改」。
- **第三步，认乘积**：两者点乘、取期望——**「让 actor 的输出动作沿 $Q$ 的上升方向滑动」**。梯度上升最大化 $Q$，而 $Q$ 又由 critic 提供——actor 的学习完全依赖 critic 的「判断力」。**critic 越准，actor 的「指路」越对**。<span class="marginnote">这个「actor 被 Q 的梯度牵着走」的设计有个隐患：如果 critic 在某处高估（第6章的最大化偏差），actor 会被牵着往「虚高的动作」走。TD3（下一课）的核心之一正是「别信单一 critic 的高估」——用双 critic 取小来防骗。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 DDPG 的 actor 可以直接套 DQN 的「硬复制目标网络」。DDPG 用**软更新**（$\tau$ 小、每步微调），不是 DQN 的「每 C 步硬复制」——软更新对「actor-critic 双网络、连续动作」的稳定性更关键。抄错更新方式，训练容易崩。

**另一个易错点**：忽略「探索噪声在训练时加、评估时不加」。DDPG 评估（测试）时用**纯确定性** $\mu(s)$，不带噪声——训练噪声是探索工具，不是策略的一部分。若评估时也加噪声，分数会被噪声污染。

**第三个易错点**：把 DDPG 当「完全离策略、无 on-policy 约束」。它确实用回放、是离策略，但 actor 的目标是「最大化 $Q$」，而 $Q$ 是**当前 critic** 的估计——critic 更新与 actor 更新相互纠缠（第11章致命三要素的「自举 + 离策略 + 函数逼近」全齐），所以 DDPG 对超参敏感、训练不稳是**结构性的**——TD3 与 SAC 正是为补这个「不稳」而来。

## 6 小结

- **DDPG**：确定性 actor + critic；actor 用**确定性策略梯度** $\nabla_\theta J = \mathbb{E}[\nabla_a Q\,\nabla_\theta \mu]$ 更新。
- **离策略**：经验回放 + 双目标网络（actor 与 critic）软更新 $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$。
- **探索**：动作加噪声（OU 或高斯），学习纯确定性策略——探索与学习分离。
- 连续动作的高样本效率来自「Q 的梯度指路」+「离策略数据复用」。
- 隐患：critic 高估会让 actor 被「骗」——TD3、SAC 为此而来。

在下一节，我们修 DDPG 的三个病灶：**TD3**——双 critic 取小、延迟更新、目标平滑，让离策略连续控制又稳又强。
