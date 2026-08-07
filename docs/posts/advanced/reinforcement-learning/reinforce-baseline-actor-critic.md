---
title: 带基线的REINFORCE与Actor-Critic方法
date: 2026-08-07
---

# 带基线的REINFORCE与Actor-Critic方法

<div class="epigraph">
<p>减去一个「不随动作变的量」，期望纹丝不动，方差却大幅缩水——这是 RL 里最便宜的数学。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第13章 §13.4–13.5 ｜ 2026-08-07</p>
</div>

## 为什么「减去基线」不改变期望

REINFORCE 的高方差有个共同的祸根：回报 $G_t$ 的绝对值被直接当权重，导致**所有动作都被往上推**（只要回报为正）。这一课引入**基线（baseline）** $b(s)$——一个只依赖状态、不依赖动作的量——把更新权重从「$G_t$」换成「$G_t - b(S_t)$」。因为「基线 × 策略梯度的期望」恒为零（上一课末尾的恒等式），**基线不改变梯度的期望，却能削掉回报里「与动作无关的那部分方差」**。再往前走一步，用**自举的 TD 误差**代替整幕回报，就得到 **Actor-Critic 方法**——策略（actor）与价值（critic）分工协作。<span class="marginnote">基线的合法性来自那条已被用过的恒等式：$\sum_a\pi(a|s)\nabla\ln\pi(a|s) = \nabla\sum_a\pi(a|s) = \nabla 1 = 0$。任何 $b(s)$ 乘上它求和都为零——所以基线想怎么选就怎么选，只要不依赖动作。</span>

## 1 基线 REINFORCE：回报「去均值」

**带基线的 REINFORCE** 更新：

$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \gamma^t\, \big[G_t - b(S_t)\big]\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t)
$$

$b(s)$ 的选择自由，但一个好的基线应该**近似 $v_\pi(s)$**——因为「$G_t$ 相对 $v_\pi(S_t)$ 的超额部分」才真正反映「这次选 $A_t$ 比平均好还是差」。标准做法是用一个**价值近似函数** $\hat v(S_t, \mathbf w)$ 当基线，参数 $\mathbf w$ 用半梯度 TD 在线学习：

$$
\mathbf w \;\leftarrow\; \mathbf w + \alpha^\mathbf w\big[G_t - \hat v(S_t,\mathbf w)\big]\nabla \hat v(S_t,\mathbf w)
$$

于是同一套经验同时驱动两条线：**策略参数 $\boldsymbol{\theta}$（怎么选）与价值参数 $\mathbf w$（基线/基准）**。<span class="marginnote">为什么基线能削方差？$G_t - b(S_t)$ 的方差 ≈ $G_t$ 的方差（基线是常数不增加抖动），但「更新的均值」被校准到 0 附近——<strong>方差没变、期望方向更准</strong>，实际效果是「该推的动作被推、该拉的被拉」，不再全量上推浪费更新。</span>

## 2 从基线到自举：Actor-Critic 的分工

REINFORCE 用整幕回报 $G_t$——方差大、等幕末。**Actor-Critic（AC）** 把「回报」换成**一步 TD 误差**：

$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \delta_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t), \qquad \delta_t = R_{t+1} + \gamma\, \hat v(S_{t+1}, \mathbf w_t) - \hat v(S_t, \mathbf w_t)
$$

- **Critic（评论家）**：价值函数 $\hat v(S,\mathbf w)$——评估「当前策略有多好」，给出 TD 误差 $\delta_t$。
- **Actor（演员）**：策略 $\pi(a|s,\boldsymbol{\theta})$——按 critic 的评论（$\delta_t$）改进自己的表演。

**$\delta_t$ 既是 critic 的更新目标（$w \leftarrow w + \alpha^\mathbf w\delta_t\nabla\hat v$），又是 actor 的更新权重**——一个误差，两处使用。<span class="marginnote">结构上，AC 就是「策略梯度 + 自举」的结合：REINFORCE 用 $G_t$（MC 回报）、AC 用 $\delta_t$（TD 目标）。代价是 AC 有偏差（自举的旧估计误差），收益是方差骤降、可在线更新——第6章「TD vs MC」的权衡在策略梯度家族里原样重演。</span>

**n步 Actor-Critic** 是两者的折中：用 n步回报的 TD 变体做误差，n 是偏差-方差旋钮——和所有 n步方法一样。**GAE（第14篇）正是这条思路的 λ 版本**。

## 3 从 AC 到完整方法：AC 的三个关键点

Actor-Critic 看似简单，但「能跑」与「跑得稳」之间隔着三个关键设计：

1. **两时标（two time-scales）**：critic 更新通常比 actor 快（步长更大），让价值估计先稳住、再指导策略——否则两者互相追逐、震荡。
2. **协同更新**：每步经验同时更新 $w$ 与 $\boldsymbol{\theta}$——critic 的误差 $\delta_t$ 就是 actor 的信号，不需要额外「回报回算」。
3. **探索与随机性**：actor 输出的是分布（softmax），天然带探索；确定性策略（连续动作）需要额外的探索噪声（第14篇 DDPG）。<span class="marginnote">AC 是「策略梯度方法」与「价值方法」的融合：它把价值函数（价值方法的资产）与直接策略优化（策略方法的资产）接在同一组经验上。现代深度 RL 的大半江山（A2C、A3C、PPO、SAC）都是 AC 家族——理解了 AC 的「actor 走梯度、critic 做裁判」，就理解了它们的骨架。</span>

## 4 公式解析：基线为什么「零期望、削方差」

$$
\underbrace{\mathbb{E}\big[(G_t - b(S_t))\,\nabla\ln\pi(A_t|S_t)\big]}_{\text{基线版梯度}} = \underbrace{\mathbb{E}\big[G_t\,\nabla\ln\pi(A_t|S_t)\big]}_{\text{原始 REINFORCE 梯度}} - \underbrace{\mathbb{E}\big[b(S_t)\nabla\ln\pi(A_t|S_t)\big]}_{=0}
$$

- **第一步，认展开**：期望可拆成两项——第一项就是原始 REINFORCE 的梯度，第二项是「基线项」。
- **第二步，认消零**：第二项对动作求和：$\sum_a\pi(a|s)\,b(s)\nabla\ln\pi(a|s) = b(s)\sum_a\nabla\pi(a|s) = b(s)\nabla 1 = 0$。**只要 $b(s)$ 不依赖动作，这一项恒为零**——基线不引入偏差。
- **第三步，认方差**：$(G_t-b)$ 与 $G_t$ 方差相同（$b$ 是已知常数），但**期望「锚定」在 0 附近**：当 $b \approx v_\pi$，$G_t-b$ 就是「超额回报」，正负分明——更新信号从「全正推」变成「推好拉坏」，学习信号的信噪比大幅提升。<span class="marginnote">选 $b = v_\\pi$ 时，$G_t - v_\\pi(S_t)$ 是「advantage（优势）」——「这个动作比平均好多少」。下一课第14篇的 GAE 就是把这个「优势」用 λ 精确化；而「baseline 选价值函数」正是把 $\\delta_t$ 与 advantage 连通的桥梁。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为基线必须用价值函数。**任何不依赖动作的函数 $b(s)$ 都可以当基线**——价值函数只是「经验上好用」的选择。理论上 $b(s)=c$（常数）也合法，只是削减方差的效果差。

**另一个易错点**：混淆「基线」与「控制变量」。基线加在**梯度**上（期望为零）；控制变量加在**回报目标**上（期望为零）。两者是同一个思想（加零期望项）在两个层面的应用——第12章的控制变量在目标侧、本章的基线在梯度侧。

**第三个易错点**：以为 Actor-Critic 没有偏差。AC 用 TD 误差（含自举）当权重，**有自举偏差**——价值估计不准时，actor 会朝着「错的评论」改进。这就是第14篇 PPO 等要「价值网络 + 裁剪」的原因之一。**「MC 无偏高方差、AC 有偏低方差」的权衡没有消失**。

## 6 小结

- **基线 REINFORCE**：$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha\gamma^t(G_t - b(S_t))\nabla\ln\pi$——期望不变、方差更聚焦。
- **基线合法性**：$b(s)\sum_a\nabla\pi(a|s)=0$，任何不依赖动作的 $b$ 都不引入偏差。
- **Actor-Critic**：critic（$\hat v$）出 TD 误差 $\delta_t$，actor（$\pi$）按 $\delta_t$ 改进——自举换方差、引入偏差。
- **n步/λ AC**：用 n步误差或 GAE 调偏差-方差。
- 现代深度 RL（A2C、PPO、SAC）都是 AC 家族的演化——AC 是它们的通用骨架。

在下一节，我们处理连续动作：**连续动作空间的策略参数化**——高斯策略与确定性策略梯度，让策略梯度方法告别「离散 softmax」的束缚。
