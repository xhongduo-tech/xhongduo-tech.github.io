---
title: A3C/A2C：（异步）优势Actor-Critic
date: 2026-08-07
---

# A3C/A2C：（异步）优势Actor-Critic

<div class="epigraph">
<p>与其让一个智能体慢慢地等自己的经验，不如让一百个分身各自去闯，再共享心得。</p>
<footer>—— 改编自沃洛迪米尔 · 姆尼赫（Volodymyr Mnih）等，2016</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：Mnih et al. 2016 ｜ 2026-08-07</p>
</div>

## 为什么 on-policy 需要「平行宇宙」

DQN 用经验回放解决「数据相关」问题，但回放是离策略方法（Q-learning）的专利——on-policy 的策略梯度不能复用旧数据。**A3C（Asynchronous Advantage Actor-Critic，异步优势演员-评论家）** 给出另一个答案：**开一堆并行的智能体（worker），各自跑自己的环境，异步地把梯度推给一个共享的全局网络**。每个 worker 的经验来自「不同的平行宇宙」，天然去相关——**不需要回放缓冲，也能获得 DQN 式的数据多样性**。这是深度学习时代「异步并行 + 策略梯度」的开山之作。<span class="marginnote">A3C 的论文标题就叫《Asynchronous Methods for Deep Reinforcement Learning》——它的贡献一半在「A3C 算法本身」，一半在「异步框架」：Q-learning、Sarsa、DQN 等都被装进同一个异步框架实验，A3C 表现最好。异步思想后来演化为 A2C（同步版）成为主流工程范式。</span>

## 1 A3C 的架构：共享全局网络 + 并行 Worker

A3C 的结构清晰：

- **全局网络（global network）**：actor（策略 $\pi$）与 critic（价值 $V$）的参数 $\boldsymbol{\theta}$，被所有 worker 共享。
- **每个 worker**：持有一份全局参数的**本地副本**、一个自己的环境副本，独立跑 n步 Actor-Critic，**每跑 $T_{\max}$ 步就计算本地梯度、异步推送给全局网络**，然后拉取最新全局参数继续。

**异步的关键收益**：worker 们各自在环境的不同区域探索（初始状态不同、随机种子不同），**数据的时序相关性被打散**——这替代了经验回放的去相关作用，让 on-policy 方法也能高效利用大量并行数据。<span class="marginnote">「异步」的具体含义是 worker 不需要互相等待：有的 worker 推梯度时，另一个可能正在拉参数——全局网络在「边写边读」的并发中前进。这种「无锁并行」简单高效，但也让训练轨迹不完全确定（对 RL 通常无所谓）。</span>

## 2 每个 worker 学什么：n步优势 + 熵奖励

每个 worker 内部跑的是**n步 Actor-Critic**（第13章的 AC 家族），配 GAE 思想的 n步简化版。收集 $T_{\max}$ 步经验后，从后往前算 n步优势：

$$
\hat A_t \;=\; \sum_{i=0}^{n-1}\gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$

总损失是三项之和：

$$
\mathcal{L} \;=\; \underbrace{\mathbb{E}\big[-\ln\pi(a|s)\,\hat A\big]}_{\text{策略损失（actor）}} + \underbrace{c_1\,\mathbb{E}\big[(V(s) - R_{\text{target}})^2\big]}_{\text{价值损失（critic）}} - \underbrace{c_2\,\mathbb{E}\big[H(\pi(\cdot|s))\big]}_{\text{熵奖励}}
$$

- **策略损失**：把策略往「优势为正的动作」推——AC 的标准。
- **价值损失**：让 $V$ 拟合 n步回报目标——critic 的回归。
- **熵奖励**：$H(\pi)$ 是策略的熵，**鼓励探索**——熵高意味着动作分布均匀、不容易过早收敛到确定性。系数 $c_2$ 控制探索强度。<span class="marginnote">熵奖励是 A3C 稳定训练的秘密武器之一：它防止策略「过早确定」——尤其在早期、价值估计还很差时，熵奖励拉住策略别一头扎进「当前认为的好动作」。这个「熵正则」后来成为几乎所有策略梯度算法的标配。</span>

## 3 A2C：同步版，更稳更好用

A3C 的「异步」虽然简单，但训练不稳定（无锁并发、梯度方差）。**A2C（Advantage Actor-Critic，同步优势演员-评论家）** 是它的同步化：**所有 worker 跑完 $T_{\max}$ 步后同步等待，梯度聚合再统一更新全局网络**。

| 维度 | A3C | A2C |
| --- | --- | --- |
| 更新时机 | 每个 worker 异步推送 | 所有 worker 同步汇总 |
| 数据使用 | 各 worker 独立、参数常滞后 | 聚合批次、更新整齐 |
| 稳定性 | 较抖（无锁并发） | 更稳（梯度聚合平滑） |
| 工程 | 实现简单但难调 | 更符合 GPU 批量训练 |

**A2C 如今是主流选择**——因为「同步批量」更贴合 GPU 并行与标准深度学习框架，且梯度聚合后方差更小。OpenAI Baselines 默认实现是 A2C 风格。<span class="marginnote">A2C 的「同步」代价是：最慢的 worker 决定整个批次的速度（straggler 问题）。但 GPU 批量训练（把多个 worker 的经验堆成 batch 一次前向/反向）带来的吞吐收益通常远超这点等待——这也是它胜过 A3C 的工程理由。</span>

## 4 公式解析：A3C 的三合一损失

$$
\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{T}\sum_t \ln\pi(a_t|s_t)\underbrace{\hat A_t}_{\text{优势}} + c_1\frac{1}{T}\sum_t \underbrace{\big(V(s_t)-R_t^{\text{target}}\big)^2}_{\text{价值误差}} - c_2\frac{1}{T}\sum_t \underbrace{H(\pi(\cdot|s_t))}_{\text{熵}}
$$

- **第一步，认 actor 项**：$-\ln\pi(a_t|s_t)\hat A_t$——优势为正的动作被鼓励（$\ln\pi$ 增大 → 损失减小）。这是 REINFORCE 到 AC 一脉相承的策略信号，只是权重用 n步优势。
- **第二步，认 critic 项**：$(V - R^{\text{target}})^2$ 是价值回归——$R^{\text{target}}$ 是 n步回报目标，$V$ 学它。critic 为 actor 提供优势与基线。
- **第三步，认熵项**：$+c_2 H(\pi)$（损失里减熵、即最大化熵）——**鼓励策略保持随机**。它在「探索不足」时给出向上的梯度，防止策略坍缩。**三合一：actor 学怎么走、critic 学值多少、熵正则别走死**。<span class="marginnote">注意损失里三个权重系数（1、$c_1$、$c_2$）的相对大小需要调：$c_1$ 通常与 $c_2$ 同量级（0.5~1），而策略项天然含优势尺度。这也是 A2C 训练时「先让 critic 收敛、actor 才学得动」的现象来源。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 A3C 的异步是为了「更快」而非「更稳」。异步的核心收益是**数据去相关 + 探索多样性**（on-policy 的回放替代品），速度是副产品。若环境已经很快（仿真），同步 A2C 往往更好——「异步」不是越快，而是「给 on-policy 一个去相关的机制」。

**另一个易错点**：混淆「n步优势」与「GAE」。A3C 用离散的 n步优势（$\sum\gamma^i r + \gamma^nV - V$），GAE 是它的 λ 连续化——两者都是「AC 的优势估计」，但一个用固定 n、一个用 λ 加权。A3C 论文用的是前者（n 常取 5），PPO 用后者。

**第三个易错点**：忽略熵系数 $c_2$ 的平衡。$c_2$ 太大→策略被「熵」绑架、学不动；太小→探索枯竭、过早收敛。**熵奖励是把双刃剑：早期保探索、晚期会拖累收敛**——很多实现会随训练衰减 $c_2$ 或依赖它对消价值误差。

## 6 小结

- **A3C**：多 worker 异步推梯度给共享全局网络——并行 + 数据去相关 + 探索多样。
- **每个 worker**：n步优势 + 策略损失 + 价值损失 + **熵奖励**三合一。
- **A2C**：同步聚合梯度——更稳、更贴合 GPU 批量训练，如今主流。
- 熵奖励防「过早确定」，是策略梯度稳定性的标配。
- 异步/同步的取舍：异步去相关、同步更稳——环境快时选 A2C。

在下一节，我们直面策略梯度的「步长难题」：**TRPO** 用「信赖域」约束每次更新的幅度，让大模型时代的策略优化有了理论底气。
