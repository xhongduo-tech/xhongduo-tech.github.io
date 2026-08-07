---
title: 集中训练分布执行（CTDE）：MADDPG
date: 2026-08-07
---

# 集中训练分布执行（CTDE）：MADDPG

<div class="epigraph">
<p>训练时看清楚所有人的底牌，出手时只按自己的牌面打——这是多智能体学习的「兵法」。</p>
<footer>—— 改编自瑞安 · 洛（Ryan Lowe）等，2017</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 多智能体强化学习 ｜ 原文：Lowe et al. 2017 ｜ 2026-08-07</p>
</div>

## 为什么 critic 可以「全知」而 actor 必须「盲」

VDN/QMIX 处理「完全合作」；但很多多智能体任务是**各自回报、混合动机**（竞拍、博弈、部分协作），动作还可能**连续**。**MADDPG（Multi-Agent DDPG，多智能体深度确定性策略梯度）** 给出一个通用框架：**每个智能体一个「只见自己」的 actor（策略），配一个「见所有人」的 critic（价值）**。训练时 critic 输入所有智能体的观测与动作（全局信息 → 消除非平稳性），部署时 actor 只用自己的观测决策。这套 **CTDE 的 actor-critic 版**同时覆盖合作、竞争、混合，是 MARL 里最常用的通用算法之一。<span class="marginnote">MADDPG 的核心洞见：<strong>训练时 critic 看到「其他人的动作」，就把「别人策略在变」这个非平稳性变成了「已知条件」</strong>——环境从智能体 $i$ 的视角看「固定了」（因为其他人动作被显式输入），actor 的梯度因此稳定。<strong>「把非平稳变成输入」</strong>是它胜过「独立 DDPG」的根本。</span>

## 1 集中式 critic：把「他人」变成输入

MADDPG 为每个智能体 $i$ 训练一个 critic $Q_i$，输入**所有**智能体的观测与动作：

$$
Q_i^{\mu}(\mathbf{x}, a_1, \dots, a_n), \qquad \mathbf{x} = (o_1, \dots, o_n)
$$

$Q_i$ 估计的是「在联合动作 $(a_1,\dots,a_n)$ 下，智能体 $i$ 的价值」——**它知道所有人做什么**。这与 VDN/QMIX 的「联合 $Q_{\text{tot}}$」不同：MADDPG 每人一个 $Q_i$（自己的回报），适合「各自回报」的任务；QMIX 是共享 $Q_{\text{tot}}$，适合「完全合作」。<span class="marginnote">「每人的 critic 见所有人」的代价是<strong>可扩展性</strong>：critic 输入随智能体数线性增长。MADDPG 论文自己也承认「几十个智能体以内」可行，再大就要用「部分观测」或「注意力」近似——「全知 critic」是理论干净的理想，工程上要打折。</span>

## 2 分布式 actor：部署时只见自己

每个智能体的 actor 是确定性策略 $\mu_i(o_i)$——只用自己的观测。actor 的更新用「集中式 critic 的梯度指路」：

$$
\nabla_{\theta_i} J(\mu_i) \;=\; \mathbb{E}\Big[\nabla_{a_i} Q_i\big(\mathbf{x}, a_1,\dots,a_n\big)\big|_{a_i = \mu_i(o_i)}\, \nabla_{\theta_i} \mu_i(o_i)\Big]
$$

**这是 DDPG 的确定性策略梯度，但 critic 换成了「多智能体版」**：$\nabla_{a_i}Q_i$ 告诉「智能体 $i$ 的动作往哪挪，它的价值（给定别人动作）变大」——**别人的动作被固定输入，actor 只对自己的动作求梯度**。训练时用全局 $\mathbf{x}$ 算梯度，部署时 actor 只用 $o_i$——CTDE 的「训练作弊、部署不作弊」在 actor-critic 里如此实现。<span class="marginnote">actor 更新的关键：$\\nabla_{a_i}Q_i$ 的梯度只经过自己的动作 $a_i$（其他人的动作是常数输入）——<strong>「不替别人改动作」</strong>。这让每个智能体只优化自己的策略，而 critic 的「全局视野」保证梯度里考虑了别人的存在。</span>

## 3 目标与稳定性：多智能体的 DQN 式稳定

MADDPG 继承 DDPG 的全部稳定技巧，并做多智能体适配：

- **经验回放**：回放里存「联合转移」$(\mathbf{x}, a_1,\dots,a_n, r_1,\dots,r_n, \mathbf{x}')$——每人的回报都存。
- **目标网络**：actor 与 critic 各有目标网络（软更新 $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$）。
- **critic 目标**：

$$
y_i = r_i + \gamma\, Q_i'\big(\mathbf{x}', \mu_1'(o_1'), \dots, \mu_n'(o_n')\big)
$$

**目标里的下一动作用「所有 actor 的目标网络」生成**——每个人「想象别人会怎么做」用冻结参数，保持稳定。这套「每人一套双网络」让 MADDPG 在多智能体下也能离线稳定训练。<span class="marginnote">「对他人动作的预测」是 MADDPG 的一个隐含假设：训练时用<strong>目标 actor</strong> 生成他人动作（不是真实采样），相当于「假设他人按当前策略行动」。若他人策略剧变（多智能体的常态），这个假设会偏——但软更新的缓慢追踪让它「追得上」。这也是它比「假设他人动作可观测」的方法更通用的原因。</span>

## 4 公式解析：为什么集中式 critic 消灭非平稳

$$
\underbrace{\nabla_{\theta_i} J}_{\text{actor i 的梯度}} = \mathbb{E}_{(\mathbf{x}, a_1,\dots,a_n)\sim\text{Replay}}\Big[\underbrace{\nabla_{a_i} Q_i\big(\mathbf{x}, a_1,\dots,a_n\big)}_{\text{critic 见所有人：他人是输入}} \cdot \underbrace{\nabla_{\theta_i}\mu_i(o_i)}_{\text{actor 只见自己}}\Big]
$$

- **第一步，认全局输入**：$Q_i$ 的参数里显式含所有智能体的观测与动作——**训练时「他人的行为」不是隐藏噪声，而是可见变量**。
- **第二步，认梯度路径**：$\nabla_{a_i}Q_i$ 只对 $a_i$ 求导（他人动作是常数）——actor 只优化自己，但「梯度值」里已包含「他人会怎么做」的信息。
- **第三步，认稳定性**：独立 DDPG 里，$Q_i$ 只含自己的动作，他人的策略变化体现为「Q 的突然变化」（非平稳）；MADDPG 里他人动作显式进 $Q_i$——**非平稳被「条件化」成平稳**。部署时 actor 只用 $o_i$（回不到全局），但**策略已经在「考虑他人」的训练下学好了**。<span class="marginnote">MADDPG 是「多智能体版的 DDPG」，也是「actor-critic 版的 CTDE」。它与 VDN/QMIX 的分工：<strong>QMIX 类解决「合作共享回报」、MADDPG 类解决「各自回报 + 连续动作 + 混合动机」</strong>——两者互补，构成 MARL 的两大支柱。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 MADDPG 只适用于合作。**它通吃合作、竞争、混合**——因为每人的 critic 见所有人、各自优化自己的回报，不需要「共享回报」假设。论文里的「追捕、交流、竞争」实验覆盖三种类型。

**另一个易错点**：把「集中式 critic」与「共享 critic」混淆。MADDPG 里**每个智能体有自己的 $Q_i$**（输入全局、估计自己的回报），不是「所有人共用一个 $Q$」。共享 $Q_{\text{tot}}$ 是 VDN/QMIX 的做法（合作专属）。

**第三个易错点**：忽略「他人目标动作」来自目标网络。critic 目标里的 $a_j' = \mu_j'(o_j')$ 用**目标 actor** 生成——不是在线 actor、也不是真实的他人动作。若直接喂「真实观测到的他人动作」，会引入「他人策略突变」的噪声；用目标网络是「假设他人按冻结策略走」的稳定化。

## 6 小结

- **MADDPG**：每人「只见自己的 actor」+「见所有人的 critic」——CTDE 的 actor-critic 版。
- **集中式 critic**：$Q_i(\mathbf{x}, a_1,\dots,a_n)$ 把「他人」变成输入——非平稳被条件化。
- **分布式 actor**：$\mu_i(o_i)$ 部署时只用局部观测；梯度由集中式 critic 指路。
- 继承 DDPG 全套稳定技巧：回放、双目标网络、软更新。
- 通吃合作/竞争/混合 + 连续动作——MARL 通用框架；可扩展性受限（critic 输入随智能体数增长）。

至此，第十七篇 多智能体强化学习 3 篇全部完成。接下来进入**第十八篇 逆强化学习与层次强化学习**。
