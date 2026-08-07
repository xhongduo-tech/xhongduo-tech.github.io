---
title: 合作式多智能体：VDN与QMIX
date: 2026-08-07
---

# 合作式多智能体：VDN与QMIX

<div class="epigraph">
<p>团队的价值不是个人的简单相加——但若能让「每人做好自己」恰好成就「团队最优」，问题就妙不可言。</p>
<footer>—— 改编自彼得 · 苏内哈格（Peter Sunehag）等，2018</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 多智能体强化学习 ｜ 原文：Sunehag et al. 2018（VDN）；Rashid et al. 2018（QMIX） ｜ 2026-08-07</p>
</div>

## 为什么「每人学自己的 Q」不够

合作式 MARL 里，最简单的做法是「每个智能体独立学自己的 Q」（independent Q-learning）——但它有两个毛病：**看不到全局状态**（队友在做什么不知道）、**环境非平稳**（队友策略在变）。**VDN 与 QMIX** 走「集中训练、分布执行（CTDE）」路线：**训练时用全局信息联合优化，执行时每个智能体只看自己的观测就能决策**。核心思想是**价值分解（value decomposition）**——把联合动作价值 $Q_{\text{tot}}$ 分解成各智能体的局部 $Q_i$，让「团队最优」与「每人最优」在数学上对齐。<span class="marginnote">CTDE（Centralized Training with Decentralized Execution）是合作式 MARL 的主流范式：训练时数据共享、价值联合（能算「团队信号」），部署时每个智能体只用局部观测与自己的 Q 决策（通信开销为零）。「训练时可作弊、部署时不作弊」是它平衡「协调」与「可扩展」的巧妙折中。</span>

## 1 VDN：联合 Q = 个人 Q 之和

**VDN（Value Decomposition Network，价值分解网络）** 提出最简单的分解：**联合动作价值等于各智能体个人价值之和**：

$$
Q_{\text{tot}}(\boldsymbol{\tau}, \mathbf{u}) \;=\; \sum_{i=1}^{n} Q_i(\tau^i, u^i)
$$

- $Q_i$ 只依赖智能体 $i$ 的**局部观测历史** $\tau^i$ 与自己的动作 $u^i$——部署时只需局部信息。
- $Q_{\text{tot}}$ 用**联合状态与联合动作**的全局数据训练（DQN 式），训练时共享信息。

**关键性质**：因为分解是「和」，**全局 $\arg\max$ 自动等于「每人各自 $\arg\max$」**：

$$
\arg\max_{\mathbf{u}} Q_{\text{tot}}(\boldsymbol{\tau},\mathbf{u}) = \big(\arg\max_{u^1}Q_1(\tau^1,u^1),\ \dots,\ \arg\max_{u^n}Q_n(\tau^n,u^n)\big)
$$

**训练优化「团队」，部署执行「个人」——两者不冲突**。VDN 是「和分解」的极致简单版。<span class="marginnote">VDN 的局限藏在「和」里：<strong>「团队价值 = 个人价值相加」假设个人贡献完全独立</strong>。但真实团队常有「协同/抵消」——两人一起搬重物价值大于各自搬之和。VDN 无法表达这种「非线性团队效应」——这是 QMIX 要修的。</span>

## 2 QMIX：单调分解，保留团队效应

**QMIX（Rashid et al. 2018）** 保留「执行时可分解」的性质，但允许「个人价值如何合成团队价值」更灵活。它用一个**混合网络（mixing network）**把个人 Q 合成 $Q_{\text{tot}}$，并施加**单调性约束**：

$$
Q_{\text{tot}}(\boldsymbol{\tau}, \mathbf{u}) = f_{\text{mix}}\big(Q_1(\tau^1,u^1), \dots, Q_n(\tau^n,u^n);\ s\big), \qquad \frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0
$$

**单调性**（$\partial Q_{\text{tot}}/\partial Q_i \ge 0$）保证：**某个智能体提高自己的 Q，不会让团队 Q 变差**。这确保了「全局最优 = 每人各自最优」——执行时可分解。混合网络以**全局状态 $s$ 为输入**（产生非负权重），从而能表达「状态相关的团队效应」。<span class="marginnote">QMIX 的「单调性」实现：混合网络的权重全部由「超网络」生成并<strong>强制非负</strong>（如 ReLU 后加小常数）——非负权重保证偏导非负。<strong>「非负权重 = 单调合成」</strong>是 QMIX 数学上的点睛之笔：它允许「状态相关的非线性」，但守住「单调」这个可分解的底线。</span>

## 3 VDN vs QMIX：表达力 vs 可分解性

| 维度 | VDN | QMIX |
| --- | --- | --- |
| 分解形式 | $Q_{\text{tot}} = \sum_i Q_i$ | $Q_{\text{tot}} = f_{\text{mix}}(Q_1,\dots,Q_n;s)$ |
| 团队效应 | 无（纯独立加和） | 有（非线性混合、状态相关） |
| 可分解执行 | 严格成立 | 单调性保证成立 |
| 复杂度 | 极简 | 一个混合网络 |

**共同点**：都是 CTDE、都用 DQN 式联合训练、都保证「部署时只需局部 Q」。**QMIX 的表达力更强**（能表达协同/抵消），在多数合作基准上超过 VDN——但 VDN 的简洁性在「可解释、易实现」上仍有价值。<span class="marginnote">一句话记两者的分工：<strong>VDN 是「加法分解」（简单直接），QMIX 是「单调分解」（灵活但不失可分解）</strong>。它们共同确立的「价值分解 + CTDE」范式，是合作式 MARL 的主流骨架——后续的 QPLEX、ResQ 等都在这个框架上增强表达力。</span>

## 4 公式解析：单调性如何「锁住」可分解

$$
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \;\ge\; 0 \quad\Longrightarrow\quad \arg\max_{\mathbf{u}} Q_{\text{tot}} = \big(\arg\max_{u^1}Q_1,\ \dots,\ \arg\max_{u^n}Q_n\big)
$$

- **第一步，认偏导**：$\partial Q_{\text{tot}}/\partial Q_i \ge 0$ 表示「个人 Q 涨，团队 Q 不会跌」——单调。
- **第二步，认推理**：若 $u^i$ 是 $Q_i$ 的最大点（$Q_i(\cdot,u^i) \ge Q_i(\cdot,u)$），那么把 $u^i$ 换成 $u$ 会让 $Q_i$ 变小、进而让 $Q_{\text{tot}}$ 不增（单调性）——**所以全局最优里每人必然取自己的最优**。
- **第三步，认实现**：混合网络权重非负（由超网络生成），保证偏导非负——**「非负权重」是「单调性」的工程实现**。全局状态 $s$ 只进混合网络（不进个人 Q），所以训练用了全局信息、执行只用局部。<span class="marginnote">QMIX 的「状态相关团队效应」：混合网络的偏置与权重都依赖 $s$——同样的个人 Q，在不同全局状态下合成不同。这让「团队配合」能随「战场形势」变化，而 VDN 的固定加法做不到。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 QMIX 能表达「任意团队价值」。**不能**——它只覆盖「对每个 $Q_i$ 单调」的价值函数。若最优联合价值需要「某个智能体价值降低、团队反而更好」（非单调），QMIX 表达不了。**「单调」是可分解的代价**——想要任意价值函数，就得放弃「执行时可分解」。

**另一个易错点**：把「个人 Q」当「个人独立学」。VDN/QMIX 的个人 Q 是在**联合训练**中学的——共享梯度、共享全局状态（在混合网络里）。**「执行时独立」≠「训练时独立」**——独立训练（independent Q-learning）没有联合价值，是另一种（更弱）方法。

**第三个易错点**：忽略「观测历史」vs「完整状态」。部署时每个智能体只有**局部观测** $\tau^i$，不是全局状态 $s$——QMIX 里 $s$ 只出现在混合网络（训练侧），部署时网络用「去掉 $s$ 的分支」。**「谁用全局信息」的边界是 CTDE 的精髓，别在实现里越界**。

## 6 小结

- **VDN**：$Q_{\text{tot}} = \sum_i Q_i$——加法分解，全局最优自动等于每人最优。
- **QMIX**：$Q_{\text{tot}} = f_{\text{mix}}(Q_i;s)$ + **单调性约束**（非负权重）——更灵活但仍可分解。
- **CTDE**：训练用全局信息、执行只用局部——协调与可扩展的折中。
- 单调性是「可分解执行」的数学保证；表达力上限是「单调价值」。
- 价值分解 + CTDE 是合作式 MARL 的主流骨架。

在下一节，我们看混合动机/连续动作下的 CTDE：**MADDPG**——集中训练分布执行的演员-评论家，让「各自的回报」也能协调。
