---
title: Double DQN与Dueling Network
date: 2026-08-07
---

# Double DQN与Dueling Network

<div class="epigraph">
<p>让一个网络挑动作，另一个网络给分——两个错误各不叠加，偏差自然变小。</p>
<footer>—— 改编自哈德尔 · 范·哈塞尔特（Hado van Hasselt）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ 深度强化学习专题 ｜ 原文：van Hasselt et al. 2016；Wang et al. 2016 ｜ 2026-08-07</p>
</div>

## 为什么 DQN 会「过度自信」

第6章我们学过**最大化偏差**：对带噪声的估计取 $\max$，结果系统性高估。DQN 的 $\max_{a'} Q(s',a';\boldsymbol{\theta}^-)$ 同样中招——目标网络冻住了参数，却没冻住「对噪声取极大」这个动作本身。**Double DQN** 把第6章的 **Double Learning** 思想移植进深度 RL：**用在线网络选动作、用目标网络给分**，把「选择」与「评估」解耦，高估被大幅削掉。同一年，**Dueling Network** 从另一个角度改进：把 $Q$ 拆成「状态价值 $V$ + 优势 $A$」，让网络在「动作不重要的状态」里学得更好。<span class="marginnote">两篇 2016 年的论文是 DQN 最著名的两个「点对点」改进：一个修「目标怎么算」（Double DQN），一个改「网络长什么样」（Dueling）。它们互不冲突、可叠加——Rainbow（下一课）把一堆这类改进打包在一起。</span>

## 1 Double DQN：选择与评估分家

DQN 的目标是「对目标网络的价值取最大」：

$$
y^{\text{DQN}} = r + \gamma \max_{a'} Q(s', a'; \boldsymbol{\theta}^-)
$$

**Double DQN** 的目标改为「在线网络选动作、目标网络给分」：

$$
y^{\text{Double}} = r + \gamma\, Q\big(s',\, \arg\max_{a'} Q(s',a';\boldsymbol{\theta}),\, \boldsymbol{\theta}^-\big)
$$

区别集中在 $Q$ 的调用方式：DQN 用同一份 $\boldsymbol{\theta}^-$ 既选又评；Double DQN 用**在线 $\boldsymbol{\theta}$ 选**（$\arg\max$）、**目标 $\boldsymbol{\theta}^-$ 评**。**如果在线网络恰好在某个动作上虚高，目标网络对它的估值未必同样虚高**——两个网络噪声独立，错误不叠加，高估被稀释。<span class="marginnote">这正是第6章 Double Q-learning 的深度版：那里是两张 Q 表轮流「选/评」，这里是「在线选 + 目标评」。核心思想一脉相承——<strong>「谁做选择」和「谁做评估」必须分开</strong>，否则对噪声取 max 的偏差被系统性放大。</span>

教材（第6章）的表格型实验已显示 Double Q-learning 把「向左」价值的高估从 ~1.4 压回 ~0.1；**Double DQN 在 Atari 上同样显著削掉高估，且在多款游戏上超过 DQN**——不仅更准，而且学得更好。

## 2 Dueling Network：价值与优势分道

**Dueling Network（决斗网络）** 改变的是网络**架构**。它把 $Q$ 的最后一层拆成两路输出：

- **状态价值流** $V(s;\boldsymbol{\theta},\boldsymbol{\beta})$：这个状态本身值多少（与动作无关）。
- **优势流** $A(s,a;\boldsymbol{\theta},\boldsymbol{\alpha})$：在 $s$ 下选 $a$ 相对平均好多少。

两路共享底层的卷积特征提取器，最后合成：

$$
Q(s,a;\boldsymbol{\theta},\boldsymbol{\alpha},\boldsymbol{\beta}) \;=\; V(s;\boldsymbol{\theta},\boldsymbol{\beta}) + A(s,a;\boldsymbol{\theta},\boldsymbol{\alpha}) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a';\boldsymbol{\theta},\boldsymbol{\alpha})
$$

**减去优势均值**是「可辨识性」的关键：只给 $Q$ 一个方程，$V$ 与 $A$ 有无穷多分解（如 $V$ 平移、$A$ 平移）；减均值把分解固定下来，让 $V$ 恰好承载「状态的平均价值」。<span class="marginnote">不加减均值项，网络可能学出「$V$ 巨大、$A$ 巨大负」之类的退化解，$Q$ 对却毫无结构。减均值让「$A$ 的和为零」成为显式约束——$V$ 与 $A$ 的分工才名副其实。</span>

## 3 为什么拆开更好：在「动作无关」的状态里学得快

Dueling 的核心洞察：**很多状态下，选什么动作几乎不影响结果**——比如赛车游戏里直道中央，方向盘微调无所谓。此时 $Q(s,a)$ 对 $a$ 的变化很小，单独学每个动作价值浪费容量；但 $V(s)$ 却有大信号（「当前处境好不好」）。

- **$V$ 流**集中学「处境价值」——在动作无关的状态里，$V$ 的梯度大、学得快。
- **$A$ 流**只学「相对优势」——只在「动作真的重要」的状态里被激活。

于是**网络把容量花在「该花的地方」**：动作无关状态主要更新 $V$，动作关键状态主要更新 $A$。实验显示 Dueling 在大量 Atari 游戏上超越 DQN，且对「动作空间大、多数动作无关」的任务尤其显著。<span class="marginnote">Dueling 的另一个实用优势：它天然输出「优势」$A(s,a)$——这正是策略梯度方法（GAE、A2C）里最想要的东西。价值与优势在同一网络里都被显式建模，为后续「值分布」「优先回放」等改进提供了干净的接口。</span>

## 4 公式解析：Double DQN 的目标「错位」

$$
\underbrace{r + \gamma \max_{a'} Q(s',a';\boldsymbol{\theta}^-)}_{\text{DQN：选与评同表}} \qquad\text{vs}\qquad \underbrace{r + \gamma\, Q\Big(s', \arg\max_{a'} Q(s',a';\boldsymbol{\theta}), \boldsymbol{\theta}^-\Big)}_{\text{Double DQN：在线选、目标评}}
$$

- **第一步，认选择**：$\arg\max_{a'}Q(s',a';\boldsymbol{\theta})$ 用**在线网络**挑「看起来最好的动作」——选择基于最新参数。
- **第二步，认评估**：挑出的动作再喂给**目标网络** $Q(\cdot,\cdot;\boldsymbol{\theta}^-)$ 给分——评估基于冻结参数。
- **第三步，认偏差削减**：若在线网络因噪声把 $a^*$ 高估，目标网络对 $a^*$ 的估值是「独立噪声」下的无偏值——不再被「取 max 本身」放大。**当两个网络的估值都接近真值时，Double DQN 的目标比 DQN 更接近 $q_*$**。<span class="marginnote">严格说，Double DQN 的高估不会完全消失（目标网络与在线网络在 $C$ 步内是同一份参数的复制，存在相关性），但它把「系统性高估」降成了「近似无偏」——实验上，Atari 里的高估从 DQN 的几十倍降到接近 0。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为 Double DQN「不再用目标网络」。它**仍然用**——目标网络给「评估分」，只是「选动作」挪到在线网络。把「选」也放进目标网络，就退回 DQN；把「评」也放进在线网络，就是第6章的「一张表自选自评」，高估更严重。

**另一个易错点**：把 Dueling 的「减均值」当成可选装饰。**减均值是让 $V/A$ 分解可辨识的必要约束**——删掉它，$V$ 与 $A$ 可以任意平移抵消，学出的结构无意义。实现里还有「用 $\max_{a'}A$ 替代均值」的变体，但「必须有中心化约束」这一点不变。

**第三个易错点**：把 Dueling 与「优势函数」混淆。Dueling 拆的是**网络结构**，不是学习目标；它输出的 $A$ 是「相对优势」的网络表达，与策略梯度里的「优势 $q-v$」概念呼应但不等同。**一个是架构、一个是统计量**。

## 6 小结

- **Double DQN**：在线选（$\arg\max Q(\cdot;\boldsymbol{\theta})$）+ 目标评（$Q(\cdot;\boldsymbol{\theta}^-)$）——选择与评估分家，高估大幅削减。
- **Dueling Network**：$Q = V + A - \text{mean}(A)$——两路输出共享底层，减均值保证分解可辨识。
- Dueling 的价值：在「动作无关」的状态里集中学 $V$，容量花在该花处。
- 两者正交可叠加，都是 Rainbow 的组件。
- 思想渊源：Double 是第6章 Double Learning 的深度版；Dueling 是「状态价值与优势分离」的架构化。

在下一节，我们把「均匀回放」升级成「优先回放」，再把一堆改进打包成 **Rainbow**——那个集 DQN 家族之大成的算法。
