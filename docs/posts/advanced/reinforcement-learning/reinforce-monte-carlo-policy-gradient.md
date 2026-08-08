---
title: REINFORCE：蒙特卡洛策略梯度
date: 2026-08-07
---

# REINFORCE：蒙特卡洛策略梯度

<div class="epigraph">
<p>做了什么不重要，重要的是「这件事本来有多可能发生」——用它来放大或缩小回报的教训。</p>
<footer>—— 改编自罗纳德 · 威廉姆斯（Ronald J. Williams）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第13章 §13.3 ｜ 2026-08-07</p>
</div>

## 为什么「采样一个动作」就能得到梯度

上一课的策略梯度定理里，对每个状态要对**所有动作**加权求和——动作一多就不可行。**REINFORCE** 用一个漂亮的「无偏采样」把它解开：**只采样实际做的那个动作 $A_t$，用整幕回报 $G_t$ 做权重**。因为 $A_t$ 正是按策略 $\pi$ 采样的，数学上「采样一个」的期望等于「对所有求和」。这个极简的替换，让策略梯度第一次变成「能跑起来」的算法——代价是回报 $G_t$ 的方差大得惊人。<span class="marginnote">REINFORCE 由 Ronald Williams 在 1992 年提出，名字是「REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility」的缩写。它是策略梯度家族里「最单纯」的一员：只用整幕回报，不要价值函数，不要自举。</span>

## 1 从求和到采样：一个动作的无偏替换

回顾定理的期望形式：

$$
\nabla J(\boldsymbol{\theta}) \;\propto\; \mathbb{E}_\pi\Big[\sum_a q_\pi(S_t, a)\, \nabla \ln \pi(a \mid S_t, \boldsymbol{\theta})\Big]
$$

现在把「对 $a$ 求和」换成「采样一个 $A_t \sim \pi(\cdot|S_t)$」，并用**回报 $G_t$ 代替 $q_\pi(S_t, A_t)$**：

$$
\mathbb{E}_\pi\big[G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta})\big]
$$

为什么两者期望相同？拆两层看：

- **内层（对动作）**：$G_t$ 的期望（给定 $S_t, A_t$）正是 $q_\pi(S_t, A_t)$；于是 $\mathbb{E}[G_t\nabla\ln\pi(A_t)] = \mathbb{E}[\sum_a \pi(a|S) q_\pi(S,a)\nabla\ln\pi(a|S)] = \mathbb{E}[\sum_a q_\pi(S,a)\nabla\pi(a|S)]$——**回到定理原式**。
- **外层（对状态）**：$S_t$ 按访问分布采样，与 $\mu$ 一致。

**所以「采样一个动作 × 回报」是定理梯度的无偏估计**——这是 REINFORCE 的全部数学合法性。<span class="marginnote">对比上一课的「全动作求和」：求和是无偏且低方差的（用完了所有信息），采样是<strong>同样无偏但方差更大</strong>（只用一个动作、一个回报）。REINFORCE 用方差换掉了「动作空间不可枚举」的瓶颈——这是它存在的理由。</span>

## 2 REINFORCE 更新：把回报「缩放」进策略

REINFORCE 的更新规则：

$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \gamma^t\, G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol{\theta}_t)
$$

三个部件各司其职：

- **$\nabla\ln\pi(A_t|S_t,\boldsymbol{\theta})$**：方向——「如何让 $A_t$ 在 $S_t$ 下更可能」。若回报为正，把它调得更可能；为负，调得更不可能。
- **$G_t$**：权重——「这一幕到底值多少」。回报越大，这次「推」越用力；$G_t$ 为负则反向推。
- **$\gamma^t$**：折扣——「越早的动作对总回报影响越大」。在 $J=v(s_0)$ 的设定下，离 $s_0$ 越远的动作其梯度贡献被 $\gamma^t$ 衰减。<span class="marginnote">$\gamma^t$ 的细节：它来自「$v_\\pi(s_0)$ 的梯度」在展开时沿途累积的折扣。若用平均奖励或持续式设定，这一项会消失或换成别的常数。很多实现把 $\gamma^t$ 省略（当 $\gamma$ 接近 1 时影响小），但理论上它是回合式 REINFORCE 的必要成分。</span>

**REINFORCE 算法框**（回合式）：每幕开始重置状态；对幕内每个 $t$：按 $\pi(\cdot|S_t,\boldsymbol{\theta})$ 采样 $A_t$，执行得到 $R_{t+1}, S_{t+1}$；幕末回算 $G_t = \sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$；对每个 $t$ 做参数更新。**它必须先跑完整幕，才能回算 $G_t$——纯蒙特卡洛，不自举**。

## 3 为什么方差大、以及 REINFORCE 的地位

REINFORCE 的方差来源有两层：**回报 $G_t$ 本身**是整幕随机性的累积（MC 的老毛病），而**$\nabla\ln\pi$ 又放大它**。一个回报极大的偶然轨迹会让策略参数猛跳一下——下一幕可能又偏回来。这与第5章 MC 的高方差一脉相承，只是对象从「价值」换成了「策略参数」。

尽管方差大，REINFORCE 仍占据一个不可替代的地位：

**它是最简单的「从零可用」的策略梯度**——两行更新、无价值网络、无经验回放。
**它是所有高级方法的理论起点**——带基线、Actor-Critic、PPO，本质上都是「REINFORCE 的方差手术」。
**它在策略空间上做无偏梯度**——收敛性分析最干净（在适当步长下收敛到局部最优）。<span class="marginnote">教材对 REINFORCE 的定位很务实：它几乎从不作为最终算法使用，但「REINFORCE 更新 + 方差削减」是理解整个策略梯度家族的钥匙。就像学微积分先学「用定义求导」——慢，但让你明白一切快捷法则在干什么。</span>

## 4 公式解析：$\gamma^t G_t$ 从哪来

$$
\underbrace{\gamma^t}_{\text{折扣累积}} \underbrace{G_t}_{\text{整幕回报}} \underbrace{\nabla\ln\pi(A_t|S_t,\boldsymbol{\theta})}_{\text{策略方向的梯度}}
$$

- **第一步，认回报**：$G_t = \sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$ 是「从 $t$ 到幕末的折扣回报」——它度量「在 $S_t$ 选 $A_t$ 之后，实际获得了多少」。作为 $q_\pi(S_t,A_t)$ 的无偏估计，它决定更新的**大小与符号**。
- **第二步，认梯度**：$\nabla\ln\pi(A_t|S_t)$ 是「把 $\pi$ 往『更常选 $A_t$』方向推」的梯度。回报为正→推；为负→拉。
- **第三步，认折扣**：$\gamma^t$ 是「从 $s_0$ 到 $S_t$ 的折扣累计」——因为性能 $J=v_\pi(s_0)$ 的梯度里，$S_t$ 的贡献要经 $s_0\to S_t$ 的折扣路径加权。**$\gamma^t$ 不是超参，而是从目标函数推导出的必然项**；删掉它，梯度方向就偏了。<span class="marginnote">把三件套连起来读：<strong>「回报告诉我值不值，梯度告诉我怎么调，折扣告诉我这件事离起点多近」</strong>——REINFORCE 的每次更新就是这三个信号的乘积。高方差的原因也一目了然：回报 $G_t$ 是全幕噪声的和，一抖，整个乘积就抖。</span>

## 5 易错点辨析

**辨析｜易错点：** 把 REINFORCE 当成「用 $G_t$ 当权重做最大似然」。它**不是**监督学习的最大似然——$G_t$ 是「回报」，不是「标签」；如果回报为负，更新是**把 $A_t$ 调得更不可能**（梯度下降方向），而最大似然只会一味上调。**「回报有正有负，更新有推有拉」**——这是策略梯度与监督学习的本质区别。

**另一个易错点**：忘记「回报 $G_t$ 必须相对 0 有意义」。REINFORCE 用「$G_t$ 的绝对值」定更新力度：所有奖励为正时，即使「次优动作」也会被上调（因为 $G_t>0$）。这导致更新「全向上推、但好的推得更猛」——能学，但方差浪费。下一课的**基线**正是为了「把权重中心移到 0」而引入。

**第三个易错点**：在持续性任务里直接套回合式 REINFORCE。回合式用 $J=v(s_0)$ 与 $\gamma^t G_t$；持续性任务要用平均奖励设定的变体（回报换差分回报 $G_t-\bar r$）。**「用哪个 $J$、回报怎么定义」必须与任务设定匹配**——否则梯度方向整体偏移。

## 6 小结

- **REINFORCE**：$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha\gamma^t G_t\nabla\ln\pi(A_t|S_t,\boldsymbol{\theta})$。
- **无偏采样**：「采样一个动作×回报」的期望 = 「全动作求和」——定理的采样实现。
- **纯蒙特卡洛**：用整幕回报 $G_t$，不自举；方差大、收敛性分析干净。
- 三件套：回报定大小符号、梯度定方向、$\gamma^t$ 定距离折扣。
- 它是全部高级策略梯度（基线、Actor-Critic、PPO）的理论起点。

在下一节，我们给 REINFORCE 装上一个「零均值助手」——**基线**：减去 $b(S_t)$ 不改变期望、却大幅削减方差；再往前走一步，就得到 **Actor-Critic**。
