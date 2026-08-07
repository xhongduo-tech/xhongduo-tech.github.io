---
title: 策略近似及其优势
date: 2026-08-07
---

# 策略近似及其优势

<div class="epigraph">
<p>与其估出「每步值多少」，不如直接学会「每一步该做什么」。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第13章 §13.1 ｜ 2026-08-07</p>
</div>

## 为什么最后一章要「绕过价值」

前十二章的算法几乎都在学习**价值函数**，策略只是「对价值贪心/软贪心」的衍生品。第13章换了一条更直接的路线：**直接参数化策略**——用一个带参数的分布 $\pi(a|s,\boldsymbol{\theta})$ 逼近最优策略，参数 $\boldsymbol{\theta}$ 沿「性能度量」的梯度更新。这条路叫**策略梯度方法（policy gradient methods）**。它有三个价值方法给不了的杀手锏：能学**随机策略**、近似面**更平滑**、且优化目标**直接就是回报本身**。<span class="marginnote">策略梯度方法其实比价值方法更「古老」——可以追溯到 1990 年代的 REINFORCE（Williams, 1992）与更早的联想搜索。但直到深度学习的时代，它才凭借「可微参数化 + 大批量并行」成为深度 RL 的主力。</span>

## 1 策略的参数化：从软偏好到动作分布

**策略近似（policy approximation）** 用一个参数向量 $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ 表示策略，记为 $\pi(a|s,\boldsymbol{\theta})$，满足 $\sum_a \pi(a|s,\boldsymbol{\theta}) = 1$。对离散动作，最常用的是**动作偏好（action preferences）**的 softmax：

$$
\pi(a \mid s, \boldsymbol{\theta}) \;=\; \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}
$$

其中 $h(s,a,\boldsymbol{\theta})$ 是「动作 $a$ 在状态 $s$ 的偏好值」，通常是一个线性/神经网络的输出。<span class="marginnote">softmax 的两个优点：一是保证概率为正且和为 1（分布合法性自动满足）；二是「偏好差」决定选择强度——偏好最高的动作不必独占 1，其余动作保留探索概率。这为「随机策略」提供了天然的参数化舞台。</span>

**softmax 的梯度**有一个优雅的封闭形式，是全部策略梯度推导的基础：

$$
\nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s, \boldsymbol{\theta}) \;=\; \nabla h(s,a,\boldsymbol{\theta}) - \sum_b \pi(b \mid s,\boldsymbol{\theta})\, \nabla h(s,b,\boldsymbol{\theta})
$$

即「动作 $a$ 的偏好梯度」减去「所有动作偏好梯度的策略加权平均」——**当前被偏爱的动作的梯度被「去均值」了**，这正是 softmax 归一化在梯度层面的体现。

## 2 三大优势：为什么值得绕路

**优势一：能学随机策略。** 价值方法（$\varepsilon$-贪心、$\max$）本质上只能产生「近似确定性」的策略；但很多问题的最优策略**必须随机**——石头剪刀布的最优策略是均匀随机，部分可观测环境里「确定性策略」会被对手/环境利用。策略梯度可以直接输出「55% 概率出石头」这种混合策略，且**能以任意精度逼近随机策略**。<span class="marginnote">对比价值方法：Q-learning 的 $\varepsilon$-贪心是「以 1−ε 概率选最好、ε 随机」，这种「随机性均匀抹平」永远不能表达「某些次优动作该被更偏爱」的精细随机策略。策略梯度则让每个动作的偏好独立可调——随机性是「学出来的」，不是「抖出来的」。</span>

**优势二：近似面更平滑。** 价值函数的微小变化可能导致贪心策略**突变**（从选 $a_1$ 跳到选 $a_2$）——价值近似的一点点误差被贪心放大成策略的剧烈跳变。策略参数化则不同：**$\pi(a|s,\boldsymbol{\theta})$ 随 $\boldsymbol{\theta}$ 平滑变化**——参数挪一点，策略挪一点，梯度信号连续、稳定。这让「沿着梯度走」这种优化在策略空间里比在价值空间里更顺。<span class="marginnote">直觉：价值方法像「在地面上标出最高点再走」，一点噪声就能让你从最高的山尖跳到错误的谷边；策略梯度像「直接沿着山坡往上爬」，每一步都是连续的改进——对噪声的鲁棒性是天生的。</span>

**优势三：直接优化回报。** 价值方法优化的是「价值函数的拟合误差」（间接），策略梯度直接最大化**性能度量** $J(\boldsymbol{\theta})$——回合任务的期望回报、持续任务的平均回报。**优化的目标就是最终要的目标**，不存在「价值拟合得好但策略次优」的错位。

## 3 性能度量与「策略梯度」的预告

策略梯度方法需要一个标量性能 $J(\boldsymbol{\theta})$，然后做梯度上升：

$$
\boldsymbol{\theta}_{t+1} \;=\; \boldsymbol{\theta}_t + \alpha\, \widehat{\nabla J(\boldsymbol{\theta}_t)}
$$

回合任务里通常取 $J(\boldsymbol{\theta}) = v_{\pi_\boldsymbol{\theta}}(s_0)$（起始状态价值）；持续任务里取 $J = \bar r(\pi)$（平均奖励）。**核心难题是：$J$ 对 $\boldsymbol{\theta}$ 的梯度怎么无偏估计？** 答案就是下一课的主角——**策略梯度定理（policy gradient theorem）**：$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a \nabla\pi(a|s,\boldsymbol{\theta})\, q_\pi(s,a)$——**梯度里出现的不是「$\nabla q_\pi$」，而是「$\nabla \pi$ 乘 $q_\pi$」**。这个「不用对价值求梯度」的结构，让策略梯度规避了第11章函数逼近的一切发散噩梦。<span class="marginnote">策略梯度定理是这个「直接优化」承诺的兑现：它把 $\nabla J$ 表达成「策略参数的梯度 × 动作价值」的加权和——价值 $q_\pi$ 只作权重、不求导。<strong>「价值作秤、策略作针」</strong>——这是策略梯度与价值方法分工的最终格局。</span>

## 4 公式解析：softmax 的「去均值梯度」

$$
\nabla \ln \pi(a|s,\boldsymbol{\theta}) \;=\; \underbrace{\nabla h(s,a,\boldsymbol{\theta})}_{\text{该动作的偏好梯度}} - \underbrace{\sum_b \pi(b|s,\boldsymbol{\theta})\, \nabla h(s,b,\boldsymbol{\theta})}_{\text{所有偏好的策略加权平均}}
$$

- **第一步，认对数**：$\nabla\ln\pi = \nabla\pi/\pi$——除以概率是为了「按当前策略的比例」缩放，让梯度不因概率大小失真。这是**分数函数（score function）**技巧的核心。
- **第二步，认减均值**：减去「所有动作偏好梯度的加权平均」。效果是：**当前被偏爱（$\pi(b)$ 大）的动作，其梯度方向被整体压低**——防止 softmax 参数「一边倒地涨」。
- **第三步，认性质**：这个梯度的**期望为零**（$\sum_a\pi(a)\nabla\ln\pi(a)=0$，因为 $\sum_a\pi(a)=1$ 恒成立）。期望为零的项是天然的「控制变量」——第39课带基线的 REINFORCE 正是利用这种「零均值」特性削减方差。<span class="marginnote">$\sum_a\pi(a)\nabla\ln\pi(a) = \nabla\sum_a\pi(a) = \nabla 1 = 0$——这个恒等式是策略梯度方差削减（基线）的理论支点。它在第13章反复出现，值得单独记住。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为「策略梯度方法不能处理确定性最优策略」。它**能**——softmax 偏好相差悬殊时，最优动作的概率趋近 1，随机策略退化为近确定性。策略梯度的「随机能力」是**额外**的（能表达任何策略），不是「只能随机」的缺陷。

**另一个易错点**：混淆「策略参数化」与「策略评估」。参数化只是「怎么表示策略」；怎么评估改进是策略梯度定理与具体算法的事。**$\pi(a|s,\boldsymbol{\theta})$ 本身不含「该不该选 $a$」的信息**——那是 $q_\pi$ 与 $J$ 的事。

**第三个易错点**：以为「平滑近似 = 无局部最优」。策略梯度的优化面平滑、可微，但**仍非凸**——神经网络参数化下局部最优、鞍点照常存在。平滑只是「梯度好走」，不是「没有坑」；这也是 TRPO/PPO（第14篇）要管「步长别太大」的原因。

## 6 小结

- **策略近似**：$\pi(a|s,\boldsymbol{\theta})$ 直接参数化策略；离散动作常用 **softmax 动作偏好**。
- **三大优势**：能学**随机策略**、近似面**更平滑**、目标**直接是回报**。
- softmax 梯度是「偏好梯度减策略加权均值」，期望为零——方差削减的支点。
- 性能度量 $J(\boldsymbol{\theta})$：回合取起始价值、持续取平均奖励；策略做梯度上升。
- **策略梯度定理**是下一课的核心：$\nabla J \propto \sum_s\mu(s)\sum_a\nabla\pi\, q_\pi$——价值作秤、策略作针。

在下一节，我们把「直接优化回报」兑现成数学定理——**策略梯度定理**：为什么 $\nabla J$ 只需要 $\nabla\pi$ 与 $q_\pi$，而完全不需要对价值求导。
