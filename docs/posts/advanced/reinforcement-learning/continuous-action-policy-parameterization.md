---
title: 连续动作空间的策略参数化方法
date: 2026-08-07
---

# 连续动作空间的策略参数化方法

<div class="epigraph">
<p>当动作不再能枚举，策略就只能用「分布」来写——而采样让一切照旧。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第13章 §13.6–13.7 ｜ 2026-08-07</p>
</div>

## 为什么连续动作是策略梯度的主场

机器人关节力矩、自动驾驶方向盘转角——这些动作是**连续向量**，没有「枚举所有动作」这回事。价值方法在这里集体失灵：Q-learning 的 $\max_a Q(s,a)$ 需要对连续 $a$ 做优化（很难），期望 Sarsa 的 $\sum_a\pi(a|s)$ 需要积分（不可行）。**策略梯度方法天生为连续动作而生**：它只要「从分布里采样一个动作」，方向由 $\nabla\ln\pi(a|s,\boldsymbol{\theta})$ 给出——**采样代替枚举，梯度代替最优**。这一课介绍连续动作的标准参数化——**高斯策略**，以及它的探索与方差设计。<span class="marginnote">这就是第13章把「连续动作」留到最后的原因：它是策略梯度相对价值方法最不可替代的优势所在。连续动作 + 策略梯度 + 深度网络，正是 DDPG、TD3、SAC 这些现代算法的地基。</span>

## 1 高斯策略：均值 + 方差的参数化

**高斯策略（Gaussian policy）** 把动作 $a \in \mathbb{R}^d$ 建模为以「参数化均值」为中心的正态分布：

$$
\pi(a \mid s, \boldsymbol{\theta}) \;=\; \frac{1}{\sqrt{2\pi}\, \sigma(s,\boldsymbol{\theta})} \exp\Big(-\frac{\big(a - \mu(s,\boldsymbol{\theta})\big)^2}{2\sigma(s,\boldsymbol{\theta})^2}\Big)
$$

其中 $\mu(s,\boldsymbol{\theta})$ 是「这个状态下应该做的最优动作」（常为线性 $\boldsymbol{\theta}^\top\mathbf x(s)$ 或神经网络），$\sigma$ 是「动作的随机性/探索程度」。**策略 = 均值（决策）+ 方差（探索）** 两个部件的参数化。<span class="marginnote">每个动作维度独立建模（多维正态的对角协方差）：$\pi(\mathbf a|s,\boldsymbol{\theta}) = \prod_i \frac{1}{\sqrt{2\pi}\sigma_i}e^{-(a_i-\mu_i)^2/2\sigma_i^2}$。全协方差矩阵更灵活但参数爆炸，对角假设是「独立维度」的工程折中。</span>

**高斯策略的分数函数**（即 $\nabla\ln\pi$）有干净的封闭形式：

$$
\nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s, \boldsymbol{\theta}) \;=\; \frac{a - \mu(s,\boldsymbol{\theta})}{\sigma(s,\boldsymbol{\theta})^2}\, \nabla_{\boldsymbol{\theta}} \mu(s,\boldsymbol{\theta})
$$

读法：**「实际动作比均值高多少（$a-\mu$），就往「均值应该更高」的方向推（$\nabla\mu$），力度除以方差」**——动作低于均值则反向推。这个「误差驱动」的分数函数是 REINFORCE/AC 在连续动作下的直接实现。

## 2 方差 σ 的角色：探索与收敛的双刃剑

高斯策略里，$\sigma$ 决定**探索**——σ 大、动作分散、探索充分；σ 小、动作集中、接近确定性。σ 可以由**固定常数**、**与状态相关**（$\sigma(s,\boldsymbol{\theta})$ 作为网络输出）或**参数化学习**。三种选择各有取舍：

- **固定 σ**：最简单，但无法随学习进度调整探索——学后期仍需大探索会不稳。
- **状态相关 σ**：让「不确定的状态多探索、确定的状态少探索」，表达力强但难优化。
- **学出来的 σ**：把 σ 也放进 $\boldsymbol{\theta}$ 梯度上升——它会被「回报」拉着变，但容易坍缩到 0（过早确定性）或漂移过大。<span class="marginnote">方差坍缩是连续策略训练的经典陷阱：σ 学得太小，策略过早确定，探索枯竭、错过更优动作；太大则动作噪声淹没信号。工程上的常见招数是对 σ 设下限（$\\sigma \\ge \\sigma_{\\min}$）或对 σ 的对数建模（保证正值）。</span>

**σ→0 的极限**给出**确定性策略**——$\pi$ 退化为「总是选 $\mu(s,\boldsymbol{\theta})$」。教材 §13.7 讨论的正是这一点：**随机策略可以任意逼近确定性策略，而确定性策略无法表达随机性**——所以「参数化用随机、必要时收敛到确定」是更通用的设计。这一观察通向 DDPG 的**确定性策略梯度**（第14篇）。

## 3 连续动作下的策略梯度：一切照旧，只是采样

把高斯策略塞进 Actor-Critic，更新没有任何「离散专属」的东西：

$$
\boldsymbol{\theta} \;\leftarrow\; \boldsymbol{\theta} + \alpha\, \delta_t\, \frac{a_t - \mu(s_t,\boldsymbol{\theta})}{\sigma^2}\, \nabla \mu(s_t,\boldsymbol{\theta}), \qquad \delta_t = R_{t+1} + \gamma \hat v(s_{t+1},\mathbf w) - \hat v(s_t,\mathbf w)
$$

**关键点：整个流程只需要「从高斯分布采样一个 $a_t$」与「算一次 $\nabla\mu$」**——没有 $\max$、没有积分、没有枚举。这正是连续动作的解法：**用「采样的随机性」替代「枚举的完备性」**。<span class="marginnote">对比价值方法在连续动作上的窘境：Q-learning 的 $\max_a$ 变成「对连续函数求最大值」（难），期望 Sarsa 的 $\sum_a$ 变成「高维积分」（不可行）。策略梯度用一次采样就绕开了两者——这是它不可替代性的最直接证据。</span>

**回报函数的可微性**在这里是个加分项：如果奖励/转移关于动作可微（如机器人仿真），还能用「重参数化技巧（reparameterization）」——$a = \mu(s) + \sigma\varepsilon,\ \varepsilon\sim\mathcal{N}(0,1)$——把随机性从动作里剥离，让梯度**穿透采样**直达 $\mu$。这是 SAC 等现代算法的基础。

## 4 公式解析：高斯分数函数的误差驱动

$$
\underbrace{\nabla\ln\pi(a|s,\boldsymbol{\theta})}_{\text{策略方向}} = \underbrace{\frac{a - \mu(s,\boldsymbol{\theta})}{\sigma^2}}_{\text{误差 / 方差}} \cdot \underbrace{\nabla\mu(s,\boldsymbol{\theta})}_{\text{均值梯度}}
$$

- **第一步，认误差**：$a - \mu$ 是「这次实际动作相对均值偏了多少」。若 $a > \mu$，说明「这个状态该要的动作比均值还高」——把均值推高；反之推低。
- **第二步，认归一化**：除以 $\sigma^2$ 让「方差不齐的动作」被公平对待——σ 大（探索大）时，单次偏差不可全信，梯度被稀释；σ 小（接近确定）时，偏差更有信息量，梯度被放大。
- **第三步，认方向**：$\nabla\mu$ 告诉「动哪些参数能让均值 $\mu(s)$ 变化」。乘积是「向量 × 标量」——沿「能改变均值的方向」，按「误差力度」更新。<span class="marginnote">与离散 softmax 的分数函数对比：softmax 是「偏好去均值」，高斯是「动作误差除方差」——形态不同，但都满足「$\mathbb{E}[\\nabla\\ln\\pi]=0$」的零均值性质，因此基线/Actor-Critic 的方差削减机制照样适用。</span>

## 5 易错点辨析

**辨析｜易错点：** 以为「连续动作的策略必须是高斯」。高斯只是「最常见」的选择——任何可采样的连续分布（如 beta 分布、混合高斯）都行。高斯的流行是因为**分数函数封闭可算、重参数化方便**，不是唯一合法。

**另一个易错点**：把 σ 当「超参数」而非「策略的一部分」。σ 若固定是超参；若参数化就是**学习目标的一部分**——它决定探索，也被梯度更新，两者需要权衡。**「σ 是探索还是噪声」取决于你是否把它放进 $\boldsymbol{\theta}$**。

**第三个易错点**：忽略动作的尺度/量纲。高斯策略假设动作「无量纲可加」——力矩、角度、位移的量级差别很大时，直接用一个 σ 会失衡。**通常要对动作归一化（或用每个维度独立的 σ）**——否则「厘米级的位移」与「牛·米级的力矩」被同一个方差对待，训练崩溃。

## 6 小结

- **高斯策略**：$\pi(a|s,\boldsymbol{\theta}) = \mathcal{N}\big(\mu(s,\boldsymbol{\theta}),\ \sigma^2\big)$——均值定决策、方差定探索。
- **分数函数**：$\nabla\ln\pi = \frac{a-\mu}{\sigma^2}\nabla\mu$——「误差 ÷ 方差」驱动的方向。
- **连续动作解法**：采样代替枚举、梯度代替最优——策略梯度不可替代的主场。
- **σ 的双刃剑**：探索与收敛；固定/状态相关/学习三种选择，注意坍缩。
- **σ→0 的极限**是确定性策略——通往 DDPG 确定性策略梯度的桥。

至此，Sutton & Barto 经典部分的 13 篇全部完成。接下来进入本专题的现代篇——**第14篇 深度强化学习专题**，从 DQN 开始。
