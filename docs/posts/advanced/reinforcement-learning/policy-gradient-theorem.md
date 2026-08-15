---
title: 策略梯度定理
date: 2026-08-07
---

# 策略梯度定理

<div class="epigraph">
<p>性能的梯度，居然不需要价值的梯度——这是策略梯度方法最深刻的赠礼。</p>
<footer>—— 改编自理查德 · 萨顿（Richard S. Sutton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 强化学习 ｜ Sutton & Barto《强化学习（第2版）》 第13章 §13.2 ｜ 2026-08-07</p>
</div>

## 为什么「回报的梯度」如此干净

上一课预告了策略梯度定理：$\nabla J$ 只含「策略参数的梯度」乘「动作价值」——**完全没有「价值的梯度」$\nabla q_\pi$**。这一课把定理证出来。它的结论之干净让人惊讶：**你不需要知道「改变参数会让价值怎么变」，只需要知道「让策略多选好动作」**。这个「免价值梯度」的特性，让策略梯度方法天然避开了第11章函数逼近的收敛噩梦——因为定理本身就不依赖价值近似的可微性。<span class="marginnote">策略梯度定理是 Sutton 等 1999 年的经典结果，它把「直接优化回报」从口号变成了可证的公式。证明的关键动作是「把 $\\nabla v_\\pi$ 递归展开，让 $\\nabla q_\\pi$ 项在相邻状态下相消」——价值梯度最终被吸收进「访问频率」里。</span>

## 1 定理陈述：回合式与持续式

**回合式（episodic）策略梯度定理**（性能 $J(\boldsymbol{\theta}) = v_{\pi_\boldsymbol{\theta}}(s_0)$）：

$$
\nabla J(\boldsymbol{\theta}) \;\propto\; \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} q_\pi(s,a)\, \nabla \pi(a \mid s, \boldsymbol{\theta})
$$

其中 $\mu(s) = \sum_{k=0}^{\infty} \gamma^k \Pr(s_0 \to s, k, \pi)$ 是**折扣加权访问频率**——「从起始状态出发、在 $\pi$ 下、折扣累计地访问 $s$ 的次数」。<span class="marginnote">$\mu(s)$ 的折扣求和意味着「离起点近的状态权重大」——这正与「回合任务从 $s_0$ 出发、越早的状态对总回报影响越大」对齐。它是状态访问的「影响权重」，不是平稳分布（平稳分布在回合任务里无定义）。</span>

**持续式（continuing）策略梯度定理**（性能 $J = \bar r(\pi)$）：

$$
\nabla J(\boldsymbol{\theta}) \;=\; \sum_s \mu(s) \sum_a q_\pi(s,a)\, \nabla \pi(a \mid s, \boldsymbol{\theta})
$$

其中 $\mu(s)$ 现在是策略 $\pi$ 下的**平稳分布**。持续式没有折扣访问频率（无 $\gamma$），因为平均奖励设定里没有「起点」。**两式结构完全相同，只差 $\mu$ 的定义与比例常数**——这就是「一个定理、两种设定」的全部差别。两种设定的对账：

| 设定 | 性能 $J$ | 状态分布 $\mu(s)$ | 有无 $\gamma$ | 比例常数 |
| --- | --- | --- | --- | --- |
| 回合式 | $v_{\pi_\boldsymbol{\theta}}(s_0)$ | 折扣加权访问频率 $\sum_k\gamma^k\Pr(s_0\to s,k,\pi)$ | 有 | $\propto$ |
| 持续式 | 平均奖励 $\bar r(\pi)$ | 平稳分布 $d_\pi(s)$ | 无 | $=$ |

这张表值得背下来：**所有带 $\gamma$ 的策略梯度公式都默认回合式，所有平均奖励公式都默认持续式**——把 $\mu$ 用错分布，梯度方向就会整体偏斜。

## 2 证明的骨架：让价值梯度相消

用「分而治之」展开 $\nabla v_\pi$。对任意状态 $s$，由价值函数定义：

$$
v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\big[r + \gamma v_\pi(s')\big]
$$

对 $\boldsymbol{\theta}$ 求导（用乘积法则）：

$$
\nabla v_\pi(s) = \sum_a \Big[\nabla\pi(a|s)\, q_\pi(s,a) \;+\; \pi(a|s) \sum_{s',r} p(s',r|s,a)\, \gamma\, \nabla v_\pi(s')\Big]
$$

**关键观察**：第一项是「想要的」（$\nabla\pi \times q_\pi$），第二项把 $\nabla v_\pi$ 递归到下一层——带着 $\gamma$ 与转移概率「传递」到 $s'$。把上式对 $s=s_0$ 迭代展开 $k$ 层，$\nabla v_\pi$ 项不断后移，最终：

$$
\nabla v_\pi(s_0) = \sum_{s} \underbrace{\sum_{k=0}^{\infty}\gamma^k \Pr(s_0\to s, k, \pi)}_{=\mu(s)} \sum_a \nabla\pi(a|s)\, q_\pi(s,a)
$$

**第二项的 $\nabla v_\pi(s')$ 被「吸收」进下一层的 $\mu(s')$ 里，而不是出现在最终式中**——价值梯度就此完全消失。<span class="marginnote">这个「递归展开 + 吸收」是证明的精髓：你不需要知道「$s'$ 的价值梯度」，只需要知道「$s'$ 被访问（加权）了多少次」——访问频率由策略与转移决定，是「可采样的」，不像价值梯度那样「不可学习」。</span>

## 3 从求和到期望：梯度怎么变成样本

定理里的双重求和（对 $s$ 与 $a$）无法直接计算，但可以改写成**期望形式**——这正是采样更新的入口。先把「对 $a$ 求和」改写：

$$
\sum_a q_\pi(s,a)\, \nabla\pi(a|s,\boldsymbol{\theta}) \;=\; \sum_a \pi(a|s)\, q_\pi(s,a)\, \nabla\ln\pi(a|s,\boldsymbol{\theta})
$$

（这里用了 $\nabla\pi = \pi\nabla\ln\pi$。）于是定理的期望形式为：

$$
\nabla J(\boldsymbol{\theta}) \;\propto\; \mathbb{E}_{\pi}\Big[\sum_a q_\pi(S_t, a)\, \nabla \ln \pi(a \mid S_t, \boldsymbol{\theta})\Big]
$$

**期望下的状态 $S_t$ 从 $\mu$（访问分布）采样**；对每个状态，还要对**所有动作**加权求和。这个「全动作求和」是 REINFORCE 与 Actor-Critic 的出发点——下一课会看到，把「对动作求和」换成「采样单个动作」，就得到可用的随机梯度。<span class="marginnote">「对动作求和」在动作空间巨大时不可行，必须采样——于是有了「REINFORCE 用一个动作」与「A2C 用策略采样」的区别。动作空间连续时（第40课），求和变成积分，只能靠采样，策略梯度方法成为唯一自然的选择。</span>

## 4 数值算例：softmax 策略的梯度长什么样

策略参数化最常用的是 **softmax 线性偏好**：$\pi(a \mid s, \boldsymbol{\theta}) = \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}$，其中偏好函数取线性形式 $h(s,a,\boldsymbol{\theta})=\boldsymbol{\theta}^\top \mathbf{x}(s,a)$。它的对数梯度有一个漂亮且好用的闭式：

$$
\nabla\ln\pi(a\mid s,\boldsymbol{\theta}) = \mathbf{x}(s,a) - \sum_b \pi(b\mid s,\boldsymbol{\theta})\,\mathbf{x}(s,b)
$$

即「所选动作的特征向量，减去策略下的加权平均特征向量」。做一组具体数字：设动作空间 $\mathcal{A}=\{a_1,a_2\}$，特征取 one-hot，$\mathbf{x}(s,a_1)=(1,0)$、$\mathbf{x}(s,a_2)=(0,1)$，$\boldsymbol{\theta}=(1.5,-0.5)$。于是偏好为 $(1.5,-0.5)$，计算概率：

$$
e^{1.5}\approx 4.48,\; e^{-0.5}\approx 0.61 \;\Rightarrow\; \pi(a_1\mid s)\approx \frac{4.48}{4.48+0.61}\approx 0.88,\quad \pi(a_2\mid s)\approx 0.12
$$

若回合中这一状态被采样且**选中的是 $a_1$**，则 $\nabla\ln\pi(a_1\mid s) = (1,0) - \big[0.88(1,0)+0.12(0,1)\big] = (0.12, -0.12)$；若**选中的是 $a_2$**，则 $\nabla\ln\pi(a_2\mid s) = (0,1) - (0.88,0.12) = (-0.88, 0.88)$。

| 实际选中的动作 | $\nabla\ln\pi$ | 对参数的影响 |
| --- | --- | --- |
| $a_1$（大概率 0.88） | $(+0.12,-0.12)$ | 温和地增 $\theta_1$、减 $\theta_2$ |
| $a_2$（小概率 0.12） | $(-0.88,+0.88)$ | 用力地降 $\theta_1$、升 $\theta_2$ |

**读法**：梯度方向总是「推着选中的动作对应的参数走」，且幅度与该动作的当前概率成反比——选小概率动作时更新猛烈，选大概率动作时更新温和。<span class="marginnote">「概率低则梯度大」正是策略梯度能探索的理由：一个几乎不会被选的动作，一旦被选中并被高回报奖励，参数会被大幅推向它。把这一格记牢，REINFORCE 的方差来源与「基线」的动机（下一节）都从这里长出来。</span>

## 5 公式解析：乘积法则里的「相消」

$$
\underbrace{\nabla v_\pi(s)}_{\text{要算的}} = \sum_a \Big[\underbrace{\nabla\pi(a|s)\, q_\pi(s,a)}_{\text{目标项：保留}} + \underbrace{\pi(a|s)\sum_{s',r}p(\cdot)\,\gamma\,\nabla v_\pi(s')}_{\text{递归项：被吸收}}\Big]
$$

- **第一步，认乘积法则**：$v_\pi$ 是「$\pi$ 与 $q_\pi$ 的乘积」，求导自然出两项：$\nabla\pi$ 项（策略怎么变）与 $\nabla q_\pi$ 项（价值怎么变，被包装进 $\nabla v_\pi(s')$）。
- **第二步，认递归**：第二项把「$s'$ 的价值梯度」乘 $\gamma$ 与转移概率后**留给下一层**。这不是消失，而是「传递」——它会在展开后成为「$s'$ 的访问权重」的一部分。
- **第三步，认吸收**：把 $s=s_0$ 处的等式反复代入 $s'$ 处，$\nabla v_\pi$ 一路后移、最终在 $T\to\infty$ 时被折扣压到 0；沿途每个状态的「贡献权重」恰好累积成 $\mu(s)$。**价值梯度被「打折消失」，访问频率顶上**——定理由此成立。<span class="marginnote">一个检验定理正确性的方式：在表格型、$\gamma\to1$、均匀初始分布下，$\mu(s)$ 趋于平稳分布，定理退化为「性能梯度 = 平稳分布下的策略加权价值」——与第10章平均奖励的直觉吻合。这个「极限对账」能帮你判断是否记错 $\mu$ 的定义。</span>

## 6 易错点辨析

**辨析｜易错点：** 把定理里的 $q_\pi$ 当成「要优化的量」。$q_\pi$ 在定理里是**已知的权重**（给定当前策略的价值），$\nabla J$ 只对 $\boldsymbol{\theta}$ 求导、不对 $q_\pi$ 求导。**$q_\pi$ 用近似值替换不影响定理的「无偏性」**（在正确的分布下），这是 Actor-Critic 敢用价值网络当权重的理论基础。

**另一个易错点**：混淆「回合式 $\mu$」与「持续式 $\mu$」。回合式是**折扣加权访问频率**（$\sum_k\gamma^k\Pr(s_0\to s,k,\pi)$），持续式是**平稳分布**。把两者记混，梯度方向就错。

**第三个易错点**：以为定理要求「策略必须可微于所有状态」。它要求 $\nabla\pi(a|s,\boldsymbol{\theta})$ 存在（策略参数化可微），但**不要求 $q_\pi$ 可微**——价值可以是表格、查表、任何不可微函数。这正是「价值作秤、策略作针」分工的数学保证。

## 7 小结

- **回合式定理**：$\nabla J \propto \sum_s\mu(s)\sum_a q_\pi(s,a)\nabla\pi(a|s,\boldsymbol{\theta})$，$\mu$ 是折扣加权访问频率。
- **持续式定理**：同结构，$\mu$ 是平稳分布，性能取平均奖励。
- **证明精髓**：乘积法则展开 $\nabla v_\pi$，$\nabla q_\pi$ 项沿状态递归、被折扣与访问频率吸收，最终消失。
- **期望形式**：$\nabla J \propto \mathbb{E}_\pi[\sum_a q_\pi(S_t,a)\nabla\ln\pi(a|S_t,\boldsymbol{\theta})]$——对动作求和是采样更新的入口。
- 价值可不可微都行——策略梯度天生免疫「价值梯度不可学习」的诅咒。

在下一节，我们把「对动作求和」换成「采样一个动作」——**REINFORCE**：蒙特卡洛策略梯度，最简单的可用实现。
