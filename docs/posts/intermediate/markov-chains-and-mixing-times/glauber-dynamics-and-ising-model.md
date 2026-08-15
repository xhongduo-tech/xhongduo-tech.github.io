---
title: Glauber 动力学与 Ising 模型
date: 2026-08-07
---

# Glauber 动力学与 Ising 模型

<div class="epigraph">
<p>临界点上，秩序与混沌只隔一层薄纱。</p>
<footer>—— 肯尼斯 · 威尔逊（Kenneth Wilson）</footer>
</div>

<div class="article-byline">
<p>第二级 · 马尔可夫链与混合时间 ｜ Levin, Peres &amp; Wilmer《Markov Chains and Mixing Times》Ch. 3 &amp; 15 ｜ 2026-08-07</p>
</div>

## 为什么从 Glauber 动力学与 Ising 模型开始

洗牌模型展示的是「没有物理背景」的纯组合混合；**Ising 模型**则带来了物理的深度。它是统计力学研究铁磁相变的最简模型：在晶格上放上 $+1/-1$ 的自旋，邻近自旋倾向一致（低能），温度决定这种倾向的强度。直接从这个模型**采样**（在 $2^n$ 个构型中按 Boltzmann 分布抽一个）是完全不可行的，但 **Glauber 动力学**给出了可行路径：每次只更新一个自旋，构造一条以 Ising 平稳分布为极限的马尔可夫链，跑足够久就得到一次采样。于是混合时间成了物理问题：**高温下链混合快，低温下混合慢，临界温度附近出现 cutoff**。<span class="marginnote">Ising 模型 + Glauber 动力学是 MCMC 最有理论深度的样板：它把「物理相变」翻译成「马尔可夫链的混合相变」。理解它，就同时理解了一半的统计物理采样与一半的混合时间理论。</span>

## 1 Ising 模型与 Boltzmann 分布

**Ising 模型（Ising model）**：图 $G = (V, E)$ 上每个顶点赋一个自旋 $\sigma_v \in \{+1, -1\}$，构型空间 $\{-1,+1\}^V$。给定逆温度 $\beta \geq 0$ 与外部磁场 $h$，构型 $\sigma$ 的能量为

$$
H(\sigma) = - \beta \sum_{\{u,v\} \in E} \sigma_u \sigma_v - h \sum_{v \in V} \sigma_v
$$

**Boltzmann（Gibbs）分布**为

$$
\pi(\sigma) = \frac{1}{Z} e^{-H(\sigma)}, \qquad Z = \sum_{\sigma'} e^{-H(\sigma')}
$$

其中 $Z$ 是配分函数——一个 $2^{|V|}$ 项的归一化常数，通常**不可计算**。<span class="marginnote">配分函数不可算正是 MCMC 的全部动机：我们只采样而不计算 $Z$。能量越低、概率越高；$\beta$ 越大，邻近自旋越倾向同向（铁磁序），这就是「低温长程序」的数学来源。</span>

**格点上的随机游走式的观察**：$h=0$ 且 $\beta=0$ 时，所有构型等概率，这是纯均匀情形；$\beta \to \infty$ 时，只有全同向构型大概率。中间的 $\beta_c$（如二维正方格子上 $\beta_c \approx 0.44$）对应相变温度。

## 2 Glauber 动力学

**Glauber 动力学（Glauber dynamics）**：每条边以速率 1 的泊松时钟更新一个端点——离散时间版本是**单点热浴（heat-bath）**：每步均匀随机选一个顶点 $v$，把 $\sigma_v$ 按「其余自旋固定时 $v$ 的条件分布」重新采样：

$$
P(\sigma, \sigma^{(v,s)}) = \frac{1}{|V|} \cdot \frac{e^{-H(\sigma^{(v,s)})}}{e^{-H(\sigma^{(v,+)})} + e^{-H(\sigma^{(v,-)})}}
$$

其中 $\sigma^{(v,s)}$ 是仅把 $\sigma_v$ 改为 $s$ 的构型，$\sigma^{(v,s)}$ 与 $\sigma$ 只在 $v$ 处不同（若 $s = \sigma_v$ 则原地不动，构成自环）。<span class="marginnote">「热浴」的名字来自每一步把选定自旋「泡进」条件热浴：新自旋以概率正比于条件 Boltzmann 权重抽取。由于新自旋只依赖 $v$ 的邻居，局部更新成本是 $O(\text{度数})$，非常适合高维采样。</span>

**Glauber 链以 $\pi$ 为平稳分布**，验证用到细致平衡：设 $\sigma$ 与 $\tau$ 只在 $v$ 处不同，则

$$
\pi(\sigma) P(\sigma, \tau) = \frac{e^{-H(\sigma)}}{Z} \cdot \frac{1}{|V|}\frac{e^{-H(\tau)}}{e^{-H(\sigma)} + e^{-H(\tau)}} = \frac{e^{-H(\tau)}}{Z} \cdot \frac{1}{|V|}\frac{e^{-H(\sigma)}}{e^{-H(\tau)} + e^{-H(\sigma)}} = \pi(\tau) P(\tau, \sigma)
$$

两边相等——**Glauber 链可逆，平稳分布就是 Ising 的 Gibbs 分布**。

## 3 公式解析：单点更新概率怎么算

以无外场 $h=0$、图上顶点 $v$ 有 $d$ 个邻居为例，邻居自旋和为 $S = \sum_{u \sim v} \sigma_u$。固定其余自旋，$v$ 的新自旋条件分布：

$$
\mathbb{P}(\sigma_v = +1 \mid \text{邻居}) = \frac{e^{\beta S}}{e^{\beta S} + e^{-\beta S}} = \frac{1}{1 + e^{-2\beta S}}
$$

逐项拆解这个式子的意义：

- **第一步，条件能量**：仅翻转 $v$ 时，能量变化来自 $v$ 与邻居的耦合：$\Delta H = 2\beta \sigma_v S$。$v=+1$ 时能量为 $-2\beta S$ 项，$v=-1$ 时为 $+2\beta S$ 项。
- **第二步，权重比**：Boltzmann 权重正比 $e^{-H}$，故两种自旋权重之比为 $e^{+2\beta S} : e^{-2\beta S}$。
- **第三步，归一化**：两权重之和 $e^{2\beta S} + e^{-2\beta S}$ 作分母，即得上式。
- **第四步，读出趋势**：若 $S > 0$（多数邻居为 $+1$），则 $e^{-2\beta S} \lt  1$，$\mathbb{P}(+1) > 1/2$——**自旋倾向跟从多数邻居**；$\beta$ 越大，跟从越强。这就是「铁磁性」的微观机制。

**具体数字**：设一个顶点有 $d = 4$ 个邻居，其中三个为 $+1$（即 $S = 2$）。若 $\beta = 0.5$，$\mathbb{P}(+1) = 1/(1+e^{-2}) \approx 0.88$——自旋以约 $88\%$ 概率跟从多数；若 $\beta = 0.1$，$\mathbb{P}(+1) = 1/(1+e^{-0.4}) \approx 0.60$，仅轻微偏向多数。同样 $S = 2$，低温下几乎必然跟从，高温下只是微弱偏向——「温度控制极化强度」在此量化可见。

## 4 高温快速混合与低温慢混合

Glauber 链的混合速度由温度决定，这是本节的物理核心。

**高温（小 $\beta$）**：自旋之间耦合弱，单点更新几乎独立，链混合快。在 $d$ 正则图上，谱隙满足

$$
\gamma \geq 1 - \tanh(\beta) \cdot (d - 1)^{\text{修正}}
$$

高维或小 $\beta$ 时，谱隙有正下界，混合时间 $t_{\mathrm{mix}} = O(|V| \log |V|)$。<span class="marginnote">直觉：小 $\beta$ 时条件分布 $\mathbb{P}(+1) \approx 1/2$，每次更新几乎等概率重置自旋，记忆快速丧失；大 $\beta$ 时条件分布极化，自旋被邻居「钉住」，忘掉初态很慢。</span>

**低温（大 $\beta$）**：自旋高度相关，出现**亚稳态**——系统可能整体卡在「几乎全 +」或「几乎全 −」很久。混合时间指数增长：

$$
t_{\mathrm{mix}} \geq e^{c\beta |V| / 2} \quad \text{或更强}
$$

瓶颈是「从全 + 翻到全 −」需翻越一个巨大的能量壁垒。**这是混合时间理论最重要的物理教训：采样低温 Ising 构型，朴素 Glauber 链要跑指数长时间。**

## 5 临界点与相变

在临界温度 $\beta_c$ 附近，谱隙与混合时间的标度发生突变。对 $n \times n$ 二维方格上的 Ising 模型：

- $\beta \lt  \beta_c$：谱隙 $\gamma \asymp 1/n^2$（扩散型慢但多项式），混合时间 $O(n^2 \log n)$；
- $\beta = \beta_c$：谱隙 $\gamma \asymp n^{-15/4}$（临界慢化，critical slowing down）；
- $\beta > \beta_c$：谱隙指数小，混合时间指数长，且发生**cutoff**。

**为什么低温混合这么慢？** 把全 $+$ 与全 $-$ 看成两个「势阱」。要从全 $+$ 变成全 $-$，必须逐一翻转自旋，中途经过大量高能量构型（正负边界越长能量越高）。这个能量壁垒高度 $\Delta E \sim \beta \cdot \text{边界长度}$，跨越它的概率按 $e^{-\beta \Delta E}$ 指数衰减，于是混合时间也指数长——这就是「亚稳态」的数学面貌，也是采样低温构型困难的根本原因。

临界慢化是计算物理的著名难题，也是「用更聪明的算法（如 cluster 算法）」的动力。<span class="marginnote">在 $n \times n$ 方格上，临界点附近关联长度发散，自旋块互相锁定，单点更新算法必须绕过这个障碍。Swendsen–Wang、Wolff 簇算法通过同时翻转整簇自旋打破临界慢化，是 MCMC 算法设计「量身定做提议分布」的范例。</span>

## 6 辨析｜易错点：Glauber 与 Metropolis 的区别

**辨析｜易错点：** Glauber 热浴是「按条件分布直接采样新自旋」，Metropolis 是「提议一个翻转，再按接受概率接受或拒绝」。两者都以 Gibbs 分布为平稳分布，但 Glauber 通常效率更高（无拒绝浪费）也更难实现；Metropolis 更通用（任何提议都行）。许多文献把两者统称「单点更新动力学」，但接受-拒绝结构与直接重采样在实现上完全不同。

**辨析｜易错点：** Glauber 链**有自环**（选中的 $v$ 重采样后可能不变），所以链非周期——不要因为「更新自旋」就想当然以为每步都变。自环还意味着链在构型图上不是简单随机游走，谱隙分析必须计入自环概率。

## 7 小结

- **Ising 模型**的 Gibbs 分布 $\pi(\sigma) = e^{-H(\sigma)}/Z$ 有不可计算的配分函数，采样依赖 MCMC。
- **Glauber 动力学**（单点热浴）每次更新一个自旋，满足细致平衡，以 $\pi$ 为平稳分布，且**可逆、非周期**。
- 单点更新概率 $\mathbb{P}(+1) = 1/(1 + e^{-2\beta S})$：$S>0$ 时倾向跟从多数邻居，$\beta$ 越大极化越强——铁磁性的微观机制。
- **高温快混**：小 $\beta$ 时谱隙有正下界，$t_{\mathrm{mix}} = O(|V|\log|V|)$；**低温慢混**：翻越能量壁垒，混合时间指数长。
- **临界慢化**：$\beta = \beta_c$ 时谱隙 $\asymp n^{-15/4}$，单点更新被自旋块锁定，需 Swendsen–Wang / Wolff 簇算法打破。
- **辨析**：Glauber 是「按条件分布直接重采样」，Metropolis 是「提议 + 接受/拒绝」，平稳分布相同但实现与效率不同。
- **工程要点**：Glauber 链有自环（重采样后可能不变），保证非周期，但谱隙分析必须计入自环概率。

在下一节，我们将看到更通用的采样引擎——**Metropolis–Hastings 算法**：它不依赖 Ising 的特殊结构，对任意目标分布都成立。