---
title: 对偶理论与最优性条件
date: 2026-08-07
---

# 对偶理论与最优性条件

<div class="epigraph">
<p>线性规划的对偶问题有一个美丽的解释：它把资源配置问题，改写成了对资源定价的问题。</p>
<footer>—— 列昂尼德 · 坎托罗维奇（Leonid Kantorovich），《最优计划中的对偶性问题》（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 最优传输理论 ｜ Villani《Optimal Transport: Old and New》第5章 ｜ 2026-08-07</p>
</div>

## 为什么从对偶理论开始

上一篇把 Monge 问题松弛成了一个（无穷维）线性规划：在耦合集 $\Pi(\mu,\nu)$ 上极小化线性代价。凡线性规划，必有对偶。**Kantorovich 对偶（Kantorovich duality）**把这个"极小化运输成本"的问题，改写成"极大化某种收益"的问题——运输公司要最小化运费，而一个聪明的观测者想证明"运费不可能低于某个值"，两者在同一数量上达到相等。对偶理论的价值有三重：它给出最优解的**表征**（怎么验证一个解是最优的）、给出**下界**（任何对偶可行解都是一个可行下界）、并且是几乎所有后续算法（Sinkhorn、Wasserstein GAN 的判别器）的数学根基。<span class="marginnote">对偶的直觉在经济学术语里就是"影子价格"：每个约束（每处供需）配一个价格，最小成本恰好等于所有资源的影子价格之和。这个直觉是列宁格勒学派（Kantorovich 与他的学生）在 1940–50 年代发展出来的。</span>

## 1 从拉格朗日乘子到对偶

回忆线性规划的对偶怎么来：把约束"吸收"进目标函数，配以乘子，再对原变量求下确界。Kantorovich 问题也一样。原始问题

$$
\min_{\pi} \; \int_{X \times Y} c(x,y)\, d\pi, \qquad \text{s.t. 边际为 } \mu, \nu
$$

配两个"拉格朗日乘子"函数 $\varphi(x)$ 与 $\psi(y)$（分别对应两条边际约束），吸收约束后对 $\pi$ 求下确界，就得到对偶问题。关键在于约束转换：

$$
\varphi(x) + \psi(y) \le c(x, y), \qquad \forall (x,y) \in X \times Y
$$

任何一对满足这个逐点不等式的函数，都给出一个下界：对任意可行 $\pi$，

$$
\int \varphi \, d\mu + \int \psi \, d\nu
= \int \big[\varphi(x) + \psi(y)\big]\, d\pi
\le \int c(x,y)\, d\pi
$$

于是原始最小值被这些"对偶可行对"从下方包住。<span class="marginnote">中间那步等式是抓住精髓的一步：因为 $\pi$ 的边际恰好是 $\mu$ 与 $\nu$，所以 $\int\varphi\,d\mu = \int\varphi(x)\,d\pi(x,y)$，$\psi$ 同理。<strong>正是边际条件，把"两个空间上的积分"合并成了"乘积空间上的积分"。</strong></span>

## 2 公式解析：Kantorovich 对偶

**Kantorovich 对偶定理**说，在温和条件下（$c$ 下半连续、$X,Y$ 为波兰空间）对偶间隙为零：

$$
\min_{\pi \in \Pi(\mu,\nu)} \int_{X \times Y} c(x,y)\, d\pi(x,y)
=
\sup_{\varphi \oplus \psi \le c} \left( \int_X \varphi\, d\mu + \int_Y \psi\, d\nu \right)
$$

拆成三步理解：

- **第一步，读懂对偶可行条件 $\varphi \oplus \psi \le c$**：这里 $\varphi \oplus \psi$ 是"直和"记号，表示函数 $(x,y) \mapsto \varphi(x) + \psi(y)$。条件要求它对所有 $(x,y)$ 都成立。物理直觉：$\varphi(x)$ 是"土在 $x$ 处的单价"，$\psi(y)$ 是"坑在 $y$ 处的单价"，任何一对 $(x,y)$ 的单价之和不得超过真实运费，否则就存在套利。
- **第二步，读懂目标 $\int\varphi\,d\mu + \int\psi\,d\nu$**：总收益 = 所有初始质量按 $\varphi$ 计价 + 所有目标质量按 $\psi$ 计价。最大化这个总收益，就是在找"最紧的下界"。
- **第三步，读懂等式**：$\min = \sup$。**原始问题的最小运输成本，恰好等于对偶问题的最大下界。** 这就是强对偶。对偶最优解 $(\varphi,\psi)$ 称为**对偶势函数（dual potentials）**。

**辨析｜易错点：** 对偶里是 $\sup$ 不是 $\max$，且对偶变量不是唯一的——把 $\varphi$ 加上常数、$\psi$ 减去同一常数，目标不变。因此实际计算时常固定 $\psi = \varphi^c$（见下节）来消去冗余自由度。

## 3 c-变换与 c-凹函数

对偶问题里的 $(\varphi,\psi)$ 可以大幅简化。给定任意 $\varphi$，定义它的 **c-变换（c-transform）**：

$$
\varphi^c(y) = \inf_{x \in X} \big[ c(x,y) - \varphi(x) \big]
$$

对偶条件 $\varphi + \psi \le c$ 成立当且仅当 $\psi \le \varphi^c$，而把 $\psi$ 取到最大（即取 $\varphi^c$）只会让目标更大，所以对偶问题可以只对 $\varphi$ 极大化：

$$
\sup_{\varphi} \left( \int_X \varphi\, d\mu + \int_Y \varphi^c \, d\nu \right)
$$

满足 $\varphi = (\varphi^c)^c$ 的函数称为 **c-凹函数（c-concave）**，它们是对偶问题的自然候选。<span class="marginnote">当 $c(x,y) = -x\cdot y$ 时，c-凹函数就是通常的凹函数（的变体），而 $\varphi^c$ 类似 Fenchel 共轭——学过第一级《凸分析》的读者会发现，最优传输的对偶理论几乎就是凸共轭理论在乘积空间上的翻版。</span>

## 4 最优性条件：互补松弛与 c-单调性

对偶理论不仅给出数值，还给出**最优解的刻划**。设 $\pi$ 原始最优、$(\varphi, \psi)$ 对偶最优，则强对偶加上不等式链要求中间的不等式全部取等号，于是得到**互补松弛（complementary slackness）**：

$$
\varphi(x) + \psi(y) = c(x, y), \qquad \pi\text{-几乎处处成立}
$$

即：**凡是被最优耦合用到的"通路" $(x,y)$，其单价之和必须恰好等于真实代价**；若某条通路有严格不等式，则最优 $\pi$ 不会把质量流过去。<span class="marginnote">这像极了最短路径里"松弛"终止时的条件，也像极了对偶单纯形里的"对偶可行 + 互补松弛 = 最优"。最优传输的现代算法（包括第 7 篇的 Sinkhorn）本质上都是在逐步逼近这组等式。</span>

由此导出最优耦合的几何特征：最优 $\pi$ 的支集 $\operatorname{supp}\pi$ 必须包含在集合

$$
\Gamma_c = \big\{ (x,y) : \varphi(x) + \psi(y) = c(x,y) \big\}
$$

之内，而 $\Gamma_c$ 是一个 **c-单调（c-monotone）** 集合：不存在有限个点对 $(x_i,y_i) \in \Gamma_c$ 与排列 $\sigma$ 使得 $\sum_i c(x_i, y_i) > \sum_i c(x_i, y_{\sigma(i)})$。直观上，c-单调集合"不允许交叉运输"——如果 $x_1$ 运到 $y_1$、$x_2$ 运到 $y_2$ 但交换更便宜，那这对就不是最优的。这个性质在后面 Brenier 定理（第 5 篇）里会变成一个非常具体的几何结论。

## 5 一个微观验证：点质量的 $W_1$ 对偶

理论讲得再多，不如一个小例子把"对偶 = 找最紧下界"钉死。设 $\mu = \delta_{0}$、$\nu = \delta_{1}$（各只有一个点，位于 0 与 1），代价 $c(x,y) = \|x-y\|$。

**原始问题**：$\Pi(\mu,\nu)$ 里只有一个耦合——两个点必须"一对一"地配成 $\pi = \delta_{(0,1)}$。总代价

$$
\min_{\pi} \int \|x-y\|\, d\pi = \|0 - 1\| = 1
$$

**对偶问题**：找势函数 $\varphi, \psi$ 极大化 $\varphi(0) + \psi(1)$，约束 $\varphi(x) + \psi(y) \le \|x-y\|$ 对所有 $x,y \in \{0,1\}$ 成立。约束展开有四条：

$$
\varphi(0)+\psi(0) \le 0, \quad \varphi(1)+\psi(1) \le 0, \quad
\varphi(0)+\psi(1) \le 1, \quad \varphi(1)+\psi(0) \le 1
$$

取 $\varphi(x) = -x$、$\psi(y) = y$。逐条检查：前两条给 0 与 0，第三条给 1，第四条给 $-1$——全部满足。目标值 $\varphi(0) + \psi(1) = 0 + 1 = 1$。

于是**原始最小值 1 = 对偶最大值 1**，强对偶成立，且我们找到了一个达到下界的势函数。<span class="marginnote">这里的 $\varphi(x)=-x$、$\psi(y)=y$ 恰好是 $W_1$ 对偶里的"最优 Lipschitz 势"：$\varphi, \psi$ 拼起来正是 Lipschitz 常数 1 的函数 $f(t) = t$，符合第 4 篇 Kantorovich–Rubinstein 公式 $\sup_{\|f\|_{\mathrm{Lip}}\le 1}\int f\,d\mu - \int f\,d\nu$ 的形态。</span>

**辨析｜易错点：** 注意对偶最优势**不唯一**。把 $\varphi$ 换成 $\varphi + a$、$\psi$ 换成 $\psi - a$，目标不变、约束不变（两边同时加 $a$ 与 $-a$）。这印证了第 2 节"消去冗余自由度"的说法——实际算法里要用归一化约定（如固定 $\psi = \varphi^c$）才能把解钉住。

## 6 小结

- **Kantorovich 对偶**：$\min_{\pi} \int c\,d\pi = \sup_{\varphi + \psi \le c}\left(\int\varphi\,d\mu + \int\psi\,d\nu\right)$，强对偶在一般条件下成立。
- 对偶变量 $( \varphi, \psi)$ 可解释为**影子价格 / 势函数**；可消去冗余化为单变量问题。
- **c-变换** $\varphi^c(y) = \inf_x [c(x,y) - \varphi(x)]$；**c-凹函数**是 $\varphi = (\varphi^c)^c$ 的函数。
- **最优性条件**：互补松弛 $\varphi(x) + \psi(y) = c(x,y)$ 在最优耦合的支集上几乎处处成立。
- 最优耦合的支集是 **c-单调**的——不允许交叉运输。

在下一节，我们把代价函数取成距离的 $p$