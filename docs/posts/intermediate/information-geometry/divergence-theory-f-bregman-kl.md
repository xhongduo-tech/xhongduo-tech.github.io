---
title: 散度理论：f-散度、Bregman 散度与 KL 散度的几何
date: 2026-08-07
---

# 散度理论：f-散度、Bregman 散度与 KL 散度的几何

<div class="epigraph">
<p>散度是统计流形上的「广义距离平方」——它不要求对称，却保留了信息论与几何的全部有用性质。</p>
<footer>—— 甘利俊一（Shun-ichi Amari）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息几何 ｜ Amari《Information Geometry and Its Applications》Ch. 1-2 ｜ 2026-08-07</p>
</div>

## 为什么从散度理论开始

前几篇里 KL 散度一直是「暗线主角」：Pythagoras 定理、投影定理、EM、自然梯度都以它为核心。但 KL 散度不是唯一的散度——f-散度与 Bregman 散度构成两个大族，它们共享同样的几何骨架：**每个散度都能在统计流形上诱导一个度量和一族联络，而且这族联络恰是 α-联络**。<span class="marginnote">这是信息几何最深刻的一课：α-联络不是凭空发明的，而是「任何散度」自然诱导出来的。散度 ↔ 度量 + α-联络 的对应，由 Amari 与 Nagaoka 在 Ch. 6 系统证明，可类比于「距离诱导黎曼度量」的经典结论。</span>

本篇目标：建立散度的两大族（f-散度、Bregman 散度），弄清 KL 散度为何同时属于两族，并展示散度如何诱导出度量和联络。

## 1 什么是散度：三公理

回顾并严格化散度的定义。流形上的函数 $D[p : q]$（记号用「:」强调不对称）满足：

**非负性**：$D[p:q] \ge 0$，且 $D[p:q] = 0 \iff p = q$。
- **局部度量性质**：在 $q = p + dp$ 时，$D[p:q] = \frac{1}{2}\sum_{i,j} g_{ij}(p)\, dp^i dp^j + o(\|dp\|^2)$，其中 $g$ 是该散度诱导的度量。
- 一般**不对称**：$D[p:q] \ne D[q:p]$；也不满足三角不等式。

第三点最重要：**散度是「带方向的距离平方」**。第一参数处的微分衡量的是「从 p 出发走向 q 的成本」，交换参数意味着换一个出发点。<span class="marginnote">正因为不对称，散度天然编码了「方向」。在优化里，最小化 D[p:q] 与最小化 D[q:p] 是两件不同的事——这对应极大似然（M 投影）与最大熵（I 投影）的差别，也是散度相对度量的真正优势。</span>

## 2 f-散度族

**f-散度（f-divergence）**：给定凸函数 $f$（$f(1) = 0$），定义

$$D_f[p : q] = \int q(x)\, f\!\left(\frac{p(x)}{q(x)}\right) dx$$

这族散度包罗大量经典对象：

- KL 散度：$f(t) = t\log t$，得 $D_{KL}[p:q] = \int p \log \frac{p}{q}$。
- 反向 KL：$f(t) = -\log t$，得 $D_{KL}[q:p]$。
- Hellinger 距离：$f(t) = (\sqrt{t} - 1)^2$。
- $\chi^2$ 散度：$f(t) = (t-1)^2$。
- Jensen-Shannon 与总变差也都在族内。

f-散度的共同性质：**信息单调性（information monotonicity）**——任何统计处理（充分化简、粗粒化）都不会增加散度。这使 f-散度成为「信息损失的天然度量」。<span class="marginnote">Csiszár 在 1963 年独立引入 f-散度（因此也叫 Csiszár 散度），阿里·西尔维（Ali &amp; Silvey）同年也有等价定义。信息单调性后来被证明是「散度」概念最本质的公理。</span>

**辨析｜易错点：** f-散度不是唯一的散度定义方式。f-散度族成员满足信息单调性，但 Bregman 散度族一般不满足。两类散度的交集恰好是 KL 散度（及其仿射变换）——**KL 是唯一同时属于两大族的散度**，这是它无处不在的原因之一。

### 一张表认识 f-散度族

把最常见的 f-散度成员列成一张表，并写出对应的生成函数 $f$：

| 名称 | 生成函数 $f(t)$ | 散度公式 |
| --- | --- | --- |
| KL 散度 | $t\log t$ | $\int p \log(p/q)$ |
| 反向 KL | $-\log t$ | $\int q \log(q/p)$ |
| Hellinger 距离 $^2$ | $(\sqrt{t}-1)^2$ | $\int (\sqrt p - \sqrt q)^2$ |
| $\chi^2$ 散度 | $(t-1)^2$ | $\int (p-q)^2/q$ |
| 总变差 | $\lvert t-1 \rvert$ | $\int \lvert p-q \rvert$ |

注意 Hellinger 距离本身是对称的（$D_H^2 = 2\left(1 - \int \sqrt{pq}\right)$），是 f-散度里少见的「准度量」；而 $\chi^2$ 在分布鲁棒优化（DRO）里是 KL 球之外最常用的约束半径。**同一个 $f$ 就决定了一种「差异的计价方式」——选 $f$，就是选统计问题里「多远算远」的答案。**<span class="marginnote">在假设检验里，f-散度还对应「二分类贝叶斯误差」的刻画：最优检验的功效由似然比的分布决定，而 f-散度恰好是它的单调函数。这是 f-散度家族与推断理论相连的最古老纽带。</span>

## 3 Bregman 散度族

**Bregman 散度**：给定光滑凸函数 $\varphi$（Bregman 生成元），定义

$$D_\varphi[p : q] = \varphi(p) - \varphi(q) - \langle \nabla\varphi(q),\, p - q \rangle$$

即「$\varphi$ 在 $q$ 处的切平面高出 $\varphi(p)$ 的距离」。Bregman 散度族同样庞大：

- 平方欧氏距离：$\varphi(x) = \|x\|^2$，得 $\|p - q\|^2$。
- KL 散度（对离散分布）：$\varphi(p) = \sum_i p_i \log p_i$（负熵），得 $D_{KL}[p:q]$。
- Itakura-Saito 距离：$\varphi(x) = -\log x$，广泛用于谱估计。

Bregman 散度的核心性质是**广义三点不等式**与**广义 Pythagoras**——这正是我们第 5 篇在指数族里用的结构：指数族的势函数 $\psi$ 就是 Bregman 生成元，KL 散度就是它对应的 Bregman 散度。<span class="marginnote">Bregman 散度的几何与「凸函数的 Legendre 变换」深度绑定：两点之间的 Bregman 距离等于「对偶空间里的一条竖线段」。这解释了为何指数族（凸函数 ψ 的世界）里 Pythagoras 定理如此干净。</span>

## 4 公式解析：散度如何诱导度量与联络

这是散度理论的心脏——**每个散度都唯一诱导一个度量和一族联络**。以 Bregman 散度 $D_\varphi$ 为例，取坐标 $p, q$ 为 $(\theta_P, \theta_Q)$，在 $q = p$ 处计算各阶导数：

$$
\begin{aligned}
g_{ij} &= \partial_{p^i} \partial_{q^j} D[p:q]\big|_{q=p} = \partial_i\partial_j \varphi(p) \\[2mm]
\Gamma^{(\alpha)}_{ij,k} &= \frac{1 - \alpha}{2}\, \partial_{p^i}\partial_{p^j}\partial_{q^k} D\big|_{q=p} \;+\; \frac{1+\alpha}{2}\, \partial_{p^i}\partial_{p^j}\partial_{p^k} D\big|_{q=p}
\end{aligned}
$$

分步理解：

- **第一步，度量来自二阶混合导**：$g_{ij} = \partial_i\partial_j \varphi$。对于 KL（$\varphi = $ 负熵），这正是 Fisher 度量——**Fisher 度量就是 KL 散度的二阶展开**。
- **第二步，联络来自三阶导**：对第一参数取两次偏导、对第二参数取一次（或全对第一参数），再按 α 权重混合，得到 α-联络。α 的不同取值对应「从哪一侧测量三阶信息」。
- **第三步，验证 KL 特例**：对 KL 散度，$\alpha = -1$ 得到 e-联络，$\alpha = 1$ 得到 m-联络——**KL 散度精确诱导出对偶平坦结构**，与第 3、4 篇完全一致。

直觉：**散度是流形的「全息编码」**——只用它一个对象，二阶信息给出度量，三阶信息给出联络，两个三阶方向的组合给出对偶结构。KL 之所以特殊，是因为它的三阶信息恰好产生一对曲率为零的联络。

## 5 两大族的应用坐标

- **f-散度**：变分推断、GAN 的 Jensen-Shannon 与总变差目标、分布鲁棒优化（DR）的 $\chi^2$ 与 KL 球约束、公平性约束。<span class="marginnote">GAN 的原始目标正是「判别器逼近 f-散度、生成器最小化它」；f-GAN 则把任意 f-散度写成可微目标。所以 GAN 的训练本质上是「散度空间里的博弈」。</span>
- **Bregman 散度**：k-means 聚类（Bregman 硬聚类）、矩阵分解（Itakura-Saito）、Boosting（对指数损失的 Legendre 对应）、以及自然梯度对应的凸优化。
- **KL 交叉点**：当你的算法既想要信息单调性又想要 Pythagoras 几何时，唯一的选择就是 KL——这解释了它在统计机器学习里无法撼动的地位。

### 速查：两大散度族对照

| 性质 | f-散度 | Bregman 散度 |
| --- | --- | --- |
| 定义 | $\int q\, f(p/q)$ | $\varphi(p)-\varphi(q)-\langle\nabla\varphi(q), p-q\rangle$ |
| 信息单调性 | 有 | 一般没有 |
| 广义 Pythagoras | 一般没有 | 有 |
| 诱导的联络 | α-联络 | α-联络 |
| 两者的交集 | KL 及其仿射变换 | |

**两类散度的交集恰好是 KL**——这就是它在统计机器学习里无处不在的根本原因。

## 6 小结

- **散度三公理**：非负、零当且仅当相等、局部二阶项给出度量；允许不对称。
- **f-散度族**：$D_f[p:q] = \int q f(p/q)$，含 KL、Hellinger、$\chi^2$，享有**信息单调性**。
- **Bregman 散度族**：$D_\varphi[p:q] = \varphi(p)-\varphi(q)-\langle\nabla\varphi(q), p-q\rangle$，含 KL、欧氏距离，享有**广义 Pythagoras**。
- **KL 是两大族的唯一交集**，同时拥有单调性与勾股几何。
- **散度全息诱导几何**：二阶导给度量，三阶导给 α-联络；KL 诱导出对偶平坦结构。

在下一节，我们进入**高阶渐近推断理论**——Edgeworth 展开与 Amari-Chentsov 张量，看曲率如何把一阶渐近理论的「同构」打碎，揭示模型间真正的差别。
