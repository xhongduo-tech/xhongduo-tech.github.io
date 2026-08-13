---
title: Lagrange 对偶：凸优化对偶问题的构造
date: 2026-08-07
---

# Lagrange 对偶：凸优化对偶问题的构造

<div class="epigraph">
<p>宇宙的构造最为完美，是造物主最智慧的作品，宇宙间没有任何事物不遵循某种极大或极小法则。</p>
<footer>—— 莱昂哈德 · 欧拉（Leonhard Euler）</footer>
</div>

<div class="article-byline">
<p>第二级 · 凸分析 ｜ Boyd《Convex Optimization》§5.1；Rockafellar《Convex Analysis》第28章 ｜ 2026-08-07</p>
</div>

## 为什么从 Lagrange 对偶开始

共轭与双共轭给了我们「用仿射下界重构凸函数」的哲学，
现在把它应用到**约束优化**。
面对问题「最小化 $f$，同时满足约束」，
Lagrange 对偶的做法是：把约束**罚进**目标，
形成拉格朗日函数；
对原始变量取极小，得到只关于对偶变量的**对偶函数**；
再最大化这个对偶函数。这套「先罚、再缩、再抬」的流程给出一个**永远成立的下界**，
并把原问题换成另一个（往往更好解）的问题。
<span class="marginnote">在「从极限到大模型」主线上，几乎所有带约束的机器学习问题（SVM 的对偶、带正则的约束最小化、博弈论中的 minimax）都以 Lagrange 对偶为骨架。
对偶变量本身还有经济学意义——约束的「影子价格」。
</span>

## 1 拉格朗日函数

考虑标准形式

$$\min f(x) \quad \text{s.t.} \quad g_i(x) \le 0,\; i=1,\dots,m, \qquad h_j(x) = 0,\; j=1,\dots,p$$

**拉格朗日函数（Lagrangian）**：

$$L(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i\, g_i(x) + \sum_{j=1}^{p} \nu_j\, h_j(x)$$

定义域为 $\operatorname{dom} f \times \mathbb{R}^m_+ \times \mathbb{R}^p$，
即**不等式约束的对偶变量 $\lambda_i \ge 0$**，
等式约束的对偶变量 $\nu_j$ 无符号限制。
<span class="marginnote">为什么 $\lambda$ 必须非负？
因为对可行点 $g_i(x) \le 0$，
若 $\lambda_i \ge 0$ 则 $\lambda_i g_i(x) \le 0$，
罚项不奖励也不过度惩罚可行解；
若 $\lambda_i \lt  0$，罚项可能反过来压低目标，
破坏「下界」性质。
符号约定是 Lagrange 对偶的第一道纪律。
</span>

**重点：** $L$ 关于 $x$ 是原目标与约束的线性组合，
关于 $(\lambda, \nu)$ 则是**仿射**的（因为 $g_i, h_j$ 与 $x$ 有关、与 $(\lambda,\nu)$ 无关）。
这个「对 $x$ 复杂、对 $(\lambda,\nu)$ 仿射」的双面结构，
是后面所有性质的根源。

## 2 对偶函数与下界性质

**对偶函数（dual function）**：

$$g(\lambda, \nu) = \inf_{x} L(x, \lambda, \nu)$$

**重点：** 由于 $L$ 对 $(\lambda, \nu)$ 是仿射的，逐点 $\inf$ 后 $g$ 是**一族仿射函数的下确界**——因此 $g$ 是**凹函数**，无论原问题是否凸！这是对偶函数「免费凹」的来源。<span class="marginnote">对偶函数定义域为 $\operatorname{dom} g = \{(\lambda,\nu) \mid \lambda \succeq 0,\ g(\lambda,\nu) > -\infty\}$。注意 $\inf$ 是在<strong>所有</strong> $x$ 上取的（包括不可行的 $x$）——正是这种「放任」让下界性质成立。</span>

**下界性质（weak duality）**：对任意 $\lambda \succeq 0$ 与任意 $\nu$，

$$g(\lambda, \nu) \le p^*$$

其中 $p^*$ 是原问题最优值。
证明一行：对可行点 $\tilde x$（满足全部约束），
$g(\lambda,\nu) \le L(\tilde x, \lambda, \nu) \le f(\tilde x)$，
再对可行 $\tilde x$ 取下确界。

## 3 对偶问题

既然 $g$ 是凹函数，最大化它就是凸优化（最大化凹 = 最小化其负）：

$$\max_{\lambda \succeq 0,\, \nu} g(\lambda, \nu)$$

称为**对偶问题**，其最优值记 $d^*$。下界性质给出

$$d^* \le p^*$$

即**弱对偶（weak duality）**永远成立。
<span class="marginnote">弱对偶是免费的午餐：<strong>任何</strong>对偶可行点都给原问题一个下界，不要求任何凸性、任何约束规格。
这个「永远成立的下界」让对偶问题在非凸情形也能用于<strong>下界估计</strong>（分支定界法、对偶界），
是全局优化的重要工具。
</span>

**辨析｜易错点：** 对偶问题的变量是 $(\lambda, \nu)$，
不是 $x$；
原问题的维度可能远高于对偶问题（约束少时对偶特别「瘦」）。
另外，**对偶问题永远是凸的**（最大化凹函数），
哪怕原问题非凸——这常让初学者惊讶。
**「原问题凹 + 对偶问题凸」是对偶理论最不对称也最有用的事实。**

## 4 公式解析：下界性质的完整证明

- **第一步，任取对偶可行 $(\lambda, \nu)$**：$\lambda \succeq 0$，$g(\lambda, \nu) > -\infty$。
- **第二步，对偶函数定义**：$g(\lambda,\nu) = \inf_x L(x,\lambda,\nu) \le L(\tilde x, \lambda, \nu)$ 对任意 $\tilde x$ 成立。
- **第三步，取可行点**：若 $\tilde x$ 可行，则 $g_i(\tilde x) \le 0$、$h_j(\tilde x) = 0$。因 $\lambda_i \ge 0$，$\lambda_i g_i(\tilde x) \le 0$，罚项不增目标：$L(\tilde x, \lambda, \nu) = f(\tilde x) + \sum \lambda_i g_i(\tilde x) \le f(\tilde x)$。
- **第四步，收紧**：$g(\lambda,\nu) \le f(\tilde x)$ 对**所有**可行 $\tilde x$ 成立，取下确界得 $g(\lambda,\nu) \le p^*$。对 $(\lambda,\nu)$ 再取上确界即 $d^* \le p^*$。

**整条证明只用了一件事：$\lambda \succeq 0$ 保证罚项符号正确。**
符号约定不是风格问题，而是下界性质成立的充分必要条件——这也解释了为什么不等式约束的对偶变量必须非负。

## 5 对偶的实例：从约束到影子价格

把 Lagrange 对偶用在一个简单但完整的例子上，
看清每个部件：

**例子：产能受限的资源分配。**
你有两种产品，利润 $f(x) = -x_1 - 2x_2$（最小化负利润），
产能约束 $x_1 + x_2 \le 1$，
非负 $x \ge 0$。
拉格朗日：

$$L(x, \lambda) = -x_1 - 2x_2 + \lambda (x_1 + x_2 - 1), \qquad \lambda \ge 0$$

对偶函数 $g(\lambda) = \inf_{x \ge 0} L(x, \lambda)$。对 $\lambda$ 分别求 $x_1, x_2$ 的系数：$x_1$ 的系数是 $-1 + \lambda$，$x_2$ 的是 $-2 + \lambda$。<span class="marginnote">当 $\lambda \lt  1$ 时，两个系数都为负，$\inf_{x\ge 0}$ 把 $x$ 推向 $+\infty$，$g(\lambda) = -\infty$（无意义）；当 $\lambda \ge 2$ 时，系数非负，最优取 $x = 0$，$g(\lambda) = -\lambda$；中间段要分情况。实际对偶最优在 $\lambda^* \in [1, 2]$ 处取到，给出 $d^* = -2$，与 $p^* = -2$ 相等——强对偶成立。</span>

**影子价格的解读。** 最优 $\lambda^*$ 是产能约束的影子价格：产能每增加一个单位，最优利润增加 $\lambda^*$。在这个例子里，增加产能能多生产高利润产品 2，故 $\lambda^* = 2$ 正好是第二产品的边际利润。<span class="marginnote">「$\lambda$ = 约束的边际价值」是经济学里对偶变量的标准解释：$d^* = p^*$ 时，$\lambda^*$ 衡量「放松约束能带来多少收益」。这解释了为什么大厂做产能/预算分配时都在解对偶问题——对偶变量就是资源的定价。</span>

**对偶问题为什么总是凸。** 无论原问题凸不凸，$g(\lambda, \nu)$ 都是凹函数，最大化凹函数是凸问题。**这意味着对偶问题可以交给标准的凸优化求解器**——即使原问题非凸，对偶问题仍给出一个可计算的下界。这是「对偶松弛」在全局优化里被广泛使用的根本原因：**对偶总是凸的，哪怕原问题不是。**

**辨析｜易错点：** 对偶函数的 $\inf_x$ 在**无界**时给出 $-\infty$，这个点不属于 $\operatorname{dom} g$（有效域）。写对偶问题时，先算 $\operatorname{dom} g$——**对偶可行域不是「自动」的**，它由「$\inf_x$ 有限」定义。上面例子里 $\lambda \lt  1$ 时 $g = -\infty$，那些 $\lambda$ 不进入对偶问题。

## 6 小结

- **拉格朗日函数** $L = f + \sum \lambda_i g_i + \sum \nu_j h_j$，$\lambda \succeq 0$、$\nu$ 自由。
- **对偶函数** $g(\lambda,\nu) = \inf_x L(x,\lambda,\nu)$：一族仿射函数的下确界，**永远凹**。
- **下界性质**：$g(\lambda,\nu) \le p^*$ 对所有对偶可行点成立，证明只需 $\lambda \succeq 0$。
- **对偶问题** $\max_{\lambda \succeq 0, \nu} g(\lambda,\nu)$ 永远是凸优化；**弱对偶** $d^* \le p^*$