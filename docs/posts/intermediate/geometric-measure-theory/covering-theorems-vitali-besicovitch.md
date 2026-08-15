---
title: 覆盖定理（Vitali / Besicovitch）
date: 2026-08-07
---

# 覆盖定理（Vitali / Besicovitch）

<div class="epigraph">
<p>测度论的一半工作是构造测度，另一半工作是找到覆盖某个集合的好方式。</p>
<footer>—— 自 H. Federer, *Geometric Measure Theory\*（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ P. Mattila, *Geometry of Sets and Measures in Euclidean Spaces\*, Ch.2 ｜ 2026-08-07</p>
</div>

## 为什么从覆盖定理开始

上一节我们把「$s$ 维大小」定义成了覆盖成本的下确界。下确界告诉我们「存在一列足够便宜的覆盖」，却没说怎么**显式地挑出**这列覆盖——而几乎所有后续定理（密度定理、Lebesgue 微分定理、Rademacher 定理）都需要回答同一个问题：**给我一族覆盖球，我能不能从中选出「互不相交得差不多」的一批，仍然盖住原集合的大部分？** 覆盖定理就是这套「选球术」的数学化。本节讲两条主线：Vitali 型覆盖（适用于球，靠半径大小排序贪心选球）与 Besicovitch 型覆盖（适用于任意点，靠空间维数的有限重叠性质选球）。<span class="marginnote">两类覆盖定理的本质区别：Vitali 用「半径之比」控制重叠，Besicovitch 用「空间维数决定重叠数」控制重叠——前者依赖于球，后者与球的形状无关。</span>

## 1 从「一列球盖住大部分」到 Vitali 覆盖类

先看一个具体场景。设 $\mu$ 是一个 Borel 正则测度，$A \subset \mathbb{R}^n$，有一族闭球 $\mathcal{F}$ 使得 $A$ 中每个点都被任意小半径的球覆盖（即对任意 $x \in A$、任意 $\varepsilon > 0$，存在 $B(x,r) \in \mathcal{F}$ 满足 $r \lt  \varepsilon$）。我们想知道：**能不能挑出一列互不相交的球，让它们的并盖住 $A$ 除零测度以外的部分？** 这要求覆盖球足够「密」，而这种「任意点都被任意小的球覆盖」的性质正是 Vitali 覆盖类的定义。

**核心概念（Vitali 覆盖类）**：设 $\mathcal{F}$ 是一族闭球，若对每个 $x \in A$ 和每个 $\varepsilon > 0$，都存在 $B(x,r) \in \mathcal{F}$ 使得 $r \lt  \varepsilon$，则称 $\mathcal{F}$ 是 $A$ 的一个 **Vitali 覆盖类（Vitali class of coverings）**。<span class="marginnote">「任意小」三个字是关键：它把「覆盖」从静态概念升级为「在每个点附近有无穷多越来越小的球」的动态概念。这样挑选出来的球才能精确逼近每个点。</span>

Vitali 覆盖定理说的是：若 $\mathcal{F}$ 是 $A$ 的 Vitali 覆盖类，则存在 $\mathcal{F}$ 中**至多可数、两两不相交**的球列 $\{B_i\}$，使得

$$
\mathcal{L}^n\left( A \setminus \bigcup_i B_i \right) = 0
$$

也就是说，互不相交的球虽然可能盖不满 $A$（毕竟它们彼此分离），但盖不住的部分是零测度的。这条定理把「覆盖」转换成「不相交的球列」，而后者的体积可以直接相加——这是密度定理的出发点。<span class="marginnote">注意 Vitali 覆盖定理要求测度是 Lebesgue 测度 $\mathcal{L}^n$（或与它绝对连续的测度），结论是「剩下的集零测度」。对一般测度需要对结论做修正，见下面的 5r 引理。</span>

## 2 贪心选球：5r 覆盖引理

Vitali 覆盖定理的证明核心是一个独立的引理，它不要求覆盖类是 Vitali 的，只要求球的半径有上界——因此适用性更广。

**核心概念（5r 覆盖引理，五倍半径引理）**：设 $\mathcal{F}$ 是 $\mathbb{R}^n$ 中一族闭球，半径有上界。则存在 $\mathcal{F}$ 中**至多可数、两两不相交**的子族 $\{B_i\}$，使得每个 $B \in \mathcal{F}$ 都被某个半径五倍的球 $5B_i$ 覆盖（$B \subset 5B_i$ 对某个 $i$）。因此

$$
\bigcup_{B \in \mathcal{F}} B \;\subset\; \bigcup_i 5B_i
$$

证明是贪心的：先选半径最大的球，然后不断选取「与已选球都不相交」的球中半径最大的那个。当半径序列趋于 0 时，关键观察是——**任何一个被漏掉的球，必然与某个「恰好大到可以盖住它」的已选球相交**，于是用五倍半径就能把它罩住。<span class="marginnote">为什么是 5 而不是 2？因为被漏掉的球 $B$ 与已选球 $B_i$ 相交（否则它早该被选上），若 $B_i$ 是覆盖 $B$ 中心的那一档，则 $B$ 的半径至多是 $B_i$ 的半径的 2 倍，加起来 $1 + 2 \cdot 2 = 5$。这个常数证明起来就够用了，最优常数可取 3，但 5 更省事。</span>

## 3 公式解析：从 5r 引理推出 Vitali 覆盖定理

把 5r 引理用到 Vitali 覆盖定理上，只需两步不等式。设 $E = A \setminus \bigcup_i B_i$，要证 $\mathcal{L}^n(E) = 0$。

- **第一步，把 $E$ 塞进五倍球**。取定 $r_0 > 0$，令 $A_{r_0}$ 是 $A$ 中被半径小于 $r_0$ 的覆盖球盖住的点。由 5r 引理，$\mathcal{L}^n(A_{r_0})$ 被 $\sum \mathcal{L}^n(5B_i)$ 控制，而 $\mathcal{L}^n(5B_i) = 5^n \mathcal{L}^n(B_i)$。
- **第二步，让 $r_0 \to 0$**。随着 $r_0$ 收缩，参与求和的球半径都趋于 0。若 $\mathcal{L}^n(E) > 0$，取一个 $\mathcal{L}^n$-稠密点 $x \in E$，把 $E$ 限制在 $B(x, R)$ 内，则

$$
\sum_{B_i \subset B(x,R)} \mathcal{L}^n(B_i) \;=\; \mathcal{L}^n\left(\bigcup_i B_i \cap B(x,R)\right) \;\le\; \mathcal{L}^n(B(x,R))
$$

- **第三步，五倍球并集的外推**。被 $E$ 里的点选中的五倍球都落在 $B(x, R + 2r_0)$ 里（因为球心在 $B(x,R)$ 内、半径不超过 $r_0$），于是 $\mathcal{L}^n$ 版 5r 引理给出 $\mathcal{L}^n(E \cap B(x,R)) \le 5^n \mathcal{L}^n(B(x, R+2r_0))$。令 $r_0 \to 0$、再令 $R$ 任意小，右边的量级是 $R^n$，与「$x$ 是 $E$ 的密度为 1 的点」矛盾，从而 $\mathcal{L}^n(E) = 0$。<span class="marginnote">这个推导用到的「五倍球并集被扩大球吸收」与「密度点」的论证，正是下一节密度定理的预演——覆盖定理与密度定理天然互相咬合。</span>

**重点：Vitali 覆盖定理的结论本质上是「零测度剩余」型的**，它把任意覆盖族化简为一列不相交球，代价是损失一个零测度集合。这个「损失零测度」的模式在测度论里反复出现：几乎处处、可数子族、零测度集合，三件套是几何测度论的标准语言。

## 4 Besicovitch 覆盖定理：摆脱球的形状

Vitali 覆盖定理依赖「球」这个形状——五倍半径论证用到了球放大的性质。但很多时候我们需要覆盖的族是任意有界集合，甚至只是「每个点配一个足够小的开集」。这时需要 Besicovitch 覆盖定理。

**核心概念（Besicovitch 覆盖定理）**：设 $A \subset \mathbb{R}^n$，$\mathcal{F}$ 是一族以 $A$ 中点为球心的闭球。存在一个仅依赖于空间维数 $n$ 的常数 $N = N(n)$，使得可以从 $\mathcal{F}$ 中选出 $N$ 个子族 $\mathcal{F}_1, \dots, \mathcal{F}_N$，其中**每个子族内部两两不相交**，且它们的并覆盖 $A$：

$$
A \;\subset\; \bigcup_{j=1}^{N} \bigcup_{B \in \mathcal{F}_j} B
$$

关键点：**$N$ 只依赖维数 $n$，不依赖球的半径、不依赖覆盖族本身**。<span class="marginnote">在 $\mathbb{R}^1$ 中 $N = 2$ 就够（一列点用两种「左/右」的球族分开），在 $\mathbb{R}^n$ 中 $N$ 以指数级增长，但始终有限。这个「有限重叠」性质让 Besicovitch 覆盖定理适用于任意测度——见下一节的密度定理。</span>

**比较表**：Vitali 与 Besicovitch 覆盖定理的差异。

| 特征 | Vitali 覆盖定理 / 5r 引理 | Besicovitch 覆盖定理 |
| --- | --- | --- |
| 覆盖对象 | 任意球族（或 Vitali 覆盖类） | 以 $A$ 中点为球心的球族 |
| 选取结果 | 一列互不相交球 + 五倍球盖住全体 | $N(n)$ 个子族，每个内部不相交 |
| 重叠控制 | 靠半径之比（5 倍） | 靠空间维数（$N(n)$） |
| 适用测度 | Lebesgue 测度（零测度剩余） | 任意 Borel 测度 |
| 证明风格 | 贪心选最大半径 | 归纳 + 有限重叠计数 |

## 5 Besicovitch 覆盖定理为何重要

Vitali 覆盖定理的「零测度剩余」在一般测度下会失效——一个对 $\mathcal{L}^n$ 是零测度的集合，对某个奇异测度 $\mu$ 可能承载全部质量。而 Besicovitch 覆盖定理的结论是「有限个不相交子族的并**完全覆盖** $A$」而不抛掉任何部分，因此对任何测度都成立。这使它成为「逐点估计 → 积分估计」的标准桥梁。

典型应用：证明 Lebesgue 微分定理时，要估计 $\sup_{r>0} \frac{1}{|B(x,r)|} \int_{B(x,r)} |f - c|$ 这种极大函数。Hardy–Littlewood 极大函数 $Mf$ 的弱 (1,1) 不等式用 Vitali 型即可，但涉及「测度与 Lebesgue 测度无关」的 Besicovitch 覆盖版本（例如对 Hausdorff 测度 $\mathcal{H}^s$ 的密度定理），就必须用 Besicovitch。<span class="marginnote">极大函数 $Mf$ 与弱 $L^1$ 估计在第二级《实变函数与测度论》中详述；这里只需记住「极大函数不等式需要覆盖定理」这一依赖关系。</span>

**辨析｜易错点：** 不要把 Vitali 覆盖定理与 Vitali 覆盖类混为一谈。覆盖类是一种「每个点被任意小的球盖住」的结构假设；覆盖定理是在该假设下的结论。5r 覆盖引理既不用到覆盖类假设，结论也更强（盖住全体球而非零测度剩余），它是更底层的地基。另一个常见误解是把 Besicovitch 常数 $N(n)$ 当成与测度有关——它是纯几何量，只取决于 $\mathbb{R}^n$ 的维数。

## 6 覆盖定理在极大函数不等式中的应用

覆盖定理最直接的受益者是 Hardy–Littlewood 极大函数。设 $f \in L^1_{\mathrm{loc}}(\mathbb{R}^n)$，定义

$$
(Mf)(x) \;=\; \sup_{r>0} \frac{1}{|B(x,r)|} \int_{B(x,r)} |f(y)|\; \mathrm{d}y
$$

极大函数 $Mf$ 度量「$f$ 在 $x$ 附近的球平均里最大能到多大」。对 $f$ 的 $L^1$ 可积性而言，$Mf$ 不一定可积，但它满足**弱 (1,1) 不等式**：

$$
\mathcal{L}^n(\{x : Mf(x) > \lambda\}) \;\le\; \frac{C_n}{\lambda} \|f\|_{L^1}
$$

证明用覆盖定理：集合 $\{Mf > \lambda\}$ 的每个点 $x$ 都对应一个球 $B(x,r_x)$ 使 $\frac{1}{|B(x,r_x)|}\int_{B(x,r_x)}|f| > \lambda$，这族球形成一个覆盖。对 $\mathcal{L}^n$ 版本，用 5r 覆盖引理挑出互不相交的球列，把测度估计化为「球内 $f$ 积分」的估计。<span class="marginnote">这个不等式是 Lebesgue 微分定理（第 3 篇密度定理）的证明引擎，而覆盖定理是它唯一的几何原料——没有选球术，极大函数的测度估计无从下手。</span>

**比较表**：两类覆盖工具在极大函数论证中的分工。

| 工具 | 在极大函数中的角色 |
| --- | --- |
| 5r 覆盖引理 | 挑互不相交球，控制 $\mathcal{L}^n(\{Mf > \lambda\})$ |
| Besicovitch 覆盖 | 处理向量值 / 一般测度情形，不依赖球形状 |

## 7 小结

- **Vitali 覆盖类**：每个点被任意小半径的球覆盖的球族，是「覆盖」概念的动力版本。
- **5r 覆盖引理**：任意球族可挑出至多可数互不相交子族，且全体球被其五倍球覆盖；证明靠贪心选最大半径球。
- **Vitali 覆盖定理**：对 Lebesgue 测度，Vitali 覆盖类可化为互不相交球列，剩余集为零测度。
- **Besicovitch 覆盖定理**：以 $A$ 中点为球心的球族可分成 $N(n)$ 个内部互不相交的子族盖住 $A$；$N(n)$