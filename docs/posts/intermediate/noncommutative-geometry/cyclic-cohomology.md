---
title: 循环上同调（Hochschild 同调、循环复形、Connes 长正合列）
date: 2026-08-17
---

# 循环上同调

<div class="epigraph">
<p>时间存在的唯一理由，是使一切不至于同时发生。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.3; Connes《Noncommutative Geometry》Ch.III ｜ 2026-08-17</p>
</div>

## 为什么从循环上同调开始

经典流形的微分几何建立在微分形式上：$d$ 算符、de Rham 上同调、Stokes 定理。但一旦走进非交换世界，微分形式没了——它们依赖「交换点乘」的 Leibniz 规则。循环上同调（cyclic cohomology）正是 Connes 在 1981 年为填补这一空缺而发明的**非交换 de Rham 理论**：用代数本身的（高阶）迹来构造上同调，让「积分」在非交换空间上重新有意义。

循环上同调的地位怎么强调都不过分：它是 Connes–Chern 特征（把 K-理论类变成显式上同调类）的载体，是配对非交换指标公式的语言，也是全书链条中「拓扑不变量 → 显式公式」的关键一环。爱因斯坦那句话在此处变成数学：循环上同调之所以叫「循环」，正是因为它的基本对象——循环上循环——被「时间（轮换）」保持不变性约束；而周期性算子 $S$ 让它在偶维与奇维之间来回摆荡，恰如 $K_0$ 与 $K_1$ 之间的 Bott 周期性。

## 1 Hochschild 同调：第一层非交换微分

### 1.1 复形

设 $A$ 是含幺代数，$M$ 是 $A$-双模。定义链群 $C_n(A, M) = M \otimes A^{\otimes n}$，**Hochschild 边界算子** $b: C_n \to C_{n-1}$：

$$
b(m \otimes a_1 \otimes \cdots \otimes a_n) = ma_1 \otimes a_2 \otimes \cdots \otimes a_n
+ \sum_{i=1}^{n-1} (-1)^i\, m \otimes a_1 \otimes \cdots \otimes a_i a_{i+1} \otimes \cdots \otimes a_n
+ (-1)^n\, a_n m \otimes a_1 \otimes \cdots \otimes a_{n-1}
$$

<b>Hochschild 同调</b> $H_n(A, M) = \ker b / \operatorname{im} b$。

<b>Hochschild 上同调</b> $H^n(A, M)$ 用对偶复形 $C^n(A, M) = \operatorname{Hom}(A^{\otimes n}, M)$ 定义。

### 1.2 低阶含义

- $H_0(A, M) = M / [A, M]$：$M$ 商去交换子——「迹」的抽象。当 $M = A$ 时，$H_0(A) = A/[A,A]$，是「$A$ 上所有迹」的余域。
- $H^1(A)$：导子模去内导子——「非交换向量场」的等价类。
- $H^2(A)$：**形变理论**：$H^2(A)$ 度量了「把 $A$ 形变成非结合代数」的障碍；这是 Gerstenhaber 形变理论的起点，也是非交换几何与量子化（Moyal 积）的联系。

<span class="marginnote">Hochschild 同调 1945 年由 Hochschild 在《On the cohomology groups of an associative algebra》中引入，直接推广了群上同调。它对 $A = C^\infty(M)$ 给出 $H_n = \Omega^n(M)$（微分形式空间），从而把 de Rham 理论还原为代数构造——这正是非交换化的切入点。</span>

### 1.3 与光滑流形的联系

对 $A = C^\infty(M)$（紧光滑流形），有著名的定理（Hochschild–Kostant–Rosenberg 1959，简称 HKR）：

$$
H_n(C^\infty(M)) \cong \Omega^n(M)
$$

即 Hochschild 同调恰好是微分形式。**HKR 定理是「循环上同调 = 非交换 de Rham」的几何证据**：交换情形下我们得到了熟悉的微分形式，非交换情形下 Hochschild 同调就充当替代品。

## 2 循环复形与循环上同调

### 2.1 循环算子的缺失

Hochschild 复形漏掉了微分形式的一个关键性质：**外微分的闭形式在轮换下不变**（对奇数维，$\omega \mapsto (-1)^n \omega$）。Connes 的关键洞察：在 Hochschild 复形上添加一个**循环算子** $\lambda$，强制轮换不变性，就得到循环复形。

### 2.2 定义

**循环上同调（cyclic cohomology）** $HC^n(A)$：对偶复形 $C_\lambda^n(A)$ = 满足轮换不变性的 Hochschild 上链：

$$
\phi(a_0, a_1, \ldots, a_n) = (-1)^n \phi(a_1, \ldots, a_n, a_0)
$$

配上 Hochschild 上边界 $b$，取 $HC^n(A) = \ker b / \operatorname{im} b$。**循环上循环（cyclic cocycle）** 就是同时满足轮换不变性与 $b\phi = 0$ 的上链。<span class="marginnote">直观理解：循环上循环是「关于轮换不变的高阶迹」。0 阶循环上循环正是普通迹 $\tau: A \to \mathbb{C}$，满足 $\tau(ab) = \tau(ba)$；高阶循环上循环推广了这条性质到 $n+1$ 个变量。</span>

### 2.3 周期性循环上同调

把 Connes 算子 $B$（由「循环化」构造）也纳入，得到 $(b, B)$ 双复形，其总上同调是**周期性循环上同调（periodic cyclic cohomology）**

$$
HP^\pm(A) = \varinjlim_{S} HC^{n+2k}(A)
$$

其中 $S: HC^n \to HC^{n+2}$ 是周期性算子。$HP^\pm$ 以 2 为周期，与 Bott 周期性遥相呼应——这是循环上同调最具魅力的结构之一。

## 3 Connes 长正合列（SBI 序列）

### 3.1 序列本身

把 $(b, B)$ 双复形按对角线取列，得到连接 Hochschild 与循环上同调的**长正合列**（Connes 1982 年的定理）：

$$
\cdots \longrightarrow HH_n(A) \xrightarrow{\,I\,} HC_n(A) \xrightarrow{\,S\,} HC_{n-2}(A) \xrightarrow{\,B\,} HH_{n-1}(A) \longrightarrow \cdots
$$

其中 $I$ 是包含（把 Hochschild 类看作循环类），$S$ 是周期性算子，$B$ 是 Connes 算子诱导的边界。

### 3.2 解读

这条序列是非交换几何的「基本三角关系」：$HH$（切空间、微分）、$HC$（积分、闭形式）、周期化 $S$（维数提升）。它说明：

- 计算 $HC$ 时，$HH$ 是高维信息的主要来源；
- $B$ 把「积分」联系到「微分」；
- 周期性算子 $S$ 让 $HC^{n+2}$ 与 $HC^n$ 通过 $HH$ 联系起来。

## 4 公式解析：循环上循环的条件

以 1 阶循环上循环为例看条件长什么样。$\phi: A^{\otimes 2} \to \mathbb{C}$ 是循环上循环，当且仅当（记 $a_0, a_1$ 为变量，$b\phi = 0$ 展开）：

$$
\phi(a_0 a_1, a_2) - \phi(a_0, a_1 a_2) + \phi(a_2 a_0, a_1) = 0, \qquad \phi(a_0, a_1) = -\phi(a_1, a_0)
$$

分解为三步理解：

- **第一步**，第一行来自 $b\phi = 0$：$b$ 对 2 元组的作用是三段求和的代数和为零——这正是 Hochschild 边界的低阶展开。
- **第二步**，第二行是轮换不变性在 $n=1$ 时的特例（符号 $(-1)^1 = -1$）：它把「对称」换成「反对称」，与微分形式的奇偶性规则一致。
- **第三步**，综合起来：1 阶循环上循环是「反对称、满足二次分配律」的二元泛函。读者可自行验证：当 $A = C^\infty(M)$ 且 $\phi(f, g) = \int_M f\, dg \wedge \omega$（$\omega$ 是闭 1 形式）时，上述两个条件都满足——**循环上循环真的把 de Rham 闭形式藏了进来**。

**这就是「非交换积分」的严格定义**：循环上循环扮演了 $\int$ 的角色，且不依赖交换性。

## 5 与 K-理论配对：Connes–Chern 特征

循环上同调不是孤立的：它通过与 K-理论的配对获得几何意义。

**Connes–Chern 特征（Chern character）**：对 K-理论类 $[e] \in K_0(A)$（$e$ 幂等元）与偶循环上同调类 $[\phi] \in HC^{2k}(A)$，配对为

$$
\langle [\phi], [e] \rangle = \sum_i (-1)^i\, \phi\big(e, e, \ldots, e\big) \quad \text{（$2i$ 个 $e$ 的项）}
$$

对 $K_1$ 类与奇循环上同调类有类似公式（用可逆元 $u$ 的项）。这个配对把抽象的 K-群变成可计算的数——它是下一节谱三元组、以及《局部指标公式》中 JLO 上循环与 Connes–Moscovici 公式的核心引擎。<span class="marginnote">Khalkhali 的书第 4 章《Connes–Chern character》专门处理这一配对，并证明它给出 K-理论到（周期）循环上同调的「非交换 Chern 特征」——在交换情形 $A = C^\infty(M)$ 下，它精确还原经典的 Chern 特征 $\operatorname{ch}: K^0(M) \to H^{\mathrm{even}}(M)$。</span>

## 6 小结

- **Hochschild 同调** $H_n(A)$：非交换微分的基础；HKR 定理 $H_n(C^\infty(M)) = \Omega^n(M)$。
- **循环上同调** $HC^n(A)$：由轮换不变性从 Hochschild 上同调提炼出来；0 阶为普通迹。
- **周期性循环上同调** $HP^\pm$：以 2 为周期，呼应 Bott 周期性。
- **Connes 长正合列**（SBI）：$\cdots \to HH_n \to HC_n \to HC_{n-2} \to HH_{n-1} \to \cdots$，把 Hochschild 与循环上同调绑在一起。
- **Connes–Chern 特征**：K-理论类与循环上同调配对，给出可计算的不变量，是非交换指标公式的语言。

在下一节，我们将把拓扑（K-理论、循环上同调）升级为**几何**——用 Dirac 算子为代数装上「距离」与「微分」，这就是**谱三元组**，非交换黎曼流形的完整定义。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.3; Connes《Noncommutative Geometry》Ch.III; Hochschild 原始论文与 HKR 定理见 Khalkhali Ch.3 的参考文献。</span>