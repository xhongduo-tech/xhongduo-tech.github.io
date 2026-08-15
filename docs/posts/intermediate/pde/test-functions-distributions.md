---
title: 基本函数空间与广义函数的定义
date: 2026-08-07
---

# 基本函数空间与广义函数的定义

<div class="epigraph">
<p>不再问「函数在一点等于什么」，而问「函数对所有光滑探针的作用是什么」。</p>
<footer>—— 洛朗·施瓦茨（Laurent Schwartz）的广义函数论</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第八章 ｜ 2026-08-07</p>
</div>

## 为什么从基本函数空间开始

δ 函数不是普通函数，但它能对任何连续函数「求值」。这个观察启发了广义函数论的根本思路：**不定义「函数本身是什么」，而定义「它对一切良好测试函数的作用是什么」**。如果两个对象对所有测试函数的作用都相同，就认为是同一个广义函数。这一节建立两个空间：**基本函数空间** $\mathcal{D}$（良好的测试函数）与**广义函数空间** $\mathcal{D}'$（$\mathcal{D}$ 上的连续线性泛函）。δ 在其中第一次有了严格身份。

## 1 基本函数空间 D

**基本函数空间（test function space）**$\mathcal{D}(\Omega)$：$\Omega$ 上**无穷光滑且支集紧**的函数全体。

**支集（support）**：$\mathrm{supp}\,f = \overline{\{x : f(x) \ne 0\}}$——函数非零点集的闭包。「支集紧」意味着函数在某个有界区域外恒为零——**测试函数是「局部存在的、光滑的、可无限求导的探针」**。<span class="marginnote">为什么要求支集紧？因为广义函数对测试函数作用时，测试函数的紧支集保证「作用」只依赖 $\Omega$ 内的局部信息，且积分无无穷远麻烦。经典例子：$\varphi(x) = e^{-1/(1-x^2)}$（$|x|<1$）、$0$（$|x|\ge1$）——光滑、支集在 $[-1,1]$，是标准测试函数。</span>

**$\mathcal{D}$ 上的收敛**：$\varphi_k \to \varphi$ 指「支集一致有界 + 各阶导数一致收敛」。这个收敛定义精确刻画了「探针族如何逼近」。

**$\mathcal{D}$ 是「最良好的函数空间」**：它不含 δ、不含间断函数，只含「怎么折腾都行」的光滑紧支函数——正因如此，它的对偶空间 $\mathcal{D}'$ 才能容纳一切「粗糙」对象。

## 2 广义函数的定义

**广义函数（distribution，分布）**：$\mathcal{D}(\Omega)$ 上的**连续线性泛函**，即映射

$$
T: \mathcal{D}(\Omega) \to \mathbb{C}, \qquad \varphi \mapsto \langle T, \varphi \rangle
$$

满足线性（$\langle T, a\varphi + b\psi\rangle = a\langle T,\varphi\rangle + b\langle T,\psi\rangle$）与连续性（$\varphi_k \to \varphi$ 时 $\langle T,\varphi_k\rangle \to \langle T,\varphi\rangle$）。全体广义函数记为 $\mathcal{D}'(\Omega)$。

**记号**：$\langle T, \varphi\rangle$ 是「广义函数 $T$ 作用在测试函数 $\varphi$ 上」的值。<span class="marginnote">直觉：广义函数是「测量仪器」，测试函数是「被测量的探针」；$\langle T,\varphi\rangle$ 是读数。普通函数是「一台只读它的积分的仪器」，δ 是「一台只读 $\varphi(0)$ 的仪器」——仪器不同，但都是合法的「测量方案」。</span>

**δ 作为广义函数**：定义

$$
\langle \delta, \varphi\rangle = \varphi(0)
$$

**δ 就是「在原点取值」这个泛函**——严格、无需「无穷大」的修辞。

## 3 正则广义函数：普通函数如何进入

**每个「局部可积」函数 $f$ 都对应一个广义函数**：

$$
\langle f, \varphi\rangle = \int_\Omega f(x)\,\varphi(x)\,dx
$$

这样的 $f$ 称为**正则广义函数（regular distribution）**；不是正则的（如 δ）称为**奇异广义函数（singular distribution）**。

**为什么可以「把 $f$ 当广义函数」？** 因为「$f$ 是什么」完全由它对所有测试函数的积分 $\int f\varphi$ 决定——**两个局部可积函数若对所有 $\varphi\in\mathcal{D}$ 有相同积分，则几乎处处相等**（变分法基本引理）。<span class="marginnote">变分法基本引理（fundamental lemma of the calculus of variations）：若 $\int f\varphi = 0$ 对所有 $\varphi\in\mathcal{D}$ 成立，则 $f = 0$（几乎处处）。它保证「用积分作用刻画函数」是诚实的——不丢信息。这条引理在本专题第十篇变分方法中还会再次登场，是「弱等价」论证的基石。</span>

**广义函数包含普通函数**：$\mathcal{D}'$ 是「局部可积函数空间」的扩张——一切普通函数都在里面，外加 δ 这类奇异对象。**广义函数 = 函数 + 更多。**

## 4 公式解析：从密度到广义函数

把「集中量」放进广义函数框架：

- **第一步，质点密度。** 质点（质量 $M$ 在原点）的密度是 $M\delta$，作为广义函数：$\langle M\delta, \varphi\rangle = M\varphi(0)$——「探针 $\varphi$ 在原点读到的值乘质量」。
- **第二步，抽样积分。** 物理中的 $\int f(x)\delta(x)dx = f(0)$ 现在严格化为 $\langle\delta, \varphi\rangle = \varphi(0)$（把 $f$ 当测试函数 $\varphi$）。
- **第三步，连续分布。** 一般质量分布（密度 $\rho$ 局部可积）对应正则广义函数 $\langle\rho,\varphi\rangle = \int\rho\varphi$——与点质量用同一个语言描述。
- **第四步，统一。** 质点、连续体、混合分布（质点 + 连续背景）全在 $\mathcal{D}'$ 里——**广义函数把「离散源」与「连续源」统一成一个框架**。

**「源」的统一语言是 PDE 基本解理论的先决条件**：$\Delta u = -\delta$ 里的 δ 是广义函数，方程本身也要在广义函数意义下理解。

## 5 广义函数的支集与局部性

**广义函数的支集（support）**：$T$ 在开集 $\omega$ 上为零，指 $\langle T,\varphi\rangle = 0$ 对所有 $\mathrm{supp}\,\varphi \subset \omega$ 成立；$\mathrm{supp}\,T$ 是「$T$ 不为零的最大开集的补」。

**δ 的支集是 $\{0\}$**——但 δ 不是「在原点取值的一个数」，它是「集中在一个点的广义函数」。**「支集大小」与「正则性」是两个正交的概念**：δ 支集最小（单点）却是奇异广义函数；常函数 $f = 1$ 支集最大（全空间）却是正则的。

**辨析｜易错点：** 不要把「广义函数在一点的值」当成普通函数的值。$T$ 作为泛函，它的「值」只在对测试函数作用时才有意义；「$T(x_0)$」一般没有定义（除非 $T$ 是正则的）。δ 的「在 $x\ne0$ 为零」应读作「δ 在 $\mathbb{R}\setminus\{0\}$ 上为零（作为广义函数）」——这是「支集在 $\{0\}$」的精确含义。**用「点值」的旧直觉理解广义函数，是初学者的最大障碍。**

## 6 广义函数的常见例子

把 $\mathcal{D}'$ 的成员盘点一遍，建立「广义函数长什么样」的直觉：

- **阶跃函数** $\Theta(x)$（$x > 0$ 取 1、$x < 0$ 取 0）：局部可积，是正则广义函数，$\langle\Theta,\varphi\rangle = \int_0^\infty\varphi\,dx$。
- **δ 及其导数** $\delta'$：定义 $\langle\delta',\varphi\rangle = -\varphi'(0)$（严格定义在下一节）。它不再是「点测量」，而是「斜率测量」——集中量的高阶信息。
- **主值积分（PV）**：$\langle\mathrm{p.v.}\,\frac{1}{x},\varphi\rangle = \lim_{\epsilon\to 0}\int_{|x|>\epsilon}\frac{\varphi(x)}{x}dx$。$1/x$ 在原点不可积，但主值给出一个合法的广义函数——**奇异但可定义**。
- **常函数 1**：正则，$\langle 1,\varphi\rangle = \int\varphi\,dx$，是「全空间质量」的测量。

**广义函数 = 连续线性测量方案的全体。** 凡是能对光滑紧支探针给出稳定读数的「对象」，都可以是广义函数；有没有「函数图像」反而不重要。

**$\mathcal{D}'$ 的规模**：$\mathcal{D}'$ 比函数空间大得多——δ、阶跃、主值、甚至「处处增长极快」的对象都能进来。一个粗略的量级感：$\mathcal{D}'$ 容纳了所有连续函数，还能容纳指数增长的对象（只要作用在紧支探针上读数就有限）。「粗糙」恰恰是它的常态，不是它的例外。

## 7 公式解析：δ 的逼近族与弱收敛

「δ 是无穷高的尖峰」是常见的口语，但严格的说法是：**δ 是许多尖峰族的公共极限**。设 $\eta \ge 0$、$\int\eta = 1$、支集紧的光滑函数，令 $\eta_\epsilon(x) = \frac{1}{\epsilon}\eta(\frac{x}{\epsilon})$——越来越窄、越来越高、面积恒为 1 的尖峰：

- **第一步，作用。** $\langle\eta_\epsilon,\varphi\rangle = \int\eta_\epsilon\varphi\,dx = \int\eta(y)\varphi(\epsilon y)\,dy$（换元 $y = x/\epsilon$）。
- **第二步，取极限。** $\epsilon \to 0$ 时 $\varphi(\epsilon y) \to \varphi(0)$，故 $\langle\eta_\epsilon,\varphi\rangle \to \varphi(0)\int\eta\,dy = \varphi(0)$。
- **第三步，结论。** $\eta_\epsilon \to \delta$（弱收敛）。**「δ 函数」不是某一片尖峰，而是所有尖峰族的公共极限。**

常见的逼近族（都以弱收敛趋向 δ）：

| 逼近族 | 显式形式 | 特点 |
| --- | --- | --- |
| 高斯 | $\frac{1}{\epsilon\sqrt{2\pi}}e^{-x^2/(2\epsilon^2)}$ | 处处光滑 |
| 拍形 | $\frac{1}{\epsilon}\eta(\frac{x}{\epsilon})$，$\eta$ 光滑紧支 | 支集紧 |
| 洛伦兹 | $\frac{1}{\pi}\frac{\epsilon}{x^2+\epsilon^2}$ | 代数衰减 |
| 抽样 | $\frac{\sin(x/\epsilon)}{\pi x}$ | 振荡、弱意义 |

**辨析｜易错点：** 弱收敛不保证「点态」或「积分」意义上的收敛。$\eta_\epsilon$ 在原点处趋于无穷（点态发散），但它对任意测试函数的作用趋于有限值 $\varphi(0)$——**「处处发散」与「弱收敛」可以同时为真**。这正是广义函数「换个角度看极限」的要点：不看函数本身，看它对探针的作用。<span class="marginnote">「弱收敛」定义在 $\mathcal{D}'$ 上：$T_k \to T$ 指对一切 $\varphi\in\mathcal{D}$ 有 $\langle T_k,\varphi\rangle \to \langle T,\varphi\rangle$。这是最宽容的收敛——它允许 δ 的逼近在广义函数意义下收敛，尽管这些逼近在普通函数意义下根本发散。第九篇「广义函数的极限与傅里叶变换」将系统使用弱收敛。</span>

**从「从极限到大模型」的视角**，δ 的逼近族很像「分布收敛 → 点状对象」的工程抽象：离散点质量、脉冲信号、集中载荷在数学上都是 δ，而它们的「实现」是不同形状的窄峰。掌握了「谁在弱意义下收敛到 δ」，就掌握了把离散源与连续源统一处理的钥匙。

## 8 小结

- 基本函数空间 $\mathcal{D}$：无穷光滑 + 紧支集的测试函数，是广义函数的「探针」。
- 广义函数：$\mathcal{D}$ 上的连续线性泛函 $T: \varphi \mapsto \langle T,\varphi\rangle$。
- δ 严格化：$\langle\delta,\varphi\rangle = \varphi(0)$，无需「无穷大」。
- 局部可积函数都是（正则）广义函数；变分法基本引理保证刻画不丢信息。
- 支集定义于「对测试函数的作用」，δ 的支集是单点 $\{0\}$。

在下一节，我们定义广义函数的极限、导数与乘子运算。
