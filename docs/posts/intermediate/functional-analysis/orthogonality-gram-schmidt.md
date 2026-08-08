---
title: 正交、正交系与格拉姆-施密特正交化
date: 2026-08-07
---

# 正交、正交系与格拉姆-施密特正交化

<div class="epigraph">
<p>垂直是内积空间最温柔的馈赠：一旦有了它，分解、投影、级数全都自然生长。</p>
<footer>—— 哈尔莫斯（Paul Halmos），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.4 ｜ 2026-08-07</p>
</div>

## 为什么正交如此重要

欧氏几何里「垂直」是日常词汇，可抽象到函数空间，「垂直」意味着什么？**$f \perp g$ 就是 $\langle f, g\rangle = 0$**——两个函数「内积为零」。信号处理里，正交信号互不干扰（傅里叶分解正是把信号拆成相互正交的分量）；概率论里，正交就是「不相关」；机器学习里，正交特征互不冗余。「垂直」让内积空间有了**分解的能力**：任意向量都能「拆」成正交分量的和，每个分量承担独立的角色。本节把「正交」「正交系」与**Gram-Schmidt 正交化**系统化——这是第四章的几何引擎。<span class="marginnote">Gram-Schmidt 过程在线性代数里你可能见过：把一组无关向量变成正交向量组。它在无穷维的用武之地更大：<strong>从任意线性无关函数列（比如幂函数 $1, t, t^2, \ldots$）构造正交函数列</strong>——这将在第十章造出正交多项式（勒让德、切比雪夫、厄米等）。</span>

## 1 正交与正交系

**定义**：内积空间 $H$ 中的两个向量 $x, y$ 称为**正交（orthogonal）**，记作 $x \perp y$，若 $\langle x, y\rangle = 0$。

**正交系（orthogonal system）**：一族向量 $\{x_\alpha\}_{\alpha \in A}$ 若两两正交（$\alpha \neq \beta$ 时 $\langle x_\alpha, x_\beta\rangle = 0$）且不含零向量，则称为正交系。若还满足 $\|x_\alpha\| = 1$ 对一切 $\alpha$，则称为**规范正交系（orthonormal system）**。

**基本性质**：

正交系是线性无关的：若 $\sum c_i x_i = 0$，对每个 $j$ 取内积 $\langle \cdot, x_j\rangle$ 得 $c_j\|x_j\|^2 = 0$，故 $c_j = 0$。<span class="marginnote">这个「取内积消去交叉项」的证明是正交性威力的第一次展示：<strong>正交让「线性组合 = 0」变成「每个系数 = 0」</strong>——因为交叉项全部为零，方程退化成单个变量。</span>
- **勾股定理**：$x \perp y$ 时 $\|x + y\|^2 = \|x\|^2 + \|y\|^2$。推广：正交系的有限和满足 $\|\sum x_i\|^2 = \sum \|x_i\|^2$。

**核心要点：正交系是「无穷维的标准正交基的候选者」**。它天然线性无关，且分解极其简单（取内积即可读出系数）。

## 2 正交分解：向量沿正交系的投影

设 $\{e_1, \ldots, e_n\}$ 是规范正交系。对任意 $x \in H$，定义

$$
P x = \sum_{i=1}^n \langle x, e_i\rangle e_i
$$

这是 $x$ 在 $\operatorname{span}\{e_i\}$ 上的**正交投影**。它满足两条关键性质：

- **最佳逼近（投影定理雏形）**：$x - Px \perp \operatorname{span}\{e_i\}$，且 $Px$ 是 $x$ 在 $\operatorname{span}\{e_i\}$ 中的最佳逼近（距离最近的点）。
- **系数就是内积**：$x$ 沿 $e_i$ 的分量恰好是 $\langle x, e_i\rangle$——**傅里叶系数的抽象定义**。

**例**：$L^2[-\pi,\pi]$ 中 $e_1 = \frac{\sin t}{\sqrt\pi}$、$e_2 = \frac{\sin 2t}{\sqrt\pi}$，$x$ 在 $\operatorname{span}\{e_1,e_2\}$ 上的投影是 $\langle x,e_1\rangle e_1 + \langle x,e_2\rangle e_2$——这就是**截断傅里叶级数**。<span class="marginnote">把 $e_i$ 换成三角函数、把 $\langle x, e_i\rangle$ 换成傅里叶系数，$Px$ 就是「只保留前 $n$ 项谐波」的近似——<strong>傅里叶级数 = 正交投影的无穷极限</strong>。这是第四章「傅里叶级数的抽象观点」一节的预告。</span>

## 3 Gram-Schmidt 正交化

**定理（Gram-Schmidt 正交化）**：设 $\{x_1, x_2, \ldots\}$ 是内积空间 $H$ 中的线性无关序列，则存在规范正交系 $\{e_1, e_2, \ldots\}$，使对每个 $n$，

$$
\operatorname{span}\{x_1, \ldots, x_n\} = \operatorname{span}\{e_1, \ldots, e_n\}
$$

（即「前 $n$ 个生成相同的子空间」）。构造递归如下：

$$
e_1 = \frac{x_1}{\|x_1\|}, \qquad e_n = \frac{x_n - \sum_{i=1}^{n-1}\langle x_n, e_i\rangle e_i}{\big\|x_n - \sum_{i=1}^{n-1}\langle x_n, e_i\rangle e_i\big\|}
$$

**理解**：第 $n$ 步先把 $x_n$ 中「落在前 $n-1$ 个方向上的分量」全部减掉，剩下的是「全新的方向」，归一化即得 $e_n$。<span class="marginnote">几何直觉：把 $x_n$ 投影到已经搭好的正交架上，减去投影（投影 = 各方向分量的和），剩余就是「垂直方向」——这正是第 2 节正交投影的反向使用。Gram-Schmidt 与 QR 分解（线性代数）是同一件事：QR 分解 $A = QR$ 里的 $Q$ 列向量就是 $e_i$。</span>

## 4 公式解析：Gram-Schmidt 的一步

把第 $n$ 步的关键运算拆开：

$$
y_n = x_n - \sum_{i=1}^{n-1} \langle x_n, e_i\rangle e_i
$$

- **第一步，投影**：$\sum_{i=1}^{n-1}\langle x_n, e_i\rangle e_i$ 是 $x_n$ 在 $\operatorname{span}\{e_1,\ldots,e_{n-1}\}$ 上的正交投影（第 2 节定义）。
- **第二步，减投影**：$y_n = x_n - (\text{投影})$。关键性质：$y_n \perp e_j$ 对一切 $j < n$——因为

$$
\langle y_n, e_j\rangle = \langle x_n, e_j\rangle - \langle x_n, e_j\rangle = 0
$$

（交叉项 $\langle e_i, e_j\rangle$ 当 $i \neq j$ 时为零，只剩 $j$ 那一项）。
- **第三步，归一化**：$e_n = y_n/\|y_n\|$。只要 $\{x_i\}$ 线性无关，$y_n \neq 0$（否则 $x_n$ 落在前 $n-1$ 个的生成空间里，线性相关）。

**关键**：正交化一次只处理「减掉旧方向」，而「减掉旧方向」的操作本身就是正交投影——**正交化就是反复使用投影**。这正是「投影」概念在全书中的第二次大显身手（第一次是第 2 节的分解）。

## 5 应用：构造正交多项式

Gram-Schmidt 最著名的应用之一：把幂函数 $\{1, t, t^2, \ldots\}$ 在 $L^2[-1,1]$（权函数 $1$）或带权 $L^2$ 空间里正交化，得到**勒让德多项式（Legendre polynomials）** $P_n(t)$：

- $L^2[-1,1]$ 中 $\{1, t, t^2, \ldots\}$ Gram-Schmidt 正交化 → 勒让德多项式。
- 带权 $e^{-t^2}$ 的 $L^2(\mathbb{R})$ → 厄米多项式（量子谐振子）。
- 带权 $e^{-t}$ 的 $L^2(0,\infty)$ → 拉盖尔多项式。

这些多项式是数学物理方程、数值逼近（Gauss 求积）的基石。<span class="marginnote">正交多项式的核心性质：<strong>$P_n$ 与一切次数低于 $n$ 的多项式正交</strong>（因为它与 $P_0,\ldots,P_{n-1}$ 张成的空间正交，而那空间恰好含所有低次多项式）。这个「最高次带头」的性质让它们在逼近论里极其好用，第十章将专门讨论。</span>

## 6 例题精讲：Gram-Schmidt 的三个具体计算

**例题一：在 $\mathbb{R}^2$ 上正交化 $\{(1,1), (1,0)\}$**。

- $e_1 = (1,1)/\sqrt2$。
- $y_2 = (1,0) - \langle(1,0), e_1\rangle e_1 = (1,0) - \tfrac{1}{\sqrt2}\cdot\tfrac{1}{\sqrt2}(1,1) = (1/2, -1/2)$。
- $e_2 = (1,-1)/\sqrt2$。检验：$\langle e_1, e_2\rangle = (1-1)/2 = 0$。

**例题二：在 $L^2[-1,1]$ 上正交化 $\{1, t\}$（勒让德前两项）**。

- $e_1(t) = 1/\sqrt2$（$\|1\|_2 = \sqrt2$）。
- $y_2(t) = t - \langle t, e_1\rangle e_1 = t$（$\int_{-1}^1 t = 0$，$t \perp 1$）。
- $e_2(t) = t/\sqrt{2/3} = \sqrt{3/2}\,t$——勒让德多项式 $P_1$ 的归一化。

**例题三：正交化一个「几乎线性相关」的组**。

- $\{v_1, v_2\}$ 中 $v_2$ 与 $v_1$ 接近平行时，$y_2$ 很小，$e_2$ 数值不稳定。
- 这是 Gram-Schmidt 的数值弱点：**修正的 Gram-Schmidt（MGS）** 每次先减再归一，稳定性更好。
- 有限元、QR 分解里都用 MGS 保证数值稳健。

**核心要点**：Gram-Schmidt 的每一步都是「减投影 + 归一化」——正交化就是反复做正交投影。

**辨析｜易错点：** 正交化不改变「前 $k$ 个张成的子空间」——$\operatorname{span}\{x_1..x_k\} = \operatorname{span}\{e_1..e_k\}$。这是它在逼近论里好用的根本。


## 7 小结

- **正交**：$\langle x,y\rangle = 0$；正交系线性无关，勾股定理 $\|x+y\|^2 = \|x\|^2 + \|y\|^2$。
- **正交投影**：$Px = \sum\langle x,e_i\rangle e_i$，是 $x$ 在 $\operatorname{span}\{e_i\}$ 的最佳逼近；系数 $\langle x,e_i\rangle$ 是抽象傅里叶系数。
- **Gram-Schmidt**：$e_n = \frac{x_n - \sum_{i<n}\langle x_n,e_i\rangle e_i}{\|x_n - \sum_{i<n}\langle x_n,e_i\rangle e_i\|}$，从线性无关列造规范正交系。
- **正交化 = 反复投影**：先减旧方向、再归一化。
- **应用**：幂函数正交化 ⟹ 勒让德/厄米/拉盖尔正交多项式，逼近论与数学物理的基石。

在下一节，我们把「正交投影」推到极致——**正交分解定理**：任意向量都能唯一地分解为「子空间上的投影 + 垂直余量」，这是 Hilbert 空间几何的核心定理。
