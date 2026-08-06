---
title: 本征值、本征向量与谱分解
date: 2026-08-07
---

# 本征值、本征向量与谱分解

<div class="epigraph">
<p>物理定律应当具有数学之美。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.1 线性代数 ｜ 2026-08-07</p>
</div>

## 为什么从本征值开始

上一篇我们认识了量子力学的两把钥匙：厄米算符描写可观测的量，幺正算符描写演化。但「厄米算符的本征值是实数」这句话我们还只是预告，没有动手算过。今天这篇就是把钥匙彻底磨亮的日子——我们要学会一件事：**把一个算符拆开来看**。

拆法的名字叫**谱分解（spectral decomposition）**：任何一个"好"的算符（尤其是厄米算符和幺正算符），都能写成

$$
A = \sum_i \lambda_i \, |i\rangle\langle i|
$$

的形式——一堆实数（或复数）$\lambda_i$ 乘以一堆投影算符 $|i\rangle\langle i|$ 的加权和。这看起来像把一台复杂的机器拆成一排简单零件，每个零件只做一个动作。这么拆为什么值得？因为量子力学的**测量公设**（第四篇将专门讲）几乎就是谱分解的物理朗读：测得某个结果的概率，正比于对应的投影系数。拆算符，就是拆测量。

## 1 本征值与本征向量

**本征向量与本征值（eigenvector and eigenvalue）**：如果非零矢量 $|\psi\rangle$ 与数 $\lambda$ 满足

$$
A\,|\psi\rangle = \lambda\, |\psi\rangle
$$

就称 $|\psi\rangle$ 是算符 $A$ 的**本征向量**，$\lambda$ 是它对应的**本征值**。几何直觉是：$A$ 对这个矢量「只拉伸、不转向」，拉伸的倍数就是 $\lambda$。<span class="marginnote">本征值 $\lambda$ 通常默认为复数；只有当 $A$ 是厄米算符时，$\lambda$ 才必然是实数。所以「实数本征值」不是普遍规律，而是厄米性的特权，这正是它被选中描写物理量的原因。</span>

求本征值的方法是解**特征方程（characteristic equation）**：

$$
\det(A - \lambda I) = 0
$$

左边是 $\lambda$ 的 $n$ 次多项式，叫**特征多项式**。对 $n$ 维空间，它一般有 $n$ 个根（计入重数），因此一个 $n\times n$ 矩阵通常有 $n$ 个本征值——这正是代数基本定理说的：$n$ 次多项式有 $n$ 个复数根。<span class="marginnote">如果某个本征值出现了多次（例如 $\lambda=1$ 是重根），就说它是<strong>简并的（degenerate）</strong>。简并时，对应本征向量不止一个方向，而是张成一个<strong>本征子空间</strong>——子空间里任选一组正交基都是合法的本征向量。量子力学里简并对应「多个态有同一个能量」，是物理中的常见景象。</span>

**辨析｜易错点：** 不要把「本征向量」和「基矢量」混为一谈。本征向量是相对于**某一个特定算符**而言的——$|0\rangle$ 是 $\sigma_z$ 的本征向量，却不是 $\sigma_x$ 的本征向量。一个矢量可以同时是很多算符的本征向量（例如 $|0\rangle$ 同时是 $\sigma_z$、$I$、$Z$ 的本征向量），但没有任何矢量能天生「本征」——它总要对某个算符来说才是。

## 2 厄米算符的三条特权

厄米算符 $A^\dagger = A$ 在谱问题上享有三条特权，它们合起来保证了「拆算符」这件事永远行得通：

- **本征值全是实数**。设 $A|\psi\rangle = \lambda|\psi\rangle$，两边取内积 $\langle\psi|A|\psi\rangle = \lambda\langle\psi|\psi\rangle = \lambda$（取 $|\psi\rangle$ 归一）。左边是实数（上一篇已证厄米算符对角元为实），所以 $\lambda$ 是实数。
- **不同本征值的本征向量正交**。设 $A|i\rangle = \lambda_i|i\rangle$，$A|j\rangle = \lambda_j|j\rangle$ 且 $\lambda_i \neq \lambda_j$，则 $\langle i | A | j \rangle = \lambda_j\langle i|j\rangle$；又因 $A$ 厄米，左边同时等于 $\langle j|A|i\rangle^* = \lambda_i^*\langle i|j\rangle = \lambda_i\langle i|j\rangle$。于是 $(\lambda_i - \lambda_j)\langle i|j\rangle = 0$，而 $\lambda_i\neq\lambda_j$，故 $\langle i|j\rangle = 0$。
- **可被一组标准正交本征基完全对角化**。也就是说，存在一组标准正交基 $\{|i\rangle\}$，使 $A$ 在这组基下是对角矩阵，对角元恰为 $\lambda_i$。

这三条特权可以浓缩成一句：**厄米算符有一套"专属"的标准正交基，在这套基下它只是一张写满实数的对角表。** 幺正算符也有类似特权，只是本征值换成了单位圆上的复数。凡是同时满足「可用正交基对角化」的算符（厄米、幺正、以及更一般的**正规算符** $A^\dagger A = AA^\dagger$），都配得上谱分解。

## 3 投影算符与谱分解

拆算符需要一种新零件：**投影算符（projector）**。对单位矢量 $|i\rangle$，定义

$$
P_i = |i\rangle\langle i|
$$

它把任意态 $|\psi\rangle$ 变成 $|\psi\rangle$ 在 $|i\rangle$ 方向上的分量 $(\langle i|\psi\rangle)\,|i\rangle$——这就是一次「沿某方向取分量」的投影。<span class="marginnote">投影算符有两个一眼可辨的特征：$P_i^2 = P_i$（投影两次等于投影一次）以及本征值只能是 0 或 1。在量子测量的语言里，$P_i$ 对应「问到 $|i\rangle$ 方向这一问题的答案」，0/1 正是「是/否」。</span>

把一组标准正交基 $\{|i\rangle\}$ 的所有投影加起来，就得到恒等算符——这叫**完备性关系（completeness relation）**：

$$
\sum_i |i\rangle\langle i| = I
$$

它说的是：把每个方向的"影子"全部加起来，正好还原整个矢量。现在把本征值和投影拼起来，就得到**谱分解（spectral decomposition）**：

$$
A = \sum_i \lambda_i \, |i\rangle\langle i|
$$

验证很直接：两边同时作用于本征向量 $|j\rangle$，左边得 $\lambda_j|j\rangle$，右边除 $j$ 项外全为零、第 $j$ 项得 $\lambda_j|j\rangle$，两边相等。**谱分解的本质是：选对基，一切算符都变简单。**

## 4 算符函数：谱分解的赠品

谱分解还有一个隐藏回报：它让我们给算符定义「函数」。对任何函数 $f$，定义

$$
f(A) = \sum_i f(\lambda_i)\, |i\rangle\langle i|
$$

即把本征值逐个送进 $f$，投影结构保持不变。这套定义下，$e^{A}$、$\sqrt{A}$、$A^{-1}$ 全都顺理成章。上一篇的指数桥 $e^{-iHt/\hbar}$ 就是最关键的实例：只要 $H$ 的本征值 $\lambda_i$ 已知，演化算符就立刻写成

$$
U(t) = e^{-iHt/\hbar} = \sum_i e^{-i\lambda_i t/\hbar}\, |i\rangle\langle i|
$$

每个本征态 $|i\rangle$ 只是整体乘上一个模长为 1 的相位 $e^{-i\lambda_i t/\hbar}$——**演化在能量本征态上"不做别的"，只转相位**。这是量子动力学最优雅的读法，也是后面相位估计算法（第五篇）的全部灵感来源。

## 5 公式解析：把 Pauli X 彻底拆开

我们完整算一遍非对角的例子：对 Pauli 矩阵 $\sigma_x = \begin{pmatrix}0&1\\1&0\end{pmatrix}$ 做谱分解。它不在对角形里，正适合练手。

- **第一步，解特征方程求本征值**：

$$
\det(\sigma_x - \lambda I) = \begin{vmatrix} -\lambda & 1 \\ 1 & -\lambda \end{vmatrix} = \lambda^2 - 1 = 0
$$

得 $\lambda_+ = +1$，$\lambda_- = -1$。果然都是实数——因为 $\sigma_x$ 是厄米的。

- **第二步，逐本征值求本征向量**。对 $\lambda_+ = 1$，解 $(\sigma_x - I)|\psi\rangle = 0$，即 $\begin{pmatrix}-1&1\\1&-1\end{pmatrix}\begin{pmatrix}a\\b\end{pmatrix}=0$，得 $a=b$，归一化后是 $|+\rangle = \frac{1}{\sqrt2}(|0\rangle + |1\rangle)$。对 $\lambda_- = -1$，得 $a=-b$，即 $|-\rangle = \frac{1}{\sqrt2}(|0\rangle - |1\rangle)$。

- **第三步，写出两个投影算符**：

$$
|+\rangle\langle+| = \frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}, \qquad
|-\rangle\langle-| = \frac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}
$$

- **第四步，按谱分解公式重组**：

$$
\sigma_x = (+1)\,|+\rangle\langle+| + (-1)\,|-\rangle\langle-|
= \frac12\begin{pmatrix}1&1\\1&1\end{pmatrix} - \frac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}
= \begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

加号一开一合，正好还原 $\sigma_x$ 本身。**这一步不是循环论证，而是确认"拆开再装回"没有信息损失**——谱分解提供了一套等价、但对理解测量与演化友好得多的表达。

## 6 小结

- **本征方程** $A|\psi\rangle = \lambda|\psi\rangle$ 与**特征方程** $\det(A - \lambda I) = 0$ 是求谱的基本工具；$n$ 维矩阵一般有 $n$ 个本征值（计入重数）。
- **厄米算符的三条特权**：本征值实数、异值本征向量正交、存在标准正交本征基完全对角化。
- **投影算符** $P_i = |i\rangle\langle i|$ 满足 $P_i^2 = P_i$，本征值只有 0 和 1；完备性 $\sum_i |i\rangle\langle i| = I$。
- **谱分解** $A = \sum_i \lambda_i |i\rangle\langle i|$ 把算符拆成本征值 × 投影的加权和，是测量公设的数学核心。
- **算符函数** $f(A) = \sum_i f(\lambda_i)|i\rangle\langle i|$，演化 $e^{-iHt/\hbar}$ 的本征态只是转相位。

在下一节，我们要回答：单个系统学会了，两个、三个、$n$ 个系统的状态空间怎么拼？这个问题的答案是**张量积**——它也是下一章纠缠的出生证。
