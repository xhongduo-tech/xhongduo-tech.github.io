---
title: 量子力学中的算子：可观测量与自伴算子
date: 2026-08-07
---

# 量子力学中的算子：可观测量与自伴算子

<div class="epigraph">
<p>位置、动量、能量——量子世界的每个物理量都是 Hilbert 空间上的自伴算子。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§10.7 ｜ 2026-08-07</p>
</div>

## 为什么具体算子值得专章

上一节是量子力学的公理框架；这一节看**具体的物理算子**——位置、动量、角动量、能量。每个都是自伴算子，但形态各异：有的有界（自旋）、有的无界（动量、Hamiltonian）、有的谱离散（束缚态）、有的谱连续（散射态）。理解这些算子的自伴性与谱结构，就是理解量子系统的能谱与测量。泛函分析的工具（闭算子、谱分解、自伴判别）在这里全部派上用场。<span class="marginnote">量子算子的谱结构直接对应物理：<strong>离散谱 = 束缚态能级，连续谱 = 散射态</strong>。氢原子的能级（$E_n = -13.6/n^2$ eV）是 Hamiltonian 的离散谱，自由电子的能量是连续谱——一个算子的谱同时容纳两者，这正是谱分解（§9.7）的意义。</span>

## 1 位置与动量算子

**位置算子**：$\hat x$：$L^2(\mathbb{R}) \to L^2(\mathbb{R})$，$(\hat x\psi)(x) = x\psi(x)$（乘法算子）。

- **无界、自伴**：$D(\hat x) = \{\psi : x\psi \in L^2\}$。$\sigma(\hat x) = \mathbb{R}$（连续谱）。
- 自伴性：$\langle x\psi,\varphi\rangle = \int x\psi\bar\varphi = \langle\psi, x\varphi\rangle$（$x$ 实值）。

**动量算子**：$\hat p = -i\hbar\frac{d}{dx}$，$D(\hat p) = \{\psi : \psi \text{ 绝对连续}, \psi' \in L^2\}$。

- **无界、自伴**（在 $L^2(\mathbb{R})$ 上）。$\sigma(\hat p) = \mathbb{R}$（连续谱）。
- 特征函数 $e^{ipx/\hbar}$（广义）——不在 $L^2$，需谱分解处理。<span class="marginnote">动量算子的自伴性依赖定义域：<strong>在 $L^2(\\mathbb{R})$ 上 $\\hat p$ 自伴，但在 $L^2(0,1)$（加边界条件）上可能只是对称</strong>（§5.7 的周期 vs Dirichlet 情形）。「自伴 vs 对称」的讨论在量子力学里是实打实的：只有自伴才有实的、物理的谱。</span>

**位置与动量的关系**：$\hat x$ 与 $\hat p$ 满足**正则对易关系（canonical commutation relation）**

$$
[\hat x, \hat p] = \hat x\hat p - \hat p\hat x = i\hbar\, I
$$

这是 Heisenberg 不确定性原理的算子根源（下节）。

## 2 角动量与自旋算子

**轨道角动量**：$\hat L = \hat r \times \hat p$，分量满足

$$
[\hat L_x, \hat L_y] = i\hbar \hat L_z
$$

（循环排列）。特征值：$\hat L_z$ 的谱 = $\{m\hbar : m \in \mathbb{Z}\}$（离散点谱）。

**自旋算子**：自旋 $\frac12$ 的态空间是 $\mathbb{C}^2$，自旋算子由 **Pauli 矩阵**

$$
\sigma_x = \begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}, \quad \sigma_y = \begin{pmatrix}0 & -i \\ i & 0\end{pmatrix}, \quad \sigma_z = \begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}
$$

给出（$\hat S_i = \frac{\hbar}{2}\sigma_i$）。每个 $\sigma_i$ 自伴，特征值 $\pm 1$（自旋 $\pm\hbar/2$）。<span class="marginnote">自旋算子是有界自伴算子的最简单例子：<strong>Pauli 矩阵是 Hermite 矩阵，谱是 $\\{\\pm 1\\}$（离散）</strong>。有限维情形下，自伴算子的谱就是 Hermite 矩阵的特征值——第四章的伴随理论与矩阵完全一致。自旋是「有限维量子系统」的典型。</span>

## 3 Hamiltonian：能量算子

**Hamiltonian**（能量算子）：

$$
\hat H = -\frac{\hbar^2}{2m}\Delta + V(x)
$$

**自伴性与谱**：

- 势阱（$V$ 有界下方）：$\hat H$ 自伴（在合适定义域上），谱 = 离散谱（束缚态）+ 连续谱（散射态）。
- **氢原子**：$V = -\frac{e^2}{4\pi\epsilon_0 r}$，谱 = $\{E_n = -\frac{13.6}{n^2}\text{ eV}\}$（离散）+ $\{E \ge 0\}$（连续）。<span class="marginnote">氢原子能级是量子力学的经典成果，也是谱理论的辉煌应用：<strong>Schrödinger 方程 $\\hat H\\psi = E\\psi$ 的 $L^2$ 解对应离散谱（束缚态），非 $L^2$ 解对应连续谱（散射态）</strong>。「为什么氢原子能量量子化？」的答案：Hamiltonian 的谱是离散的。这是谱理论对物理最直接的贡献。</span>

**例（谐振子）**：$V(x) = \frac12 m\omega^2 x^2$，能级 $E_n = (n + \frac12)\hbar\omega$——等间距能级。特征函数是**厄米函数**（厄米多项式 × Gauss 权重），构成 $L^2$ 的正交基。

## 4 公式解析：位置与动量的对易

把对易关系 $[\hat x, \hat p] = i\hbar I$ 的推导写清：

$$
(\hat x\hat p\psi)(x) = x\Big(-i\hbar\frac{d\psi}{dx}\Big), \qquad (\hat p\hat x\psi)(x) = -i\hbar\frac{d}{dx}\big(x\psi\big) = -i\hbar\Big(\psi + x\frac{d\psi}{dx}\Big)
$$

- **第一步，先 $\hat p$ 后 $\hat x$**：$\hat x\hat p\psi = -i\hbar x\psi'$。
- **第二步，先 $\hat x$ 后 $\hat p$**：$\hat p\hat x\psi = -i\hbar(x\psi)' = -i\hbar\psi - i\hbar x\psi'$（乘积法则）。
- **第三步，相减**：$(\hat x\hat p - \hat p\hat x)\psi = -i\hbar x\psi' + i\hbar\psi + i\hbar x\psi' = i\hbar\psi$。
- **第四步，结论**：$[\hat x, \hat p] = i\hbar I$。

**关键**：对易关系来自「乘法与求导不对易」——**$x\frac{d}{dx} \neq \frac{d}{dx}x$（差一个恒等项）**。这个「非交换性」是量子力学的本质，也是不确定性原理的算子来源（下节）。

## 5 例题精讲：量子算子的结构

**例题一：谐振子的升降算符**。

- 定义 $a = \sqrt{\frac{m\omega}{2\hbar}}\big(\hat x + \frac{i}{m\omega}\hat p\big)$、$a^\dagger$（伴随）。
- $[a, a^\dagger] = I$，$\hat H = \hbar\omega(a^\dagger a + \frac12)$。
- 能级 $E_n = (n+\frac12)\hbar\omega$——代数方法（升降算符）给出谱。

**例题二：Pauli 矩阵的自伴性与谱**。

- $\sigma_z$ 自伴（Hermite），特征值 $\pm 1$，特征向量 $|{\uparrow}\rangle, |{\downarrow}\rangle$。
- $\sigma_x, \sigma_y, \sigma_z$ 两两反对易且 $\sigma_i^2 = I$。
- 自旋测量：$P(\pm\hbar/2) = |\langle\psi|\text{本征态}\rangle|^2$。

**例题三：自由粒子的连续谱**。

- $\hat H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}$ 于 $L^2(\mathbb{R})$，谱 = $[0, \infty)$（连续）。
- 广义特征函数 $e^{ikx}$（$E = \hbar^2k^2/2m$）不在 $L^2$。
- 散射态：波包展开（Fourier 积分）——连续谱的物理实现。

**核心要点**：量子算子的三个练习——谐振子（代数谱）、自旋（有限维）、自由粒子（连续谱）——展示自伴算子的谱理论如何直接给出物理能谱。

**辨析｜易错点：** 无界算子的「乘积」要小心定义域：$\hat x\hat p$ 与 $\hat p\hat x$ 的公共定义域不一定相同。对易关系 $[\hat x,\hat p] = i\hbar I$ 只在适当的（稠密）定义域上严格成立——严格处理需要「无界算子的对易」理论（超出本节范围）。

## 6 小结

- **位置** $\hat x = M_x$：无界自伴，谱 $\mathbb{R}$（连续）。
- **动量** $\hat p = -i\hbar d/dx$：无界自伴，谱 $\mathbb{R}$；自伴性依赖定义域。
- **对易关系** $[\hat x, \hat p] = i\hbar I$：非交换性的量子签名。
- **角动量与自旋**：$[\hat L_x,\hat L_y] = i\hbar\hat L_z$；Pauli 矩阵 = 有限维自伴算子。
- **Hamiltonian**：谱 = 离散（束缚）+ 连续（散射）；氢原子、谐振子是典型。
- **定位**：量子算子的谱结构 = 量子系统的能谱，谱理论（第九章）的直接应用。

在下一节，我们研究**不确定性原理的算子表述**——用对易关系与 Cauchy-Schwarz 严格证明海森堡不确定性。
