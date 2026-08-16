---
title: 希尔伯特空间（正交投影、Riesz 表示定理）
date: 2026-08-07
---

# 希尔伯特空间（正交投影、Riesz 表示定理）

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》第4章 ｜ 2026-08-07</p>
</div>

## 为什么希尔伯特空间「太好了」

Banach 空间可以装下函数与数列，但没有「垂直」和「夹角」。**内积**把这两个概念带回来，于是有了希尔伯特空间——**带内积的完备空间**。它是泛函分析里「最舒服」的空间：几何直觉（垂直、投影、分解）与无穷维并存。<span class="marginnote">希尔伯特（1862—1943）在 1906 年前后研究积分方程时引入 $L^2$ 的几何语言；冯 · 诺依曼 1929 年把「希尔伯特空间」确立为公理化概念——<strong>量子力学（第10篇）从此有了数学舞台</strong>。当大模型说「向量空间」「点积」「相似度」时，它们说的就是这里的内积空间。</span>

这一篇是希尔伯特空间的总览：内积、正交、投影、正交基、Riesz 表示定理——一条直线串起全部。

## 1 内积：垂直与夹角从哪来

**核心概念（内积）**：线性空间 $H$ 上的映射 $\langle\cdot,\cdot\rangle: H \times H \to \mathbb{C}$ 称为**内积**，若

1. 对第一个变量线性：$\langle \alpha x + \beta y, z \rangle = \alpha\langle x,z\rangle + \beta\langle y,z\rangle$；
2. **共轭对称**：$\langle x, y \rangle = \overline{\langle y, x\rangle}$；
3. 正定：$\langle x, x\rangle \ge 0$，且 $\langle x,x\rangle = 0 \iff x = 0$。

**重点：内积 ⟹ 范数。** $\|x\| = \sqrt{\langle x, x\rangle}$ 自动满足三条范数公理。反过来不一定——范数不一定来自内积（见 §2）。

**经典例子**：

- $\mathbb{C}^n$：$\langle x, y\rangle = \sum_{i=1}^n x_i \overline{y_i}$。
- $l^2$：$\langle x, y\rangle = \sum_{n\ge1} x_n \overline{y_n}$（收敛性由 Hölder 保证）。
- $L^2(a,b)$：$\langle f, g\rangle = \int_a^b f(t)\overline{g(t)}\,dt$。
- $C[a,b]$ 配上 $\langle f,g\rangle = \int f\overline g$ **不完备**（缺连续极限）——所以不构成希尔伯特空间，必须换 $L^2$。

## 2 柯西-施瓦茨不等式与平行四边形公式

**定理（Cauchy-Schwarz）**：内积空间中对一切 $x, y$：

$$
|\langle x, y\rangle| \le \|x\|\,\|y\|, \qquad \text{且等号} \iff x, y \text{ 线性相关}
$$

**直觉**：两个向量内积的「模」至多是两者长度之积——夹角余弦落在 $[-1,1]$ 的无穷维翻版。Cauchy-Schwarz 是最重要的不等式之一，$L^2$ 里它正是 Hölder 不等式在 $p=q=2$ 的情形。<span class="marginnote">柯西 1821 年给出有限维情形，施瓦茨 1888 年处理积分情形。它在整个分析里反复出现——<strong>几乎所有「一个量 ≤ 两个量乘积」的估计都藏着它</strong>。</span>

**核心概念（平行四边形公式）**：

$$
\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2
$$

**反问题（Jordan-von Neumann 定理）**：一个范数来自内积，**当且仅当**它满足平行四边形公式。据此可判断哪些 Banach 空间不是希尔伯特空间：

- $l^p$（$p \ne 2$）、$L^p$（$p \ne 2$）、$C[a,b]$：不满足平行四边形公式（取 $f = 1_{[0,1/2]}$、$g = 1_{[1/2,1]}$ 型函数验证），**不来自内积**。
- $l^2, L^2, \mathbb{C}^n$：满足，来自内积。

## 3 正交与正交分解

**核心概念（正交）**：$x \perp y$，若 $\langle x, y\rangle = 0$。$M^\perp = \{x : x \perp m, \forall m \in M\}$ 称为 $M$ 的**正交补**。

**定理（正交分解定理）**：设 $M$ 是希尔伯特空间 $H$ 的**闭**子空间，则

$$
H = M \oplus M^\perp
$$

即每个 $x \in H$ 可**唯一**写成 $x = m + n$，$m \in M$、$n \in M^\perp$。<span class="marginnote"><strong>正交分解是「分解」思想的极致形态</strong>：信号里的「信号 + 噪声」、逼近论里的「最佳逼近 + 残差」、量子力学里的「态空间直和」，全是它的化身。闭性是唯一条件——无穷维里「$M$ 闭」不可省（有理系数多项式在 $L^2$ 中不闭）。</span>

**公式解析：分解为什么存在且唯一。**

- **第一步，唯一性**：若 $x = m + n = m' + n'$，则 $m - m' = n' - n \in M \cap M^\perp = \{0\}$，故 $m = m'$、$n = n'$。
- **第二步，存在性（几何法）**：取 $d = \inf_{m \in M}\|x - m\|$，取极小化列 $\{m_n\}$，用平行四边形公式证明 $m_n$ 是 Cauchy 列：

$$
\|m_n - m_k\|^2 = 2\|x - m_n\|^2 + 2\|x - m_k\|^2 - 4\|x - \tfrac{m_n + m_k}{2}\|^2 \le 2(\cdots) - 4d^2 \to 0
$$

- **第三步**：$M$ 完备（$M$ 闭 + $H$ 完备）⟹ $m_n \to m$，令 $n = x - m$，验证 $n \perp M$（对任意 $m_0 \in M$，$\|x - m - tm_0\|^2 \ge d^2$ 让 $t$ 取极小即得 $\langle n, m_0\rangle = 0$）。

## 4 投影算子与最佳逼近

**核心概念（投影算子）**：正交分解的「取 $M$ 分量」映射 $P_M: x \mapsto m$ 是**线性算子**，满足 $P_M^2 = P_M$（幂等）、$P_M^* = P_M$（自伴，见第七篇）、$\|P_M\| = 1$（$M \ne \{0\}$）。

**核心概念（最佳逼近）**：对 $x \notin M$，$P_M x$ 正是 $M$ 中**离 $x$ 最近**的点：

$$
\|x - P_M x\| = \operatorname{dist}(x, M) = \min_{m \in M}\|x - m\|
$$

**且 $P_M x$ 唯一**。这是逼近论（第10篇《最佳逼近元的存在性与唯一性》）在希尔伯特空间的精确形态——也是最小二乘的几何答案。

**例子（最小二乘）**：解超定线性方程组 $Ax = b$（$A$ 为 $m \times n$ 矩阵，$m > n$）。令 $M = \operatorname{ran} A$，则最小二乘解 $\hat x$ 满足 $A\hat x = P_M b$，即 $A^T A \hat x = A^T b$——**正规方程**。投影把「无解方程」变成「可解方程」：$\operatorname{ran}A$ 上的投影正交化残差。

## 5 规范正交基与 Riesz 表示定理

**核心概念（规范正交基）**：$H$ 中的集 $\{e_\alpha\}$，若 $\langle e_\alpha, e_\beta\rangle = \delta_{\alpha\beta}$（规范正交）且其有限张成在 $H$ 中稠密（完备），则称其为**规范正交基（orthonormal basis）**。对 $x \in H$ 有

$$
x = \sum_\alpha \langle x, e_\alpha\rangle e_\alpha, \qquad \|x\|^2 = \sum_\alpha |\langle x, e_\alpha\rangle|^2 \ \text{（Parseval 等式）}
$$

$l^2$ 的标准基 $\{e_n\}$、$L^2[0, 2\pi]$ 的傅里叶基 $\{\frac{1}{\sqrt{2\pi}}e^{inx}\}$ 都是例子。**傅里叶级数（第一级《数学分析》三）正是「在规范正交基上展开」——正交基观点把傅里叶理论变成线性代数。**

**定理（Riesz 表示定理）**：设 $H$ 是希尔伯特空间，$f \in H^*$，则存在**唯一** $y_f \in H$ 使

$$
f(x) = \langle x, y_f\rangle \quad (\forall x \in H), \qquad \|f\| = \|y_f\|
$$

**并且映射 $f \mapsto y_f$ 是共轭线性等距同构**——$H$ 与 $H^*$「几乎同一」。<span class="marginnote"><strong>Riesz 表示定理 = 希尔伯特空间的「免检通行证」</strong>：每个连续线性泛函都「其实」是某个内积。于是 $H$ 与 $H^*$ 可视为同一空间——这解释了对偶算子在希尔伯特空间里为何「直接回到自身」（第七篇）。</span>

**公式解析：Riesz 表示定理为什么成立。** 三步：

- **第一步**：若 $f = 0$，取 $y_f = 0$。否则 $\ker f$ 是**闭**真子空间，由正交分解取 $z \perp \ker f$，$\|z\| = 1$（归一化）。
- **第二步**：对任意 $x$，$x - \frac{f(x)}{f(z)}z \in \ker f$（直接代 $f$ 验证），故与 $z$ 正交，$\langle x, z\rangle = \frac{f(x)}{f(z)}\langle z,z\rangle = \frac{f(x)}{f(z)}$，即 $f(x) = f(z)\langle x, z\rangle = \langle x, \overline{f(z)}z\rangle$。
- **第三步**：$y_f = \overline{f(z)}z$，唯一性由「若 $\langle x, y\rangle = 0$ 对所有 $x$ 则 $y = 0$」保证。

## 6 小结

- **内积空间**：内积 ⟹ 范数；Cauchy-Schwarz 与平行四边形公式把关。
- **希尔伯特空间** = 完备内积空间；$l^2, L^2$ 是主角，$C[a,b]$ 不是。
- **正交分解**：$H = M \oplus M^\perp$，$M$ 闭是唯一前提。
- **投影算子**：$P_M$ 幂等自伴、范数 1；最佳逼近 = 投影，唯一且显式。
- **规范正交基**：$x = \sum \langle x,e_\alpha\rangle e_\alpha$；傅里叶级数即正交基展开。
- **Riesz 表示**：$H^* \cong H$——一切连续线性泛函都是内积。

在下一节，我们研究希尔伯特空间上的「变换」——**希尔伯特空间上的算子（伴随、自伴算子）**。
