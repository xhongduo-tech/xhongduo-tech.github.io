---
title: 变分原理与里茨方法
date: 2026-08-07
---

# 变分原理与里茨方法

<div class="epigraph">
<p>把无穷维的变分问题限制到有限维子空间，里茨方法把「找函数」变成「解方程组」。</p>
<footer>—— 瓦尔特 · 里茨（Walther Ritz），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§10.4 ｜ 2026-08-07</p>
</div>

## 为什么需要「变分原理」与「里茨方法」

E-L 方程把变分问题变成微分方程，但很多方程无法精确求解。**变分原理（variational principle）**反过来想：不解 E-L 方程，而是**直接在函数空间里找极小化元**——它把「解方程」换成「取极小」。**里茨方法（Ritz method）**再把「函数空间」换成「有限维子空间」（如 $n$ 次多项式），于是「找函数」退化成「求 $n$ 个系数」——一个普通的最优化问题。这就是有限元方法（FEM）与谱方法的思想源头：**变分问题在有限维子空间上的离散化**。<span class="marginnote">里茨方法的核心步骤：<strong>在有限维子空间 $V_n$ 里最小化 $J$，得到 $J$ 在 $V_n$ 上的极小元 $u_n$</strong>。若 $V_n \\uparrow$ 稠密且 $J$ 是「好」的泛函，$u_n \\to u$（真解）。「离散逼近 + 收敛性」是里茨方法的全部内容，也是有限元的原型。</span>

## 1 变分原理

**定理（变分原理 / Dirichlet 原理）**：设 $A$ 是自伴正定算子（$\langle Ax, x\rangle \ge c\|x\|^2$），则方程

$$
Au = f
$$

的解 $u$ 等价于「能量泛函」的极小元：

$$
J(v) = \frac12\langle Av, v\rangle - \operatorname{Re}\langle f, v\rangle
$$

**即**：$u$ 满足 $Au = f$ ⟺ $u$ 使 $J$ 极小。<span class="marginnote">这是「变分原理」的抽象形态：<strong>「解方程」与「最小化能量」是一回事</strong>。对 Dirichlet 问题 $A = -\\Delta$，$J(v) = \\frac12\\int|\\nabla v|^2 - \\int fv$——「位能 + 载荷能」的极小。物理上，系统趋向「能量最小」的位形，而能量最小的位形正是方程的解。这就是「最小作用量原理」的线性版本。</span>

**证明（一阶条件）**：对任意 $\varphi$，$J(u + t\varphi)$ 在 $t = 0$ 取极小 ⟹ $\frac{d}{dt}\big|_0 J(u + t\varphi) = \operatorname{Re}\langle Au - f, \varphi\rangle = 0$ 对一切 $\varphi$ ⟹（变分基本引理）$Au = f$。

## 2 里茨方法

**方法（Ritz）**：设 $V_n \subset H$ 是有限维子空间（基 $\{\varphi_1, \ldots, \varphi_n\}$）。求 $u_n = \sum c_i \varphi_i$ 使 $J$ 在 $V_n$ 上极小：

- **第一步，代入**：$J(u_n) = \frac12\sum c_i \bar c_j \langle A\varphi_i, \varphi_j\rangle - \sum \bar c_i \langle f, \varphi_i\rangle$。
- **第二步，一阶条件**：$\frac{\partial J}{\partial \bar c_k} = 0$ 给出**线性方程组**

$$
\sum_j \langle A\varphi_i, \varphi_j\rangle\, c_j = \langle f, \varphi_i\rangle, \qquad i = 1, \ldots, n
$$

- **第三步，求解**：这是 $n \times n$ 线性方程组（刚度矩阵 $\langle A\varphi_i, \varphi_j\rangle$），解出 $c_i$ 得 $u_n$。<span class="marginnote">里茨方程组与「投影」完全同构：<strong>$u_n$ 是 $u$ 在 $V_n$ 上的「$A$-正交投影」（$A$-内积 $\langle Av, w\\rangle$ 下的投影）</strong>。刚度矩阵 $\langle A\\varphi_i, \\varphi_j\\rangle$ 就是 $A$-内积的 Gram 矩阵——「里茨 = $A$-投影」这条视线把变分法接回第四章的正交分解。</span>

**核心要点：里茨方法 = 在有限维子空间上最小化能量泛函**——解一个线性方程组得到近似解 $u_n$。

## 3 收敛性：里茨解逼近真解

**定理（里茨方法的收敛性）**：设 $A$ 自伴正定，$V_n \uparrow H$（$V_n$ 递增且在 $H$ 中稠密）。则里茨近似 $u_n$ 在「能量范数」$\|v\|_A = \sqrt{\langle Av, v\rangle}$ 下收敛到真解 $u$：

$$
\|u_n - u\|_A \to 0
$$

**证明**：$u_n$ 是 $u$ 在 $V_n$ 上的 $A$-投影，由最佳逼近（$A$-内积版），$\|u - u_n\|_A = \inf_{v \in V_n}\|u - v\|_A \to 0$（稠密性）。<span class="marginnote">收敛性证明只用了「最佳逼近」一条定理：<strong>$u_n$ 是 $u$ 在 $V_n$ 里的 $A$-最近点，$V_n$ 稠密 ⟹ $A$-距离趋于零</strong>。这是「变分法 + Hilbert 空间」的完美结合——里茨方法的收敛性不比「正交投影的收敛性」更复杂。</span>

**例**：Dirichlet 问题 $-u'' = f$（$u(0)=u(1)=0$），$V_n$ = 分段线性函数（有限元）或三角多项式（谱方法）。$u_n \to u$ 在 $H^1$ 能量范数下收敛。

## 4 公式解析：里茨方程组的推导

把「能量极小的条件变成线性方程组」拆开：

$$
J(u_n) = \frac12 \sum_{i,j} c_i \bar c_j K_{ij} - \sum_i \bar c_i b_i, \qquad K_{ij} = \langle A\varphi_i, \varphi_j\rangle,\ b_i = \langle f, \varphi_i\rangle
$$

- **第一步（能量展开）**：$J(u_n) = \frac12\langle A\sum c_i\varphi_i, \sum c_j\varphi_j\rangle - \operatorname{Re}\langle f, \sum c_i\varphi_i\rangle$——展开成 $c$ 的二次型。
- **第二步（求导）**：$\frac{\partial J}{\partial \bar c_k} = \frac12\sum_j K_{kj}c_j - b_k = 0$（对 $\bar c_k$ 求导，实部处理）。
- **第三步（方程组）**：$K c = b$——刚度矩阵 $K$、载荷向量 $b$。
- **第四步（求解）**：$K$ 正定（$A$ 正定 + 基线性无关），方程有唯一解 $u_n$。

**关键**：整个推导是「二次函数的极值」——**能量是系数的二次型，极值条件是一组线性方程**。里茨方法把变分问题化归为「解 $Kc = b$」，与最小二乘（$A^TAx = A^Tb$）完全平行。

## 5 例题精讲：里茨方法的计算

**例题一：$-u'' = 1$（$u(0)=u(1)=0$），一次里茨近似**。

- 基 $\varphi_1 = t(1-t)$。能量 $J(v) = \frac12\int_0^1 v'^2 - \int_0^1 v$。
- 一阶条件：$c_1\int \varphi_1'^2 = \int \varphi_1$，解得 $c_1 = 5$。
- $u_1 = 5t(1-t)$，真解 $u = \frac{t(1-t)}{2}$——一次近似已接近（误差约 10%）。

**例题二：三角基的里茨（谱方法）**。

- 基 $\sin(k\pi t)$。刚度矩阵 $K_{ij} = \pi^2 i^2\delta_{ij}$（对角！）。
- 解 $c_i = \frac{2\langle f, \sin i\pi t\rangle}{\pi^2 i^2}$——里茨方程对角化。
- 三角基让刚度矩阵对角——这是谱方法效率的来源。

**例题三：有限元（分段线性基）**。

- $V_n$ = 分段线性（节点 $t_i$）。刚度矩阵三对角。
- 求解 $Kc = b$——稀疏线性系统。
- 有限元 = 里茨方法 + 局部支撑基。收敛阶与网格大小相关。

**核心要点**：里茨方法的三个实例——多项式、三角、分段线性——展示「基的选择」决定刚度矩阵的形态（稠密/对角/稀疏）。

**辨析｜易错点：** 里茨方法要求 $A$ 自伴正定（变分原理的前提）。非对称算子（如对流项）不能直接最小化 $J$，需用 Galerkin 方法（弱形式，§10.5）——里茨是 Galerkin 的特例（对称正定）。

## 6 常见误区与辨析

**误区一：把里茨方法当「精确解法」**。

- 里茨给的是有限维子空间上的最优近似。
- 收敛性靠子空间稠密 + 投影最佳逼近。

**误区二：忘记变分原理要求自伴正定**。

- 非对称算子不能用能量极小化，需 Galerkin。
- 里茨是 Galerkin 的自伴正定特例。

**误区三：混淆刚度矩阵与一般矩阵**。

- $K_{ij} = \langle A\varphi_i, \varphi_j\rangle$ 是 $A$-内积 Gram 矩阵。
- 基的选择决定 $K$ 的稀疏性/对角性。

**核心要点：里茨 = $A$-投影 = 有限维能量极小化**——变分法的计算引擎。


## 7 小结

- **变分原理**：$Au = f$ ⟺ $u$ 极小化 $J(v) = \frac12\langle Av,v\rangle - \operatorname{Re}\langle f,v\rangle$（$A$ 自伴正定）。
- **里茨方法**：在有限维 $V_n$ 上极小化 $J$，得线性方程组 $Kc = b$（$K_{ij} = \langle A\varphi_i,\varphi_j\rangle$）。
- **收敛性**：$u_n$ 是 $u$ 的 $A$-投影，$V_n$ 稠密 ⟹ 能量范数收敛。
- **联系**：里茨 = $A$-投影 = Galerkin 的特例（对称正定情形）。
- **实例**：多项式、三角（对角）、分段线性（稀疏）基。
- **定位**：变分原理与里茨方法是有限元、谱方法、边值问题（下节）的共同基础。

在下一节，我们研究**微分方程边值问题的变分形式**——把边值问题改写成弱形式，用里茨/Galerkin 求解。
