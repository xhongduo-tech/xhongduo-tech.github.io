---
title: 非交换环面与其他例子（irrational rotation algebra、量子群 SUq(2)）
date: 2026-08-17
---

# 非交换环面与其他例子

<div class="epigraph">
<p>一幅由物理与数学织成的挂毯。</p>
<footer>—— 沃恩 · 琼斯（Vaughan Jones）评 Connes《Noncommutative Geometry》</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ GBVF《Elements of Noncommutative Geometry》Ch.12; Connes《Noncommutative Geometry》Ch.IV; Khalkhali《Basic Noncommutative Geometry》Ch.2 ｜ 2026-08-17</p>
</div>

## 为什么从例子开始

七篇文章的理论装备已经齐了：Gelfand 对偶、Morita 等价、K-理论、循环上同调、谱三元组、微分结构、指标公式。但非交换几何不是抽象教条——它需要**真正的例子**来检验机器、喂养直觉。Vaughan Jones 把 Connes 的书称作「物理与数学织成的挂毯」，而挂毯的花纹正是由一个个具体的非交换空间织出来的。

本节介绍两个最核心的样板：**非交换环面**（irrational rotation algebra，$A_\theta$）与**量子群** $SU_q(2)$。前者是「最简单的非平凡非交换流形」，后者把 Lie 群 $SU(2)$ 量子化。再加上一组其他例子（Penrose 拼图、群 C\*-代数、Moyal 平面、Cuntz 代数），我们就能看到：非交换几何的对象不是孤例，而是一个完整的世界。

## 1 非交换环面 $A_\theta$

### 1.1 定义

**非交换环面（noncommutative torus）** $A_\theta$（$\theta$ 无理数）是含幺 C\*-代数，由两个幺正元 $U, V$ 生成，满足

$$
VU = e^{2\pi i \theta}\, UV
$$

当 $\theta = 0$ 时 $UV = VU$，$A_0 = C(\mathbb{T}^2)$ 就是经典环面上的连续函数代数——$U = e^{2\pi i x}$、$V = e^{2\pi i y}$。<span class="marginnote">「无理旋转代数」的名字来自它的一个具体实现：$U$ 是平移，$V$ 是按无理角 $2\pi\theta$ 旋转——它们合起来生成 $\mathbb{T}^2$ 上一个遍历群作用的交叉积。Rieffel 1970 年代末证明 $A_\theta$ 与 $A_{\theta'}$ 强 Morita 等价当且仅当 $\theta, \theta'$ 在 $SL(2,\mathbb{Z})$ 作用下同轨。</span>

### 1.2 为什么它是「流形」

$A_\theta$ 有光滑子代数 $A_\theta^\infty$（由在 $\mathbb{Z}^2$ 上快速衰减的系数 $\sum_{m,n} a_{mn} U^m V^n$ 构成），上面有两个基本导子：

$$
\delta_1(U) = U,\ \ \delta_1(V) = 0; \qquad \delta_2(U) = 0,\ \ \delta_2(V) = V
$$

$A_\theta^\infty$ 配这两个导子，构成一个**光滑非交换流形**：它有迹 $\tau(\sum a_{mn}U^mV^n) = a_{00}$，有谱三元组（Dirac 算子 $D = \delta_1 \otimes \sigma_1 + \delta_2 \otimes \sigma_2$），有联络与曲率。它是「非交换几何工具箱」每次上线时第一个被测试的对象。

### 1.3 K-理论

**定理（Pimsner–Voiculescu, 1980）**：$A_\theta$ 的 K-理论为

$$
K_0(A_\theta) \cong \mathbb{Z}^2, \qquad K_1(A_\theta) \cong \mathbb{Z}^2
$$

**核心对比表**：经典环面 vs 非交换环面

| 性质 | $C(\mathbb{T}^2)$ | $A_\theta$ |
| --- | --- | --- |
| K₀ | $\mathbb{Z}^2$ | $\mathbb{Z}^2$（Pimsner–Voiculescu） |
| K₁ | $\mathbb{Z}^2$ | $\mathbb{Z}^2$ |
| 点 | 存在（连续函数） | 无点 |
| 导子 | 两个 | 两个（$\delta_1, \delta_2$） |
| 迹 | 积分 | $\tau$（不变迹） |
| 谱三元组 | 有 | 有 |

令人惊讶的是：**非交换环面与经典环面的 K-理论相同**——尽管没有点，它的「拓扑骨架」与经典环面一模一样。这印证了 K-理论是「非交换拓扑」的说法。

## 2 量子群 $SU_q(2)$

### 2.1 Woronowicz 的紧矩阵量子群

1987 年，Woronowicz 引入了**紧矩阵量子群（compact matrix quantum group）**：一个含幺 C\*-代数 $A$，带矩阵元 $u_{ij}$ 与余积 $\Delta$、余单位 $\epsilon$、反极 $S$（Hopf 代数结构），且矩阵 $u = (u_{ij})$ 酉。$SU(2)$ 的量子化 $SU_q(2)$ 是最重要的例子。

### 2.2 定义与关系

$SU_q(2)$（$q \in (0,1)$）由四个生成元 $a, b, c, d$ 生成的含幺 $*$-代数，满足矩阵 $u = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ 幺正，且满足量子化交换关系（记 $q$ 为实参数）：

$$
ab = q\, ba, \quad ac = q\, ca, \quad bd = q\, db, \quad cd = q\, dc, \quad bc = cb
$$

$$
ad - da = (q - q^{-1})\, bc, \qquad ad - q\, bc = 1 \quad \text{（量子行列式）}
$$

当 $q \to 1$ 时这些关系退化为 $SU(2)$ 的坐标函数交换关系，$SU_q(2)$ 还原为 $SU(2)$。<span class="marginnote">$SU_q(2)$ 的表示论与经典 $SU(2)$ 平行：不可约表示仍由「量子自旋」$j = 0, \tfrac12, 1, \dots$ 标记，但维度公式量子化为 $[2j+1]_q = (q^{2j+1}-q^{-2j-1})/(q-q^{-1})$。Woronowicz 1987 年构造了它的 Haar 态与表示论，是非交换几何与量子群交叉的起点。</span>

### 2.3 Podleś 球面

由 $SU_q(2)$ 的量子齐性空间可构造 **Podleś 球面（Podleś sphere）**——$q$-形变的 $S^2$。它同样有谱三元组、Dirac 算子与 K-理论，是非交换球面的标准例子。

### 2.4 公式解析：交换关系背后的物理

以 $A_\theta$ 的核心关系为例拆解「为什么要让它不交换」：

$$
VU = e^{2\pi i\theta}\, UV
$$

- **第一步**，把 $U, V$ 想成「两个坐标方向的指数函数」$e^{2\pi i x}$、$e^{2\pi i y}$。经典情形它们交换，对应经典环面 $\mathbb{T}^2$ 有独立坐标。
- **第二步**，加入相位 $e^{2\pi i\theta}$：$U, V$ 不再交换，但相差一个**中心相位**。这正是量子力学里位置与动量、或 Heisenberg 代数的类似物：$PQ - QP = i\hbar$。$\theta$ 扮演 $\hbar$ 的角色——**形变参数**。
- **第三步**，量子力学里 $\hbar \to 0$ 回到经典；这里 $\theta \to 0$ 回到经典环面。因此 $A_\theta$ 是一个「带 $B$-场/非交换坐标的环面」——在弦理论中，$A_\theta$ 正是 D-膜在有常量 $B$-场背景下的开弦端点坐标代数。

**一句话**：非交换环面把「环面 + 常数 $B$-场」编码成一个算子代数，而 $\theta$ 是普朗克常数式的形变参数。

## 3 更多例子：非交换空间的世界

### 3.1 Penrose 拼图空间

Penrose 的非周期拼图（准晶体）整体上没有平移对称性。Connes 证明其「拼图空间」是一个非交换空间：其 C\*-代数是 **Connes–Moser 群胚 C\*-代数**，K-理论给出 $\mathbb{Z}^4$ 等不变量，并与准晶体的衍射谱对应。<span class="marginnote">这是「非交换商」的绝佳例子：Penrose 拼图模去平移作用的商空间太坏（不可 Hausdorff），经典拓扑无法处理；非交换几何用交叉积代数绕过了它。见 Connes Ch.II §3《The space X of Penrose tilings》。</span>

### 3.2 群 C\*-代数与约化

对离散群 $\Gamma$，其约化 C\*-代数 $C^*_r(\Gamma)$ 把「$\Gamma$ 的酉表示空间」编码成一个非交换空间。$C^*_r(\mathbb{Z}^2) = C(\mathbb{T}^2)$ 即回到环面；对非交换群则得到真正的非交换空间。Novikov 猜想正是关于 $C^*_r(\Gamma)$ 的 K-理论的陈述（见 Connes Ch.II/III）。

### 3.3 Moyal 平面与 Cuntz 代数

- **Moyal 平面**：$\mathbb{R}^2$ 上的形变量子化，交换子 $[x^\mu, x^\nu] = i\theta^{\mu\nu}$；是量子力学相空间（Weyl 量子化）的代数实现。
- **Cuntz 代数** $\mathcal{O}_n$：由 $n$ 个等距生成、纯非交换的简单 C\*-代数，$K_0 = \mathbb{Z}/(n-1)\mathbb{Z}$，$K_1 = 0$——展示非交换空间可以有多「奇异」。
- **叶状结构/轨道空间**：叶子空间与群作用轨道空间是最早驱动非交换几何的几何例子（Connes Ch.I/II）。

## 4 小结

- **非交换环面 $A_\theta$**：$VU = e^{2\pi i\theta}UV$；最简单的非交换流形，有导子、迹、谱三元组与完整 K-理论（Pimsner–Voiculescu：$K_0 = K_1 = \mathbb{Z}^2$）。
- **量子群 $SU_q(2)$**（Woronowicz 1987）：紧矩阵量子群的样板；$q\to 1$ 时还原经典 $SU(2)$；量子行列式 $ad - qbc = 1$ 是核心关系。
- **Podleś 球面**、**Penrose 拼图空间**、**群 C\*-代数**、**Moyal 平面**、**Cuntz 代数**——非交换空间是一个丰富而多样的世界。
- 共同的模式：**形变参数**（$\theta$ 或 $q$）把经典对象量子化，经典极限还原，而 K-理论/循环上同调等不变量在形变下保持或可控。

在下一节，我们将看到这套例子与指标公式在**物理学**中的集大成——**标准模型的谱作用与非交换规范理论**，其中包括著名的「谱作用原理」如何从纯几何推导出粒子物理的拉格朗日量。

<span class="marginnote">本文参考：GBVF《Elements of Noncommutative Geometry》Ch.12《Tori》; Connes《Noncommutative Geometry》Ch.IV; Khalkhali Ch.2。Pimsner–Voiculescu 的 K-理论计算见其 1980 年论文《Exact sequences for K-groups and Ext-groups of certain cross-product C\*-algebras》。</span>