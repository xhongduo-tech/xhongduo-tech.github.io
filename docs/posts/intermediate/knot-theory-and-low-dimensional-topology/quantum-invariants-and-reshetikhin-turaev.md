---
title: 量子不变量与 Reshetikhin-Turaev 构造
date: 2026-08-07
---

# 量子不变量与 Reshetikhin-Turaev 构造

<div class="epigraph">
<p>给结的每一条绳绑一个表示，缠结就变成了张量网络。</p>
<footer>—— 本文作者按</footer>
</div>

<div class="article-byline">
<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第20章 · Adams《The Knot Book》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从「量子群」开始

Jones 多项式来自 Temperley-Lieb 代数，而 Temperley-Lieb 代数只是「量子群」这个更大机器的冰山一角。**量子群（quantum group）**是李代数 $U(\mathfrak{g})$ 的「量子变形」$U_q(\mathfrak{g})$：让结构常数带参数 $q$，$q \to 1$ 时回到经典李代数。1991 年，Reshetikhin 与 Turaev 给出系统构造：**用量子群的表示（R-矩阵）把每条 tangle 映射成一个张量网络，再取迹，得到结不变量**。

这套 **Reshetikhin-Turaev 构造（RT 构造）**是量子拓扑的中枢：Jones、HOMFLY、Kauffman 全部是它的特例，三维流形的 Witten 不变量（第3篇之五）也建立在它之上。<span class="marginnote">量子群不是「群」——它是霍普夫代数（Hopf algebra），一种携带乘法、余乘法、对极的代数结构。「群」之名是德林费尔德（Drinfeld）1990 菲尔兹奖演讲留下的历史称谓。R-矩阵满足的 Yang-Baxter 方程是量子群的核心结构公理，也正是结图 R3 移动的代数化身。</span>

## 1 从 Temperley-Lieb 到量子群

回顾第3篇之二：Jones 多项式 = $TL_n$ 上 Markov 迹对辫子的求值。$TL_n$ 与 $U_q(\mathfrak{sl}_2)$ 有深刻联系：

**定理**：$TL_n$ 是 $U_q(\mathfrak{sl}_2)$ 的「量子双」中某个子代数的商；更直接地，Jones 多项式是 $U_q(\mathfrak{sl}_2)$ 的二维表示对应的不变量。

关键洞察：**Temperley-Lieb 代数对应「二维表示」**。若改用高维表示（三维、四维、……）或改用其他李代数（$\mathfrak{sl}_N$、正交/辛型），就得到一整族新不变量。这就是 RT 构造的「参数空间」：

| 李代数 | 表示 | 得到的不变量 |
| --- | --- | --- |
| $U_q(\mathfrak{sl}_2)$ | 二维 | Jones 多项式 |
| $U_q(\mathfrak{sl}_N)$ | 基本表示 | $A_N$ 型量子不变量（HOMFLY 的量子化） |
| 正交/辛型 | 基本表示 | Kauffman 多项式型不变量 |
| $U_q(\mathfrak{sl}_2)$ | $N$ 维 | **着色 Jones 多项式**（colored Jones） |

## 2 R-矩阵与 Yang-Baxter 方程

量子群的表示论给出一个关键对象：**R-矩阵（R-matrix）** $R \in \operatorname{End}(V \otimes V)$，它交换两个表示的张量积，且满足 **Yang-Baxter 方程**：

$$
R_{12}\, R_{13}\, R_{23} = R_{23}\, R_{13}\, R_{12}.
$$

（下标 $12, 13, 23$ 表示 $R$ 作用在张量积 $V \otimes V \otimes V$ 的对应因子上。）

- **几何含义**：$R$ 就是「一个交叉」——把 $V \otimes V$ 里的两条线交换，即「让两根绳子交叉」。
- **Yang-Baxter 方程 ⟺ R3 移动**：$R_{12}R_{13}R_{23} = R_{23}R_{13}R_{12}$ 正是「第三根绳滑越交叉」的代数版本。R 矩阵满足此方程，结图等价性自动保持。<span class="marginnote">Yang-Baxter 方程起源于统计力学（Yang 1967、Baxter 1972）：它保证可积模型的转移矩阵可对易。量子群提供 R-矩阵，于是「可积系统的代数」与「结的拓扑」共享同一个方程——这是「数学物理统一」最著名的实例之一。</span>

## 3 Reshetikhin-Turaev 构造

RT 构造把「结图 → 张量网络」写成一套配方：

1. **给绳子上色**：把链环的每条分量标上一个表示 $V_i$（选 $U_q(\mathfrak{g})$ 的一个表示）。每条绳子携带一个「颜色」。
2. **给交叉指定 R-矩阵**：一个交叉 = $R$ 或 $R^{-1}$（按上下与正负）；交叉处两条绳的颜色确定 $R$ 作用在哪个 $V \otimes V$ 上。
3. **给闭圈取迹**：每一条封闭的绳段（分量）对应「对颜色取迹」（把绳端接合）。整条链环对应一个**标量**：

$$
J_{L}^{(V_1, \ldots, V_\mu)} = \operatorname{tr}\big(\text{把 } R^{\pm 1} \text{ 沿 } L \text{ 相乘后的算子}\big).
$$

4. **归一化**：适当除以「平凡结的贡献」，得到不变量。

**定理（Reshetikhin-Turaev，1991）**：对任何李代数 $\mathfrak{g}$、任何表示选择，上述构造给出链环的拓扑不变量。

**易错点｜颜色与不变量**：同一链环、不同颜色（表示），得到**不同的不变量**。$N$ 维表示的着色 Jones 多项式 $J_N(K)$ 是一族无穷多不变量——比单条 Jones 多项式信息量大得多。把「着色」看作「层次」，RT 构造是在「表示论」的每一层各放一枚不变量。

## 4 公式解析：三叶结的着色 Jones

以 $U_q(\mathfrak{sl}_2)$、二维表示 $V$ 为例。R-矩阵取为

$$
R = q^{1/2} \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & q & 1 - q^2 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

（在 $V \otimes V$ 的基下）。三叶结是 $\sigma_1^3$ 的闭包，于是其着色 Jones 为

$$
J_{3_1}(V) = \operatorname{tr}\big(R^3\big) \qquad (\text{归一化后}).
$$

- **第一步，$R$ 是 4 × 4 矩阵**：$\dim(V \otimes V) = 4$。$R^3$ 是「三个交叉」的矩阵幂——三叶结三个交叉全同向，矩阵乘三次。
- **第二步，取迹**：闭包把绳两端接合，对应矩阵的迹（沿对角求和）。$\operatorname{tr}(R^3)$ 是 $q$ 的多项式。
- **第三步，归一化**：除以平凡结（空辫子）的贡献并代入 $t = q^2$ 或 $t = q^4$ 等，得到标准 Jones 多项式 $V_{3_1}(t) = -t^{-4} + t^{-3} + t^{-1}$。<span class="marginnote">矩阵迹与 Markov 迹在此合流：RT 构造的「取迹」就是上一节 Markov 迹的实现方式——$R$ 矩阵相乘 = 辫子相乘，矩阵迹 = 闭包。Temperley-Lieb 的图迹是「$R = A e - A^{-1}$ 特例」下的矩阵迹。两条路线其实同一条路。</span>

## 5 性质与应用

**着色 Jones 多项式 $J_N$**：一族无穷不变量，$N = 2$ 时即 Jones。$J_N$ 携带「更高表示」信息，能分辨更多结。
**体积猜想（Volume Conjecture）**：$J_N$ 在 $q$ 为单位根处的渐近行为猜想给出**双曲体积**（第3篇之六）：$\lim_{N} \frac{2\pi}{N} \log |J_N(e^{2\pi i/N})| = \operatorname{Vol}(K)$。这是「量子不变量 ⟷ 几何」最著名的未解桥梁。
- **三维流形不变量**：对链环做手术（第4篇之二）并用 RT 不变量「缝合」，得到三维流形的量子不变量——Witten 不变量（第3篇之五）的严格化。<span class="marginnote">RT 构造最重要的遗产之一是把「结不变量」升级为「三维流形不变量」：对三维流形取手术描述，把每个结的 RT 不变量组合起来，就得到流形的不变量。这直接通向 Witten 用 Chern-Simons 理论定义的拓扑量子场论——「量子群 → 手术 → 流形」是低维拓扑在 1990 年代最深的突破。</span>

**辨析｜量子群 vs 经典李代数**：$q$ 是「量子参数」，$q \to 1$ 时 R-矩阵退化为平凡的置换算子（交叉无信息），不变量退化（所有结都平凡）。「量子」的必要性正在于此：**非平凡的结不变量要求 $q \neq 1$**——拓扑信息藏在「变形」里。

### 辫群表示：RT 构造的代数载体

RT 构造的严格表述依赖「辫群在量子群表示上的作用」。关键对象：

**定理（RT 构造的辫群版本）**：量子群 $U_q(\mathfrak{g})$ 的每个有限维表示 $V$ 给出辫群表示 $\rho_V : B_n \to \operatorname{End}(V^{\otimes n})$，由

$$
\rho_V(\sigma_i) = \mathrm{Id}^{\otimes (i-1)} \otimes R_V \otimes \mathrm{Id}^{\otimes (n-i-1)}
$$

给出——第 $i$ 个交叉对应「R-矩阵作用在第 $i$ 与第 $i+1$ 个因子上」。

- **第一步，交叉 = R**：每个 $\sigma_i$（辫子的基本交叉）映射到「在 $V^{\otimes n}$ 的第 $i, i+1$ 因子上作用 $R_V$」。
- **第二步，辫关系自动满足**：因为 $R_V$ 满足 Yang-Baxter 方程，辫群关系 $\sigma_i\sigma_{i+1}\sigma_i = \sigma_{i+1}\sigma_i\sigma_{i+1}$ 在表示下自动成立——辫群表示良定义。
- **第三步，闭包 = 取迹**：闭辫 $\widehat{\beta}$ 对应 $\operatorname{tr}(\rho_V(\beta))$——矩阵迹把「辫子」变成「数」。

**这条表示论路线为何有力**：它把「结不变量」完全打包进「表示论数据」——只要给量子群和表示，不变量自动生成。RT 构造的普适性（覆盖 Jones、HOMFLY、Kauffman、着色系列）来自「量子群 + 表示」这一套参数化。

### 张量范畴视角

RT 构造还有更抽象的表达：**辫张量范畴（braided tensor category）**。把「向量空间」换成「范畴」，把「张量积」换成范畴的张量积，把「辫子」换成范畴的辫结构——RT 构造变成：

**定理（RT 构造的范畴版本）**：任何辫张量范畴上的 ribbon 结构都给出链环不变量。

- 量子群表示构成一个辫张量范畴（量子群 = 范畴的来源）；RT 不变量 = 「辫张量范畴上的 ribbon 不变量」的特例。
- 这个视角让「不变量」从「具体的矩阵计算」抽象为「范畴的普遍性质」——TQFT（第3篇之五）的数学地基正在于此。

## 6 小结

- **量子群** $U_q(\mathfrak{g})$ 是李代数的量子变形；R-矩阵满足 Yang-Baxter 方程（= R3 移动的代数版）。
- **Reshetikhin-Turaev 构造**：给绳子上色（选表示）、交叉 = R-矩阵、闭圈取迹，得到链环不变量。
- Jones 是 $U_q(\mathfrak{sl}_2)$ 二维表示的特例；换表示/李代数得到 HOMFLY、Kauffman、着色 Jones 一族。
- **着色 Jones 多项式** $J_N$