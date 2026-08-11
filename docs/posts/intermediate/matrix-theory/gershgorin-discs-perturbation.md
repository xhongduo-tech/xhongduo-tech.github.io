---
title: 特征值定位与扰动理论：Gershgorin 圆盘定理、Weyl 不等式与 Hoffman-Wielandt 定理
date: 2026-08-11
---

# 特征值定位与扰动理论：Gershgorin 圆盘定理、Weyl 不等式与 Hoffman-Wielandt 定理

<div class="epigraph">
<p>特征值是我们无法直接看见的对象，但我们可以用圆盘圈住它、用不等式夹住它——扰动理论就是给看不见之物画圈的艺术。</p>
<footer>—— 化用自塞米恩·格什戈林（Semyon Aronovich Gershgorin）</footer>
</div>

<div class="article-byline">
<p>第二级 · 矩阵论 ｜ Horn & Johnson《Matrix Analysis》 ｜ 2026-08-11</p>
</div>

## 为什么从特征值定位开始

特征多项式 $\det(\lambda I - A) = 0$ 在理论上完全决定了特征值，但**显式求解**对
$n \ge 5$ 的矩阵几乎不可能（没有通用求根公式）。现实中我们更常遇到的问题是：不用精确求解，能否知道特征值落在哪里？
当我们修改矩阵的某些元素，特征值会漂移多远？这就是**特征值定位与扰动理论**——用几何圆盘和不等式把谱「圈」住。
它是数值方法收敛性、图上谱分析、以及大模型数值稳定性分析的数学底座<span class="marginnote">从极限到大模型的连接：训练中权重被噪声、量化、低秩近似扰动，
模型是否崩溃取决于扰动后特征值是否越过临界区域。Gershgorin 圆盘定理是「谱会不会跑出安全区」的快速体检工具。
</span>。

## 1 Gershgorin 圆盘定理：不看谱也能圈住谱

**Gershgorin 圆盘定理**：设 $A = [a_{ij}]$ 是 $n \times n$ 复矩阵，对每个
$i$ 定义**盖尔什戈林圆盘**

$$D_i = \left\{ z \in \mathbb{C} : |z - a_{ii}| \le \sum_{j \neq i} |a_{ij}| \right\}$$

即「圆心在 $a_{ii}$、半径等于第 $i$ 行非对角元绝对值之和」的闭圆盘。定理断言：

1. $A$ 的**每个特征值都至少落在某一个** $D_i$ 中（谱
$\subseteq \bigcup_i D_i$）；
2. 若 $k$ 个圆盘的并集与其余圆盘**不相交**，则该并集内恰有 $k$ 个特征值（计代数重数）。<span class="marginnote">第 2 条是「连通分支计数」：不相交的圆盘联盟各自独占一定数量的特征值。
它使圆盘定理从「谱在哪」升级为「每个分支有几个谱」，对判断奇异性与正定性非常有力。</span>

证明的核心是反证法：设 $Ax = \lambda x$、$\|x\|_\infty = 1$，取 $|x_k| = 1$
的最大分量下标 $k$，则 $\lambda x_k = \sum_j a_{kj}x_j$，移项得
$|\lambda - a_{kk}| \le \sum_{j\neq k}|a_{kj}||x_j| \le \sum_{j\neq k}|a_{kj}|$——特征值
$\lambda$ 被第 $k$ 行圆盘圈住。<span class="marginnote">这证明里选的范数是
$\infty$-范数，所以圆盘来自<strong>行</strong>；若改用 $1$-范数，则得到<strong>列</strong>版本的圆盘定理：圆心
$a_{jj}$、半径 $\sum_{i \neq j}|a_{ij}|$。行版本与列版本可以同时使用、取更紧的交集。
</span>

**推论**：若 $A$ 严格对角占优（$\sum_{j\neq i}|a_{ij}| < |a_{ii}|$ 对所有
$i$），则原点不在任何圆盘内，$A$ **可逆**。对角占优矩阵的可逆性由此一句话证得——这是圆盘定理最实用的礼物。

## 2 Weyl 不等式：Hermite 矩阵的谱扰动

对 Hermite 矩阵，谱是实数且可排序，扰动问题有一个极其精确的答案。设 $A, B$ 均为 $n \times n$
Hermite 矩阵，特征值按非增排列：

$$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n, \qquad \mu_1 \ge \mu_2 \ge \cdots \ge \mu_n$$

**Weyl 不等式**断言：对每个 $i = 1, \dots, n$，

$$\lambda_i(A) + \lambda_n(B) \le \lambda_i(A + B) \le \lambda_i(A) + \lambda_1(B)$$

特别地，取 $B$ 为扰动矩阵 $E$，则每个特征值的位移都被 $E$ 的最大、
最小特征值夹住：$\lambda_i(A+E) \in [\lambda_i(A) + \lambda_{\min}(E),\, \lambda_i(A) + \lambda_{\max}(E)]$。
<span class="marginnote">Weyl（外尔，1885—1955）在 1912 年给出这个不等式，
它是「谱的半连续依赖」的定量版本。直观：往 Hermite 矩阵上加一个正定扰动，每个特征值只能上移；加一个负定扰动，
每个特征值只能下移。</span>

**理解**：Hermite 矩阵的扰动是「有序的」——第 $i$ 大特征值的位置被 $B$ 的极值特征值整体平移，
不会交叉穿越。这是实谱排序带来的独特红利，对非 Hermite 矩阵没有如此干净的不等式。

## 3 Hoffman–Wielandt 定理：谱的配对

Weyl 只用到 $B$ 的极值特征值，Hoffman–Wielandt 定理则把**整个谱向量**的偏差与扰动矩阵的
Frobenius 范数直接挂钩。

**定理**：设 $A, B$ 均为 $n \times n$ **正规**矩阵，特征值分别为
$\lambda_1, \dots, \lambda_n$ 与 $\mu_1, \dots, \mu_n$，则存在一个排列
$\pi$ 使

$$\sum_{i=1}^{n} |\lambda_i - \mu_{\pi(i)}|^{2} \le \|A - B\|_F^{2}$$

对 Hermite 矩阵，排序后的配对即满足该式：**谱向量的 $\ell_2$ 位移不超过扰动矩阵的 Frobenius
范数**。<span class="marginnote">把特征值看作 $\mathbb{C}^n$ 中的点（计重数），
Hoffman–Wielandt 说的是：谱点集在最优配对下的位移，被扰动矩阵的 Frobenius 范数整体控制。它比
Weyl 精细——不只控制单个极值，而是控制整个谱的全局偏差。</span>

**辨析｜易错点：** Hoffman–Wielandt 要求 $A, B$ **都正规**。对非正规矩阵，
特征值对扰动的敏感度可以远超 Frobenius 范数的控制范围——病态矩阵的谱可以「剧烈漂移」。
这解释了为什么数值线性代数如此强调「正规化」：**谱的稳定性是正规矩阵的专利，非正规矩阵可能对扰动极其过敏**。

## 4 更紧的圈：Brauer 卵形与谱半径上界

Gershgorin 圆盘定理是好用的「第一圈」，但常常画得**过松**。为了得到更紧的谱定位，Brauer 在 1947
年提出用**卡西尼卵形（Cassini oval）**替换圆盘。

**Brauer 卵形定理**：对 $A$ 的每一对不同的下标 $i, j$，考虑卵形

$$\mathcal{C}_{ij} = \left\{ z : |z - a_{ii}|\,|z - a_{jj}| \le R_i R_j \right\}, \qquad R_i = \sum_{k \neq i} |a_{ik}|$$

则 $A$ 的每个特征值都落在**至少一个**卵形 $\mathcal{C}_{ij}$ 中。
卵形是「两个圆心乘积小于常数」的区域——当两圆相距很远时它分裂成两瓣，靠近时连成一片。<span class="marginnote">为什么更紧：圆盘断言谱在 $\bigcup_i D_i$ 中，Brauer 断言谱在
$\bigcup_{i<j}\mathcal{C}_{ij}$ 中，而每个卵形都包含在对应两圆的并集内——Brauer
区域是 Gershgorin 区域的子集，通常紧得多，代价是要检查 $O(n^2)$ 个卵形。</span>

**用圆盘定理直接给谱半径上界**：由谱包含于圆盘并集，可立即得到

$$\rho(A) \le \max_i \left( |a_{ii}| + \sum_{j\neq i}|a_{ij}| \right) = \|A\|_\infty$$

以及列版本 $\rho(A) \le \|A\|_1$。这些上界在理论上简单、在应用里够用，是「不求解就估谱」的快捷方式。
<span class="marginnote">更精致的上界：$\rho(A) \le \max_i |a_{ii}| + \sqrt{\max_i\sum_j |a_{ij}| \max_j\sum_i|a_{ij}|}$（用
Frobenius 与奇异值关系改进），以及 $\rho(A) \le \max_i\sum_j|a_{ij}|$
对不可约非负矩阵取等号的 Perron–Frobenius 语境——定位理论与非负矩阵理论在此交汇。</span>

**定位理论的整体图景**：Gershgorin 圆盘是「一阶」定位（只看单行），Brauer
卵形是「二阶」（看两行乘积），更高阶的推广（Fiedler 的 $k$-阶区域）继续收紧，但计算代价上升。
**「圈住谱」的本质是：用矩阵的局部信息换取全局结论，圈越紧、代价越高，
但结论永远保守可靠**——这是所有谱定位方法的共同气质。

**辨析｜易错点：** Brauer 卵形要求取**不同**下标 $i \neq j$，不能取 $i = j$（否则退化）。
另外卵形个数随 $n$ 二次增长，对大规模稀疏矩阵，工程上仍优先用圆盘与列-行双版本取交集，
而不是直接枚举全部卵形——**理论上的紧致与应用上的代价要平衡**。

## 5 公式解析：Gershgorin 圆盘 $|z - a_{kk}| \le \sum_{j \neq k} |a_{kj}|$

这条圆盘不等式是整个定位理论的起点，拆四步：

- **第一步，从特征方程出发**：$Ax = \lambda x$ 的第 $k$ 个分量方程是 $\sum_j a_{kj}x_j = \lambda x_k$，把 $k = j$ 项移到一边：$(\lambda - a_{kk})x_k = \sum_{j\neq k}a_{kj}x_j$。
- **第二步，放缩的杠杆**：对 $x$ 取 $\infty$-范数并规范化 $\|x\|_\infty = 1$，总有一个分量满足 $|x_k| = 1$；取它为基准，其余分量 $|x_j| \le 1$。
- **第三步，三角不等式封顶**：两边取模，$|\lambda - a_{kk}| = |\sum_{j\neq k}a_{kj}x_j| \le \sum_{j\neq k}|a_{kj}||x_j| \le \sum_{j\neq k}|a_{kj}|$。**特征值 $\lambda$ 距对角元 $a_{kk}$ 不超过该行非对角元绝对值和**。
- **第四步，遍历全部行**：对每个特征值，必存在某个「最大分量下标」$k$ 使它落入第 $k$ 个圆盘；故谱含于所有圆盘的并集。若改用 $\|\cdot\|_1$，同样的论证把「行」换成「列」，得到第二组圆盘，两者相交更紧。

## 6 小结

- **Gershgorin 定理**：谱含于以 $a_{ii}$ 为圆心、以行（或列）非对角元绝对值和为半径的圆盘并集；不相交分支各占对应个数的特征值。
- **对角占优 ⇒ 可逆**：严格对角占优矩阵可逆，是圆盘定理的直接推论。
- **Weyl 不等式**：Hermite 矩阵加扰动 $E$，第 $i$ 大特征值被 $E$ 的极值特征值夹逼，位移有序不交叉。
- **Hoffman–Wielandt**：正规矩阵谱的 $\ell_2$ 位移 ≤ 扰动矩阵的 Frobenius 范数；非正规矩阵的谱可能对扰动剧烈敏感。
- 易错点：行圆盘用 $\infty$-范数、列圆盘用 $1$-范数；Hoffman–Wielandt 的双正规条件是硬门槛。

在下一节，我们把谱的符号条件推到极致——当特征值全部为正，矩阵就获得了正定性，那是二次型、优化与概率论共享的金矿。
