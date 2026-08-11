---
title: Jordan 标准型与广义特征向量
date: 2026-08-11
---

# Jordan 标准型与广义特征向量

<div class="epigraph">
<p>不能对角化的矩阵并非不可理喻——它们只是拒绝被打散，坚持用上三角的链条把自己锁成一个整体。</p>
<footer>—— 化用自卡米尔·若尔当（Camille Jordan）</footer>
</div>

<div class="article-byline">
<p>第二级 · 矩阵论 ｜ Horn & Johnson《Matrix Analysis》 ｜ 2026-08-11</p>
</div>

## 为什么从 Jordan 标准型开始

第二级《线性代数》告诉我们：可对角化的矩阵最好办，特征向量铺成基即可。
但现实中有大量矩阵**不可对角化**——几何重数严格小于代数重数。它们不是病态的反例，而是真实出现在微分方程组、幂零算子、
约当块中的核心对象。Jordan 标准型给出了「最接近对角」的规整形态：每个特征值对应一串**Jordan 块**，
块内是「对角元 + 一条上对角线」的链。要理解它，必须先引入比特征向量更广义的概念——**广义特征向量**<span class="marginnote">为什么要广义特征向量：可对角化要求每个特征值 $g = m$（几何重数 =
代数重数）。当 $g < m$，特征向量只张出 $g$ 维空间，剩下的 $m - g$
维要靠"广义特征向量"来补——它们是 $A - \lambda I$ 反复作用后"被磨平"的向量。</span>。

## 1 广义特征向量与亏损

**广义特征向量（generalized eigenvector）**：设 $\lambda$ 是 $A$ 的特征值，
若存在正整数 $k$ 使

$$(A - \lambda I)^{k} x = 0, \qquad x \neq 0$$

则称 $x$ 为属于 $\lambda$ 的**广义特征向量**。$k=1$ 时就是普通特征向量。
全体使某次幂为零的向量构成**广义特征子空间**
$\mathcal{G}_\lambda = \ker(A - \lambda I)^{m}$（取足够大的 $m$，到
$m = $ 代数重数即稳定）。<span class="marginnote">直观：普通特征向量是「一次就被
$\lambda$ 吸收」的方向；广义特征向量是「多磨几次才被吸收」的方向。零矩阵的幂零部分
$(A-\lambda I)$ 是"磨盘"，广义特征向量正是磨盘下渐渐归零的向量链。</span>

**亏损矩阵（defective matrix）**：存在特征值满足几何重数 < 代数重数的矩阵。亏损矩阵不可对角化。
<span class="marginnote">例：$A = \begin{pmatrix}\lambda&1\\0&\lambda\end{pmatrix}$
只有一个特征向量（不计倍数），几何重数 1、代数重数 2，亏损。它是对角阵加了 $1$ 个"上斜元素"，
却彻底改变了对角化可能——这就是约当链的种子。</span>

广义特征向量张成整个空间：**每个 $n \times n$ 复矩阵 $A$ 都有
$\mathbb{C}^{n} = \bigoplus_{\lambda} \mathcal{G}_\lambda$**，
即广义特征子空间直和铺满全空间。这是 Jordan 标准型存在的几何基础。

## 2 Jordan 块与 Jordan 标准型

**Jordan 块（Jordan block）**：形如

$$J_k(\lambda) = \begin{pmatrix} \lambda & 1 & & \\ & \lambda & \ddots & \\ & & \ddots & 1 \\ & & & \lambda \end{pmatrix}$$

的 $k \times k$ 上三角矩阵：对角元全为 $\lambda$，紧邻上对角线全为 $1$，其余为 $0$。它满足
$(J_k(\lambda) - \lambda I)^{k} = 0$ 而
$(J_k(\lambda)-\lambda I)^{k-1} \neq 0$——链长恰为 $k$。

**Jordan 标准型（Jordan canonical form）**：任何复方阵 $A$ 都相似于一个由
Jordan 块构成的分块对角矩阵：

$$A = PJP^{-1}, \qquad J = \operatorname{diag}(J_{k_1}(\lambda_1), \dots, J_{k_s}(\lambda_s))$$

块的大小与个数由 $A$ 唯一决定（不计顺序）。**每个特征值 $\lambda$ 对应的诸块链长之和等于它的代数重数
$m$，块的个数等于它的几何重数 $g$**。<span class="marginnote">对应关系是「抽屉原理」式的：$m$ 个广义特征向量要排成 $g$
条链（每条链首是普通特征向量），链长总和 $m$、链数 $g$。$g = m$ 时每条链长 1，
退化为对角化——可对角化恰是 Jordan 标准型「所有块 $1\times1$」的特例。</span>

## 3 幂零部分：Jordan 分解的核心

把 Jordan 块拆成对角与严格上三角两部分：$J_k(\lambda) = \lambda I_k + N_k$，其中
$N_k$ 是幂零矩阵（$N_k^k = 0$）。于是 $A = PJP^{-1}$ 可写为

$$A = D + N, \qquad DN = ND$$

其中
$D = P\operatorname{diag}(\dots, \lambda_i I, \dots)P^{-1}$
是可对角化部分（半单），
$N = P\operatorname{diag}(\dots, N_{k_i}, \dots)P^{-1}$ 是幂零部分。
**这就是 Jordan 分解（Jordan–Chevalley 分解）：任何矩阵唯一地分解为可交换的"半单 +
幂零"两部分**。<span class="marginnote">为什么强调 $DN = ND$：可交换性使得二项式展开
$(D+N)^{k} = \sum_i \binom{k}{i}D^{k-i}N^{i}$ 合法，
这是下一组《矩阵函数》计算 $e^{A}$ 时用到的关键杠杆。幂零部分 $N$ 则刻画了"不可对角化的残留"。</span>

**幂零矩阵的指数**：$N$ 的**幂零指数**是使 $N^r = 0$ 的最小 $r$，等于最大 Jordan
块的尺寸。幂零部分让 $e^{A}$ 退化为有限和——这是后面计算矩阵指数时「Jordan 块上的指数自动截断」的来源。

**辨析｜易错点：** 最容易混淆的是「可对角化」与「相似」的界限。**可对角化 ⇔ 全部 Jordan 块为
$1 \times 1$ ⇔ $g = m$ 处处成立**。
两个矩阵有相同的特征多项式（甚至相同的最小多项式）仍可能不相似——Jordan 块的**尺寸分布**才是相似类的完整不变量。
例：$\begin{pmatrix}0&1&0\\0&0&0\\0&0&0\end{pmatrix}$ 与
$\begin{pmatrix}0&1&0\\0&0&1\\0&0&0\end{pmatrix}$ 特征多项式都是
$t^3$，但前者块为 $2\times1$、后者为 $3\times1$，**不相似**。

## 4 应用：用 Jordan 标准型解线性微分方程组

Jordan 标准型不只是分类工具，它直接给出线性常微分方程组 $\dot{x} = Ax$ 的**通解结构**。
这可能是它最重要的应用。

**可对角化情形的解**：$A = S\Lambda S^{-1}$ 时，令 $y = S^{-1}x$，方程组解耦成
$n$ 个独立标量方程 $\dot y_i = \lambda_i y_i$，通解为
$x(t) = \sum_i c_i e^{\lambda_i t} v_i$——每个初始分量沿各自特征方向以指数
$e^{\lambda_i t}$ 演化。<span class="marginnote">解的稳定性判据由此一目了然：$\operatorname{Re}(\lambda_i) < 0$
对所有 $i$ 时，$x(t) \to 0$；有 $\operatorname{Re}(\lambda_i) > 0$
时发散；实部为零且虚部非零时做等幅振荡（如简谐运动）。稳定性完全由谱的实部符号决定。</span>

**亏损情形的解：多项式尾巴**。若 $A$ 有 Jordan 块 $J_k(\lambda)$，其解不再是纯指数。考虑
$J = \lambda I + N$（$N$ 幂零、$N^k = 0$），对应子系统为 $\dot y = Jy$，解为

$$y(t) = e^{tJ}y(0) = e^{\lambda t} e^{tN}y(0) = e^{\lambda t}\left(I + tN + \frac{t^2}{2!}N^2 + \cdots + \frac{t^{k-1}}{(k-1)!}N^{k-1}\right)y(0)$$

**亏损系统的解是「指数 × 多项式」的混合**：$e^{\lambda t}$ 决定增长/衰减的骨架，而幂零部分的截断项
$t^{j}/j!$ 带来多项式因子。$k$ 越大，多项式尾巴越长。<span class="marginnote">直观：亏损矩阵的广义特征向量链让「同频模式」之间产生共振式耦合——它们不以完全相同的速率演化，
而是逐级"拖拽"，于是出现 $t e^{\lambda t}$、$t^2 e^{\lambda t}$ 这样的项。这是
Jordan 链在动力系统中"露面"的直接证据。</span>

**一个完整的 2×2
例子**：$A = \begin{pmatrix}\lambda & 1 \\ 0 & \lambda\end{pmatrix}$（一个
$2\times2$ Jordan 块）。通解为

$$\begin{pmatrix}x_1(t)\\x_2(t)\end{pmatrix} = e^{\lambda t}\begin{pmatrix}1 & t \\ 0 & 1\end{pmatrix}\begin{pmatrix}c_1\\c_2\end{pmatrix} = \begin{pmatrix}e^{\lambda t}(c_1 + c_2 t)\\ c_2 e^{\lambda t}\end{pmatrix}$$

注意 $x_1$ 含有 $t e^{\lambda t}$ 项：即使 $\lambda < 0$，
系统也会先被多项式尾巴推着走一段，再被指数拉回——**"先放大后衰减"的瞬态行为正是亏损结构的签名**。

**对稳定性分析的意义**：亏损矩阵若 $\operatorname{Re}\lambda = 0$（纯虚特征值），
解中会出现 $t$、$t^2$ 等多项式因子，
系统不再等幅振荡而会**线性增长**——这是「临界情形」下亏损性决定稳定与否的著名现象。<span class="marginnote">工程含义：控制系统、结构力学里若系统矩阵有零实部特征值且亏损，
微小扰动会导致随时间线性增长的响应，必须避免——"亏损"与"临界稳定"的组合是设计上的红线。</span>

**辨析｜易错点：** 亏损矩阵的「幂零尾巴」只在**特征向量不足**时出现——若每个特征值都 $g = m$，则无尾巴、
纯指数。另一个易错点：解里多项式尾巴的**次数**等于对应 Jordan 块**尺寸减一**（$J_k$ 给出到
$t^{k-1}$），不是代数重数本身；有多个块时各块独立贡献尾巴。

## 5 公式解析：Jordan 块上的幂 $J_k(\lambda)^m$

计算 $J_k(\lambda)$ 的幂是理解幂零效应的关键，拆四步：

- **第一步，二项展开**：$J = \lambda I + N$，$IN = NI$，故 $J^{m} = \sum_{i=0}^{m} \binom{m}{i}\lambda^{m-i}N^{i}$。可交换性使二项式定理直接可用。
- **第二步，$N$ 的作用是"平移"**：$N$ 是移位矩阵，$N^{i}$ 把矩阵的上对角带整体上移 $i$ 格：$(N^{i})_{p,q} = 1$ 当且仅当 $q - p = i$。
- **第三步，截断**：$N^{i} = 0$ 当 $i \ge k$（块尺寸 $k$），所以求和只需 $i = 0, \dots, \min(m, k-1)$。**$J^{m}$ 的第 $i$ 条上对角带为 $\binom{m}{i}\lambda^{m-i}$**。
- **第四步，为什么这就是广义特征向量**：$(J - \lambda I)^i = N^{i}$ 逐次把「磨盘」推进一格，链上的第 $i+1$ 个广义特征向量是 $N^{i}$ 作用后仍非零、$N^{i+1}$ 作用后归零的向量。**Jordan 链的长度 = 幂零指数**，一条链就是一个不可约的"磨盘"单元。

## 6 小结

- **广义特征向量**：$(A-\lambda I)^k x = 0$；广义特征子空间直和铺满 $\mathbb{C}^{n}$。
- **亏损矩阵**：某特征值 $g < m$，不可对角化；几何重数 = Jordan 块个数，代数重数 = 链长之和。
- **Jordan 标准型**：$A = PJP^{-1}$，$J$ 由 Jordan 块构成，块尺寸分布是相似类的完整不变量。
- **Jordan 分解**：$A = D + N$，半单 $D$ 与幂零 $N$ 可交换；$N$ 决定不可对角化的残留。
- 易错点：特征多项式相同 ≠ 相似；可对角化 ⇔ 全部块 $1\times1$。

在下一节，我们将用一个漂亮的定理把「矩阵与它的特征多项式」绑在一起——Cayley-Hamilton 定理，
以及随之而来的最小多项式。
