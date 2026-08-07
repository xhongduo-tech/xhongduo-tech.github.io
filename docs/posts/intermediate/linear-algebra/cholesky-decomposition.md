---
title: Cholesky 分解与正定矩阵的判定
date: 2026-08-08
---

# Cholesky 分解与正定矩阵的判定

<div class="epigraph">
<p>正定矩阵像一盒拼图：它总能被拆成一个「三角形因子 × 自己的转置」，$A = LL^T$——对称之美，分解成乘积之简。</p>
<footer>—— 肖莱斯基（André-Louis Cholesky，法国军官与数学家）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§4.2 ｜ 2026-08-08</p>
</div>

## 为什么从 Cholesky 分解开始

正定矩阵（第五篇）处处为正、特征值全正，还隐藏着另一个宝藏：**它能被分解成 $A = LL^T$**，其中 $L$ 是下三角矩阵。这就是 **Cholesky 分解**——正定矩阵专属的「对称 LU」，比普通 LU 快一倍、更省内存、数值更稳。<span class="marginnote">Cholesky 分解是正定矩阵的「谱证据」：<strong>一个对称矩阵正定 ⟺ 它存在 Cholesky 分解</strong>。这个判据在统计里天天用——协方差矩阵、Gram 矩阵、正规方程 $A^TA$ 都要靠 Cholesky 分解来求逆与采样（多元正态抽样）。</span>

本节给出 Cholesky 分解的定义、算法、与正定判定的关系。

## 1 Cholesky 分解的定义

**定理（Cholesky 分解）**：设 $A$ 是 $n$ 阶实对称正定矩阵，则存在**唯一**的下三角矩阵 $L$（对角元为正）使得

$$
A = LL^T
$$

**核心概念**：$L$ 称为 **Cholesky 因子（Cholesky factor）**。分解 $A = LL^T$ 是「LU 的对称版本」：因为 $A$ 对称，LU 分解中的 $U$ 恰好等于 $D L^T$（$D$ 对角），合并后就是 $LL^T$。

**重点**：$L$ 的对角元全正（由 $A$ 正定保证）。Cholesky 分解**唯一**——比 LU 更「专一」。这个唯一性使它成为验证正定性的可靠判据。

**辨析｜易错点：** Cholesky 分解要求 $A$ **对称正定**——只是对称不够，只是正定也必须先取对称部分。若 $A$ 对称但非正定，Cholesky 计算中途会出现「对负数开方」而失败——这个「失败」本身就是「非正定」的诊断。

## 2 算法：逐列求 $L$

对 $A = LL^T$ 展开，第 $(i, j)$ 元（$i \ge j$）：

$$
a_{ij} = \sum_{k=1}^{j} l_{ik} l_{jk}
$$

逐列解出 $l_{ij}$（主对角线下方）：

- **对角元**：$a_{jj} = \sum_{k=1}^j l_{jk}^2 \Rightarrow l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$；
- **非对角元**（$i > j$）：$a_{ij} = \sum_{k=1}^{j} l_{ik}l_{jk} \Rightarrow l_{ij} = \frac{a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk}}{l_{jj}}$。

**重点**：开方只出现在对角元，且被开方数必须为正——这正是「$A$ 正定」的体现。若被开方数出现负值或零，立即知道 $A$ 非正定。

**一个完整例子**：$A = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$。

- $l_{11} = \sqrt{a_{11}} = 2$；
- $l_{21} = a_{21}/l_{11} = 2/2 = 1$；
- $l_{22} = \sqrt{a_{22} - l_{21}^2} = \sqrt{3 - 1} = \sqrt2$。

故 $L = \begin{pmatrix} 2 & 0 \\ 1 & \sqrt2 \end{pmatrix}$，验证 $LL^T = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$ ✓。

## 3 公式解析：为什么 Cholesky 等于「正定判据」

把 Cholesky 与正定性打通，拆成四步：

- **第一步，正定 ⇒ Cholesky**：$A$ 正定则 LU 中 $U$ 的对角元全正，配成 $LL^T$ 时每个对角元取正平方根，得到对角元为正的 $L$。
- **第二步，Cholesky ⇒ 正定**：若 $A = LL^T$（$L$ 可逆，下三角对角元正），对任意 $\mathbf{x} \ne \mathbf{0}$：$\mathbf{x}^TA\mathbf{x} = \|L^T\mathbf{x}\|^2 > 0$（$L^T\mathbf{x} \ne \mathbf{0}$）。**正定性的定义式直接成立**。
- **第三步，判据形成**：对称矩阵正定 ⟺ Cholesky 分解存在且唯一。
- **第四步，工程意义**：计算 Cholesky 只需约 $\frac{n^3}{3}$ 次运算（LU 的 $\frac{2n^3}{3}$ 的一半），且不需要选主元（正定保证主元正）。**又快又稳**。

<span class="marginnote"><strong>「$\|L^T x\|^2 > 0$」这一行是 Cholesky 与正定性的桥梁</strong>：分解的存在使「处处为正」从「验证无穷多个 $x$」变成「看一个分解」。类似地，$A = BB^T$（$B$ 行满秩）也是半正定的分解判据——数据矩阵的 $X^TX$ 正是这种形式。</span>

## 4 Cholesky 的应用

- **解方程组**：$Ax = b$ ⇒ $Ly = b$（前代）+ $L^Tx = y$（回代），正定系统的最优解法。
- **求逆与行列式**：$\det A = (\det L)^2 = \prod l_{ii}^2$——**行列式 = 对角元平方之积**。
- **多元正态抽样**：若 $\Sigma = LL^T$，则 $L\mathbf{z}$（$\mathbf{z}$ 为标准正态向量）服从 $N(0, \Sigma)$——**用 Cholesky 因子把不相关样本变成相关样本**。
- **最小二乘**：正规方程 $A^TA\hat{x} = A^Tb$ 中 $A^TA$ 正定（$A$ 列满秩），可用 Cholesky 分解（数值上仍推荐 QR/SVD，但 Cholesky 是小规模正规方程的经典算法）。
- **矩阵补全与数理统计**：条件协方差、卡尔曼滤波的更新步都用 Cholesky。

**重点**：Cholesky 是「正定矩阵的瑞士军刀」——**一个分解，解方程、行列式、抽样、回归全部搞定**。在数据科学里，任何「对称正定」出现的地方，Cholesky 都是默认的数值工具。

## 5 Cholesky 与半正定、可逆性的辨析

- **正定** ⇒ 存在 $LL^T$，$L$ 对角元正、可逆；
- **半正定** ⇒ 存在 $LL^T$，但 $L$ 对角元可能为零（$L$ 不可逆）；
- **可逆但不对称** ⇒ 没有 Cholesky，用 LU；
- **对称但非正定** ⇒ Cholesky 中途失败（负开方）。

**辨析｜易错点：** 不要混淆 $LL^T$（Cholesky，$L$ 方阵下三角）与 $A = BB^T$（$B$ 可以是长方形、$m \times n$，对应半正定）。**「$A = X^TX$ 恒为半正定」**是数据科学最常用的事实——任何数据矩阵 $X$ 的 Gram 矩阵 $X^TX$ 都半正定（第十一篇协方差矩阵）。

**补充｜Cholesky 与「随机模拟」**：多元正态抽样的标准做法是 Cholesky：若 $\Sigma = LL^T$，生成独立标准正态 $\mathbf{z}$，则 $L\mathbf{z} \sim N(\mathbf{0}, \Sigma)$。为什么有效？$\operatorname{Cov}(L\mathbf{z}) = L\operatorname{Cov}(\mathbf{z})L^T = LL^T = \Sigma$——**协方差结构由 $L$ 精确搬运**。这个技巧在蒙特卡洛模拟、金融定价、贝叶斯计算里无处不在。**「用 Cholesky 因子把白噪声染上相关性」**是统计模拟的经典开场。

**补充｜Cholesky 与「随机模拟」**：多元正态抽样的标准做法是 Cholesky：若 $\Sigma = LL^T$，生成独立标准正态 $\mathbf{z}$，则 $L\mathbf{z} \sim N(\mathbf{0}, \Sigma)$。为什么有效？$\operatorname{Cov}(L\mathbf{z}) = L\operatorname{Cov}(\mathbf{z})L^T = LL^T = \Sigma$——**协方差结构由 $L$ 精确搬运**。这个技巧在蒙特卡洛模拟、金融定价、贝叶斯计算里无处不在。**「用 Cholesky 因子把白噪声染上相关性」**是统计模拟的经典开场。

**辨析｜易错点：** Cholesky 与 LU 的关系：

- Cholesky 是正定矩阵的「对称 LU」：$A = LL^T$，$L$ 直接是下三角；
- 普通 LU 是 $A = LU$（$L$ 单位下三角、$U$ 上三角），两者不冲突——对正定对称矩阵，$U$ 与 $L^T$ 只差一个对角缩放；
- **运算量**：Cholesky 约 $n^3/3$，LU 约 $2n^3/3$——Cholesky 快一半。

**「对称 + 正定 ⇒ Cholesky」**是它的适用前提，缺一不可。

**补充｜Cholesky 的「一句话」**：**「正定矩阵的对称版 LU = $A = LL^T$」**——比 LU 快一半、不用选主元、对角元即正定性的证据。见到「对称正定」，第一反应就是 Cholesky。

**补充｜Cholesky 的适用判断清单**：

- $A$ 对称吗？不对称 → 用 LU；
- $A$ 正定吗？不正定 → Cholesky 中途会「对负数开方」而失败；
- 要解方程/求逆/抽样？正定对称 → Cholesky 是最优选择。

**「对称 + 正定 ⇒ Cholesky」**，这是它相对 LU 的全部优势来源。

## 6 小结

- **Cholesky 分解**：$A = LL^T$，$L$ 下三角对角元为正，对称正定矩阵唯一分解。
- **算法**：逐列求，对角元开方、非对角元做差除法。
- **判据**：对称矩阵正定 ⟺ Cholesky 存在；半正定 ⟺ 存在 $LL^T$（$L$ 可不逆）。
- **应用**：正定方程组求解、$\det A = \prod l_{ii}^2$、多元正态抽样、最小二乘。
- **效率**：$\approx \frac{n^3}{3}$ 次运算，无需选主元，LU 的一半成本。

在下一节，我们将用特征分解的语言重新陈述「谱」——**谱定理：对称矩阵的特征分解**，把实对称矩阵的一切结构收进一个公式。
