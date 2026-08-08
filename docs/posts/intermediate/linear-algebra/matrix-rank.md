---
title: 矩阵的秩及其性质
date: 2026-08-07
---

# 矩阵的秩及其性质

<div class="epigraph">
<p>秩是矩阵的「身份证号」：无论你用行还是列、用哪个基去度量，它都给出同一个数——矩阵真正携带的信息量。</p>
<footer>—— 弗罗贝尼乌斯（Ferdinand Georg Frobenius）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§3.3 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵的秩开始

前面的消元、求逆都围绕「这个矩阵能不能解、可不可逆」打转。但要回答「这个矩阵到底有多『满』」——有多少行/列是真正独立、不含冗余的——需要一个数来度量，这个数就是**矩阵的秩（rank）**。<span class="marginnote">秩是全课程最「稳健」的量：它不随初等变换改变，是初等变换下的不变量；可逆矩阵的秩是满的，奇异矩阵的秩「缺一截」。在数据科学里，数据矩阵的秩决定数据真正占用的维度（第十一篇），SVD 的截断低秩近似（第十篇）也以秩为核心。</span>

秩把之前分散的判据全部收编：$\operatorname{rank} A = n \Leftrightarrow A$ 可逆。更重要的是，它将引领我们进入第四篇——秩是「向量组线性无关个数」的精确度量。

## 1 秩的定义

**核心概念**：设 $A$ 是 $m \times n$ 矩阵，对 $A$ 施行初等行变换化为**行阶梯形**，其**非零行的个数** $r$ 称为矩阵 $A$ 的**秩**，记作 $\operatorname{rank}(A)$ 或 $R(A)$。

**重点**：秩与化成哪种行阶梯形无关——行阶梯形不唯一，但**非零行个数唯一**。这一事实来自行最简形的唯一性：RREF 是唯一的「指纹」，主元个数唯一，而行阶梯形的非零行数 = 主元数。

例：$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}$，第二行是第一行的 2 倍、第三行是第一行的 3 倍，消元后只剩一行非零，$\operatorname{rank} A = 1$。

几个立即成立的性质：

$0 \le \operatorname{rank}(A) \le \min(m, n)$——秩不超过行数也不超过列数。
$\operatorname{rank}(A) = 0 \Leftrightarrow A = O$（零矩阵）。
$A$ 是 $n$ 阶方阵时，$\operatorname{rank} A = n \Leftrightarrow A$ 可逆，此时称 $A$ 为**满秩矩阵**。

## 2 行秩 = 列秩：秩的双面身份

秩的定义用了「非零行个数」，但换一个方向也成立：

**定理（行秩 = 列秩）**：矩阵 $A$ 的**行向量组的秩**（行向量中极大线性无关组所含向量个数）等于其**列向量组的秩**，且都等于 $\operatorname{rank} A$。

这个定理说明：**秩不偏向「行」也不偏向「列」**——一个 $3 \times 100$ 的矩阵，列很多但秩至多 3；它的「真实维度」由行秩和列秩共同决定。<span class="marginnote">行秩 = 列秩是线性代数最令人惊讶的早期定理之一。直观上，行是「方程的个数视角」，列是「变量的个数视角」，二者表面无关，秩却强制它们相等——这正是方程组「有效方程数」与「有效变量维数」最终对账的原因。</span>

**重点**：初等**行**变换保持行空间不变（行向量组的秩不变）；初等**列**变换保持列空间不变。而因为行秩 = 列秩，初等行变换其实**同时不改变列秩**——这是「初等变换保持秩」这一基本性质的深层理由。

## 3 秩的性质：与运算如何配合

秩与矩阵运算满足一组重要不等式：

$$
\operatorname{rank}(A + B) \le \operatorname{rank} A + \operatorname{rank} B
$$

$$
\operatorname{rank}(AB) \le \min\{\operatorname{rank} A, \operatorname{rank} B\}
$$

第一条说明「两个矩阵相加，秩最多叠加」；第二条说明「乘积的秩不超过任一因子的秩」——乘法会让信息**变少或持平，绝不会凭空增多**。这与「映射复合不会创造新方向」的直觉一致：$B$ 的输出空间被 $A$ 压缩，不可能恢复 $B$ 已丢失的信息。

**辨析｜易错点：** $\operatorname{rank}(AB) \le \operatorname{rank} A$ 允许「等号成立」（比如 $A$ 可逆时 $\operatorname{rank}(AB) = \operatorname{rank} B$）。**左乘可逆矩阵不改变秩**：$\operatorname{rank}(PA) = \operatorname{rank} A$（$P$ 可逆）。同理右乘可逆矩阵也不改变秩。但若 $P$ 奇异，秩可能下降——这正是不等式而非等式的原因。

## 4 公式解析：可逆左乘不改变秩

$\operatorname{rank}(PA) = \operatorname{rank} A$（$P$ 可逆）这条性质值得单独拆开：

- **第一步，为何左乘可逆不降秩**：$P$ 可逆，所以 $P$ 的列线性无关，$P$ 把任何非零向量映到非零向量（单射）。于是 $A$ 的列若线性无关，$PA$ 的对应列仍线性无关——列秩不降。
- **第二步，为何不增秩**：$PA$ 的每一列都是 $P$ 左乘 $A$ 的一列，是「$A$ 的各列经 $P$ 作用」的结果。列空间被映射，不可能超出 $P$ 的值域，而 $P$ 可逆时值域是全空间——实际上 $\operatorname{rank}(PA) \le \operatorname{rank} A$ 由一般不等式给出。
- **第三步，合起来**：$\operatorname{rank}(PA) \le \operatorname{rank} A$ 且因 $P$ 可逆又有 $\operatorname{rank} A = \operatorname{rank}(P^{-1}PA) \le \operatorname{rank}(PA)$，两边夹即得等式。
- **第四步，推论**：初等矩阵都可逆，所以初等变换保持秩；可逆矩阵本质上是「初等变换的乘积」，故左乘可逆矩阵 = 一系列保持秩的操作。

<span class="marginnote">这个「两边夹」证明是线性代数中不等式论证的典型范式：先证 $\le$，再用逆矩阵把方向反过来，从而得等式。同样的手法在 SVD、范数不等式里反复出现。</span>

## 5 秩的应用：判断方程组解的个数

秩是判断线性方程组解的「总开关」。设 $A$ 是 $m \times n$ 矩阵，增广矩阵 $(A \mid b)$，则（下一节将系统展开）：

- $Ax = b$ **无解** $\Leftrightarrow \operatorname{rank}(A) < \operatorname{rank}(A \mid b)$（矛盾行出现）；
- $Ax = b$ **有唯一解** $\Leftrightarrow \operatorname{rank}(A) = \operatorname{rank}(A \mid b) = n$；
- $Ax = b$ **有无穷多解** $\Leftrightarrow \operatorname{rank}(A) = \operatorname{rank}(A \mid b) < n$，自由变量个数 $= n - \operatorname{rank} A$。

**重点**：这三个判据合起来是全书最重要的应用公式之一。它把「解的情况」完全压缩进两个秩的比较里——只算秩，不真正解方程，就能预知解的形态。

**补充｜秩的「连续化」：数值秩与有效秩**：浮点计算里「秩」是脆弱的——一个元素上微小扰动就会改变精确秩。数值线性代数用**数值秩（numerical rank）**替代：奇异值大于某个阈值（如 $\sigma_{\max}\cdot\varepsilon$）的个数。数据科学更常用**有效秩**：奇异值累积能量达到某比例（如 95%）所需的个数。**「秩从离散到连续」**——精确秩是「有没有信息」，数值/有效秩是「有多少信息」——这正是 SVD 低秩近似（第十篇）的出发点。

**补充｜秩与「自由度」的直观**：秩 = 独立行数 = 独立列数 = 独立约束数 = 保留的自由度。一个 $3 \times 3$ 秩 1 矩阵，三行全都指向同一方向——它描述的信息「只有一条线」。这个「秩 = 有效信息维度」的解读，是 PCA、压缩、回归里反复使用的直觉。

**补充｜秩的「连续化」：数值秩与有效秩**：浮点计算里「秩」是脆弱的——一个元素上微小扰动就会改变精确秩。数值线性代数用**数值秩（numerical rank）**替代：奇异值大于某个阈值（如 $\sigma_{\max}\cdot\varepsilon$）的个数。数据科学更常用**有效秩**：奇异值累积能量达到某比例（如 95%）所需的个数。**「秩从离散到连续」**——精确秩是「有没有信息」，数值/有效秩是「有多少信息」——这正是 SVD 低秩近似（第十篇）的出发点。

**补充｜秩与「自由度」的直观**：秩 = 独立行数 = 独立列数 = 独立约束数 = 保留的自由度。一个 $3 \times 3$ 秩 1 矩阵，三行全都指向同一方向——它描述的信息「只有一条线」。这个「秩 = 有效信息维度」的解读，是 PCA、压缩、回归里反复使用的直觉。

**辨析｜易错点：** 秩与行列式的关系在考试里极常考：

- $\operatorname{rank} A = n$ ⇔ $\det A \ne 0$（$n$ 阶方阵满秩）；
- $\operatorname{rank} A < n$ ⇔ $\det A = 0$（方阵降秩）；
- $\operatorname{rank} A = r$ ⇔ 存在非零 $r$ 阶子式，且所有 $r+1$ 阶子式为零——这是「用子式定义秩」的等价说法。

**「行列式判满秩、子式判具体秩」**是两条互补的路径：前者快，后者精细。

**补充｜秩的应用清单（复习与延伸）**：

- 判断线性方程组解的情况：$\operatorname{rank}A$ 与 $\operatorname{rank}(A\mid b)$ 的比较（第三篇）；
- 判断向量组相关/无关：$\operatorname{rank}A = n$ ⇔ 列无关（第四篇）；
- 判断可逆性：$n$ 阶方阵满秩 ⇔ 可逆 ⇔ $\det A \ne 0$；
- 判断基础解系个数：$\dim\operatorname{Nul}(A) = n - \operatorname{rank}A$；
- 四子空间维数：列/行空间维数 $= \operatorname{rank}A$（第七篇）；
- 奇异值个数：$\operatorname{rank}A$ = 非零奇异值个数（第十篇）。

**「一个秩，六个用途」**——它是全书出场率最高的概念之一。

**补充｜一句话**：**「秩 = 矩阵有效信息的维度」**——它是消元、行列式、子空间、SVD 之间共同的「度量单位」。

## 6 小结

- **定义**：$\operatorname{rank} A$ = 行阶梯形非零行个数 = 主元个数，唯一确定。
- **行秩 = 列秩**：行向量组与列向量组的秩相等，统一为矩阵的秩。
- **性质**：$\operatorname{rank}(A+B) \le \operatorname{rank} A + \operatorname{rank} B$；$\operatorname{rank}(AB) \le \min\{\operatorname{rank} A, \operatorname{rank} B\}$。
- **可逆不改变秩**：$\operatorname{rank}(PA) = \operatorname{rank} A$（$P$ 可逆）；满秩方阵 $\operatorname{rank} = n \Leftrightarrow$ 可逆。
- **解判据**：$Ax = b$ 的解情况由 $\operatorname{rank} A$ 与 $\operatorname{rank}(A \mid b)$ 的比较完全决定。

在下一节，我们将把秩这把尺子用到解方程组上——**线性方程组的解：高斯消元与解的判定定理**，系统给出「无解 / 唯一解 / 无穷多解」的完整判别法与几何图景。
