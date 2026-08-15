---
title: Borel 集类与 Fσ、Gδ 型集
date: 2026-08-07
---

# Borel 集类与 Fσ、Gδ 型集

<div class="epigraph">
<p>我从开集出发，用可数并、可数交与补集编织一张网，网住的集合就叫 Borel 集。</p>
<footer>—— 埃米尔 · 波莱尔（Émile Borel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》§2.7 ｜ 2026-08-07</p>
</div>

## 为什么从 Borel 集开始

开集与闭集是「干净」的集合，但分析学很快遇到需要「脏」集合的场合：连续函数的收敛点集、可微点集、三角级数的收敛集，往往既不开也不闭，却恰好是「开集的可数交」或「闭集的可数并」。Borel 集类正是为这类集合准备的**分层容器**：从开集出发，反复施加可数并、可数交与取补，得到的全部集合。

Borel 集类的重要性怎么强调都不过分：**Lebesgue 可测集的家族就是以 Borel 集为骨架、再添上零测集拼成的**。可测函数、可测性判定的所有论证，最终都归结为「某集合是 Borel 集」或「某集合与 Borel 集相差零测集」。学 Borel 集，是在给后面全部测度论证铺设分类学。<span class="marginnote">Borel 集类的严格定义需要<strong>超限递归</strong>：$\Sigma_1^0$ 是开集，$\Sigma_{\alpha+1}^0$ 是 $\Pi_\alpha^0$ 的可数并，$\Pi_\alpha^0$ 是 $\Sigma_\alpha^0$ 的补……沿所有可数序数往上叠。这产生了 Borel 层级（Borel hierarchy），是描述集合论的起点。</span>

## 1 Fσ 集与 Gδ 集：Borel 的第一层

先看最常用的两类二阶集合：

**定义（Fσ 集）**：能写成**可数个闭集的并**的集合，记作 $F_\sigma$ 型集。

**定义（Gδ 集）**：能写成**可数个开集的交**的集合，记作 $G_\delta$ 型集。

记号来源：$F$ 取自法语 *fermé*（闭），$\sigma$ 表示可数和；$G$ 取自德语 *Gebiet\*（区域，开集），$\delta$ 表示可数交（Durchschnitt）。它们是对开、闭的**一层提升**：

- **例（Fσ）**：$\mathbb{Q}=\bigcup_{r\in\mathbb{Q}}\{r\}$ 是单点集（闭）的可数并，故 $\mathbb{Q}$ 是 $F_\sigma$ 集。$\mathbb{R}\setminus\mathbb{Q}$ 呢？它是 $G_\delta$ 集（无理数集 $=\bigcap_{r\in\mathbb{Q}}(\mathbb{R}\setminus\{r\})$，每个 $\mathbb{R}\setminus\{r\}$ 是开集）。
- **例（Gδ）**：连续函数的连续点集是 $G_\delta$ 集；单调函数的不连续点集是 $F_\sigma$ 集。

**重点：$F_\sigma$ 与 $G_\delta$ 是互补对。** $E$ 是 $F_\sigma$ $\iff$ $E^c$ 是 $G_\delta$。这是因为可数并的补是可数交：$\left(\bigcup_k F_k\right)^c=\bigcap_k F_k^c$，闭的补是开。这个对偶贯穿 Borel 层级每一层——「闭/开」被提升为「可数并的闭 / 可数交的开」。

## 2 更高层与 Borel 集类

继续提升：$F_{\sigma\delta}$ 是可数个 $G_\delta$ 的并，$G_{\delta\sigma}$ 是可数个 $F_\sigma$ 的交，如此层层交替。**Borel 集类（Borel class）** $\mathcal{B}$ 定义为：从开集出发，对可数并、可数交、取补三种运算反复封闭得到的**最小**集族。

$$G\ \text{开}\Rightarrow G\in\mathcal{B};\qquad \mathcal{B}\ \text{关于}\ \bigcup_{k=1}^{\infty},\ \bigcap_{k=1}^{\infty},\ ^c\ \text{封闭}$$

**定理（Borel 集的基数）**：$\mathcal{B}$ 的基数为 $c$（连续统基数）。

证明要点：Borel 集可以由「构造它的运算序列」编码——每次运算选一个可数子族，一个 Borel 集对应一个「可数长度的运算树」，这样的树的全体与实数集等势。**结论：Borel 集有 $c$ 个，而 $[0,1]$ 的子集有 $2^c$ 个，所以「绝大多数子集不是 Borel 集」。** 这为下一章的不可测集预留了巨大的空间。

**辨析｜易错点：$F_\sigma$ 与 $G_\delta$ 是「一层」，Borel 集是「无穷层」，两者是包含关系而非相等。** $F_{\sigma\delta}$ 集未必是 $F_\sigma$ 或 $G_\delta$。事实上 Borel 层级严格单调上升：$\Sigma_1^0\subsetneq\Pi_1^0\subsetneq\Sigma_2^0\subsetneq\Pi_2^0\subsetneq\cdots$，每一层都有「全新」的集合。<span class="marginnote">「是否存在既非 $F_\sigma$ 也非 $G_\delta$ 的 Borel 集」的答案是肯定的，且逐层都有新例。用<strong>对角线法</strong>可以在 $\omega_1$ 处造出不在任何可数层里的 Borel 集——这已是描述集合论的内容，此处只需知道层级不坍缩。</span>

## 3 Borel 集与开闭集的关系：正则性

Borel 集类虽大，仍被开集与闭集「夹」得很紧，这就是**正则性**：

**定理（正则性）**：设 $E$ 是 Borel 集。则对任意 $\varepsilon>0$，存在闭集 $F$ 与开集 $G$，使得

$$F\subset E\subset G,\qquad m(G\setminus F)<\varepsilon$$

其中 $m$ 是 Lebesgue 外测度（下一章）。即：**Borel 集可以从内部用闭集逼近、从外部用开集逼近，误差任意小。**

证明思路：先对开集验证（开集本身可取 $F$ 为「去掉薄边」的闭子集）；「可数并/交/补」三种操作都保持这种逼近性质，由归纳推及全体 Borel 集。这条性质在 Lebesgue 测度的定义中举足轻重：**外测度用开集定义、正则性保证「可测集 ≈ 开集（外）≈ 闭集（内）」**，是下一章逼近定理的 Borel 版预告。

## 4 公式解析：一个集合如何一层层长成 Borel 集

以「单调函数的不连续点集是 $F_\sigma$ 集」为例，走一遍典型论证：

$$D(f)=\left\{x:\omega_f(x)>0\right\}=\bigcup_{n=1}^{\infty}\underbrace{\left\{x:\omega_f(x)\ge\tfrac1n\right\}}_{=:\,D_n}$$

其中 $\omega_f(x)$ 是 $f$ 在 $x$ 点的振幅（oscillation）。逐步拆解：

- **第一步，把「不连续」改写成「振幅大于 0」**：$f$ 在 $x$ 连续 $\iff$ $\omega_f(x)=0$。于是 $D(f)=\{x:\omega_f(x)>0\}$。
- **第二步，把「$>0$」分解成「$\ge\tfrac1n$」的可数并**：$\{x:\omega_f(x)>0\}=\bigcup_{n=1}^{\infty}D_n$，因为「正数」总落在某个 $\tfrac1n$ 之上。「$>$」改写成「$\ge\tfrac1n$」是分析学把开区间拆成闭区间闭包的标准手法。
- **第三步，证明每个 $D_n$ 是闭集**：若 $x_k\in D_n$ 且 $x_k\to x$，则振幅函数满足 $\omega_f(x)\ge\limsup\omega_f(x_k)\ge\tfrac1n$，故 $x\in D_n$。**可数个闭集之并 $=F_\sigma$ 集，证毕。**

**这套「开区间拆闭区间」的手法贯穿全书**：它把「正测度 / 非空」这类开性条件，翻译成「$\ge\tfrac1n$」的闭性条件，从而让有限覆盖、Borel 层级这些闭集工具派上用场。

## 6 数值演练与 Borel 层级速查

**算例一（$\mathbb{Q}$ 的 Borel 身份）**：$\mathbb{Q}=\bigcup_{r\in\mathbb{Q}}\{r\}$——单点（闭）的可数并，$F_\sigma$ 集。$\mathbb{R}\setminus\mathbb{Q}=\bigcap_{r\in\mathbb{Q}}(\mathbb{R}\setminus\{r\})$——开集的可数交，$G_\delta$ 集。**稠密可数集与其补集正好分居 $F_\sigma$ 与 $G_\delta$ 两侧。**

**算例二（连续点集是 $G_\delta$）**：$f$ 的连续点集 $C(f)=\bigcap_n\bigcup_{\delta}\{x:\sup_{|x'-x|<\delta}|f(x')-f(x)|<\tfrac1n\}$——每个 $\{\cdot\}$ 对固定 $n$ 取合适的 $\delta$ 是开集，可数交得 $G_\delta$。**「连续」这一性质在 Borel 层级上落在 $G_\delta$ 层。**

**对照表：Borel 层级速查**

| 层级 | 定义 | 例 |
| --- | --- | --- |
| $\Sigma_1^0$ | 开集 | $(0,1)$ |
| $\Pi_1^0$ | 闭集 | $[0,1]$ |
| $F_\sigma$ | 可数闭并 | $\mathbb{Q}$ |
| $G_\delta$ | 可数开交 | $\mathbb{R}\setminus\mathbb{Q}$ |
| $F_{\sigma\delta}$ | 可数 $G_\delta$ 并 | 更深层新例 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| $F_\sigma$ | 可数闭并 |
| $G_\delta$ | 可数开交 |
| $\mathcal{B}$ | Borel 集类 |
| 振幅 $\omega_f$ | 局部振荡大小 |

**辨析｜易错点：Borel 集不限于 $F_\sigma$ 或 $G_\delta$——层级无穷深。** $F_{\sigma\delta}$ 集未必 $F_\sigma$。**「多数子集不是 Borel 集」**：$[0,1]$ 有 $2^c$ 个子集，Borel 集仅 $c$ 个——不可测的嫌疑对象极其众多。

### 三步看穿「$F_\sigma$ 论证」

- **改写**：不连续 ⇔ 振幅 $>0$。
- **分层**：$>0$ ⇔ $\bigcup_n\{\omega_f\ge\tfrac1n\}$。
- **闭化**：每层 $\{\omega_f\ge1/n\}$ 闭，可数并得 $F_\sigma$。

**延伸（与测度论连接）**：Lebesgue 可测集正是「Borel 集 ⊕ 零测修补」；正则性（开外闭内逼近）保证每个可测集都被 Borel 集夹住。**「Borel 骨架 + 零测血肉」是可测集的结构定理。**

**一道收束练习**：证明连续函数 $f$ 的值域是区间（连通）但「$f$ 的不可微点集」是 $F_\sigma$ 集——用振幅手法把「可微性」也翻译成 Borel 语言。

## 7 小结

- **$F_\sigma$ / $G_\delta$**：可数个闭集的并 / 可数个开集的交；互补对（$E\in F_\sigma\iff E^c\in G_\delta$）。
- **Borel 集类**：对开集施以可数并、可数交、补集三种运算反复封闭的最小族；基数 $c$。
- **层级不坍缩**：$F_{\sigma\delta}$、$G_{\delta\sigma}$……严格递增，且「多数子集不是 Borel 集」（$2^c$ 个子集中只有 $c$ 个是）。
- **正则性**：Borel 集可用开集外逼近、闭集内逼近，误差任意小——Lebesgue 测度逼近定理的雏形。
- **振幅手法**：「$>0$」拆成「$\ge\tfrac1n$」的可数并，是分析论证的常备武器。
- **数值**：$\mathbb{Q}$ 是 $F_\sigma$，$\mathbb{R}\setminus\mathbb{Q}$ 是 $G_\delta$。

在下一节，我们将进入紧致性的世界：证明 **有限覆盖定理与可数覆盖定理**，这是「开区间」覆盖「闭区间」时「无穷」与「有限」第一次握手。
