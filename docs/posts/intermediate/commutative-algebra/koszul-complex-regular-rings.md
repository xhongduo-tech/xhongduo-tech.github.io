---
title: Koszul 复形与正则环
date: 2026-08-11
---

# Koszul 复形与正则环

<div class="epigraph">
<p>正合列是这个学科的语法，而 Koszul 复形是其中最自然的一句。</p>
<footer>—— 让-路易 · 科斯居尔（Jean-Louis Koszul）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 交换代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从正则序列开始

上一节，正则局部环用「嵌入维数 = 维数」来定义——数切空间。这一节换一副眼镜：**正则性能否用「乘法算子是否无核」来刻画？** 答案是肯定的，而且刻画工具是一个精巧的复形——**Koszul 复形**。它把一个元素序列的「正则性」（每个元素对商模都不是零因子）翻译成复形的正合性，最终导出 Serre 的同调刻画：**正则局部环 = 有限同调维数的环**。<span class="marginnote">Koszul 复形由 Jean-Louis Koszul（1921—2018）1950 年在巴黎的博士论文中系统研究。它在代数几何里描述「正则序列张出的几何」，在表示论里描述「对角嵌入的系数」——同一个复形，多张面孔。</span>

这一篇是「同调代数进交换代数」的第一课：需要一点复形、同调、正合列的语言（第1篇《理想、模与环同态》打好了底），换来的是正则性、深度、Cohen–Macaulay 性全部同调化。

## 1 正则序列：逐层「非零因子」

**正则序列（regular sequence）**：$M$ 是 $A$-模，序列 $x_1, \dots, x_r \in A$ 称为 $M$ 上的正则序列，若
$x_1$ 是 $M$ 上的**非零因子**（$x_1 m = 0 \Rightarrow m = 0$）；
- 且 $x_i$ 是 $M/(x_1, \dots, x_{i-1})M$ 上的非零因子（$i = 2, \dots, r$）；
- 且 $M/(x_1,\dots,x_r)M \neq 0$。

标准例子：$k[x,y]$ 中 $x, y$ 是正则序列（$x$ 非零因子，$y$ 在 $k[x,y]/(x) \cong k[y]$ 上非零因子）；$x, xy$ 不是（在 $k[x,y]/(x) \cong k[y]$ 上 $xy = 0$，$xy$ 是零因子）。

**重点：正则序列是「乘法算子无核的序列」。** 每次除以前几个生成元，新的生成元在商上仍不做坏事——这个「层层无核」正是几何里「横截相交」的代数化：一个方程在簇上不是零因子，意味着它真的把维数压低一维。<span class="marginnote">直觉例子：在 $k[x,y]$ 中 $x$ 是整环上非零因子，$V(x)$ 是直线，维数从 2 到 1；$x, y$ 连取两个非零因子，维数到 0——「非零因子」每次压低一维，正则序列 = 每次都压低一维的坐标序列。</span>

**辨析｜易错点：** 正则序列对**顺序**敏感，且要求「最后的商非零」。$x, y$ 是 $k[x,y]$ 的正则序列，但 $y, x$ 也是；而 $x^2, xy$ 不是正则序列（$x^2$ 非零因子，但 $xy$ 在 $k[x,y]/(x^2)$ 上是零因子：$xy \cdot x = x^2 y = 0$，而 $x \neq 0$）。**判断正则性要逐层检查，不能只看总体。**

## 2 Koszul 复形：把一个元素序列装进复形

对单个元素 $x \in A$ 与模 $M$，定义复形 $K(x; M)$：

$$K(x; M): \qquad 0 \longrightarrow M \overset{\cdot x}{\longrightarrow} M \longrightarrow 0.$$

对序列 $x_1, \dots, x_r$，取张量积

$$K(x_1, \dots, x_r; M) = K(x_1; A) \otimes_A \cdots \otimes_A K(x_r; A) \otimes_A M.$$

它形如

$$0 \longrightarrow \Lambda^r \to \Lambda^{r-1} \to \cdots \to \Lambda^1 \to \Lambda^0 \longrightarrow 0$$

（$\Lambda^p$ 是自由模，$\Lambda^0 = M$，$\Lambda^r = M$），各阶同调记为 $H_i(x; M)$，称为 **Koszul 同调**。$H_0(x; M) = M/(x_1,\dots,x_r)M$。

例子：$K(x, y; k[x,y])$ 是

$$0 \to k[x,y] \xrightarrow{\binom{-y}{x}} k[x,y]^2 \xrightarrow{(x\ \ y)} k[x,y] \to 0$$

——这正是「$k[x,y]$ 是它的商 $k$ 的自由分解」的起点，也是代数几何里「余切复形」的原型。<span class="marginnote">当 $x_1,\dots,x_r$ 生成 $\mathfrak{m}$ 时，$K(x; R)$ 就是 $k = R/\mathfrak{m}$ 的一个自由分解的<strong>候选者</strong>；它正合与否，恰好回答「参数系是不是正则的」。这是后续深度理论的关键机制。</span>

## 3 Koszul 同调与正则性

**重点：$x_1, \dots, x_r$ 是 $M$ 上的正则序列，当且仅当 $H_i(x; M) = 0$ 对所有 $i \geq 1$ 成立。** 此时 $K(x; M)$ 是 $M/(x)M$ 的一个自由分解——「正则序列 ⇔ Koszul 复形正合」。

证明用单元素情形归纳：$K(x; M)$ 的正合性（$H_1 = 0$）恰恰等价于 $\cdot x$ 是单射，即 $x$ 是非零因子；把模换成 $M/(x_1,\dots,x_{i-1})M$ 逐层套用，就得到全序列版本。**这是「乘法无核」与「复形正合」的第一次握手。**

**Serre 定理（同调刻画）**：Noether 局部环 $(R, \mathfrak{m}, k)$ 是正则的，当且仅当存在参数系生成的正则序列（即 $\mathfrak{m}$ 由正则序列生成），当且仅当**每个有限生成 $R$-模都有有限投射维数**：

$$R \text{ 正则} \iff \operatorname{gldim} R < \infty \iff \operatorname{pd}_R(k) < \infty.$$

**辨析｜易错点：** Koszul 复形正合性只对「正则序列」成立；「$\mathfrak{m}$ 由 $d$ 个元素生成」只是参数系，未必正则。$R = k[x,y]/(x^2, xy)$ 时参数系 $\{x\}$ 张出 $\mathfrak{m}$-准素理想，但 $x$ 在 $R$ 上是零因子（$x \cdot y = 0$），Koszul 复形不正合——**正合性才是正则性的真相**，光数个数不够。<span class="marginnote">这个反例同时预告深度理论：正则序列的长度上限就是「深度」，而 $R = k[x,y]/(x^2,xy)$ 的深度是 0。</span>

## 4 公式解析：单元素的 Koszul 正合性

最微小的例子已经承载全部机制。设 $x \in A$，$M$ 是 $A$-模：

$$K(x; M): \quad 0 \to M \xrightarrow{\cdot x} M \to 0 \qquad\text{正合} \iff x \text{ 是 } M \text{ 上的非零因子}.$$

- **第一步，读复形**：两个非零项之间只有 $\cdot x$ 这个映射。正合性要求 $\ker(\cdot x) = 0$（中间处的核等于像）且 $\operatorname{im}(\cdot x) = M$（末端的像等于整个模）。
- **第二步，等价翻译**：$\ker(\cdot x) = 0$ 就是「$xm = 0 \Rightarrow m = 0$」即非零因子；而 $\operatorname{im}(\cdot x) = M$ 需要 $x$ 可逆——但这只是 $H_0$ 位置的平凡等式，通常不要求。于是单元素情形的正合性「几乎」就是非零因子性。
- **第三步，升级**：多元素时把 $M$ 换成逐层商模，$H_i = 0$（$i \geq 1$）就逐条等价于正则序列的全部条件。**一个「非零因子」的检查，在 Koszul 语言里变成一整条复形的正合性。**

**辨析｜易错点：** 别把「Koszul 复形正合」与「$\cdot x$ 是满射」混为一谈。Koszul 复形只要求中间各阶同调消失；$H_0 = M/(x)M$ 不必为零。正合是「层层无核」，不是「一一对应」。

## 5 小结

- **正则序列**：每个元素在逐层商模上都是非零因子——几何上「每次都压低一维」。
- **Koszul 复形** $K(x_1,\dots,x_r; M)$：由乘法算子搭成的复形，$H_0 = M/(x)M$。
- **判据**：正则序列 ⇔ $H_i(x; M) = 0$（$i \geq 1$）⇔ Koszul 复形是商模的自由分解。
- **Serre 定理**：正则局部环 ⇔ 有限同调维数；「正则性 = 同调有限」。

在下一节，我们做一个基础的拼装：**张量积与平坦模**——它是局部化、基变换、纤维的全部底层机制，也是「张量积为什么只保一半正合」问题的正式回答。
