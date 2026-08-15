---
title: 完备化与 Hensel 引理
date: 2026-08-07
---

# 完备化与 Hensel 引理

<div class="epigraph">
<p>完备性……是数学分析的心脏；把它搬到代数里，就得到 $p$-进数。</p>
<footer>—— 库尔特 · 亨泽尔（Kurt Hensel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 交换代数 ｜ Atiyah–Macdonald Ch. 10 ｜ 2026-08-07</p>
</div>

## 为什么从完备化开始

分析学里，$\mathbb{R}$ 是 $\mathbb{Q}$ 的完备化——把「越来越接近」的点补成极限点。交换代数用完全相同的思路：给定环 $A$ 与理想 $\mathfrak{a}$，把「按 $\mathfrak{a}$ 的幂越来越接近」的点补全，得到 **$\mathfrak{a}$-进完备化** $\widehat{A}$。最著名的例子是 $p$-进整数环 $\mathbb{Z}_p$——亨泽尔 1897 年正是从「$p$-进度量下的完备化」构造了它。<span class="marginnote">Hensel 当初造 $\mathbb{Z}_p$ 的动机是类比幂级数：$\mathbb{Q}_p$ 之于 $\mathbb{Z}_p$，正如 $\mathbb{Q}((t))$ 之于 $\mathbb{Q}[[t]]$。数字在 $p$-进世界里从右往左展开，$p$ 的幂决定「大小」——距离越近的是 $p$-整除越多者，与实数直观正好相反。</span>

完备化的威力在两条：一是把复杂问题**线性化/截断化**——$A/\mathfrak{a}^{n+1}$ 只看前 $n$ 阶信息，$n \to \infty$ 补齐；二是让方程在有「近似解」时能精确求解，这就是 **Hensel 引理**，多项式版的牛顿迭代。分析、数论、几何在这一篇正式合流。

## 1 逆极限：从截断到完备

完备化的标准语言是**逆极限**。对降链

$$A/\mathfrak{a} \longleftarrow A/\mathfrak{a}^2 \longleftarrow A/\mathfrak{a}^3 \longleftarrow \cdots$$

**逆极限（inverse limit）**：

$$\widehat{A} = \varprojlim A/\mathfrak{a}^n = \Big\{(x_n) \in \prod_n A/\mathfrak{a}^n \;\Big|\; x_{n+1} \bmod \mathfrak{a}^n = x_n\Big\}.$$

即「一串互相一致的截断」。自然的对角映射 $A \to \widehat{A}$ 给出 $\mathfrak{a}$-进完备化。

标准例子：
- $A = \mathbb{Z}$、$\mathfrak{a} = (p)$：$\widehat{\mathbb{Z}} = \mathbb{Z}_p$，$p$-进整数环。
- $A = k[x]$、$\mathfrak{a} = (x)$：$\widehat{A} = k[[x]]$，幂级数环——$k[x]$ 的「Taylor 展开视角」。
- $A = k[x_1,\dots,x_n]$、$\mathfrak{a} = (x_1,\dots,x_n)$：形式幂级数环 $k[[x_1,\dots,x_n]]$。

**核心对照表：几种 $\mathfrak{a}$-进完备化**

| 环 $A$ | 理想 $\mathfrak{a}$ | 完备化 $\widehat{A}$ |
| --- | --- | --- |
| $\mathbb{Z}$ | $(p)$ | $\mathbb{Z}_p$ |
| $k[x]$ | $(x)$ | $k[[x]]$ |
| $k[x_1,\dots,x_n]$ | $(x_1,\dots,x_n)$ | $k[[x_1,\dots,x_n]]$ |
| $k[x,y]/(y^2-x^3)$ | $(x,y)$ | 尖端的形式完备化 |

用 $\mathbb{Z}_p$ 体会「相容系」：一个元素是一串 $x_n \bmod p^n$，且 $x_{n+1} \equiv x_n \pmod{p^n}$。取 $x = (1, 1+p, 1+p+p^2, \dots)$，它正是 $1/(1-p)$ 的 $p$-进展开——逐阶逼近一个极限，恰如「从右往左写数字」。

**重点：完备化把所有「余-有限阶信息」一网打尽。** $A/\mathfrak{a}^n$ 是「只看前 $n$ 阶」的近似，逆极限把近似串成精确对象。<span class="marginnote">完备化与「连续函数空间 $C(X)$」的思想同源：把点态极限补进来。交换代数把「极限」做成「相容系」，从而不需要拓扑语言——严格说 $\widehat{A}$ 上确实带有 $\mathfrak{a}$-进拓扑。</span>

**辨析｜易错点：** $A \to \widehat{A}$ 不必是单射，核是 $\bigcap_n \mathfrak{a}^n$。Noether 局部环里 Krull 交定理保证 $\bigcap_n \mathfrak{m}^n = 0$（真理想情形），此时映射单射。看到「完备化」先问「核是不是零」——在 $A = k[x, 1/x]$ 之类非局部环上会有微妙差别。

## 2 完备环与 $\mathbb{Z}_p$ 的结构

称 $A$ 在 $\mathfrak{a}$ 处**完备**，若 $A \cong \widehat{A}$。完备局部环 $(R, \mathfrak{m})$ 有漂亮的结构：

**重点：完备局部环 $(R, \mathfrak{m}, k)$ 是「从剩余域 $k$ 长出来的环」。** 关键事实：
- **Hensel 引理**成立（见下节）；
- $R$ 的每个元素可写成「从剩余域逐阶展开」的级数；
- **Cohen 结构定理**：若 $R$ 含域 $k$，则 $R$ 是 $k$ 上幂级数环的商（带有限多个参数与关系）。

**辨析｜完备化 vs 局部化：** 两者都从 $\mathbb{Z}$ 出发给出新环——$\mathbb{Z}_{(p)}$（局部化）与 $\mathbb{Z}_p$（完备化），容易混。局部化「让某些元素可逆」，完备化「把极限补进来」；$\mathbb{Z}_{(p)}$ 一维但不完备，$\mathbb{Z}_p$ 既完备又是 DVR（第1篇《离散赋值环》）。几何上局部化看「点附近」，完备化看「点处的无穷小邻域」——后者比前者更细，因此很多性质（如 Hensel 引理）只在完备侧成立。

$\mathbb{Z}_p$ 的算术由此变得透明：每个 $x \in \mathbb{Z}_p$ 唯一写成

$$x = a_0 + a_1 p + a_2 p^2 + \cdots, \qquad a_i \in \{0, 1, \dots, p-1\}$$

「$p$-进展开」。$x$ 可逆 ⇔ $a_0 \neq 0$；不可逆元恰是 $p\mathbb{Z}_p$——$\mathbb{Z}_p$ 是 DVR（局部一维主理想整环），极大理想 $(p)$，剩余域 $\mathbb{F}_p$。**$\mathbb{Z}_p$ 与 $k[[t]]$ 是「同一类环」在算术与几何里的两个化身。**<span class="marginnote">这个类比是数论几何化的钥匙：$\operatorname{Spec} \mathbb{Z}_p$ 像 $\operatorname{Spec} k[[t]]$ 一样「只有一个点加一个一般点」，但后者是 $t=0$ 处的一条曲线局部，前者是素数 $p$ 处的「环的局部」——算术点与几何点被同一种语言接管。</span>

## 3 Hensel 引理：近似解提升为精确解

完备化的第一个丰收是 Hensel 引理，它在分析里的原型是牛顿法的收敛性。

**Hensel 引理**：设 $(R, \mathfrak{m})$ 完备、$k = R/\mathfrak{m}$，$f \in R[x]$。若 $f$ 在 $k$ 上有一个简单根 $\bar{a}$（即 $f(\bar{a}) = 0$、$f'(\bar{a}) \neq 0$ 于 $k$），则存在唯一的 $a \in R$，使

$$f(a) = 0, \qquad a \equiv \bar{a} \pmod{\mathfrak{m}}.$$

**重点：「模 $\mathfrak{m}$ 的简单根能唯一提升到完备环里的真根」。** 证明就是牛顿迭代：从 $a_0$（$\bar{a}$ 的任一提升）出发，令 $a_{n+1} = a_n - f(a_n)/f'(a_n)$，用完备性取极限 $a = \lim a_n$。<span class="marginnote">每步迭代把误差从「$\mathfrak{m}^n$ 量级」压到「$\mathfrak{m}^{n+1}$ 量级」，平方收敛——与实数牛顿法完全平行，只是「绝对值小」被「$p$-进靠近」替代。</span>

经典例子：$x^2 \equiv -1 \pmod 5$ 有根 $\bar{a} = 2$（$2^2 = 4 \equiv -1$），且 $2x = 4 \not\equiv 0 \pmod 5$，所以 $x^2 + 1$ 在 $\mathbb{Z}_5$ 中有真根——$i$ 的「$5$-进版本」存在！<span class="marginnote">这个例子震惊了许多人：$x^2+1=0$ 在 $\mathbb{R}$ 中无解，在 $\mathbb{Q}_5$ 中却有解。代数数的「存在性」依赖你所处的完备化——这正是「局部域」观念的核心。</span>

**辨析｜易错点：** 简单根条件（$f'(\bar{a}) \neq 0$）不可省。$x^2 \equiv 0 \pmod{p}$ 在 $p \neq 0$ 时根 $\bar{a} = 0$ 是**多重**根，提升存在（$a = 0$）但**不唯一**——多重根的 Hensel 需要更精细的「分解为互素因子」版本（Hensel 引理的全形式）。

## 4 公式解析：$\mathbb{Z}_p$ 中解的构造

用 Hensel 引理做一个完整的算例：在 $\mathbb{Z}_5$ 中求 $x^2 = -1$ 的根。设 $a = \lim a_n$，$a_0 = 2$。

- **第一步，验证近似解**：$a_0 = 2$，$f(a_0) = 5$，误差恰为 $5$ 量级；$f'(a_0) = 4$，模 $5$ 非零——简单根条件成立，可以开始迭代。
- **第二步，牛顿迭代**：$a_{n+1} = a_n - (a_n^2 + 1)/(2a_n)$，在 $\mathbb{Z}_5$ 中逐阶计算：
  $a_1 = 2 - 5/4$：$4^{-1} \equiv 4 \pmod 5$（$4\cdot 4 \equiv 1$），修正量 $5\cdot 4 = 20 \equiv 0 \pmod{5^2}$？需小心逐阶……标准结果 $a_1 \equiv 7 \pmod{25}$（$7^2 = 49 \equiv -1 \pmod{25}$）。
- **第三步，取极限**：完备性保证序列 $\{a_n\}$ 在 $5$-进意义下收敛到 $a \in \mathbb{Z}_5$，$f(a) = 0$。逐阶近似 $\{2, 7, 57, \dots\}$（$a_n \bmod 5^{n+1}$）正是 $a$ 的 $5$-进展开系数——**Hensel 引理 = 完备环上的牛顿法**。

把前几步迭代数据列成表，平方收敛看得一清二楚：

| $n$ | $a_n$ | $a_n^2 + 1$ 的误差阶 |
| --- | --- | --- |
| 0 | $2$ | $5$，模 $5$ 为零 |
| 1 | $7$ | $50$，模 $25$ 为零 |
| 2 | $57$ | $3250$，模 $125$ 为零 |
| 3 | $182$ | $33125$，模 $625$ 为零 |

每走一步，误差从「$5^n$ 量级」压到「$5^{n+1}$ 量级」——误差数量级每次翻倍，正是牛顿法的平方收敛。$182^2 + 1 = 33125 = 53 \cdot 625$，四阶以后 $x^2 \equiv -1$ 在 $\mathbb{Z}_5$ 里已经「看不见误差」。

**辨析｜易错点：** 「模 $\mathfrak{m}$ 有根」与「$R$ 里有根」在非完备环上差得很远。$\mathbb{Z}$ 中 $x^2 \equiv -1 \pmod{5^n}$ 对每个 $n$ 有解（$2, 7, 57, \dots$ 逐阶可解），但 $x^2 = -1$ 在 $\mathbb{Z}$ 中无解——**「近似解逐阶存在」只保证「完备化里有解」，不保证「原环里有解」**。这正是「局部可解 ≠ 全局可解」的算术版。

**术语速查表**

| 术语 | 一句话含义 |
| --- | --- |
| 逆极限 $\varprojlim$ | 一串相容截断的极限 |
| $\mathfrak{a}$-进完备化 | $\varprojlim A/\mathfrak{a}^n$ |
| $p$-进整数环 $\mathbb{Z}_p$ | $\varprojlim \mathbb{Z}/p^n$，DVR |
| 完备环 | $A \cong \widehat{A}$ |
| Hensel 引理 | 模 $\mathfrak{m}$ 的简单根唯一提升为真根 |
| Cohen 结构定理 | 含域完备局部环是幂级数环的商 |

## 5 小结

- **逆极限** $\widehat{A} = \varprojlim A/\mathfrak{a}^n$ 定义完备化；$\mathbb{Z}_p = \varprojlim \mathbb{Z}/p^n$、$k[[x]] = \varprojlim k[x]/(x^n)$。
- 完备局部环是「从剩余域长出来」的环，元素有级数展开；$\mathbb{Z}_p$ 与 $k[[t]]$ 同一族。
- **Hensel 引理**：模 $\mathfrak{m}$ 的简单根唯一提升为完备环真根；证明即牛顿迭代。
- 近似解逐阶存在 ⇒ 完备化有解，但原环不一定有解。

在下一节，我们攀上维数的高地：定义 **Krull 维数**、用 Hilbert 函数度量「体积」，看维数如何成为交换代数与代数几何之间最本质的桥梁。
