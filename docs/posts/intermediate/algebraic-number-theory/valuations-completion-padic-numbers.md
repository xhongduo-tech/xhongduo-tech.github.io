---
title: 赋值、完备化与 p-adic 数
date: 2026-08-11
---

# 赋值、完备化与 p-adic 数

<div class="epigraph">
<p>我们必须知道，我们必将知道。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert，Wir müssen wissen, wir werden wissen）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 p-adic 数开始

有理数 $\mathbb{Q}$ 有**很多种**度量，不只平常那种。对素数 $p$，可以定义一种「$p$ 进度量」：两个数相差「越多 $p$ 的因子」就**越接近**。比如在 $3$-进度量下，$1$ 与 $1 + 3^5$ 非常接近（差被 $3^5$ 整除）。按这个度量把 $\mathbb{Q}$ 完备化（补齐柯西列），就得到 $p$-**adic 数域 $\mathbb{Q}_p$**。<span class="marginnote">把「收敛」的观念从实数搬到 $p$-adic 世界，初看诡异，却威力无穷：很多在实数域里无解、无根的方程，在 $\mathbb{Q}_p$ 里反而可解——局部信息拼装全局结论（Hasse 局部—整体原则）由此成为数论的日常工具。</span>本节讲清楚赋值、完备化与 $\mathbb{Q}_p$ 的结构，为后面类域论的分歧理论铺路。

## 1 赋值与 Ostrowski 定理

**绝对值（absolute value）** 是 $\mathbb{Q} \to \mathbb{R}_{\ge 0}$ 满足三条公理的函数（正定性、乘法性、三角不等式）。$\mathbb{Q}$ 上的绝对值分三类：

1. **平凡**：$|x| = 1$（$x \ne 0$）；
2. **通常绝对值 $|\cdot|_\infty$**；
3. **$p$-adic 绝对值 $|\cdot|_p$**：设 $x = p^k \cdot \frac{m}{n}$（$m, n$ 不被 $p$ 整除，$k \in \mathbb{Z}$），则
$$|x|_p = p^{-k}, \qquad |0|_p = 0.$$

**Ostrowski 定理**：$\mathbb{Q}$ 上一切非平凡绝对值，要么等价于 $|\cdot|_\infty$，要么等价于某个 $|\cdot|_p$。<span class="marginnote">这是「数域的完整度量图景」的起点：$\mathbb{Q}$ 被这些绝对值「安放」在无穷多个局部对象里——一个来自通常度量，其余每个素数一个。把这套想法推广到任意数域，就是 Neukirch 第四章「局部—全局」的骨架。</span>

**关键换算**：$|p|_p = \frac{1}{p}$，$|1|_p = 1$，$|p^k|_p = p^{-k}$。注意 $p$ 的幂次越大，$p$-adic 绝对值越小——与「整除」的概念完美互补。

## 2 非阿基米德与超度不等

$p$-adic 绝对值满足比三角不等式强得多的性质：

$$
|x + y|_p \le \max(|x|_p, |y|_p), \qquad \text{且 } |x|_p \ne |y|_p \implies |x + y|_p = \max(|x|_p, |y|_p)
$$

这条**超度不等式（ultrametric inequality）** 带来两个反直觉结论：**任何三角形都是等腰的**；**球内任意一点都是球心**（所有点要么距离相等、要么其中一个在另一个的球内）。<span class="marginnote">拓扑学里的直觉在此全部失效：$\mathbb{Q}_p$ 是<strong>完全不连通</strong>的（每个点的开球都同时是闭的），却又是紧集的逆极限。这种「碎成一个个球」的结构，正是分歧理论里惯性群、剩余类域发挥作用的舞台。</span>

**辨析｜易错点：** 别把 $|\cdot|_p$ 当「另一个指数距离」。$|2|_2 = \frac12$、$|4|_2 = \frac14$，**$2$ 比 $4$ 更接近 $1$**；而 $|\frac12|_2 = 2$ 可以很大。所以「$p$-adic 小的数」=「$p$ 幂次整除得多的数」，与通常意义恰好相反。序列 $p^n$ 在实数里爆炸，在 $\mathbb{Q}_p$ 里却收敛到 $0$。

## 3 完备化：$\mathbb{Q}_p$ 与 $\mathbb{Z}_p$

把 $\mathbb{Q}$ 关于 $|\cdot|_p$ 的柯西列取模去零序列，得到**完备化** $\mathbb{Q}_p$。它的结构非常清晰：

$$
\mathbb{Q}_p = \left\{ x = \sum_{k = m}^{\infty} a_k p^k \;:\; m \in \mathbb{Z},\; a_k \in \{0, 1, \dots, p-1\} \right\}
$$

从某个（可能负的）指数开始的 $p$ 幂级数。<span class="marginnote">这正好对应实数的小数展开——只不过「小数点」位置相反：实数往 $10^{-1}, 10^{-2}, \dots$ 无限延伸（小数部分），$\mathbb{Q}_p$ 往 $p^1, p^2, \dots$ 无限延伸（整数部分）。实数 $10$ 进制下 $\frac13 = 0.333\ldots$ 无限循环，$3$-adic 下 $\frac13 = 1\cdot 3^{-1}$（$m = -1$，一位写完）——两者的「无限」恰好跑到相反方向。错位相消的典型是 $-1 = 2 + 2\cdot 3 + 2\cdot 3^2 + \cdots$（每位都是 $p-1 = 2$：$2(1+3+9+\cdots) = \frac{2}{1-3} = -1$）。</span>

三个核心子对象：

- **整数环 $\mathbb{Z}_p$**：$m \ge 0$ 的部分（$p$-adic 绝对值 $\le 1$ 的元素），是局部环；
- **极大理想 $\mathfrak{p} = p\mathbb{Z}_p$**：$a_0 = 0$ 的元素（$|x|_p < 1$）；
- **剩余类域 $\mathbb{Z}_p / p\mathbb{Z}_p \cong \mathbb{F}_p$**。

并且 $\mathbb{Z}_p$ 可看成有限环的**逆极限**：

$$
\mathbb{Z}_p = \varprojlim_n \mathbb{Z}/p^n\mathbb{Z}
$$

**辨析｜易错点：** $\mathbb{Q}_p$ 的**特征仍是 $0$**（它含 $\mathbb{Q}$），但**剩余类域特征**是 $p$。这个「特征 0 的域配特征 $p$ 的剩余类域」的组合，是后面分歧理论里「野分歧」现象的根源。也别把 $\mathbb{Z}_p$ 误当成「$p$ 元域 $\mathbb{Z}/p\mathbb{Z}$」——那是 $\mathbb{F}_p$，是商环不是子环。

**局部紧与调和分析**：$\mathbb{Z}_p$ 是紧群、$\mathbb{Q}_p$ 是局部紧阿贝尔群——Pontryagin 对偶 $\widehat{\mathbb{Q}_p} \cong \mathbb{Q}_p$ 使 $p$-adic 上的调和分析成为可能，这是 Tate 学位论文（1937）与 Iwasawa 理论的地基，也是「$p$-adic 分析」这门分支的入口。

## 4 公式解析：$p$-adic 展开

$$
x = \sum_{k=m}^{\infty} a_k p^k, \qquad m \le 0 \text{ 或 } m > 0, \quad a_k \in \{0,\dots,p-1\}, \quad a_m \ne 0
$$

- **第一步，理解 $m$**：$m$ 是 $x$ 的**$p$-adic 估值**（$v_p(x)$）。$m < 0$ 表示 $x$ 有「$p$ 的负幂次」——即 $x$ 的 $p$-adic 绝对值大于 $1$（$x$ 在 $\mathbb{Q}_p \setminus \mathbb{Z}_p$）；$m \ge 0$ 则 $x \in \mathbb{Z}_p$。
- **第二步，理解系数 $a_k$**：$a_k$ 是「余数递归」的产物：$\frac{1}{p}\big(x - a_m\big)$ 的整数部分逐层剥出。用 $a_0$（若 $m \le 0$）或 $a_m$ 判估值，像十进制展开但方向相反。
- **第三步，为什么收敛**：部分和 $S_N$ 与 $x$ 之差被 $p^{N+1}$ 整除，$|x - S_N|_p \le p^{-(N+1)} \to 0$。**级数在这个度量下自然收敛**——这验证了「$p$-adic 分析」的有效性：幂级数、微积分都可以在 $\mathbb{Q}_p$ 上重建（Hensel 引理是它的第一朵花）。（顺带：$m < 0$ 正是「$x \in \mathbb{Q}_p$ 但 $x \notin \mathbb{Z}_p$」的情形，如 $\frac1p$ 的展开 $m = -1$、$a_{-1} = 1$。）

## 5 Hensel 引理与局部—整体

**Hensel 引理**：设 $f \in \mathbb{Z}_p[x]$，若存在 $a_0$ 使 $f(a_0) \equiv 0 \pmod p$ 且 $f'(a_0) \not\equiv 0 \pmod p$，则存在 $a \in \mathbb{Z}_p$ 使 $f(a) = 0$ 且 $a \equiv a_0 \pmod p$。

即「模 $p$ 有简单根」就能升到 $\mathbb{Q}_p$ 里的真根——牛顿法的算术版。<span class="marginnote">例：$x^2 \equiv -1 \pmod 5$ 有根 $x \equiv 2$（$4 \equiv -1$），$f'(2) = 4 \not\equiv 0$，故 $x^2 = -1$ 在 $\mathbb{Q}_5$ 里有真根——<strong>模 $p$ 的代数，比模 $p^k$ 的一层层逼近更接近「完整真相」</strong>。</span> Hensel 引理把「局部可解」变成「$p$-adic 可解」，再经 Hasse 原则拼成全局结论（如 $ax^2 + by^2 = c$ 的整性判定），是类域论「局部决定全局」哲学的第一章。**一个标准应用**：$x^2 = a$ 在 $\mathbb{Q}_p$ 可解的判据是——$p$ 奇时只需模 $p$ 可解，$p = 2$ 时需模 $8$ 的精细条件（Hensel + 二次互反律联合给出）。

**$p$-adic 的树直觉**：$\mathbb{Z}_p$ 是「$p$ 元树」的逆极限——每个节点有 $p$ 个子节点（$a_k$ 的 $p$ 种选择），$x \equiv \sum_{k=0}^{n} a_k p^k \pmod{p^{n+1}}$ 是深度 $n$ 处的节点，$x$ 的展开就是沿树的一条无限路径。

## 6 扩展：数域上的赋值

**规范化赋值**：对 $\mathbb{Q}_p$ 常用 $v_p(x) = -\log_p |x|_p$，即 $v_p(p^k) = k$。它满足

$$
v_p(xy) = v_p(x) + v_p(y), \qquad v_p(x + y) \ge \min(v_p(x), v_p(y))
$$

「乘法变加法、加法变取小」——这是**离散赋值环**的公理，把 $p$-adic 结构抽象出来。

**赋值扩张到数域**：设 $K$ 是数域，$\mathfrak{p}$ 是素理想，$v_{\mathfrak{p}}(\alpha)$ 定义为「$\alpha$ 的 $\mathfrak{p}$-adic 展开中 $\mathfrak{p}$ 的最高幂」。它给出 $K$ 上的绝对值

$$
|\alpha|_{\mathfrak{p}} = \mathrm{N}(\mathfrak{p})^{-v_{\mathfrak{p}}(\alpha)}
$$

**Ostrowski 的全数域版**：$K$ 的一切非平凡赋值，要么来自某个素理想（非阿基米德），要么来自实/复嵌入（阿基米德）。于是「$K$ 的素数」= 素理想 $\cup$ 嵌入——这就是 idèle 论和类域论的输入数据，也是下一节分歧理论的发生地。

**例**：$K = \mathbb{Q}_5(\sqrt{2})$。因为 $2$ 不是模 $5$ 的平方（平方类 $\{1,4\}$），$\sqrt2 \notin \mathbb{Q}_5$，故 $K/\mathbb{Q}_5$ 是次数 $2$ 的**无分歧**扩张，$v_K(\sqrt2) = 0$（$\sqrt2$ 是单位），剩余类域 $\mathbb{F}_5 \to \mathbb{F}_{25}$ 扩张（$f = 2$，$e = 1$）。<span class="marginnote">这个「$v_K$ 是 $v_5$ 的延拓」的语言，正是分歧理论（分解群、惯性群）的局部骨架——每个 $\mathfrak{P} \mid \mathfrak{p}$ 对应一个「$v_{\mathfrak{p}}$ 的延拓到 $L$ 的方式」。</span>

**辨析｜易错点：** $v_p(x) = k$ 与 $|x|_p = p^{-k}$ 是同一信息两种写法：$v_p$ 越大「越被整除」、$|x|_p$ 越小。计算时先想 $v_p$（整数算术）再换算绝对值。另外别把「$v_p$」与「$p$-adic 展开的起始指数 $m$」当作两个概念——它们相等，只是一个数论量、一个级数记号。

## 7 小结

- **$p$-adic 绝对值** $|x|_p = p^{-v_p(x)}$：被 $p$ 整除越多越小；Ostrowski 定理说 $\mathbb{Q}$ 的度量只有通常与 $p$-adic 两类。
- **超度不等式**：$|x+y|_p \le \max(|x|_p, |y|_p)$，一切三角形等腰、球内皆球心，$\mathbb{Q}_p$ 完全不连通。
- **$\mathbb{Q}_p$ = $p$ 幂级数**；$\mathbb{Z}_p$ 是局部环、逆极限 $\varprojlim \mathbb{Z}/p^n\mathbb{Z}$，剩余类域 $\mathbb{F}_p$。
- **Hensel 引理**：模 $p$ 简单根升为 $\mathbb{Q}_p$ 真根——局部—整体思想的第一朵花。
- $\mathbb{Z}_p$ 紧、$\mathbb{Q}_p$ 局部紧阿贝尔：Pontryagin 对偶使 $p$-adic 调和分析（Tate thesis、Iwasawa 理论）成为可能。

在下一节，我们将回到数域扩张 $L/K$，研究有理素数在 $K$ 里如何「分裂、惯性、分歧」——**分歧理论**以分解群、惯性群与高阶分歧群为骨架，把素理想分解的局部信息系统化。
