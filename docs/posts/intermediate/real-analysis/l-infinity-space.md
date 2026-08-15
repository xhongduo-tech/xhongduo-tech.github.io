---
title: L∞ 空间与本性有界函数
date: 2026-08-07
---

# L∞ 空间与本性有界函数

<div class="epigraph">
<p>$L^\infty$ 是 $L^p$ 谱系的极限站——当 $p$ 趋于无穷，可积性退化为「本性有界」。</p>
<footer>—— 斯坦尼斯瓦夫 · 萨克斯（Stanisław Saks）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第七章 ｜ 2026-08-07</p>
</div>

## 为什么从 L∞ 空间开始

$L^p$ 家族的最后一位成员是 **$L^\infty$**——当 $p\to\infty$，$p$-范数 $\left(\int|f|^p\right)^{1/p}$ 的极限退化为「本性上确界」：$f$ 在「除去零测集外」的最大值。$L^\infty$ 是**本性有界函数**的空间，它是 $L^p$ 谱系的对偶端点（$L^1$ 的对偶是 $L^\infty$），也是控制收敛、算子范数、测度论对偶理论中的关键角色。

理解 $L^\infty$ 的核心是「**本性**」二字：$L^\infty$ 不关心单点上的值（可以在零测集上任意大），只关心「a.e. 有界」且「a.e. 上界的最小者」。它是「本质有界」而非「处处有界」——这是 Lebesgue 理论的彻底贯彻。<span class="marginnote">$L^\infty$ 是 $L^1$ 的<strong>对偶空间</strong>（在 $\sigma$-有限测度下）：$L^1$ 上的每个有界线性泛函都是「对某个 $g\in L^\infty$ 的积分」。这把「积分算子」与「$L^\infty$ 元素」一一对应——<strong>Riesz 表示定理</strong>是泛函分析的开山之作，$L^\infty$ 是它的主角之一。</span>

## 1 本性上确界与 L∞ 的定义

**定义（本性上确界）**：$f$ 可测，定义

$$\operatorname*{ess\,sup}|f|=\inf\{M:|f(x)|\le M\ \text{a.e.}\}$$

即「使 $|f|\le M$ a.e. 成立的最小 $M$」。

**定义（$L^\infty$）**：$L^\infty(E)=\{f\ \text{可测}:\operatorname*{ess\,sup}|f|<\infty\}$，配范数

$$\|f\|_\infty=\operatorname*{ess\,sup}|f|$$

**例**：$f=\chi_{\mathbb{Q}\cap[0,1]}$：$\|f\|_\infty=1$（虽然它在有理点上取 1，在无理点上取 0，但「本性」上界是 1？——不，$f$ 在 a.e. 点为 0，$\operatorname*{ess\,sup}|f|=0$！因为 $|f|\le0$ 在零测例外处成立）。**$\chi_\mathbb{Q}$ 的 $L^\infty$ 范数是 0**——它 a.e. 等于零函数。

**重点：「本性」意味着单点/零测集上的值被忽略。** $\chi_\mathbb{Q}$ 处处取值 0/1，但它的 $L^\infty$ 范数是 0——因为「$|f|\le M$ a.e.」只要求零测例外。**$L^\infty$ 的元素是 a.e. 等价类，本性上确界是「等价类的最紧上界」。**

## 2 L∞ 的基本性质

**性质一（范数三公理）**：$\|\cdot\|_\infty$ 是范数——正定（$\|f\|_\infty=0\iff f=0$ a.e.）、齐次、三角不等式（$\operatorname*{ess\,sup}|f+g|\le\operatorname*{ess\,sup}|f|+\operatorname*{ess\,sup}|g|$，由逐点三角不等式 + 本性处理）。

**性质二（完备）**：$L^\infty$ 是 Banach 空间（Cauchy 列一致有界 a.e.，极限仍本性有界）。

**性质三（乘法代数）**：$f,g\in L^\infty$ ⇒ $fg\in L^\infty$，$\|fg\|_\infty\le\|f\|_\infty\|g\|_\infty$——**$L^\infty$ 是 Banach 代数**（有单位元 1）。

**性质四（与 $L^p$ 的关系）**：$f\in L^\infty$ 且 $m(E)<\infty$ ⇒ $f\in L^p$ 对一切 $p$（$\int|f|^p\le\|f\|_\infty^p m(E)$）。**$L^\infty\subset\bigcap_{p<\infty}L^p$（有限测度上）**——本性有界是最强的可积性。

**辨析｜易错点：$\|f\|_\infty$ 不是 $\sup|f|$。** 对处处取大值的点，$\sup$ 与 $\operatorname*{ess\,sup}$ 不同：$f=\chi_{\{0\}}$（单点 1），$\sup|f|=1$ 但 $\operatorname*{ess\,sup}|f|=0$（单点零测）。**「本性上确界」忽略零测集，是 $L^\infty$ 与「有界函数空间」$B$ 的区别**——$L^\infty$ 更宽容（等价类）。

## 3 作为 p→∞ 的极限

**定理（$L^p$ 与 $L^\infty$ 的极限关系）**：设 $f\in L^p$ 对一切 $p$（或 $f$ 本性有界）。则

$$\|f\|_\infty=\lim_{p\to\infty}\|f\|_p$$

**证明（有界情形）**：$|f|\le M=\|f\|_\infty$ a.e.，则 $\|f\|_p\le Mm(E)^{1/p}\to M$（有限测度）。反方向：$\|f\|_p\ge M(1-\varepsilon)m(\{|f|>M(1-\varepsilon)\})^{1/p}\to M(1-\varepsilon)$（正测度集贡献）。两边夹得 $\|f\|_p\to M$。<span class="marginnote">极限关系揭示 $L^\infty$ 是 $L^p$ 谱系的「几何极限」：<strong>$p$-范数把「大值区域的占比」逐步放大，$p\to\infty$ 时只剩「最大值」</strong>。这个视角让「$L^\infty$ 是 $p=\infty$ 的 $L^p$」不只是记号，而是真实的极限过程。</span>

**推论（谱系完整性）**：$1\le p\le\infty$ 的 $L^p$ 谱系，端点 $L^1$（绝对可积）与 $L^\infty$（本性有界），中间 $L^p$ 平滑过渡。**「$L^p$ 的完整谱系 = 从可积到有界」**。

## 4 公式解析：$\|f\|_\infty$ 的 inf 定义

本性上确界的「$\inf$」定义是理解它的钥匙：

$$\|f\|_\infty=\operatorname*{ess\,sup}|f|=\inf\{M:|f|\le M\ \text{a.e.}\}$$

- **第一步，读「$|f|\le M$ a.e.」**：$M$ 是「本性上界」——要求「除零测集外 $|f|\le M$」。**「a.e.」是关键字**：单点超 $M$ 不影响资格。
- **第二步，读「$\inf$ 的意义」**：所有本性上界的**下确界**就是「最小的本性上界」。**「本性上确界 = 本性上界的下确界」**——上确界与下确界在本性意义下合流（类似 $\sup$ = 上界的最小者）。
- **第三步，读「$\inf$ 是否被取到」**：$\{M:|f|\le M\ \text{a.e.}\}$ 是 $[0,\infty]$ 的子集，下确界存在；且这个 $\inf$ 本身也是本性上界（$\{|f|>\|f\|_\infty\}$ 是零测集之并可零测）。**「本性上确界被取到」**——$L^\infty$ 范数是可达的，这点比 $\sup$ 更「紧」。

**「a.e. 上界的下确界」**，是 $L^\infty$ 范数的完整语义——它精确地是「等价类的最紧 a.e. 上界」。

## 5 例子与常见陷阱

**例子一（$\chi_\mathbb{Q}$ 的范数）**：$f=\chi_{\mathbb{Q}\cap[0,1]}$。$\sup|f|=1$（处处有界函数空间的范数），但 $\|f\|_\infty=\operatorname*{ess\,sup}|f|=0$——因为 $|f|\le0$ 在 $[0,1]\setminus\mathbb{Q}$（余集零测）上成立。**同一个函数，在「有界函数空间 $B$」范数是 1，在「$L^\infty$」范数是 0**——本质差别是「是否忽略零测集」。

**例子二（$L^\infty$ 不是 $B$ 的等价类商）**：$L^\infty$ 的元素是「本性有界函数的 a.e. 等价类」。$f=\chi_\mathbb{Q}$ 与 $g\equiv0$ 是同一个 $L^\infty$ 元素（$\|f-g\|_\infty=0$），但它们处处不同。**「本性」的代价是放弃逐点身份**——这正是 Lebesgue 理论「函数 = 等价类」的最终贯彻。

**例子三（$\|f\|_p\to\|f\|_\infty$ 的直观）**：$f(x)=x$ 在 $[0,1]$ 上，$\|f\|_p=\left(\tfrac1{p+1}\right)^{1/p}\to1=\|f\|_\infty$。$p$ 越大，$p$-范数越被「靠近最大值」的区域主导——**$p$-范数在「平均」与「最大」之间滑动，$p=\infty$ 停在最大**。

**辨析｜易错点：$\|f\|_\infty$ 有限 ≠ $f$ 有界。** $f=\chi_{\mathbb{Q}\cap[0,1]}$ 本性有界但处处无界（在无理点取 0，有理点取 1，每个点都有界——不，它处处有界取 0/1；更好的反例：$f(x)=\tfrac1x$ 在 $(0,1]$ 上，$\|f\|_\infty=\infty$ 且处处无界）。真正的差别例：$f$ 在零测集上取 $+\infty$，其余为 0——$\|f\|_\infty=0$ 但 $f$ 在零测集上「无界」。**「本性有界」允许零测集上的任意大值。**

## 7 数值演练与 $L^\infty$ 速查

**算例一（本性上确界的数值）**：$f=\chi_{\mathbb{Q}\cap[0,1]}$。$\sup|f|=1$，但 $\operatorname*{ess\,sup}|f|=0$——$|f|\le0$ 在 $[0,1]\setminus\mathbb{Q}$（余集零测）成立。**$\chi_\mathbb{Q}$ 是 $L^\infty$ 的零元素**（与 $0$ 是同一等价类）。

**算例二（$\|f\|_p\to\|f\|_\infty$ 的数值）**：$f(x)=x$ 于 $[0,1]$。$\|f\|_p=(\tfrac1{p+1})^{1/p}\to1=\|f\|_\infty$。$p=1$：$\tfrac12$；$p=2$：$\sqrt{\tfrac13}\approx0.577$；$p=10$：$(\tfrac1{11})^{1/10}\approx0.786$——**$p$ 越大，范数越被「靠近 1 的区域」主导**。

**对照表：$L^\infty$ 与有界函数空间 $B$**

| 空间 | 范数 | 元素 |
| --- | --- | --- |
| $B$（有界） | $\sup\|f\|$ | 处处有界函数 |
| $L^\infty$ | ess sup | 本性有界等价类 |
| $L^p$（$p<\infty$） | $(\int\|f\|^p)^{1/p}$ | $p$ 次可积 |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| $\operatorname*{ess\,sup}$ | 本性上确界 |
| Banach 代数 | 有乘法单位的完备赋范代数 |
| 对偶空间 | $(L^1)^*=L^\infty$ |
| 本性有界 | a.e. 有界 |

**辨析｜易错点：$\|f\|_\infty<\infty$ 不蕴含 $f$ 处处有界。** $f$ 在零测集上取 $+\infty$、其余为 $0$：$\|f\|_\infty=0$ 但 $f$ 在零测集上「无界」。**「本性」的宽容正是 $L^\infty$ 与 $B$ 的分界。**

### 三步计算 $L^\infty$ 范数

- **找上界**：$\{M:|f|\le M\ \text{a.e.}\}$。
- **取下确界**：最小的本性上界。
- **验证**：$\operatorname*{ess\,sup}|f|$ 本身是本性上界（零测集并可零测）。

**延伸（与泛函分析连接）**：$L^\infty$ 是 $L^1$ 的对偶——「每个有界线性泛函都是对某个 $g\in L^\infty$ 的积分」是 Riesz 表示定理。**控制收敛的「边界」、算子范数的「载体」都落在 $L^\infty$**，它是 $L^p$ 谱系的对偶端点。

**一道收束练习**：证明 $f\in L^\infty$、$m(E)<\infty$ 时 $f\in L^p$ 且 $\|f\|_p\le\|f\|_\infty m(E)^{1/p}$——$L^\infty\subset\bigcap_{p<\infty}L^p$（有限测度上）的定量版本。

## 8 小结

- **本性上确界**：$\operatorname*{ess\,sup}|f|=\inf\{M:|f|\le M\ \text{a.e.}\}$——忽略零测集的最紧上界。
- **$L^\infty$ 定义**：本性有界函数的等价类空间，范数 $\|f\|_\infty=\operatorname*{ess\,sup}|f|$。
- **性质**：Banach 空间、Banach 代数、有限测度上 $L^\infty\subset\bigcap L^p$。
- **$p\to\infty$**：$\|f\|_p\to\|f\|_\infty$——$L^\infty$ 是谱系的几何极限。
- **地位**：$L^1$ 的对偶、控制收敛的边界、算子范数的载体。
- **数值**：$\chi_\mathbb{Q}$ 的 $\|f\|_\infty=0$；$f=x$ 于 $[0,1]$ 时 $\|f\|_p\to1$。

至此，第八篇「Lᵖ 空间」与整个《实变函数与测度论》专题的全部条目写作完成。
