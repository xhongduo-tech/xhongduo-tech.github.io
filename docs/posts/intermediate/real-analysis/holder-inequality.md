---
title: Hölder 不等式
date: 2026-08-07
---

# Hölder 不等式

<div class="epigraph">
<p>乘积的积分，被「两个因子的 $L^p$ 范数」夹住——这是分析学最有用的一条不等式。</p>
<footer>—— 奥托 · 赫尔德（Otto Hölder）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第七章 ｜ 2026-08-07</p>
</div>

## 为什么从 Hölder 不等式开始

$L^p$ 空间要成为赋范空间，最关键的一步是证明 Minkowski（三角不等式）。而 Minkowski 的基石是 **Hölder 不等式**：它是「乘积积分」的万能控制——$\int|fg|\le\|f\|_p\|g\|_q$，其中 $p,q$ 是共轭指数（$\tfrac1p+\tfrac1q=1$）。这条不等式是分析学最频繁使用的工具：$L^p$ 空间的配对、内积、卷积不等式、概率论中的矩不等式，全都要靠它。

Hölder 不等式的美丽在于「$p,q$ 共轭」的对称性：$f$ 的 $p$ 次可积与 $g$ 的 $q$ 次可积，乘积自动可积。$p=q=2$ 时它就是 **Cauchy–Schwarz 不等式**——Hölder 是它的推广。<span class="marginnote">Hölder 不等式在概率论中就是「$E|XY|\le(E|X|^p)^{1/p}(E|Y|^q)^{1/q}$」——<strong>矩的乘积控制</strong>。$p=q=2$ 是 $|E[XY]|\le\sqrt{E[X^2]E[Y^2]}$（Cauchy–Schwarz），是相关性、协方差、以及「$L^2$ 内积」理论的全部基础。</span>

## 1 共轭指数与 Young 不等式

**定义（共轭指数）**：$1<p,q<\infty$ 满足 $\tfrac1p+\tfrac1q=1$，即 $q=\tfrac{p}{p-1}$，称为**共轭指数**。$p=2$ 时 $q=2$（自共轭）；$p\to1$ 时 $q\to\infty$。

**预备引理（Young 不等式）**：对 $a,b\ge0$：

$$ab\le\frac{a^p}{p}+\frac{b^q}{q}$$

**证明**：由对数函数的凹性，$\ln(ab)=\ln a+\ln b=\tfrac1p\ln(a^p)+\tfrac1q\ln(b^q)\le\ln\left(\tfrac{a^p}{p}+\tfrac{b^q}{q}\right)$（Jensen/对数凹）。取指数即得。<span class="marginnote">Young 不等式的几何：$ab$ 是矩形面积，$\tfrac{a^p}p+\tfrac{b^q}q$ 是两条曲线 $y=x^{p-1}$ 与 $x=y^{q-1}$ 围成的面积之和——矩形总被包含在内。<strong>「乘积 ≤ 幂的平均」</strong>是 Hölder 的全部种子。</span>

**重点：Young 不等式把「乘积」分解为「幂」**——$ab$ 被两个「各论各的」项控制，权重 $1/p,1/q$ 之和为 1。这是 Hölder 证明的第一块砖。

**共轭指数速查表**：常用配对 $(p,q)$ 直接查，省去每次现算：

| $p$ | $q=\tfrac{p}{p-1}$ | 备注 |
| --- | --- | --- |
| $2$ | $2$ | 自共轭，即 Cauchy–Schwarz |
| $3$ | $\tfrac32$ | 处理 $x^{-1/4}$ 型奇异函数 |
| $4$ | $\tfrac43$ | 幂指数越小越「安全」 |
| $\tfrac32$ | $3$ | 与 $p=3$ 对称 |
| $p\to\infty$ | $q\to1^+$ | 退化到 $L^\infty$-$L^1$ 配对 |

**记忆法**：$p$ 增大则 $q$ 减小，$p=2$ 是唯一自共轭点——正因为自共轭，$L^2$ 才独占内积结构（见《$L^2$ 空间》）。

## 2 Hölder 不等式的陈述与证明

**定理（Hölder 不等式）**：设 $1\le p,q\le\infty$ 共轭（$\tfrac1p+\tfrac1q=1$），$f\in L^p$，$g\in L^q$。则 $fg\in L^1$，且

$$\left|\int_Efg\,dm\right|\le\int_E|fg|\,dm\le\|f\|_p\|g\|_q$$

**证明（$1<p,q<\infty$ 情形）**：若 $\|f\|_p=0$ 或 $\|g\|_q=0$，平凡（a.e. 零）。否则归一化：令 $\tilde f=f/\|f\|_p$，$\tilde g=g/\|g\|_q$，则 $\int|\tilde f|^p=1$、$\int|\tilde g|^q=1$。由 Young 不等式逐点：

$$|\tilde f(x)\tilde g(x)|\le\frac{|\tilde f(x)|^p}{p}+\frac{|\tilde g(x)|^q}{q}$$

积分（非负，线性）：

$$\int|\tilde f\tilde g|\le\frac1p\int|\tilde f|^p+\frac1q\int|\tilde g|^q=\frac1p+\frac1q=1$$

代回 $\tilde f,\tilde g$：$\int|fg|\le\|f\|_p\|g\|_q$。<span class="marginnote">证明的机关是「<strong>归一化 + Young 逐点 + 积分</strong>」：先把范数缩放成 1（让 Young 的右边恰好是 $1/p+1/q=1$），逐点用 Young，再积分。三个步骤环环相扣——<strong>归一化把「乘积 ≤ 幂平均」变成「积分 ≤ 1」</strong>。</span>

**等号条件**：$\int|fg|=\|f\|_p\|g\|_q$ 当且仅当 $|f|^p$ 与 $|g|^q$ 几乎处处成比例（Young 等号条件 + 积分等号）。

**边界情形（$p=1,q=\infty$）**：$\int|fg|\le\|f\|_1\|g\|_\infty$——由 $|f(x)g(x)|\le|f(x)|\,\|g\|_\infty$ 逐点成立再积分即得。<strong>Hölder 在最极端的共轭对上也成立</strong>，这正是 $L^\infty$ 与 $L^1$ 的对偶配对的依据（见《$L^\infty$ 空间与本性有界函数》）。

**备注（为何只看 $\int|fg|$）**：Hölder 的结论先给 $\int|fg|$，再推出 $\left|\int fg\right|\le\int|fg|$（三角不等式）。<strong>「先绝对值、后符号」的顺序保证取绝对值的时刻不丢失信息</strong>——这是 Lebesgue 积分「绝对可积」纪律在不等式层面的回声。

## 3 推论与应用

**推论一（Cauchy–Schwarz）**：$p=q=2$ 时，$\int|fg|\le\|f\|_2\|g\|_2$。

**推论二（有限测度上的 $L^q\subset L^p$）**：$m(E)<\infty$，$1\le p<q$，$f\in L^q$ ⇒ $f\in L^p$。证明：$|f|^p=|f|^p\cdot1$，用 Hölder（指数 $q/p$ 与 $\tfrac{q}{q-p}$）：$\int|f|^p\le\|f\|_p^{q}\,m(E)^{1-p/q}$。

**推论三（$L^p$ 内积配对）**：$f\in L^p,g\in L^q$ ⇒ $fg\in L^1$，映射 $(f,g)\mapsto\int fg$ 是 $L^p\times L^q\to\mathbb{R}$ 的连续双线性配对。**$L^p$ 与 $L^q$ 互为对偶（$p\ne\infty$ 时）**——这是泛函分析对偶理论的入口。

**推论四（多因子版本）**：若 $f\in L^p,g\in L^q,h\in L^r$ 且 $\tfrac1p+\tfrac1q+\tfrac1r=1$，则 $fgh\in L^1$——把 $fg$ 视作整体、指数 $\tfrac{pr}{p+r}$，或直接对推广版 Hölder 使用均可。<strong>Hölder 可以逐次套用</strong>，三线性估计在调和分析里因此成立。

**应用（卷积不等式）**：$\|f*g\|_r\le\|f\|_p\|g\|_q$（Young 卷积不等式），其中 $\tfrac1r=\tfrac1p+\tfrac1q-1$——调和分析的核心工具，靠 Hölder 证明。

**例（数值验证）**：$f(x)=x$，$g(x)=1$ 在 $[0,1]$ 上，$p=q=2$。$\|f\|_2=(\int_0^1x^2)^{1/2}=\tfrac1{\sqrt3}$，$\|g\|_2=1$，$\int fg=\int_0^1x=\tfrac12$。验证 $\tfrac12\le\tfrac1{\sqrt3}\approx0.577$——成立且严格。

**例（等号取到）**：取 $f=g=x$：$\int x^2=\tfrac13=\|f\|_2\|g\|_2$——等号成立，因为 $f$ 与 $g$ 支撑相同且强度成比例（等号条件生效）。两个例子对照：<strong>「$f=g$ 时 Hölder 取等」是直觉起点，而「不相干的 $f,g$ 只会严格小于」</strong>。

**辨析｜易错点：Hölder 需要「$p,q$ 共轭」，不是「任意 $p,q$」。** 若 $\tfrac1p+\tfrac1q\ne1$，不等式形式要调整（Young 卷积不等式那种更一般的版本）。**「共轭」是 Hölder 成立的前提**——$1/p+1/q=1$ 让「幂平均的权重和为 1」。

## 4 公式解析：归一化的 Young 积分

把 Hölder 证明的完整链条写出：

$$\int|\tilde f\tilde g|\ \overset{\text{Young}}{\le}\ \frac1p\int|\tilde f|^p+\frac1q\int|\tilde g|^q\ =\ \frac1p+\frac1q\ =\ 1\ \Longrightarrow\ \int|fg|\le\|f\|_p\|g\|_q$$

- **第一步，读「逐点 Young」**：$|\tilde f(x)\tilde g(x)|\le\tfrac{|\tilde f|^p}{p}+\tfrac{|\tilde g|^q}{q}$ 对每个 $x$——**Young 是逐点的**，积分后仍是「≤」（非负，线性保序）。
- **第二步，读「归一化的魔术」**：$\int|\tilde f|^p=1$、$\int|\tilde g|^q=1$，右边恰好 $=\tfrac1p+\tfrac1q=1$。**「范数归一为 1」让 Young 的积分值精确等于 1**——这就是为什么选择 $1/p+1/q=1$ 的共轭条件。
- **第三步，读「代回」**：$\int|fg|\le\|f\|_p\|g\|_q\int|\tilde f\tilde g|\le\|f\|_p\|g\|_q$。**缩放因子恰好是两范数之积**——不等式形状优美，等号条件由 Young 给出。

**「逐点 Young + 归一化 + 积分」**，是 Hölder 证明的标准三步——它是「乘积积分」的全部控制力所在。

## 5 例子：Hölder 的等号与典型应用

**例一（等号条件）**：$f(x)=x^{1/3}$，$g(x)=x^{1/4}$ 在 $[0,1]$ 上，取 $p=3$、$q=3/2$（$1/3+2/3=1$）。$|f|^3=x$ 与 $|g|^{3/2}=x^{1/2}\cdot$——不直接成比例，故 Hölder 严格不等。等号要求 $|f|^p=c|g|^q$ a.e.：例如 $f=x^{1/4}$，$g=x^{1/2}$，$p=4,q=4/3$，$|f|^4=x$，$|g|^{4/3}=x^{2/3}$——仍不成比例。真正取等的例子：$f=\chi_A$，$g=\chi_A$ 任意 $p,q$（$\int|fg|=\|f\|_p\|g\|_q=m(A)$）。**「两个函数的支撑相同且强度成比例」是等号的本质**。

**例二（有限测度上的 $L^q\subset L^p$ 的具体值）**：$m(E)=1$，$f=x^{-1/4}$ 在 $(0,1]$：$\|f\|_1=\int x^{-1/4}=\tfrac43$，$\|f\|_2=(\int x^{-1/2})^{1/2}=\sqrt2$，$\|f\|_4=(\int x^{-1})^{1/4}=\infty$。**$f\in L^1\cap L^2$ 但 $f\notin L^4$**——$p$ 越大要求越高，有限测度上的包含 $L^4\subset L^2\subset L^1$ 由此方向确认。

**例三（Cauchy–Schwarz 的直接应用）**：$f,g\in L^2$ 时 $fg\in L^1$——两个平方可积函数的乘积自动可积。这保证 $L^2$ 内积 $\langle f,g\rangle=\int fg$ 良定义，是 $L^2$ 空间全部几何（正交、投影、Fourier）的前提。**「Hölder 保证内积良定义」是它在 $L^2$ 理论中的第一贡献。**

**重点：Hölder 的实用性在于「把乘积的积分拆成两个范数」。** 遇到「$\int fg$ 或 $\int f^ag^b$ 型」的量，第一反应就是 Hölder：先看 $p,q$ 怎么配（让指数共轭），再分别估计范数。**「乘积积分 → 范数乘积」是分析不等式的基本动作**，Hölder 是它的总闸门。

**例四（与概率论对接）**：若 $X,Y$ 是随机变量，$E|XY|\le(E|X|^p)^{1/p}(E|Y|^q)^{1/q}$。取 $p=q=2$：$|\mathrm{Cov}(X,Y)|\le\sqrt{E[X^2]E[Y^2]}$——相关系数有界的全部依据。<strong>Hölder 在概率论里就叫「矩不等式」</strong>，大数定律与中心极限定理的估计处处用到它。

## 6 小结

- **Hölder 不等式**：$\int|fg|\le\|f\|_p\|g\|_q$（$1/p+1/q=1$）。
- **证明**：Young（对数凹）+ 归一化 + 积分。
- **$p=q=2$**：Cauchy–Schwarz——Hölder 是 CS 的推广。
- **推论**：有限测度 $L^q\subset L^p$；$L^p$-$L^q$ 对偶配对；Young 卷积不等式。
- **地位**：$L^p$ 空间与泛函分析对偶理论的第一块基石。
- **边界**：$p=1,q=\infty$ 也成立；等号条件 = 支撑相同且强度成比例。
- **速查**：共轭对 $(2,2),(3,\tfrac32),(4,\tfrac43)$；$p$ 增 $q$ 减。
- **配对**：$L^p$ 与 $L^q$ 互为对偶；卷积不等式 $\|f*g\|_r\le\|f\|_p\|g\|_q$。

在下一节，我们用 Hölder 证明 **Minkowski 不等式**，完成 $L^p$ 的范数结构。
