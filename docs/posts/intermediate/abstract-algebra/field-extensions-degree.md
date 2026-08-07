---
title: 域扩张与扩张次数
date: 2026-08-07
---

# 域扩张与扩张次数

<div class="epigraph">
<p>把一个域扩大，让它装下方程的根——域扩张是「解方程」的代数本质。</p>
<footer>—— 自 题（域扩张笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§10.2 ｜ 2026-08-07</p>
</div>

## 为什么从域扩张与扩张次数开始

上一节的素域是「最小底座」，这一节开始攀登：**域扩张（field extension）**把一个域 $F$ 扩成更大的域 $E$（$F \subseteq E$）。为什么要扩张？——**为了解方程**。$x^2 + 1 = 0$ 在 $\mathbb{R}$ 里无解，扩张到 $\mathbb{C}$ 就有解了；$x^3 - 2 = 0$ 在 $\mathbb{Q}$ 里无解，扩张到 $\mathbb{Q}(\sqrt[3]{2})$ 就有解了。域扩张是「把方程的根装进去」的形式化。

扩张的次数 $[E : F]$ 是扩张的「维度」——$E$ 作为 $F$-向量空间的维数。它满足漂亮的「乘法塔」公式 $[E : K] = [E : F][F : K]$，是整个域论结构（有限域、分裂域、Galois 理论）的度量骨架。本节把扩张的定义、次数、乘法塔与有限扩张讲透。

## 1 域扩张与向量空间视角

**域扩张（field extension）**：设 $F, E$ 是域且 $F \subseteq E$（$F$ 是 $E$ 的子域），则称 $E$ 是 $F$ 的**域扩张**，记作 $E/F$（读作「$E$ over $F$」）。$F$ 称为**基域**。

**关键视角：$E$ 是 $F$ 上的向量空间。** 域 $E$ 的加法是向量加法，$F$ 的元素是标量（$F \subseteq E$，标量乘法即域乘法）。于是线性代数的一切概念（基、维数、线性相关）都可搬进域论。

**扩张次数（degree of extension）**：$E$ 作为 $F$-向量空间的维数称为 $E/F$ 的扩张次数，记作

$$
[E : F] = \dim_F E
$$

若 $[E : F] < \infty$，称 $E/F$ 是**有限扩张**；否则是**无限扩张**。<span class="marginnote">「$E$ 是 $F$-向量空间」是域扩张理论的第一洞察：把域论问题翻译成线性代数问题，维数就成了度量。$\mathbb{C}/\mathbb{R}$：$\mathbb{C}$ 是 $\mathbb{R}$ 上的 2 维空间，基 $\{1, i\}$，$[\mathbb{C} : \mathbb{R}] = 2$。$\mathbb{R}/\mathbb{Q}$：$\mathbb{R}$ 是 $\mathbb{Q}$ 上的无穷维空间，$[\mathbb{R} : \mathbb{Q}] = \infty$。</span>

**例：**
- $[\mathbb{C} : \mathbb{R}] = 2$（基 $\{1, i\}$）；
- $[\mathbb{Q}(\sqrt{2}) : \mathbb{Q}] = 2$（$\mathbb{Q}(\sqrt2) = \{ a + b\sqrt2 \}$，基 $\{1, \sqrt2\}$）；
- $[\mathbb{R} : \mathbb{Q}] = \infty$；
- $[\mathbb{F}_{p^n} : \mathbb{F}_p] = n$（有限域是 $n$ 维 $\mathbb{F}_p$-空间）。

## 2 生成扩张：F(a) 与 F(a₁,...,aₙ)

**由集合生成的扩张**：设 $S \subseteq E$，包含 $F$ 与 $S$ 的最小扩域记作 $F(S)$，称为「$F$ 添加 $S$ 生成的扩张」。

**单扩张（simple extension）**：$S = \{ a \}$ 时记作 $F(a)$，称为**单扩张**（由单个元素生成）。

**例：**
- $\mathbb{Q}(\sqrt{2}) = \{ a + b\sqrt{2} \}$（添加 $\sqrt2$）；
- $\mathbb{Q}(i) = \{ a + bi \}$（添加 $i$）；
- $\mathbb{C} = \mathbb{R}(i)$；
- $\mathbb{Q}(\sqrt[3]{2})$ = 全部 $\{ a + b\sqrt[3]{2} + c\sqrt[3]{4} \}$（添加 $\sqrt[3]2$）。

**直觉**：$F(a)$ 是把 $a$「塞进」$F$，再用加、减、乘、除闭合成域的最小结果。$F(a)$ 的「大小」由 $a$ 的「代数性质」决定——$a$ 若是某方程的根（代数元），$F(a)$ 有限维；否则（超越元）无限维（下一篇）。<span class="marginnote">「$F(a)$ 是含 $a$ 的最小域」与「$\langle a\rangle$ 是含 $a$ 的最小群/理想」同构同源——都是「生成」的通用模板。但域里的「生成」除了加减乘，还要对除法封闭（取倒数），所以 $F(a)$ 通常不是「多项式」而是「有理函数」$p(a)/q(a)$ 的全体。下一篇的「代数元」会让这个有理函数退化为多项式。</span>

## 3 扩张次数的乘法塔

扩张次数最重要的性质是「可乘」。

**定理（乘法塔 / Tower Law）：** 设 $F \subseteq K \subseteq E$ 是三个域，则

$$
[E : F] = [E : K] \cdot [K : F]
$$

**证明：** 设 $\{ e_i \}$ 是 $E$ 的 $K$-基（$[E:K]$ 个），$\{ k_j \}$ 是 $K$ 的 $F$-基（$[K:F]$ 个）。断言 $\{ e_i k_j \}$ 是 $E$ 的 $F$-基：

- **张成**：任意 $e \in E$ 写为 $\sum \alpha_i e_i$（$\alpha_i \in K$），每个 $\alpha_i = \sum \beta_{ij} k_j$（$\beta_{ij} \in F$），代入得 $e = \sum \beta_{ij} e_i k_j$；
- **线性无关**：若 $\sum \beta_{ij} e_i k_j = 0$，先按 $i$ 分组 $\sum_i (\sum_j \beta_{ij} k_j) e_i = 0$，$\{e_i\}$ 的 $K$-无关性给 $\sum_j \beta_{ij} k_j = 0$（每 $i$），$\{k_j\}$ 的 $F$-无关性给 $\beta_{ij} = 0$。$\blacksquare$<span class="marginnote">乘法塔的证明是「两套基张成一套基」的线性代数论证：$E$ 在 $K$ 上的基与 $K$ 在 $F$ 上的基相乘，得到 $E$ 在 $F$ 上的基。它让「扩张维度」像「分数相乘」一样可拼装——$\mathbb{C}/\mathbb{Q}$ 的扩张次数是 $[\mathbb{C}:\mathbb{R}][\mathbb{R}:\mathbb{Q}]$，但注意 $\mathbb{R}/\mathbb{Q}$ 无穷，所以 $\mathbb{C}/\mathbb{Q}$ 也无穷。</span>

**例：**
- $[\mathbb{C} : \mathbb{Q}] = [\mathbb{C} : \mathbb{R}] \cdot [\mathbb{R} : \mathbb{Q}] = 2 \cdot \infty = \infty$；
- $[\mathbb{F}_{p^{6}} : \mathbb{F}_p] = [\mathbb{F}_{p^6} : \mathbb{F}_{p^2}] \cdot [\mathbb{F}_{p^2} : \mathbb{F}_p]$，若 $\mathbb{F}_{p^2} \subseteq \mathbb{F}_{p^6}$（$2 \mid 6$）。

**推论（次数整除性）：** 若 $K$ 夹在 $F$ 与 $E$ 之间，则 $[E : K]$ 与 $[K : F]$ 都整除 $[E : F]$。**扩张次数限制中间域的可能性**——有限域里 $\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n}$ 当且仅当 $d \mid n$（下一篇）。

## 4 公式解析：F(a) 的次数 = 最小多项式的次数

单扩张 $F(a)$ 的次数是域论最常用的计算，它与「$a$ 的最小多项式」挂钩。

**定理（单代数扩张的次数）：** 设 $a$ 是 $F$ 上的**代数元**（存在非零 $f \in F[x]$ 使 $f(a) = 0$），$m(x)$ 是 $a$ 的**最小多项式**（次数最小的首一零化多项式），则

$$
[F(a) : F] = \deg m
$$

且 $\{ 1, a, a^2, \dots, a^{\deg m - 1} \}$ 是 $F(a)$ 的 $F$-基。

**证明（三步）：**
- **第一步，$F(a) \cong F[x]/\langle m \rangle$**：求值同态 $\varphi : F[x] \to E$，$f \mapsto f(a)$ 的核是 $\langle m \rangle$（最小多项式生成），由第一同构定理 $F[x]/\langle m \rangle \cong \operatorname{Im}\varphi$；$m$ 不可约 ⟹ $\langle m\rangle$ 极大 ⟹ $F[x]/\langle m\rangle$ 是域，故 $\operatorname{Im}\varphi$ 是含 $F$ 与 $a$ 的域，即 $F(a)$。
- **第二步，维数**：$F[x]/\langle m \rangle$ 作为 $F$-空间的基是 $\{ \bar 1, \bar x, \dots, \bar x^{\deg m - 1} \}$（带余除法保证任何多项式模 $m$ 后次数 < $\deg m$），维数 = $\deg m$。
- **第三步，同构保维**：$F(a) \cong F[x]/\langle m\rangle$，$[F(a):F] = \deg m$。$\blacksquare$<span class="marginnote">这条定理是「商环造域」与「扩张次数」的首次会师：$F(a)$ 的次数完全由 $a$ 的最小多项式次数决定。$\mathbb{Q}(\sqrt2)$ 的次数 = $\deg(x^2 - 2) = 2$；$\mathbb{Q}(\sqrt[3]2)$ 的次数 = $\deg(x^3 - 2) = 3$；$\mathbb{Q}(i)$ 的次数 = $\deg(x^2 + 1) = 2$。<strong>「单扩张的次数 = 最小多项式次数」是有限域与 Galois 理论的计算核心。</strong></span>

**例：** $[\mathbb{Q}(\sqrt[3]{2}) : \mathbb{Q}] = 3$（最小多项式 $x^3 - 2$，艾森斯坦判定不可约）。$\mathbb{Q}(\sqrt[3]2) = \{ a + b\sqrt[3]2 + c\sqrt[3]4 \}$，基 $\{1, \sqrt[3]2, \sqrt[3]4\}$。

## 5 例：有限扩张的运算

把有限扩张的次数计算练熟，它是整个域论的算术基础。

**例 1（复合扩张）**：$[\mathbb{Q}(\sqrt2, \sqrt3) : \mathbb{Q}]$。$\mathbb{Q}(\sqrt2)$ 次数 2；$\sqrt3$ 在 $\mathbb{Q}(\sqrt2)$ 上的最小多项式是 $x^2 - 3$（$\sqrt3 \notin \mathbb{Q}(\sqrt2)$），故 $[\mathbb{Q}(\sqrt2, \sqrt3) : \mathbb{Q}(\sqrt2)] = 2$，乘法塔给出 $[\mathbb{Q}(\sqrt2,\sqrt3):\mathbb{Q}] = 2 \times 2 = 4$。基 $\{1, \sqrt2, \sqrt3, \sqrt6\}$。

**例 2（有限域）**：$[\mathbb{F}_4 : \mathbb{F}_2] = 2$（$\mathbb{F}_4 = \mathbb{F}_2[x]/\langle x^2 + x + 1\rangle$，最小多项式次数 2）；$[\mathbb{F}_8 : \mathbb{F}_2] = 3$。<span class="marginnote">有限域 $\mathbb{F}_{p^n}$ 的次数 $n$ 完全决定它的「大小」：$|\mathbb{F}_{p^n}| = p^n$（$n$ 维 $\mathbb{F}_p$-空间有 $p^n$ 个元素）。下一篇《有限域》会证明：对每个 $n$ 恰好存在一个 $p^n$ 阶有限域，且 $\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n} \iff d \mid n$。次数 $n$ 是有限域的「身份证」。</span>

## 6 小结

- **域扩张** $E/F$：$F \subseteq E$；$E$ 是 $F$-向量空间，次数 $[E:F] = \dim_F E$。
- **生成扩张**：$F(S)$、单扩张 $F(a)$；$F(a)$ 是含 $a$ 的最小域。
- **乘法塔**：$[E:F] = [E:K][K:F]$；中间域次数整除扩张次数。
- **单代数扩张**：$[F(a):F] = \deg(\text{最小多项式})$，基 $\{1, a, \dots, a^{n-1}\}$。
- 有限域 $\mathbb{F}_{p^n}$ 是 $\mathbb{F}_p$ 的 $n$ 次扩张，$|\mathbb{F}_{p^n}| = p^n$。

在下一节，我们区分扩张的两类「大小」：**单扩张：单代数扩张与单超越扩张**。生成元是「方程之根」还是「自由变量」，决定了扩张是有限维还是无穷维。
