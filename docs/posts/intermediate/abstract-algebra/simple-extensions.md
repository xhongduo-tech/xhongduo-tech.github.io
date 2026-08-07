---
title: 单扩张：单代数扩张与单超越扩张
date: 2026-08-07
---

# 单扩张：单代数扩张与单超越扩张

<div class="epigraph">
<p>添加一个元素得到的扩张，只有两种命运：它要么是方程的根，要么是自由变量。</p>
<footer>—— 自 题（单扩张笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§10.3 ｜ 2026-08-07</p>
</div>

## 为什么从单扩张开始

域扩张里最简单、也最重要的一类是**单扩张（simple extension）** $F(a)$——由添加单个元素 $a$ 生成的扩张。单扩张只有两种形态：**代数扩张**（$a$ 是某个多项式的根，$F(a)$ 有限维）与**超越扩张**（$a$ 是「自由变量」，$F(a)$ 无穷维）。

单扩张是域论的「原子操作」：任何有限扩张 $F(a_1, \dots, a_n)$ 都可以「逐个添加」拆成一串单扩张。理解 $F(a)$ 的结构（同构于 $F[x]/\langle m\rangle$ 或 $F(x)$），就理解了域扩张的全部「拼装单元」。本节把单代数扩张与单超越扩张的完整结构讲透。

## 1 代数元与超越元

**代数元（algebraic element）**：设 $E/F$ 是域扩张，$a \in E$。若存在非零多项式 $f \in F[x]$ 使 $f(a) = 0$，则称 $a$ 是 $F$ 上的**代数元**；否则称 $a$ 是 $F$ 上的**超越元（transcendental element）**。

**例：**
- $\sqrt2, i, \sqrt[3]2$ 都是 $\mathbb{Q}$ 上的代数元（分别是 $x^2-2$、$x^2+1$、$x^3-2$ 的根）；
- $\pi, e$ 是 $\mathbb{Q}$ 上的超越元（林德曼定理：$\pi$ 超越；埃尔米特定理：$e$ 超越）；
- $x$（不定元）在 $F(x)$ 中是 $F$ 上的超越元（没有多项式以 $x$ 为根）。<span class="marginnote">「代数 vs 超越」把添加的元素分成两类：代数元「受到多项式方程的束缚」，超越元「完全自由」。$\mathbb{Q}$ 上代数元可数而超越元不可数（绝大多数实数都超越），但证明某个具体数超越极难——$\pi$ 的超越性直到 1882 年才由林德曼证明（还顺带解决了化圆为方问题，第十一篇会讲）。</span>

## 2 单代数扩张的结构：F(a) ≅ F[x]/⟨m⟩

**定理（单代数扩张结构定理）：** 设 $a$ 是 $F$ 上的代数元，$m(x) \in F[x]$ 是 $a$ 的最小多项式（次数 $n$）。则

1. $m$ 不可约（在 $F[x]$ 中）；
2. **$F(a) \cong F[x] / \langle m \rangle$**，同构 $f(x) + \langle m\rangle \mapsto f(a)$；
3. $[F(a) : F] = n$，基为 $\{ 1, a, \dots, a^{n-1} \}$。

**证明要点**：求值同态 $\varphi : F[x] \to E$（$f \mapsto f(a)$）的核是 $\langle m \rangle$（$m$ 是次数最小的零化多项式，任何零化多项式被 $m$ 整除）。由第一同构定理 $F[x]/\langle m\rangle \cong \operatorname{Im}\varphi$。$m$ 不可约 ⟹ $\langle m\rangle$ 极大 ⟹ $F[x]/\langle m\rangle$ 是域，故 $\operatorname{Im}\varphi$ 是含 $F$ 与 $a$ 的域，即 $F(a)$。$\blacksquare$<span class="marginnote">「$F(a) \cong F[x]/\langle m\rangle$」是域论最核心的同构之一：它把「添加一个代数元」翻译成「商掉一个不可约多项式」。$a$ 的全部算术（加、乘、求逆）都由 $m$ 控制：$a$ 的幂超过 $n-1$ 就按 $m(a) = 0$ 化简。$\mathbb{C} \cong \mathbb{R}[x]/\langle x^2+1\rangle$ 是最早的例子，现在它成为「单代数扩张」的通用模板。</span>

**例：** $\mathbb{Q}(\sqrt2) \cong \mathbb{Q}[x]/\langle x^2 - 2\rangle$；$\mathbb{F}_4 = \mathbb{F}_2(\alpha) \cong \mathbb{F}_2[x]/\langle x^2 + x + 1\rangle$（$\alpha$ 是 $x^2+x+1$ 的根）。

## 3 单超越扩张的结构：F(a) ≅ F(x)

**定理（单超越扩张结构定理）：** 设 $a$ 是 $F$ 上的超越元。则

$$
F(a) \cong F(x)
$$

即 $F(a)$ 同构于**有理函数域** $F(x) = \{ \frac{p(x)}{q(x)} \mid p, q \in F[x], q \ne 0 \}$，同构 $a \mapsto x$。

**证明**：求值同态 $\varphi : F[x] \to E$（$f \mapsto f(a)$）在超越情形**单射**（核是 $\{0\}$，因为 $a$ 无零化多项式）。由同态基本定理 $F[x] \cong F[a]$（多项式环嵌入）；再取商域（分数），$F(a) = \{ \frac{p(a)}{q(a)} \} \cong \operatorname{Frac}(F[x]) = F(x)$。$\blacksquare$<span class="marginnote">超越元的「自由」：$a$ 没有任何关系约束，所以 $F(a)$ 就是「把 $a$ 当不定元」的有理函数域 $F(x)$。$F(a)$ 因此是无穷维扩张（$[F(x):F] = \infty$，因为 $1, x, x^2, \dots$ 线性无关）。<strong>代数 ⟹ 有限维，超越 ⟹ 无穷维</strong>——单扩张的两类完全分野。</span>

**例：** $\mathbb{Q}(\pi) \cong \mathbb{Q}(x)$（$\pi$ 超越）；$\mathbb{Q}(e) \cong \mathbb{Q}(x)$。这两个「无理数添加」的扩张都同构于有理函数域——因为 $\pi, e$ 都是超越元。

## 4 公式解析：最小多项式与 F(a) 的次数

把「单代数扩张的次数 = 最小多项式的次数」从机制上拆透，它是全部计算的枢纽。

**定理：** $a$ 代数元，最小多项式 $m$，则 $[F(a) : F] = \deg m$。

- **第一步，$F(a)$ 的元素形态。** 由 $F(a) \cong F[x]/\langle m\rangle$，$F(a)$ 中每个元素都可写成 $f(a)$ 的形式（$f \in F[x]$）。

- **第二步，带余除法压次数。** $f = mq + r$（带余除法，$\deg r < \deg m$），代入 $a$ 得 $f(a) = r(a)$——$F(a)$ 的元素总可用次数 < $\deg m$ 的多项式表示。

- **第三步，基的确定。** $\{ 1, a, \dots, a^{n-1} \}$（$n = \deg m$）张成 $F(a)$（上一步）；线性无关（若 $\sum c_i a^i = 0$ 且系数不全零，则 $a$ 有次数 < $n$ 的零化多项式，与 $m$ 最小矛盾）。于是它是 $F$-基，$[F(a):F] = n$。$\blacksquare$

- **第四步，直觉。** 最小多项式「测量」$a$ 受约束的程度：约束越紧（$m$ 次数越低），$F(a)$ 越小；约束越松，$F(a)$ 越大。超越元无约束，$F(a)$ 无穷大。**「代数元 = 有限维单扩张」是域论最重要的等价。**

## 5 例：单扩张的实战

把单扩张在几个关键例子上算透。

**例 1：$\mathbb{Q}(\sqrt[3]{2})$。** $\sqrt[3]2$ 的最小多项式是 $x^3 - 2$（艾森斯坦，$p = 2$），次数 3。故 $[\mathbb{Q}(\sqrt[3]2) : \mathbb{Q}] = 3$，基 $\{1, \sqrt[3]2, \sqrt[3]4\}$，$\mathbb{Q}(\sqrt[3]2) = \{ a + b\sqrt[3]2 + c\sqrt[3]4 \}$。$\sqrt[3]2$ 的「立方根算术」完全由 $(\sqrt[3]2)^3 = 2$ 控制。

**例 2：$\mathbb{F}_2$ 上添加不可约二次根。** $x^2 + x + 1$ 在 $\mathbb{F}_2$ 不可约（无根：$0^2+0+1 = 1$、$1^2+1+1 = 1$），$\alpha$ 是它的根，则 $\mathbb{F}_2(\alpha) \cong \mathbb{F}_2[x]/\langle x^2+x+1\rangle$ 有 4 个元素 $\{ 0, 1, \alpha, \alpha + 1 \}$，是 4 阶域 $\mathbb{F}_4$。**这是有限域 $\mathbb{F}_{2^n}$ 的构造雏形**（下一篇全面展开）。<span class="marginnote">$\mathbb{F}_4 = \{ 0, 1, \alpha, \alpha+1 \}$（$\alpha^2 = \alpha + 1$）是「单代数扩张造有限域」的最简示范：$\mathbb{F}_2$ 添加一个不可约二次式的根，得到 4 个元素，构成 4 阶域。<strong>「有限域 = $\mathbb{F}_p$ 添加不可约多项式之根」</strong>——单扩张是有限域的全部引擎。</span>

**例 3：$\mathbb{Q}(\sqrt2, \sqrt3)$ 是单扩张。** 取 $a = \sqrt2 + \sqrt3$，则 $\mathbb{Q}(\sqrt2, \sqrt3) = \mathbb{Q}(a)$（$a$ 的最小多项式 $x^4 - 10x^2 + 1$ 次数 4）。**「有限生成扩张常常是单扩张」**——本原元定理（Galois 理论里的经典）保证特征 0 的有限扩张都是单扩张。

## 6 例子：单扩张的次数计算集锦

单扩张的次数 $[F(a):F] = \deg m$（$m$ 是最小多项式）是域论计算的地基，把一批经典例子集中算一遍。

**例 1：$\mathbb{Q}(\sqrt[3]{2})$。** $x^3 - 2$ 在 $\mathbb{Q}$ 不可约（艾森斯坦，$p = 2$），故 $[\mathbb{Q}(\sqrt[3]2):\mathbb{Q}] = 3$。基 $\{1, \sqrt[3]2, \sqrt[3]4\}$，每个元素 $a + b\sqrt[3]2 + c\sqrt[3]4$。

**例 2：$\mathbb{Q}(\sqrt2 + \sqrt3)$。** $\alpha = \sqrt2 + \sqrt3$ 的最小多项式？$\alpha^2 = 5 + 2\sqrt6$，$(\alpha^2 - 5)^2 = 24$，故 $\alpha$ 满足 $x^4 - 10x^2 + 1 = 0$。此多项式不可约（可用艾森斯坦的平移变体或直接判），$[\mathbb{Q}(\sqrt2+\sqrt3):\mathbb{Q}] = 4$。

**例 3：$\mathbb{F}_2$ 上添加三次根。** $\mathbb{F}_2$ 上 $x^3 + x + 1$ 无根（$0^3+0+1 = 1$、$1^3+1+1 = 1$），不可约。设 $\alpha$ 是其根，$\mathbb{F}_2(\alpha) \cong \mathbb{F}_2[x]/\langle x^3+x+1\rangle$ 有 $2^3 = 8$ 个元素，是 $\mathbb{F}_8$。$\alpha$ 满足 $\alpha^3 = \alpha + 1$（特征 2 里 $-1 = 1$），$\mathbb{F}_8 = \{ a + b\alpha + c\alpha^2 \}$。

**例 4（超越）**：$\mathbb{Q}(\pi) \cong \mathbb{Q}(x)$，$[\mathbb{Q}(\pi):\mathbb{Q}] = \infty$。$\pi$ 超越，$1, \pi, \pi^2, \dots$ 线性无关。

**观察**：代数元的最小多项式次数 = 扩张次数，而超越元给出无穷扩张。**「算单扩张次数 = 算最小多项式次数」**——第九篇的不可约性判据（艾森斯坦、有理根定理）在这里是全部计算工具。

## 7 小结

- **代数元 vs 超越元**：是否有零化多项式；$\sqrt2$ 代数、$\pi$ 超越。
- **单代数扩张**：$F(a) \cong F[x]/\langle m\rangle$，$[F(a):F] = \deg m$，基 $\{1, a, \dots, a^{n-1}\}$。
- **单超越扩张**：$F(a) \cong F(x)$（有理函数域），无穷维。
- **次数 = 最小多项式次数**：代数元有限维、超越元无穷维的分野。
- **应用**：$\mathbb{F}_4$、$\mathbb{Q}(\sqrt[3]2)$ 都是单扩张；有限扩张常为单扩张（本原元定理）。

在下一节，我们把单扩张推广到任意扩张：**代数扩张与超越扩张**。一个扩张的每个元素是否都代数，决定了扩张的性质与结构。
