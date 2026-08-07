---
title: 本原多项式与高斯（Gauss）引理
date: 2026-08-07
---

# 本原多项式与高斯（Gauss）引理

<div class="epigraph">
<p>本原多项式把「整数系多项式」与「有理系多项式」的不可约性完美挂钩——高斯引理就是那根挂钩。</p>
<footer>—— 自 题（高斯引理笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§9.8 ｜ 2026-08-07</p>
</div>

## 为什么从本原多项式与高斯引理开始

$\mathbb{Z}[x]$ 不是 PID，但它是 UFD——这件事并不显然。它的证明核心是**高斯引理（Gauss's Lemma）**：**本原多项式（系数互素的多项式）在 $\mathbb{Z}[x]$ 中不可约，当且仅当它在 $\mathbb{Q}[x]$ 中不可约。**

这条引理把「整数系数多项式的不可约性」归结为「有理系数多项式的不可约性」——而后者有更强的工具（次数、带余除法、艾森斯坦判别法）。它同时证明「$D$ 是 UFD ⟹ $D[x]$ 是 UFD」（高斯引理可推广到任意 UFD），从而 $\mathbb{Z}[x]$、多变量多项式环的 UFD 性一网打尽。本节把本原多项式的概念、高斯引理的陈述与证明、以及「UFD ⟹ $D[x]$ UFD」讲透。

## 1 本原多项式

**本原多项式（primitive polynomial）**：设 $D$ 是 UFD，$f \in D[x]$ 非零。若 $f$ 的系数没有非平凡的公共因子（即系数的 gcd 是单位），则称 $f$ 是**本原多项式**。

**例：**
- $2x^2 + 3x + 1$ 在 $\mathbb{Z}[x]$ 是本原的（系数 $2, 3, 1$ 的 gcd 是 1）；
- $2x^2 + 4x + 2$ 在 $\mathbb{Z}[x]$ **不是**本原的（gcd 是 2）；
- 任意非常数多项式在域 $F[x]$ 上都本原（$F$ 里任何非零系数都是单位）。<span class="marginnote">「本原」度量系数的「互素程度」：本原 = 系数不能再提公共因子。在 $\mathbb{Z}[x]$ 里，任何非零多项式都能「提公因子」写成「内容 × 本原部分」：$2x^2 + 4x + 2 = 2 \cdot (x^2 + 2x + 1)$，其中 $2$ 是内容（content）、$x^2 + 2x + 1$ 是本原部分。<strong>「内容 × 本原部分」是本原概念的算术化。</strong></span>

**内容（content）**：$f$ 的系数的 gcd 称为 $f$ 的内容，记作 $\mathrm{cont}(f)$。则 $f = \mathrm{cont}(f) \cdot f^{\mathrm{prim}}$（本原部分）。本原 ⟺ $\mathrm{cont}(f)$ 是单位。

## 2 高斯引理：本原多项式的乘积

**定理（高斯引理）：** 设 $D$ 是 UFD，$f, g \in D[x]$ 都本原，则 $fg$ 也本原。

**证明（用素元）：** 反证，设 $fg$ 不本原，则存在素元 $p$ 整除 $fg$ 的所有系数。考虑模 $p$ 的商环 $(D/\langle p\rangle)[x]$，$fg$ 的像为零（所有系数被 $p$ 整除）。但 $f, g$ 本原，$f$ 的像与 $g$ 的像都非零（$p$ 不整除它们的全部系数）；$D/\langle p \rangle$ 是整环（$p$ 素），$(D/\langle p\rangle)[x]$ 也是整环（整环上多项式环整环），非零元的乘积非零——矛盾。故 $fg$ 本原。$\blacksquare$<span class="marginnote">高斯引理的证明用「模 $p$ 投影」：把「$fg$ 有公共因子 $p$」翻译成「$fg \bmod p = 0$」，而模 $p$ 后 $f, g$ 都非零，$(D/\langle p\rangle)[x]$ 是整环，非零乘积非零——矛盾。<strong>素元的除法性（$D/\langle p\rangle$ 是整环）是全部武器。</strong> 注意这里需要 $D$ 是 UFD（素元分解的存在），高斯引理对 UFD 成立。</span>

**推论（内容乘积律）：** 对 $D$ 是 UFD，$\mathrm{cont}(fg) = \mathrm{cont}(f)\,\mathrm{cont}(g)$（计相伴）。这来自「$fg$ 的本原部分 = 两本原部分之积」。

## 3 高斯引理的核心应用：Z[x] 与 Q[x] 的不可约性

**定理（$\mathbb{Z}[x]$ 与 $\mathbb{Q}[x]$ 的不可约性对应）：** 设 $f \in \mathbb{Z}[x]$ 是本原的且次数 ≥ 1。则 $f$ 在 $\mathbb{Z}[x]$ 中不可约 ⟺ $f$ 在 $\mathbb{Q}[x]$ 中不可约。

**证明（⟸，$\mathbb{Q}$ 不可约 ⟹ $\mathbb{Z}$ 不可约）：** 若 $f = gh$ 在 $\mathbb{Z}[x]$ 中分解（$g, h$ 都非单位），则同样的分解在 $\mathbb{Q}[x]$ 中成立，$f$ 在 $\mathbb{Q}$ 上可约，矛盾。

**证明（⟹，$\mathbb{Z}$ 不可约 ⟹ $\mathbb{Q}$ 不可约）：** 反证，设 $f = GH$ 在 $\mathbb{Q}[x]$ 中分解（$\deg G, \deg H \ge 1$）。把 $G, H$ 的分母清掉：$G = \frac{a}{b} G_0$、$H = \frac{c}{d} H_0$（$G_0, H_0$ 本原、整系数），则 $f = \frac{ac}{bd} G_0 H_0$。由 $f$ 本原且 $G_0H_0$ 本原（高斯引理），$\frac{ac}{bd}$ 必为单位（比较内容），$f = \pm G_0 H_0$ 是 $\mathbb{Z}[x]$ 中的真分解，矛盾。$\blacksquare$<span class="marginnote">「$\mathbb{Q}$ 可约 ⟹ $\mathbb{Z}$ 可约」的关键是「清分母 + 高斯引理」：有理分解 $G H$ 先清分母成本原多项式，乘积本原（高斯引理），剩下的常数因子必须是单位，于是分解落在 $\mathbb{Z}[x]$ 里。<strong>这条定理让我们在 $\mathbb{Z}$ 上放心使用一切有理系数的不可约判据（艾森斯坦等）。</strong></span>

**推论（判定不可约的通道）**：要证 $f \in \mathbb{Z}[x]$ 不可约，只需：① 提内容，$f = \mathrm{cont}(f) f^{\mathrm{prim}}$；② 证本原部分在 $\mathbb{Q}[x]$ 不可约（用艾森斯坦判别法、有理根定理等）。**整数系多项式的不可约性 = 有理系数的不可约性 + 本原性。**

## 4 公式解析：UFD ⟹ D[x] 是 UFD

高斯引理的最深应用是「UFD 的 UFD 性向上继承」。

**定理：** 设 $D$ 是 UFD，则 $D[x]$ 是 UFD。

**证明骨架（用内容分解）：** 对 $f \in D[x]$：

- **第一步，分离内容。** $f = \mathrm{cont}(f) \cdot f^{\mathrm{prim}}$。$\mathrm{cont}(f) \in D$，$D$ 是 UFD，内容部分可唯一分解；$f^{\mathrm{prim}}$ 是本原部分。

- **第二步，本原部分在 $\mathbb{Q}(D)$ 上分解。** 记 $F = \operatorname{Frac}(D)$（商域），$F[x]$ 是 UFD（PID），故 $f^{\mathrm{prim}}$ 在 $F[x]$ 中有唯一分解：$f^{\mathrm{prim}} = g_1 \cdots g_k$（不可约，$g_i \in F[x]$）。

- **第三步，清分母。** 每个 $g_i$ 清分母成 $D[x]$ 中的本原不可约多项式 $\tilde g_i$（高斯引理保证清分母后仍不可约）。于是 $f^{\mathrm{prim}} = u \cdot \tilde g_1 \cdots \tilde g_k$（$u$ 单位，因两边都本原）。

- **第四步，合拢。** $f = \mathrm{cont}(f) \cdot u \cdot \tilde g_1 \cdots \tilde g_k$，其中内容分解 + 本原分解各自唯一，合起来给出 $f$ 在 $D[x]$ 中的唯一分解。$\blacksquare$<span class="marginnote">「$D$ UFD ⟹ $D[x]$ UFD」的证明把 $D[x]$ 的分解拆成「内容（在 $D$ 里分解）+ 本原部分（在 $F[x]$ 里分解）」。$F[x]$ 的 UFD 性由「$F$ 域 ⟹ $F[x]$ PID ⟹ UFD」供给，高斯引理负责「清分母不破坏不可约性」。<strong>归纳应用：$\mathbb{Z}[x_1, \dots, x_n]$ 与 $F[x_1, \dots, x_n]$ 都是 UFD。</strong></span>

**推论：** $\mathbb{Z}[x]$、$F[x_1, \dots, x_n]$、$\mathbb{Z}[x_1, \dots, x_n]$ 都是 UFD——**高斯引理把 UFD 性从系数环推广到任意有限个变量的多项式环**。

## 5 例：用本原性与高斯引理判定

把本原性与高斯引理用在实战判定上。

**例 1：** $f = 4x^2 + 6x + 2 \in \mathbb{Z}[x]$。$\mathrm{cont}(f) = 2$，$f^{\mathrm{prim}} = 2x^2 + 3x + 1$。$f^{\mathrm{prim}}$ 在 $\mathbb{Q}$ 上可约（$2x^2 + 3x + 1 = (2x+1)(x+1)$），故 $f$ 在 $\mathbb{Z}$ 上可约：$4x^2 + 6x + 2 = 2(2x+1)(x+1)$。$\checkmark$

**例 2：** $f = x^2 + 1 \in \mathbb{Z}[x]$ 本原。$x^2 + 1$ 在 $\mathbb{Q}[x]$ 不可约（无有理根且二次），由高斯引理在 $\mathbb{Z}[x]$ 不可约。

**例 3（内容乘积律的用法）：** 设 $f = 2(x^2 + 1)$、$g = 3(x + 1)$，$\mathrm{cont}(f) = 2$、$\mathrm{cont}(g) = 3$，$\mathrm{cont}(fg) = \mathrm{cont}(2\cdot3(x^2+1)(x+1)) = 6 = 2 \cdot 3$。$\checkmark$<span class="marginnote">内容乘积律 $\mathrm{cont}(fg) = \mathrm{cont}(f)\mathrm{cont}(g)$ 是「gcd 的乘法性」：$\gcd$ 系数的 gcd 等于各自 gcd 的乘积（在 UFD 里）。这是小学「最大公因数可分别提」的严格版，也是「$fg$ 的本原部分 = 各自本原部分乘积」的简洁写法。</span>

## 6 例子：高斯引理的应用链

高斯引理的价值在于它是一根「传递链」——把 UFD 性从系数环一路推到多项式环。沿链条走一遍，体会它的普适性。

**$\mathbb{Z}[x]$ 是 UFD（但非 PID）。** 由「$D$ UFD ⟹ $D[x]$ UFD」，取 $D = \mathbb{Z}$ 得 $\mathbb{Z}[x]$ 是 UFD。而 $\mathbb{Z}[x]$ 不是 PID（$\langle 2, x\rangle$ 非主理想）——**UFD 与 PID 在多项式环上分离**，高斯引理正是这分离的「分界线」。

**多变量多项式环是 UFD。** 归纳应用「$D$ UFD ⟹ $D[x]$ UFD」：$\mathbb{Z}[x_1, \dots, x_n] = (\cdots((\mathbb{Z}[x_1])[x_2])\cdots)[x_n]$ 是 UFD。**高斯引理让「有限个变量的多项式环」永远保持 UFD 性**——无论系数是 $\mathbb{Z}$ 还是域 $F$。

**$F[x, y]$ 是 UFD 但非 PID。** $\langle x, y\rangle$（常数项为 0 的二元多项式）不是主理想，但 $F[x, y]$ 仍是 UFD。**「多变量多项式环 = UFD 而 PID 彻底失效」**——这是代数几何里「多项式环的谱」理论的基础观察。

**一条判定链**：要判 $f \in \mathbb{Z}[x_1, \dots, x_n]$ 的不可约性，可以「先提内容、再把它当 $\mathbb{Q}[x_1, \dots, x_n]$ 里的多项式判」，高斯引理保证两者一致。**「整数系 ↔ 有理系」的不可约性对应，在任意多个变量上都成立**——这是高斯引理在计算中的终极形态。

## 7 小结

- **本原多项式**：系数 gcd 为单位；$f = \mathrm{cont}(f) \cdot f^{\mathrm{prim}}$（内容 × 本原部分）。
- **高斯引理**：本原多项式之积仍本原（模 $p$ 投影 + 整环性证明）。
- **$\mathbb{Z}[x]$ ⟷ $\mathbb{Q}[x]$ 不可约对应**：本原时两者不可约性等价（清分母 + 高斯引理）。
- **UFD ⟹ $D[x]$ UFD**：内容分解 + 商域分解 + 高斯引理清分母。
- **推论**：$\mathbb{Z}[x]$、多变量多项式环都是 UFD。

在下一节，我们拿到整环上多项式不可约性的最强判别法：**艾森斯坦（Eisenstein）不可约判别法**。它只需看系数的整除关系，就能判定一批重要多项式不可约。
