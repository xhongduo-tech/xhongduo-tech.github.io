---
title: 有限域（伽罗瓦域）的存在唯一性与结构
date: 2026-08-07
---

# 有限域（伽罗瓦域）的存在唯一性与结构

<div class="epigraph">
<p>有限域按阶完全分类：对每个素数幂 p^n，恰有一个有限域——伽罗瓦域是代数的完美结晶。</p>
<footer>—— 自 题（有限域笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§10.7 ｜ 2026-08-07</p>
</div>

## 为什么从有限域开始

有限域（又称**伽罗瓦域 Galois field**，记作 $\mathbb{F}_q$）是元素个数有限的域。有限域的完全分类是整个抽象代数最漂亮的成就之一：

- **存在性**：对每个素数幂 $q = p^n$，存在 $q$ 阶有限域 $\mathbb{F}_q$；
- **唯一性**：任何两个 $q$ 阶有限域同构；
- **结构**：$\mathbb{F}_{p^n} = \mathbb{F}_p[x]/\langle f \rangle$（$f$ 是 $n$ 次不可约多项式），且 $\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n} \iff d \mid n$。

有限域是现代密码学（AES 的 $\mathbb{F}_{2^8}$、椭圆曲线的 $\mathbb{F}_p$、Reed-Solomon 码）的全部舞台。理解有限域的分类与结构，是连接抽象代数与计算机科学的桥梁。本节把三大定理与构造彻底讲透。

## 1 有限域的特征与阶

**有限域（finite field）**：元素个数有限的域。其阶必为素数幂。

**定理：** 有限域的阶必为 $p^n$（$p$ 素数、$n \ge 1$），且有限域是其素域 $\mathbb{F}_p$ 的有限扩张。

**证明：** 有限域 $F$ 的特征是 $p$（特征 0 的域必无穷），素域是 $\mathbb{F}_p$；$F$ 作为 $\mathbb{F}_p$ 上的有限维向量空间（有限 ⟹ 有限维），维数 $n$，故 $|F| = p^n$。$\blacksquare$<span class="marginnote">「有限域 = $\mathbb{F}_p$ 上的有限维向量空间」给有限域的阶一个完全确定的形态：$p^n$。于是有限域完全由一对数（$p, n$）标记——素域 $\mathbb{F}_p$ 和扩张次数 $n$。反过来，$p^n$ 阶有限域是否存在、是否唯一，就是接下来的两大定理。</span>

**记号**：$\mathbb{F}_q$（$q = p^n$）或 $\mathrm{GF}(q)$（伽罗瓦域）。$\mathbb{F}_p = \mathbb{Z}/p\mathbb{Z}$，$\mathbb{F}_{p^n}$ 是其 $n$ 次扩张。

## 2 存在性：用不可约多项式构造

**定理（存在性）：** 对每个素数幂 $q = p^n$，存在 $q$ 阶有限域。

**证明（构造）：** 需要一个 $\mathbb{F}_p$ 上 $n$ 次不可约多项式 $f$。存在性由「$\mathbb{F}_p$ 上有任意次不可约多项式」保证（计数论证：$p$ 元域上的首一 $n$ 次多项式共 $p^n$ 个，可约的远少于总数）。取 $f$，则

$$
\mathbb{F}_q = \mathbb{F}_p[x] / \langle f \rangle
$$

有 $p^n = q$ 个元素（每个元素是次数 < $n$ 的多项式的同余类），且 $f$ 不可约 ⟹ $\mathbb{F}_p[x]/\langle f\rangle$ 是域。$\blacksquare$<span class="marginnote">构造「$\mathbb{F}_p$ 上任意次不可约多项式」的计数证明：首一 $n$ 次多项式共 $p^n$ 个，而可约的「由低次不可约式拼成」，数量远小于 $p^n$（可用莫比乌斯反演精确计数），故 $n$ 次不可约式必存在。艾森斯坦判别法不能直接用于 $\mathbb{F}_p$（没有「素数」），所以存在性靠计数而非显式构造——「知道有、未必写得出」是有限域构造的常态。</span>

**例：$\mathbb{F}_4$。** $\mathbb{F}_2$ 上二次不可约多项式是 $x^2 + x + 1$。$\mathbb{F}_4 = \mathbb{F}_2[x]/\langle x^2+x+1\rangle = \{ 0, 1, \alpha, \alpha+1 \}$（$\alpha^2 = \alpha + 1$），4 个元素，是域。

**例：$\mathbb{F}_8$。** $\mathbb{F}_2$ 上三次不可约多项式 $x^3 + x + 1$。$\mathbb{F}_8 = \mathbb{F}_2[x]/\langle x^3+x+1\rangle$，8 个元素。

## 3 唯一性：x^q - x 的分裂域

**定理（唯一性）：** 任何两个 $q = p^n$ 阶有限域同构。

**证明（$\mathbb{F}_q$ 是 $x^q - x$ 的分裂域）：** 设 $F$ 是 $q$ 阶有限域，$F^\times$ 是 $q-1$ 阶乘法群（拉格朗日），故对 $a \in F^\times$，$a^{q-1} = 1$，$a^q = a$；$a = 0$ 也满足 $0^q = 0$。于是**每个 $a \in F$ 都是 $x^q - x$ 的根**。$x^q - x$ 是 $q$ 次多项式、有 $q$ 个根，故 $x^q - x = \prod_{a \in F}(x - a)$ 在 $F$ 中完全分裂，$F$ 是 $x^q - x$ 在 $\mathbb{F}_p$ 上的分裂域。由分裂域的唯一性，任何 $q$ 阶有限域都同构于 $x^q - x$ 的分裂域。$\blacksquare$<span class="marginnote">唯一性证明的两步都漂亮：第一步「$a^q = a$ 对一切 $a$」来自乘法群 $F^\times$ 的阶是 $q-1$（拉格朗日定理）；第二步「$x^q - x$ 有 $q$ 个不同的根」用导数 $(x^q-x)' = qx^{q-1} - 1 = -1 \ne 0$（特征 $p$ 里 $q = p^n = 0$），无重根。于是有限域 = $x^q - x$ 的分裂域，唯一性由分裂域唯一性给出。<strong>「$x^q - x$ 的分裂域」是有限域的标准实现。</strong></span>

**推论：** $\mathbb{F}_{p^n}$ 是 $\mathbb{F}_p$ 上 $x^{p^n} - x$ 的分裂域。$\mathbb{F}_{p^n} = \{ \text{$x^{p^n} - x$ 的全部根} \}$。

## 4 结构：子域格与 Galois 群

**定理（子域与 Galois 群）：** 设 $n = dm$。

1. **$\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n}$ ⟺ $d \mid n$**（子域按整除排格）；
2. **$\operatorname{Gal}(\mathbb{F}_{p^n} / \mathbb{F}_p) \cong \mathbb{Z}_n$**（循环群），由 **Frobenius 自同构** $\sigma(a) = a^p$ 生成。

**证明（子域条件）：** 若 $\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n}$，则 $[\mathbb{F}_{p^n} : \mathbb{F}_p] = [\mathbb{F}_{p^n} : \mathbb{F}_{p^d}] \cdot [\mathbb{F}_{p^d} : \mathbb{F}_p]$，故 $n = (\text{整数}) \cdot d$，$d \mid n$。反之若 $d \mid n$，$\mathbb{F}_{p^n}$ 中满足 $a^{p^d} = a$ 的元素构成 $p^d$ 阶子域 $\mathbb{F}_{p^d}$（$x^{p^d} - x$ 的根）。$\blacksquare$<span class="marginnote">子域条件 $d \mid n$ 让有限域的子域格与「$n$ 的因子格」同构：$\mathbb{F}_{p^{12}}$ 的子域是 $\mathbb{F}_{p^d}$（$d \mid 12$，即 $d = 1, 2, 3, 4, 6, 12$）。这条整除性规则在密码学（双线性配对、有限域算术）里反复使用——选择域的大小必须让安全子域合适。</span>

**证明（Galois 群）：** $\sigma(a) = a^p$ 是 $\mathbb{F}_{p^n}$ 保持 $\mathbb{F}_p$ 的自同构（Frobenius）。$\sigma^n(a) = a^{p^n} = a$（$a^{q} = a$），故 $\sigma^n = \mathrm{id}$；而 $\sigma^k = \mathrm{id}$ 要求 $a^{p^k} = a$ 对一切 $a$，即 $x^{p^k} - x$ 有全部 $p^n$ 个根，$p^k \ge p^n$，$k \ge n$——$\sigma$ 的阶恰为 $n$。故 $\langle \sigma \rangle$ 是 $\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p)$ 的 $n$ 阶子群，而 Galois 群的大小 $= [\mathbb{F}_{p^n} : \mathbb{F}_p] = n$，所以 $\operatorname{Gal} \cong \langle \sigma \rangle \cong \mathbb{Z}_n$。$\blacksquare$<span class="marginnote">有限域的 Galois 群是循环群 $\mathbb{Z}_n$，由 Frobenius 生成。这是「Galois 群 = 分裂域对称」的最干净例子：$\mathbb{F}_{p^n}$ 的对称只有「$p$-次幂置换」反复迭代这 $n$ 种。对密码学，这意味着有限域上「没有太多隐藏对称」，安全性建立在循环群的结构上（离散对数问题）。</span>

## 5 公式解析：x^q - x 在 F_q 中完全分裂

「$\mathbb{F}_q = x^q - x$ 的分裂域」是有限域一切定理的枢纽，拆透它。

- **第一步，乘法群的阶。** $|F^\times| = q - 1$（去掉 0），拉格朗日给出 $a^{q-1} = 1$ 对一切 $a \ne 0$。

- **第二步，翻译成方程。** $a^{q-1} = 1$ 两边乘 $a$：$a^q = a$ 对一切 $a \ne 0$；$a = 0$ 也满足。**$F$ 的每个元素都是 $x^q - x$ 的根。**

- **第三步，根的数目。** $x^q - x$ 次数 $q$，至多 $q$ 个根；$F$ 有 $q$ 个元素全是根，故恰好 $q$ 个根且各不相同（$(x^q - x)' = -1 \ne 0$ 保证无重根）。

- **第四步，分裂域结论。** $x^q - x = \prod_{a \in F}(x-a)$ 在 $F$ 中完全分裂，$F$ 由根生成（根就是全部元素），$F$ 是分裂域。**唯一性随之而来**——任何 $q$ 阶域都「是」这个分裂域。

## 6 对照速查：三个最小有限域的算术

把抽象的定理落进三个具体的最小有限域，感受「存在性—唯一性—结构」如何变成可以手算的表格。

| 域 | 素域 | 不可约多项式 | 元素 | 加法群 | 乘法群 |
| --- | --- | --- | --- | --- | --- |
| $\mathbb{F}_2$ | 自身 | — | $\{0,1\}$ | $\mathbb{Z}_2$ | 平凡群 $\{1\}$（阶 1） |
| $\mathbb{F}_4$ | $\mathbb{F}_2$ | $x^2+x+1$ | $\{0,1,\alpha,\alpha+1\}$ | $\mathbb{Z}_2^2$ | 循环群 $\mathbb{Z}_3$ |
| $\mathbb{F}_8$ | $\mathbb{F}_2$ | $x^3+x+1$ | 8 个元素 | $\mathbb{Z}_2^3$ | 循环群 $\mathbb{Z}_7$ |

以 $\mathbb{F}_4$ 为例把乘法算透。特征 2 里减号即加号，$\alpha^2 = \alpha + 1$，于是 $\alpha^3 = \alpha(\alpha+1) = \alpha^2 + \alpha = (\alpha+1)+\alpha = 1$。<span class="marginnote">$\alpha$ 的阶恰为 3，生成整个 $\mathbb{F}_4^\times = \{1,\alpha,\alpha+1\}$，而 $3 = q - 1$。一般地 $\mathbb{F}_q^\times$ 是 $q-1$ 阶循环群——<strong>有限域的乘法群是循环群</strong>，记为 $\mathbb{F}_q^\times \cong \mathbb{Z}_{q-1}$，这是有限域最重要的结构性事实，也是密码学里离散对数问题的舞台。</span>

**数值核对**：$\mathbb{F}_4$ 中 $(\alpha+1)^2 = \alpha^2 + 2\alpha + 1 = \alpha^2 + 1 = \alpha$（特征 2），而 $\alpha^{-1} = \alpha^2$（因为 $\alpha \cdot \alpha^2 = 1$）。每个非零元素都有逆元，域的公理逐条可验。

**用加法表再验一次**：$\mathbb{F}_4$ 的加法是逐位异或：$1 + \alpha = \alpha+1$，$\alpha + (\alpha+1) = 1$，$(\alpha+1)+(\alpha+1) = 0$。这些关系完全由特征 2（$x + x = 0$）与 $\alpha^2 = \alpha+1$ 决定，没有第二种可能——这正是「有限域的算术由素域特征与不可约多项式唯一确定」的直观版本。

**子域校验**：按第四节子域条件 $d \mid n$，$\mathbb{F}_2 \subseteq \mathbb{F}_4$（$1 \mid 2$）成立，而 $\mathbb{F}_4 \subseteq \mathbb{F}_8$ 不成立（$2 \nmid 3$）。$\mathbb{F}_4$ 中满足 $a^2 = a$ 的元素恰是 $0$ 与 $1$——正是子域 $\mathbb{F}_2$。用「$x^{p^d} - x$ 的根」定位子域，第四节定理就变成可手算的操作。

**易错辨析｜加法群 vs 乘法群。** $\mathbb{F}_4$ 的加法群 $\cong \mathbb{Z}_2 \times \mathbb{Z}_2$（克莱因四元群，非循环），乘法群 $\cong \mathbb{Z}_3$（循环）。**同一批元素配上不同运算，就得到结构迥异的两个群**——「代数结构由运算决定」在此一目了然。误以为「$\mathbb{F}_4^\times$ 有 4 个元素」是高频错误：$\mathbb{F}_4$ 总共 4 个元素，去掉 0 的乘法群只有 3 阶。<span class="marginnote">把「域的阶」与「乘法群的阶」分开记：$|\mathbb{F}_{p^n}| = p^n$，而 $|\mathbb{F}_{p^n}^\times| = p^n - 1$。差出来的这 1 个，是零元。</span>

**规模往上的规律**：$\mathbb{F}_{2^8}$（AES 的字节域）乘法群阶 $255 = 3 \times 5 \times 17$，仍是循环群，但比 $\mathbb{F}_4$、$\mathbb{F}_8$ 复杂得多。循环群理论保证本原元存在，具体找它却要靠试算——「存在性由定理保证，构造仍需动手」，这正是第一节计数证明的余韵。

**一句话总结**：有限域由一对数 $(p, n)$ 完全标记——素域 $\mathbb{F}_p$ 定特征、扩张次数 $n$ 定阶；加法是向量空间、乘法是循环群、对称是 $\mathbb{Z}_n$，三者拼出有限域的全部面孔。

**把三个例子串起来**：$\mathbb{F}_2$（1 位）、$\mathbb{F}_4$（2 位）、$\mathbb{F}_8$（3 位）是字节域 $\mathbb{F}_{2^8}$ 的前三个台阶。每一级都满足：阶是素数幂、乘法群循环、子域按整除排格。把 $\mathbb{F}_4$ 的手算练熟，$\mathbb{F}_{2^8}$ 的一切就只是规模放大。

**回归主题**：为什么 AES 选 $\mathbb{F}_{2^8}$ 而不是「模 256 算术」？因为 $256 = 2^8$ 是素数幂，$\mathbb{F}_{2^8}$ 是域、每个非零字节都可逆；而 $\mathbb{Z}_{256}$ 有零因子（偶数不可逆）。「阶是否为素数幂」决定了结构的名字是域还是环。

## 7 小结

- **有限域的阶必为素数幂** $p^n$；有限域 = $\mathbb{F}_p$ 的有限扩张。
- **存在性**：$\mathbb{F}_p$ 上有任意次不可约多项式；$\mathbb{F}_q = \mathbb{F}_p[x]/\langle f\rangle$。
- **唯一性**：$\mathbb{F}_q$ 是 $x^q - x$ 的分裂域；任何 $q$ 阶域同构。
- **结构**：$\mathbb{F}_{p^d} \subseteq \mathbb{F}_{p^n} \iff d \mid n$；$\operatorname{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p) \cong \mathbb{Z}_n$ 由 Frobenius 生成。
- 有限域是 AES（$\mathbb{F}_{2^8}$）、椭圆曲线（$\mathbb{F}_p$）、编码理论（Reed-Solomon）的共同舞台。

在下一节，我们进入第十一篇 Galois 理论：**域的自同构与 Galois 群**。有限域的 Galois 群是循环群，而一般分裂域的 Galois 群将决定方程能否根式求解。
