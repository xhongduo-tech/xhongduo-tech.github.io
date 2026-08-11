---
title: Dedekind Zeta 函数与类数公式
date: 2026-08-11
---

# Dedekind Zeta 函数与类数公式

<div class="epigraph">
<p>317 是素数，不是因为我们认为它是，而是因为数学实在就是这样构成的。</p>
<footer>—— 戈弗雷 · 哈罗德 · 哈代（G. H. Hardy，317 is a prime, not because we think so, or because our minds are shaped in one way or another, but because it is, because mathematical reality is built that way）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Dedekind Zeta 函数开始

解析数论与代数数论的第一次深吻，是 Riemann zeta 函数与数域的相遇。**Dedekind zeta 函数** $\zeta_K(s)$ 把素数乘积的 Euler 积推广到「理想」层面，并神奇地把整个专题前十一节的几个不变量——类数 $h_K$、调整数 $R_K$、单位根数 $w_K$、判别式 $d_K$、签名 $(r_1, r_2)$——**全部缝进 $s = 1$ 处的一个留数**。这个**解析类数公式（analytic class number formula）**是代数数论最深的「结算中心」：算术量的整体结构，被写成一条单点残差的等式。<span class="marginnote">这类「$L$-函数特殊值 = 算术不变量」的模式，正是 Langlands 纲领与 BSD 猜想（椭圆曲线）的祖先——你在这里看到的公式结构，会在《椭圆曲线》专题以更微妙的形式重现。</span>

## 1 定义与 Euler 积

**Dedekind zeta 函数**：对 $\mathrm{Re}(s) > 1$，令 $\mathfrak{a}$ 跑遍 $\mathcal{O}_K$ 的非零整理想：

$$
\zeta_K(s) = \sum_{\mathfrak{a}} \frac{1}{\mathrm{N}(\mathfrak{a})^{s}}
$$

由于每个整理想唯一分解为素理想之积，$\mathrm{N}$ 完全可乘，得 **Euler 积**：

$$
\zeta_K(s) = \prod_{\mathfrak{p}} \left(1 - \mathrm{N}(\mathfrak{p})^{-s}\right)^{-1}, \qquad \mathrm{Re}(s) > 1
$$

$\mathfrak{p}$ 跑遍素理想。<span class="marginnote">当 $K = \mathbb{Q}$，素理想就是素数，$\zeta_{\mathbb{Q}}(s)$ 恰为 Riemann zeta $\zeta(s)$——Dedekind zeta 是其全数域版的推广。收敛域 $\mathrm{Re}(s) > 1$ 与 $\sum \mathrm{N}(\mathfrak{p})^{-s} < \infty$（质数密度）一致。</span>

**例（二次域）**：$K = \mathbb{Q}(\sqrt{d})$（$d$ 平方自由、$d \equiv 1 \bmod 4$，取 $K$ 的判别式 $d_K$）。素理想按素数 $p$ 的分裂方式分组：

$$
\zeta_K(s) = \zeta(s)\cdot L(s, \chi_{d_K}), \qquad L(s,\chi) = \sum_{n=1}^{\infty}\frac{\chi_{d_K}(n)}{n^{s}}
$$

$\chi_{d_K}$ 是 Kronecker 符号 $(\frac{d_K}{\cdot})$（判别式特征标）。**二次域的 Dedekind zeta 是 Riemann zeta 与一个 Dirichlet L-函数的乘积**——最简的非平凡可分解案例。

## 2 解析延拓与函数方程

**定理（Hecke）：** $\zeta_K(s)$ 解析延拓为全平面上的亚纯函数，唯一的极点在 $s = 1$，且是**单极点**。它满足**函数方程**。定义完备化

$$
\Lambda_K(s) = |d_K|^{\frac{s}{2}}\left(\frac{\Gamma(s/2)}{\pi^{s/2}}\right)^{r_1}\left(\frac{\Gamma(s)}{(2\pi)^s}\right)^{r_2} \zeta_K(s)
$$

则 $\Lambda_K(s) = \Lambda_K(1 - s)$，且 $\Lambda_K$ 在 $s = 0, 1$ 处有单极点，其余为整函数。<span class="marginnote">这是 Riemann 函数方程 $\Lambda(s) = \Lambda(1-s)$ 的全数域版，$r_1, r_2$ 通过 $\Gamma$ 因子入场——签名再次决定「分析形状」。函数方程的意义远超技巧：它把 $\zeta_K$ 在 $s$ 与 $1-s$ 两侧的行为互锁，是「素数分布与 $\zeta$ 零点」这一经典套路的前提。</span>

**零点与素数分布**：$\zeta_K(s)$ 的平凡零点（在负整数与 $s = 0$）对应分歧与 $\Gamma$ 因子；非平凡零点所在临界带 $\mathrm{Re}(s) = \frac12$ 上的**广义 Riemann 猜想（GRH）**是最著名的开放问题。素数定理的数域版由「$\zeta_K$ 无 $s = 1$ 以外的 $1$-带零点」推出——Riemann 论证的忠实复刻。

## 3 解析类数公式

**定理（解析类数公式）：** $\zeta_K(s)$ 在 $s = 1$ 处的留数为

$$
\boxed{\;\lim_{s \to 1}(s-1)\, \zeta_K(s) \;=\; \frac{2^{r_1}(2\pi)^{r_2}\, h_K\, R_K}{w_K \sqrt{|d_K|}}\;}
$$

其中 $h_K$ = 类数，$R_K$ = 调整数，$w_K$ = $K$ 中单位根个数，$r_1, r_2$ = 签名，$d_K$ = 判别式。

**这条公式把前面所有「几何—算术」不变量串成一条链**：类群的大小 $h_K$、单位格子的体积 $R_K$、有限挠 $w_K$、格子的总体积 $\sqrt{|d_K|}$，全部决定 $\zeta_K$ 在极点处的「质量」。<span class="marginnote">反过来说：只要能<strong>解析地</strong>算出留数（比如通过 $L$-函数特殊值、正则化行列式、Tate 方法），就能读出类数——对很多域，这是得到 $h_K$ 的唯一可行途径。类数公式因此既是理论上的结算中心，也是实际计算的工具箱。</span>

**例（虚二次域）**：$K = \mathbb{Q}(\sqrt{-d})$，$d > 0$。此时 $r_1 = 0, r_2 = 1, w_K = 2$（或 4、6 在特殊情形），$R_K = 1$，公式化为

$$
\lim_{s\to 1}(s-1)\zeta_K(s) = \frac{2\pi\, h_K}{w_K \sqrt{|d_K|}}
$$

而 $\zeta_K(s) = \zeta(s)L(s, \chi_{d_K})$，用 $L(1, \chi) = \frac{2\pi}{\sqrt{|d_K|}}\,h_K/w_K \cdot (\dots)$ 之类的关系即可把类数从 $L$-函数特殊值读出。Dirichlet 早在 1839 年用这个公式**算出了虚二次域类数**——比类域论完整成形早了一个世纪。

## 4 公式解析：留数公式里的每一项

$$
\lim_{s\to 1}(s-1)\zeta_K(s) = \frac{2^{r_1}(2\pi)^{r_2}\, h_K\, R_K}{w_K \sqrt{|d_K|}}
$$

- **第一步，$2^{r_1}(2\pi)^{r_2}$**：来自 $\Gamma$ 因子的残差（$\Gamma$ 在 $0$ 处的行为与 $\Lambda_K$ 的规范化），本质是「实嵌入与复嵌入的数量权重」——与函数方程里的 $\Gamma$ 因子一脉相承。
- **第二步，分子 $h_K R_K$**：$h_K$ 是「理想偏离主理想的量」，$R_K$ 是「单位格的体积」。两者都来自**算术结构**；留数定理把 zeta 的极点强度分解为「类群 × 单位格」——这就是「解析/算术」对偶的第一次显形。
- **第三步，分母 $w_K \sqrt{|d_K|}$**：$w_K$ 计数「挠单位」（零秩自由部分的除数），$\sqrt{|d_K|}$ 是格子的协体积（Minkowski 嵌入）。分母越大，留数越小——几何体积越大、算术量越被「摊薄」。
- **第四步，取极限的意义**：$s \to 1$ 时 $\zeta_K(s) \sim \frac{\mathrm{res}}{s-1}$。**素数在数域的 Euler 积在 $s=1$ 发散的「速率」，精确等于这个算术组合**——类数公式是「分析发散强度 = 算术代数不变量」的教科书式等式。

## 5 从 zeta 到 Artin L-函数

$\zeta_K$ 只是解析算术的第一层。对 Galois 扩张 $L/K$，**Artin L-函数**

$$
L(s, \rho) = \prod_{\mathfrak{p}} \det\left(1 - \rho(\mathrm{Frob}(\mathfrak{p}))\,\mathrm{N}(\mathfrak{p})^{-s}\Big| V^{I_{\mathfrak{p}}}\right)^{-1}
$$

用表示 $\rho$ 和惯性群不动子空间把每个素理想的分歧也编码进去，并满足分解律 $\zeta_L(s) = \prod_{\rho} L(s,\rho)^{\dim \rho}$。<span class="marginnote">这正是 Chebotarev 定理证明里用到的分解——$L(s,\rho)$ 的正则性与特殊值决定素数的分布，而它们的特殊值（像 $\zeta_K$ 的留数那样）编码更深的不变量。<strong>Artin 互反律 ⟹ $L(s, \chi)$ 自守</strong>，这条链通向 Langlands。</span>

**递推**：数域的分歧、惯性、单位、类数被 zeta 编码，而 zeta 的零点分布又反推素数的行为——**代数不变量与分析函数的双向锁**。BSD 猜想（椭圆曲线 $L$-函数在 $s=1$ 的零点阶 = 秩）是同一模式的当代化身：特殊值几何化。

## 6 实例：Euler 积与留数的核算

**例 1（Euler 积按分裂分组）**：$K = \mathbb{Q}(\sqrt{5})$（$d_K = 5$）。对有理素数 $p$：$p = 5$ 分歧（一个素理想，范 $5$）；$(\frac5p) = 1$ 分裂（两个素理想，各范 $p$）；$(\frac5p) = -1$ 惯性（一个素理想，范 $p^2$）。于是

$$
\zeta_{\mathbb{Q}(\sqrt5)}(s) = \frac{1}{1-5^{-s}} \prod_{(\frac5p)=1}\frac{1}{(1-p^{-s})^2} \prod_{(\frac5p)=-1}\frac{1}{(1-p^{-2s})} = \zeta(s)\,L(s,\chi_5)
$$

其中 $\chi_5 = (\frac5{\cdot})$——这就是「二次域的 Dedekind zeta 分解」的显式写法。

**例 2（类数公式核算）**：$K = \mathbb{Q}(\sqrt{-5})$：$r_1 = 0, r_2 = 1, h_K = 2, R_K = 1, w_K = 2, |d_K| = 20$，

$$
\lim_{s\to1}(s-1)\zeta_K(s) = \frac{2\pi \cdot 2 \cdot 1}{2\sqrt{20}} = \frac{2\pi}{\sqrt{20}} = \frac{\pi}{\sqrt5} \approx 1.405
$$

与 $\zeta(s)L(s, \chi_{-20})$ 在 $s = 1$ 的留数核对一致——四个算术量（$h, R, w, d$）在一个极限里团圆。<span class="marginnote">类数公式之所以重要，正是因为它把<strong>看似互不相干</strong>的量绑在一起：类数是代数、调整数是几何、单位根数是组合、判别式是格体积——而它们共同决定 $\zeta_K$ 的极点强度。对现代计算，Tate 的正则化方法让留数可解析算出，从而反推类数。</span>

**辨析｜易错点：** $\zeta_K$ 的零点分「平凡」（来自 $\Gamma$ 因子与分歧，在负整数与 $s=0$）与「非平凡」（临界带内）两类。**类数公式只与 $s=1$ 的留数有关，与零点分布无关**；但「素数分布的精细结果」靠零点的无零区。别把「$s=1$ 的单极点」与「$s=0$ 的平凡零点」混记。

## 7 小结

- **定义** $\zeta_K(s) = \sum_{\mathfrak{a}} \mathrm{N}(\mathfrak{a})^{-s}$（$\mathrm{Re}\,s > 1$），Euler 积 $\prod_{\mathfrak{p}}(1-\mathrm{N}(\mathfrak{p})^{-s})^{-1}$；$K = \mathbb{Q}$ 退化为 Riemann $\zeta$。
- 二次域：$\zeta_K = \zeta \cdot L(s, \chi_{d_K})$——可分解的首例。
- **Hecke**：$\zeta_K$ 亚纯延拓、唯一单极点 $s=1$；函数方程 $\Lambda_K(s) = \Lambda_K(1-s)$；GRH 关乎临界带零点。
- **解析类数公式**：$\mathrm{res}_{s=1}\zeta_K = \dfrac{2^{r_1}(2\pi)^{r_2} h_K R_K}{w_K \sqrt{|d_K|}}$。
- Artin L-函数与 $\zeta_L = \prod L(s,\rho)^{\dim\rho}$：通向 Langlands 与 BSD 的分析主线。

在下一节，我们转向最后一个经典工具——**Gauss 和与 Jacobi 和**：它们把分圆根的和式变成「有限域上的 Fourier 分析」，用代数信息精确数出 $x^n + y^n = 1$ 在模 $p$ 下的解数，为椭圆曲线与密码学埋下伏笔。
