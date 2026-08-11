---
title: Dirichlet 单位定理
date: 2026-08-11
---

# Dirichlet 单位定理

<div class="epigraph">
<p>上帝创造了整数，其余一切都是人的作品。</p>
<footer>—— 利奥波德 · 克罗内克（Leopold Kronecker，God made the integers, all the rest is the work of man）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从单位理论开始

数环 $\mathcal{O}_K$ 里有一类元素看似普通却至关重要——**单位**：可逆元。在 $\mathbb{Z}$ 里单位只有 $\pm 1$，平淡无奇；但在 $\mathbb{Q}(\sqrt{2})$ 里，$1 + \sqrt{2}$ 满足 $\mathrm{N}(1+\sqrt{2}) = 1 - 2 = -1$，于是它是单位，而且 $(1+\sqrt{2})^k$ 全是单位。佩尔方程 $x^2 - 2y^2 = \pm 1$ 的**全部整数解**，本质上就是这些单位的全体。Dirichlet 单位定理一举刻画了所有数域的单位群结构，把「找方程整数解」变成「看清一个格的形状」。

## 1 嵌入与 $r_1, r_2$

先搭一个工具：$K$ 到 $\mathbb{C}$ 的 $\mathbb{Q}$-嵌入。设 $K$ 是数域，$[K:\mathbb{Q}] = n$。共有 $n$ 个嵌入 $\sigma: K \hookrightarrow \mathbb{C}$，其中：

- $r_1$ 个是**实嵌入**（像落在 $\mathbb{R}$ 里）；
- 其余 $2r_2$ 个成对共轭（互为复共轭的**复嵌入**），所以 $n = r_1 + 2r_2$。

**例**：$\mathbb{Q}$：$r_1 = 1, r_2 = 0$；$\mathbb{Q}(\sqrt{2})$：两个实嵌入 $\sqrt{2} \mapsto \pm\sqrt{2}$，故 $r_1 = 2, r_2 = 0$；$\mathbb{Q}(i)$：嵌入 $i \mapsto \pm i$，$r_1 = 0, r_2 = 1$。<span class="marginnote">$r_1, r_2$ 是数域的「签名（signature）」，Minkowski 界里的 $r_2$ 就是它。直觉：实嵌入多，域「贴着实数轴」，单位可以又大又正负交错；纯虚的域单位群反而很小（只能是单位根）。</span>

**对数嵌入（logarithmic embedding）**：把 $\alpha \in K^\times$ 映到

$$
\ell(\alpha) = \big(\log|\sigma_1(\alpha)|, \dots, \log|\sigma_{r_1}(\alpha)|, 2\log|\sigma_{r_1+1}(\alpha)|, \dots, 2\log|\sigma_{r_1+r_2}(\alpha)|\big) \in \mathbb{R}^{r_1+r_2}
$$

关键性质：对任意 $\alpha$，各分量之和 $\sum \ell(\alpha)_i = \log|\mathrm{N}(\alpha)|$。因此**单位**（范 $= \pm 1$）的像落在超平面 $H = \{(x_i) : \sum x_i = 0\}$ 中，$H \cong \mathbb{R}^{r_1 + r_2 - 1}$。

## 2 单位定理的陈述

**Dirichlet 单位定理：**

$$
\mathcal{O}_K^\times \;\cong\; \mu(K) \times \mathbb{Z}^{\,r}, \qquad r = r_1 + r_2 - 1
$$

其中 $\mu(K)$ 是 $K$ 中单位根（**挠元**）构成的有限循环群。也就是说，**除去有限个单位根，单位群是一个秩为 $r$ 的自由阿贝尔群**。<span class="marginnote">定理的证明思路：对数嵌入把 $\mathcal{O}_K^\times$ 送入 $H$ 成为一个<strong>格</strong>（离散、秩 $r$ 的 $\mathbb{Z}$-子模），核正好是单位根。于是「单位群的无限生成部分」被还原为「$H$ 里的一个格子」——把数论问题翻译成格的几何，这是 Minkowski 思想的又一次胜利。</span>

**秩公式** $r = r_1 + r_2 - 1$ 的三个典型案例：

| $K$ | $r_1$ | $r_2$ | 单位群 |
| --- | --- | --- | --- |
| $\mathbb{Q}(\sqrt{2})$ | 2 | 0 | $\{\pm 1\} \times (1+\sqrt{2})^{\mathbb{Z}}$（佩尔解） |
| $\mathbb{Q}(i)$ | 0 | 1 | $\{\pm 1, \pm i\}$，$r = 0$（有限） |
| $\mathbb{Q}(\sqrt{-3})$ | 0 | 1 | $\mu_6$（$e^{i\pi/3}$ 的幂），$r = 0$ |

**辨析｜易错点：** 秩 $r = 0$（纯虚二次域）时单位群是**有限**的，只有单位根；秩 $r \ge 1$ 时单位群是**无限**的。千万别以为「有复数嵌入就有无限单位」——单位根的数量由范 $= \pm 1$ 的元素个数决定，虚二次域的 $\mathcal{O}_K^\times$ 永远有限。**$\mu(K)$ 的阶 $w_K$ 在类数公式里是重要的除数**，不要忘掉它。

## 3 基本单位与佩尔方程

秩为 1 的情形最有名：$K = \mathbb{Q}(\sqrt{d})$，$d > 0$ 平方自由。此时 $r = 1$，单位群 = $\{\pm 1\} \times \varepsilon^{\mathbb{Z}}$，其中 $\varepsilon > 1$ 是最小的大于 $1$ 的单位，称为**基本单位（fundamental unit）**。

**佩尔方程 $x^2 - d y^2 = \pm 1$** 的解 $x + y\sqrt{d}$ 恰好就是 $\mathcal{O}_K$ 的单位（$\mathrm{N}(x + y\sqrt{d}) = x^2 - d y^2$）。于是

$$
\text{佩尔方程的所有解} \;=\; \pm \varepsilon^k, \quad k \in \mathbb{Z}
$$

**例**：$d = 2$ 时基本单位 $\varepsilon = 1 + \sqrt{2}$，$x^2 - 2y^2 = \pm 1$ 的全部正解由 $(1+\sqrt{2})^k$ 给出：$k = 1$ 得 $1^2 - 2\cdot 1^2 = -1$；$k = 2$ 得 $3^2 - 2\cdot 2^2 = 1$；$k = 3$ 得 $7^2 - 2\cdot 5^2 = -1$……<span class="marginnote">$d = 2$ 的基本单位来自 $(1+\sqrt2)^2 = 3 + 2\sqrt2$ 一族。找基本单位没有万能公式（连分数、类数逼近都要上场），但 Dirichlet 定理保证它<strong>一定存在</strong>——存在性比构造容易得多，这在数学里很常见。</span>

**调整数（regulator）$R_K$**：$\mathbb{R}$ 中格的协体积，即 $\det$ 在对数嵌入像的基下的体积。它是类数公式里的又一个关键量：$r = 1$ 时 $R_K = \log \varepsilon$，$r = 0$ 时约定 $R_K = 1$。

**分圆域的单位秩**：$K = \mathbb{Q}(\zeta_n)$（$n > 2$）时 $r_1 = 0$、$r_2 = \varphi(n)/2$，故

$$
r = \frac{\varphi(n)}{2} - 1
$$

对 $n = 3$（即 $\mathbb{Q}(\omega)$）得 $r = 0$——单位群只有单位根；对 $n = 5$ 得 $r = 1$——存在一个「第一基本单位」$1 - \zeta_5$ 之类的生成元。**单位秩从纯组合数 $\varphi(n)$ 自动读出**，这正是定理的普适性。

## 4 公式解析：秩公式 $r = r_1 + r_2 - 1$

$$
\underbrace{r}_{单位群秩} = \underbrace{r_1 + r_2}_{嵌入空间维数} \; - \; 1
$$

两步讲透这条「减一」：

- **第一步，为什么是 $r_1 + r_2$**：对数嵌入把单位送到 $\mathbb{R}^{r_1+r_2}$，这里复共轭对共用一维（取了模长 $\times 2$）。单位群的格就活在这个嵌入空间里。
- **第二步，为什么减 $1$**：单位必须满足范 $= \pm 1$，即各分量对数之和为 $0$——它们全部落在超平面 $H: \sum x_i = 0$ 上，$H$ 的维数是 $r_1 + r_2 - 1$。格的秩被这条约束压掉一维。

**直觉版**：单位是「乘法结构」里无限生成的骨架，它的自由度等于「可以自由伸缩的对数坐标方向」数目；而乘积为 $1$（范的约束）恰好把其中一个方向钉死。剩下的每个自由度，都对应一个无限循环生成元 $\varepsilon$。

## 5 补算：基本单位与调整数

**基本单位计算**：实二次域 $K = \mathbb{Q}(\sqrt{d})$ 时 $r = 1$，基本单位 $\varepsilon$ 是最小的大于 $1$ 的单位：

| $d$ | 基本单位 $\varepsilon$ | $\mathrm{N}(\varepsilon)$ | 备注 |
| --- | --- | --- | --- |
| $2$ | $1 + \sqrt{2}$ | $-1$ | 佩尔 $x^2 - 2y^2 = \pm 1$ 的种子 |
| $3$ | $2 + \sqrt{3}$ | $1$ | $x^2 - 3y^2 = -1$ 无解（模 $3$ 排除） |
| $5$ | $\frac{1+\sqrt{5}}{2}$ | $-1$ | **黄金比例本身就是单位** |
| $6$ | $5 + 2\sqrt{6}$ | $1$ | $(5+2\sqrt6)(5-2\sqrt6) = 1$ |

**调整数直算**：$K = \mathbb{Q}(\sqrt{2})$ 时对数嵌入 $\ell(1+\sqrt2) = (\log(1+\sqrt2), \log|1-\sqrt2|) = (\log(1+\sqrt2), -\log(1+\sqrt2))$，它的「长度」（协体积）是

$$
R_K = \log(1+\sqrt{2}) \approx 0.881
$$

这就是单位格的体积——类数公式分子里的那个量。<span class="marginnote">找基本单位没有万能公式，但连分数很管用：$\sqrt{d}$ 的连分数周期长度为 $1$ 当且仅当 $d = a^2+1$（此时 $\varepsilon = a + \sqrt{d}$）；一般情形用周期中途的近似分数逼近。这个「存在但难构造」的张力，正是 Dirichlet 定理的存在性之美。</span>

**辨析｜易错点：** 基本单位可能范 $= -1$（如 $1+\sqrt2$）也可能范 $= 1$（如 $2+\sqrt3$）。「$x^2 - dy^2 = -1$ 是否可解」是判定范 $=-1$ 单位是否存在的一维问题（Nagetell–Rabinowitsch 情形的退化版）。**千万别默认基本单位一定范 $1$**——$d$ 的小变化就能翻转范的符号。

## 6 小结

- 数域签名 $(r_1, r_2)$：$n = r_1 + 2r_2$，决定一切「嵌入类」不变量。
- **Dirichlet 单位定理**：$\mathcal{O}_K^\times \cong \mu(K) \times \mathbb{Z}^{r}$，$r = r_1 + r_2 - 1$。
- 秩 $r$：纯虚二次域 $r = 0$（单位有限）；实二次域 $r = 1$（佩尔方程基本单位）。
- 对数嵌入把单位群变成 $H$ 中的格；**调整数 $R_K$** 是格的体积，未来类数公式的分子。

在下一节，我们把「每个数域的算术」放到一个全新的视野里看——为每个素数配备一把「尺子」$|\cdot|_p$，这就是**赋值与 $p$-adic 数**：把有理数完备化成 $p$-adic 数域，让数论从「全局」进入「局部」。
