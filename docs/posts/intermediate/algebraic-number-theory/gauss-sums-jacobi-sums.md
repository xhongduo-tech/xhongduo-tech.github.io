---
title: Gauss 和与 Jacobi 和
date: 2026-08-11
---

# Gauss 和与 Jacobi 和

<div class="epigraph">
<p>读读欧拉吧，读读欧拉，他是我们所有人的老师。</p>
<footer>—— 皮埃尔-西蒙 · 拉普拉斯（Pierre-Simon Laplace，Read Euler, read Euler, he is the master of us all）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Gauss 和与 Jacobi 和开始

分圆单位根 $\zeta = e^{2\pi i/p}$ 是一把「模 $p$ 的频谱分析仪」：把 $\mathbb{F}_p$ 上的函数变成它的离散 Fourier 系数，同余方程的解数就可以被精确数出来。**Gauss 和**给一个 Dirichlet 特征标做「Fourier 系数」，**Jacobi 和**是两个特征标的卷积——它们精确给出 $x^n + y^n = 1 \pmod p$ 的解数，也给出高斯对二次互反律最漂亮的证明之一。<span class="marginnote">这是本专题的收官工具：前面的 zeta 函数、类数公式都是「长程」解析，而 Gauss/Jacobi 和是「短程」组合——它们用最直白的方法，把素数的对称性从函数方程里「硬数」出来。费马大定理 $n = 3$、$p \equiv 1 \bmod 3$ 的素数可分解 $a^2 + 3b^2$，都藏在 Jacobi 和里。</span>

## 1 Dirichlet 特征标与 Gauss 和

**Dirichlet 特征标模 $p$（Dirichlet character mod $p$）**：乘法同态 $\chi: (\mathbb{Z}/p\mathbb{Z})^\times \to \mathbb{C}^\times$，扩展到 $\mathbb{F}_p$：$\chi(0) = 0$。它的值落在一组单位根中，$\chi(-1) = \pm 1$ 区分奇偶。**平凡**特征标 $\varepsilon$（恒为 $1$）、**二次**特征标 $\chi_p$（勒让德符号）是最常用的两个。

**Gauss 和（Gauss sum）**：设 $\zeta_p = e^{2\pi i/p}$，

$$
g(\chi) = \sum_{a=0}^{p-1} \chi(a)\,\zeta_p^{a} = \sum_{a=1}^{p-1} \chi(a)\,\zeta_p^{a}
$$

它把特征标「投影」到 $p$ 次单位根上。<span class="marginnote">Gauss 和出现在：二次互反律证明、解数计算、类数公式、以及「$p$ 的可分解性」判定里。高斯的灵感源于他 17 岁发现 $\sqrt{p}$ 与单位根和的关系——$g(\chi_p)$ 的平方恰好是他要的对称性。</span>

**核心绝对值公式**：对非平凡 $\chi$，

$$
|g(\chi)|^2 = p, \qquad g(\chi)\,g(\bar\chi) = \chi(-1)\,p
$$

对二次特征标 $\chi_p$，$g(\chi_p)^2 = \chi_p(-1)\,p = (-1)^{\frac{p-1}{2}}p$——**高斯当年算出的「$\sqrt{\pm p}$」就藏在单位根的和里**。这给出二次互反律的一个经典证明路径：$g(\chi_p)$ 的共轭作用搭桥 $p$ 与 $q$ 两侧。

## 2 Jacobi 和与卷积公式

**Jacobi 和（Jacobi sum）**：对非平凡特征标 $\chi, \psi$（$\chi\psi$ 也非平凡）：

$$
J(\chi, \psi) = \sum_{a=0}^{p-1} \chi(a)\,\psi(1-a)
$$

**乘积公式**（把卷积转成乘法）：

$$
J(\chi, \psi) = \frac{g(\chi)\, g(\psi)}{g(\chi\psi)}
$$

由此立得 $|J(\chi,\psi)| = \sqrt p$（两非平凡且 $\chi\psi$ 非平凡时）。<span class="marginnote">这个公式是「有限域上的 Fourier 恒等式」：特征标的卷积的 Fourier 系数 = 各自 Fourier 系数之积再归一。它把两个看似独立对象的「交互」化简为三次 Gauss 和的除法——结构一目了然。</span>

**例**：$p \equiv 1 \pmod 4$，$\chi_p$ 为二次特征标。则 $J(\chi_p, \chi_p)$ 落在 $\mathbb{Z}[i]$ 中且满足 $J = a + bi$、$a^2 + b^2 = p$——**直接把「$p$ 表成两个平方和」证出来**（Fermat 两平方和定理的 Gauss 和证明）。

## 3 应用：$x^n + y^n = 1$ 的解数

设 $N_n$ 是方程 $x^n + y^n = 1$ 在 $\mathbb{F}_p$ 中的解数。用单位根筛选（$\frac1p\sum_t \zeta^{tn}$ 是「$n \equiv 0$」的指示函数），把 $N_n$ 展开成 Gauss 和与 Jacobi 和的组合：

对 $n \mid p-1$ 与 $a \ne 0$，解数由 $n$ 次特征标对的 Jacobi 和精确表出：

$$
N(x^n + y^n = a) = \sum_{\chi^n = \varepsilon} \sum_{\psi^n = \varepsilon} J(\chi,\psi)\,\chi(a)\,\psi(a)
$$

其中退化对 $\chi\psi = \varepsilon$ 用 $J(\chi, \bar\chi) = -\chi(-1)$ 补正（平凡对 $\varepsilon, \varepsilon$ 给出主项 $p$；$\chi(0) = 0$ 的约定自动处理 $a_0 = 0$ 的边界）。<span class="marginnote">这个公式数出「费马曲线 $x^n + y^n = 1$ 的有限域点数」，是椭圆曲线/更高亏格曲线 Hasse 界 $\#E(\mathbb{F}_p) = p + 1 - a_p$ 的直接祖先——解数永远落在「$p$ 加减几个 $\sqrt{p}$」的区间内，这正是 Weil 猜想（黎曼假设的函数域版）的最初形态。</span>

**例（$n = 2$，二次型）**：$N_2 = \#\{(x,y) : x^2 + y^2 = 1\}$。用 $J(\chi_p,\chi_p)$ 算出

$$
N_2 = p - \chi_p(-1) = \begin{cases} p - 1, & p \equiv 1 \pmod 4 \\ p + 1, & p \equiv 3 \pmod 4 \end{cases}
$$

核对 $p = 5$：$x^2 + y^2 = 1$ 的解为 $(\pm1, 0), (0, \pm1)$ 共 $4 = p - 1$ ✓；$p = 3$：解为 $(\pm1, 0),(0,\pm1)$ 共 $4 = p + 1$ ✓。**Gauss 和把「几何数点」变成「代数求和」，再变成精确的封闭式**——整个过程无一近似。

## 4 公式解析：$g(\chi)\,g(\bar\chi) = \chi(-1)\,p$

$$
g(\chi)\,g(\bar\chi) = \sum_{a,b} \chi(a)\,\bar\chi(b)\,\zeta^{a+b} = \sum_{a,b} \chi(a)\chi(b)^{-1}\zeta^{a+b}
$$

- **第一步，换元**：$a = 1$ 情形 $b \mapsto ab$（$a \ne 0$），得 $g(\chi)g(\bar\chi) = \sum_{a\ne 0}\sum_b \chi(b)^{-1}\zeta^{a(1+b)}$。
- **第二步，内层和**：固定 $b$，$\sum_{a\ne0}\zeta^{a(1+b)}$ 当 $b = -1$ 时为 $p-1$，否则为 $-1$（单位根满和）——内层被「正交性」干净地筛选。
- **第三步，合并**：$= (p-1) - \sum_{b \ne -1}\chi(b)^{-1} = (p-1) + \chi(-1)$（因 $\sum_b \chi(b)^{-1} = 0$）……整理即得 $\chi(-1)p$。**核心机制：单位根的和把所有「非关键项」正交化掉，只剩下特征标在 $-1$ 处的值**。
- **第四步，取范**：$|g(\chi)| = \sqrt p$ 由此立得——Gauss 和的模长被正交性锁定，这正是「有限域 Fourier」的简洁力量。

## 5 从 Jacobi 和到算术加密学

Gauss 和与 Jacobi 和的现代后裔遍布数论与密码：

- **有限域点数的精确公式**：$\#\{x^n + y^n = a\}$ 全由 Jacobi 和给出；曲线点数进入密码学的安全性论证（椭圆曲线离散对数）。
- **二次互反律的 Gauss 和证明**：$g(\chi_p)$ 在 $\mathrm{Gal}(\mathbb{Q}(\zeta_p)/\mathbb{Q})$ 下的共轭给出 $g(\chi_p)^q \equiv \cdots$，直接导出 $(\frac{p}{q})$ 与 $(\frac{q}{p})$ 的配平。<span class="marginnote">这给专题一个圆满的回环：第 10 节的 Artin 互反律与本节的高斯和证明是同一枚硬币的两面——一个用类域论、一个用分圆和式，都回答「素数之间的对称」。密码学（RSA、Diffie-Hellman、椭圆曲线）把「有限域的乘法结构」当作强度来源，其安全性论证处处依赖这类精确计数。</span>
- **Langlands / 模形式**：Gauss 和是「有限域上的 Fourier 变换」的种子，Hecke 特征、自守 L-函数的局部因子都是它的高阶化身。

**辨析｜易错点：** Gauss 和 $g(\chi)$ 依赖**特征标是模 $p$ 的还是模 $N$ 的**、以及 $\zeta$ 的阶——公式里的 $p$ 是「特征标模长」，换了模就要换单位根。而 Jacobi 和的**非平凡条件**（$\chi, \psi, \chi\psi$ 皆非平凡）一旦放宽，乘积公式与 $|J| = \sqrt p$ 都失效——边界情形必须逐一处理。

## 6 实例：$p = 5$ 的全流程

**例 1（Gauss 和的直接计算）**：$\chi = \chi_5$ 为二次特征标，$\zeta = e^{2\pi i/5}$。$\chi(1) = 1, \chi(2) = -1, \chi(3) = -1, \chi(4) = 1$，故

$$
g(\chi) = \zeta - \zeta^2 - \zeta^3 + \zeta^4 = 2\cos\frac{2\pi}{5} + 2\cos\frac{\pi}{5} = \frac{\sqrt5-1}{2} + \frac{\sqrt5+1}{2} = \sqrt5
$$

于是 $g(\chi)^2 = 5 = \chi(-1)\cdot 5$（因 $5 \equiv 1 \pmod 4$，$\chi(-1) = 1$）——绝对值公式的实证。

**例 2（退化 Jacobi 和）**：$J(\chi, \chi) = \sum_a \chi(a)\chi(1-a)$ 逐项（$a = 2, 3, 4$ 贡献非零）：

$$
a=2:\ \chi(2)\chi(-1) = -1, \qquad a=3:\ \chi(3)\chi(-2) = 1, \qquad a=4:\ \chi(4)\chi(-3) = -1
$$

得 $J(\chi,\chi) = -1 = -\chi(-1)$。注意这里 $\chi\psi = \chi^2 = \varepsilon$ 退化，所以「$|J| = \sqrt p$」不适用（$|J| = 1$）——**乘积公式的前提「$\chi, \psi, \chi\psi$ 皆非平凡」必须逐项检查**，这是初学者最常踩的坑。

**核对解数**：$N(x^2 + y^2 = 1) = p + J(\chi,\chi) = 5 - 1 = 4$，与枚举 $(\pm1, 0), (0, \pm1)$ 完全一致——Gauss 和把「数点」变成「封闭式」的实例。

## 7 小结

- **Gauss 和** $g(\chi) = \sum_a \chi(a)\zeta^a$：特征标在单位根上的 Fourier 系数；$|g(\chi)|^2 = p$，$g(\chi)g(\bar\chi) = \chi(-1)p$。
- **Jacobi 和** $J(\chi,\psi) = \sum_a \chi(a)\psi(1-a)$；乘积公式 $J = \frac{g(\chi)g(\psi)}{g(\chi\psi)}$。
- **解数公式** $N_n = p + \sum J(\chi,\psi)$：$x^n + y^n = 1$ 的 $\mathbb{F}_p$-点数精确封闭；$n=2$ 得 $p \pm 1$。
- 应用：两平方和定理、二次互反律的 Gauss 和证明、费马曲线点数、现代密码学与模形式的种子。

至此，本专题 13 篇把代数数论的经典主干走完：从代数整数、Dedekind 整环、类群、单位定理，到赋值与 $p$-adic 数、分歧理论、Minkowski 几何、差积与判别式，再到局部类域论、Artin 互反律、Chebotarev 密度定理、Dedekind zeta 与类数公式、Gauss/Jacobi 和——下一站，你可以顺着这条主线进入《代数几何》《椭圆曲线》与《Langlands 纲领》，让互反律在更高维的表示论里重生。
