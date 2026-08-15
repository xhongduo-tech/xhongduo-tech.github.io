---
title: 解析延拓与函数方程
date: 2026-08-07
---

# 解析延拓与函数方程

<div class="epigraph">
<p>数学是给不同的事物起相同名字的艺术。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第二级 · 解析数论 ｜ T. M. Apostol《Introduction to Analytic Number Theory》第12章 ｜ 2026-08-07</p>
</div>

## 为什么必须有解析延拓

上一节结束时留了一个大坑：$\zeta(s) = \sum n^{-s}$ 只在 $\sigma > 1$ 有定义，而黎曼假设、素数定理全部关心 $\sigma \le 1$ 的区域。要迈过去，靠的是复分析里一个「白捡」的定理：**解析函数如果在一个连通区域内两处重合，就处处重合**。这给了我们唯一的合法操作——把 $\zeta$ 从半平面 $\sigma > 1$ 一路延拓出去，延拓方式是唯一的，不取决于你走哪条路。

这一节我们干两件事：给出两条具体的延拓路径（Euler–Maclaurin 与 $\eta$ 函数），然后见证最美丽的公式之一——**Zeta 的函数方程**，以及它带来的「平凡零点」。

## 1 延拓路径一：Euler–Maclaurin 求和公式

把 $\zeta$ 看作「对 $n^{-s}$ 求和」，而初等分析里有现成的工具能把「离散求和」翻译成「积分 + 修正项」：**Euler–Maclaurin 求和公式**。

对 $\sigma > 1$ 可写

$$
\zeta(s) = \frac{1}{s-1} + \frac12 + s\int_1^{\infty} \frac{\{x\} - 1/2}{x^{s+1}}\, dx
$$

其中 $\{x\}$ 是 $x$ 的小数部分。<span class="marginnote">公式的来源在《数学分析》里学过：分部积分每做一次，就「多剥出」一层 $\{x\}$ 的周期性修正；被积函数里 $x^{-s-1}$ 的衰减让积分在 $\sigma > 0$ 收敛。这是「把和变成积分」的标准套路，素数定理里的 Abel 求和也是亲戚。</span>

右边第二项在 $\sigma > 0$ 收敛（$|\{x\}-1/2| \le 1/2$，积分至多像 $\int x^{-\sigma-1}$ 那样收），于是 $\zeta$ 被延拓到 $\sigma > 0$ 且 $\sigma \neq 1$；而 $\frac1{s-1}$ 正是**极点**。反复运用求和公式（把高阶 Bernoulli 多项式剥出来）可以把延拓一路推到整个复平面。这条路线的好处是**完全初等**，缺点是只揭示局部性质。

这条公式也解释了极点的留数：把两边乘上 $(s-1)$ 再取 $s \to 1$，得 $\lim_{s\to1}(s-1)\zeta(s) = 1$，留数 1 直接可读。实际上由更精细的分析还能得到 Laurent 展开 $\zeta(s) = \frac{1}{s-1} + \gamma + \cdots$，其中 $\gamma \approx 0.5772$ 正是 **Euler 常数**——这个常数在后面的均值定理里还会反复出场。<span class="marginnote">$\gamma$ 的定义是 $\lim_{n\to\infty}\left(\sum_{k\le n}\frac1k - \log n\right)$，它是解析数论里最「低调的万能常数」：$\sum_{n\le x}d(n) = x\log x + (2\gamma-1)x + O(\sqrt{x})$ 里就有它的身影。</span>

Euler–Maclaurin 还是**计算** $\zeta$ 数值的工具：在 $N$ 处截断、加上积分修正与 Bernoulli 修正项，误差可以压到任意小。例如取 $N=10$、修正到 $B_{10}$ 项，$\zeta(2)$ 就能算到与真值 $1.6449340668\ldots$ 一致到十位小数——「把离散和变积分」的路线在数值上也同样好使。

**重点：$\zeta$ 在 $s = 1$ 处有一个简单极点，留数为 $1$，除此之外整个复平面解析**。这个极点不是缺陷，而是「素数无穷」的分析化身——上一节欧拉乘积在 $s \to 1$ 发散，正是这个极点的影射。

## 2 延拓路径二：$\eta$ 函数与直接解析公式

更优雅的做法是借道 Dirichlet $\eta$ 函数。对 $\sigma > 0$ 有

$$
\eta(s) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{n^s} = \left(1 - 2^{1-s}\right) \zeta(s)
$$

第二个等号只需把 $\zeta$ 的级数按奇偶拆分：

$$
\zeta(s) = \sum_{n} \frac1{n^s} = \sum_{m} \frac1{(2m)^s} + \sum_{m} \frac1{(2m-1)^s} = 2^{-s}\zeta(s) + \sum_{m} \frac{1}{(2m-1)^s}
$$

同理 $\eta(s) = \sum \frac1{(2m-1)^s} - 2^{-s}\zeta(s)$。两式相减、移项即得。

因为 $\eta$ 是**交错级数**，它在 $\sigma > 0$ 收敛，于是

$$
\zeta(s) = \frac{\eta(s)}{1 - 2^{1-s}}
$$

给出了 $\sigma > 0$、$s \neq 1$ 上的延拓。注意 $1 - 2^{1-s}$ 在 $s = 1 + 2\pi i k/\log 2$ 处为零，但因为 $\eta$ 在这些点也恰好为零（并且分子分母同阶），最终 $\zeta$ 只在 $s=1$ 有极点——这是解析延拓「走哪条路结果都唯一」的活证据。

拿 $s=2$ 验证一下这条恒等式：左边 $\eta(2) = 1 - \frac14 + \frac19 - \frac1{16} + \cdots = \frac{\pi^2}{12} \approx 0.8225$；右边 $\left(1 - 2^{1-2}\right)\zeta(2) = \frac12 \cdot \frac{\pi^2}{6} = \frac{\pi^2}{12}$，两边一致。这个「交错级数先收敛、再借 $\zeta$ 精确求值」的例子，正是 $\eta$ 路线价值的浓缩。

两条延拓路线性格互补：**Euler–Maclaurin 完全初等、宜计算**；$\eta$ 函数干净利落、宜证明。而第三节的 $\Gamma$ 路线最深刻，直接通向函数方程——理解延拓的「唯一性」，就是理解这三条路为什么殊途同归。

## 3 函数方程：临界线两侧的镜像

Riemann 在 1859 年给出了深刻的延拓方式，它直接来自 $\Gamma$ 积分：

$$
\pi^{-s/2}\Gamma\left(\frac{s}{2}\right)\zeta(s) = \pi^{-(1-s)/2}\Gamma\left(\frac{1-s}{2}\right)\zeta(1-s)
$$

这个式子叫 **Zeta 的函数方程**。它把 $\zeta$ 在 $s$ 与 $1-s$ 两点的值连起来——临界线 $\mathrm{Re}\, s = 1/2$ 正好是 $s \leftrightarrow 1-s$ 的对称轴。为了对称美观，通常定义

$$
\xi(s) = \frac12 s(s-1)\, \pi^{-s/2}\Gamma\left(\frac{s}{2}\right)\zeta(s)
$$

则 $\xi(s) = \xi(1-s)$，且 $\xi$ 是**整函数**（处处解析），在 $s=1$ 的极点被因子 $s-1$ 消掉。<span class="marginnote"><strong>辨析｜易错点：</strong> 很多书把函数方程写成四种不同样子（$\sin$ 形式、$\Gamma$ 形式、$\xi$ 形式），它们是等价的，靠 $\Gamma$ 的反射公式 $\Gamma(z)\Gamma(1-z) = \pi/\sin \pi z$ 互推。查资料时先认准 $\xi$ 形式再对号入座。</span>

| 形式 | 公式 | 特点 |
| --- | --- | --- |
| $\xi$ 形式 | $\xi(s) = \xi(1-s)$ | $\xi$ 是整函数，最对称 |
| $\Gamma$ 形式 | $\pi^{-s/2}\Gamma(\frac{s}{2})\zeta(s) = \pi^{-(1-s)/2}\Gamma(\frac{1-s}{2})\zeta(1-s)$ | 来自 Mellin 变换 |
| $\sin$ 形式 | $\zeta(s) = 2^s \pi^{s-1}\sin\frac{\pi s}{2}\,\Gamma(1-s)\zeta(1-s)$ | 便于数值检查 |
| 反推关系 | 由 $\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin\pi z}$ 互推 | 四者等价 |

用函数方程还能直接读出负整数处的值：结合 $\zeta$ 在 $\sigma>1$ 的值与 $\Gamma$ 的极点结构，得到 $\zeta(-m) = -\frac{B_{m+1}}{m+1}$（$B_k$ 为 Bernoulli 数）。于是 $\zeta(-1) = -1/12$、$\zeta(-3) = 1/120$ 全部可由函数方程与 Bernoulli 数算出——第二篇表格里的负数行，到这里才算有了完整出处。

**平凡零点的来历**：$s$ 取负偶数 $s = -2, -4, \ldots$ 时，$\Gamma(s/2)$ 有极点，而 $\zeta(1-s) = \zeta(3), \zeta(5), \ldots \neq 0$，要维持等式平衡，$\zeta(s)$ 必须为零。这些零点叫 **平凡零点**。函数方程没有告诉我们的是临界带 $0 < \sigma < 1$ 内的**非平凡零点**在哪里——那正是黎曼假设（第六篇）的战场。

## 4 公式解析：为什么 $\Gamma$ 会闯进来

函数方程里冒出一个 $\Gamma$ 函数，读者多半会疑惑：数论的和怎么会冒出阶乘的推广？拆解一遍 $\Gamma$ 积分的妙用：

$$
\Gamma\left(\frac{s}{2}\right) = \int_0^{\infty} t^{s/2 - 1} e^{-t}\, dt
$$

- **第一步，换元 $t = \pi n^2 x$**：得到 $n^{-s} = \pi^{-s/2} \int_0^{\infty} x^{s/2 - 1} e^{-\pi n^2 x}\, dx$。这一步把 $n^{-s}$ 写成了对 $n$ 的积分，代价是引入 $\pi^{s/2}$ 与 $\Gamma(s/2)$。
- **第二步，对 $n$ 求和**：$\pi^{-s/2}\Gamma(s/2)\zeta(s) = \int_0^\infty x^{s/2-1} \sum_{n\ge1} e^{-\pi n^2 x}\, dx$。于是数论和变成了一个关于 **theta 函数** $\theta(x) = \sum_n e^{-\pi n^2 x}$ 的 Mellin 变换。
- **第三步，利用 theta 的函数方程**：Poisson 求和公式给出 $\theta(1/x) = \sqrt{x}\,\theta(x)$。把它代回积分、用 $x \leftrightarrow 1/x$ 拆开积分区间，等式两边正好出现 $\zeta(s)$ 与 $\zeta(1-s)$——函数方程诞生。

**重点：函数方程不是「猜出来的」，而是 Poisson 求和公式的数论面孔**——「对整数求和」在 Fourier 变换下「镜像对称」，最终折射成临界线两侧的对称。这也是为什么 $\xi$ 形式如此自然。<span class="marginnote">Poisson 求和公式是「离散 ↔ 连续」最锋利的刀刃之一，从数论到信号处理（抽样定理）再到统计物理都靠它。它是第三级《数学物理方法》与傅里叶分析的又一交叉点。</span>

再补一个数值侧面：theta 函数在 $x=1$ 处取值 $\theta(1) = \sum_{n\in\mathbb{Z}} e^{-\pi n^2} \approx 1.0864$。这个数本身没有闭式，但 Poisson 求和给它的对称性 $\theta(1/x) = \sqrt{x}\,\theta(x)$ 却是精确的——「数值上说不清、结构上完全对称」，正是 $\zeta$ 这一整族的缩影。

为什么要在函数方程里乘上 $s(s-1)$ 造出 $\xi$？因为 $\zeta$ 在 $s=1$ 的极点会破坏「整函数」的对称美感：乘 $(s-1)$ 消掉极点，再乘 $s$ 让 $\xi$ 在 $s=0$ 处也正则。一个被「收拾干净」的整函数 $\xi$，是所有零点研究（第七、八篇）的合法舞台。

函数方程还能从 **Mellin 变换**的视角理解：$\pi^{-s/2}\Gamma(s/2)\zeta(s)$ 正是 theta 函数 $\theta(x) - 1$ 的 Mellin 变换在 $s$ 处取值的对称形式。Mellin 变换把「$x \to 1/x$ 的对称」翻译成「$s \to 1-s$ 的对称」——这是「离散 ↔ 连续」翻译管线最纯正的形态。

## 5 小结

- 解析函数**延拓唯一**，故 $\zeta$ 从 $\sigma>1$ 的延拓与路径无关；它延拓后在整个复平面除 $s=1$ 外解析，$s=1$ 是**留数 1 的简单极点**。
- 两条延拓路径：**Euler–Maclaurin**（$\zeta(s) = \frac1{s-1} + \frac12 + s\int_1^\infty \frac{\{x\}-1/2}{x^{s+1}}dx$）与 **$\eta$ 函数** $\zeta(s) = \eta(s)/(1-2^{1-s})$。
- **函数方程** $\pi^{-s/2}\Gamma(s/2)\zeta(s) = \pi^{-(1-s)/2}\Gamma((1-s)/2)\zeta(1-s)$，对称轴是临界线 $\mathrm{Re}\,s=1/2$；$\xi$ 形式给出整函数。
- **平凡零点** $s = -2, -4, \ldots$ 来自 $\Gamma$ 的极点；临界带内的**非平凡零点**才是素数分布的秘密所在。
- **负整数处的值**可由 $\zeta(-m) = -\frac{B_{m+1}}{m+1}$ 与 Bernoulli 数算出（如 $\zeta(-3)=1/120$），第二篇表格的负数行由此得到出处。
- 函数方程有 $\xi$ 形式、$\Gamma$ 形式、$\sin$ 形式等四种等价写法，靠 $\Gamma$ 反射公式互推——**查资料先认 $\xi$ 形式**。

现在我们已经把 $\zeta$ 请进了整个复平面。下一节，用它去拿数论史上的第一块金牌——**素数定理**。
