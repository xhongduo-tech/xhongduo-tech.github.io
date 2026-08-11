---
title: 素数定理
date: 2026-08-11
---

# 素数定理

<div class="epigraph">
<p>大约在 1792 或 1793 年，我还是个少年时，就已观察到素数的平均密度约为 1/log x。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（致恩克的信, 1849）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 解析数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么素数定理是王冠

数论最古老的问题之一：小于 $x$ 的素数有多少个？记这个数为 $\pi(x)$，那么 $\pi(10) = 4$（2, 3, 5, 7），$\pi(100) = 25$，$\pi(10^6) = 78498$。这些数字毫无规律——但高斯在少年时代就猜想，它们服从一条**光滑规律**：$\pi(x)$ 大约等于 $x/\log x$。这个猜想花了一百多年才被证明，而证明它的工具正是我们前几节搭好的整套机器。

**素数定理（Prime Number Theorem, PNT）**：$\pi(x) \sim \dfrac{x}{\log x}$，即 $\lim_{x\to\infty} \frac{\pi(x)}{x/\log x} = 1$。

1896 年阿达马（Hadamard）与德·拉·瓦莱-普桑（de la Vallée Poussin）**独立地**用 $\zeta$ 函数证明了它；1949 年塞尔伯格与爱多士又给出（困难得多的）初等证明。无论哪条路，PNT 都是分析进入数论后的第一个顶点成果。<span class="marginnote">PNT 不是「素数多到几乎没有」——它说的是<strong>精确的比例</strong>：当 $x$ 极大时，第 $n$ 个素数约等于 $n\log n$，素数占整数的比例是 $1/\log x$，趋于 0 但方式完全确定。下一节我们会用它精确的极限来反推素数分布的全部信息。</span>

## 1 历史：猜想与逼近的竞赛

1798 年勒让德猜测 $\pi(x) \approx \frac{x}{\log x - 1.08366}$；高斯则看上了**对数积分** $\mathrm{Li}(x) = \int_2^x \frac{dt}{\log t}$。高斯是对的：$\mathrm{Li}(x)$ 是比 $x/\log x$ 好得多的逼近。下表给出真实值对比：

| $x$ | $\pi(x)$ | $x/\log x$ | $\mathrm{Li}(x)$ | 相对误差 $(x/\log x)$ |
| --- | --- | --- | --- | --- |
| $10^3$ | 168 | 145 | 177 | 13.7% |
| $10^4$ | 1229 | 1086 | 1246 | 11.6% |
| $10^5$ | 9592 | 8686 | 9630 | 9.4% |
| $10^6$ | 78498 | 72382 | 78628 | 7.8% |
| $10^9$ | 50847534 | 48254942 | 50849235 | 5.1% |

**重点：$x/\log x$ 与 $\mathrm{Li}(x)$ 的比值趋于 1，但 $\mathrm{Li}(x)$ 追得更紧**——两者的差大约是 $\pi(x) - \mathrm{Li}(x) = O(x e^{-c\sqrt{\log x}})$。一百多年里，大量数值证据让数学家相信 $\mathrm{Li}(x)$ 永远偏大（Littlewood 却在 1914 年证明它**必然后来居上**，甚至符号翻转无穷多次——这正是分析能给出「初等直觉根本看不到」的结论的例子）。

## 2 等价形式：为什么非要用 $\psi$ 函数

直接对 $\pi(x)$ 下手是笨的；数论家把目光转向一个「加权计数」：**Chebyshev 函数** $\psi(x) = \sum_{n \le x} \Lambda(n)$，即对 $n = p^k \le x$ 累加 $\log p$。它的意义是「把所有素数的对数按含幂权重加起来」。

**Chebyshev 定理（核心等价）**：以下三条彼此等价：

$$
\pi(x) \sim \frac{x}{\log x} \quad\Longleftrightarrow\quad \theta(x) = \sum_{p \le x} \log p \sim x \quad\Longleftrightarrow\quad \psi(x) \sim x
$$

其中 $\theta(x)$ 是不含幂权重的对数和。等价性的证明只需一次**分部求和（Abel 求和）**：素数幂比素数稀疏得多，把 $\psi$ 中 $p^k$（$k\ge2$）的部分单独估掉，误差至多 $O(\sqrt{x}\log x)$。<span class="marginnote">这就是「为什么不用 $\pi(x)$ 直接证」的答案：$\pi(x)$ 是阶梯函数，零点信息被粗糙地压扁了；$\psi(x)$ 与 $\zeta$ 的耦合是<strong>乘性</strong>的——因为 $\log \zeta(s) = \sum \Lambda(n)n^{-s}$，而 $-\zeta'/\zeta$ 更是 $\Lambda$ 的 Dirichlet 级数。一个函数和一个复函数直接挂钩，方便多了。</span>

## 3 证明的骨架：为什么需要零点

为什么 $\zeta$ 的零点能决定素数分布？把线拉直：

- **第一步**：由欧拉乘积取对数，$\log \zeta(s) = \sum_n \Lambda(n)n^{-s}$。于是 $\Lambda$ 的全部信息都藏进了 $\zeta$ 的对数。
- **第二步**：**Perron 公式**（Mellin 反演）把「累加 $\Lambda(n)$ 到 $x$」翻译成「在复平面上对 $-\zeta'/\zeta$ 做积分」：$\psi(x) = \frac{1}{2\pi i}\int_{c-i\infty}^{c+i\infty} \left(-\frac{\zeta'(s)}{\zeta(s)}\right) \frac{x^s}{s}\, ds$。
- **第三步**：把积分线向左挪。**$-\zeta'/\zeta$ 的极点（即 $\zeta$ 的零点）会贡献出主项与误差项**。若 $\zeta$ 在 $\mathrm{Re}\,s > 1$ 有零点，主项会失控；而我们在第一节说过，$\sigma > 1$ 没有零点——**问题的全部重心移到：$\zeta$ 在 $\mathrm{Re}\,s = 1$ 上有没有零点？**
- **第四步（1896 年的关键）**：阿达马与德·拉·瓦莱-普桑各自证明了 $\zeta$ 在直线 $\sigma = 1$ 上**无零点**。由此 Perron 积分可以安全挪动，主项 $\psi(x) \sim x$ 弹出，PNT 得证。

**重点：素数定理等价于「$\zeta$ 在 $\mathrm{Re}\,s = 1$ 上无零点」**。把「多少个素数」这样一个初等问题翻译成「一个解析函数在一条直线上有没有零点」，这就是黎曼划时代的思路——也立刻打开了通往黎曼假设的门。

## 4 公式解析：误差项与零点的距离

PNT 的强弱完全由误差项刻画。事实上可证

$$
\psi(x) = x + O\!\left(x\, e^{-c\sqrt{\log x}}\right), \qquad \pi(x) = \mathrm{Li}(x) + O\!\left(x\, e^{-c\sqrt{\log x}}\right)
$$

拆解这条公式：

- **主项 $x$**：来自积分线挪到 $\sigma = 1$ 时 $-\zeta'/\zeta$ 在 $s=1$ 的极点（留数 1）。
- **指数衰减 $e^{-c\sqrt{\log x}}$**：来自允许把积分线挪到**零区域** $\sigma \ge 1 - c/\log t$ 的边界——这个区域是「$\sigma=1$ 无零点」的量化版本，越是深入（把边界向左推）误差项越小。
- **直觉**：$e^{-c\sqrt{\log x}}$ 比任何 $\log^{-A}x$ 都大、比任何 $x^{-\delta}$ 都大，但它**趋于 0**——所以 PNT 的误差是「亚多项式但亚指数」的。

**辨析｜易错点：** PNT 只告诉你「趋于 $x$」，不告诉你 $\mathrm{Li}(x)$ 与 $\pi(x)$ 谁大。$\pi(x) < \mathrm{Li}(x)$ 在所有数值表里都成立，但它其实是错的——Littlewood 定理（1914）保证符号翻转。把「数值证据」当「定理」是数论里最贵的教训。

## 5 一张图：三种函数赛跑

![素数计数函数 $\pi(x)$ 与 $x/\log x$、$\mathrm{Li}(x)$ 的对比](/images/analytic-number-theory/prime-number-theorem-1.svg)

这幅示意图画出 $x$ 增大时三个函数的关系：$x/\log x$ 与 $\mathrm{Li}(x)$ 都紧跟 $\pi(x)$，而 $\mathrm{Li}(x)$ 追得更近——这正是高斯少年直觉的视觉化。<span class="marginnote">示意图取对数-线性坐标，曲线只示意形状，真实数值请以上表为准。把「逼近」与「相等」区分开，是读懂这张图的关键。</span>

## 6 小结

- **素数定理**：$\pi(x) \sim x/\log x$，等价于 $\psi(x) \sim x$ 与 $\theta(x) \sim x$；187 年后由阿达马与德·拉·瓦莱-普桑在 1896 年证明。
- 证明骨架：$\log \zeta = \sum \Lambda n^{-s}$ → Perron 公式 → 把积分线挪进零区域 → **$\zeta$ 在 $\sigma=1$ 无零点** 保证主项成立。
- **误差项**：$\psi(x) = x + O(x e^{-c\sqrt{\log x}})$；误差的大小完全由零区域（$\zeta$ 零点离直线 $\sigma=1$ 多远）决定。
- Littlewood 定理：$\pi(x) - \mathrm{Li}(x)$ 的符号翻转无穷多次——数值证据永远不能替代证明。

下一个问题顺理成章：零点到底能离 $\sigma = 1$ 多远？这把我们带向**零区域与 Deuring–Heilbronn 现象**，以及更深的、关于零点分布的一切——**黎曼假设**。
