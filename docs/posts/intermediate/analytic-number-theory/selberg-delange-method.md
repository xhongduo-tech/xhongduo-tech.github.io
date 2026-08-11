---
title: 均值定理深化：Selberg-Delange 方法与算术函数均值
date: 2026-08-11
---

# 均值定理深化：Selberg-Delange 方法与算术函数均值

<div class="epigraph">
<p>一个方程对我没有意义，除非它表达了上帝的一个思想。</p>
<footer>—— 斯里尼瓦瑟 · 拉马努金（Srinivasa Ramanujan）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 解析数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么 Wirsing–Halász 还不够

第九篇的均值定理给出了乘法函数均值的**极限**，但误差只有 $o(1)$——对具体计算来说「趋于某常数」远远不够。我们需要的是**带精度的均值**：$\sum_{n\le x} a(n) = \text{主项} \cdot (1 + O(1/\log x))$。这要更强的工具，而它来自分析的一个精准观察：**多数自然出现的算术函数，其 Dirichlet 级数在 $s=1$ 附近的奇点长得像 $\zeta(s)$ 的幂**。

Selberg 与 Delange 在 1940–1950 年代发展出一套方法，把这类奇点变成精确的均值渐近。这一节我们会看到主项里那个意味深长的因子 $(\log x)^{z-1}/\Gamma(z)$，并拿它算出「恰有 $k$ 个素因子的整数有多少」。

## 1 观察：一切都像 $\zeta(s)^z$

先看三个例子，它们的 Dirichlet 级数都「几乎等于」$\zeta(s)$ 的某个幂：

| $a(n)$ | $F(s) = \sum a(n)/n^s$ | 在 $s=1$ 附近的形态 |
| --- | --- | --- |
| $\mathbf{1}$ | $\zeta(s)$ | $\zeta(s)^1$ |
| $d(n)$（因子数） | $\zeta(s)^2$ | $\zeta(s)^2$ |
| $\tau_k(n)$（$k$ 元因子数） | $\zeta(s)^k$ | $\zeta(s)^k$ |
| $z^{\Omega(n)}$（对因子个数加权） | $\prod_p (1 - zp^{-s})^{-1}$ | $\zeta(s)^z \cdot H(s)$ |

其中 $z^{\Omega(n)}$ 那行是关键的引子：$\Omega(n)$ 是带重数统计的素因子个数，它的加权生成函数是 $\prod_p(1 - zp^{-s})^{-1}$，而 $\prod_p (1 - p^{-s})^{-1} = \zeta(s)$，比值

$$
H(s) = \prod_p (1 - p^{-s})^z (1 - zp^{-s})^{-1}
$$

在 $\sigma > 1$ 解析且非零。<span class="marginnote"><strong>辨析｜易错点：</strong> $z^{\Omega(n)}$ 的欧拉乘积每项是 $(1 - zp^{-s})^{-1}$ 而不是 $(1 - zp^{-s})^z$——第一个 $z$ 在「底数」位置，第二个 $z$ 是幂。把「素因子个数」当独立伯努利变量是概率直觉，但落到级数上必须是这个乘积，错一处就全错。</span>

**重点：凡 Dirichlet 级数在 $s=1$ 附近能写成 $\zeta(s)^z G(s)$（$G$ 解析非零），它就落入 Selberg–Delange 的射程**。$\zeta$ 本身是「一阶极点的原型」，$\zeta^z$ 则是「$z$ 阶极点」（$z$ 可以是任意复数）的原型。

## 2 定理陈述：主项里的 $\Gamma$ 和 $(\log x)^{z-1}$

**Selberg–Delange 定理（主项形式）**：设 $F(s) = \sum a(n)/n^s$ 满足 $F(s) = \zeta(s)^z G(s)$，其中 $G$ 在 $\sigma \ge 1$ 解析、非零，且满足温和的增长条件，$z \in \mathbb{C}$。则对 $x \ge 3$，

$$
\sum_{n \le x} a(n) = x\, (\log x)^{z-1}\, \frac{G(1)}{\Gamma(z)} \left(1 + O\!\left(\frac1{\log x}\right)\right)
$$

（$z$ 落在 $\Gamma$ 的极点之外；$G(1)$ 是 $G$ 在 $s=1$ 的值。）<span class="marginnote">当 $z$ 取非正整数时 $\Gamma(z)$ 有极点，主项公式需要换成带 $\log x$ 的对数校正的版本——那是方法本身的细节，我们这里只讨论「干净」的 $z$。实际上对 $z=1$ 退化为 $\sum \mathbf{1} = x(1+O(1/\log x))$，对 $z=2$ 给出 $\sum d(n) = x\log x\,G(1)(1+O(1/\log x))$——但注意这时 $G(1) = \prod_p (1-p^{-s})^{-2}\zeta(s)^{-2}$ 在 $s=1$ 处不为 1，常数与 Dirichlet 级数的定义细节相关。</span>

这个定理比 Wirsing–Halász 强在哪？它给出**乘法常数 $G(1)/\Gamma(z)$ 与对数阶 $(\log x)^{z-1}$ 一起**的完整主项，且误差只有相对 $O(1/\log x)$——足以做进一步的减法、除法与求导（比如第十一篇大筛法的均值估计就要用到这种精度）。

## 3 公式解析：$(\log x)^{z-1}/\Gamma(z)$ 从哪来

主项的形状不是猜的，它从「$\zeta(s)^z$ 的 Mellin 反演」里**被迫**长出来。拆三步：

$$
\sum_{n\le x} a(n) = \frac{1}{2\pi i}\int_{2-i\infty}^{2+i\infty} \zeta(s)^z G(s)\, \frac{x^s}{s}\, ds
$$

- **第一步，Perron 公式**：和等于 Mellin 反演，积分线立在 $\sigma = 2$。把围道向左挪到 $\sigma = 1$ 左侧，主项来自 $s=1$ 附近的奇点，误差来自其余部分（归入 $O(1/\log x)$）。
- **第二步，换元到奇点附近**：在 $s=1$ 附近，$\zeta(s)^z \approx (s-1)^{-z}G(1)$。令 $w = (s-1)\log x$，则 $x^s = x\, e^{w}$、$s^{-1} \approx 1$，积分主项变成
  $x\,G(1) \cdot \frac{1}{2\pi i}\int \left(\frac{w}{\log x}\right)^{-z} e^{w}\,\frac{dw}{\log x} = x\,G(1)(\log x)^{z-1} \cdot \frac{1}{2\pi i}\int w^{-z} e^w\, dw$。
- **第三步，认出 Hankel 积分**：$\frac{1}{2\pi i}\int w^{-z}e^w\, dw = \frac{1}{\Gamma(z)}$——这正是 **Γ 函数的 Hankel 围道积分表示**（把积分线绕成绕割线的钥匙孔围道，$w^{-z}$ 给出割线的相位跳跃）。于是主项里出现 $(\log x)^{z-1}/\Gamma(z)$。

**直觉**：$(\log x)^{z-1}$ 来自「$s=1$ 处 $z$ 阶奇点在 $\log x$ 尺度上积分」，而 $1/\Gamma(z)$ 是这个积分的归一化常数——**不是凭空插入的装饰，而是 Hankel 围道绕割线绕出来的自然结果**。<span class="marginnote">Hankel 围道是复分析里绕「割线」积分的标准姿势，第二篇讲 $\Gamma$ 的反射公式时提过它的表亲。一个方法里同时出现 $\Gamma$ 与 Perron 积分，说明它站在第三篇（Mellin/函数方程）与第九篇（均值定理）的交汇点上。</span>

## 4 应用：恰有 $k$ 个素因子的整数有多少

Selberg–Delange 最漂亮的成名作：记 $N_k(x) = \#\{n \le x : \Omega(n) = k\}$。对固定的 $k$，

$$
N_k(x) \sim \frac{x}{\log x}\, \frac{(\log\log x)^{k-1}}{(k-1)!}
$$

**这意味着「$n \le x$ 的素因子个数」像参数为 $\log\log x$ 的 Poisson 分布**：主项里 $(\log\log x)^{k-1}/(k-1)!$ 正是 Poisson 概率 $e^{-\lambda}\lambda^{k-1}/(k-1)!$ 的核心形状（$e^{-\lambda}$ 被归一化吸收）。<span class="marginnote">推导钥匙：在 $z^{\Omega(n)}$ 的级数上对 $z$ 取系数。$\frac{1}{k!}\frac{d^k}{dz^k}\big|_{z=0} z^{\Omega(n)} = [\Omega(n)=k]$，把 Selberg–Delange 的公式作用到每个 $z$ 幂再取 $z^k$ 系数，就得到上面的 Poisson 主项。这是一次「生成函数 + 系数提取」的标准作业。</span>

**重点：Poisson 分布（第九篇 Erdős–Kac 的极限是正态）在这里以更精确的形态现身**——固定 $k$ 时计数是 Poisson 型，$k$ 随 $x$ 变大后再取极限，Poisson 又滑向正态。数论与概率的这条暗线，从均值定理一路贯穿到 Selberg–Delange。

## 5 与 Erdős–Kac 的对表：为什么 $(\log\log x)$ 到处出现

把两套结果放一起看：Erdős–Kac 说 $\omega(n)$ 的分布以 $\log\log x$ 为均值、方差；Selberg–Delange 说「恰有 $k$ 个」的计数是 Poisson($\log\log x$) 型。它们不是两个孤立结果，而是同一现象的两种精度：

**定性**（Erdős–Kac）：分布形状 = 独立变量之和 → 中心极限定理；
- **定量**（Selberg–Delange）：每个 $k$ 的概率都精确知道 → Poisson 结构。

**辨析｜易错点：** Poisson 的「平均」$\log\log x$ 增长极慢，但它是**真正的期望**而非近似——$N_k(x)$ 的渐近确实以 $(\log\log x)^{k-1}/(k-1)!$ 为主项。把「平均 $\log\log x$ 个素因子」误读成「素数密度」，是初学均值理论最常踩的坑（第九篇也警告过）。<span class="marginnote">从「从极限到大模型」的主线看，这个「$\log\log$ 尺度下的普适计数分布」与词频长尾、激活对数正态同属一类「大而无序的计数趋于普适」现象——素数因子个数、单词出现次数、矩阵特征值，在不同标尺上重演同一套统计剧情。</span>

## 6 小结

- **Selberg–Delange 方法**：当 $F(s) = \zeta(s)^z G(s)$（$G$ 解析非零）时，$\sum_{n\le x}a(n) = x(\log x)^{z-1}\frac{G(1)}{\Gamma(z)}(1+O(1/\log x))$。
- **主项来源**：$s=1$ 处 $\zeta(s)^z \approx (s-1)^{-z}$，Mellin 反演 + **Hankel 围道**给出 $(\log x)^{z-1}/\Gamma(z)$——$\Gamma$ 不是装饰，是绕割线的积分常数。
- **应用**：恰有 $k$ 个素因子的整数计数 $N_k(x) \sim \frac{x}{\log x}\frac{(\log\log x)^{k-1}}{(k-1)!}$，呈现 **Poisson($\log\log x$)** 结构。
- 与 **Erdős–Kac** 定理互补：定量（每个 $k$ 精确）与定性（极限正态）合流为同一统计图景。
- 误差是**相对** $O(1/\log x)$，足以支持除法、求导等二次操作——下一节大筛法正需要这种精度。

下一节我们进入均值估计的「重武器」：**大筛法与 Bombieri–Vinogradov 定理**——如何在一个**平均**意义上绕过 Siegel 零点、对几乎所有模数同时得到素数分布的均匀性。
