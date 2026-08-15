---
title: 面积公式与余面积公式
date: 2026-08-07
---

# 面积公式与余面积公式

<div class="epigraph">
<p>面积公式把「变换后的面积」折算回「原像的面积」，余面积公式把高维积分拆成一叠低维切片——几何测度论的两把万能钥匙。</p>
<footer>—— 自 L. C. Evans & R. F. Gariepy, *Measure Theory and Fine Properties of Functions\*（意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ P. Mattila, *Geometry of Sets and Measures in Euclidean Spaces\*, §7–8 ｜ 2026-08-07</p>
</div>

## 为什么从面积公式开始

微积分里最熟悉的面积变换是**重积分换元公式**：光滑双射 $T$ 把区域 $U$ 映到 $T(U)$，面积（体积）按 $|\det DT|$ 缩放。但实际场景很少是双射：一条曲线蜷曲着穿过平面（像 0 测度却可能覆盖正面积的区域）、一张被折叠的参数曲面、一个把多片材料压到一起的 Lipschitz 映射。**面积公式（area formula）** 把换元公式推广到 Lipschitz 映射与 $\mathcal{H}^m$ 测度，允许非单射、允许重叠——重叠处按「覆盖层数」加权计数。**余面积公式（coarea formula）** 则是对偶的另一面：把 $\mathbb{R}^n$ 上的积分按「到目标空间某一点的纤维（层）」切片，再沿目标积分。两者合起来，是所有整流集、电流、变分问题计算面积和体积的基础工具。<span class="marginnote">两个公式的直觉：面积公式回答「$f$ 把 $m$ 维材料摊在 $\mathbb{R}^n$ 上盖了多少面积」，余面积公式回答「$\mathbb{R}^n$ 上的体积如何被 $f$ 的每一层 $f^{-1}(t)$ 切片」。一个水平切片，一个竖向切丝，合起来就是一张完整的「换底公式」。</span>

## 1 换元公式的困局：非单射与重叠

回顾经典换元公式：$T: U \to \mathbb{R}^n$ 是 $C^1$ 单射，则

$$
\int_{T(U)} g(y)\, \mathrm{d}\mathcal{L}^n(y) \;=\; \int_{U} g(T(x))\, |\det DT(x)|\, \mathrm{d}\mathcal{L}^n(x)
$$

这里「单射」和「$C^1$」是两个奢侈的假设。Lipschitz 映射可能把多块区域压到同一处（$T(x) = x^2$ 在 $[-1,1]$ 上，两块区间映到同一段 $[0,1]$）；甚至可能把一个正测度集合映到低维（$T(x,y) = x$ 把单位正方形映成区间）。这两个问题分别由面积公式与余面积公式处理。

**核心概念（Jacobian）**：对 Lipschitz 映射 $f: \mathbb{R}^n \to \mathbb{R}^m$（$m \ge n$ 时定义面积公式，$m \le n$ 时定义余面积公式），在 $f$ 几乎处处可微的点定义 **$n$ 维 Jacobian**

$$
J f(x) \;=\; \sqrt{\det\bigl( Df(x)^T Df(x) \bigr)}
$$

当 $n = m$ 时 $Jf = |\det Df|$；当 $m > n$ 时 $Jf$ 度量「$n$ 维体积元被嵌入 $m$ 维空间后放大多少」。<span class="marginnote">$Jf$ 的几何意义：$Df(x)$ 把单位 $n$ 方体映成一个 $n$ 维平行体，$Jf(x)$ 恰是该平行体的 $n$ 维体积——用 Gram 行列式 $\det(Df^T Df)$ 计算。Rademacher 定理保证 $Jf$ 几乎处处存在。</span>

## 2 面积公式：非单射情形的换元

**核心概念（面积公式）**：设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是 Lipschitz 映射，$n \le m$，$g: \mathbb{R}^n \to \mathbb{R}$ 是非负 Borel 函数。则

$$
\int_{\mathbb{R}^n} g(x)\, Jf(x)\; \mathrm{d}\mathcal{L}^n(x) \;=\; \int_{\mathbb{R}^m} \left( \sum_{x \in f^{-1}(y)} g(x) \right)\; \mathrm{d}\mathcal{H}^n(y)
$$

特别地，取 $g = 1$、$A = f(E)$ 时得到「面积形式」：

$$
\mathcal{H}^n(f(E)) \;=\; \int_E Jf(x)\; \mathrm{d}\mathcal{L}^n(x), \qquad \text{若 } f|_{E} \text{ 几乎处处单射}
$$

**关键结论：Lipschitz 映射不增加维数结构，却按 Jacobian 放大面积。** 内层和式 $\sum_{x \in f^{-1}(y)} g(x)$ 统计「点 $y$ 被多少个原像覆盖」，把重叠处的面积加起来——这正是非单射的补偿方式。<span class="marginnote">面积公式可以看作「把换元公式中单射假设删除」的补偿：单射时 $f^{-1}(y)$ 至多一个点，内层和退化成 $g(f^{-1}(y))$，右端回到经典的 $\int g \, |\det DT|$；非单射时，重叠层数被精确计入。</span>

**辨析｜易错点：** 面积公式在「$f|_E$ 几乎处处单射」时才有简洁的面积形式；对一般的多重点，必须保留内层计数和式。另外注意等式右端是 $\mathcal{H}^n$（目标空间的 $n$ 维测度），不是 $\mathcal{L}^n$——当 $m > n$ 时像落在低维子流形上，用 $\mathcal{H}^n$ 度量才恰当。这是初学者最容易写错的地方。

## 3 公式解析：面积公式的三层含义

把核心等式拆开看：

- **左端，$\int g \, Jf \, \mathrm{d}\mathcal{L}^n$（原像侧）**：在原空间积分，被积函数是 $g$ 与 Jacobian 的乘积。Jacobian 扮演「局部面积缩放因子」，把原空间的体积元换算成目标空间的面积元。
- **内层和，$\sum_{x \in f^{-1}(y)} g(x)$（重叠计数）**：对目标点 $y$，把它所有的原像上的 $g$ 值加起来。当 $g \equiv 1$ 时，这个和就是「$y$ 被覆盖的次数」——数学上叫**覆盖函数（multiplicity）** $N(f, y) = \#\{x : f(x) = y\}$。
- **右端，$\int \cdots \, \mathrm{d}\mathcal{H}^n(y)$（目标侧）**：沿目标空间用 $n$ 维 Hausdorff 测度积分。这是「面积」的度量——像集若是光滑 $n$ 维子流形，$\mathcal{H}^n$ 就是通常的曲面面积。
- **几乎处处约定**：等式两端都只要求几乎处处意义下定义良好；$f$ 不可微的点、$Jf$ 发散的点都属于零测度例外集。

**重点：面积公式把「几何面积」与「解析 Jacobian」统一起来。** 经典曲面面积 $\mathcal{H}^n(f(E))$ 可以纯粹用 Jacobian 积分计算，而 Jacobian 又可纯粹用测度定义——几何与分析的鸿沟由此弥合。这是整流集理论中「重数 = 密度」的测度论根源。

## 4 余面积公式：把积分切成层

现在转向对偶情形：$f: \mathbb{R}^n \to \mathbb{R}^m$ 是 Lipschitz 映射，$n \ge m$。此时 $f$ 把高维空间压到低维目标，每一层 $f^{-1}(t)$ 是 $n-m$ 维的纤维。

**核心概念（余面积公式）**：设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是 Lipschitz 映射，$n \ge m$，$g: \mathbb{R}^n \to \mathbb{R}$ 是非负 Borel 函数。则

$$
\int_{\mathbb{R}^n} g(x)\, Jf(x)\; \mathrm{d}\mathcal{L}^n(x) \;=\; \int_{\mathbb{R}^m} \left( \int_{f^{-1}(t)} g(x)\; \mathrm{d}\mathcal{H}^{n-m}(x) \right)\; \mathrm{d}\mathcal{L}^m(t)
$$

当 $m = 1$ 时（$f: \mathbb{R}^n \to \mathbb{R}$ 是标量函数），它是 Fubini 定理的几何化：

$$
\int_{\mathbb{R}^n} g(x)\, |\nabla f(x)|\; \mathrm{d}\mathcal{L}^n(x) \;=\; \int_{-\infty}^{\infty} \int_{f^{-1}(t)} g(x)\; \mathrm{d}\mathcal{H}^{n-1}(x)\; \mathrm{d}t
$$

**关键结论：$|\nabla f|$ 扮演「切片厚度修正因子」。** 等值面 $f^{-1}(t)$ 的间距不均匀时，$|\nabla f|$ 补偿「单位高度变化对应的实际层厚」，使得「按 $t$ 积分」与「按空间体积积分」精确相等。<span class="marginnote">为什么需要 $|\nabla f|$？直观上，若 $f$ 的梯度大，等值面挤在一起，一层薄薄的 $t$ 变化对应很厚的空间区域；$|\nabla f|$ 就是这个「层厚」的换算因子。梯度为零的点（临界点）不贡献层厚，对应 Sard 定理的零测度例外。</span>

## 5 公式解析：余面积公式的纤维语言

逐项拆解余面积公式：

**$Jf(x)$（Jacobian，目标维数低时）**：当 $n \ge m$ 时 $Jf$ 度量「$n$ 维体积元被压到 $m$ 维目标后剩余多少」，它把纤维方向的厚度折算进权重。$f: \mathbb{R}^2 \to \mathbb{R}$ 时 $Jf = |\nabla f|$。
**纤维 $f^{-1}(t)$（层）**：对每个 $t$，$f^{-1}(t)$ 是 $n-m$ 维的水平集。在一般 Lipschitz 情形，纤维不必是流形，但由共面积公式的测度论版本，$\mathcal{H}^{n-m}$-几乎处处纤维是整流集——这让层切片有严格的积分意义。
**内层积分，$\int_{f^{-1}(t)} g \, \mathrm{d}\mathcal{H}^{n-m}$（沿纤维）**：先沿每一层积分，把「层」压缩成一个数。
- **外层积分，$\int \cdots \, \mathrm{d}\mathcal{L}^m(t)$（沿目标）**：再对 $t$ 积分，把层「摞」起来。

**辨析｜易错点：** 余面积公式不是「无条件的分层积分」。当 $f$ 有大量临界点（$\nabla f = 0$ 的集合有正测度）时，直接写 $\int_{\mathbb{R}^n} g = \int \int_{f^{-1}(t)} g \, \mathrm{d}\mathcal{H}^{n-1} \mathrm{d}t$ 是错的——正确的公式必须带上 $|\nabla f|$。例如 $f(x,y) = 0$ 是常数，等值面只有一个（整个平面），分层积分毫无意义，而 $|\nabla f| = 0$ 使左端恒为 0，公式保持正确。**$|\nabla f|$ 的存在正是为了在临界处自动「熄灭」切片。**

## 6 面积 / 余面积公式的应用

两个公式在几何测度论内外都硕果累累。

其一，**Sard 定理的测度论版本**：余面积公式直接给出——Lipschitz 映射 $f: \mathbb{R}^n \to \mathbb{R}^m$（$n \ge m$）的临界值集合（$\{f(x) : Jf(x) = 0\}$）是 $\mathcal{L}^m$-零测度的。因为在这些点上积分权为 0，切片积分为 0。<span class="marginnote">经典 Sard 定理要求 $C^k$ 光滑（$k$ 足够大），Lipschitz 情形的临界值零测度由余面积公式一步可得——公式的威力可见一斑。</span>

其二，**整流测度的密度 = 重数**：面积公式的测度形式说明，对整流测度 $\mu = \theta \mathcal{H}^m \llcorner E$，球内质量 $\mu(B(x,r))$ 渐近等于 $\theta(x) \omega_m r^m$，从而密度就是重数（呼应第 5 篇）。

其三，**等周不等式与 Sobolev 函数**：余面积公式把 Sobolev 函数的范数估计化为「每层水平集的周长估计」，是证明 Sobolev 嵌入、等周不等式的标准路径。<span class="marginnote">等周不等式：给定体积，球的表面积最小。用余面积公式对 $u(x) = |x|$ 分层，可把表面积表示成层周长的积分，再逐层用等周不等式，拼出整体结论——这是几何分析里的经典戏法。</span>

**比较表**：面积公式与余面积公式的对照。

| 特征 | 面积公式 | 余面积公式 |
| --- | --- | --- |
| 维数 | $n \le m$（升维 / 嵌入） | $n \ge m$（降维 / 压扁） |
| 度量目标 | $\mathcal{H}^n$（像的 $n$ 维面积） | $\mathcal{L}^m$（目标的 $m$ 维体积） |
| 切片方向 | 沿目标统计原像层数 | 沿目标对原像分层 |
| 特殊情况 | 单射时回到换元公式 | $m=1$ 时是带 $|\nabla f|$ 的 Fubini |
| 补偿因子 | 覆盖函数（重数） | $|\nabla f|$（层厚） |

## 7 小结

- **面积公式**：$\int g\, Jf = \int \sum_{x \in f^{-1}(y)} g(x)\, \mathrm{d}\mathcal{H}^n(y)$，用 Jacobian 与重叠计数把换元公式推广到非单射 Lipschitz 映射。
- **Jacobian**：$Jf = \sqrt{\det(Df^T Df)}$，度量体积元在映射下的缩放；几乎处处存在（Rademacher）。
- **余面积公式**：$\int g\, Jf = \int \int_{f^{-1}(t)} g\, \mathrm{d}\mathcal{H}^{n-m} \, \mathrm{d}\mathcal{L}^m(t)$，把高维积分切成 $n-m$ 维纤维层。
- $|\nabla f|$