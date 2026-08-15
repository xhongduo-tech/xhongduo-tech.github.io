---
title: Rademacher 定理与 Lipschitz 函数
date: 2026-08-07
---

# Rademacher 定理与 Lipschitz 函数

<div class="epigraph">
<p>几乎处处可微，是一切「足够受控」的函数的命运。</p>
<footer>—— 自 H. Rademacher（汉斯 · 拉多马赫，1892–1969），对其 1919 年定理的概括</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ L. C. Evans & R. F. Gariepy, *Measure Theory and Fine Properties of Functions\*, §3.1–3.2 ｜ 2026-08-07</p>
</div>

## 为什么从 Lipschitz 函数开始

几何测度论研究的对象从「静态的集合」扩展到「把集合映到集合的映射」，马上遇到一个根本问题：**什么样的映射才值得研究？** 光滑映射（$C^1$）性质好但太少——康托尔函数、锯齿测度、分形自相似映射全都不是光滑的；连续映射又太野——Weierstrass 函数处处连续却无处可微。折中的选择落在 **Lipschitz 连续**：它足够温和，保证几乎所有几何量都能计算；又足够灵活，囊括了大多数「由实际问题长出来」的映射。Lipschitz 函数最漂亮的定理由 Rademacher 于 1919 年给出：**Lipschitz 函数几乎处处可微。** 这条定理让「不可微函数」的世界重新获得微积分工具的眷顾，是整个几何测度论从集合过渡到映射的枢纽。<span class="marginnote">从分析学的角度看，Rademacher 定理是「绝对连续函数几乎处处可微」（Lebesgue 微分定理）在高维的翻版，但结论强得多：Lipschitz 条件直接控制住了差商，使得几乎处处可微成为必然。</span>

## 1 Lipschitz 函数：定义与第一性质

**核心概念（Lipschitz 连续）**：设 $A \subset \mathbb{R}^n$，$f: A \to \mathbb{R}^m$。若存在常数 $L \ge 0$，使得对一切 $x, y \in A$，

$$
|f(x) - f(y)| \;\le\; L\,|x - y|
$$

则称 $f$ 是 **Lipschitz 连续的**，最小的可行 $L$ 记作 $\mathrm{Lip}(f)$，称为 $f$ 的 **Lipschitz 常数**。当 $L \lt  1$ 时叫压缩映射，$L = 1$ 时叫非扩张映射。

Lipschitz 连续把「连续」这个拓扑概念升级成了「受控」的度量概念：函数值的震荡被输入距离成比例地约束。由此立刻得到三件好事的清单：

- **保 Hausdorff 维数**：$\dim_{\mathrm{H}} f(A) \le \dim_{\mathrm{H}} A$（Lipschitz 映射不提高维数，因为 $\mathcal{H}^s(f(A)) \le L^s \mathcal{H}^s(A)$）。
- **保零测度**：若 $\mathcal{L}^n(A) = 0$ 且 $f$ 是 Lipschitz 的，则 $\mathcal{L}^m(f(A)) = 0$——零测度集的 Lipschitz 像是零测度的。
- **几乎处处可微**：这正是 Rademacher 定理的内容。

**辨析｜易错点：** Lipschitz 连续 $\Rightarrow$ 一致连续 $\Rightarrow$ 连续，反方向都不成立。$f(x) = \sqrt{|x|}$ 在 $0$ 附近连续甚至一致连续，但导数无界，不是 Lipschitz；$f(x) = x^2$ 在任何紧区间上是 Lipschitz 的，但在整个 $\mathbb{R}$ 上不是。判定「局部 Lipschitz」与「全局 Lipschitz」要区分开。<span class="marginnote">函数 $x \mapsto x^2$ 在 $[0,1]$ 上 $\mathrm{Lip} = 2$，在 $\mathbb{R}$ 上 Lipschitz 常数发散——「局部 Lipschitz」是「每点有邻域是 Lipschitz」，远弱于全局控制。</span>

## 2 McShane 延拓：Lipschitz 函数的万能延拓

Lipschitz 函数不仅几乎处处可微，还拥有一种「把定义域扩到全空间」的能力——这是连续函数没有的奢侈品。

**核心概念（McShane 延拓定理）**：设 $A \subset \mathbb{R}^n$，$f: A \to \mathbb{R}$ 是 Lipschitz 函数（$L = \mathrm{Lip}(f)$）。则存在 $\mathbb{R}^n$ 上定义的 Lipschitz 函数 $\tilde f$，使得 $\tilde f|_A = f$ 且 $\mathrm{Lip}(\tilde f) = L$。显式构造为

$$
\tilde f(x) \;=\; \inf_{y \in A}\bigl\{ f(y) + L\,|x - y| \bigr\}
$$

这个公式的直觉是：$\tilde f(x)$ 是「所有从 $A$ 出发的 $L$-Lipschitz 下界函数在 $x$ 处的最大下界」。用下确界构造保证了它不高于任何合理延拓，而 Lipschitz 性质由 $L$ 控制。对向量值函数，可逐分量延拓。<span class="marginnote">McShane 延拓的公式本质是「以 $L$ 为斜率从 $A$ 上的点向外张锥」。它把 $f$ 延拓成整个空间上最「紧」的 Lipschitz 函数，是证明 Rademacher 定理的关键工具之一。</span>

**重点：延拓定理的价值在于，研究 Lipschitz 函数时可以假定它定义在全空间 $\mathbb{R}^n$ 上。** 这极大简化了后面面积公式、余面积公式的处理——局部性质可以延拓成整体，再把结果拉回原定义域。

## 3 Rademacher 定理：内容与证明骨架

**核心概念（Rademacher 定理）**：设 $U \subset \mathbb{R}^n$ 是开集，$f: U \to \mathbb{R}^m$ 是 Lipschitz 连续的。则 $f$ 在 $U$ 中 $\mathcal{L}^n$-几乎处处可微，即对几乎所有 $x \in U$，存在 $m \times n$ 矩阵 $Df(x)$ 使得

$$
\lim_{y \to x} \frac{f(y) - f(x) - Df(x)(y-x)}{|y - x|} \;=\; 0
$$

且 $Df(x)$ 满足 $|Df(x)| \le \mathrm{Lip}(f)$。对向量值函数，$m$ 个分量分别应用标量情形即可。<span class="marginnote">「几乎处处可微」意味着微分的定义排除了一个零测度集合——这正是上一节密度定理的语言。Rademacher 定理保证不可微点至多零测度，但不保证不可微点「少」到可数。</span>

证明分三步，骨架非常清晰：

**第一步，降为一维**：利用 Lipschitz 函数在几乎每条线上都绝对连续，对每个坐标方向应用一维的 Lebesgue 微分定理，得到 $f$ 沿坐标方向的偏导数几乎处处存在。
**第二步，方向导数到全导数**：取可数个有理方向，用 Lipschitz 条件与测度论证证明「沿所有有理方向的方向导数存在」蕴涵「几乎处处全导数存在」。这里用到密度定理：把「每个方向都微好」的点集取交集，余集是零测度的。
**第三步，延拓收尾**：利用 McShane 延拓把 $f$ 延拓到全空间，用卷积磨光（smoothing）构造逼近，说明微分矩阵就是梯度。<span class="marginnote">磨光序列 $f_\varepsilon = f * \rho_\varepsilon$ 光滑且 Lipschitz 常数一致有界，逐点逼近 $f$，它们的导数序列在 $L^\infty$ 弱-* 收敛到 $f$ 的分布导数，从而识别出 $Df$。这个「先磨光、再取极限」的技巧在 PDE 里同样常用。</span>

## 4 公式解析：Rademacher 定理的微分公式

Rademacher 定理的结论在应用中通常写成如下「链式」形式：对 Lipschitz 函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，几乎处处有

$$
f(y) - f(x) \;=\; Df(x)(y-x) + o(|y-x|), \qquad \text{a.e. } x
$$

逐项拆解：

- **$Df(x)(y-x)$（线性主部）**：这是 $f$ 在 $x$ 处的切映射作用在增量上的结果。$Df(x)$ 是 $m \times n$ 矩阵，其 $i$ 行第 $j$ 列是 $\partial f_i / \partial x_j$（几乎处处意义下存在）。
- **$o(|y-x|)$（余项）**：余项比增量更快地趋于 0。它的存在正是「可微」与「有偏导数」的区别——偏导数存在只保证沿坐标方向的行为，全导数存在要求余项沿所有方向一致趋于 0。
- **几乎处处（a.e.）**：等式只对 $x$ 属于某个全测度集合成立。不可微点（如 $f(x)=|x|$ 在 $0$ 处）允许存在，但不能占据正测度。
- **常数控制 $|Df(x)| \le \mathrm{Lip}(f)$**：微分矩阵的算子范数被 Lipschitz 常数控制，说明微分不放大超过函数本身的能力——这是 Lipschitz 条件在无穷小尺度的回声。

**重点：Rademacher 定理保证的是「几乎处处可微」，不是「可微点集是开的」。** 事实上不可微点可以是康托尔集那样的无处稠密集合，但它必须零测度。这个「几乎处处」的容忍度，让几何测度论能对 Lipschitz 映射整体地使用积分工具，而不必逐点操心光滑性。

## 5 Rademacher 定理的几何应用预览

Rademacher 定理是整棵几何测度论树的树干，从这里分出三条粗枝。

其一，**面积公式与余面积公式**（第 6 篇）的证明需要先知道 Lipschitz 映射几乎处处有 Jacobian $Jf(x) = \sqrt{\det(Df(x)^T Df(x))}$，而 Jacobian 的定义依赖几乎处处可微。

其二，**整流集与切空间**（第 5 篇）的核心是「被 Lipschitz 图像覆盖的集合」，Lipschitz 图像 $y = f(x)$ 的切空间恰好是 $Df(x)$ 的图形——Rademacher 定理保证了切空间几乎处处存在。

其三，**变分问题**（第 8、10 篇）中的能量泛函只对 Lipschitz 或 BV 函数有意义，而对这些函数做变分计算、导出 Euler–Lagrange 方程，第一步就是「几乎处处可微，从而可以求导」。

**比较表**：Lipschitz 与相邻函数类的对照。

| 函数类 | 连续性 | 几乎处处可微 | 典型代表 |
| --- | --- | --- | --- |
| 连续（$C^0$） | 有 | 不一定（Weierstrass 函数无处可微） | $x \mapsto \sum 2^{-k}\sin(4^k x)$ |
| 绝对连续（AC） | 一致 | 有（一维） | $x \mapsto x^2 \sin(1/x^2)$（修正后） |
| Lipschitz | 一致且受控 | 有（Rademacher） | 距离函数、压缩映射、测地线 |
| $C^1$ | 强于一致 | 处处（且导数连续） | 光滑参数曲线 |

## 6 Lipschitz 映射的测度像：一条链

Rademacher 定理与 Lipschitz 性质共同保证的「可微结构」，让 Lipschitz 映射在测度论里格外温和。把分散的性质串成一条链：

- **Lipschitz 保零测度**：$\mathcal{L}^n(A) = 0 \Rightarrow \mathcal{L}^m(f(A)) = 0$——用可数覆盖 + 半径缩放 $|f(x)-f(y)| \le L|x-y|$ 直接得。
- **Lipschitz 保维数**：$\dim_{\mathrm{H}} f(A) \le \dim_{\mathrm{H}} A$——因为 $\mathcal{H}^s(f(A)) \le L^s \mathcal{H}^s(A)$。
- **面积公式的 Lipschitz 版**（第 6 篇）：$\int g\, Jf = \int \sum g\, \mathrm{d}\mathcal{H}^n$——Jacobian 由 Rademacher 导数构成，几乎处处有意义。
- **链式法则**：两个 Lipschitz 映射的复合仍 Lipschitz，且 $D(g\circ f) = Dg(f)\cdot Df$ 几乎处处成立——Rademacher 保证两因子几乎处处存在。<span class="marginnote">链式法则的「几乎处处」版本是整流集面积计算的前提：参数化 Lipschitz 映射与度量函数的复合求导，都归结为这一条 Rademacher 级的链式法则。</span>

这条链说明 Lipschitz 类是被测度论「钦定」的映射类：可微结构、测度变换、维数控制三件事同时成立。这也是为什么从 Rademacher 定理出发能一举推出面积公式（第 6 篇）、整流集切空间（第 5 篇）的原因——**Lipschitz 是测度论意义上的「好映射」的最低门槛**。

**辨析｜易错点：** Lipschitz 保维数是「$\le$」，不是「$=$」：投影（Lipschitz）可以把二维集压成一维。等号需要额外条件（如面积公式中的几乎处处单射 + Jacobian 正定）。「保维数」只是上界，下界要由面积公式、射影定理（第 11 篇）等工具补足。

## 7 小结

- **Lipschitz 连续**：$|f(x)-f(y)| \le L|x-y|$，是「受控」的连续性；保维数、保零测度、几乎处处可微。
- **McShane 延拓**：Lipschitz 函数可延拓到全空间且 Lipschitz 常数不变，构造公式 $\inf_y\{f(y) + L|x-y|\}$