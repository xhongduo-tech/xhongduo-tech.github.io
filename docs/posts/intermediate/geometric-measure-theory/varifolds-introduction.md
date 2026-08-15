---
title: 变分几何（varifolds）引论
date: 2026-08-07
---

# 变分几何（varifolds）引论

<div class="epigraph">
<p>一个曲面并不只是它所在的位置——它还带着它在每一点的切空间。varifold 把这两者都当作质量来度量。</p>
<footer>—— 自 F. J. Almgren（弗雷德里克 · 阿尔姆格伦，1933–1997）与 W. K. Allard（威廉 · 阿拉德）的思想</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ W. K. Allard, *On the First Variation of a Varifold\*（1972） ｜ 2026-08-07</p>
</div>

## 为什么从 varifold 开始

前面几节处理的是「静止的集合」：测度、密度、整流集、切空间。但几何测度论最有生命力的部分，是**让曲面动起来**——在面积约束下收缩、变形、寻找极小曲面。经典的变分法要求曲面光滑，可肥皂膜在吹破的瞬间、极小曲面在出现奇点的时候，全都不是光滑的。**varifold（变分流形，字面「变化 + 流形」）** 是 Almgren 在 1960 年代引入的补救方案：**把「曲面」重新包装成一个测度，这个测度不仅记录「曲面占据哪些位置」，还记录「每个位置处曲面的切空间分布」。** 这样一来，曲面极限仍是 varifold，变分计算可以一路做到底而不跳出框架。<span class="marginnote">「varifold」=「varying + manifold」，由 Almgren 造词。它和下一节的「current（电流）」都是把几何对象「测度化」以换取紧致性的产物——区别在于 varifold 携带切空间分布，current 携带定向。</span>

## 1 从曲面到测度：位置 + 切空间的二重信息

一条光滑曲面 $M \subset \mathbb{R}^n$ 可以被编码成两重信息：

**位置**：$M$ 作为集合占据的空间；
**方向**：每点 $x \in M$ 处的切空间 $T_x M$。

若只保留位置，取 $\mu = \mathcal{H}^m \llcorner M$，则 $M$ 变成了「没有方向」的测度，极限下会丢失切空间信息——而切空间恰恰是计算面积、法向量、平均曲率的必需数据。varifold 的想法是**把切空间也当作测度的自变量**。

设 $G(n,m)$ 是 $\mathbb{R}^n$ 中所有 $m$ 维线性子空间构成的**格拉斯曼流形（Grassmannian）**。

**核心概念（$m$ 维 varifold）**：$\mathbb{R}^n$ 上的一个 **$m$ 维 varifold** 是 $G(n,m) \times \mathbb{R}^n$ 上的一个 Radon 测度 $V$，其中 $G(n,m) \times \mathbb{R}^n$ 中的点记作 $(S, x)$，$S$ 是切空间、$x$ 是位置。<span class="marginnote">把「曲面」重写为 $G(n,m) \times \mathbb{R}^n$ 上的测度，等于把每一点的切空间「摊」成一个独立的自由度。这个抬升让极限运算闭包：一串曲面的极限仍是一个 varifold，哪怕极限位置与极限切空间都面目全非。</span>

对「好」的 varifold，测度 $\|V\|(A) = V(G(n,m) \times A)$ 是位置空间的测度，称为 varifold 的**质量（mass）测度**。

## 2 整流 varifold 与 varifold 的紧致性

不是所有 varifold 都值得研究——测度论太宽了，需要挑出「由曲面长出来」的那一类。

**核心概念（整流 varifold）**：称 varifold $V$ 是 **$m$ 维整流 varifold（rectifiable varifold）**，如果存在 $m$ 维整流集 $M$ 与重数 $\theta$，使得对一切检验函数 $\phi(S, x)$，

$$
V(\phi) \;=\; \int_{M} \phi\bigl(T_x M,\, x\bigr)\, \theta(x)\; \mathrm{d}\mathcal{H}^m(x)
$$

即：质量测度 $\|V\| = \theta\, \mathcal{H}^m \llcorner M$，且切空间「几乎处处钉在」整流集的近似切空间 $T_x M$ 上。整流 varifold 是「带切空间的整流测度」。

**重点：varifold 的紧致性定理（Allard / Almgren）**：若一族 $m$ 维 varifold 的质量测度在任意紧集上一致有界（$\|V_i\|(B(0,R)) \le C_R$），则存在子列弱收敛到一个 $m$ 维 varifold。这条紧致性定理没有任何光滑性、定向性要求——极限自动是 varifold，即使它不再是整流 varifold。<span class="marginnote">这正是 varifold 比整流测度更进一步之处：整流集类对「面积一致有界的弱极限」不封闭（极限可能变成纯不可分的尘埃），而 varifold 类封闭。代价是：极限 varifold 可能失去「整流性」，正则性分析因此要处理更广的对象。</span>

**辨析｜易错点：** 弱收敛的 varifold 极限不一定保持整流性。例如一串圆盘越来越细密地皱成一条线，位置测度极限是一维的，但切空间信息可能散布——这不再是整流 varifold。这是「紧致 + 正则」这一对矛盾的经典体现：紧致性容易得到，正则性要额外挣。

## 3 第一变分：varifold 的导数

varifold 理论的核心工具是**第一变分（first variation）**——它回答「把 varifold 沿一个向量场流着推一下，质量怎么变」，是面积泛函的导数。

设 $X$ 是 $\mathbb{R}^n$ 上的光滑向量场，$\phi_t$ 是 $X$ 生成的流。对 varifold $V$，定义沿 $X$ 的变分 $V_t = (\phi_t)_\# V$（把 $V$ 的每个点按 $\phi_t$ 推前）。

**核心概念（第一变分）**：varifold $V$ 沿向量场 $X$ 的**第一变分**定义为

$$
\delta V(X) \;=\; \left.\frac{\mathrm{d}}{\mathrm{d}t}\right|_{t=0} \|V_t\|(B(0,R))\Big|_{R \to \infty} 的规范化
$$

更常用的是展开式：对整流 varifold，第一变分可以写成

$$
\delta V(X) \;=\; \int_M \mathrm{div}_{T_x M} X(x)\; \theta(x)\; \mathrm{d}\mathcal{H}^m(x)
$$

其中 $\mathrm{div}_{T} X$ 是向量场 $X$ 沿切空间 $T$ 的**切散度**。<span class="marginnote">切散度 $\mathrm{div}_{T} X$ 度量「沿着曲面方向，向量场 $X$ 是散开还是收缩」——它替代了经典变分里的平均曲率项。对嵌入的曲面，$\mathrm{div}_T X = \mathrm{div} X - H \cdot X$ 与平均曲率向量 $H$ 通过 $\mathrm{div} X$ 联系，经典公式是它的特例。</span>

## 4 公式解析：第一变分与平均曲率

展开 $\mathrm{div}_{T_x M} X(x)$ 到基底：设 $e_1, \dots, e_m$ 是 $T_x M$ 的正交基，则

$$
\mathrm{div}_{T_x M} X(x) \;=\; \sum_{i=1}^{m} \langle \nabla_{e_i} X(x),\, e_i \rangle
$$

逐项拆解：

- **$\nabla_{e_i} X(x)$（方向导数）**：向量场 $X$ 沿第 $i$ 个切方向 $e_i$ 的变化率。这是经典方向导数，Rademacher 定理保证在整流集上几乎处处有定义。
- **$\langle \cdot, \cdot\rangle$（投影到切方向）**：只保留导数在切方向上的分量。垂直方向的分量不改变「沿曲面」的质量，故不计入切散度。
- **求和 $\sum_{i=1}^m$**：对 $m$ 个切方向求和。直观上，$\mathrm{div}_T X$ 是「$X$ 的流在曲面内部是扩张还是收缩」的总体速率。
- **与平均曲率的关系**：当 $V$ 是光滑曲面 $M$ 的整流 varifold 且取 $X$ 为法向扰动，$\mathrm{div}_T X = - H \cdot X$，第一变分变为 $\delta V(X) = -\int H \cdot X \, \mathrm{d}\mathcal{H}^m$。**平均曲率向量 $H$ 正是「面积对法向扰动的一阶导数」，第一变分把经典几何的曲率观念推广到了任意 varifold。**

**重点：stationary（驻定）varifold 是变分问题的解概念。** 称 varifold $V$ 是 **stationary**，如果 $\delta V(X) = 0$ 对一切紧支撑向量场 $X$ 成立。直觉上，stationary 意味着「任何局部变形都不改变面积到一阶」，即面积临界点——极小曲面、肥皂膜、浸没子流形都是 stationary varifold。<span class="marginnote">stationary 只是「一阶必要条件」，不是「局部极小」。极小化序列的极限自动 stationary（由第一变分连续性），但要证明它「真是极小曲面」还要第二变分与正则性分析——这就是第 10 篇极小曲面正则性的入口。</span>

## 5 varifold 的应用与 Allard 正则性定理

varifold 理论最著名的成果是 **Allard 正则性定理**（1972）：若一个 $m$ 维 rectifiable varifold 是 stationary，且它的质量测度满足「在任意点的 $m$ 维密度有正下界」（即曲面在每个点附近都有正面积密度），则除一个 $\mathcal{H}^m$-零测度集合外，它其实是一个 $C^{1,\alpha}$ 光滑的 $m$ 维子流形。<span class="marginnote">Allard 定理说的是「足够好的 stationary varifold 自动光滑」：从「几乎处处有切空间」出发，靠 stationarity 的椭圆性推出 Hölder 连续性，最终得到 $C^{1,\alpha}$。这是几何测度论里「正则性」的样板定理，与 De Giorgi 极小曲面正则性、以及 PDE 中的 Schauder 估计同源。</span>

varifold 的典型应用场景：

- **肥皂膜与极小曲面**：把肥皂膜建模成 stationary varifold，允许破裂、允许分支，紧致性保证最小化序列有极限。
- **几何流（mean curvature flow）**：平均曲率流在有限时间可能产生奇点，用 varifold（或 Brakke 流）作为奇点后的弱解继续演化，是当前活跃的方向。
- **图像处理与几何测度**：图像中边缘检测、数字几何中的表面积估计，本质是「测度 + 切空间」的重建问题，varifold 提供了理论框架。

**比较表**：varifold 与整流集的对照。

| 特征 | $m$ 维整流集 | $m$ 维整流 varifold |
| --- | --- | --- |
| 数据 | 位置集合 $M$ | 位置 + 切空间 $(T_xM, x)$ |
| 测度 | $\mathcal{H}^m \llcorner M$ | $\theta \mathcal{H}^m$，切空间作变量 |
| 方向信息 | 派生（近似切空间） | 内置（作为测度坐标） |
| 极限封闭性 | 不封闭 | 封闭（质量一致有界） |
| 变分工具 | 无内置 | 第一变分 $\delta V$ |

## 6 varifold 与平均曲率流

varifold 框架最活跃的应用之一是**平均曲率流（mean curvature flow）**的弱解。对一族浸入曲面 $M_t$，平均曲率流要求

$$
\partial_t x \;=\; H(x)
$$

即每个点沿平均曲率方向以曲率速率运动。光滑解在有限时间内会形成奇点（颈缩、尖点），经典解无法继续。**Brakke 流**（1980 年代）用 varifold 语言写出平均曲率流的弱形式：一族 varifold $V_t$ 满足一个变分不等式

$$
\frac{\mathrm{d}}{\mathrm{d}t}\|V_t\|(\phi) \;\le\; \int \left( \mathrm{div}_{T}\phi - |H|^2 \phi \right)\; \mathrm{d}\|V_t\|
$$

对一切非负检验函数 $\phi$ 成立。这个不等式把「面积单调下降」与「曲率平方积分」同时编码，允许解在奇点处失去光滑性却仍保持测度论意义下的演化。<span class="marginnote">Brakke 流的存在性正是靠 varifold 的紧致性（第 2 节）建立的：把光滑解的序列取弱极限，极限自动是 varifold 流。与 Plateau 问题不同，平均曲率流的「极限」不能保证唯一性，奇点后的演化路径选择至今是研究重点。</span>

**重点：varifold 让「奇点后的几何流」有了严格的弱解。** 这是「测度化换取紧致性」思想在动态问题中的延续：静态时给极限一个身份，动态时给奇点一个继续演化的载体。

## 7 小结

- **varifold**：$G(n,m) \times \mathbb{R}^n$ 上的 Radon 测度，同时编码位置与切空间；**整流 varifold** 由整流集 + 重数给出。
- **紧致性**：质量一致有界的 varifold 族有弱收敛子列；极限自动是 varifold（可能非整流）。
- **第一变分** $\delta V(X) = \int \mathrm{div}_{T_xM} X \, \theta \, \mathrm{d}\mathcal{H}^m$，是面积泛函的导数；stationary varifold 是面积临界点。
- 光滑曲面的法向扰动下，第一变分退化为 $\int H \cdot X$，平均曲率向量 $H$ 由此进入弱框架。
- **Allard 正则性定理**：stationary + 密度正下界 ⟹ 几乎处处 $C^{1,\alpha}$