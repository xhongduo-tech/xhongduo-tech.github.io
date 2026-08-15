---
title: Hausdorff 测度与 Hausdorff 维数
date: 2026-08-07
---

# Hausdorff 测度与 Hausdorff 维数

<div class="epigraph">
<p>云雾不是球，山不是锥，海岸线不是圆，树皮不是光滑的，闪电也不是直线。</p>
<footer>—— 伯努瓦 · 曼德尔布罗（Benoit B. Mandelbrot），*The Fractal Geometry of Nature\*</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ P. Mattila, *Geometry of Sets and Measures in Euclidean Spaces\*, Ch.3–4 ｜ 2026-08-07</p>
</div>

## 为什么从 Hausdorff 测度开始

几何测度论回答一个朴素的问题：**粗糙的、不光滑的、到处是碎片的集合，该用怎样的尺子去量它的「大小」？** Lebesgue 测度是一把好尺子，但它只提供整数的维数——一段线段是一维的，一片曲面是二维的。可世界上偏偏存在比线更「厚」、比面更「薄」的集合：康托尔三分集、科赫雪花、海岸线。它们长度为零（一维测度为零）、面积也为零（二维测度为零），却明显「比点更多」。要定量刻画这种介于维度之间的量，需要两件新工具：一把可以调节刻度的尺子，以及一个由这把尺子自然导出的维数概念。这就是本节的主题——**Hausdorff 测度**与**Hausdorff 维数**。<span class="marginnote">康托尔集、科赫曲线等经典分形我们在第二级《分形几何》专题中系统展开，本专题只用到它们的构造作例子。</span>

## 1 一维的尺子量不出二维的粗糙

先回顾一个经验：**用长度量面积必然得到无穷大，用面积量线段必然得到零。** 把一根单位线段看成面积对象去量，它的二维 Lebesgue 测度是 0；把一张正方形看成长度对象去量，它的一维测度是无穷。这说明「量测度」与「选维数」必须配套——尺子的刻度如果与对象的内在维数不匹配，测量结果要么是 0，要么是 ∞，全都失去分辨力。

Hausdorff 的想法是：**把尺子的刻度参数化，引入实数参数 $s$，让尺子可以连续地「变细」或「变粗」。** 当 $s$ 从小往大移动时，一个给定集合的 $s$ 维测度会经历「从 ∞ 跳到 0」的临界转变，这个临界点本身就是一个几何量——它正是集合的 Hausdorff 维数。因此 Hausdorff 测度不是一把单独的尺子，而是一整族尺度连续的尺子，维数是其中「最灵敏」的那一把。

**核心概念（Hausdorff $s$ 维测度）**：对 $0 \le s \lt  \infty$，集合 $A \subset \mathbb{R}^n$ 的 $s$ 维 Hausdorff 测度 $\mathcal{H}^s(A)$，是用「直径的 $s$ 次幂」作为成本、覆盖 $A$ 的最便宜价格。<span class="marginnote">「直径的 $s$ 次幂」捕捉了关键直觉：一个直径 $r$ 的小块，在 $s$ 维意义下「应当」有大小 $r^s$。对 $s=1$ 它回到长度，对 $s=2$ 它回到（差个常数倍的）面积。</span>

## 2 构造：先限制覆盖块的大小，再让限制消失

构造分两步，把「用多大的块覆盖」与「覆盖得多便宜」分离。

第一步，对固定的小尺度 $\delta > 0$，只允许用直径不超过 $\delta$ 的集合去覆盖 $A$，记

$$
\mathcal{H}^s_\delta(A) \;=\; \inf \left\{ \sum_{i=1}^{\infty} (\mathrm{diam}\, E_i)^s :\; A \subset \bigcup_{i=1}^{\infty} E_i,\;\; \mathrm{diam}\, E_i \le \delta \right\}
$$

这里的下确界取遍所有可数覆盖 $\{E_i\}$，覆盖块的直径都限制在 $\delta$ 以内。$E_i$ 可以是任意集合，不要求是球——因为「球」这个形状在纯度量的框架里不是本质的。

第二步，让 $\delta$ 缩小、允许的块越来越小。**注意：块越小，能套住 $A$ 的方案越多，下确界越可能变小**，所以 $\mathcal{H}^s_\delta(A)$ 随 $\delta$ 递减，于是定义

$$
\mathcal{H}^s(A) \;=\; \lim_{\delta \to 0} \mathcal{H}^s_\delta(A) \;=\; \sup_{\delta > 0} \mathcal{H}^s_\delta(A)
$$

极限存在（允许等于 $+\infty$），因为 $\delta \mapsto \mathcal{H}^s_\delta(A)$ 是递减函数。<span class="marginnote">为什么先限制 $\delta$ 再取极限？直觉在于：一条锯齿曲线用粗线段量，会高估它的长度；只有让覆盖块无限细，才能看到「真正的长度」。限制 $\delta$ 就是强迫测量分辨率趋于无穷。</span>

## 3 公式解析：把「直径的 $s$ 次幂」拆开看

定义的核心是 $\sum (\mathrm{diam}\, E_i)^s$。逐项拆解：

- **第一项，$\mathrm{diam}\, E_i$（直径）**：$E_i$ 中任意两点距离的上确界。它是「$E_i$ 有多大」的纯度量描述，不依赖坐标轴、不依赖形状。
- **第二项，$s$ 次幂 $(\mathrm{diam}\, E_i)^s$**：这是整把尺子的刻度。当 $s=1$ 时它退回「直径之和」，恰好是测量长度的传统方式；当 $s=2$ 时它近似「直径平方」，量级上对应面积。
- **第三项，求和与下确界**：对所有可数覆盖求和，再对所有覆盖方案取下确界——「取最便宜的方案」。这个「最便宜」让 $\mathcal{H}^s$ 成为可数子可加、单调的度量外测度，从而在集合类上导出真正意义上的测度。
- **第四步，极限**：$s$ 固定时，$\delta \to 0$ 的下确界之极限给出「高分辨率下的真实价格」。

把 $s$ 当作变量观察一个关键现象：**若 $\mathcal{H}^s(A) \lt  \infty$ 且 $t > s$，则 $\mathcal{H}^t(A) = 0$；若 $\mathcal{H}^s(A) > 0$ 且 $t \lt  s$，则 $\mathcal{H}^t(A) = \infty$。** 这是因为覆盖成本 $\sum (\mathrm{diam}\, E_i)^t$ 与 $(\mathrm{diam}\, E_i)^{t-s}$ 同阶缩放，而块直径趋于 0，指数差 $t-s>0$ 会把成本压到 0。这个「单调的阈值行为」正是维数定义的依据。

**重点：对任何集合 $A$，函数 $s \mapsto \mathcal{H}^s(A)$ 最多只有一个「0 到 ∞ 的跳跃点」。** 测度要么在此点之前恒为 ∞、之后恒为 0，要么恰好在此点取有限正数——这个跳跃点就是维数。

## 4 Hausdorff 维数：把阈值定义成维数

**核心概念（Hausdorff 维数）**：集合 $A$ 的 Hausdorff 维数是

$$
\dim_{\mathrm{H}} A \;=\; \inf \{\, s \ge 0 : \mathcal{H}^s(A) = 0 \,\}
$$

即让 $s$ 维测度塌缩为零的所有 $s$ 的下确界。等价地，它是 $\mathcal{H}^s(A) = \infty$ 与 $\mathcal{H}^s(A) = 0$ 之间的分界：当 $s \lt  \dim_{\mathrm{H}} A$ 时 $\mathcal{H}^s(A) = \infty$（大尺子量，测度发散）；当 $s > \dim_{\mathrm{H}} A$ 时 $\mathcal{H}^s(A) = 0$（小尺子量，测度消失）。在临界点本身，$\mathcal{H}^{s_0}(A)$ 可以取 $0$、$+\infty$ 或有限正数，三种情况都可能发生。<span class="marginnote">例如一条光滑曲线 $C$ 有 $\dim_{\mathrm{H}} C = 1$，但 $\mathcal{H}^1(C)$ 可能等于 0（比如康托尔集式的不可求长曲线）。「维数 = 1」与「一维测度为正」是两回事。</span>

Hausdorff 维数天然满足几条理想的性质：单调性（$A \subset B \Rightarrow \dim_{\mathrm{H}} A \le \dim_{\mathrm{H}} B$）、可数稳定性（可数并的维数等于维数的上确界）、以及**同胚不变量之上的更强性质——等距与相似变换下的不变性**。<span class="marginnote">维数对等距不变，但注意它不必对一般同胚不变：一个把单位正方形扭成锯齿带的双-Lipschitz 映射不会改变维数，而任意连续映射可能把维数推高到任意值。</span> 这些性质使它成为「集合有多粗糙」的稳健度量。

## 5 正则化常数：与 Lebesgue 测度接轨

Hausdorff 测度与 Lebesgue 测度不是两套互相独立的制度：在整数维数 $s = n$ 时，两者只差一个常数。事实上

$$
\mathcal{H}^n \;=\; c_n \, \mathcal{L}^n, \qquad c_n \;=\; \frac{\omega_n}{2^n} \;=\; \frac{\pi^{n/2}}{2^{n}\, \Gamma\left(\frac{n}{2}+1\right)}
$$

其中 $\omega_n$ 是单位球的体积，$c_n$ 是**正则化常数**。$\mathcal{H}^n$ 用「直径的 $n$ 次幂」当成本，而 $n$ 维球被 $n$ 维立方体覆盖时，直径与边长差一个因子 $2$，这解释了 $2^{-n}$ 的来源。有了这个常数，整数维情形下 Hausdorff 测度与 Lebesgue 测度就只差一个正倍数，两边结论可以互相翻译。

**辨析｜易错点：** 不要把「$\mathcal{H}^n = c_n \mathcal{L}^n$」理解成「Hausdorff 测度就是 Lebesgue 测度换个名字」。两者定义机制不同：$\mathcal{H}^n$ 是覆盖性的、对任意集合（包括病态集）都有定义的度量外测度；$\mathcal{L}^n$ 是 Carathéodory 外测度经可测性理论构造的。相等只是一个「数值恒等式」，且只在整维时成立——对分数维 $s$，两者根本无法比较，因为 $\mathcal{L}^n$ 根本没有 $s$ 维的版本。

## 6 实例：康托尔三分集的维数

康托尔三分集 $C$：把 $[0,1]$ 去掉中间的开区间 $(1/3, 2/3)$，再对剩下的两段各自去掉中间三分之一，无限操作后留下的集合。$C$ 与 $[0,1]$ 等势（所以不可数），但 $\mathcal{L}^1(C) = 0$。

用相似维数直接猜：$C$ 由两片自相似拷贝拼成，每片是整体的 $1/3$，若 $\mathcal{H}^s$ 在维数临界点表现出「自相似测度」的均匀性，则要求

$$
1 = \mathcal{H}^s(C) = 2 \cdot \left(\frac{1}{3}\right)^{s} \mathcal{H}^s(C)
$$

（左式到右式是「覆盖分成两半，每半成本缩放到 $3^{-s}$」的自相似方程）。由此解得

$$
2 \cdot 3^{-s} = 1 \quad\Longrightarrow\quad s = \frac{\log 2}{\log 3} \approx 0.6309
$$

**辨析｜易错点：** 上面的推导是「猜测维数」而不是「证明维数」。它假定了两份拷贝不重叠、且 $\mathcal{H}^s(C)$ 是有限正值，这两点都需要严格验证；但作为直觉，自相似方程给出的恰好是正确的 $\dim_{\mathrm{H}} C = \log 2 / \log 3$。更严格的「维数 ≥」需要 Frostman 引理（本节后面第 7 篇），「维数 ≤」用一列长度 $3^{-k}$ 的小区间覆盖即可。<span class="marginnote">严格证明 $\dim_{\mathrm{H}} C \le \log 2/\log 3$：第 $k$ 步留下的 $2^k$ 段区间每段长 $3^{-k}$，于是 $\mathcal{H}^s_{3^{-k}}(C) \le 2^k \cdot (3^{-k})^s = (2 \cdot 3^{-s})^k$，当 $s = \log 2/\log 3$ 时恰为 1。</span>

**比较表**：三种「维数」在同一集合上的表现。

| 概念 | 定义方式 | 康托尔集 $C$ | 单位正方形 |
| --- | --- | --- | --- |
| 拓扑维数 | 覆盖的序（Lebesgue 维数） | 0 | 2 |
| Lebesgue 测度维数 | $\mathcal{L}^d(A)>0$ 的最小整数 $d$ | 无（测度恒 0） | 2 |
| Hausdorff 维数 | $\mathcal{H}^s(A)=0$ 的阈值 | $\log 2/\log 3$ | 2 |

## 7 小结

- **Hausdorff $s$ 维测度**用「直径的 $s$ 次幂」作覆盖成本，先限制覆盖块直径 $\le \delta$、再取 $\delta \to 0$ 的极限：$\mathcal{H}^s(A) = \lim_{\delta\to 0} \mathcal{H}^s_\delta(A)$。
- **单调阈值**：$s \mapsto \mathcal{H}^s(A)$ 从 $\infty$ 跳到 $0$ 至多一次，这个跳跃点定义了 **Hausdorff 维数** $\dim_{\mathrm{H}} A$。
- 整数维接轨：$\mathcal{H}^n = c_n \mathcal{L}^n$，$c_n = \omega_n / 2^n$；分数维情形 Lebesgue 测度完全失效。
- 康托尔三分集 $\dim_{\mathrm{H}} C = \log 2 / \log 3$，且 $\mathcal{L}^1(C) = 0$