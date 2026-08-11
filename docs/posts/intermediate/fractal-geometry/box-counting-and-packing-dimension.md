---
title: Box-counting 维数与 Packing 维数
date: 2026-08-11
---

# Box-counting 维数与 Packing 维数

<div class="epigraph">
<p>大不列颠的海岸线有多长？——统计自相似与分数维。</p>
<footer>—— 本华 · 曼德博（Benoit Mandelbrot），1967 年同名论文标题</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 分形几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Box-counting 维数开始

Hausdorff 维数在数学上完美，用起来却痛苦：要算 $\inf$、要取极限、还要验证测度性质。对真实数据——一段海岸线的地图、一张卫星照片、一张待压缩的图片——你根本不可能做这样的运算。这时需要一把「工程上可用」的尺子：**数格子**。把平面铺上边长为 $\delta$ 的方格，数一数有多少格碰到集合，记为 $N_\delta(E)$；然后看 $N_\delta(E)$ 随 $\delta$ 缩小的增长速度。这就是 box-counting 维数，也叫**盒子维数 / 闵可夫斯基维数**。它牺牲了一部分理论优雅，却换来几乎所有实际应用——从图像压缩到材料断裂面的粗糙度测量，都用它。本节的 Packing 维数则负责「补课」：把 box 维数在理论上丢掉的严谨性（比如对可数并的不稳定性）抢救回来。

## 1 数格子：最实用的维数

对 $\mathbb{R}^n$ 中的有界集合 $E$，记 $N_\delta(E)$ 为「直径不超过 $\delta$、且能覆盖 $E$ 的集合的最少个数」，或等价地「与 $E$ 相交的边长 $\delta$ 的网格立方体个数」。对大多数集合二者给出的维数相同。定义：

**Box-counting 维数（box-counting dimension）：**

$$\dim_B E = \lim_{\delta \to 0} \frac{\log N_\delta(E)}{-\log \delta}$$

直觉很直白：若 $N_\delta(E) \approx c\,\delta^{-s}$，则 $\log N_\delta \approx \log c + s \log(1/\delta)$，斜率就是 $s$。下图以康托尔集为例：$\delta = 1/3$ 时数出 2 格，$\delta = 1/9$ 时数出 4 格，格子数按 $\delta^{-0.63}$ 增长。

![康托尔集的 box-counting 计数](/images/fractal-geometry/box-counting-dimension-cantor.svg)

几个标准例子：直线段 $\dim_B = 1$；平面区域 $\dim_B = 2$；康托尔集 $\dim_B = \log 2/\log 3$；科赫曲线 $\dim_B = \log 4/\log 3 \approx 1.26$——每把尺度缩小 $1/3$，曲线变长 $4$ 倍，正是曼德博口中「海岸线随尺子变长」的数学化。<span class="marginnote">刘易斯 · 理查森（Lewis Fry Richardson）在 1961 年发现：用越来越短的尺子量海岸线，测得的总长按幂律增长。曼德博 1967 年用 log–log 图的斜率给出了分数维，直接催生了整门学科。</span>

## 2 上下 box 维数与不收敛的尴尬

问题在于：极限不一定存在。$\delta \to 0$ 时 $\log N_\delta / (-\log \delta)$ 可能上下震荡，于是有了**下 box 维数**与**上 box 维数**：

$$\underline{\dim}_B E = \liminf_{\delta \to 0} \frac{\log N_\delta(E)}{-\log \delta}, \qquad \overline{\dim}_B E = \limsup_{\delta \to 0} \frac{\log N_\delta(E)}{-\log \delta}$$

当两者相等时，才把公共值记为 $\dim_B E$。<span class="marginnote">即使对不少「正经」集合，上下 box 维数也不相等——有人构造出 $\underline{\dim}_B E \ne \overline{\dim}_B E$ 的紧集。好在绝大多数教科书中的分形（康托尔集、科赫曲线、谢尔宾斯基三角）都「行为良好」，上下相等。</span>

更本质的缺陷是：**box 维数不具备可数稳定性**。可数个点的并的 Hausdorff 维数是 0，但 $\mathbb{Q} \cap [0,1]$（有理数，可数集）的 box 维数却是 1——因为有理数稠密，任何尺度下它都填满整个区间。一个「本应 0 维」的集合，box 维数却给出 1。这是 box 维数在理论上最大的硬伤。

## 3 Packing 维数：补课方案

Packing 维数把「覆盖」换成「填充」，再补上可数稳定化。先定义 **$s$ 维 Packing 测度**：固定 $\delta > 0$，考虑两两不相交、中心在 $E$ 内、半径不超过 $\delta$ 的一族闭球 $\{B_i\}$，记

$$P_\delta^s(E) = \sup \left\{ \sum_i (2 r_i)^s : \{B_i\} \text{ 如上} \right\}, \qquad P_0^s(E) = \lim_{\delta \to 0} P_\delta^s(E)$$

注意 $P_0^s$ 对可数并仍不老实，于是再做一次可数稳定化：$P^s(E) = \inf\{ \sum_i P_0^s(E_i) : E \subset \bigcup_i E_i \}$。这个 $P^s$ 才是真正的外测度，叫 **Packing 测度**。再由与 Hausdorff 维数完全相同的「临界值」机制定义 **Packing 维数**：

$$\dim_P E = \inf\{ s : P^s(E) = 0 \} = \sup\{ s : P^s(E) = \infty \}$$

**Packing 维数是「把 box 维数的外层取极限换成测度化」的结果**，它同时保住了 box 维数的实用性直觉（填充球的规模）与 Hausdorff 维数的可数稳定性。

## 4 公式解析：$\dim_B$ 的算式

$$
\dim_B E = \lim_{\delta \to 0} \frac{\log N_\delta(E)}{-\log \delta}
$$

三步拆解：

- **第一，$N_\delta(E)$ 是计数**：它与 $E$ 相交的网格立方体个数，或覆盖 $E$ 的最小集合数。工程上取前者——逐像素判断「这个格子被碰到没有」，算法廉价。
- **第二，$\log N_\delta$ 与 $\log(1/\delta)$ 是两边取对数**：幂律 $N_\delta \approx c\delta^{-s}$ 在取对数后变成线性关系 $\log N_\delta = \log c + s\log(1/\delta)$，维数就是斜率。
- **第三，$\delta \to 0$ 的极限是把斜率定准**：真实数据只有有限多个尺度可用，于是实践中退化为「在可用的几个 $\delta$ 上做线性回归取斜率」——这就是 log–log 图（log-log plot）的由来。

一句话：**box 维数就是「网格变细时，碰到的格子数在 log–log 图上的斜率」**。它把「维数」从测度论概念降级成了统计量——正是这份「降级」，让它能被任何会数数的人用起来。

## 5 辨析｜易错点

- **$\dim_H E \le \underline{\dim}_B E \le \overline{\dim}_B E$**：Hausdorff 维数永远不超过上下 box 维数——因为覆盖 $E$ 的最小盒子数就是最好的 $\delta$-覆盖，box 维数天然「更宽松」。绝大多数分形上等号成立，但（如 $\mathbb{Q}\cap[0,1]$）会严格小于。
- **可数稳定性是分水岭**：$\dim_H$ 与 $\dim_P$ 满足「可数并取上确界」，box 维数不满足。**这决定了「谁才是理论上的维数」**：凡依赖可数可加性的论证，都得退回 Hausdorff 或 Packing。
- **工程与理论脱钩**：数据驱动场景几乎只用 box 维数，但必须警惕伪分形——有限尺度的 log–log 图斜率可以处处光滑地变化，测出的「维数」未必对应任何测度论意义下的维数。
- **离散化误差**：网格对齐会引入偏差，实践中常用多个偏移取平均；网格方向也会影响结果，对自仿集尤其严重（见第 10 篇《自仿集》）。

## 6 小结

- Box 维数 $\dim_B E = \lim_{\delta\to0} \log N_\delta(E)/(-\log\delta)$，是「数格子」的 log–log 斜率，工程上最实用。
- 上下 box 维数可以不等；box 维数不具可数稳定性，对 $\mathbb{Q}\cap[0,1]$ 这类可数稠密集给出错误的 1。
- Packing 测度用「不相交球填充」构造，再可数稳定化，得到 Packing 维数，满足 $\dim_H \le \dim_P \le \overline{\dim}_B$ 且具备可数稳定性。

在下一节，我们把构造从「外在的尺子」转回「内在的复制」：自相似集与迭代函数系统。你会看到，很多分形根本不是「画出来的」，而是「同一个操作迭代出来的」——而它们的维数，一行公式就能算完。
