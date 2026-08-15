---
title: Levi 形式与强伪凸超曲面
date: 2026-08-07
---

# Levi 形式与强伪凸超曲面

<div class="epigraph">
<p>Levi 形式是超曲面的「曲率仪」：它不关心法向的弯折，只测量复切方向的鼓起与凹陷。</p>
<footer>—— 仿 欧金尼奥 · 伊莱维（Eugenio Elia Levi），《多复变函数与解析超曲面》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从超曲面上的 Levi 形式开始

第 7 篇我们见过 Levi 形式——那时它作为**区域伪凸性**的边界判据登场。现在把它独立出来：**Levi 形式本身就是超曲面 $M \subset \mathbb{C}^n$ 的内蕴曲率**，不依赖任何「内部区域」。对 CR 流形而言，Levi 形式回答了：**复切方向上的二阶几何是「凸」还是「凹」**——它决定 CR 函数的延拓性、$\bar\partial_b$ 方程的正则性、以及超曲面的几何分类。**强伪凸超曲面**（Levi 形式正定）是其中最「好」的一类，几乎拥有全部理想性质。<span class="marginnote">Levi 形式在抽象 CR 流形上也能定义（不嵌入 $\mathbb{C}^n$）：取单位法向 $N$，$\mathcal L(X, Y) = \langle [X, JY], N\rangle$。嵌入情形它正是第 7 篇的复 Hessian 限制。所以 Levi 形式是「CR 流形的曲率张量」，正定/半正定/不定的符号分类即超曲面的几何分类。</span>

## 1 Levi 形式在超曲面上的再定义

设 $M = \{\rho = 0\} \subset \mathbb{C}^n$，$p \in M$。**Levi 形式**是复切空间 $H_p M$ 上的 Hermitian 形式：

$$
\mathcal L_p(X, Y) = \left\langle \partial\bar\partial \rho(p), X \wedge \bar Y \right\rangle, \qquad X, Y \in H_p M
$$

即 $\mathcal L_p(w) = \sum_{j,k} \partial^2\rho/\partial z_j\partial\bar z_k\, w_j \bar w_k$（$w \in H_p M$）。它的**符号**（正定/半正定/不定）不依赖定义函数 $\rho$ 的选择（乘以正因子不变），因而是 $M$ 的几何不变量。<span class="marginnote">关键对比（再次强调，因为它太重要）：<strong>Levi 形式只在复切方向上取值</strong>。实 Hessian 在 $M$ 的<strong>实切方向</strong>上取值——那是黎曼几何的曲率；Levi 形式只取复切方向——那是 CR 几何的曲率。一个超曲面可以实凸（实 Hessian 正定）而 Levi 形式不定，反之亦然。<strong>复切方向才是多复变的「生长方向」。</strong></span>

## 2 强伪凸超曲面：定义与等价刻画

**强伪凸超曲面（strongly pseudoconvex hypersurface）**：$M$ 的 Levi 形式在每点**正定**。

等价刻画：

1. **几何**：$M$ 局部可写成 $M = \{ \rho \lt  0 \} \cap \{\rho = 0\}$ 的边界，其中 $\rho$ 在边界邻域内**强 psh**（复 Hessian 正定）。
2. **法向曲率**：对 $p \in M$ 与单位法向 $N$，$\mathcal L_p$ 正定 ⟺ $M$ 在 $p$ 处沿复切方向「鼓向 $-\nabla\rho$ 一侧」——想象 $\mathbb{C}^n$ 中的球面 $\{|z|^2 = 1\}$：它是强伪凸的（Levi 形式正定），因为球面在每个复切方向都「向外鼓」。
3. **CR 版本**：对抽象 CR 流形，$\mathcal L$ 正定给出「strongly pseudoconvex CR manifold」。

**标准例子**：球面 $S^{2n-1} = \{|z| = 1\} \subset \mathbb{C}^n$ 强伪凸；双曲空间型的边界、以及任何强伪凸域的边界都是。<span class="marginnote">球面的 Levi 形式：取 $\rho = |z|^2 - 1$，复 Hessian 是恒等矩阵，在复切方向上正定——强伪凸。而 $\mathbb{C}^n$ 中的「马鞍面」$M = \{\mathrm{Im}\, z_n = \sum |z_j|^2\}$（Heisenberg 型）的 Levi 形式符号取决于 $\sum |z_j|^2$ 前的系数——正系数是强伪凸，负系数是强凹。</span>

## 3 为什么强伪凸如此「好」

强伪凸超曲面之所以统治边界理论，是因为它让一切「严格性」都成立：

**（1）CR 函数局部可延拓**：强伪凸超曲面上的 CR 函数在每点局部可延拓为（一侧的）全纯函数——这是 **Lewy 延拓定理**（§4 公式解析）。理由：Levi 正定保证「在复切方向沿法向有可积结构」。

**（2）$\bar\partial_b$ 方程的次椭圆正则性**：在强伪凸 CR 流形上，$\bar\partial_b$ 及其 Kohn Laplacian $\square_b = \bar\partial_b\bar\partial_b^* + \bar\partial_b^*\bar\partial_b$ 满足**次椭圆估计**（增益 $1/2$ 阶，同第 21 篇），从而有有限维上同调与光滑解——Kohn 的奠基定理。<span class="marginnote">对比一般伪凸 CR 流形：Levi 形式半正定但非正定时，$\square_b$ 可能不是次椭圆的，甚至 $H^1$ 可以无限维（Folland–Kohn 的退化例子）。强伪凸 = 次椭圆 = 一切正则性定理的通行证，这个铁律从 $\bar\partial$ 到 CR 一路贯彻。</span>

**（3）边界值的全纯延拓**：强伪凸域边界的 CR 函数可整体延拓到域内（Khenkin、Rossi、Boggess 等），这是下一组末篇的主题。

## 4 公式解析：Lewy 延拓定理

**定理（Lewy，1956）**：设 $M \subset \mathbb{C}^n$ 是强伪凸超曲面，$p \in M$。则 $M$ 上每个光滑 CR 函数 $f$ 在 $p$ 附近可唯一延拓为 $M$ 的**伪凸一侧**上的全纯函数 $F$（$F|_{M} = f$ 在 $p$ 附近）。

证明的核心是构造局部积分表示，其中 Levi 正定性通过如下**不等式**登场：

$$
\sum_{j,k} \frac{\partial^2 \rho}{\partial z_j \partial \bar z_k} w_j \bar w_k \;\geq\; \delta |w|^2, \qquad w \in H_p M
$$

- **第一步，看不等式的作用**：Levi 正定 ⟹ 存在 $\delta > 0$，使复切方向上二阶项一致有下界。这保证「延拓核」$F(z) = \int_M f(\zeta) \Omega(z,\zeta)$ 中的积分核在复切方向上是**可控的正主值**，积分收敛且满足全纯方程。
- **第二步，为什么只是局部**：不等式在**单点**成立，由连续性在 $p$ 的**小邻域**内一致成立——所以延拓是局部的。整体延拓需要更强的条件（强伪凸域的全局性）。
- **第三步，延拓的唯一性**：$F$ 在伪凸一侧全纯、限制回 $M$ 等于 $f$，由全纯函数的唯一性定理，$F$ 唯一。**CR 函数在强伪凸边界上「自动」变成全纯函数**——这是 CR 理论最具戏剧性的结论。

## 5 辨析与延伸：强伪凸超曲面的五个要点

**辨析 1：强伪凸超曲面的「强」只指 Levi 形式**。强伪凸 = Levi 形式正定。它与区域的「强伪凸域」一致（若 $M$ 是 $D$ 的边界）。**但 CR 版本的强伪凸不依赖区域——它是 CR 流形内蕴的性质**。<span class="marginnote">球面 $S^{2n-1}$ 是标准例子：Levi 形式正定。而 Heisenberg 型 $M=\{v=|z|^2\}$ 是强凹（Levi 负定）。Levi 形式的符号把 CR 流形分成「凸族」与「凹族」。</span>

**辨析 2：Levi 形式的符号为什么是 CR 不变量**。换定义函数 $\rho \to e^{\psi}\rho$，Levi 形式在复切方向上乘正因子——符号不变。**符号（正/负/不定）是 CR 几何的分类标签**，正如 Ricci 曲率符号是黎曼几何的分类标签。

**辨析 3：Lewy 延拓定理的「一侧性」**。强伪凸超曲面的 CR 函数只能延拓到**伪凸一侧**（区域内部），不能延拓到另一侧。这是「延拓的方向性」——Levi 正定选出一个「内部」，负定选出一个「外部」。**方向由 Levi 形式符号决定**。

**辨析 4：次椭圆性与强伪凸的绑定**。$\bar\partial_b$ 的 Kohn Laplacian $\square_b$ 在强伪凸 CR 流形上次椭圆（$1/2$ 阶），在一般 CR 流形上可能完全退化。**次椭圆 = 强伪凸的「正则性签名」**——这条规律贯穿整个 CR 理论。

**误区清单**：

- **误区 1**：以为「强伪凸超曲面一定闭」。
  正解：可以是开的 CR 流形；强伪凸是 Levi 形式的性质，与紧致性无关。
- **误区 2**：以为「Levi 形式与定义函数无关」。
  正解：Levi 形式的值依赖定义函数，只有符号（在复切方向）不依赖。
- **误区 3**：以为「CR 函数可延拓到两侧」。
  正解：只能延拓到伪凸一侧。
- **误区 4**：以为「强伪凸 ⟹ CR 方程全局可解」。
  正解：局部可解 + 次椭圆正则性；全局可解还需更多条件（如紧性、上同调）。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 强伪凸超曲面 | strongly pseudoconvex hypersurface | Levi 正定 |
| 强凹超曲面 | strongly pseudoconcave | Levi 负定 |
| Levi 零方向 | Levi null direction | 正定性失效方向 |
| Lewy 延拓 | Lewy extension | 一侧延拓 |
| Kohn Laplacian | Kohn Laplacian | $\square_b$ |
| 次椭圆 | subelliptic | 半阶增益 |
| 伪凸一侧 | pseudoconvex side | 内部方向 |

## 6 历史注记与知识树

**历史**：Levi 形式由 E. E. Levi（1907）在研究延拓时引入；Lewy（1956）证明强伪凸超曲面上 CR 函数的局部延拓；Kohn（1963）建立 $\square_b$ 的次椭圆理论；Chern–Moser（1974）把 Levi 形式纳入等价的 CR 不变量体系。强伪凸超曲面至今仍是 CR 几何与复几何最常研究的对象。

**知识树**：

- 向后：CR 结构与复切空间（本组第 25 篇）、Levi 形式（第 2 组第 7 篇）。
- 向前：CR 函数延拓（本组第 27 篇）、Lewy 反例（本组第 28 篇）。
- 横向：黎曼几何的曲率符号分类——Levi 符号是 CR 版的「Ricci 符号」。

**一句话记忆**：强伪凸 = Levi 正定 = 次椭圆 = 延拓可行；Levi 符号是 CR 几何的分类标签，方向由符号决定。

## 7 小结

- **Levi 形式**（超曲面版本）：复切空间上的 Hermitian 形式，符号是几何不变量；只测复切方向的曲率。
- **强伪凸超曲面**：Levi 形式正定；球面是标准例子。
- **次椭圆正则性**：$\square_b$ 在强伪凸 CR 流形上次椭圆（$1/2$ 阶增益）——正则性定理的通行证。
- **Lewy 延拓定理**：强伪凸超曲面上的 CR 函数局部延拓为全纯函数。
- **次椭圆 = 正则性通行证**：强伪凸 ⟹ $\square_b$ 次椭圆（$1/2$ 阶增益）⟹ 有限维上同调与光滑解；一般伪凸 CR 流形上可能完全退化。
- **Lewy 延拓定理**：强伪凸超曲面上的 CR 函数局部延拓为全纯函数，方向由 Levi 符号决定（只能延到伪凸一侧）。
- **一句话记忆**：强伪凸 = Levi 正定 = 次椭圆 = 延拓可行；Levi 符号是 CR 几何的分类标签。

在下一节，我们进入 CR 函数的正面理论：**CR 函数的局部性质与可延拓性**——何时延拓、延拓多远、障碍是什么。