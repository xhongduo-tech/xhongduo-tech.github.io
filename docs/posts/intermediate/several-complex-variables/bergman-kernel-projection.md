---
title: Bergman 核与 Bergman 投影
date: 2026-08-07
---

# Bergman 核与 Bergman 投影

<div class="epigraph">
<p>Bergman 核是区域的全纯函数空间中最深藏的秘密——它把 $L^2$ 的几何与全纯性编织成一只积分核。</p>
<footer>—— 仿 斯特凡 · 伯格曼（Stefan Bergman），《正交函数系与核函数》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第9章；史济怀 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从 Bergman 核开始

$\bar\partial$ 理论给了我们解方程的存在性，但多复变还有一个**更古典、更几何**的对象：**Bergman 核**。它的想法非常直接：把区域 $D$ 上的全纯平方可积函数做成希尔伯特空间 $A^2(D)$（**Bergman 空间**），考虑「求值泛函」$f \mapsto f(z)$——它连续（由 Cauchy 估计保证），于是 Riesz 表示定理给出一个核函数 $K(z,w)$，使 $f(z) = \int_D f(w) K(z,w)\, dw$。这个 $K$ 就是 **Bergman 核**，它同时是 $A^2(D)$ 上正交投影（**Bergman 投影**）的积分核。<span class="marginnote">Bergman 核的价值：它把「区域的形状」编码进一个全纯函数的积分表示里。两个区域双全纯等价当且仅当它们的 Bergman 核通过雅可比因子关联——所以 Bergman 核是<strong>双全纯不变量</strong>的源泉，也是复几何里度量（Bergman 度量）的出发点。</span>

## 1 Bergman 空间与核的定义

设 $D \subset \mathbb{C}^n$ 有界区域（保证 $A^2$ 非零且求值泛函连续）。定义

$$
A^2(D) = \{ f \in \mathcal O(D) : \int_D |f|^2 \, d\lambda \lt  \infty \}
$$

配 $L^2$ 内积。$A^2(D)$ 是 $L^2(D)$ 的闭子空间（全纯函数的 $L^2$ 极限仍全纯——由内部正则性）。**求值泛函** $\delta_z: f \mapsto f(z)$ 由 Cauchy 估计有界：$|f(z)| \leq C_z \|f\|_{L^2}$。

由 Riesz 表示，存在唯一的 $K(\cdot, z) \in A^2(D)$ 使 $f(z) = \langle f, K(\cdot, z) \rangle$。定义 **Bergman 核**：

$$
K(z,w) = \overline{K(w,z)} = \text{求值泛函 $\delta_z$ 的 Riesz 表示核}
$$

**Bergman 投影** $P$：$L^2(D) \to A^2(D)$ 的正交投影，写为 $Pf(z) = \int_D f(w) K(z,w)\, d\lambda(w)$。<span class="marginnote">Bergman 核与单复变的 Szegő 核/单位圆盘情形：$D = \mathbb D \subset \mathbb C$ 时 $K(z,w) = \frac{1}{\pi}\frac{1}{(1-z\bar w)^2}$——「Bergman 核把全纯函数的再生性（$f(z)=\int fK$）变成显式公式」。多复变中 $K$ 一般无闭式，但满足同样的再生性。</span>

## 2 Bergman 核的基本性质

**性质 1（再生性 / 对角增长）**：$f(z) = \int_D f(w) K(z,w)\, dw$。对 $z = w$，$K(z,z) = \sum_\alpha |\varphi_\alpha(z)|^2 \geq 0$（$\{\varphi_\alpha\}$ 是 $A^2$ 的标准正交基），且 $K(z,z) \to +\infty$ 当 $z \to \partial D$。<span class="marginnote">对角增长 $\to \infty$ 是「区域边界不可穿越」的积分表示版本：越靠近边界，$A^2$ 里的函数越「集中在边界」，求值范数发散。这个增长行为在强伪凸域上被 Fefferman 定理精确刻画：$K(z,z) \sim \mathrm{dist}(z,\partial D)^{-(n+1)}$。</span>

**性质 2（全纯与共轭全纯）**：对固定 $w$，$z \mapsto K(z,w)$ 全纯；对固定 $z$，$w \mapsto K(z,w)$ 共轭全纯。$K$ 是 $D \times D$ 上「对角附近」的 $C^\infty$ 函数，在边界有奇性。

**性质 3（双全纯变换律）**：若 $\Phi: D \to \Omega$ 双全纯，则

$$
K_\Omega(\Phi(z), \Phi(w)) = K_D(z,w) \cdot \left[ \det \Phi'(z) \right]^{-1} \overline{\left[ \det \Phi'(w) \right]^{-1}}
$$

由此 `K_D(z,z) |\det \Phi'(z)|^2 = K_\Omega(\Phi(z),\Phi(z))`——对角值是密度型不变量。<span class="marginnote">这条变换律说明 Bergman 核随双全纯映射像「$(n,n)$-形式」一样变换。由它可构造 <strong>Bergman 度量</strong> $ds^2 = \sum \partial^2 \log K(z,z)/\partial z_j \partial \bar z_k\, dz_j d\bar z_k$，它是双全纯不变的 Kähler 度量——复几何研究的利器。</span>

## 3 Bergman 核与 $\bar\partial$ 理论的关系

Bergman 核看似独立于 $\bar\partial$ 理论，实则深刻相连：**Bergman 投影 $P$ 与 $\bar\partial$-Neumann 解算子 $N$ 有关**。对强伪凸域，有经典公式：

$$
P = I - \bar\partial^* N \bar\partial
$$

（其中 $N$ 是 $\bar\partial$-Neumann 算子，$P$ 是 $L^2$ 到 $A^2$ 的正交投影）。<span class="marginnote">这个公式说明：<strong>Bergman 投影是「$\bar\partial$-调和形式」的正交投影</strong>。$I - \bar\partial^*N\bar\partial$ 作用后消灭「$\bar\partial$ 可解部分」，留下全纯部分。所以 Bergman 核的边界奇性、$C^\infty$ 性完全由 $N$ 的正则性（上一节 Kohn 定理）决定——<strong>积分核理论与 $\bar\partial$ 理论在此合流</strong>。</span>

## 4 公式解析：Riesz 表示与再生性

$$
f(z) = \int_D f(w) \, K(z,w) \, d\lambda(w), \qquad f \in A^2(D)
$$

- **第一步，把求值泛函写成内积**：$\delta_z(f) = f(z)$ 是有界线性泛函。Riesz 表示：存在唯一的 $k_z \in A^2$ 使 $\delta_z(f) = \langle f, k_z\rangle = \int f \overline{k_z}$。令 $K(z,w) = \overline{k_z(w)}$，即得 $f(z) = \int f(w) K(z,w)$。
- **第二步，为什么 $\delta_z$ 连续**：由 Cauchy 估计，$|f(z)|$ 被 $f$ 在含 $z$ 的小多圆柱上的 $L^2$ 范数控制，进而被整个 $D$ 上的 $L^2$ 范数控制。**连续性 + 内积结构 = 核的存在**——这是希尔伯特空间方法的通用礼物。
- **第三步，基展开视角**：取 $A^2$ 的标准正交基 $\{\varphi_\alpha\}$，则 $K(z,w) = \sum_\alpha \varphi_\alpha(z)\overline{\varphi_\alpha(w)}$（级数在紧集上一致收敛）。对角值 $K(z,z) = \sum |\varphi_\alpha(z)|^2 \geq 0$——「所有基函数在 $z$ 处的能量和」。

## 5 辨析与延伸：Bergman 核的五个要点

**辨析 1：Bergman 核是「区域的指纹」**。两个区域双全纯等价当且仅当 Bergman 核通过雅可比因子关联——所以核的奇性、增长、零点模式编码了区域的**双全纯不变量**。**「区域几何 → 核分析」是一台精密的翻译机**。<span class="marginnote">特别是对角值 $K(z,z)$ 在边界的行为（如 Fefferman 定理：强伪凸域上 $K(z,z) \sim \mathrm{dist}^{-n-1}$）刻画了区域的「伪凸强度」。核是区域的「心电图」。</span>

**辨析 2：Bergman 空间非空需要区域有界（或足够好）**。无界区域上 $A^2(D)$ 可能只有零函数（如 $\mathbb{C}$ 上全纯平方可积函数只有 0）。所以 Bergman 核理论通常假设 $D$ 有界。**「求值泛函连续」需要 Cauchy 估计 + 有界性**。

**辨析 3：Bergman 核与 $\bar\partial$-Neumann 的联系**。$P = I - \bar\partial^*N\bar\partial$（Bergman 投影）把 $L^2$ 投到 $A^2$。这个公式说明：**Bergman 核的正则性完全由 $N$ 的正则性决定**——强伪凸域上 $N$ 保光滑，故 $K$ 在 $D\times D$ 内光滑（对角除外）。

**辨析 4：正交基展开的理解**。$K(z,w) = \sum_\alpha \varphi_\alpha(z)\overline{\varphi_\alpha(w)}$。这个级数在紧集上一致收敛，但**不能**逐项取极限到边界（边界上发散）。对角值 $K(z,z) = \sum |\varphi_\alpha(z)|^2 \to \infty$ 当 $z\to\partial D$——「基函数能量在边界累积」。

**误区清单**：

- **误区 1**：以为「Bergman 核总是显式可算」。
  正解：一般无闭式；有界对称域等特殊区域才有。
- **误区 2**：以为「$K(z,w)$ 处处光滑」。
  正解：$D\times D$ 上 $C^\infty$，但在边界附近有奇性。
- **误区 3**：以为「Bergman 空间对无界区域也非平凡」。
  正解：无界区域 $A^2$ 可能只有零函数。
- **误区 4**：以为「Bergman 度量与区域无关」。
  正解：Bergman 度量由核构造，随区域变化，是双全纯不变量。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| Bergman 空间 | Bergman space | $A^2(D)$ |
| 再生核 | reproducing kernel | 求值泛函的核 |
| Bergman 投影 | Bergman projection | $L^2\to A^2$ 正交投影 |
| 对角值 | diagonal value | $K(z,z)$ |
| Bergman 度量 | Bergman metric | 由核构造的 Kähler 度量 |
| 双全纯不变量 | biholomorphic invariant | 变换律 |

## 6 历史注记与知识树

**历史**：Bergman（1920s–30s）引入核函数与正交系方法；单位圆盘上 $K(z,w) = \frac1\pi\frac1{(1-z\bar w)^2}$ 是经典例子。Fefferman（1974）精确刻画强伪凸域上对角奇性，标志核理论进入现代阶段。Bergman 度量至今仍是复几何与几何函数论的活跃工具。

**知识树**：

- 向后：$\bar\partial$-Neumann 与正则性（本组第 21 篇）、强伪凸域（第 2 组）。
- 向前：Szegő 核与 Hardy 空间（本组第 24 篇）——内部 vs 边界再生核。
- 横向：泛函分析的再生核理论（RKHS，第三级《机器学习》）——同一数学结构的应用。

**一句话记忆**：Bergman 核 = $A^2(D)$ 的再生核 = 区域的指纹；$P = I - \bar\partial^*N\bar\partial$ 连接核与 $\bar\partial$ 理论。

## 7 小结

- **Bergman 空间** $A^2(D)$：全纯平方可积函数；求值泛函连续 ⟹ 核存在。
- **Bergman 核** $K(z,w)$：再生核，$f(z) = \int fK$；对角增长到边界。
- **Bergman 投影** $P = I - \bar\partial^* N \bar\partial$：$L^2$ 到 $A^2$ 的正交投影，与 $\bar\partial$