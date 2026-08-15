---
title: Marstrand 射影定理
date: 2026-08-07
---

# Marstrand 射影定理

<div class="epigraph">
<p>一个集合的维数，在几乎所有方向上看过去，都不会变小——只要它足够「胖」。</p>
<footer>—— 自 J. M. Marstrand（约翰 · 马斯特兰德），1954 年射影定理</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何测度论 ｜ P. Mattila, *Geometry of Sets and Measures in Euclidean Spaces\*, §9 ｜ 2026-08-07</p>
</div>

## 为什么从 Marstrand 射影定理开始

前几节反复用到「把集合投影到某个方向上看」的技巧。投影会改变集合的维数：一条曲线在 $\mathbb{R}^2$ 中投影到它垂直的方向会塌成点（维数变 0），投影到平行方向保持维数 1。对光滑对象，投影的维数损失取决于「是否与切方向相切」；但对分形、对一般 Borel 集，**几乎所有方向上的投影行为是一致的**——这正是 **Marstrand 射影定理** 的内容：若集合的维数 $\le 1$，则几乎所有投影保持维数不变；若维数 $> 1$，则几乎所有投影都「填满」目标空间（有正 Lebesgue 测度）。这条定理是射影维数理论的基石，也是连接几何测度论与傅里叶分析（通过 Frostman 测度的傅里叶变换）的桥梁。<span class="marginnote">「几乎所有方向」指的是参数空间里除去一个零测度集合以外的所有方向。这个「几乎处处方向无关性」是几何测度论反复出现的主题：测度性质往往在几乎每个方向、几乎每个点上都是规则的。</span>

## 1 射影的维数损失：先看光滑情形

设 $A \subset \mathbb{R}^2$，$\pi_\theta$ 是把 $\mathbb{R}^2$ 正交投影到与 $x$ 轴夹角为 $\theta$ 的直线上的映射。问：$\dim_{\mathrm{H}} \pi_\theta(A)$ 与 $\dim_{\mathrm{H}} A$ 的关系如何？

对光滑 $m$ 维子流形 $M \subset \mathbb{R}^n$，投影的微分是满秩线性映射（除非投影方向落在切空间的某个退化方向上）。因此「几乎处处方向」上，投影保持维数：$\dim_{\mathrm{H}} \pi_\theta(M) = \dim_{\mathrm{H}} M = m$。退化只发生在「与切空间相切」的特殊方向——这些方向构成一个零测度集合。

但对分形，问题变得微妙：康托尔集在 $\mathbb{R}^2$ 里绕成「康托尔尘」，任意方向的投影几乎总是区间（维数 1）还是尘埃（维数 0.63）？直觉上，旋转一下坐标系，尘埃的结构会被打散，投影应该「变大」。Marstrand 定理把这个直觉精确化。<span class="marginnote">一个经典例子：乘积康托尔集 $C \times C$（$C$ 是 $\{0,1\}$ 进制三分康托尔集）维数 $2\log2/\log3 \approx 1.26$，但投影到 45° 方向会塌成 $C$ 的某种变形（可能维数降到 0.63）。Marstrand 定理保证「几乎所有」方向保持维数，但<strong>存在</strong>个别坏方向——坏方向集的维数结构本身是活跃研究课题。</span>

## 2 Marstrand 射影定理：两个临界区间

**核心概念（Marstrand 射影定理，$\mathbb{R}^2$ 情形）**：设 $A \subset \mathbb{R}^2$ 是 Borel 集，$\pi_\theta$ 为与 $x$ 轴夹角 $\theta$ 的正交投影。

- 若 $\dim_{\mathrm{H}} A \le 1$，则对几乎所有 $\theta$，
  $$
  \dim_{\mathrm{H}} \pi_\theta(A) \;=\; \dim_{\mathrm{H}} A
  $$
- 若 $\dim_{\mathrm{H}} A > 1$，则对几乎所有 $\theta$，
  $$
  \mathcal{L}^1\bigl(\pi_\theta(A)\bigr) > 0
  $$

第一条说「维数 $\le 1$ 的集合几乎处处投影保维」；第二条说「维数 $> 1$ 的集合几乎处处投影有正长度」。第二条的结论比「保维」更强——它断言投影**填满目标空间的一维区间**（正 Lebesgue 测度），而不仅是维数等于 1。<span class="marginnote">临界值 1 对应目标空间 $\mathbb{R}^1$ 的维数。对 $\mathbb{R}^n \to \mathbb{R}^k$ 的投影，临界值是 $k$：$\dim A \le k$ 时几乎处处保维，$\dim A > k$ 时几乎处处投影有正 $k$ 维测度。</span>

**重点：Marstrand 定理把「集合的内在维数」与「投影的可测大小」桥接起来**——维数不超过目标维数的集合，在几乎每个方向上都不被「压扁」。

## 3 公式解析：投影维数与 Frostman 能量

证明 Marstrand 定理（$\mathbb{R}^2 \to \mathbb{R}^1$ 情形）的标准路线借道 Frostman 能量（第 7 篇）。设 $A$ 上存在概率测度 $\mu$ 满足 Frostman $t$ 条件 $\mu(B(x,r)) \le C r^t$。

考虑投影测度 $\mu_\theta = (\pi_\theta)_\# \mu$，它是 $\mathbb{R}$ 上的测度。其 $s$-能量为

$$
I_s(\mu_\theta) \;=\; \int_{\mathbb{R}}\int_{\mathbb{R}} \frac{\mathrm{d}\mu_\theta(u)\, \mathrm{d}\mu_\theta(v)}{|u - v|^s}
$$

把 $\mu_\theta$ 写成 $\mu$ 的推前，再用「$\pi_\theta(x) = \pi_\theta(y) \iff$ 弦 $xy$ 平行于方向 $\theta$」的几何，得到关键的**傅里叶形式**：

$$
I_s(\mu_\theta) \;=\; \int_{\mathbb{R}} |\widehat{\mu_\theta}(\xi)|^2 |\xi|^{s-1} \mathrm{d}\xi
$$

逐项拆解：

- **$\widehat{\mu_\theta}(\xi)$（投影测度的傅里叶变换）**：把「投影方向的几何信息」编码成傅里叶域的衰减。
- **$|\xi|^{s-1}$（幂权）**：$\xi$ 越大权重越大，能量小 ⟺ 高频衰减快 ⟺ 测度「光滑」。对投影测度，高频行为由 $\mu$ 的径向傅里叶衰减决定。
- **$\theta$ 取平均**：对 $\theta$ 在 $[0,\pi)$ 上积分平均，可以用球面坐标把 $\iint |\widehat{\mu_\theta}|^2 |\xi|^{s-1}$ 化简成 $\iint |x-y|^{-(s+1)}$ 形式的项，从而用 $\mu$ 的 $t$-Frostman 条件控制。
- **结论推导**：若 $t \lt  1$ 且 $s \lt  t$，则对几乎所有 $\theta$，$I_s(\mu_\theta) \lt  \infty$，由 Frostman 能量判据得 $\dim_{\mathrm{H}} \pi_\theta(A) \ge s$，再取 $s \to \dim_{\mathrm{H}} A$ 得第一条。<span class="marginnote">傅里叶形式的妙处：它把「对 $\theta$ 几乎处处」变成「对 $\theta$ 积分平均」——只要能证明平均能量有限，就能推出几乎处处能量有限（有限例外集是零测度）。这是「逐点几乎处处」与「积分有限」互相转化的标准戏法。</span>

## 4 一般维数的推广与 Kaufman 定理

Marstrand 定理对 $m$ 维投影（$\mathbb{R}^n \to \mathbb{R}^k$）有直接推广：若 $\dim A \le k$，则对几乎所有 $k$ 维投影 $\pi$，$\dim_{\mathrm{H}} \pi(A) = \dim_{\mathrm{H}} A$；若 $\dim A > k$，则 $\mathcal{L}^k(\pi(A)) > 0$ 对几乎所有 $\pi$ 成立。这个版本同样可用 Frostman 能量 + 傅里叶方法证明。

当维数严格小于 $k$ 时，Marstrand 定理只给出「保维」而没说投影集合「有多大」。**Kaufman 定理**（1968）给出更精细的下界：对 $A \subset \mathbb{R}^n$，几乎所有 $k$-维投影都满足

$$
\dim_{\mathrm{H}} \pi(A) \;\ge\; \frac{\dim_{\mathrm{H}} A}{k}
$$

（在 $\dim_{\mathrm{H}} A \le k$ 情形）。这个下界弱于保维，但在「投影几乎处处变胖」的方向上更进一步。对「打满」情形的临界行为，Mattila 和后续研究者给出了精确的临界维数条件。<span class="marginnote">Kaufman 下界与 Marstrand 保维结论的差距，反映了「维数下界」与「维数相等」的差别：保维需要更精细的 Frostman 论证，Kaufman 下界只需要粗一点的能量估计。这个 gap 在更高维仍未完全闭合，是活跃的开放课题。</span>

**辨析｜易错点：** Marstrand 定理是「几乎所有方向」而非「所有方向」。坏方向确实存在（如乘积康托尔集投影到 45° 方向可能降维），但坏方向构成零测度集合。初学者易把定理误记为「所有投影都保维」——正确说法是「几乎处处保维」，且「正测度」情形（$\dim A > k$）断言的是投影有正 $k$ 测度而非维数恰好等于 $k$。

## 5 Marstrand 定理的应用：Kakeya 问题与 Besicovitch 集

Marstrand 射影定理最深刻的用武之地之一，是 **Kakeya（贝西科维奇）问题**。

**核心概念（Besicovitch 集）**：$\mathbb{R}^n$ 中一个**Besicovitch 集**是包含每个方向的一条单位线段、但 Lebesgue 测度为 0 的集合。$n = 2$ 时这样的集合存在（Besicovitch 构造），但它的 Hausdorff 维数是多少？Kakeya 猜想断言 $\dim_{\mathrm{H}} = 2$。<span class="marginnote">Kakeya 猜想：$\mathbb{R}^n$ 里任何 Besicovitch 集的 Hausdorff 维数是 $n$。$n = 2$ 已证（Davies），$n \ge 3$ 是著名的开放问题，与调和分析（Bochner–Riesz 猜想、限制猜想）深度纠缠。Marstrand 型的投影/方向估计是这条线上最基本的地基。</span>

Marstrand 定理在其中扮演的角色：若一个集合的投影「几乎所有方向都保持维数」，那么它不可能「在每个方向都坍缩」。用反证：若 Besicovitch 集维数 $\lt  2$，取它的 1 维投影，Marstrand 说几乎处处投影保维——但 Besicovitch 集在每个方向都含一条线段，投影至少包含一段，维数至少 1。这给出相容但不够紧的约束；真正困难的是把「维数恰好」推向「维数 2」。这套「用射影定理约束方向结构」的思路，是调和分析通往 Kakeya 问题的桥梁。

**比较表**：射影维数理论的三个层次。

| 定理 | 条件 | 结论 | 工具 |
| --- | --- | --- | --- |
| 光滑情形的投影 | 子流形 + 非退化方向 | 保维 | 秩定理 |
| Marstrand 定理 | $\dim A \le k$ / $> k$ | 保维 / 正 $k$ 测度 | Frostman + 傅里叶 |
| Kaufman 定理 | $\dim A \le k$ | 维数下界 $\ge (\dim A)/k$ | 能量估计 |

## 6 随机版本的射影定理与例外集分析

Marstrand 定理对「几乎所有方向」成立，但「例外方向」本身的结构同样迷人。对给定集合 $A$，记

$$
E(A) \;=\; \left\{ \theta : \dim_{\mathrm{H}} \pi_\theta(A) \lt  \dim_{\mathrm{H}} A \right\}
$$

为例外方向集。Marstrand 定理说 $\mathcal{L}^1(E(A)) = 0$，但 $E(A)$ 可以有多大？已知结果：$E(A)$ 的 Hausdorff 维数可以是任意接近 1 的值，且其维数由 $A$ 的「相切结构」决定。对自相似集，Furstenberg 猜想（部分证明中）刻画了 $\pi_\theta(A)$ 的维数如何随 $\theta$ 变化。

随机分形则提供了「几乎所有」的另一种解释。设 $A_\omega$ 是随机构造的分形集（例如随机康托尔集：每层以概率 $p$ 保留子块），则

$$
\dim_{\mathrm{H}} \pi_\theta(A_\omega) \;=\; \min\{ \dim_{\mathrm{H}} A_\omega, 1 \}, \qquad \text{对所有 } \theta \text{ 几乎必然成立}
$$

对随机分形，例外方向**几乎必然消失**——所有方向的投影都保维。这个「随机正则化」现象说明：方向上的退化需要精细的确定性结构，随机性把它冲掉。<span class="marginnote">例外方向集的维数研究是「射影维数」方向的活跃课题：从 Besicovitch 时代的「例外集测度为零」，到如今「例外集的维数上下界」，每隔十年都有实质推进。</span>

**比较表**：确定性与随机分形的投影行为。

| 分形类型 | 例外方向 | 投影维数 |
| --- | --- | --- |
| 确定性自相似集 | 零测度，但可正维数 | 几乎处处保维，坏方向降维 |
| 随机康托尔集 | 几乎必然无例外 | 几乎必然所有方向保维 |

## 7 小结

- **Marstrand 射影定理**（$\mathbb{R}^n \to \mathbb{R}^k$）：$\dim A \le k$ 时几乎所有投影保维；$\dim A > k$ 时几乎所有投影有正 $k$ 测度。
- 证明借道 **Frostman 能量 + 傅里叶变换**：把「对几乎所有 $\theta$」换成「对 $\theta$ 平均」再取例外集零测度。
- 临界值 $k$ 等于目标空间的维数；坏方向存在但构成零测度集合。
- **Kaufman 定理**给出投影维数的更粗下界 $\dim \pi(A) \ge (\dim A)/k$