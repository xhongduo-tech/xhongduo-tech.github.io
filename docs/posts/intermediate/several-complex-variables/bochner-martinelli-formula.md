---
title: 积分表示公式：Bochner-Martinelli 与 Cauchy 型
date: 2026-08-07
---

# 积分表示公式：Bochner-Martinelli 与 Cauchy 型

<div class="epigraph">
<p>在单复变里，Cauchy 核是唯一的；在多复变里，每个区域都有自己的 Cauchy 核——这是丰饶，也是责任。</p>
<footer>—— 仿 萨洛蒙 · 博赫纳（Salomon Bochner），《多复变函数的积分表示》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Krantz 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从积分表示公式开始

单复变的全部理论都站在一个公式上：$f(z) = \frac{1}{2\pi i}\oint \frac{f(\zeta)}{\zeta-z} d\zeta$。进入 $\mathbb{C}^n$，这个公式遭遇一个深刻的困难：**没有唯一的 Cauchy 核**。第一课的多圆柱迭代公式依赖「多圆柱」这个特殊形状；对一般区域，需要寻找适合区域边界的积分表示。**Bochner–Martinelli 公式**提供了一个**普适**（不依赖区域形状）的替代：它的核只依赖欧氏度量，适用于任何有界区域的光滑边界。本篇先讲 Bochner–Martinelli，再看 **Cauchy 型积分**（把核沿任意边界积分）及其边界值理论。<span class="marginnote">为什么值得学 Bochner–Martinelli？因为它是多复变的「通用积分表示」：不需要区域凸或伪凸，只要边界光滑。它的核是全空间 $\mathbb{C}^n$ 内生的，虽然「太大」（对全纯函数它给出的是光滑解而非全纯），但它能给出<strong>光滑性的先验控制</strong>——在建立估计时不可替代。</span>

## 1 Bochner–Martinelli 核与公式

定义 **Bochner–Martinelli 核**：

$$
\omega_{BM}(z, \zeta) = \frac{(n-1)!}{(2\pi i)^n} \frac{1}{|\zeta - z|^{2n}} \sum_{j=1}^n (\bar\zeta_j - \bar z_j) \, d\bar\zeta_1 \wedge \cdots \wedge \widehat{d\bar\zeta_j} \wedge \cdots \wedge d\bar\zeta_n \wedge d\zeta_1 \wedge \cdots \wedge d\zeta_n
$$

其中 $\widehat{\quad}$ 表示去掉该项。**Bochner–Martinelli 公式**：设 $D \Subset \mathbb{C}^n$ 有 $C^1$ 边界，$f$ 在 $\overline D$ 邻域上全纯，则对 $z \in D$：

$$
f(z) = \int_{\partial D} f(\zeta)\, \omega_{BM}(z,\zeta)
$$

即：**全纯函数由边界上的值通过 $BM$ 核再生**。<span class="marginnote">$n=1$ 时 $\omega_{BM}(z,\zeta) = \frac{1}{2\pi i} \frac{d\zeta}{\zeta - z}$——精确回到经典 Cauchy 核。所以 Bochner–Martinelli 是 Cauchy 公式的<strong>忠实推广</strong>（对任意 $n$），但它不是「全纯核」（作用在全纯函数上才再生），这既是它的局限，也是它的普适性来源。</span>

## 2 为什么 BM 核是普适的

对比三条积分表示路线：

| 表示 | 核 | 区域要求 | 全纯性 | 用途 |
| --- | --- | --- | --- | --- |
| 多圆柱迭代 Cauchy | $1/((\zeta_1-z_1)\cdots(\zeta_n-z_n))$ | 多圆柱 | 全纯核 | 局部理论、幂级数 |
| Bochner–Martinelli | $BM$ 核（欧氏度量） | 任何光滑边界 | 非全纯核 | 普适表示、先验估计 |
| 积分核（Hua / 强伪凸） | 区域依赖核 | 强伪凸等 | 全纯核 | Bergman 型理论 |

**关键观察**：多圆柱核把 $\zeta_j - z_j$ **逐坐标**分开（交叉项不出现），所以只适合「盒状」区域；BM 核用 $|\zeta - z|^{2n}$ 整体度量距离，对任何区域都合法，但代价是**不再全纯**（含 $\bar\zeta$ 因子）。<span class="marginnote">这个权衡是理解多复变积分表示的一把钥匙：<strong>「通用」与「全纯」不可兼得</strong>。想要全纯核（如强伪凸域上的 Cauchy–Fantappiè 核），必须付出区域几何的限制；想要普适，就得接受非全纯核。Bochner–Martinelli 站在「普适」这一极。</span>

## 3 Cauchy 型积分与边界值

**Cauchy 型积分**：对边界 $\partial D$ 上的（任意可积）函数 $g$，定义

$$
\mathcal C g(z) = \int_{\partial D} g(\zeta)\, \omega_{BM}(z,\zeta), \qquad z \notin \partial D
$$

当 $g = f\big|_{\partial D}$（$f$ 全纯）时，$\mathcal C g$ 在 $D$ 内等于 $f$，在 $D$ 外等于 $0$（对 $n=1$）；对 $n \geq 2$，**$BM$ 核的奇点集是 $z=\zeta$ 附近的一个 $2n-2$ 维流形**，行为与 Cauchy 核不同，边界值要小心处理。<span class="marginnote">对 $n \geq 2$，$\omega_{BM}$ 的奇性比 Cauchy 核弱：$|\zeta-z|^{-(2n-1)}$ 沿奇点流形的积分不是主值可积的普通形式。Kytmanov 等人发展了 $BM$ 型积分的边界值理论（主值 + 半值公式），它与 CR 函数（第 5 篇组）的边界值有密切联系。</span>

**应用**：Cauchy 型积分把边界值问题化为积分方程问题——研究边界函数 $g$ 何时是全纯函数的边界值（即 **CR 函数的边值表示**），这直接通向第 5 篇组的 CR 理论。

## 4 公式解析：BM 核的降维结构

只看核心项（忽略组合系数）：

$$
\omega_{BM}(z,\zeta) \sim \frac{1}{|\zeta - z|^{2n}} \sum_j (\bar\zeta_j - \bar z_j)\, d\bar\zeta^{\hat j} \wedge d\zeta
$$

- **第一步，距离因子 $|\zeta-z|^{-2n}$**：这是 $\mathbb{C}^n$ 上（实维 $2n$）的**基本解**的导数规模：$\Delta_{\mathbb R^{2n}}$ 的基本解 $\sim |x|^{2-2n}$，其一阶导数 $\sim |x|^{1-2n}$ 的模方……更直接：$\omega_{BM}$ 是「$\partial_{z_j}$ 作用于基本解」的整合。$2n$ 的幂次保证积分在边界 $\partial D$（实维 $2n-1$）上收敛。
- **第二步，为什么会有 $\bar\zeta$ 因子**：$\omega_{BM}$ 含 $\bar\zeta_j - \bar z_j$，所以不是 $\zeta$ 的全纯形式——这导致它作用在全纯 $f$ 上才给出 $f$，作用在一般光滑函数上给出「$\bar\partial$ 方程解的积分表示」（Koppelman 公式的退化）。**核的形式泄露了它的代数性质**。
- **第三步，$n=1$ 自检**：$n=1$ 时 $|\zeta-z|^{-2} = (\zeta-z)^{-1}(\bar\zeta-\bar z)^{-1}$，$\sum_j$ 只剩一项，$\omega_{BM} = \frac{1}{2\pi i}\frac{d\zeta}{\zeta-z}$——Cauchy 核重现，公式复原。

## 5 辨析与延伸：积分表示的五个要点

**辨析 1：BM 核的「非全纯」不是缺陷，是权衡**。BM 核含 $\bar\zeta$ 因子，作用在全纯函数上才再生 $f$；作用在一般光滑函数上给出 $\bar\partial$ 方程解的表示。**「普适」与「全纯」的权衡是多复变积分表示的核心张力**：要普适（任何区域），就得放弃全纯核。<span class="marginnote">这解释了为什么强伪凸域理论会专门构造「Cauchy–Fantappiè 型」全纯核——用区域几何换回全纯性。BM 核是「无区域假设」时的默认选择。</span>

**辨析 2：$n=1$ 自检的重要性**。BM 核在 $n=1$ 精确退回 Cauchy 核。这既是记忆锚点，也是检验公式正确性的利器。**多复变一切积分表示都应通过 $n=1$ 自检**——若退化不对，公式必有误。

**辨析 3：BM 型积分的边界值**。对 $n\geq2$，BM 核的奇点集是 $2n-2$ 维流形（比 $n=1$ 的点状奇点复杂），Cauchy 型积分 $\mathcal C g$ 在边界上的主值要小心处理。Kytmanov 等发展了完整的 BM 型积分边值理论。**边界值理论是多复变积分表示的深水区**。

**辨析 4：与 Bergman/Szegő 核的分工**。BM 核用于「普适表示 + 先验估计」；Bergman 核用于「$L^2$ 全纯投影」；Szegő 核用于「边界 $L^2$ 投影」。三者互补，各有侧重。**选核取决于问题**：要估计选 BM，要投影选 Bergman，要边界选 Szegő。

**误区清单**：

- **误区 1**：以为「BM 核是全纯核」。
  正解：BM 核含 $\bar\zeta$，非全纯。
- **误区 2**：以为「BM 公式只在凸域成立」。
  正解：对任何 $C^1$ 边界成立，这是它普适的原因。
- **误区 3**：以为「Cauchy 型积分边界值平凡」。
  正解：$n\geq2$ 时需主值理论，非平凡。
- **误区 4**：以为「BM 核在 $n=1$ 与 Cauchy 核不同」。
  正解：$n=1$ 时完全重合。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| Bochner–Martinelli 核 | Bochner–Martinelli kernel | 普适核 |
| Cauchy 型积分 | Cauchy-type integral | 边界函数积分 |
| 主值 | principal value | 奇点正则化 |
| 全纯核 | holomorphic kernel | 区域依赖核 |
| 先验估计 | a priori estimate | 表示给出控制 |
| 再生表示 | reproducing formula | 边界再生 |

## 6 历史注记与知识树

**历史**：Bochner 与 Martinelli 在 1940s 独立发现该公式；此后 Koppelman 给出含 $\bar\partial$ 项的完整同伦公式（Koppelman 公式），统一了 Cauchy 型与 BM 型表示。Hua（华罗庚）与 Lu 等系统研究有界对称域的显式全纯核，把积分表示推向具体计算。

**知识树**：

- 向后：Cauchy 积分公式（第 1 组）、Bergman 核（本组第 22 篇）。
- 向前：Szegő 核与边界值（本组第 24 篇）、CR 理论（第 5 组）。
- 横向：调和分析的奇异积分（第三级《调和分析》）——主值与边界值理论。

**一句话记忆**：BM 核 = 普适的 Cauchy 核替代品；「通用」与「全纯」不可兼得；$n=1$ 自检永远是第一关。

## 7 小结

- **Bochner–Martinelli 公式**：$f(z) = \int_{\partial D} f\, \omega_{BM}$，对任何 $C^1$ 边界成立，$n=1$ 时是 Cauchy 公式。
- **普适性代价**：BM 核非全纯——「通用」与「全纯」不可兼得。
- **Cauchy 型积分**：边界函数经 BM 核的积分；边界值理论通向 CR 函数。
- **对比表**：多圆柱核（局部）、BM 核（普适）、区域核（全纯但受限）三分天下。

在下一节，我们研究另一族边界值理论：**Szegő 核、Hardy 空间与边界值**——$L^2$