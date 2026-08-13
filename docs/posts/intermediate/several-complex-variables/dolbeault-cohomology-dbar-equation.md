---
title: Dolbeault 上同调与 ∂̄ 方程
date: 2026-08-07
---

# Dolbeault 上同调与 ∂̄ 方程

<div class="epigraph">
<p>$\bar\partial$ 方程是多复变的牛顿方程：解出它，全纯函数就纷纷现身。</p>
<footer>—— 仿 让 · 皮埃尔·多波（Jean-Pierre Dolbeault），《Dolbeault 上同调理论》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章；史济怀 第5章 ｜ 2026-08-07</p>
</div>

## 为什么从 Dolbeault 上同调开始

前面几节反复借用一个「黑箱」：全纯凸域上 $H^q(D,\mathcal O)=0$，理由是「$\bar\partial$ 方程可解」。现在是时候把黑箱打开。**Dolbeault 上同调** 是「用光滑微分形式计算 $\mathcal O$-上同调」的机器：它把「$\bar\partial f = g$ 何时有解」这个**偏微分方程**问题，转化为「**Dolbeault 上同调何时为零**」这个**代数**问题。Dolbeault 定理给出两者之间的同构，而 $\bar\partial$ 方程的可解性则是 Hörmander L² 理论（下一组）的主战场。<span class="marginnote">为什么这个「翻译」如此重要？因为 $\bar\partial$ 方程是<strong>线性、常系数、可显式积分</strong>的方程（用 Cauchy 核 / Bochner–Martinelli 核积分表示解），比抽象的层论更好下手。把「存在全纯函数」的一切问题都化成「解 $\bar\partial$ 方程」，是多复变的制胜一击。</span>

## 1 微分形式的双分级：$(p,q)$-形式

在 $\mathbb{C}^n$ 中，光滑微分形式按「$z$ 与 $\bar z$」分成双分级。记 $dz^I = dz_{i_1}\wedge\cdots\wedge dz_{i_p}$，$d\bar z^J = d\bar z_{j_1}\wedge\cdots\wedge d\bar z_{j_q}$。**$(p,q)$-形式**是形如

$$
\omega = \sum_{|I|=p,\,|J|=q} f_{IJ}\, dz^I \wedge d\bar z^J
$$

的光滑形式。外微分分解为 $d = \partial + \bar\partial$，其中

$$
\partial: \Omega^{p,q} \to \Omega^{p+1,q}, \qquad \bar\partial: \Omega^{p,q} \to \Omega^{p,q+1}
$$

由 $d = \partial + \bar\partial$ 与 $d^2=0$ 推出 $\partial^2 = \bar\partial^2 = 0$ 且 $\partial\bar\partial + \bar\partial\partial = 0$。<span class="marginnote">$\bar\partial$ 的显式作用：对 $(0,0)$-函数 $f$，$\bar\partial f = \sum_j \frac{\partial f}{\partial \bar z_j} d\bar z_j$。<strong>$\bar\partial f = 0$ 正是 Cauchy-Riemann 方程组</strong>——所以「$\bar\partial$-闭」的 $(0,0)$-形式就是全纯函数。这是全纯性在现代语言下的精确定义。</span>

## 2 Dolbeault 上同调

对开集 $D$，定义

$$
H^{p,q}_{\bar\partial}(D) = \frac{\ker(\bar\partial: \Omega^{p,q}(D) \to \Omega^{p,q+1}(D))}{\mathrm{im}(\bar\partial: \Omega^{p,q-1}(D) \to \Omega^{p,q}(D))}
$$

它度量「$\bar\partial$-闭的 $(p,q)$-形式有多大程度不是 $\bar\partial$-精确的」。特别地：

- $H^{0,0}_{\bar\partial}(D) = $ 全纯函数（$\bar\partial$-闭的 $(0,0)$-形式）；
- $H^{0,q}_{\bar\partial}(D) = 0$ ⟺ 每个 $\bar\partial$-闭的 $(0,q)$-形式都是某个 $(0,q-1)$-形式的 $\bar\partial$ 像——即 **$\bar\partial u = g$ 对满足 $\bar\partial g = 0$ 的 $g$ 恒可解**（可解性条件）。

**Dolbeault 引理（局部可解）**：在**多圆柱**（更一般地，任意可缩开集）上，$H^{p,q}_{\bar\partial} = 0$ 对 $q \geq 1$。即 $\bar\partial$ 方程局部恒可解。<span class="marginnote">这是单复变 Cauchy 积分公式的多维推广：解 $u$ 用 Bochner–Martinelli / Cauchy 核的积分表示显式写出。局部可解 + 伪凸域的整体方法（L² 估计），拼出全局可解。Dolbeault 引理是「局部正确」的保证，整体正确则靠下一组的 L² 理论。</span>

## 3 Dolbeault 定理：上同调的桥梁

**Dolbeault 定理**：设 $D$ 是 $\mathbb{C}^n$ 中的开集，则对一切 $p, q \geq 0$：

$$
H^q(D, \Omega^p) \;\cong\; H^{p,q}_{\bar\partial}(D)
$$

其中 $\Omega^p$ 是全纯 $p$-形式层。特别地，对 $p=0$：

$$
H^q(D, \mathcal O) \;\cong\; H^{0,q}_{\bar\partial}(D)
$$

**这条同构是本专题的枢纽**：层上同调（代数）与 $\bar\partial$ 方程（分析）被证明是同一个东西。<span class="marginnote">证明思路：$\Omega^{p,q}$（光滑 $(p,q)$-形式层）是 $\mathcal O$ 的「a 级消解」——用 Poincaré 引理（光滑情形 Dolbeault 引理）与层的正合性，把 $\mathcal O \to \Omega^{p,\bullet}$ 做成软层的消解，再由「软层上同调为零」推出同构。这是「用消解算上同调」的标准套路。</span>

**推论**：$D$ 是全纯凸域 ⟺ 对一切 $q \geq 1$，$H^{0,q}_{\bar\partial}(D) = 0$。也就是说，**在伪凸域上，$\bar\partial u = g$（$\bar\partial g = 0$）恒有解**——这就是上一组反复引用的「$\bar\partial$ 可解性」。

## 4 公式解析：$\bar\partial$ 方程的可解性条件

$$
\bar\partial u = g, \qquad g \in \Omega^{0,q}(D),\;\bar\partial g = 0
$$

- **第一步，可解的必要条件 $\bar\partial g = 0$**：若 $\bar\partial u = g$ 且 $u$ 光滑，则 $\bar\partial g = \bar\partial^2 u = 0$。所以「$g$ 是 $\bar\partial$-闭的」是任何解存在的**必要条件**——这不是人为限制，而是恒等式。
- **第二步，充分性靠区域**：局部（Dolbeault 引理）$g$ 闭即可解；整体上，闭性是否足够由 $H^{0,q}_{\bar\partial}(D)$ 决定。全纯凸域上 $H^{0,q} = 0$（$q\geq1$），所以「闭 ⟹ 可解」。
- **第三步，对比实情形的 Poincaré 引理**：实分析中 $dg = 0$（闭）⟹ 局部可解（Poincaré），但整体要 $H^q_{\text{de Rham}} = 0$。多复变完全平行：$\bar\partial g = 0$ ⟹ 局部可解，整体要 $H^{0,q}_{\bar\partial} = 0$。**Dolbeault 上同调就是复结构的 de Rham 上同调**。

## 5 辨析与延伸：Dolbeault 上同调的五个要点

**辨析 1：$\bar\partial$ 方程是「复结构的 Laplace 方程」**。实分析中 $\Delta u = f$ 是核心；多复变中 $\bar\partial u = g$（配 $\bar\partial^*$）是核心。$\square = \bar\partial\bar\partial^* + \bar\partial^*\bar\partial$ 是复结构的 Laplacian（Kähler 流形上等于 $\frac12\Delta$）。**理解 $\bar\partial$，就等于理解复几何的全部 PDE**。<span class="marginnote">在 Kähler 流形上 $\square = \frac12\Delta$（Hodge 恒等式），Dolbeault 上同调与 de Rham 上同调通过 Hodge 分解联系起来——这是 Hodge 理论的核心，也是 Dolbeault 上同调在几何中如此重要的原因。</span>

**辨析 2：$(p,q)$ 分级是复结构特有的**。实微分形式按总次数 $k$ 分级；复形式按 $(p,q)$ 双分级，因为 $d = \partial + \bar\partial$ 分别提升 $p$ 和 $q$。**双分级的存在是「复结构」的签名**——没有复结构就没有 $(p,q)$ 分级。

**辨析 3：$\bar\partial f = 0$ 就是 Cauchy–Riemann 方程**。$(0,0)$-形式 $f$ 的 $\bar\partial f = \sum_j \frac{\partial f}{\partial\bar z_j} d\bar z_j$。它为零 ⟺ 每个 $\partial f/\partial\bar z_j = 0$ ⟺ Cauchy–Riemann 方程组成立 ⟺ $f$ 全纯。**「全纯 = $\bar\partial$-闭的 $(0,0)$-形式」是现代多复变的第一句话**。

**辨析 4：Dolbeault 引理 vs Poincaré 引理**。Poincaré 引理（实）：闭形式局部精确。Dolbeault 引理（复）：$\bar\partial$-闭局部 $\bar\partial$-精确。两者结构完全平行，区别只在算子（$d$ vs $\bar\partial$）。**多复变是「复化的 Poincaré 引理」**。

**误区清单**：

- **误区 1**：以为「$d = \partial + \bar\partial$ 是 trivial」。
  正解：这是复结构下的 Hodge 分解第一层，蕴含 $p,q$ 双分级。
- **误区 2**：以为「$\bar\partial f = 0$ 与全纯无关」。
  正解：$\bar\partial f = 0$ 正是 Cauchy–Riemann 方程组。
- **误区 3**：以为「$H^{p,q}_{\bar\partial}$ 与 $H^{p+q}_{\text{de Rham}}$ 相同」。
  正解：不同；两者通过 Hodge 理论在 Kähler 流形上部分联系。
- **误区 4**：以为「Dolbeault 定理是平凡的」。
  正解：它用层消解证明，是多复变与代数方法合流的标志。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| $(p,q)$-形式 | $(p,q)$-form | 复形式双分级 |
| 全纯 $p$-形式 | holomorphic $p$-form | $\Omega^p$ 层 |
| Dolbeault 上同调 | Dolbeault cohomology | $\bar\partial$ 障碍 |
| Dolbeault 引理 | Dolbeault lemma | 局部可解 |
| Dolbeault 定理 | Dolbeault theorem | 上同调同构 |
| $\bar\partial$-闭 | $\bar\partial$-closed | $\bar\partial g = 0$ |

## 6 小结

- **$(p,q)$-形式**与 $d = \partial + \bar\partial$：$\bar\partial$ 是复结构下的微分算子，$\bar\partial f = 0$ 即全纯性。
- **Dolbeault 上同调** $H^{p,q}_{\bar\partial}(D)$：度量 $\bar\partial$ 方程的可解障碍。
- **Dolbeault 引理**：多圆柱上 $q \geq 1$ 时 $H^{p,q}_{\bar\partial} = 0$（局部可解）。
- **Dolbeault 定理**：$H^q(D, \Omega^p) \cong H^{p,q}_{\bar\partial}(D)$——层论与分析合流。
- **推论**：全纯凸域上 $\bar\partial$ 方程恒可解 ⟺ $H^q(D,\mathcal O) = 0$。

在下一节，我们把「$\bar\partial$