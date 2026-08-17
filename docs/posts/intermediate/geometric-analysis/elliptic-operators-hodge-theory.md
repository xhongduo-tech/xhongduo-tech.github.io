---
title: 流形上的椭圆算子（Laplace–Beltrami 算子、Hodge 理论）
date: 2026-08-07
---

# 流形上的椭圆算子（Laplace–Beltrami 算子、Hodge 理论）

<div class="epigraph">
<p>「数学是一门给不同事物取同一个名字的艺术。」</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）《科学与方法》（Science and Method, 1908）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Jost《Riemannian Geometry and Geometric Analysis》调和形式与 de Rham 上同调章 ｜ Peter Li《Geometric Analysis》Ch. 3 ｜ 2026-08-07</p>
</div>

## 为什么从椭圆算子开始

测地线与比较定理给了流形的「几何骨架」，现在要给流形装上「分析引擎」。**Laplace–Beltrami 算子（Laplace–Beltrami operator）**是欧氏空间拉普拉斯算子在黎曼流形上的整体推广，它是几何分析最核心的一阶算子：热方程、波动方程、特征值、调和函数、调和形式全都由它驱动。而 **Hodge 理论**则揭示了一件惊人的事：**调和形式的维数就是流形的 Betti 数——分析学的解空间读出了纯拓扑的信息。**

从课程体系看，本篇是分析进入几何的正式起点。你在第一级《向量微积分》学过的梯度、散度、拉普拉斯，在第二级《PDE 引论》学过的椭圆方程与正则性，到这里全部升级为「与度量有关的几何算子」；而 Hodge 理论与你之前接触的 de Rham 上同调（第三级《代数拓扑》）在此接榫。

<span class="marginnote">庞加莱这句「给不同事物取同一个名字」完美诠释 Hodge 理论：同一个调和形式的方程，同时是分析问题（偏微分方程）与拓扑问题（上同调类）——一个对象，两个名字。这套思想后来孕育了 Atiyah–Singer 指标定理（见第四级《指标定理》专题的入口）。</span>

## 1 Laplace–Beltrami 算子：梯度与散度的复合

在欧氏空间，$\Delta f = \sum_i \partial_i^2 f = \operatorname{div}(\operatorname{grad}f)$。在黎曼流形上，**梯度**与**散度**都需要用度量重写：

$$(\nabla f)^i = g^{ij}\partial_j f, \qquad \operatorname{div} X = \frac{1}{\sqrt{\det g}}\partial_i\big(\sqrt{\det g}\, X^i\big)$$

于是 **Laplace–Beltrami 算子（注意本博客采用的分析学符号约定，它是正的）**定义为

$$\Delta f = -\operatorname{div}(\nabla f) = -\frac{1}{\sqrt{\det g}}\,\partial_i\Big(\sqrt{\det g}\, g^{ij}\,\partial_j f\Big)$$

<span class="marginnote">符号约定是几何分析的第一大坑：几何学家常定义 $\Delta = \operatorname{div}\nabla$（负），分析学家常定义 $\Delta = -\operatorname{div}\nabla$（正）。Peter Li 与 Jost 的教材用正号：此时 $\Delta$ 的特征值 $\lambda\ge0$，热方程写作 $\partial_t u = -\Delta u$。读任何文献前先确认符号，否则一切不等式差一个负号。</span>

**正则性从哪里来**：椭圆算子的本质特征是「主符号正定」——局部看，它几乎是欧氏 Laplacian。椭圆正则性定理（Schauder 估计）说：若系数光滑、$u$ 是方程 $Lu=f$ 的弱解且 $f$ 光滑，则 $u$ 光滑。这就是 Hodge 定理「每个上同调类有光滑代表元」的底层原因，也是之后 Yamabe 方程、调和映射正则性讨论的通用起点。

为什么 $\Delta$ 是「正的」：它满足 Green 恒等式

$$\int_M f\,\Delta f\, dV = \int_M |\nabla f|^2\, dV$$

右边非负，因此 $\Delta$ 是一个自伴、非负的椭圆算子。**椭圆性**指的是主符号（$\sum g^{ij}\xi_i\xi_j$）正定——这保证了特征值离散、有正则性理论、有极大值原理，是与流形「兼容」的分析核心。

## 2 调和函数与极大值原理

**调和函数（harmonic function）**满足 $\Delta u = 0$。它们继承并推广了欧氏空间调和函数的全部性质，其中最关键的是**强极大值原理（strong maximum principle）**：

**强极大值原理**：连通流形上非常值的调和函数不能在内部取到最大值（或最小值）。

这一条是椭圆算子的「结构性事实」，它来自 Hopf 引理与椭圆正则性的结合。<span class="marginnote">极大值原理是分析学「白送」给几何的工具：热核估计、Ricci 流曲率界、调和映射的正则性，几乎处处靠它封顶。它也是「解的整体行为由边界/无穷远决定」这一物理直觉的数学形式。</span>

调和函数在流形上还有一个特有的「均值性质」：对以 $p$ 为心、半径充分小的测地球，调和函数的平均值等于它在球心的值——这是欧氏情形向曲率几何的自然推广，也是 Hodge 理论里调和形式的正则性与唯一性论证的入口。

**一个具体计算：$S^n$ 上的调和 1-形式**。球面的 $\operatorname{Ric}=(n-1)g$ 严格正。对调和 1-形式 $\omega$，Bochner 公式给出 $0 = \Delta|\omega|^2 = 2|\nabla\omega|^2 + 2\operatorname{Ric}(\omega^\sharp,\omega^\sharp) \ge 2(n-1)|\omega|^2$——故 $\omega=0$，即 $S^n$ 上 $H^1=0$、$b_1=0$。这是「正曲率 ⇒ 拓扑平凡」最直接的一击，也是《调和映射》篇 Bochner 公式的预热。

## 3 Hodge 分解与 de Rham 上同调

把 Laplace 算子从函数推广到微分形式：对 $k$-形式 $\omega$，**Hodge Laplacian** 定义为

$$\Delta \omega = (d\delta + \delta d)\omega$$

其中 $d$ 是外微分，$\delta = (-1)^{n(k+1)+1}\star d\star$ 是它的 **形式伴随（codifferential）**，$\star$ 是 Hodge 星算子。<span class="marginnote">$\delta$ 是 $d$ 的「伴随」：$\langle d\alpha,\beta\rangle = \langle \alpha,\delta\beta\rangle$。定义里的符号 $(-1)^{n(k+1)+1}$ 保证 $\delta$ 的自伴性与 $\Delta$ 的正性——它在计算中容易出错，却是整个理论的符号命脉。</span>

**定义：调和形式（harmonic form）**是满足 $\Delta\omega=0$ 的 $k$-形式，记为 $\mathcal{H}^k(M)$。由于

$$\langle \Delta\omega,\omega\rangle = \|d\omega\|^2 + \|\delta\omega\|^2$$

调和形式等价于「既闭又余闭」：$d\omega = 0$ 且 $\delta\omega = 0$。于是每个上同调类里都有「最光滑」的代表元——这就是 **Hodge 定理（Hodge's theorem）**。

## 4 公式解析：Hodge 分解定理

**Hodge 分解定理（Hodge decomposition theorem）**把整个空间切成三块互补的部分：

$$\Omega^k(M) = d\,\Omega^{k-1}(M) \;\oplus\; \delta\,\Omega^{k+1}(M) \;\oplus\; \mathcal{H}^k(M)$$

逐项拆解：

- **第一步，看懂对象**：$\Omega^k$ 是光滑 $k$-形式的全体；$d\Omega^{k-1}$ 是「可微分的」形式（精确形式），$\delta\Omega^{k+1}$ 是「可余微分的」形式，$\mathcal{H}^k$ 是调和形式。
- **第二步，正交性**：$d\alpha$ 与 $\delta\beta$ 自动正交（伴随关系）；调和形式与两者都正交（$d$、$\delta$ 的像是 $\ker d$、$\ker\delta$ 的补）。所以这是正交直和。
- **第三步，存在性**：证明的核心是椭圆正则性——把「给定 $k$-形式 $\eta$，找 $\omega$ 使 $\Delta\omega = \eta$ 且 $\omega\perp\mathcal{H}^k$」化为可解的椭圆方程。紧致流形上 $\Delta$ 有离散谱、Fredholm 性质，保证解存在。这一步把拓扑定理还原为一个 PDE 的存在性问题。
- **第四步，为什么是「同一个名字」**：取任意闭 $k$-形式 $\alpha$，它唯一分解为 $\alpha = \omega + d\beta$，其中 $\omega$ 调和。于是映射

$$\mathcal{H}^k(M) \cong H^k_{dR}(M), \qquad \omega \mapsto [\omega]$$

是同构。**调和形式的最小维数 = de Rham 上同调的维数 = Betti 数 $b_k$**。

这个同构把「代数拓扑：上同调维数」与「分析：调和形式空间」合二为一，并附带一个重要事实：**流形上的正则性**（椭圆正则性）意味着每个上同调类都能被光滑（乃至实解析）的调和形式代表——于是上同调的计算可以在「最光滑」的代表元上进行。

## 5 Hodge 理论在几何分析中的作用

Hodge 分解不只是漂亮的代数结构，它是几何分析的常备工具：

- **Poisson 方程的可解性**：$\Delta u = f$ 可解的充要条件是 $f$ 与常值函数正交——这是 Hodge 分解取 $k=0$ 的直译，调和映射、极小曲面存在性里到处用它。
- **Ricci 曲率与调和形式**：由 **Bochner 公式**（见《调和映射》篇的推导），非负 Ricci 曲率下调和 1-形式平行、且 $H^1 = 0$——**正曲率约束拓扑**（此时 $b_1 = 0$）。这是「曲率正 ⇒ 拓扑简单」的 Hodge 版本。
- **热流方法**：$d\omega/dt + \Delta\omega = 0$ 保持闭形式类不变，随时间把形式「推向」调和代表元——这是 Hodge 定理的热流证明，也是之后调和映射热流、Ricci 流的雏形。
- **等周不等式与特征值**：$k=1$ 的 Hodge 分解与向量场分解给出流形上的 Helmholtz 分解，连接等周常数与谱（见《谱几何》篇）。

| 层级 | 对象 | 核心方程 | 几何/拓扑信息 |
| --- | --- | --- | --- |
| 函数 | $u: M\to\mathbb{R}$ | $\Delta u = 0$ | 调和函数、极大值原理 |
| 1-形式 | $\omega$ | $d\omega=\delta\omega=0$ | $H^1_{dR}$，Bochner 公式 $b_1$ |
| $k$-形式 | $\omega\in\Omega^k$ | $\Delta\omega = 0$ | $H^k_{dR} \cong \mathcal{H}^k$，$b_k$ |
| 张量 | $\sigma$（如对称张量） | 张量 Laplacian | 保 Ricci 流、保调和映射张量 |

**辨析｜易错点：** Hodge 定理要求流形**紧致、无边界**（或对非紧流形要求适当的增长条件）。开流形上调和形式与上同调维数可以不再相等——「调和形式 = 上同调」只在紧致情形成立。

**术语速查**：

| 记号 / 术语 | 含义 | 要点 |
| --- | --- | --- |
| $\Delta$（正约定） | $-\operatorname{div}\nabla$ | 特征值 $\lambda \ge 0$；几何/分析约定相反 |
| 椭圆正则性 | 弱解 ⇒ 光滑解 | Schauder 估计，一切正则性论证的起点 |
| 强极大值原理 | 调和函数不在内部取极值 | Hopf 引理 |
| 调和形式 $\mathcal{H}^k$ | $\Delta\omega = 0$ | 等价于 $d\omega = \delta\omega = 0$ |
| Hodge 星算子 $\star$ | 正交补的定向同构 | 定义 $\delta$ 的核心 |
| Hodge 分解 | $\Omega^k = d\Omega^{k-1}\oplus\delta\Omega^{k+1}\oplus\mathcal{H}^k$ | 正交直和 |
| Betti 数 $b_k$ | $\dim\mathcal{H}^k = \dim H^k_{dR}$ | 分析读出拓扑 |

## 6 小结

- **Laplace–Beltrami 算子** $\Delta = -\operatorname{div}\nabla$：正、自伴、椭圆，Green 恒等式 $\int f\Delta f = \int|\nabla f|^2$。
- **调和函数**满足强极大值原理，是几何分析「封顶」的工具。
- **Hodge Laplacian** $\Delta = d\delta + \delta d$，调和形式 = 闭且余闭。
- **Hodge 分解** $\Omega^k = d\Omega^{k-1}\oplus\delta\Omega^{k+1}\oplus\mathcal{H}^k$，从而 $\mathcal{H}^k \cong H^k_{dR}$，调和形式维数 = Betti 数。
- **Bochner 思想**：曲率条件通过 Bochner 公式约束调和形式 → 约束上同调。

在下一节，我们要给这套静态的椭圆理论装上时间维——**热方程与热核**：从 $\partial_t u = -\Delta u$ 出发，研究热核的渐近展开与 Li–Yau 的梯度估计，看「热量如何沿测地线扩散」。
