---
title: 调和映射（能量泛函、Bochner 公式、Eells–Sampson 存在性定理）
date: 2026-08-07
---

# 调和映射（能量泛函、Bochner 公式、Eells–Sampson 存在性定理）

<div class="epigraph">
<p>「自然总是用最少的东西做最多的事。」</p>
<footer>—— 约翰内斯 · 开普勒（Johannes Kepler）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Jost《Riemannian Geometry and Geometric Analysis》调和映射章 ｜ Schoen–Yau《Lectures on Differential Geometry》 ｜ 2026-08-07</p>
</div>

## 为什么从调和映射开始

测地线是「曲线到流形」的能量极小者，**调和映射（harmonic map）**把它推广到「流形到流形」：$\varphi:(M,g)\to(N,h)$ 是能量泛函的临界点。这个理论把几何分析的几乎全部武器——变分法、Bochner 公式、热流、正则性——揉成一个问题，并为极小曲面（等距浸入且能量极小 → 调和）、Yamabe 型非线性方程与拓扑分类提供了统一框架。**Eells–Sampson 存在性定理（1964）**则是一个里程碑：它首次用**热流方法（heat flow method）**证明非线性椭圆方程的全局解存在。

从课程体系看，本篇把第二级《变分法》的 Lagrange 乘子思想与《PDE 引论》的热流方法搬到「映射空间」，并与前面的 Hodge 理论（Bochner 公式的模板）和 Yamabe 问题（非线性椭圆的存在性模板）接榫。调和映射也是第四级《广义相对论》中「调和坐标」与规范场论中 Yang–Mills 理论的近亲。

<span class="marginnote">开普勒这句话（「自然用最小量做事」）正是极小原则的哲学。调和映射的力学图像：把 $M$ 想成一张橡皮膜，$N$ 想成一个曲面，膜被边界钉在 $N$ 上，松开后它收缩到能量最小的形态——这形态就是调和映射。费马原理、弹性膜、肥皂膜全部是同一原则。</span>

## 1 能量泛函与调和映射的定义

设 $\varphi:M\to N$ 光滑。它的**微分（differential）** $d\varphi$ 把切空间送到切空间，而**能量密度（energy density）**用两边的度量度量「伸缩程度」：

$$e(\varphi) = \frac12\,|d\varphi|^2 = \frac12\, g^{ij}\,\langle \partial_i\varphi,\partial_j\varphi\rangle_{h}$$

**能量泛函（energy functional）**为

$$E(\varphi) = \int_M e(\varphi)\, dV_g$$

**定义：调和映射（harmonic map）是能量泛函的临界点。** 一阶变分公式给出**张力场（tension field）**

$$\tau(\varphi) = \operatorname{tr}_g(\nabla d\varphi) = g^{ij}\big(\partial_{ij}\varphi - \Gamma^k_{ij}\partial_k\varphi + \Gamma^{\alpha}_{\beta\gamma}(\varphi)\,\partial_i\varphi^\beta\partial_j\varphi^\gamma\big)$$

调和 ⇔ $\tau(\varphi) = 0$。当 $M$ 是一维区间，这就是测地线方程；当 $N = \mathbb{R}$（平直目标），这就是 Laplace 方程 $\Delta\varphi=0$。<span class="marginnote">张力场的表达式可这样读：第一项是「流形上的二阶导」，第二项修正源流形 $M$ 的弯曲，第三项修正目标 $N$ 的弯曲。所以「调和」的意思是「在两边流形的弯曲参照下，映射不产生张力」——两个弯曲互相抵消。</span>

### 1.1 三个基本例子

- **测地线**：$M$ 是一维区间或圆，能量泛函就是测地线能量，调和映射即测地线——上一篇的全部理论在此回收。
- **谐函数**：$N=\mathbb{R}$（平直目标），张力场退化为主 Laplacian $\Delta\varphi$，调和映射即调和函数——本专题第四篇的领地。
- **全纯函数**：从 Riemann 曲面到球面或黎曼面的**全纯映射自动调和**（由 Cauchy–Riemann 方程直接验证）。全纯 ⇔ 调和是复几何与几何分析的第一层连接，也预示了 Kähler 几何中「调和形式与全纯对象」的交织。

## 2 Bochner 公式：曲率如何作用于调和映射

调和映射理论的第一把几何钥匙是 **Bochner 公式（Bochner formula）**。设 $\varphi$ 调和，$f = e(\varphi)$ 是能量密度，则

$$\Delta f = |\nabla d\varphi|^2 + \big\langle \nabla f,\dots\big\rangle + \operatorname{Ric}_M(d\varphi,d\varphi) - \big\langle R^N(d\varphi,d\varphi)d\varphi,\;d\varphi\big\rangle$$

更标准的形式（去掉交叉项后）对调和映射的能量密度有

$$\Delta e(\varphi) \ge |\nabla d\varphi|^2 - C\,|d\varphi|^2 - |\operatorname{Ric}_M^+||d\varphi|^2 - |R^N||d\varphi|^4$$

逐项拆解其**几何意义**（这正是 Schoen–Yau《Lectures on Differential Geometry》中的核心工具）：

- **曲率项 $\operatorname{Ric}_M(d\varphi,d\varphi)$**：源流形 $M$ 的 Ricci 曲率沿映射「铺开」的方向挤压或拉伸能量。若 $\operatorname{Ric}_M \ge 0$，这一项非负，倾向于把能量密度推高。
- **曲率项 $R^N(d\varphi,d\varphi)$**：目标流形 $N$ 的曲率。若 $N$ 的截面曲率 $\le 0$，则该项非负——**负曲率目标「吸收」能量而非「制造」**。
- **$\Delta f$ 的解释**：对调和映射，能量密度的 Laplacian 与两边的曲率、以及 $|d\varphi|$ 的次幂相耦合。**极大值原理**由此上线：若两边的曲率条件恰当，则 $f$ 被均匀有界，进而 $|d\varphi|$ 有界，再由椭圆正则性推出 $d\varphi$ 光滑——这就是**调和映射的「曲率条件 ⇒ 先验估计」**模板。

**重点：Bochner 公式把「存在性」变成「先验估计」问题。** 一旦 $|d\varphi|$ 有一致上界，正则性理论（Moser 迭代 + De Giorgi–Nash）自动跟上。<span class="marginnote">Bochner 公式得名于 Salomon Bochner（1940 年代），他在 Hodge 理论的背景下证明「正 Ricci 曲率的紧流形上 $H^1=0$」。调和映射的版本由 Eells–Sampson 与 Schoen–Yau 系统使用。它是一个「几何 → 分析」的翻译器：曲率条件翻译成能量密度的偏微分不等式。</span>

## 3 公式解析：Eells–Sampson 热流方法

**Eells–Sampson 存在性定理（Eells–Sampson theorem, 1964）**：设 $(M,g)$ 紧致，$\operatorname{Ric}_M \ge 0$，$(N,h)$ 的截面曲率 $\le 0$。则任一光滑映射 $\varphi_0:M\to N$ 同伦于一个调和映射。

证明的关键是把「解非线性椭圆方程」换成「解非线性抛物方程」——**调和映射热流（harmonic map heat flow）**：

$$\partial_t \varphi = \tau(\varphi), \qquad \varphi(0) = \varphi_0$$

逐项拆解这个「时间化」的魔法：

- **第一步，为什么是 $\tau$**：能量沿热流的导数恰好是 $-E(\varphi_t)$ 的时间导数 $= -\int|\tau|^2$——热流是能量泛函的**梯度流**，能量单调下降，就像橡皮膜被弹簧拉向能量低谷。
- **第二步，Bochner 公式救场**：在 $\operatorname{Ric}_M\ge0$、$K_N\le0$ 下，能量密度满足

$$\partial_t e \le -\Delta e + C\,|d\varphi|^4 \ \Rightarrow\ \partial_t \max e \le C\,(\max e)^2$$

由极大值原理，能量密度在有限时间内有一致上界（经典抛物极大值原理给出 $L^\infty$ 估计，配合 Sobolev 上界）。
- **第三步，解的收敛**：能量单调且非负，故收敛；再对 $e$ 的上界配合抛物正则性，推出 $t\to\infty$ 时 $\tau(\varphi_t)\to0$。需要进一步的紧性论证（能量恒等 + 正则性）得到光滑极限 $\varphi_\infty$。
- **第四步，为什么成立**：核心是**先验估计**——曲率条件保证热流在 $t<\infty$ 内不爆破（$e$ 一致有界），于是热流能一直跑下去并收敛。**「曲率条件 ⇒ 不爆破 ⇒ 存在性」**是 Eells–Sampson 的全部哲学，后来被 Hamilton 原封不动地搬到 Ricci 流上（见《Ricci 流引论》篇）。

**一个漂亮的推论（Schoen–Yau / Eells–Sampson）**：若 $(N,h)$ 截面曲率 $\le 0$，则从任何紧流形到 $N$ 的每个同伦类里都有调和代表元——特别地 $N$ 的**每个同伦类由调和映射分类**。负曲率流形的拓扑由调和映射「探测」。

**唯一性（Hartman）**：若目标截面曲率 $<0$ 且映射落入某个凸区域，则同伦类里的调和映射唯一。这使「调和代表元」成为同伦类的规范形——负曲率目标的拓扑被调和映射完全规范化，为「调和映射分类拓扑」提供了精确的数学陈述。

## 4 调和映射与几何分类

调和映射的价值一半在存在性，一半在它作为「几何探针」：

- **极小曲面是特殊调和映射**：等距浸入 $\varphi:M\to N$ 是调和当且仅当 $M$ 的像在 $N$ 中是极小曲面（$H=0$，见《极小曲面与平均曲率流》篇）。调和映射理论因此覆盖了极小曲面理论。
- **正曲率源流形**：若 $\operatorname{Ric}_M > 0$，由 Bochner 公式，任何调和映射的能量密度满足更强的刚性——**Rigidity**：若 $N$ 也是正曲率且映射非常值，则要求 $d\varphi=0$ 或满足严格的不等式链，从而推出同伦平凡。这是「正曲率 ⇒ 映射刚性」的几何分析表达。
- **到双曲空间、负曲率对称空间的调和映射**：Corlette、Gromov–Schoen 把调和映射用于研究格的刚性（Margulis 超刚性），是调和映射「进入群论」的著名应用。

| 源流形曲率 | 目标流形曲率 | 结论（Eells–Sampson 谱系） |
| --- | --- | --- |
| $\operatorname{Ric}_M \ge 0$ | $K_N \le 0$ | 每同伦类有调和代表元 |
| $\operatorname{Ric}_M > 0$，$K_N \le 0$ | 上一条 + 非平凡类刚性 | 正 Ricci 约束到负曲率目标的映射 |
| $K_M \ge 0$，$K_N < 0$ | 同伦类唯一性 | 负曲率目标的同伦唯一性 |
| 任意 | $N=\mathbb{R}^n$ | 调和 = 谐函数，回到 Laplace 方程 |

<span class="marginnote">调和映射的「探针」用法最震撼的例子：Gromov–Schoen 用它研究 $p$-进群的表示，Corlette 用它证明某些格（lattice）的超刚性——把「映射是否调和」当作判别「群结构是否刚性」的仪器。几何分析由此进入代数。这是本专题「前沿」方向的一瞥。</span>

**存在性之外：正则性**。调和映射在临界维度 $n=2$（能量维数为 2）是特殊的：任意有限能量的弱解自动光滑（Sacks–Uhlenbeck、Schoen–Uhlenbeck）。$n\ge3$ 时则可能出现能量集中的气泡奇点——**气泡分析（bubble analysis）**是调和映射现代理论的中心课题。

**辨析｜易错点：** Eells–Sampson 要求目标曲率 $\le 0$；目标正曲率时（如 $N=S^2$），调和映射可能在有限时间内**爆破**——能量集中成「气泡」，这是调和映射理论的现代核心（bubble analysis，见《前沿专题》篇对 Perelman 的铺垫）。存在性不再是「免费」的。

**术语速查**：

| 记号 / 术语 | 含义 | 要点 |
| --- | --- | --- |
| 能量密度 $e(\varphi)$ | $\frac12\|d\varphi\|^2$ | 两流形度量下的伸缩程度 |
| 张力场 $\tau(\varphi)$ | $\operatorname{tr}_g\nabla d\varphi$ | 调和映射 $\tau=0$ |
| Bochner 公式 | $\Delta e \ge \|\nabla d\varphi\|^2 - C\|d\varphi\|^2 - \cdots$ | 曲率条件 → 先验估计 |
| 调和映射热流 | $\partial_t\varphi = \tau(\varphi)$ | 能量梯度流 |
| Eells–Sampson | $\operatorname{Ric}_M\ge0$，$K_N\le0$ ⇒ 调和代表元 | 热流不爆破 + 收敛 |
| 气泡（bubble） | $n\ge3$ 时能量的点集中 | 目标正曲率时发生 |
| 调和代表元 | 同伦类中的调和映射 | 负曲率目标时唯一 |

## 5 小结

- **调和映射** $\tau(\varphi)=0$：能量泛函的临界点，测地线与谐函数的共同推广。
- **Bochner 公式**：把曲率条件翻译成能量密度的微分不等式，是「存在性 ⇐ 先验估计」的桥梁。
- **Eells–Sampson 定理**：$\operatorname{Ric}_M\ge0$、$K_N\le0$ 时热流不爆破、能量下降并收敛，每同伦类有调和代表元。
- **热流方法模板**：梯度流 + Bochner 先验估计 + 抛物正则性 —— 后来 Ricci 流、平均曲率流的直接祖先。
- **探针用途**：负曲率目标的同伦分类、格刚性、以及极小曲面的统一。

在下一节，我们把「能量极小」的思想从映射推向**子流形**——**极小曲面与平均曲率流**：研究面积泛函的临界点、稳定性不等式、单调性公式与奇点分析。
