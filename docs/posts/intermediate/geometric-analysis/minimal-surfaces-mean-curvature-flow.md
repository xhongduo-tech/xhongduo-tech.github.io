---
title: 极小曲面与平均曲率流（稳定性不等式、单调性公式、奇点分析）
date: 2026-08-07
---

# 极小曲面与平均曲率流（稳定性不等式、单调性公式、奇点分析）

<div class="epigraph">
<p>「自然乐于简洁。」</p>
<footer>—— 艾萨克 · 牛顿（Isaac Newton）</footer>
</div>

<div class="article-byline">
<p>第二级 · 几何分析 ｜ Schoen–Yau《Lectures on Differential Geometry》 ｜ Jost 子流形章 ｜ 2026-08-07</p>
</div>

## 为什么从极小曲面开始

面积泛函的临界点是**极小曲面（minimal surface）**——肥皂膜、普拉托问题、正质量定理，都是它的故事。而**平均曲率流（mean curvature flow, MCF）**让曲面随时间沿平均曲率方向流动，是面积泛函的梯度流。这一篇把前面学到的变分法、Bochner 公式、热流方法全部用到「子流形」上，并用**单调性公式（monotonicity formula）**与**奇点分析（singularity analysis）**回答：曲面如何收缩、在何处爆破、如何分类。

从课程体系看，本篇与《调和映射》互为表里（极小曲面是等距的调和映射），并直接通向第四级《广义相对论》的正质量定理——Schoen–Yau 用极小曲面证明正能量。它也是 Perelman 处理 Ricci 流奇点（用曲率集中）的几何直觉来源。

<span class="marginnote">牛顿「自然乐于简洁」恰是极小曲面的信条：肥皂膜自会找到面积最小（严格说是极小）的构型。普拉托（Plateau）19 世纪用铁丝框浸肥皂液观察的「最小面积曲面」，直到 1930 年才由 Douglas 与 Radó 严格证明存在——Douglas 因此获 1936 年首届费尔兹奖之一。极小曲面与几何分析的交汇是 20 世纪数学的高光。</span>

## 1 第一变分与平均曲率

设 $\Sigma^n \subset M^{n+1}$ 是浸入子流形，面积泛函 $\operatorname{Area}(\Sigma) = \int_\Sigma dV$。一族形变 $\Sigma_t$ 由法向速度场 $V = V^\perp$ 驱动，**第一变分公式（first variation formula）**为

$$\frac{d}{dt}\operatorname{Area}(\Sigma_t)\Big|_{t=0} = -\int_\Sigma \langle V, H\rangle \, dV$$

其中 **平均曲率向量（mean curvature vector）** $H = \operatorname{tr} A$ 是第二基本形式的迹（$A$ 是第二基本形式张量，$\operatorname{tr}A$ 沿法向）。

**定义：极小曲面（minimal surface）**：平均曲率 $H \equiv 0$ 的浸入子流形——面积泛函的临界点。注意「极小」≠「面积局部最小」：临界点可能只是鞍点。面积局部最小的叫**稳定极小（stable minimal）**，需要第二变分检查。<span class="marginnote">经典的例子：平面（$H=0$）、悬链面（catenoid）、螺旋面（helicoid）、以及球面（$H\ne0$，是 MCF 的收缩解而非极小）。$H$ 的符号依赖法向选取，但「$H=0$」是良定义的。普拉托问题（Plateau problem）——给定边界找面积最小曲面——由 Douglas、Radó 解决，是变分法在几何中的第一个重大胜利。</span>

**普拉托问题的边界形态**：平面边界给出平面盘；两个平行圆环给出悬链面（catenoid）；正交的线段给出螺旋面。这些经典极小曲面是「能量极小的变分存在性」的具体见证，也是研究稳定性与奇点的天然实验室。

## 2 第二变分与稳定性不等式

**第二变分公式（second variation）**决定临界点的稳定性。设 $\Sigma$ 极小，法向变分 $f\nu$（$f$ 是标量函数，$\nu$ 是单位法向），则

$$\frac{d^2}{dt^2}\operatorname{Area}\Big|_{t=0} = \int_\Sigma \Big(|\nabla f|^2 - \big(|A|^2 + \operatorname{Ric}_M(\nu,\nu)\big) f^2\Big)\, dV$$

**稳定性不等式（stability inequality）**：$\Sigma$ 是（局部）面积最小 ⇒ 上式对一切 $f$ 非负，即

$$\int_\Sigma |\nabla f|^2 \ge \int_\Sigma \big(|A|^2 + \operatorname{Ric}_M(\nu,\nu)\big) f^2$$

逐项拆解其几何内容：

- **$|\nabla f|^2$ 项**：曲面「弯曲变分」的代价，类比测地线的扭曲惩罚项。
- **$|A|^2$ 项**：第二基本形式模长平方——曲面自身的弯曲越大，越容易失稳（因为它能「自发皱缩」）。
- **$\operatorname{Ric}_M(\nu,\nu)$ 项**：环境流形沿法向的 Ricci 曲率。**环境正曲率使极小曲面更容易不稳定**——这正是正质量定理的直觉来源：正能量/正曲率环境里，极小曲面会「被抓进」弯曲而坍缩。

取 $f\equiv1$（平行移动）得到：稳定极小曲面满足 $\int (|A|^2+\operatorname{Ric}(\nu,\nu)) \le 0$ 不可能对全正曲率成立——**正 Ricci 曲率的紧致流形中不存在稳定（闭）极小曲面**。这就是 Schoen–Yau 证明正质量定理的第一个杠杆。

## 3 平均曲率流与单调性公式

**平均曲率流（mean curvature flow）**是面积泛函的梯度流：曲面 $\Sigma_t$ 满足

$$\partial_t x = H = \operatorname{tr}A$$

其中 $x:\Sigma_t\to M$ 是位置向量场。与调和映射热流、Ricci 流同一血脉：**几何量沿自己的梯度流演化**。典型行为：平面不动；半径 $R$ 的球面以 $\dot R = -\frac{n}{R}$ 收缩，$T = R_0^2/2n$ 时刻坍缩成一点（自相似收缩，一个「奇点」）；圆柱面在某个方向上收缩成线段。

MCF 最深刻的工具是 **Huisken 单调性公式（monotonicity formula, 1984）**：在 $\mathbb{R}^{n+1}$ 中，对后向热核（backward heat kernel）

$$\rho(x,t) = \frac{1}{(4\pi(T-t))^{n/2}}e^{-\frac{|x|^2}{4(T-t)}}$$

有

$$\frac{d}{dt}\Big((T-t)^{-n/2}\int_{\Sigma_t} e^{-\frac{|x|^2}{4(T-t)}} dV\Big) \le 0$$

**几何意义**：沿 MCF，「以奇点时刻 $T$ 为中心的高斯权重面积」单调递减。这使奇点分析成为可能：把曲面在奇点处「放大」（blow-up），极限曲面恰好是**自相似收缩解**（shrinkers），且因单调性而满足刚性方程。**奇点分类 = 自相似解分类**，这是 Huisken、White、Ilmanen 学派的基本纲领。<span class="marginnote">Huisken 单调性公式是「奇点分析」的标尺：它把「何时、何处爆破」的模糊直觉变成精确的单调量。Perelman 在 Ricci 流中发明了完全平行的单调量（W 泛函、约化体积），自述灵感来自 Huisken 的单调性与熵——这是几何流领域最著名的思想移植之一。</span>

## 4 公式解析：奇点分析的自相似收缩

奇点分析的典型结论来自 **Huisken 定理（Huisken's theorem, 1990）**：紧凸超曲面在 $\mathbb{R}^{n+1}$ 中沿 MCF，在有限时刻 $T$ 收缩为一点；放大后（blow-up）收敛到**圆球** $S^n$。关键步骤：

- **第一步，凸性保持**：MCF 下，平均曲率 $\ge 0$（凸性）被极大值原理保持；Huisken 证明更强的「第二基本形式张量的凸性传播」。
- **第二步，曲率比值控制**：通过 Harnack 型估计与单调性公式，控制主曲率 $\kappa_i$ 的比值：$\kappa_{\max}/\kappa_{\min} \to 1$——曲率趋向各向同性。
- **第三步，吹胀极限**：在奇点 $(0,T)$ 处取时空放大 $x \mapsto \lambda(x-x_0), t \mapsto \lambda^2(t-T)$，单调性公式给出极限曲面是**自相似收缩解**：$H = \frac{1}{2}\langle x,\nu\rangle$（shrinker 方程）。
- **第四步，分类**：$\mathbb{R}^{n+1}$ 中紧自相似收缩解中，唯一稳定的是圆球（Huisken）。由凸性保持，极限必须是圆球，从而奇点是「球型」——**整族曲面在奇点处被圆球封装**。

**重点：奇点分析 = 单调性（给出极限存在）+ 自相似分类（决定极限形状）+ 刚性传播（把极限信息拉回有限时间）。** 这套三步曲在后来的 Ricci 流奇点分析（Hamilton–Perelman）、调和映射气泡分析（Sacks–Uhlenbeck）中反复上演。

## 5 极小曲面在几何分析中的角色

极小曲面与 MCF 不只是「自己的故事」，它们是几何分析最锋利的探针：

- **正质量定理（Schoen–Yau, 1979）**：用极小曲面作为「测试曲面」，把 ADM 质量的非负性转化为极小曲面的稳定性不等式——见《前沿专题》篇详述。
- **恒曲率与对称性**：等参曲面、以及具有常数平均曲率的曲面（CMC）在相对论中作为「准局域质量」的载体（见《广义相对论》专题）。
- **拓扑探测**：稳定极小曲面在负曲率环境下的存在性约束环境流形的拓扑（Schoen–Yau 的拓扑刚性结果）。
- **MCF 在现代**：低维拓扑中不可压缩曲面的流、以及 2020 年代以来 MCF 在代数几何（奇点理论）的横空应用。

**稳定性不等式的等价物**：在黎曼流形而非欧氏空间中，稳定性不等式里出现环境曲率 $\operatorname{Ric}_M(\nu,\nu)$；当环境是渐近平坦流形（如三维空间外加上一个孤立系统）时，这个环境曲率项恰由 ADM 质量控制——这就是 Schoen–Yau 把极小曲面稳定性翻译为正质量的关键一步。

**一个数值印象**：三维欧氏空间中，给定边界的极小构型往往对应肥皂膜；闭曲面的 Willmore 能量下界 $4\pi$（球面取到）是经典的 **Willmore 猜想**（已由 Marques–Neves 证明），它把「曲率积分」与「拓扑」重新焊在一起。

| 对象 | 定义 | 关键工具 | 应用 |
| --- | --- | --- | --- |
| 极小曲面 | $H=0$ | 第二变分、稳定性不等式 | 普拉托问题、正质量定理 |
| 稳定极小 | 面积局部最小 | 稳定性不等式 | Schoen–Yau 正质量证明 |
| MCF | $\partial_t x = H$ | Huisken 单调性 | 奇点分析、分类 |
| Shrinker | $H = \tfrac12\langle x,\nu\rangle$ | 自相似分类 | 奇点极限、Perelman 熵 |

<span class="marginnote">「极小曲面 ⇔ 面积极小 + 调和」与「MCF ⇔ 面积梯度流 + 单调性」这两条平行线，本质都是「能量（面积）泛函的变分理论」。把「面积」换成「$L^2$ 曲率泛函」，就得到 Willmore 流；换成「标量曲率泛函」，就得到 Yamabe 流——极小曲面是这张变分全景图的起点。</span>

**辨析｜易错点：** 「极小」是 $H=0$（临界点），不等于「绝对面积最小」。且 $H$ 的定义在余维数 $>1$ 时是法向向量（非标量）；「$H$ 是标量」只在超曲面（余维 1）且选定单位法向时成立。另外 MCF 的单调性公式在 $\mathbb{R}^{n+1}$ 中才有最干净的形状，在一般环境流形中需要曲率修正项。

**术语速查**：

| 记号 / 术语 | 含义 | 要点 |
| --- | --- | --- |
| 平均曲率向量 $H$ | 第二基本形式的迹 | 极小曲面 $H=0$ |
| 稳定极小 | 第二变分非负 | 稳定性不等式 |
| 第一变分公式 | $\frac{d}{dt}\mathrm{Area} = -\int\langle V,H\rangle$ | 面积梯度方向 |
| 单调性公式 | 高斯权重面积单调递减 | Huisken 1984 |
| Shrinker | 自相似收缩解 $H=\frac12\langle x,\nu\rangle$ | 奇点极限 |
| Blow-up 放大 | $x\mapsto\lambda(x-x_0)$，$t\mapsto\lambda^2(t-T)$ | 奇点处取极限 |
| 普拉托问题 | 给定边界求面积最小曲面 | Douglas / Radó 1930s |

## 6 小结

- **第一变分**：面积变化率 $= -\int\langle V,H\rangle$，极小曲面 $H=0$。
- **第二变分 / 稳定性不等式**：$\int|\nabla f|^2 \ge \int(|A|^2+\operatorname{Ric}(\nu,\nu))f^2$——环境正曲率使极小曲面失稳。
- **MCF** $\partial_t x = H$：面积梯度流，球面收缩、平面不动。
- **Huisken 单调性公式**：高斯权重面积单调递减，奇点分析由此入手。
- **奇点分析三步曲**：单调性 → 自相似分类 → 刚性传播；与 Ricci 流奇点理论同构。

在下一节，我们终于研究「整个度量」随时间如何流动——**Ricci 流引论**：Hamilton 的短时间存在性、极大型原理，以及为什么它让几何分析的武器库在三维爆发。
