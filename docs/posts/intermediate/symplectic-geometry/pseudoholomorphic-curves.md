---
title: 伪全纯曲线与 Gromov 紧致性
date: 2026-08-07
---

# 伪全纯曲线与 Gromov 紧致性

<div class="epigraph">
<p>1985 年，Gromov 把全纯曲线带进辛几何，辛拓扑就此诞生。</p>
<footer>—— 海因里希 · 格罗莫夫 传（Dusa McDuff 语）</footer>
</div>

<div class="article-byline">
<p>第二级 · 辛几何 ｜ McDuff & Salamon 第11章 ｜ 2026-08-07</p>
</div>

## 为什么从伪全纯曲线开始

在《近复结构与相容三元组》里我们学会了给辛流形配相容近复结构 $J$。现在让 $J$ 干大事：**伪全纯曲线（pseudoholomorphic curve）** 是黎曼面到辛流形的「$J$-全纯映射」——它把复结构 $j$ 送到近复结构 $J$。为什么这种曲线是辛几何的超级武器？因为**辛流形（一般没有复结构）里居然存在「全纯曲线」，而这些曲线携带的拓扑信息是辛不变量**。Gromov 在 1985 年的革命性洞察有两层：第一，伪全纯曲线总是存在（给定同调类与相容 $J$，有紧致模空间）；第二，它们满足**紧致性**——曲线族不会跑丢，只会「冒泡」（bubbling）。紧致性 + 模空间的有限维性，让「数曲线」成为可能，这就是后面 Floer 同调、Gromov-Witten 不变量的分析地基。这一篇讲定义、能量恒等式与紧致性的直观骨架。<span class="marginnote">从课程地图看：这是第3篇的心脏。前面所有铺垫（相容三元组、辛纤维丛、约化）都在为「用曲线探测辛流形」服务；后面非压缩定理、Hofer 几何、Floer 同调全从这里取血。</span>

## 1 伪全纯曲线与 Cauchy-Riemann 方程

设 $(\Sigma, j)$ 是黎曼面（带复结构 $j$，如 $S^2$ 或带孔曲面），$(M, \omega, J)$ 是辛流形带相容近复结构。

**伪全纯曲线（$J$-holomorphic curve）**：光滑映射 $u: \Sigma \to M$ 满足

$$
du \circ j = J \circ du
$$

即 $u$ 的切映射「保复结构」。这等价于 Cauchy-Riemann 型方程

$$
\bar\partial_J u := \frac{1}{2}(du + J \circ du \circ j) = 0
$$

**$J$-全纯 = $\bar\partial_J u = 0$**。<span class="marginnote">在局部坐标下，若 $j = i$（$\Sigma = \mathbb{C}$），方程变成 $u_x + J(u)u_y = 0$——这是「非线性柯西-黎曼方程」。当 $M = \mathbb{C}^n$、$J = i$ 时它就是经典全纯映射的条件 $u_{\bar z} = 0$。所以伪全纯曲线是「全纯映射」在任意辛流形上的推广。</span>

**为什么需要 $J$ 与 $\omega$ 相容？** 因为能量（下节）要用 $g = \omega(\cdot, J\cdot)$ 定义，而且相容性保证「能量 = 面积」的恒等式成立。没有相容性，理论的分析性质（紧致性、正则性）会崩坏。

## 2 能量与面积：能量恒等式

**能量（energy）**：对 $u: \Sigma \to (M, \omega, J)$，

$$
E(u) = \frac{1}{2}\int_\Sigma |du|^2 \, d\mathrm{vol}_\Sigma
$$

范数由 $g = \omega(\cdot, J\cdot)$ 诱导。

**能量恒等式**：若 $u$ 是 $J$-全纯的，则

$$
E(u) = \int_\Sigma u^*\omega
$$

特别地，$\int u^*\omega$ 只依赖同调类 $[u] \in H_2(M;\mathbb{Z})$——**能量是拓扑量**。证明要点：把 $|du|^2$ 分解为「全纯部分」与「反全纯部分」，$\bar\partial u = 0$ 时反全纯部分为零，剩下 $\int u^*\omega$。<span class="marginnote">这是整套理论的第一条命脉：<strong>能量由同调类控制</strong>。于是「固定同调类的曲线族」有界能量，紧致性问题变成「能量有界 ⇒ 有收敛子列」的分析问题。能量恒等式的另一个形式：对任何映射 $E(u) \ge \int u^*\omega$，等号当且仅当 $J$-全纯——「全纯曲线是能量最小者」。</span>

**推论（为什么曲线不会消失）**：$[u] = A \in H_2(M;\mathbb{Z})$ 固定时，$E(u) = \int_A\omega$ 固定。若 $\int_A\omega > 0$，曲线族在固定同调类里「有正的能量预算」，可以讨论极限——紧致性就有了立足点。

## 3 模空间与指标

固定（同调类 $A$，曲线亏格 $g$，标记点 $k$），考虑所有 $J$-全纯曲线构成的**模空间（moduli space）**

$$
\mathcal{M}_{g,k}(M, A) = \{ (u, z_1, \dots, z_k) : u \text{ $J$-全纯}, [u] = A \} / \text{参数化同构}
$$

除以「黎曼面的共形自同构」与重参数化。**模空间是有限维流形（对正则 $J$）**，维数由 **Atiyah-Singer 指标定理**给出：

$$
\dim \mathcal{M}_{g,k} = (1-g)(\dim_\mathbb{C} M - 3) + 2c_1(A) + 2k
$$

（对 $\Sigma = S^2$：$\dim = 2c_1(A) + 2n + 2k - 6$，其中 $\dim_\mathbb{C}M = n$）。<span class="marginnote">指标公式的直觉：线性化 $\bar\partial$ 算子 $D_u = \bar\partial_J$ 在 $u$ 处的核（模空间切向）与余核（障碍）之差由拓扑量 $c_1(A)$ 决定。正则性定理说「对一般 $J$，$D_u$ 满射，模空间是光滑流形」。Fredholm 指标是这套有限维理论的温度计。</span>

**"正则 $J$"（regular almost complex structure）**：使得所有模空间光滑的 $J$，在 $\mathcal{J}(M,\omega)$ 中是**第二纲（Baire 剩余）**的。**「一般 $J$ 的模空间是光滑流形」——这是 Gromov 紧致性的分析前提**，后面数曲线全靠它。

## 4 公式解析：能量恒等式

**核心公式：**

$$
E(u) = \frac{1}{2}\int_\Sigma |du|^2\, d\mu = \int_\Sigma u^*\omega \quad \text{（$u$ $J$-全纯）}
$$

拆解：

- **第一步，分解导数**：把 $du$ 按 $J$ 分成全纯与反全纯部分：$du = \partial u + \bar\partial u$（$\partial u = \frac{1}{2}(du - J\circ du\circ j)$，$\bar\partial u = \frac{1}{2}(du + J\circ du\circ j)$）。两点满足 $|du|^2 = |\partial u|^2 + |\bar\partial u|^2$（正交分解）。
- **第二步，连接能量与面积**：计算 $u^*\omega = \omega(du\cdot, du\cdot)$。对 $J$-相容三元组，$u^*\omega = (|\partial u|^2 - |\bar\partial u|^2)\, d\mu$（$d\mu$ 是 $\Sigma$ 上的黎曼面积元）——**面积 = 全纯能量 - 反全纯能量**。
- **第三步，用方程消项**：$u$ $J$-全纯 ⇒ $\bar\partial u = 0$ ⇒ $|\bar\partial u|^2 = 0$。于是 $\int u^*\omega = \int |\partial u|^2 d\mu = \frac{1}{2}\int |du|^2 d\mu = E(u)$。
- **第四步，拓扑化**：$[\omega]$ 是闭形式，$\int_\Sigma u^*\omega$ 只依赖 $[u] = A$：等于 $\langle [\omega], A \rangle$。**所以 $E(u) = \langle[\omega], A\rangle$ 只由同调类决定**——能量是拓扑的，紧致性由此出发。

**直觉总结：** 能量恒等式把「分析量（导数范数）」与「拓扑量（辛配对）」焊在一起。它同时给出两条命脉：能量有界 ⇒ 有紧致性；能量是拓扑量 ⇒ 固定类有固定预算。**没有它，整个理论无从谈起。**

## 5 Gromov 紧致性：冒泡

**Gromov 紧致性定理**：设 $u_\nu: S^2 \to (M, \omega, J)$ 是一列 $J$-全纯曲线，能量一致有界 $E(u_\nu) \le C$，且同调类 $[u_\nu] = A$ 固定。则存在子列（重参数化后）收敛，极限是**稳定的破裂曲线（stable broken curve）**：

- 一串曲线 $u^{(1)}, \dots, u^{(m)}$，每段在 $S^2$ 上 $J$-全纯；
- 相邻段在一个点相接（节点）；
- **能量守恒**：$\sum E(u^{(k)}) = E(A)$，且每段的能量 $\ge \hbar$（最小能量阈值，见下）。

**冒泡（bubbling）**：曲线的「一部分能量」在极限处**收缩成一个小球**，形成新的曲线段——就像气泡从主体上分离。关键量是**最小能量阈值**

$$
\hbar := \inf\{ E(A) : A \neq 0, A \in H_2(M;\mathbb{Z}), \int_A \omega > 0 \} > 0
$$

**$\hbar > 0$ 的存在性是紧致性的核心**：因为能量是整数系数的正量，冒泡只能发生有限次——「每冒一次泡至少花掉 $\hbar$ 的能量，总预算 $E(A)$ 有限，所以泡的数量有限」。<span class="marginnote">$\hbar > 0$ 为什么成立？因为 $A \in H_2(M;\mathbb{Z})$ 是离散的，$\int_A\omega > 0$ 是正整数倍的最小值。若 $H_2 = 0$ 则没有曲线、也没有冒泡——一切退化为平凡。冒泡理论是「能量离散性」的直接推论，这是 Gromov 紧致性比一般椭圆理论紧致性「免费获得」的地方。</span>

**紧致性的用途**：模空间 $\mathcal{M}_{g,k}$ 允许紧化（Gromov 紧化 $\bar{\mathcal{M}}$），紧化的边界是带节点的破裂曲线。**「数曲线」因此变成「数紧化边界上的对象」**——这是 Gromov-Witten 不变量与量子上同调的定义基础，也是下一节的直接预告。

**辨析｜易错点：** 紧致性说的是「能量有界的曲线族有收敛子列」，但收敛到**破裂曲线**而非单条曲线——极限可能分裂成多段。初学者以为「能量有界 ⇒ 单条曲线收敛」，错——**正确表述是「收敛到稳定破裂曲线」，多段 + 节点是常态**。稳定的概念（每段的自动同构群有限）保证模空间「不塌缩」。

## 6 小结

- **伪全纯曲线**：$du \circ j = J \circ du$，即非线性 Cauchy-Riemann 方程 $\bar\partial_J u = 0$；局部模型是全纯映射。
- **能量恒等式**：$E(u) = \int u^*\omega = \langle[\omega], [u]\rangle$——能量是拓扑量。
- **模空间**：正则 $J$ 下是光滑流形，维数由指标定理给出；「一般 $J$」正则。
- **Gromov 紧致性**：能量有界的曲线族收敛到稳定破裂曲线；**冒泡**是能量收缩成新段的现象。
- **最小能量阈值 $\hbar > 0$