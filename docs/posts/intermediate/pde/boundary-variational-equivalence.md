---
title: 边值问题与变分问题的等价性
date: 2026-08-08
---

# 边值问题与变分问题的等价性

<div class="epigraph">
<p>解边值问题，就是在所有候选函数中找能量最低的那一个。</p>
<footer>—— Dirichlet 原理（Dirichlet's principle）</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 姜礼尚《数学物理方程讲义》第八章 ｜ 2026-08-08</p>
</div>

## 为什么从等价性开始

上一节看到拉普拉斯方程是 Dirichlet 能量的欧拉方程。这一节把这条「欧拉方程」的链路做成**双向等价定理**：边值问题（PDE + 边界）与变分问题（能量极小）在精确的意义上互为充要条件。这个等价性是 **Dirichlet 原理**的现代版本——它的意义非同小可：**把「解微分方程」换成「求能量极小」，为存在性证明与数值方法打开了全新通道**。这一节先讲清等价性的两个方向，再看 Poisson 方程与自然边界条件的对应。

## 1 变分问题的提法

考虑 Poisson 方程的 Dirichlet 问题：

$$
-\Delta u = f \ \text{在 } \Omega, \qquad u|_{\partial\Omega} = 0
$$

**对应的变分问题**：在 $H_0^1(\Omega)$（第九篇：边界为零的 Sobolev 空间）中极小化能量泛函

$$
J[v] = \frac{1}{2}\int_\Omega|\nabla v|^2dx - \int_\Omega f\,v\,dx
$$

第一项是「弯曲代价」（Dirichlet 能量），第二项是「外力的功」（$f$ 做功）。**求「总能量最低」的 $v$。**

<span class="marginnote">为什么要在 $H_0^1$ 而不是 $C^2$ 里求极小？因为 $H_0^1$ 是完备的（第九篇）——极小化序列的极限自动落在空间里，存在性才谈得上。这是「变分法比 PDE 直接法更适合证明存在性」的技术根源。经典 Dirichlet 原理的漏洞（Weierstrass 指出：$C^2$ 里极小未必存在）正是靠 Sobolev 空间补上的。</span>

## 2 方向一：极小 ⇒ 边值问题

设 $u \in H_0^1$ 是 $J$ 的极小点。对任意测试函数 $\varphi \in C_0^\infty(\Omega)$，取变分 $u + \varepsilon\varphi$，极小性要求

$$
0 = \frac{d}{d\varepsilon}J[u+\varepsilon\varphi]\Big|_{\varepsilon=0} = \int_\Omega\nabla u\cdot\nabla\varphi\,dx - \int_\Omega f\varphi\,dx
$$

对第一项分部积分（$\varphi$ 支集紧，边界项为零）：

$$
\int_\Omega(-\Delta u - f)\varphi\,dx = 0
$$

**由变分法基本引理**，$-\Delta u - f = 0$（几乎处处）——**极小点满足 Poisson 方程**。边界 $u|_{\partial\Omega}=0$ 已由 $u \in H_0^1$ 内置。**方向一证毕：极小 ⇒ PDE。**

## 3 方向二：边值问题解 ⇒ 极小

反方向：设 $u$ 是 Poisson 边值问题的解，证明 $J[u] \le J[v]$ 对所有 $v \in H_0^1$。令 $w = v - u \in H_0^1$：

$$
J[v] = J[u + w] = \frac{1}{2}\int|\nabla u + \nabla w|^2dx - \int f(u+w)dx
$$

- **第一步，展开。** $= J[u] + \int\nabla u\cdot\nabla w\,dx - \int fw\,dx + \frac{1}{2}\int|\nabla w|^2dx$。
- **第二步，用方程消交叉项。** 分部积分：$\int\nabla u\cdot\nabla w\,dx - \int fw\,dx = \int(-\Delta u - f)w\,dx = 0$（$u$ 满足方程）。
- **第三步，剩下非负项。** $J[v] = J[u] + \frac{1}{2}\int|\nabla w|^2dx \ge J[u]$。
- **第四步，结论。** $u$ 是极小点，且极值唯一（$J[v]=J[u]$ 时 $\int|\nabla w|^2=0$，$w=0$）。

**方向二证毕：PDE 解 ⇒ 能量极小。** 两条方向合起来：

$$
\boxed{\;-\Delta u = f,\ u|_{\partial\Omega}=0 \ \Longleftrightarrow\ u \text{ 是 } J \text{ 在 } H_0^1 \text{ 中的唯一极小}\;}
$$

**这就是 Dirichlet 原理：边值问题 = 变分问题。** 它同时给出存在性路径（找极小）与唯一性（能量严格凸，极小唯一）。

## 4 自然边界条件

变分问题还自动处理 Neumann 边界。考虑**无边界约束**的泛函

$$
J[v] = \frac{1}{2}\int_\Omega|\nabla v|^2dx - \int_\Omega f\,v\,dx - \oint_{\partial\Omega}g\,v\,dS
$$

对 $v$ 不加边界约束。一阶变分为零给出（分部积分后）

$$
\int_\Omega(-\Delta u - f)\varphi\,dx + \oint_{\partial\Omega}\Big(\frac{\partial u}{\partial n} - g\Big)\varphi\,dS = 0
$$

**辨析｜易错点：** 这里 $\varphi$ 在边界不必为零，边界项无法自动消失。两项独立为零（$\varphi$ 在内部与边界可独立变化）给出

$$
-\Delta u = f \ \text{在 } \Omega, \qquad \frac{\partial u}{\partial n} = g \ \text{在 } \partial\Omega
$$

**Neumann 边界「免费」地作为变分问题的自然边界条件（natural boundary condition）出现**——无需显式要求，能量极小自动给出。这正是上一节提到的「不加约束的端点 ⇒ 法向导数条件」。<span class="marginnote">对比：Dirichlet 边界（给 $u$ 值）叫<strong>本质边界条件（essential）</strong>，必须显式放进试探函数空间（$H_0^1$）；Neumann 边界（给 $\partial u/\partial n$）叫<strong>自然边界条件</strong>，自动从能量泛函中涌现。有限元方法里这个区分极其实用：本质条件要强加、自然条件不用管。</span>

## 5 等价性的意义

边值问题与变分问题的等价，是 PDE 理论的一个枢纽：

| 视角 | 边值问题 | 变分问题 |
| --- | --- | --- |
| 表述 | 逐点微分方程 | 整体能量极值 |
| 存在性 | 构造难 | 极小存在（完备空间） |
| 唯一性 | 极值原理 | 严格凸 |
| 数值 | 有限差分 | Ritz/Galerkin/有限元 |
| 稳定性 | 能量估计 | 能量范数 |

**「求 PDE 解」与「求能量极小」从此可以自由切换。** 存在性从「构造解」变成「证明泛函有极小」（直接法）；数值从「离散微分算子」变成「离散能量泛函」（后三节的方法）。**这一条等价定理，是整个变分方法大厦的地基。**<span class="marginnote">从本专题的脉络看，这条等价性把「椭圆型 = 平衡态」的直觉精确化：平衡态不仅是「方程的解」，更是「能量的驻点」。力学中「平衡 = 势能极小」的古老直觉，在 PDE 理论中获得了完整的现代形态——从「从极限到大模型」的主线看，这也是「优化」与「求解」两大范式的交汇点。</span>

## 6 小结

- Poisson 边值问题等价于在 $H_0^1$ 中极小化 $J[v] = \frac12\int|\nabla v|^2 - \int fv$。
- 方向一：极小 ⇒ 一阶变分为零 + 基本引理 ⇒ 方程。
- 方向二：方程解 ⇒ 交叉项消去 + 非负余项 ⇒ 极小（且唯一）。
- Neumann 边界是变分问题的自然边界条件，自动涌现。
- 等价性把存在性、唯一性、数值方法全部接入「能量」轨道。

在下一节，我们定义弱解（广义解）。
