---
title: 格林（Green）公式及其推论
date: 2026-08-08
---

# 格林（Green）公式及其推论

<div class="epigraph">
<p>散度定理把体积分与面积分连在一起，格林公式把它用于拉普拉斯算子。</p>
<footer>—— 乔治·格林（George Green）</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第三章 ｜ 2026-08-08</p>
</div>

## 为什么从格林公式开始

调和函数在边界上的信息如何决定内部的值？答案藏在**格林公式**里——它是散度定理（Gauss 定理）在拉普拉斯算子上的应用，是调和函数积分理论的总开关。平均值定理、积分表达式、基本解、格林函数，第六篇几乎所有定理都从格林公式出发。这一节先把格林第一、第二公式及其关键推论讲透——它们本身也极有用：调和的梯度的总通量为零、对称性、以及「边界值决定内部值」的第一条定量线索。

## 1 格林第一公式

设 $\Omega$ 是光滑有界区域，$u, v \in C^2(\bar\Omega)$，$\boldsymbol{n}$ 是边界 $\partial\Omega$ 的外法向。散度定理

$$
\int_\Omega \nabla\cdot\boldsymbol{F}\,dx = \oint_{\partial\Omega}\boldsymbol{F}\cdot\boldsymbol{n}\,dS
$$

取 $\boldsymbol{F} = v\,\nabla u$，则 $\nabla\cdot(v\nabla u) = \nabla v\cdot\nabla u + v\,\Delta u$，得**格林第一公式**

$$
\boxed{\;\int_\Omega \big(\nabla v\cdot\nabla u + v\,\Delta u\big)\,dx = \oint_{\partial\Omega} v\,\frac{\partial u}{\partial n}\,dS\;}
$$

其中 $\frac{\partial u}{\partial n} = \nabla u\cdot\boldsymbol{n}$ 是 $u$ 沿外法向的方向导数。<span class="marginnote">格林第一公式就是「分部积分在多重积分里的版本」：它把 $v$ 对 $\Delta u$ 的二阶导「转移」到 $\nabla v\cdot\nabla u$ 上，边界余项 $\oint v\,\partial u/\partial n$ 类比一维分部积分 $\int v u'' = -\int v'u' + [vu']$。凡遇到 $\int v\,\Delta u$，第一公式就是第一反应。</span>

**第一公式的灵魂：二阶导「还」成一阶导的积分 + 边界项。** 它是能量方法、极值原理、变分原理（第十篇）共同的运算基础。

## 2 格林第二公式（格林公式）

对调换 $u, v$ 再写第一公式，两式相减消去 $\int\nabla v\cdot\nabla u$，得到**格林第二公式（通常直接叫格林公式）**

$$
\boxed{\;\int_\Omega \big(v\,\Delta u - u\,\Delta v\big)\,dx = \oint_{\partial\Omega}\Big(v\,\frac{\partial u}{\partial n} - u\,\frac{\partial v}{\partial n}\Big)\,dS\;}
$$

它把「两个函数的拉普拉斯之差」与「两个函数在边界上的通量之差」联系起来。

**推论 1（调和函数的通量为零）**：若 $u$ 调和（$\Delta u = 0$），取 $v = 1$，则

$$
\oint_{\partial\Omega}\frac{\partial u}{\partial n}\,dS = 0
$$

**调和函数沿闭曲面（或闭曲线）的法向导数积分为零。** 物理含义：稳态温度场/静电场的总通量为零——流入量与流出量平衡，因为没有源。<span class="marginnote">这个推论也是 Neumann 问题可解性的第一线索：若边界上给定 $\partial u/\partial n = g$，则必须有 $\oint g\,dS = 0$——否则无解。这就是第六篇《Neumann 内问题有解的相容性条件》的源头。</span>

**推论 2（对称性）**：若 $u, v$ 都调和，则格林公式右端为零，即

$$
\oint_{\partial\Omega}\Big(v\,\frac{\partial u}{\partial n} - u\,\frac{\partial v}{\partial n}\Big)\,dS = 0
$$

这称为**互易定理（reciprocity）**：两个调和场在边界上「互换」各自的法向导数，积分不变。

## 3 公式解析：用格林公式看「边界决定内部」

格林公式最深刻的用途是：**用调和函数 $v$ 当「探针」，读出 $u$ 在边界上的信息。** 取 $v$ 为某个特殊函数（比如后面基本解 $1/|x-y|$），让 $v$ 在 $x=y$ 处有奇点，格林公式就能「提取」$u(y)$。看这个推理的骨架：

- **第一步，选取探针 $v$。** 若 $v$ 除 $x = y$ 外调和，则 $\Delta v = 0$（在去心区域），格林公式左端 $\int v\Delta u - u\Delta v = \int v\Delta u$。
- **第二步，处理奇点。** 用一个小球 $B_\epsilon(y)$ 挖掉奇点，格林公式作用在 $\Omega \setminus B_\epsilon(y)$ 上，得到一个含 $\oint_{\partial B_\epsilon}$ 的项。
- **第三步，让 $\epsilon \to 0$。** 奇点项在极限下「吐出」$u(y)$ 本身（乘上 $v$ 的奇性系数），边界项保留 $\oint_{\partial\Omega}$ 上的 $u$ 与 $\partial u/\partial n$。
- **第四步，结论。** $u(y) = \int_{\partial\Omega}\big(\text{核}\cdot u + \text{核}\cdot\partial u/\partial n\big)dS$——**调和函数的内部值由边界值 + 边界法向导数共同决定**。

这个推理在下一节《基本解》《调和函数的积分表达式》中具体实现。此刻先记住框架：**格林公式 + 奇点探针 = 边界信息提取器**。

## 4 格林公式在物理中的身份

格林公式在不同物理语境里有熟悉的名字：

| 数学形式 | 物理/工程名 | 含义 |
| --- | --- | --- |
| $\oint_{\partial\Omega}\boldsymbol{F}\cdot\boldsymbol{n}dS = \int_\Omega\nabla\cdot\boldsymbol{F}dx$ | Gauss 散度定理 | 通量 = 源的总和 |
| 格林第一公式 | 分部积分 / 能量恒等式 | 梯度配对 |
| 格林第二公式 | 互易定理（Betti 互易） | 两个场交换边界信息 |
| $\oint\frac{\partial u}{\partial n}dS = 0$ | 通量守恒 | 无源场总流入为零 |

**格林公式是「通量平衡」的数学语言。** 它把「内部发生什么」（体积分）与「边界流出什么」（面积分）连成等式——这正是守恒律的核心，也是后面泊松方程、Neumann 问题的一切论证支点。<span class="marginnote">互易定理在工程上有直接应用：结构力学中的 Betti 互易定理（A 点载荷在 B 点产生的位移 = B 点载荷在 A 点产生的位移）、电磁学中的洛伦兹互易定理，都是格林第二公式的不同化身。同一个数学结构，贯穿力学与电磁。</span>

## 5 小结

- 格林第一公式：$\int(v\Delta u + \nabla v\cdot\nabla u) = \oint v\,\partial u/\partial n$——分部积分的多重积分版。
- 格林第二公式：$\int(v\Delta u - u\Delta v) = \oint(v\,\partial u/\partial n - u\,\partial v/\partial n)$。
- 推论 1：调和函数的法向导数沿闭曲面总积分为零。
- 推论 2：互易定理——两个调和函数互换边界法向导数，积分不变。
- 格林公式 + 奇点探针 = 边界信息提取器，是积分表达式与格林函数方法的骨架。

在下一节，我们引入基本解——调和函数理论中的「点源探针」。
