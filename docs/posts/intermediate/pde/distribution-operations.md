---
title: 广义函数的极限、导数与乘子运算
date: 2026-08-07
---

# 广义函数的极限、导数与乘子运算

<div class="epigraph">
<p>在广义函数的世界里，每个对象都能求导——而且求导永远合法。</p>
<footer>—— 分布论的「免费微分」</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第八章 ｜ 2026-08-07</p>
</div>

## 为什么从运算开始

广义函数理论的威力在于它把「极限、求导、乘法」这些运算推广到一切广义函数上——**尤其是求导**：任何广义函数都可以无限次求导，而这在经典意义下根本不可能（如阶跃函数在跳跃点不可导）。这一节定义广义函数的收敛、导数与乘子运算。核心是「**把运算搬到测试函数上**」的转移原则：$T$ 的导数对 $\varphi$ 的作用，等于 $T$ 对 $-\varphi'$ 的作用。这一条转移原则，让 §δ 与阶跃的导数都有了严格且自然的定义。

## 1 广义函数的收敛

**广义函数列收敛**：$T_k \to T$（在 $\mathcal{D}'$ 中）指对每个测试函数 $\varphi$：

$$
\langle T_k, \varphi\rangle \to \langle T, \varphi\rangle
$$

**「逐测试函数收敛」是广义函数的收敛定义。**<span class="marginnote">这个收敛定义让「δ 是窄峰的极限」有了严格含义：$\delta_\varepsilon = \frac{1}{2\varepsilon}\mathbf{1}_{[-\varepsilon,\varepsilon]}$（宽 $2\varepsilon$、高 $1/(2\varepsilon)$ 的矩形峰），对任何连续 $\varphi$，$\langle\delta_\varepsilon,\varphi\rangle = \frac{1}{2\varepsilon}\int_{-\varepsilon}^{\varepsilon}\varphi \to \varphi(0)$——故 $\delta_\varepsilon \to \delta$。上一节「任何单位质量窄峰族都趋于 δ」在此严格化。</span>

**广义函数收敛的好处**：只要「逐测试函数」收敛，就能安全取极限、求导、积分——**极限与导数在广义函数意义下几乎总是可交换**。这是经典函数论里最头疼的问题（何时能在积分号下取极限？）在分布框架下的「免费」解决。

## 2 广义函数的导数

**广义导数的定义**（转移原则）：对 $T \in \mathcal{D}'$，定义 $T'$ 为

$$
\langle T', \varphi\rangle = -\langle T, \varphi'\rangle
$$

**为什么这么定义？** 若 $T$ 是光滑函数 $f$，经典分部积分（$\varphi$ 支集紧，边界项为零）：

$$
\int f'\varphi\,dx = -\int f\varphi'\,dx \quad\Longrightarrow\quad \langle f',\varphi\rangle = -\langle f,\varphi'\rangle
$$

**把分部积分公式「倒过来」当成定义**——经典意义下成立的公式，在广义意义下变成定义，从而对一切广义函数成立。

**推论（惊人的自由）**：**每个广义函数都无穷可导**。因为 $\varphi$ 无穷光滑，$\varphi^{(n)} \in \mathcal{D}$，所以 $T^{(n)}$ 由 $\langle T^{(n)},\varphi\rangle = (-1)^n\langle T,\varphi^{(n)}\rangle$ 良定义。**求导不再是「特权」而是「天赋」**——δ、阶跃、乃至最粗糙的分布，全都能求任意阶导。

## 3 公式解析：关键例子的导数

**例 1（阶跃的导数）**：$H' = \delta$

- **第一步，用定义。** $\langle H', \varphi\rangle = -\langle H, \varphi'\rangle = -\int_0^\infty\varphi'(x)dx$。
- **第二步，算积分。** $-\int_0^\infty\varphi'dx = -[\varphi(\infty) - \varphi(0)] = \varphi(0)$（$\varphi$ 紧支，$\varphi(\infty)=0$）。
- **第三步，认出 δ。** $\langle H',\varphi\rangle = \varphi(0) = \langle\delta,\varphi\rangle$，故 $H' = \delta$ ✓。

**例 2（δ 的导数）**：$\langle\delta',\varphi\rangle = -\langle\delta,\varphi'\rangle = -\varphi'(0)$——δ 的导数「读出测试函数在原点的导数值（带负号）」。

**例 3（跳跃函数的导数）**：$f$ 在 $x=0$ 有跳跃 $[f]$、其余处处可导，则

$$
f' = \{f'\}_{\text{经典}} + [f]\,\delta
$$

**「跳跃处的导数 = 跳跃量 × δ」**——经典导数与集中导数分离。这条公式是「间断解」进入 PDE 理论的入口：守恒律的弱解在间断处自动携带 δ 贡献。

## 4 乘子运算

**广义函数乘光滑函数**：对 $a \in C^\infty$，定义

$$
\langle aT, \varphi\rangle = \langle T, a\varphi\rangle
$$

因为 $a\varphi \in \mathcal{D}$（光滑 × 紧支 = 紧支），右端良定义。**光滑函数是广义函数的「合法乘子」**。

**乘积公式（对导数）**：$(aT)' = a'T + aT'$——莱布尼茨法则在广义意义下成立（用定义验证）。

**辨析｜易错点：** **两个广义函数的乘积一般无定义。** $T_1T_2$ 需要「$T_2\varphi$」是测试函数，但 $T_2$ 作用后不一定是紧支光滑函数——δ 与 δ 的乘积、δ 与阶跃的乘积都没有自然定义。（$f\delta$ 当 $f$ 连续时可定义，因为「$f$ 在 0 的值」乘 δ 有意义；但 $\delta^2$ 无意义。）**「乘光滑函数可以，乘任意分布不行」是分布理论的重要边界**——非线性 PDE 在分布框架下的困难正源于此。<span class="marginnote">这个「无乘积」的限制是分布理论对非线性 PDE 的局限：$\Delta u = u^2$ 这类方程在 $\mathcal{D}'$ 里没法逐点理解。现代非线性理论用「重整化」「Colombeau 广义函数」等工具绕过它——但入门阶段记住「分布乘法受限」就够用。这解释了为什么本专题第十篇的弱解理论（变分方法）避开乘法困难，用「能量积分」而非「逐点方程」。</span>

## 5 运算的相容性

广义函数运算与经典运算**相容**：当 $T$ 是经典可微函数时，广义导数 = 经典导数；当 $T_k$ 是经典函数且经典意义下收敛到 $T$（一致/局部 $L^1$），则广义收敛也成立。**广义函数理论不是「另一套数学」，而是「经典数学的合法延拓」**——经典结论全部保留，外加新的自由度。

| 运算 | 经典限制 | 广义推广 |
| --- | --- | --- |
| 求导 | 需可微 | 一切分布可导 |
| 极限 | 需一致/控制收敛 | 逐测试函数收敛 |
| 乘光滑函数 | 可 | 可 |
| 乘分布 | 通常不可 | 仍不可 |

**广义导数是 PDE 弱解理论的基石**：$u_t + u u_x = 0$ 的弱解（间断解）正是在广义意义下满足方程——间断处的 δ 贡献由「跳跃条件」吸收（Rankine–Hugoniot 条件）。第九篇的 Sobolev 空间（下下节）把这些广义导数放进可度量的框架。

## 6 数值算例：分段函数的广义导数

取 $f(x) = \begin{cases}0,& x<0\\ x,& x\ge 0\end{cases}$（斜坡函数）：

- **第一步，经典导数。** $x>0$ 处 $f' = 1$、$x<0$ 处 $f' = 0$。
- **第二步，跳跃量。** 在 $x=0$，$f$ 连续（$f(0)=0$），跳跃 $[f] = 0$。
- **第三步，广义导数。** $f' = H(x)$（阶跃）——没有 δ 项，因为 $f$ 连续（跳跃量为零）。
- **第四步，再求导。** $f'' = H'(x) = \delta$——**斜坡的二阶导是 δ**。物理上：斜坡函数的「拐点」在分布意义下是一个集中加速度。

再取 $g(x) = \begin{cases}0,& x<0\\ 1,& x\ge 0\end{cases}$：$g' = \delta$（跳跃 $[g]=1$）。**「跳跃量就是 δ 的系数」**——这个对应在守恒律（Rankine–Hugoniot 条件）里是标准操作。

## 7 高维广义导数与弱解

把广义导数推广到 $n$ 维：

- **多指标导数**：$\langle \partial^\alpha T, \varphi\rangle = (-1)^{|\alpha|}\langle T, \partial^\alpha\varphi\rangle$——一维转移原则的逐坐标推广。
- **梯度、散度、拉普拉斯**：$\langle \nabla T, \varphi\rangle = -\langle T, \nabla\varphi\rangle$（分量逐个定义），$\langle \Delta T, \varphi\rangle = \langle T, \Delta\varphi\rangle$。**拉普拉斯算子在广义意义下是自伴的**：它对测试函数的作用转成测试函数的拉普拉斯。
- **弱解定义**：$-\Delta u = f$ 的弱解指 $\int\nabla u\cdot\nabla\varphi = \int f\varphi$ 对所有测试函数成立——这是第十篇变分方法的出发点，也是 Sobolev 空间（下一节）里「解」的定义。

**辨析｜易错点：** 广义导数与经典导数「相容」的代价是「正则性信息丢失」——广义导数存在并不保证经典导数存在。例如 $|x|$ 的广义导数是符号函数 $\mathrm{sgn}(x)$（经典意义处处除 0 外可导），再求导是 $2\delta$。**「可求广义导」是比「可求经典导」弱得多的条件**——这正是弱解能容纳间断解的原因。<span class="marginnote">从「从极限到大模型」的视角，广义导数把「求导」从「光滑函数的特权」变成了「所有对象的免费权利」——正如分布把「点值」从「函数的特权」变成了「泛函的作用」。这种「把特权普适化」的哲学，在测度论（把面积普适化）、Lebesgue 积分（把积分普适化）里一以贯之：先定义大空间，再把原来的对象放进去。</span>

## 8 小结

- 广义函数收敛 = 逐测试函数收敛；窄峰族趋于 δ 得以严格化。
- 广义导数 $\langle T',\varphi\rangle = -\langle T,\varphi'\rangle$，由分部积分公式反转而来。
- 每个广义函数无穷可导：$H' = \delta$、$\langle\delta',\varphi\rangle = -\varphi'(0)$。
- 跳跃函数的导数 = 经典导数 + 跳跃量 × δ。
- 乘光滑函数合法，乘分布一般不合法——分布理论的乘法边界。

在下一节，我们研究广义函数的卷积与傅里叶变换。
