---
title: 从 Stokes 定理看 Green 公式、Gauss 公式与 Gauss-Bonnet
date: 2026-08-07
---

# 从 Stokes 定理看 Green 公式、Gauss 公式与 Gauss-Bonnet

<div class="epigraph">
<p>当一条定理的名字冠在四个人头上，你就知道它的内涵远比任何一个人想象的更深。</p>
<footer>—— 弗拉基米尔 · 阿诺尔德（Vladimir Arnold）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§7.8 ｜ 2026-08-07</p>
</div>

## 为什么从统一视角开始

上一节我们宣布了 Stokes 定理 $\int_M d\omega = \int_{\partial M}\omega$，并声称它统一了四大公式。这一节不满足于「声称」——我们把每个经典公式**显式地**从 Stokes 定理推导出来，并且更进一步，用 Stokes 定理重看 **Gauss-Bonnet**。

收获会是双重的：一方面，Green、Gauss、Stokes 三大经典公式不再是「三件要背的家具」，而是「同一件家具的三个抽屉」；另一方面，**Gauss-Bonnet 定理本身是 Stokes 定理的深度应用**——它把「局部曲率」与「整体拓扑」用微分形式的语言无缝连接。学完这一节，你对「统一」二字会有切肤的体会。<span class="marginnote">「统一」的价值：背三个公式的人，遇到新问题要判断「该用哪个」；懂一条定理的人，直接写 $\int_M d\omega = \int_{\partial M}\omega$。「理解的深度 = 记忆的广度」——Stokes 定理是微积分里「少即是多」的最佳例子。</span>

## 1 从 Stokes 到 Green 公式

**Green 公式**：平面区域 $D$ 上

$$
\iint_D \Big(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\Big)\,dx\,dy = \oint_{\partial D} P\,dx + Q\,dy
$$

从 Stokes 推导：

- **第一步，取形式**：$\omega = P\,dx + Q\,dy$（1-形式），$M = D$（2 维定向流形）。
- **第二步，算外微分**：$d\omega = dP\wedge dx + dQ\wedge dy$。其中 $dP = P_x dx + P_y dy$，外积后 $dx\wedge dx = 0$，得
  $$
  d\omega = P_y\,dy\wedge dx + Q_x\,dx\wedge dy = (Q_x - P_y)\,dx\wedge dy
  $$
- **第三步，套 Stokes**：$\iint_D (Q_x - P_y)\,dx\wedge dy = \oint_{\partial D} P\,dx + Q\,dy$——正是 Green 公式。

**重点：Green 公式 = $n=2$、$\omega$ 为 1-形式的 Stokes。** 旋度项 $Q_x - P_y$ 就是 $d\omega$ 的系数——「$d$ 自动给出旋度」。

## 2 从 Stokes 到 Gauss 散度公式

**Gauss 散度公式**：三维体 $V$ 上

$$
\iiint_V \nabla\cdot\mathbf{F}\,dV = \iint_{\partial V} \mathbf{F}\cdot d\mathbf{S}
$$

从 Stokes 推导：

- **第一步，取形式**：$\mathbf{F} = (F_1, F_2, F_3)$，取 2-形式
  $$
  \omega = F_1\,dy\wedge dz + F_2\,dz\wedge dx + F_3\,dx\wedge dy
  $$
- **第二步，算外微分**：
  $$
  d\omega = \Big(\frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z}\Big)\,dx\wedge dy\wedge dz = (\nabla\cdot\mathbf{F})\,dV
  $$
  外微分把「通量 2-形式」变成「散度 × 体积 3-形式」。
- **第三步，套 Stokes**：$\iiint_V (\nabla\cdot\mathbf{F})\,dV = \iint_{\partial V}\mathbf{F}\cdot d\mathbf{S}$——散度定理。

**重点：$\nabla\cdot$ 是「$d$ 作用在 2-形式上」的坐标表达。** 「散度」这个名字来自「$d$ 把通量变成体积源」——源强 = 散度。

## 3 公式解析：Green/Gauss/Stokes 三公式的同构

把三个经典公式并排，看出它们的「同构」：

| 公式 | 维数 | $\omega$ 阶数 | $d\omega$ | 边界 |
| --- | --- | --- | --- | --- |
| 牛顿-莱布尼茨 | 1 | 0 | $f'\,dx$ | 两点 |
| Green | 2 | 1 | $(Q_x - P_y)dx\wedge dy$ | 闭合曲线 |
| Gauss 散度 | 3 | 2 | $(\nabla\cdot\mathbf{F})dV$ | 闭合曲面 |
| 经典 Stokes | 3（曲面） | 1 | 旋度 × 面积 | 边界曲线 |

**模式**：$\omega$ 是「$k$-形式」，$d\omega$ 是「$(k+1)$-形式的密度」，等式把「内部密度积分」与「边界 $k$-形式积分」连起来。**维数、阶数不同，结构全同。**<span class="marginnote">经典 Stokes 公式（$\iint_S \nabla\times\mathbf{F}\cdot d\mathbf{S} = \oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}$）是「$n=3$ 流形、$\omega$ 为 1-形式、但积分在 2 维曲面 $S$ 上」——注意它把 Stokes 用在 $S$（2 维）而非整个 $\mathbb{R}^3$ 上，$\omega = F_1dx+F_2dy+F_3dz$ 限制到 $S$，$d\omega$ 的切向分量即旋度。三维空间有两个不同的「经典 Stokes」，容易混淆——都统一于 $\int_M d\omega = \int_{\partial M}\omega$。</span>

## 4 从 Stokes 看 Gauss-Bonnet

Gauss-Bonnet 定理 $\iint_S K\,dA = 2\pi\chi(S)$ 如何与 Stokes 联系？

**微分形式版本**：在高斯映射下，曲率形式 $K\,dA$ 是「球面像的面积形式」的拉回：

$$
K\,dA = N^*(\text{面积形式})
$$

**局部的关键事实**：存在 1-形式（联络形式 / 旋转形式）$\omega$ 使

$$
K\,dA = d\omega \qquad \text{（局部）}
$$

于是 Stokes 定理给出「局部」：

$$
\iint_R K\,dA = \int_R d\omega = \int_{\partial R}\omega
$$

——这正是 Gauss-Bonnet 局部形式的微分形式版本：**区域的曲率积分 = 边界上 1-形式的积分**（测地曲率与转角的信息藏在 $\omega$ 里）。<span class="marginnote">$\omega$ 是「旋转联络形式」（connection 1-form）：它编码「切向量沿曲线平行移动时的转角」。$\int_{\partial R}\omega$ = 平行移动的和乐角 = 测地曲率积分 + 外角——与第四篇的「和乐角 = 曲率积分」精确对应。Gauss-Bonnet 局部形式 $= \int_R d\omega = \int_{\partial R}\omega$，就是 Stokes 定理。</span>

**重点：Gauss-Bonnet 局部形式是 Stokes 定理的特例。** 把「旋转联络 1-形式」$\omega$ 代入 $\int d\omega = \int_\partial \omega$，左边是曲率积分、右边是边界转角——几何与分析的统一。<span class="marginnote">整体 Gauss-Bonnet 需要额外一步：$\omega$ 只在局部是「精确的」（$K\,dA = d\omega$），整体上 $K\,dA$ 不精确（$\int K\,dA = 2\pi\chi \neq 0$）。「闭而不精确」的部分正是欧拉类——它由拓扑（$\chi$）决定。这是微分形式理论「闭形式度量拓扑」的经典演示。</span>

## 5 统一的遗产

从 Stokes 看经典公式，留下的不只是「少背三章」：

**定理的「家庭」**：牛顿-莱布尼茨、Green、Gauss、Stokes、Gauss-Bonnet 是同一个思想的家族——「边界与内部、微分与积分的对偶」。
**现代延伸**：广义 Stokes（带奇异）、纤维丛上的积分、超对称量子场论里的 Stokes 型恒等式——全是这条思想的升级。
**物理直觉**：「通量 = 源」「环量 = 旋涡」这些物理图像，本质都是「边界 = 内部的 $d$」的不同投影。

**重点：Stokes 定理是「积分几何」的宪法。** 它把「求导」（$d$）与「边界」（$\partial$）钉在一起，让从微积分到相对论的一切理论共享同一个「守恒骨架」：**内部变化的总量，等于边界上的进出量。**<span class="marginnote">这条「守恒骨架」在物理里无处不在：质量守恒 $\int_V \partial_t\rho = \int_{\partial V}$ 流量、电荷守恒、能量守恒——全是「内部变化 = 边界通量」的 Stokes 形式。「守恒律 = Stokes 定理」是连续介质力学与场论的统一视角。</span>

## 6 小结

- **Green** = 2 维、1-形式的 Stokes；**Gauss 散度** = 3 维、2-形式的 Stokes。
- 经典 Stokes（旋度）= 曲面上的 1-形式 Stokes。
- 统一模式：$\int_M d\omega = \int_{\partial M}\omega$——维数/阶数不同、结构全同。
- **Gauss-Bonnet 局部形式 = Stokes 定理**：$K\,dA = d\omega$（联络形式），曲率积分 = 边界转角。
- 整体 Gauss-Bonnet：$K\,dA$ 闭而不精确，非精确部分 = 欧拉类 = 拓扑。

在下一节，我们进入第八篇：**黎曼几何初步**——把曲面内蕴几何推广到任意维流形。
