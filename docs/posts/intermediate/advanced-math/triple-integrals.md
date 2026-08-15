---
title: 三重积分
date: 2026-08-07
---

# 三重积分

<div class="epigraph">
<p>三维空间的累加，是二重积分的自然升华。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §10.3 ｜ 2026-08-07</p>
</div>

## 为什么从三重积分开始

二重积分累加平面区域上的量，三重积分则累加**空间立体**上的量——立体质量（体密度）、总电荷、转动惯量。它的定义、性质与二重积分完全平行，只是「维数再升一级」：区域从平面 $D$ 变成空间 $\Omega$，面积微元 $d\sigma$ 变成体积微元 $dV = dx\,dy\,dz$。三重积分的计算同样「化三重为累次」，且引入了两个新坐标系——**柱坐标**与**球坐标**——它们是处理旋转对称与球对称问题的利器。三重积分是物理（密度、质心、转动惯量）与概率（三维联合分布）的必备工具。<span class="marginnote">三重积分的直观：<strong>$\iiint_\Omega f\,dV$ = 立体 $\Omega$ 上「密度 $f$ 的总量」</strong>。$f\equiv1$ 时 $\iiint_\Omega dV$ 是体积；$f=\rho$（密度）时是质量。从一维（线）到二维（面）到三维（体），「密度 × 微元」的模板一以贯之——元素法思想的最完整呈现。</span>

## 1 三重积分的定义与性质

设 $f(x,y,z)$ 在有界闭空间区域 $\Omega$ 上有界。将 $\Omega$ 任意分成 $n$ 个小闭区域 $\Delta V_i$（也表示体积），每块内任取 $(\xi_i,\eta_i,\zeta_i)$，作和 $\sum f(\xi_i,\eta_i,\zeta_i)\Delta V_i$。若 $\lambda\to0$ 时极限存在且与分割、取点无关，则称该极限为 $f$ 在 $\Omega$ 上的**三重积分**：

$$\iiint_\Omega f(x,y,z)\,dV = \lim_{\lambda\to0}\sum_{i=1}^{n} f(\xi_i,\eta_i,\zeta_i)\,\Delta V_i$$

直角坐标下 $dV = dx\,dy\,dz$。**性质**与二重积分完全平行：线性性、区域可加性、保序性、估值定理、中值定理；$\iiint_\Omega dV = V_\Omega$（体积）。<span class="marginnote">三重积分的一切都「比二重多一维」：区域 $\Omega$ 是三维立体、微元是体积 $dV$、几何意义是「四维量」（三维体积 × 密度）……但因为定义结构完全相同，你不必重新学一套理论，只需把「二维直觉」升维。这种「模式升维」正是学习更高维数学的通用策略。</span>

## 2 三重积分的累次积分

三重积分的累次积分是「先一维、再二维」或「先二维、再一维」的两次嵌套。以「先 $z$ 后 $(x,y)$」为例：设 $\Omega$ 可写成「$z$ 从 $z_1(x,y)$ 到 $z_2(x,y)$，投影区域为 $D$」：

$$\iiint_\Omega f\,dV = \iint_D\left[\int_{z_1(x,y)}^{z_2(x,y)} f(x,y,z)\,dz\right]dx\,dy$$

**三种次序**：按「先 $z$、先 $y$、先 $x$」三种内层选择，共有六种排列，选「区域边界简单、内层好积」的一种。<span class="marginnote">三重积分的累次积分有两种组织方式：<strong>「先竖柱、再平面」（先 $z$ 后 $x,y$）与「先平面、再整体」（先 $x,y$ 后 $z$）</strong>。前者适合「上下界函数简单」的立体，后者适合「截面已知」的立体（截面法）。画图 + 看边界是选择的关键。</span>

**投影法（先 $z$）**：$\Omega$ 的投影区域 $D$ + 上下曲面 $z_1(x,y), z_2(x,y)$——「切竖柱」。
**截面法（先 $x,y$）**：用平面 $z=c$ 截立体得截面 $D_z$，$\iiint_\Omega f\,dV = \int_a^b\left[\iint_{D_z} f\,dx\,dy\right]dz$——「切薄片再叠」。

## 3 柱坐标与球坐标

**柱坐标**（$r,\theta,z$）：平面用极坐标、$z$ 保持直角。变换 $x = r\cos\theta$、$y = r\sin\theta$、$z = z$，体积微元

$$dV = r\,dr\,d\theta\,dz$$

适用：区域或被积函数含 $x^2+y^2$、绕 $z$ 轴旋转的立体（圆柱、锥体、旋转抛物面）。

**球坐标**（$\rho,\varphi,\theta$）：$\rho$ 是到原点距离，$\varphi$ 是极角（与 $z$ 轴夹角），$\theta$ 是方位角。变换

$$x = \rho\sin\varphi\cos\theta, \qquad y = \rho\sin\varphi\sin\theta, \qquad z = \rho\cos\varphi$$

体积微元

$$dV = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$$

适用：球域、含 $x^2+y^2+z^2$ 的被积函数。<span class="marginnote">两个坐标系的记忆：<strong>柱坐标 = 极坐标 + 竖坐标</strong>（体积微元多乘 $r$）；<strong>球坐标微元 $\rho^2\sin\varphi$ = 两个「放大因子」相乘</strong>——$\rho^2$ 来自「到原点越远，同一角度差对应的表面积越大」，$\sin\varphi$ 来自「极点附近角度差对应的环带越窄」。到《多元积分》与《数学物理方程》，柱、球坐标是解 Laplace 方程的基础工具。</span>

**公式解析：球坐标微元 $\rho^2\sin\varphi$ 的来历**

$$dV = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$$

- **第一步，看微元形状**：$\rho,\varphi,\theta$ 的微元构成一个「弯曲长方体」。
- **第二步，量三条边**：径向长度 $d\rho$；沿 $\varphi$ 方向的弧长 $\rho\,d\varphi$（半径 $\rho$、圆心角 $d\varphi$）；沿 $\theta$ 方向的弧长 $\rho\sin\varphi\,d\theta$（到 $z$ 轴的距离 $\rho\sin\varphi$ 为半径、圆心角 $d\theta$）。
- **第三步，相乘**：体积 ≈ $d\rho \cdot \rho\,d\varphi \cdot \rho\sin\varphi\,d\theta = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$。

**关键**：三条边中两条是「弧长」（半径 × 角微元），所以出现 $\rho^2\sin\varphi$——这与柱坐标的 $r$、二重积分的 $r$ 同源：**曲线坐标下体积/面积微元 = 雅可比因子**。

## 4 公式解析：球坐标计算球体体积

计算半径 $R$ 的球体体积：

- **第一步，选坐标**：球域 + 体积（$f=1$）——球坐标最合适。$\Omega$：$0\le\rho\le R$、$0\le\varphi\le\pi$、$0\le\theta\le2\pi$。
- **第二步，写积分**：$V = \int_0^{2\pi}\int_0^\pi\int_0^R \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$。
- **第三步，逐层积**：$\int_0^R \rho^2d\rho = \frac{R^3}{3}$；$\int_0^\pi\sin\varphi\,d\varphi = 2$；$\int_0^{2\pi}d\theta = 2\pi$。
- **第四步，相乘**：$V = \frac{R^3}{3}\cdot 2\cdot 2\pi = \frac{4}{3}\pi R^3$——球体积公式被三重积分严格导出。

**关键**：球坐标下「三个独立变量」的积分完全分离（被积函数 $\rho^2\sin\varphi$ 是分离变量的），三重积分化为三个一元积分的乘积——「变量可分离时，多重积分 = 单积分乘积」。

## 5 三重积分的应用

- **质量与质心**：体密度 $\rho(x,y,z)$ 的立体质量 $M = \iiint_\Omega \rho\,dV$，质心 $\bar x = \frac{1}{M}\iiint x\rho\,dV$。<span class="marginnote">质心公式是「加权平均」的三维版：$\bar x = \frac{\int x\,dm}{\int dm}$——把每个位置 $x$ 按质量 $dm$ 加权。这个「加权平均 = 分子积分 ÷ 分母积分」的结构在概率论（期望）、物理学（质心、重心）与机器学习（加权损失）中处处出现。</span>
- **转动惯量**：$I_z = \iiint_\Omega (x^2+y^2)\rho\,dV$——质量 × 到轴距离平方的积分。
- **引力与静电力**：三维空间中的势与场强计算。
- **概率**：三维联合密度 $\iiint_\Omega p(x,y,z)\,dV$ 给出立体概率。

## 7 数值算例：柱坐标计算体积

用柱坐标计算由 $z = x^2 + y^2$ 与平面 $z = 4$ 围成的立体体积。

**第一步，选坐标**：区域含 $x^2+y^2$（抛物面 + 平面），用柱坐标最合适。
**第二步，写积分**：$V = \int_0^{2\pi}\int_0^2\int_{r^2}^{4} r\,dz\,dr\,d\theta$——$z$ 从抛物面 $r^2$ 到平面 $4$。
**第三步，逐层积**：内层 $\int_{r^2}^4 r\,dz = r(4 - r^2)$；中层 $\int_0^2 r(4-r^2)dr = \left[2r^2 - \frac{r^4}{4}\right]_0^2 = 8 - 4 = 4$；外层 $\int_0^{2\pi}4\,d\theta = 8\pi$。

**要点**：柱坐标把「抛物面 + 平面」的立体变成「$z$ 从 $r^2$ 到 $4$、$r$ 从 0 到 2」的简单积分——「含 $x^2+y^2$ 用柱坐标」的直觉在此兑现。<span class="marginnote">柱坐标的「分工」：<strong>$z$ 处理「高度方向」、$(r,\theta)$ 处理「旋转方向」</strong>——旋转对称的立体在柱坐标里边界变得极其简单。到《数学物理方程》解圆域 Laplace 方程时，柱坐标是分离变量的标准选择。</span>

## 8 对照表：三种坐标系

| 坐标系 | 微元 | 适用 |
| --- | --- | --- |
| 直角坐标 | $dx\,dy\,dz$ | 长方体区域 |
| 柱坐标 | $r\,dr\,d\theta\,dz$ | 绕 $z$ 轴旋转对称 |
| 球坐标 | $\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$ | 球对称 |

## 9 常见错误自查清单

| 错误 | 正确做法 |
| --- | --- |
| 忘乘雅可比因子 | 柱坐标乘 $r$、球坐标乘 $\rho^2\sin\varphi$ |
| 累次积分次序搞乱 | 内层先积，按边界嵌套 |
| 边界范围写错 | 先画图确定投影区域与上下界 |
| 球坐标 $\varphi$ 范围写错 | $0\le\varphi\le\pi$（北极到南极） |

## 10 三重积分与现代科学

三重积分是三维物理量的标准工具：

- **物理**：质量、质心、转动惯量、引力势——「密度 × 体积微元」；
- **概率**：三维联合分布 $\iiint p\,dV$；
- **电磁学**：电荷分布的总电荷 $\iiint\rho\,dV$；
- **有限元**：三维网格上的积分是数值仿真的基础。

「在三维立体上累加密度」的能力，是理解质量、能量、电荷等一切「总量」的数学前提。

## 11 小结

- **三重积分**：$\iiint_\Omega f\,dV$，性质与二重平行；$f\equiv1$ 时是体积。
- **累次积分**：投影法（先 $z$）+ 截面法（先 $x,y$），六种次序选「边界简单、内层好积」。
- **柱坐标**：$x=r\cos\theta$，$dV = r\,dr\,d\theta\,dz$——绕 $z$ 轴旋转对称。
- **球坐标**：$dV = \rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$——球对称。
- 柱坐标的 $r$ 与球坐标的 $\rho^2\sin\varphi$ 都是雅可比因子，别忘乘。
- 应用：质量、质心、转动惯量、引力、三维概率。

在下一节，我们将学习重积分的物理应用——**重积分的应用**。
