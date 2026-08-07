---
title: 三重积分：直角坐标、柱坐标、球坐标计算
date: 2026-08-07
---

# 三重积分：直角坐标、柱坐标、球坐标计算

<div class="epigraph">
<p>把二重积分再升一维——三重积分在空间区域上累积，而柱坐标与球坐标用雅可比因子把圆筒与球体「拉直」。</p>
<footer>—— 黎曼（Bernhard Riemann），多元积分理论（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§21.6 ｜ 2026-08-07</p>
</div>

## 为什么需要三种坐标系

三重积分 $\iiint_\Omega f(x,y,z)dx\,dy\,dz$ 是「空间区域的累积」——密度不均匀物体的质量、转动惯量、引力都靠它。与二重积分一样，关键是**选坐标系**：直角坐标适合长方体/直角棱柱，**柱坐标**适合圆柱/圆筒（极坐标 × 直线），**球坐标**适合球体/球壳。三个坐标系的雅可比因子 $1,\ r,\ r^2\sin\varphi$ 决定体积微元的伸缩。

三重积分的计算 = 二重积分的「升维」：先选坐标系、再写累次积分（三次定积分）、逐层计算。**「区域形状选坐标」与二重积分的「选次序」是同一智慧**。<span class="marginnote">三种坐标系的雅可比因子是「空间伸缩」的记忆表：直角坐标 $dV=dx\,dy\,dz$（均匀格子），柱坐标 $dV=r\,dr\,d\theta\,dz$（极坐标 × 高度，$r$ 是「离 $z$ 轴的距离」），球坐标 $dV=r^2\sin\varphi\,dr\,d\varphi\,d\theta$（$r^2$ 是「球面面积放大」、$\sin\varphi$ 是「纬度处的收缩」）。<strong>「离轴越远、格子越大」是柱坐标与球坐标的共同直觉</strong>——因子正是这个放大的度量。</span>

## 1 三重积分的定义与累次积分

**三重积分**：设 $\Omega$ 是空间有界闭区域，$f(x,y,z)$ 在 $\Omega$ 上连续。分割 $\Omega$ 成小区域 $\Delta V_i$，作黎曼和取极限：

$$\iiint_\Omega f(x,y,z)\,dV=\lim_{\|\Delta\|\to0}\sum f(\xi_i,\eta_i,\zeta_i)\Delta V_i.$$

**累次积分**（富比尼定理的升维）：$\Omega$ 是「$z$ 型」区域（$z$ 从上边界到下边界）：

$$\iiint_\Omega f\,dV=\iint_D\left(\int_{\varphi_1(x,y)}^{\varphi_2(x,y)}f(x,y,z)\,dz\right)dx\,dy.$$

先对 $z$ 积分（固定 $(x,y)$ 竖直柱），再对 $(x,y)$ 做二重积分——**「三重积分 = 内层一次定积分 + 外层一次二重积分」**。

**示范**：$f=z$ 在单位立方体 $[0,1]^3$：

$$\iiint z\,dV=\int_0^1\int_0^1\int_0^1z\,dz\,dy\,dx=\int_0^1\int_0^1\frac12dy\,dx=\frac12.$$

> **辨析｜易错点：**三重积分累次积分的**上下限层次**：最内层上下限可依赖全部外层变量、中间层可依赖最外层、最外层必须是常数。写错层次（内层依赖不该依赖的变量）是高频错误。**「从内到外：变量越早积分，上下限越自由」**是写累次积分的纪律。

## 2 柱坐标

**柱坐标（cylindrical coordinates）**：$(r,\theta,z)$，$x=r\cos\theta,\ y=r\sin\theta,\ z=z$。**雅可比**：

$$\frac{\partial(x,y,z)}{\partial(r,\theta,z)}=r.$$

**体积微元**：

$$dV=r\,dr\,d\theta\,dz.$$

**适用**：区域或被积函数关于 $z$ 轴对称（圆柱、圆锥、旋转体），或含 $x^2+y^2$。

**示范**：$\displaystyle\iiint_\Omega z\,dV$，$\Omega$ 是圆柱 $x^2+y^2\le R^2$、$0\le z\le H$。柱坐标 $0\le r\le R,\ 0\le\theta\le2\pi,\ 0\le z\le H$：

$$\iiint_\Omega z\,dV=\int_0^{2\pi}\int_0^R\int_0^Hz\cdot r\,dz\,dr\,d\theta=2\pi\cdot\frac{R^2}{2}\cdot\frac{H^2}{2}=\frac{\pi R^2H^2}{2}.$$

**公式解析：柱坐标三步**

**第一步，代换**。$x^2+y^2=r^2$、$dV=r\,dr\,d\theta\,dz$——**极坐标的 $r$ 因子 + 直线 $z$**；

**第二步，定区域**。$D'$ 是「$z$ 从下到上、$r$ 从内到外、$\theta$ 一圈」——$z$ 在最内层（先积）、$\theta$ 在最外层；

**第三步，逐层积分**。$z\to r\to\theta$ 三次定积分。

## 3 球坐标

**球坐标（spherical coordinates）**：$(\rho,\varphi,\theta)$，$x=\rho\sin\varphi\cos\theta,\ y=\rho\sin\varphi\sin\theta,\ z=\rho\cos\varphi$，其中 $\rho\ge0$（半径）、$0\le\varphi\le\pi$（极角，离 $z$ 轴）、$0\le\theta\le2\pi$（方位角）。**雅可比**：

$$\frac{\partial(x,y,z)}{\partial(\rho,\varphi,\theta)}=\rho^2\sin\varphi.$$

**体积微元**：

$$dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta.$$

**适用**：球体、球壳、含 $x^2+y^2+z^2$ 的被积函数。

**示范（球体体积）**：$\displaystyle\iiint_\Omega dV$，$\Omega$：$x^2+y^2+z^2\le R^2$。球坐标 $0\le\rho\le R,\ 0\le\varphi\le\pi,\ 0\le\theta\le2\pi$：

$$V=\int_0^{2\pi}\int_0^\pi\int_0^R\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta=2\pi\cdot2\cdot\frac{R^3}{3}=\frac43\pi R^3.$$

**球体体积 $\frac43\pi R^3$ 由三重积分精确导出**——§10.2 截面法的结果在此用球坐标「重算」一遍。<span class="marginnote">「$\iiint dV=\frac43\pi R^3$」是球坐标的第一次胜利。球坐标里 $\rho^2\sin\varphi$ 的 $\rho^2$ 是「球面面积在半径方向的放大」、$\sin\varphi$ 是「高纬度处的收缩」——正是「纬度越高、同样角度扫过的面积越小」的地理直觉。第二级《电动力学》里的点电荷场、电磁场的球谐函数分解，全在球坐标系里做——球坐标是「球对称问题」的母语。</span>

## 4 坐标系选择

**示范（含 $x^2+y^2+z^2$ 的球坐标积分）**：$\displaystyle\iiint_\Omega\sqrt{x^2+y^2+z^2}dV$，$\Omega$ 是单位球。球坐标 $\sqrt{x^2+y^2+z^2}=\rho$：

$$\int_0^{2\pi}\int_0^\pi\int_0^1\rho\cdot\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta=2\pi\cdot2\cdot\frac14=\pi.$$

**「球对称被积函数 × 球域 → 球坐标，$\sqrt{x^2+y^2+z^2}$ 变成 $\rho$」**——被积函数与区域同时简化。

**示范（含 $x^2+y^2$ 的柱坐标）**：$\displaystyle\iiint_\Omega(x^2+y^2)dV$，$\Omega$ 是圆柱 $x^2+y^2\le1$、$0\le z\le1$：

$$\int_0^{2\pi}\int_0^1\int_0^1r^2\cdot r\,dz\,dr\,d\theta=2\pi\cdot\frac14=\frac\pi2.$$

**坐标系选择总表**：

| 区域/被积函数 | 坐标系 | 雅可比 | 简化效果 |
| --- | --- | --- | --- |
| 长方体 | 直角坐标 | $1$ | 上下限常数 |
| 圆柱/圆筒 | 柱坐标 | $r$ | $x^2+y^2\to r^2$ |
| 球/球壳 | 球坐标 | $\rho^2\sin\varphi$ | $x^2+y^2+z^2\to\rho^2$ |
| 圆锥/旋转体 | 柱或球 | 视形状 | 边界变常数 |

> **辨析｜易错点：****柱坐标与球坐标的适用别混淆**：圆柱对称（绕 $z$ 轴）用柱坐标，球对称（球心）用球坐标。另一个易错点：**球坐标的 $\varphi$ 范围是 $[0,\pi]$、$\theta$ 范围是 $[0,2\pi]$**——极角 $\varphi$ 从北极到南极、方位角 $\theta$ 绕一圈，别与柱坐标的记号混。还有：**含 $x^2+y^2$（不含 $z^2$）用柱坐标、含 $x^2+y^2+z^2$ 用球坐标**——被积函数的结构决定坐标。

## 5 三重积分的地位

三重积分是重积分理论的「最高维」（本章内），它：

- **物理**：物体质量 $M=\iiint\rho\,dV$、质心、转动惯量、引力（§21.7）；
- **概率**：三维随机向量、联合分布的积分（第二级《概率论》）；
- **电磁学**：电荷分布的场（第二级《电动力学》）；
- **第二十二章**：曲面积分与高斯/斯托克斯公式的空间舞台。

**「选坐标系 + 写累次积分 + 逐层算」**是三重积分的全部方法论，与二重积分一脉相承。<span class="marginnote">三重积分在物理与工程里是「体积量的通用语言」：转动惯量 $I=\iiint r^2\rho\,dV$（$r$ 是到轴的距离）、引力势 $U=-G\iiint\frac{\rho}{r}dV$——§21.7 的重积分应用将系统展开。到第二级《电动力学》与《理论力学》，「对连续分布积分求总场」是基本动作，而今天的三重积分是那个动作的数学骨架。</span>

## 6 小结

- **三重积分**：$\iiint_\Omega f\,dV$——空间区域累积；累次积分 = 一次定积分 + 一次二重积分。
- **直角坐标**：$dV=dx\,dy\,dz$——长方体、直角棱柱。
- **柱坐标**：$dV=r\,dr\,d\theta\,dz$——圆柱对称、含 $x^2+y^2$。
- **球坐标**：$dV=\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta$——球对称、含 $x^2+y^2+z^2$。
- **选择**：区域与被积函数的结构决定坐标系；「先 $z$、再径向、再角度」的层次。

在下一节，我们完成重积分章节：**重积分的应用——曲面面积、质心、转动惯量与引力**。「切微元写积分」模板在多元世界的全面兑现。
