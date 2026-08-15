---
title: 第一型曲面积分：定义与计算
date: 2026-08-07
---

# 第一型曲面积分：定义与计算

<div class="epigraph">
<p>一张质量不均匀的曲面，它的总质量是多少？沿曲面的「密度积分」——第一型曲面积分，是二重积分的「曲面版」。</p>
<footer>—— 黎曼（Bernhard Riemann），曲面积分理论（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§22.1 ｜ 2026-08-07</p>
</div>

## 为什么「对面积的曲面积分」先出场

第二十二章的两型曲面积分中，**第一型**（对面积的积分）是「标量场沿曲面累积」：曲面面密度 $\rho(x,y,z)$ 的总质量 $M=\iint_S\rho\,dS$。它与第一型曲线积分（§20.1）完全平行——曲线是「沿弧长」，曲面是「沿面积」。与方向无关，物理意义直观。

它也是第二型曲面积分（§22.2）、高斯公式（§22.3）的「温和前奏」——先把「沿曲面积分」这个新概念建立起来，再引入「定向」的复杂性。<span class="marginnote">第一型曲面积分 $\iint_S f\,dS$ 中 $dS$ 是曲面面积微元——「在曲面上取一小块面积 $dS$，采到 $f$ 的值」。物理直觉：面密度 $\rho$ 的曲面总质量 $=\iint_S\rho\,dS$、电荷面密度的总电荷同理。<strong>「$f$ 是每单位面积的量，积分就是总量」</strong>——与第一型曲线积分「密度 × 弧长」完全同构，只是弧长换成面积。这也是为什么它「先出场」：结构最简单、方向无关。</span>

## 1 定义

**第一型曲面积分**：设光滑曲面 $S$，$f(x,y,z)$ 在 $S$ 上有定义。把 $S$ 分割成小曲面片 $\Delta S_i$，取点 $(\xi_i,\eta_i,\zeta_i)\in\Delta S_i$，作和

$$\sum_{i=1}^{n}f(\xi_i,\eta_i,\zeta_i)\Delta S_i.$$

若当 $\max$（$\Delta S_i$ 的直径）$\to0$ 时（无论分割与取点）和式趋于同一数，则称该数为 $f$ 在 $S$ 上的**第一型曲面积分**：

$$\iint_S f(x,y,z)\,dS.$$

**与二重积分的关系**：第一型曲面积分是「沿曲面」的积分，二重积分是「沿投影区域」的积分——两者通过「曲面微元 $dS$ 与投影面积 $dx\,dy$ 的关系」换算。

> **辨析｜易错点：**第一型曲面积分的性质：**与方向无关**（$dS$ 是面积，无方向）；**线性与可加性**（曲面可分割）；**$f\equiv1$ 时 $\iint_SdS=$ 曲面面积**（§21.7 曲面面积公式的自洽检验）。另一个易错点：**$S$ 与它的参数化**——$S$ 可以分成多片（分段光滑曲面），积分按片相加。还有：**第一型曲面积分与「曲面在哪一侧」无关**——它是「标量沿曲面」，不需要定向。

## 2 计算公式

**定理：设光滑曲面 $S$ 由 $z=z(x,y)$（$(x,y)\in D$）给出，$f$ 在 $S$ 上连续，则**

$$\iint_S f(x,y,z)\,dS=\iint_D f(x,y,z(x,y))\sqrt{1+z_x^2+z_y^2}\,dx\,dy.$$

**公式解析：三步拆解**

**第一步，写曲面微元**。$dS=\sqrt{1+z_x^2+z_y^2}\,dx\,dy$（§21.7 曲面面积公式的微元版）——**曲面微元 = 投影面积 × 伸缩因子**；

**第二步，代入 $f$**。$f(x,y,z)$ 换成 $f(x,y,z(x,y))$——把曲面上的点坐标代入；

**第三步，二重积分**。$\iint_D f(x,y,z(x,y))\sqrt{1+z_x^2+z_y^2}dx\,dy$——**化为投影区域上的二重积分**。

**示范**：$f=x^2+y^2+z^2$ 沿单位球面 $x^2+y^2+z^2=1$。在球面上 $f\equiv1$，故 $\iint_Sf\,dS=\iint_SdS=$ 球面面积 $=4\pi$。**「$f$ 在曲面上为常数时，积分 = 常数 × 曲面面积」**。

**示范二（显式曲面）**：$f=xy$ 沿平面 $z=1$ 在 $D=[0,1]\times[0,1]$ 上的部分。$z_x=z_y=0$，$\sqrt{1+0+0}=1$：

$$\iint_Sxy\,dS=\iint_Dxy\,dx\,dy=\left(\int_0^1xdx\right)\left(\int_0^1ydy\right)=\frac14.$$

**平面（平坦）的曲面微元 $dS=dx\,dy$**——退化为二重积分。✓

## 3 参数曲面的计算

**定理（参数曲面）：设 $S$ 由参数方程 $\vec r(u,v)=(x(u,v),y(u,v),z(u,v))$ 给出（$(u,v)\in D$），则**

$$\iint_S f\,dS=\iint_D f(\vec r(u,v))\,|\vec r_u\times\vec r_v|\,du\,dv,$$

**其中 $\vec r_u=(\frac{\partial x}{\partial u},\frac{\partial y}{\partial u},\frac{\partial z}{\partial u})$、$\vec r_v=(\frac{\partial y}{\partial v},\cdots)$ 是切向量，$|\vec r_u\times\vec r_v|$ 是面积伸缩因子。**

**公式解析：为什么面积因子是「叉积的模」**

**第一步，切向量**。$\vec r_u,\vec r_v$ 是曲面在参数方向的切向量——它们张成切平面；

**第二步，平行四边形面积**。参数微元 $du\,dv$ 对应的曲面微元 ≈ 以 $\vec r_u du,\ \vec r_v dv$ 为边的平行四边形，面积 $=|\vec r_u\times\vec r_v|du\,dv$（叉积的模 = 平行四边形面积，§22 三维向量几何）；

**第三步，积分**。$dS=|\vec r_u\times\vec r_v|\,du\,dv$，代入 $f$ 后二重积分。

**示范**：球面参数化 $\vec r(\theta,\varphi)=(\sin\varphi\cos\theta,\sin\varphi\sin\theta,\cos\varphi)$（$0\le\theta\le2\pi,\ 0\le\varphi\le\pi$）。算 $|\vec r_\theta\times\vec r_\varphi|=\sin\varphi$，故

$$\iint_Sf\,dS=\int_0^{2\pi}\int_0^\pi f(\vec r(\theta,\varphi))\sin\varphi\,d\varphi\,d\theta.$$

**$f\equiv1$ 时**：$\iint_SdS=\int_0^{2\pi}\int_0^\pi\sin\varphi\,d\varphi\,d\theta=4\pi$——球面面积。**参数曲面的面积因子 $|\vec r_u\times\vec r_v|$ 是「参数网格的伸缩」**，与三重积分球坐标的 $\rho^2\sin\varphi$ 一脉相承。

> **辨析｜易错点：**第一型曲面积分与二重积分的区别在**「微元 $dS$ 与 $dx\,dy$ 的关系」**——$dS=\sqrt{1+z_x^2+z_y^2}dx\,dy$（显式）或 $|\vec r_u\times\vec r_v|du\,dv$（参数）。**忘记伸缩因子（直接用 $dx\,dy$）会把曲面「压平」**，与 §10.3 弧长、§21.7 曲面面积的教训一致。另一个易错点：**参数化的选择**——同一个曲面可以有不同参数化，面积因子随之不同，但积分值相同（伸缩因子补偿参数化差异）。

## 4 第一型曲面积分的应用

**应用一：曲面质量与质心**。面密度 $\rho(x,y,z)$ 的曲面质量 $M=\iint_S\rho\,dS$；质心

$$\bar x=\frac{\iint_Sx\rho\,dS}{\iint_S\rho\,dS},$$

类似 $\bar y,\bar z$。**「$\iint f\,dS$ 是「质量微元 $\rho\,dS$ 的加权」**——与 §21.7 质心的重积分版同构，只是积分域从区域换成曲面。

**应用二：转动惯量**。曲面绕 $z$ 轴的转动惯量 $I_z=\iint_S(x^2+y^2)\rho\,dS$——**「到轴距离平方 × 质量微元」**。

**应用三：静电场/热流的曲面积分**（预告 §22.2）——第一型曲面积分是「总通量」的标量版，第二型是它的定向版。<span class="marginnote">第一型曲面积分在物理里常以「总质量/总电荷」出现：薄壳的面密度积分、曲面的电荷分布。到第二十二场论初步，你会看到第一型（标量通量）与第二型（定向通量）如何切换——「$\iint_S\rho\,dS$ 是总质量，$\iint_S\vec F\cdot d\vec S$ 是总通量」——一个标量、一个向量，共享「沿曲面累积」的骨架。</span>

## 5 第一型曲面积分总览

| 曲面表示 | 面积因子 | 积分公式 |
| --- | --- | --- |
| 显式 $z=z(x,y)$ | $\sqrt{1+z_x^2+z_y^2}$ | $\iint_D f\sqrt{1+z_x^2+z_y^2}dx\,dy$ |
| 参数 $\vec r(u,v)$ | $\|\vec r_u\times\vec r_v\|$ | $\iint_D f\|\vec r_u\times\vec r_v\|du\,dv$ |
| 球面参数 | $\sin\varphi$ | $\iint f\sin\varphi\,d\varphi\,d\theta$ |

**核心是「面积因子」**——它把曲面微元 $dS$ 换成参数域/投影域的微元。三种表示只是面积因子的不同写法。

## 6 计算示范：第一型曲面积分的实战

**示范三（锥面上的积分）**：$f=x^2+y^2$ 沿锥面 $z=\sqrt{x^2+y^2}$ 在 $x^2+y^2\le1$ 的部分。$z_x=\frac{x}{\sqrt{x^2+y^2}},\ z_y=\frac{y}{\sqrt{x^2+y^2}}$，面积因子 $\sqrt{1+z_x^2+z_y^2}=\sqrt2$：

$$\iint_S(x^2+y^2)dS=\iint_{x^2+y^2\le1}r^2\cdot\sqrt2\,r\,dr\,d\theta=\sqrt2\int_0^{2\pi}\int_0^1r^3dr\,d\theta=\frac{\sqrt2\pi}{2}.$$

**锥面的面积因子恒为 $\sqrt2$**——与 §21.7 示范一（圆锥面积）一致，被积函数换成 $r^2$ 后极坐标顺算。

**示范四（柱面的面积）**：用参数化算圆柱面 $x=\cos\theta,\ y=\sin\theta,\ z=t$（$0\le\theta\le2\pi,\ 0\le t\le1$）的面积。$\vec r_\theta=(-\sin\theta,\cos\theta,0)$、$\vec r_t=(0,0,1)$，$|\vec r_\theta\times\vec r_t|=1$，面积 $=\int_0^{2\pi}\int_0^1\,dt\,d\theta=2\pi$——**柱面面积 $=$ 底圆周长 × 高**。<span class="marginnote">柱面的面积因子恒为 1（参数网格是「保面积」的），这让柱面上的积分格外干净：$\iint_S f\,dS=\int_0^{2\pi}\int_0^1 f(\cos\theta,\sin\theta,t)\,dt\,d\theta$——「沿柱面展开成矩形」后的二重积分。§22.2 第二型曲面积分里，柱面同样是首选例（高斯公式的经典验证面）。</span>

**示范五（平面的积分）**：$f=x$ 沿平面 $z=2x+3y$ 在 $D=[0,1]\times[0,1]$ 上。$z_x=2,\ z_y=3$，面积因子 $\sqrt{1+4+9}=\sqrt{14}$：

$$\iint_Sx\,dS=\int_0^1\int_0^1x\sqrt{14}\,dy\,dx=\sqrt{14}\cdot\frac12.$$

**平面的面积因子是常数 $\sqrt{1+a^2+b^2}$**——「倾斜平面」的伸缩因子与位置无关，这正是「平面面积 = 投影面积 / 方向余弦」的积分版。

## 7 小结

- **第一型曲面积分**：$\iint_S f\,dS$——标量场沿曲面累积；与方向无关。
- **计算**：$f$ 代入 + 面积因子（$\sqrt{1+z_x^2+z_y^2}$ 或 $\|\vec r_u\times\vec r_v\|$）→ 二重积分。
- **$f\equiv1$ 给面积**：$\iint_SdS=$ 曲面面积。
- **参数曲面**：面积因子 = 切向量叉积的模——「参数网格的伸缩」。
- **应用**：曲面质量、质心、转动惯量、通量的标量版。

在下一节，我们进入**第二型曲面积分**：曲面的侧、定义与计算。它携带方向（定向），对应「向量场穿出曲面的通量」，是高斯公式与斯托克斯公式的主角。
