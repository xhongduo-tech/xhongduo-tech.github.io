---
title: 斯托克斯公式：曲面积分与曲线积分的联系
date: 2026-08-07
---

# 斯托克斯公式：曲面积分与曲线积分的联系

<div class="epigraph">
<p>曲面上的旋度总量，等于边界曲线的环量——斯托克斯公式把「面」与「边」连接，是格林公式在三维曲面上空中的展开。</p>
<footer>—— 乔治·加布里埃尔·斯托克斯（George Gabriel Stokes），1854 年（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§22.4 ｜ 2026-08-07</p>
</div>

## 为什么斯托克斯是「格林公式的空中版」

格林公式（§21.3）说：平面区域的边界环量 = 内部旋度面积分。斯托克斯公式把它从「平面区域」提升到「空间曲面」：

$$\oint_{\partial S}\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S.$$

**曲面边界曲线的环量 = 曲面上的旋度通量。** 格林公式是「曲面恰好躺在平面上」的斯托克斯特例——斯托克斯是格林公式在「弯曲曲面」上的推广。场论三定理至此集齐：格林（平面）、高斯（散度）、斯托克斯（旋度）。<span class="marginnote">斯托克斯公式的直觉：「旋涡的度量」。旋度 $\text{curl}\,\vec F$ 是「每点的小旋涡」，$\iint_S\text{curl}\cdot d\vec S$ 是「曲面上的旋涡总量」，$\oint\vec F\cdot d\vec r$ 是「沿边界看出的净环量」——<strong>边界环量 = 内部旋涡总和</strong>。格林公式的「环量 = 旋度面积分」（§21.3）在弯曲曲面上同样成立，只是曲面积分要按曲面法向投影——这就是斯托克斯公式的全部内容。安培定律 $\oint\vec B\cdot d\vec l=\mu_0I$ 正是「磁场的环量 = 穿过曲面的电流」——斯托克斯公式的物理形态。</span>

## 1 旋度

**旋度（curl）**：向量场 $\vec F=(P,Q,R)$ 的旋度是向量场

$$\text{curl}\,\vec F=\left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z},\ \frac{\partial P}{\partial z}-\frac{\partial R}{\partial x},\ \frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)=\nabla\times\vec F.$$

**「$\nabla$ 与 $\vec F$ 的叉积」**——旋度是「场的小旋转向量」，方向指向旋转轴、大小是旋转强度。

**二维退化**：若 $\vec F=(P(x,y),Q(x,y),0)$ 与 $z$ 无关，则 $\text{curl}\,\vec F=(0,0,Q_x-P_y)$——**第三个分量正是二维旋度 $Q_x-P_y$**（§21.3 格林公式的旋度）。斯托克斯公式在平面曲面 $z=0$ 上退化为格林公式。

> **辨析｜易错点：**旋度是**向量**、散度是**标量**——两者容易混。记忆：旋度 $\nabla\times\vec F$（叉积 → 向量）、散度 $\nabla\cdot\vec F$（点积 → 标量）。另一个易错点：**旋度的分量循环规律**：$\text{curl}$ 的第 $i$ 分量是「另两个坐标的偏导差」，按循环 $(x\to y\to z\to x)$ 记忆——$\text{curl}_z=Q_x-P_y$（$x,y$ 循环），$\text{curl}_x=R_y-Q_z$（$y,z$ 循环），$\text{curl}_y=P_z-R_x$（$z,x$ 循环）。

## 2 斯托克斯公式

**定理（斯托克斯公式 / Stokes' Theorem）：设 $S$ 是分片光滑的有向曲面，边界 $\partial S$ 是分段光滑的封闭曲线（定向与 $S$ 的法向符合右手定则），$\vec F=(P,Q,R)$ 在 $S$ 上有连续偏导数，则**

$$\oint_{\partial S}P\,dx+Q\,dy+R\,dz=\iint_S\left[\left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}\right)dy\,dz+\left(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}\right)dz\,dx+\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\,dy\right].$$

**即** $\oint_{\partial S}\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S$。

**公式解析：三步拆解**

**第一步，化归平面（格林公式）**。设 $S$ 是 $xy$ 平面上的区域 $D$（法向朝上）。此时 $\vec F\cdot d\vec r=Pdx+Qdy+Rdz$ 中 $dz=0$，斯托克斯公式退化为格林公式 $\oint(Pdx+Qdy)=\iint_D(Q_x-P_y)dx\,dy$——**格林公式就是「平的斯托克斯」**；

**第二步，参数化曲面**。一般曲面 $\vec r(u,v)$，把 $\oint\vec F\cdot d\vec r$ 用链式法则换成对 $u,v$ 的线积分，再用平面格林公式在参数域上转化；

**第三步，收集项**。把参数域上的二重积分收集成 $\iint_S\text{curl}\,\vec F\cdot d\vec S$（面积因子与投影分量合并）——**「边界环量 = 旋度通量」成立**。∎

**要点**：**证明的思路是「把弯曲曲面摊平」**——先用格林公式在参数平面（平的）上转化，再把结果「卷」回原曲面（通过 $d\vec S$）。这个「摊平再卷回」是斯托克斯公式证明的灵魂。

## 3 斯托克斯公式的应用

**应用一：把线积分化面积分**。$\displaystyle\oint_C y\,dx+z\,dy+x\,dz$ 沿圆周 $x^2+y^2+z^2=1,\ x+y+z=0$（单位球与平面的交线）。$\vec F=(y,z,x)$，$\text{curl}\,\vec F=(-1,-1,-1)$（算：$\text{curl}_x=R_y-Q_z=0-1=-1$ 等）。交线围成的圆盘面积 $=\pi r^2$，半径 $r=\sqrt{1-\frac13}=\sqrt{\frac23}$（球心到平面的距离 $\frac1{\sqrt3}$），面积 $\frac{2\pi}{3}$。斯托克斯：

$$\oint_C\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S=(-1,-1,-1)\cdot\vec n\cdot S.$$

平面 $x+y+z=0$ 的单位法向 $\vec n=\frac{(1,1,1)}{\sqrt3}$，$\text{curl}\cdot\vec n=-\sqrt3$，故积分 $=-\sqrt3\cdot\frac{2\pi}{3}=-\frac{2\sqrt3\pi}{3}$。**「旋度是常向量时，积分 = 常向量 · 法向 · 面积」**。

**应用二：面积分化线积分**。若 $\text{curl}\,\vec F$ 比 $\vec F$ 复杂，反向用斯托克斯（线积分替代面积分）。**「两条路选好算的」**是斯托克斯的核心应用原则。<span class="marginnote">「$\oint\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S$」是「线积分 ↔ 面积分」的自由通道：线积分难算时改成面积分（选旋度简单），面积分难算时改成线积分（选边界简单）。工程里「用斯托克斯定理简化」是场论计算的标准动作。安培定律 $\oint\vec B\cdot d\vec l=\mu_0\iint\vec J\cdot d\vec S$ 正是「磁场的环量 = 电流通量」——麦克斯韦方程组里「旋度方程」的积分形态，第二级《电动力学》的核心工具。</span>

**应用三：路径无关的三维判据**。斯托克斯公式给出三维路径无关条件：$\vec F$ 在单连通区域上，$\oint\vec F\cdot d\vec r=0$（一切闭曲线）⇔ $\text{curl}\,\vec F=0$——**「旋度为零」是三维保守场的判据**（§21.4 的升维版）。$\text{curl}\,\vec F=0$ 时存在势函数 $\varphi$ 使 $\vec F=\nabla\varphi$。

## 4 三个公式的统一

场论三定理与微积分基本定理构成完整的「广义斯托克斯」家族：

| 定理 | 公式 | 边界与内部 |
| --- | --- | --- |
| 微积分基本定理 | $f(b)-f(a)=\int_a^bf'$ | 端点 = 内部导数 |
| 格林公式 | $\oint\vec F\cdot d\vec r=\iint\text{curl}\,\vec F$ | 平面边界 = 内部旋度 |
| 斯托克斯公式 | $\oint\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S$ | 空间边界 = 曲面旋度 |
| 高斯公式 | $\oiint\vec F\cdot d\vec S=\iiint\text{div}\,\vec F$ | 曲面边界 = 内部散度 |

**「边界上的积分 = 内部导数的积分」是贯穿一切维度的统一原理**——格林、斯托克斯、高斯都是它的具体形态。<span class="marginnote">「广义斯托克斯公式」在第二级《微分几何》里被写成极简的一句：$\int_{\partial M}\omega=\int_M d\omega$——微分形式 $\omega$ 在边界上的积分 = 外微分 $d\omega$ 在内部上的积分。格林、斯托克斯、高斯是这句话在 1、2、3 维的具体展开。这个「边界算子与外微分对偶」的思想，是近代数学最深刻的统一之一，而你在本书最后一章亲历了它的全部三维内容。</span>

## 5 场论三定理的应用场景

| 问题类型 | 选用的公式 |
| --- | --- |
| 平面曲线环量 ↔ 平面区域 | 格林公式 |
| 空间曲线环量 ↔ 空间曲面 | 斯托克斯公式 |
| 封闭曲面通量 ↔ 空间体积 | 高斯公式 |
| 路径无关判据 | 旋度为零（斯托克斯）/ 散度相关 |

**「看维度和对象选公式」**——平面用格林、曲面用斯托克斯、立体用高斯。

## 6 计算示范：斯托克斯公式的完整演练

**示范一（直接验证公式成立）**：$\vec F=(y,x,0)$，取 $S$ 为 $xy$ 平面上的单位圆盘，$\partial S$ 是单位圆。

- **左侧（环量）**：参数化 $x=\cos t,\ y=\sin t,\ z=0$，$dx=-\sin t\,dt,\ dy=\cos t\,dt$：
  $$\oint_{\partial S}\vec F\cdot d\vec r=\int_0^{2\pi}\bigl(y\,dx+x\,dy\bigr)=\int_0^{2\pi}\bigl(-\sin^2t+\cos^2t\bigr)dt=\int_0^{2\pi}\cos 2t\,dt=0.$$
- **右侧（旋度通量）**：$\text{curl}\,\vec F=(0,0,Q_x-P_y)=(0,0,1-1)=(0,0,0)$，面积分 $=0$。

**两侧都为 0，公式成立**——一个简单到几乎平凡的验证，却确认了「环量 = 旋度通量」在两边的每一步计算都是自洽的。

**示范二（常向量旋度选平面）**：$\displaystyle\oint_C(y\,dz+z\,dx+x\,dy)$，$C$ 是三角形 $A(1,0,0),B(0,1,0),C(0,0,1)$ 的边界（逆 $z$ 正向看）。$\vec F=(x,y,z)$? 重写：$y\,dz+z\,dx+x\,dy$ 对应 $\vec F=(z,x,y)$。$\text{curl}\,\vec F=(y_z-x_y,\ z_x-y_z,\ x_y-z_x)=(1-1,\ 1-1,\ 1-1)=(0,0,0)$——**旋度为零，环量必为零**：$\oint=0$。这个结论不依赖路径形状，正是「旋度为零 ⇔ 保守场」的直接应用。

**示范三（真实算一遍）**：$C$ 为平面 $z=0$ 上 $x^2+y^2=4$ 的圆（逆时针），$\vec F=(-y,x,0)$。参数化 $x=2\cos t,\ y=2\sin t$：$dx=-2\sin t\,dt,\ dy=2\cos t\,dt$，$\oint(-y\,dx+x\,dy)=\int_0^{2\pi}\bigl(4\sin^2t+4\cos^2t\bigr)dt=8\pi$。右侧：$\text{curl}\,\vec F=(0,0,2)$，圆盘面积 $4\pi$，通量 $=2\cdot4\pi=8\pi$。**两侧 $8\pi$ 一致**——这是「环量 = 旋度 · 面积」在旋度恒定向量的标准形态。

> **辨析｜易错点：**用斯托克斯公式最常踩的三个坑：① **定向不一致**——边界曲线与曲面法向必须满足右手定则，方向取反结果差负号；② **曲面要选「好算」的**——同一闭曲线可张成无数曲面（§22.3），选旋度简单、法向简单的那个，不要拘泥于「题目给的曲面」；③ **边界要「完整、光滑分段」**——闭曲线必须整体包含，缺一段都不行。示范二还提醒：**旋度为零时环量必为零，无需算积分**，这是斯托克斯最省力的应用。

**示范一至三给出的完整方法**：写 $\vec F$ → 算 $\text{curl}\,\vec F$ → 选曲面与定向 → 算通量（或算环量）→ 核对两侧。

## 7 小结

- **旋度**：$\text{curl}\,\vec F=\nabla\times\vec F$——「小旋转向量」；二维退化 $Q_x-P_y$。
- **斯托克斯公式**：$\oint_{\partial S}\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S$——边界环量 = 旋度通量。
- **证明**：摊平（参数域格林公式）→ 卷回（$d\vec S$）——「平的斯托克斯 = 格林」。
- **应用**：线积分 ↔ 面积分互化、三维路径无关（旋度为零）、安培定律。
- **统一**：格林、斯托克斯、高斯 + 微积分基本定理 = 「边界 = 内部导数」家族。
- **实战流程**：写 $\vec F$ → 算 $\text{curl}\,\vec F$ → 选曲面与定向 → 算通量/环量 → 核对两侧。
- **三大坑**：定向（右手定则）、曲面选择（好算优先）、边界完整（分段光滑闭合）。
- **省力技巧**：$\text{curl}\,\vec F=0$ 时环量必为零，无需积分直接得结论。

在下一节，我们以**场论初步**收官全书：梯度场、散度场与旋度场——$\nabla$ 算子的三大产物，以及它们之间的恒等式与物理意义。那里的无旋场、无源场分类，正是本节「旋度为零 ⇔ 保守场」判据的系统化。
