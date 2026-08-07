---
title: 高斯公式：三重积分与曲面积分的联系
date: 2026-08-07
---

# 高斯公式：三重积分与曲面积分的联系

<div class="epigraph">
<p>穿出封闭曲面的总通量，等于曲面内部「源」的总量——高斯公式把面积分与体积分连接起来，是三维世界的格林公式。</p>
<footer>—— 卡尔·弗里德里希·高斯（Carl Friedrich Gauss），1813 年（节意）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§22.3 ｜ 2026-08-07</p>
</div>

## 为什么高斯公式是「场论第一定理」

格林公式（§21.3）说「平面边界线积分 = 内部面积分」。高斯公式把它升维到三维：

$$\oiint_S\vec F\cdot d\vec S=\iiint_\Omega\text{div}\,\vec F\,dV.$$

**封闭曲面的通量 = 内部散度的体积分。** 这里的 **散度（divergence）** $\text{div}\,\vec F=P_x+Q_y+R_z$ 是向量场的新概念——它度量「场的源头强度」。高斯公式是「广义斯托克斯公式」家族的三维成员，与格林、斯托克斯共同构成场论三定理。

物理上，高斯公式就是**高斯定律**：穿出封闭曲面的电通量 = 内部电荷（源）总量。它是流体力学、电磁学最基础的积分定律。<span class="marginnote">高斯公式的直觉是「源头计数」：一个封闭曲面里的「源」越多，穿出的「流」就越多。散度 $\text{div}\,\vec F$ 是「每一点产生流的强度」，体积分 $\iiint\text{div}$ 是「所有源的总量」，通量 $\oiint\vec F\cdot d\vec S$ 是「从曲面漏出去的量」——<strong>「漏出的 = 生成的」</strong>，物质守恒。静电场里 $\text{div}\,\vec E=\frac{\rho}{\varepsilon_0}$ 正是「电荷是电场的源」，高斯定律由此而来。第二级《电动力学》的麦克斯韦方程，第一条就是 $\nabla\cdot\vec E=\rho/\varepsilon_0$——高斯公式的微分形态。</span>

## 1 高斯公式

**定理（高斯公式 / Gauss's Divergence Theorem）：设 $\Omega$ 是有界闭区域，边界 $\partial\Omega$ 是分片光滑的封闭曲面，$\vec F=(P,Q,R)$ 在 $\Omega$ 上有连续偏导数，则**

$$\oiint_{\partial\Omega}P\,dy\,dz+Q\,dz\,dx+R\,dx\,dy=\iiint_\Omega\left(\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}\right)dV,$$

**即** $\oiint_{\partial\Omega}\vec F\cdot d\vec S=\iiint_\Omega\text{div}\,\vec F\,dV$，其中 $\partial\Omega$ 取**外侧**定向。

**公式解析：三步拆解**

**第一步，拆成三个分量**。只需证 $\oiint P\,dy\,dz=\iiint P_x$（另两项同理相加）。把 $\vec F$ 拆成 $(P,0,0)+(0,Q,0)+(0,0,R)$，每个分量单独用高斯公式再相加——**线性性**；

**第二步，先证 $R$ 分量**。设 $\Omega$ 是「$z$ 型」区域（$z$ 从 $z_1(x,y)$ 到 $z_2(x,y)$）。体积分：

$$\iiint_\Omega R_z\,dV=\iint_D\left[\int_{z_1}^{z_2}R_z\,dz\right]dx\,dy=\iint_D\left[R(x,y,z_2)-R(x,y,z_1)\right]dx\,dy;$$

**第三步，与通量对上**。$\oiint R\,dx\,dy$ 沿边界：上曲面（$z=z_2$，上侧）贡献 $\iint_D R(x,y,z_2)dx\,dy$，下曲面（$z=z_1$，下侧，定向朝下取负）贡献 $-\iint_DR(x,y,z_1)dx\,dy$，侧曲面（竖直）贡献 0（$dx\,dy=0$）。总和 = 上式的体积分。∎

**要点**：**证明的骨架 = 化归「$z$ 型区域 + 累次积分 + 边界曲面通量」**——与格林公式「$X$ 型区域 + 累次积分 + 边界线积分」完全同构，只是升一维。**「每个方向分量的积分 = 该方向偏导的体积分」**是拆解的核心。

## 2 散度

**散度（divergence）**：向量场 $\vec F=(P,Q,R)$ 的散度是标量场

$$\text{div}\,\vec F=\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}=\nabla\cdot\vec F.$$

**「$\nabla$ 与 $\vec F$ 的点积」**——nabla 算子 $\nabla=(\frac{\partial}{\partial x},\frac{\partial}{\partial y},\frac{\partial}{\partial z})$ 形式地作用在 $\vec F$ 上。

**散度的几何意义**：$\text{div}\,\vec F(P_0)$ 是「$\vec F$ 在 $P_0$ 处的源强度」——正散度 = 源（场向外发散），负散度 = 汇（场向内汇聚），零散度 = 无源（如不可压缩流体的速度场）。

**示范**：$\vec F=(x,y,z)$，$\text{div}\,\vec F=3$——处处源强度 3。$\vec F=(-y,x,0)$（旋转场），$\text{div}=0$——旋转不产生源。

**高斯公式的应用价值**：**把难算的曲面积分化成体积分**（或反过来）——由 $\vec F$ 的散度选择计算路径。

> **辨析｜易错点：**高斯公式的**条件**：$\Omega$ 是闭区域（含边界）、$\partial\Omega$ 是封闭曲面（不封闭不能直接用）、取外侧定向、$\vec F$ 有连续偏导数（内部有奇点时需「挖洞」）。**方向**：外侧通量为正，内侧则反号。另一个易错点：**「$\oiint$」是沿封闭曲面的积分记号**——不封闭的曲面（如半球面）需要「补个盖」凑成封闭区域再用高斯公式。

## 3 高斯公式的应用

**应用一：用体积分算通量**。$\displaystyle\oiint_S x^3\,dy\,dz+y^3\,dz\,dx+z^3\,dx\,dy$ 沿单位球面外侧。散度 $=3(x^2+y^2+z^2)$：

$$\oiint_S=\iiint_\Omega3(x^2+y^2+z^2)dV=3\int_0^{2\pi}\int_0^\pi\int_0^1\rho^2\cdot\rho^2\sin\varphi\,d\rho\,d\varphi\,d\theta=\frac{12\pi}{5}.$$

**「高斯公式把面积分变成体积分，体积分用球坐标秒算」**——这是高斯公式最实用的形态。

**应用二：高斯定律（物理）**。电场 $\vec E$ 穿出封闭曲面的通量 = 内部电荷总量除以 $\varepsilon_0$：$\oiint\vec E\cdot d\vec S=\frac{Q_{\text{内部}}}{\varepsilon_0}$。由高斯公式，等价于 $\text{div}\,\vec E=\frac{\rho}{\varepsilon_0}$（麦克斯韦第一方程）。**「积分形式 ↔ 微分形式」由高斯公式连接**——这是电磁学理论的核心技术。<span class="marginnote">「$\oiint\vec E\cdot d\vec S=\frac{Q}{\varepsilon_0}$」与「$\nabla\cdot\vec E=\frac{\rho}{\varepsilon_0}$」是高斯定律的积分形式与微分形式，由高斯公式互化。麦克斯韦方程组有四条，其中两条（电高斯、磁高斯）直接来自「通量 = 源」——即高斯公式的物理内容。第二级《电动力学》里，「对称性 + 高斯定律」是算电场的最快捷径（球对称、柱对称、平面对称），而这一切的数学根基就是本节的高斯公式。</span>

**应用三：用面积分算体积分**。若 $\text{div}\,\vec F$ 简单而 $\vec F$ 复杂，反向用高斯公式（面积分替代体积分）——**「两条路选好算的」**。

## 4 奇点与挖洞

若 $\vec F$ 在 $\Omega$ 内部有奇点（如 $\vec F=\frac{\vec r}{r^3}$ 在原点无定义），高斯公式不能直接用于含原点的区域。**挖洞法**：挖掉以奇点为心的小球 $B_\varepsilon$，在 $\Omega\setminus B_\varepsilon$ 上用高斯公式，再令 $\varepsilon\to0$。

**示范**：$\displaystyle\oiint_S\frac{\vec r}{r^3}\cdot d\vec S$ 沿包围原点的曲面 $S$。在 $\mathbb R^3\setminus\{0\}$ 上 $\text{div}\,\frac{\vec r}{r^3}=0$，故挖洞后体积分为 0，只剩小球边界：$\oiint_S=\oiint_{B_\varepsilon}=\frac1{\varepsilon^3}\cdot4\pi\varepsilon^2=\frac{4\pi}{\varepsilon}$？不——$\frac{\vec r}{r^3}$ 在 $r=\varepsilon$ 处，$\vec F\cdot\vec n=\frac{\varepsilon}{\varepsilon^3}\cdot1=\frac1{\varepsilon^2}$，通量 $=\frac1{\varepsilon^2}\cdot4\pi\varepsilon^2=4\pi$。**$\oiint_S=4\pi$ 与曲面形状无关**——「绕源的通量恒定」，这是「立体角」概念的雏形。<span class="marginnote">「$\oiint\frac{\vec r}{r^3}\cdot d\vec S=4\pi$（对包围原点的任意曲面）」是「源的通量不依赖曲面形状」的经典结果——它对应「点电荷的电场穿出任意包围它的曲面的电通量恒定」$\frac Q{\varepsilon_0}$。这个「绕源通量恒定」在第二级《电动力学》与《复变函数》里分别对应「高斯定律」与「留数定理」——「奇点贡献不依赖路径/曲面」是分析学与物理学的共同主题。</span>

## 5 高斯公式的地位

高斯公式是「场论三定理」的第二位成员：

| 定理 | 内容 | 维度 |
| --- | --- | --- |
| 格林公式 | $\oint_{\partial D}\vec F\cdot d\vec r=\iint_D\text{curl}\,\vec F$ | 2（平面） |
| 高斯公式 | $\oiint_{\partial\Omega}\vec F\cdot d\vec S=\iiint_\Omega\text{div}\,\vec F$ | 3（空间） |
| 斯托克斯公式 | $\oint_{\partial S}\vec F\cdot d\vec r=\iint_S\text{curl}\,\vec F\cdot d\vec S$ | 3（曲面） |

**三兄弟统一在「边界积分 = 内部导数积分」**——格林是平面版，高斯是「散度版」，斯托克斯是「旋度版」（§22.4）。

## 6 小结

- **高斯公式**：$\oiint_{\partial\Omega}\vec F\cdot d\vec S=\iiint_\Omega\text{div}\,\vec F\,dV$——封闭曲面通量 = 内部散度积分。
- **散度**：$\text{div}\,\vec F=P_x+Q_y+R_z=\nabla\cdot\vec F$——「源强度」。
- **证明**：拆分量 + 化归 $z$ 型区域 + 累次积分 + 边界曲面通量。
- **应用**：通量 ↔ 体积分互化、高斯定律（$\oiint\vec E\cdot d\vec S=\frac Q{\varepsilon_0}$）、对称性算场。
- **奇点处理**：挖洞法；绕源通量恒定（立体角雏形）。

在下一节，我们完成场论三定理的最后一位：**斯托克斯公式——曲面积分与曲线积分的联系**。旋度在曲面上的积分 = 边界曲线的环量。
