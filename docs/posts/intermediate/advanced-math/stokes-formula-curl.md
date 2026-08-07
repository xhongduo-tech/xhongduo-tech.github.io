---
title: 斯托克斯公式、环流量与旋度
date: 2026-08-07
---

# 斯托克斯公式、环流量与旋度

<div class="epigraph">
<p>曲面边界的环流量，由曲面内部的旋度决定——漩涡的数学现身了。</p>
<footer>—— 乔治 · 加布里埃尔 · 斯托克斯（George Gabriel Stokes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §11.7 ｜ 2026-08-07</p>
</div>

## 为什么从斯托克斯公式、环流量与旋度开始

三大公式的最后一块拼图：**斯托克斯公式**把「空间曲面边界的环流量」与「曲面内部的旋度」相连。高斯公式管「散度」（源），斯托克斯公式管「旋度」（涡）——两个算子构成了向量微积分的完整图景。斯托克斯公式是格林公式的三维推广（格林是平面情形，斯托克斯是曲面情形），也是麦克斯韦方程组中「变化的磁场产生电场」的数学表述。**旋度** $\mathrm{rot}\,\mathbf{F}$（curl）刻画场的「旋转强度」，是流体涡旋、电磁感应、天气预报中「涡度」的数学核心。<span class="marginnote">斯托克斯公式的物理直觉：<strong>「绕曲面边界转一圈的净环流，等于曲面内所有小漩涡强度的总和」</strong>。它把「大环流」（边界积分）分解成「小漩涡」（内部旋度积分）——就像把一整条河的旋涡运动拆成每个小水涡的贡献。麦克斯韦把「电场绕闭合回路」与「磁场变化率」用斯托克斯公式相连，得到电磁感应方程。</span>

## 1 斯托克斯公式

设 $\Sigma$ 是分片光滑的有向曲面，其边界是分段光滑闭曲线 $\Gamma$，$\mathbf{F} = P\mathbf{i}+Q\mathbf{j}+R\mathbf{k}$ 的各分量有连续偏导数，则

$$\oint_\Gamma P\,dx + Q\,dy + R\,dz = \iint_\Sigma\left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z}\right)dy\,dz + \left(\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x}\right)dz\,dx + \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\,dy$$

其中 $\Gamma$ 的**正向与 $\Sigma$ 的侧**符合右手定则（右手四指沿 $\Gamma$ 弯曲，拇指指向 $\Sigma$ 的法向）。向量形式：

$$\oint_\Gamma \mathbf{F}\cdot d\mathbf{s} = \iint_\Sigma \mathrm{rot}\,\mathbf{F}\cdot\mathbf{n}\,dS$$

**记忆**：被积函数的三个括号正是旋度 $\mathrm{rot}\,\mathbf{F}$ 的三个分量。<span class="marginnote">右手定则约定：<strong>「边界方向与曲面侧由右手定则绑定」</strong>——沿边界 $\Gamma$ 走，曲面在左手边（像格林公式的约定一样），则 $\Sigma$ 取上侧。若 $\Sigma$ 取相反侧，边界方向也要反转。方向约定是三大公式共有的「安全绳」，务必先定方向再套公式。</span>

## 2 旋度

**旋度（curl / rotation）**：向量场 $\mathbf{F} = P\mathbf{i}+Q\mathbf{j}+R\mathbf{k}$ 的旋度是向量场

$$\mathrm{rot}\,\mathbf{F} = \nabla\times\mathbf{F} = \begin{vmatrix}\mathbf{i} & \mathbf{j} & \mathbf{k}\\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z}\\ P & Q & R\end{vmatrix}$$

即 $\mathrm{rot}\,\mathbf{F} = \left(\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z},\ \frac{\partial P}{\partial z}-\frac{\partial R}{\partial x},\ \frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)$。

**旋度的物理意义**：$\mathrm{rot}\,\mathbf{F}$ 的方向是「旋转轴方向」，大小是「旋转强度」——衡量场在该点的「涡旋程度」。

- $\mathrm{rot}\,\mathbf{F} = \mathbf{0}$：**无旋场（保守场）**——路径积分与路径无关、存在势函数；
- $\mathrm{rot}\,\mathbf{F} \neq \mathbf{0}$：场有涡旋（如水中漩涡、绕导线的磁场）。<span class="marginnote">旋度的直觉：<strong>「放一个小风车在场里，它会不会转？」</strong>——旋度是「单位面积的环流量」：$\mathbf{n}\cdot\mathrm{rot}\,\mathbf{F} = \lim_{S\to0}\frac{\oint_{\partial S}\mathbf{F}\cdot d\mathbf{s}}{S}$。无旋场放风车不转（如重力场），有旋场放风车转（如漩涡）。旋度与散度一起构成「场」的两大局部特征——涡与源。</span>

## 3 公式解析：斯托克斯公式的应用

利用斯托克斯公式计算 $\oint_\Gamma (y^2 - z^2)\,dx + (z^2 - x^2)\,dy + (x^2 - y^2)\,dz$，其中 $\Gamma$ 是平面 $x+y+z=1$ 与三坐标面的交线，方向按右手定则：

- **第一步，算旋度**：$P = y^2-z^2$、$Q = z^2-x^2$、$R = x^2-y^2$。旋度三个分量：
  $$\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z} = -2y - 2z, \quad \frac{\partial P}{\partial z}-\frac{\partial R}{\partial x} = -2z - 2x, \quad \frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y} = -2x - 2y$$
- **第二步，选曲面**：$\Sigma$ 取平面 $x+y+z=1$ 在第一卦限的部分，法向量 $\mathbf{n} = \frac{(1,1,1)}{\sqrt3}$。
- **第三步，套公式**：$\oint_\Gamma \mathbf{F}\cdot d\mathbf{s} = \iint_\Sigma \mathrm{rot}\,\mathbf{F}\cdot\mathbf{n}\,dS$。计算 $\mathrm{rot}\,\mathbf{F}\cdot\mathbf{n} = \frac{-2(y+z)-2(z+x)-2(x+y)}{\sqrt3} = \frac{-4(x+y+z)}{\sqrt3} = \frac{-4}{\sqrt3}$（利用 $x+y+z=1$）。
- **第四步，面积分**：$\iint_\Sigma \frac{-4}{\sqrt3}\,dS = \frac{-4}{\sqrt3}\cdot S_\Sigma$。$\Sigma$ 是边长 $\sqrt2$ 的等边三角形，面积 $\frac{\sqrt3}{2}$，故结果 $= \frac{-4}{\sqrt3}\cdot\frac{\sqrt3}{2} = -2$。

**关键**：斯托克斯公式的应用流程——**算旋度 → 选合适曲面（让 $\mathrm{rot}\,\mathbf{F}\cdot\mathbf{n}$ 简单）→ 面积分**。选「让计算最简单」的曲面是斯托克斯公式的艺术——本题选平面三角形让旋度点积变成常数。

## 4 旋度与保守场

**平面场的旋度与格林公式的呼应**：平面场 $\mathbf{F} = (P,Q)$ 的旋度 $\mathrm{rot}\,\mathbf{F}$ 只有 $z$ 分量 $\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}$——这正是格林公式里的被积函数！所以**斯托克斯公式在平面情形退化为格林公式**（$\Sigma$ 是平面区域、$\Gamma$ 是其边界）。<span class="marginnote">三大公式的退化链条：<strong>斯托克斯（空间曲面）$\xrightarrow{\text{平面}}$ 格林（平面区域）</strong>。理解了斯托克斯，格林是它的平面特例。反过来，「旋度为零 ⟺ 保守场 ⟺ 有势函数」的空间版本也由此成立——空间曲线积分与路径无关的条件是 $\mathrm{rot}\,\mathbf{F}=\mathbf{0}$（在单连通区域）。</span>

**保守场三等价（空间版）**：在单连通空间区域内，以下等价：

- $\oint_\Gamma \mathbf{F}\cdot d\mathbf{s} = 0$ 对所有闭路 $\Gamma$ 成立；
- 曲线积分与路径无关；
- $\mathrm{rot}\,\mathbf{F} = \mathbf{0}$（无旋）；
- 存在势函数 $u$ 使 $\mathbf{F} = \nabla u$。

## 5 斯托克斯公式的应用

- **麦克斯韦方程组（电磁感应）**：$\oint_\Gamma \mathbf{E}\cdot d\mathbf{s} = -\frac{d}{dt}\iint_\Sigma \mathbf{B}\cdot d\mathbf{S}$——变化的磁场产生电场，正是斯托克斯公式的物理化身。<span class="marginnote">法拉第电磁感应定律的积分形式就是「电场沿闭合回路的环流量 = 磁通量变化率的负值」。斯托克斯公式把「环流量」翻译成「磁场的旋度」，得到微分形式 $\nabla\times\mathbf{E} = -\frac{\partial\mathbf{B}}{\partial t}$——<strong>麦克斯韦方程组的一半就靠斯托克斯公式建立</strong>。到《大学物理》与《电动力学》你会系统看到这套语言。</span>
- **流体力学**：涡度 $\boldsymbol{\omega} = \mathrm{rot}\,\mathbf{v}$ 刻画流体旋转，卡门涡街、大气涡旋都用旋度分析。
- **天气预报**：大气的涡度（cyclone 的旋转）是气象学的核心量。
- **曲线积分与路径无关判定**：空间保守场的判定直接用 $\mathrm{rot}\,\mathbf{F}=\mathbf{0}$。

## 6 小结

- **斯托克斯公式**：$\oint_\Gamma \mathbf{F}\cdot d\mathbf{s} = \iint_\Sigma \mathrm{rot}\,\mathbf{F}\cdot\mathbf{n}\,dS$，边界方向与曲面侧用右手定则约定。
- **旋度**：$\mathrm{rot}\,\mathbf{F} = \nabla\times\mathbf{F}$，刻画场的「涡旋强度与方向」。
- 旋度为零 ⟺ 保守场 ⟺ 路径无关 ⟺ 存在势函数（单连通区域）。
- 斯托克斯公式的平面情形退化为**格林公式**。
- 应用：麦克斯韦方程组、流体涡度、天气预报、保守场判定。

第十一章至此收束：从两类曲线积分到两类曲面积分，再到格林、高斯、斯托克斯三大公式，我们建立了向量场积分的完整理论。在下一章，我们将进入无穷级数的世界——**无穷级数**。
