---
title: 高斯公式、通量与散度
date: 2026-08-07
---

# 高斯公式、通量与散度

<div class="epigraph">
<p>穿出表面的总流量，等于内部源的总强度——这是三维的微积分基本定理。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §11.6 ｜ 2026-08-07</p>
</div>

## 为什么从高斯公式、通量与散度开始

格林公式是「二维边界积分 = 内部旋度积分」；三维的对应物是**高斯公式（散度定理）**：闭合曲面上场穿过表面的总通量，等于曲面所围立体内部「散度」的三重积分。它是三维的微积分基本定理，也是物理学的基石——高斯定律（电通量 = 内部电荷）、流体力学（净流出 = 内部源强）、热传导（热量平衡）全部建立于此。**散度** $\mathrm{div}\,\mathbf{F}$ 是场的「源强度」：每一点的散度告诉你「这点是源、汇还是无源」。<span class="marginnote">高斯公式与格林公式同族：<strong>「边界的积分 = 内部的积分」</strong>。一维是 $F(b)-F(a)=\int f'dx$，二维是格林（旋度），三维是高斯（散度）。它们都是「微积分基本定理」在不同维度的化身——这一族定理是整个积分学的顶峰。</span>

## 1 高斯公式

设空间闭区域 $\Omega$ 由分片光滑闭曲面 $\Sigma$ 围成，$\mathbf{F} = P\mathbf{i}+Q\mathbf{j}+R\mathbf{k}$ 在 $\Omega$ 上具有一阶连续偏导数，则

$$\iiint_\Omega\left(\frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}\right)dV = \oint\kern{-5pt}\iint_\Sigma P\,dy\,dz + Q\,dz\,dx + R\,dx\,dy$$

其中 $\Sigma$ 取**外侧**。向量形式：

$$\iiint_\Omega \mathrm{div}\,\mathbf{F}\,dV = \oint\kern{-5pt}\iint_\Sigma \mathbf{F}\cdot\mathbf{n}\,dS$$

**重点：外侧约定**——闭合曲面取外法向为正，这是「穿出为正」的物理约定。<span class="marginnote">高斯公式是「源与流」的守恒律：<strong>内部所有源（$\mathrm{div}\,\mathbf{F}$ 的总和）加起来，正好等于穿出表面的净通量</strong>。没有源（散度处处为零），穿出穿入抵消、净通量为零——流体的不可压缩无源流正是这样。这个「内部源强 = 表面通量」的思想在守恒律方程里是普适的。</span>

## 2 散度

**散度（divergence）**：向量场 $\mathbf{F} = P\mathbf{i}+Q\mathbf{j}+R\mathbf{k}$ 的散度是标量场

$$\mathrm{div}\,\mathbf{F} = \nabla\cdot\mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

用算子 $\nabla = \left(\frac{\partial}{\partial x},\frac{\partial}{\partial y},\frac{\partial}{\partial z}\right)$ 与 $\mathbf{F}$ 点积。**散度的物理意义**：$\mathrm{div}\,\mathbf{F}(M)$ 是 $M$ 点的「源强度」——单位体积的净流出率：

- $\mathrm{div}\,\mathbf{F} > 0$：该点是**源**（向外发散，如充电点的电场）；
- $\mathrm{div}\,\mathbf{F} < 0$：该点是**汇**（向内汇聚，如排水口的水流）；
- $\mathrm{div}\,\mathbf{F} = 0$：**无源场**（如不可压缩无源流体，穿出穿入平衡）。

**公式解析：散度的物理含义**

$$\mathrm{div}\,\mathbf{F} = \lim_{V\to 0}\frac{\oint\kern{-5pt}\iint_{\partial V}\mathbf{F}\cdot\mathbf{n}\,dS}{V}$$

- **第一步，看定义**：散度 = 「围绕 $M$ 的小闭曲面 $\partial V$ 的通量 ÷ 体积」，当体积趋于 0 的极限。
- **第二步，解读**：通量是「净穿出量」，除以体积是「单位体积的净流出率」，取极限是「该点的瞬时源强度」。
- **第三步，与高斯公式呼应**：高斯公式正是这个「局部定义」的积分版——把每点的源强度加起来等于总通量。

**关键**：散度是「局部量」（每一点的源强度），高斯公式是「整体守恒」（内部总源 = 表面总流）——**局部与整体的连接**，正是高斯公式最深的价值。

## 3 公式解析：用高斯公式算通量

求 $\mathbf{F} = (x, y, z)$ 穿过单位球面外侧的通量：

- **第一步，算散度**：$\mathrm{div}\,\mathbf{F} = 1 + 1 + 1 = 3$。
- **第二步，套高斯公式**：通量 $= \iiint_\Omega 3\,dV = 3V_\Omega$。
- **第三步，代球体积**：$V_\Omega = \frac{4}{3}\pi$，通量 $= 4\pi$。
- **第四步，直接验证**：球面上 $\mathbf{F}\cdot\mathbf{n} = 1$（$\mathbf{F}$ 沿径向、模为 1，$\mathbf{n}$ 也是径向单位向量），通量 $= 1\cdot S = 4\pi$，一致。

**关键**：高斯公式把「曲面通量」变成「体散度积分」——对 $\mathrm{div}\,\mathbf{F}$ 是常数的场，通量 = 常数 × 体积，一步到位。**先算散度，再乘体积**，比直接做曲面积分快得多。

## 4 高斯公式的应用

- **高斯定律（电学）**：$\oint\kern{-5pt}\iint_\Sigma \mathbf{E}\cdot d\mathbf{S} = \frac{Q_{\text{内}}}{\varepsilon_0}$——穿过闭合曲面的电通量等于内部电荷除以介电常数，是麦克斯韦方程组之一。<span class="marginnote">高斯定律用散度表述是 $\nabla\cdot\mathbf{E} = \frac{\rho}{\varepsilon_0}$——「电场的散度 = 电荷密度 ÷ 介电常数」。这是「电场线的源头是电荷」的精确数学：<strong>电荷是电场的源</strong>。到《大学物理》电磁学，高斯定律是求对称电荷分布电场的最强工具。</span>
- **流体力学**：连续性方程 $\frac{\partial\rho}{\partial t} + \mathrm{div}(\rho\mathbf{v}) = 0$——质量守恒的微分形式，靠散度表述。
- **热传导**：热量平衡方程含散度项，描述热量如何从高温区「发散」到低温区。
- **体积公式**：$\mathrm{div}\,\mathbf{F} = 3$ 的场（$\mathbf{F}=(x,y,z)$）给出 $V = \frac13\oint\kern{-5pt}\iint (x\,dy\,dz + y\,dz\,dx + z\,dx\,dy)$——用表面积分算体积。

## 5 三大公式的统一

格林、高斯、斯托克斯三大公式是同一家族的成员：

| 公式 | 维数 | 边界积分 | 内部积分 |
| --- | --- | --- | --- |
| 微积分基本定理 | 1 | 端点值 $F(b)-F(a)$ | $\int f'(x)dx$ |
| 格林公式 | 2 | 边界环流量 | $\iint(Q_x-P_y)dx\,dy$（旋度） |
| 高斯公式 | 3 | 曲面通量 | $\iiint \mathrm{div}\,\mathbf{F}\,dV$（散度） |
| 斯托克斯公式 | 3（曲面） | 边界曲线环流量 | $\iint \mathrm{rot}\,\mathbf{F}\cdot\mathbf{n}\,dS$（旋度） |

<span class="marginnote">它们都是「<strong>边界的积分 = 内部的微分积分</strong>」（广义斯托克斯定理 $\int_{\partial\Omega}\omega = \int_\Omega d\omega$）。高斯公式管「散度」（源），斯托克斯公式管「旋度」（涡）——两个算子 $\mathrm{div}$ 与 $\mathrm{rot}$（curl）是向量微积分的两大主角，贯穿电磁学与流体力学。下一节斯托克斯公式将完成这幅图景。</span>

## 6 小结

- **高斯公式**：$\oint\kern{-5pt}\iint_\Sigma \mathbf{F}\cdot\mathbf{n}\,dS = \iiint_\Omega \mathrm{div}\,\mathbf{F}\,dV$（$\Sigma$ 取外侧）。
- **散度**：$\mathrm{div}\,\mathbf{F} = \frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}$，是「每一点的源强度」。
- 散度 > 0 源、< 0 汇、= 0 无源；散度是局部量，高斯公式是整体守恒。
- 应用：高斯定律、连续性方程、热传导、用表面积分算体积。
- 三大公式 = 微积分基本定理家族：格林（2D 旋度）、高斯（3D 散度）、斯托克斯（3D 旋度）。

在下一节，我们将学习三大公式的最后一个——**斯托克斯公式、环流量与旋度**。
