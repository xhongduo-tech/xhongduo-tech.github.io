---
title: 电磁场的能量、动量与坡印廷矢量
date: 2026-08-07
---

# 电磁场的能量、动量与坡印廷矢量

<div class="epigraph">
<p>电磁场不仅携带能量，还让能量「流动」起来——坡印廷矢量告诉你能量流向何方、流速几何。</p>
<footer>—— 约翰 · 亨利 · 坡印廷（John Henry Poynting），1884</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从坡印廷矢量开始

第 55、66 节我们知道了电场与磁场的能量密度，但能量如何**流动**？一个电池供电的灯泡、一条传输电能的电缆、一束射向远方的电磁波——能量从源到负载的路径由**坡印廷矢量（Poynting vector）**描述。它是电磁场能量守恒定律的「流密度」，也是理解电磁波辐射、电路能量传输、天线功率的核心。这一节推导坡印廷矢量、能量密度与电磁动量。

## 1 电磁场能量密度

电磁场的能量密度（第 55、66 节）：

$$w = \frac{1}{2}\varepsilon E^2 + \frac{1}{2}\frac{B^2}{\mu} = \frac{1}{2}(\boldsymbol{E}\cdot\boldsymbol{D} + \boldsymbol{H}\cdot\boldsymbol{B})$$

**电场能量 + 磁场能量**：$w = \frac{1}{2}\varepsilon E^2 + \frac{B^2}{2\mu}$。电磁波中两者各占一半（第 70 节）。

## 2 坡印廷矢量

**坡印廷矢量（Poynting vector）**：单位时间通过单位面积（垂直能量流动方向）的电磁能量，即能流密度：

$$\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{H} = \frac{1}{\mu_0}\boldsymbol{E}\times\boldsymbol{B}$$

- 方向：能量流动方向（$\boldsymbol{E}\times\boldsymbol{H}$ 右手定则）；
- 大小：$S = EH$（$\boldsymbol{E}$、$\boldsymbol{H}$ 垂直时）——单位 W/m²；
- 对平面电磁波：$S = \varepsilon_0 cE^2$（瞬时），平均 $\bar{S} = \frac{1}{2}\varepsilon_0cE_0^2$（正比于振幅平方）。

**重点：坡印廷矢量 $\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{H}$ 给出电磁能量的流动方向与速率——能量沿 $\boldsymbol{E}\times\boldsymbol{H}$ 方向传播。** 它是电磁场能量守恒的「流密度」：能量不会凭空消失，只是从一个区域流到另一个区域。<span class="marginnote">「能量流动的直觉」：电磁波中 $\boldsymbol{E}$、$\boldsymbol{H}$ 互相垂直，$\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{H}$ 指向传播方向——能量随波跑。静电场+静磁场共存时（如磁铁旁的带电体），坡印廷矢量也可非零（能量在循环），但要结合边界条件整体理解。</span>

## 3 电磁能量守恒定律

由麦克斯韦方程组可推导**电磁能量守恒（坡印廷定理）**：

$$-\frac{\partial w}{\partial t} = \nabla\cdot\boldsymbol{S} + \boldsymbol{j}\cdot\boldsymbol{E}$$

- $\boldsymbol{j}\cdot\boldsymbol{E}$：电磁场对电荷做功的功率密度（转化为焦耳热/动能）；
- $-\partial w/\partial t$：单位体积能量减少率；
- $\nabla\cdot\boldsymbol{S}$：能量流出该体积的速率。

**积分形式**：区域能量减少率 = 流出通量 + 对电荷做功功率。**电磁能量守恒：场的能量减少 = 流出 + 转化给电荷。**

**重点：坡印廷定理是电磁场的能量守恒定律——$-\frac{\partial w}{\partial t} = \nabla\cdot\boldsymbol{S} + \boldsymbol{j}\cdot\boldsymbol{E}$。** 它说明电磁能量有三种去向：储存在场中（$\partial w/\partial t$）、流过边界（$\nabla\cdot\boldsymbol{S}$）、转化为电荷动能/热（$\boldsymbol{j}\cdot\boldsymbol{E}$）。<span class="marginnote">「电路能量的真实路径」：导线把能量输送给负载，能量真的「沿导线内部」流吗？不——能量在导线<strong>外部</strong>的电磁场中流动（坡印廷矢量方向从源指向负载），导线只是引导场的边界。这就是为什么同轴电缆/双绞线的能量在外层电磁场中传输。坡印廷矢量改变了「能量沿导线流」的直觉。</span>

## 4 公式解析：直流电路的能量流动

一根载流直导线（电流 $I$、电压降 $U$），分析坡印廷矢量的方向与功率。

$$
\oint\boldsymbol{S}\cdot\mathrm{d}\boldsymbol{A} = IU
$$

- **第一步，写场**：导线表面有切向电场 $E$（沿电流方向，来自电阻压降）与环向磁场 $B$（安培定律）。
- **第二步，算坡印廷矢量**：$\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{H}/\mu_0$——方向**垂直指向导线**（从场流入导线）。
- **第三步，对导线表面积分**：$\oint\boldsymbol{S}\cdot\mathrm{d}\boldsymbol{A} = IU$——流入导线的电磁能量等于电阻消耗的功率（焦耳热）。
- **第四步，解读**：能量不是「沿导线内部流」，而是「从周围的电磁场流入导线」——导线把电磁能转化为焦耳热。这个结果刷新了「电流带能量」的直觉：能量在电磁场中，导线只是耗散点。

**辨析｜易错点：**坡印廷矢量的方向是 $\boldsymbol{E}\times\boldsymbol{H}$，与传播方向一致——不要与 $\boldsymbol{H}\times\boldsymbol{E}$ 混淆（那会得到相反方向）。功率计算用 $\oint\boldsymbol{S}\cdot\mathrm{d}\boldsymbol{A}$（净流出）与能量守恒定理配合。对电磁波，时间平均 $\bar{S} = \frac{1}{2}\varepsilon_0cE_0^2$ 带 $\frac{1}{2}$（余弦平方平均）。

## 5 电磁动量与辐射压

电磁场还携带动量。动量密度（单位体积）：

$$\boldsymbol{g} = \mu_0\varepsilon_0\boldsymbol{S} = \frac{\boldsymbol{S}}{c^2} = \frac{\boldsymbol{E}\times\boldsymbol{H}}{c^2}$$

电磁波打到物体上，动量转移产生**辐射压（radiation pressure）**（第 70 节）：

$$p = \frac{I}{c} \quad \text{（完全吸收）}, \qquad p = \frac{2I}{c} \quad \text{（完全反射）}$$

**重点：电磁场携带动量——辐射压是电磁动量的宏观表现。** 光压虽小（正午阳光约 $4\times10^{-6}$ Pa），却是太阳帆推进、彗尾、光镊（激光操控微粒）的物理基础。<span class="marginnote">「电磁动量与光的粒子性」：辐射压可以从波的角度（电磁动量密度）解释，也可以从粒子角度（光子动量 $p = h\nu/c$）解释——两条路殊途同归（第 98 节康普顿效应）。电磁场携带能量与动量，进一步确认「场是物理实在」：它不只是数学描述，而是有能量、动量的真实存在。</span>

## 6 小结

- **能量密度**：$w = \frac{1}{2}\varepsilon E^2 + \frac{B^2}{2\mu}$。
- **坡印廷矢量**：$\boldsymbol{S} = \boldsymbol{E}\times\boldsymbol{H}$，方向 = 能量流动方向，大小 = 能流密度（W/m²）。
- **坡印廷定理**（能量守恒）：$-\frac{\partial w}{\partial t} = \nabla\cdot\boldsymbol{S} + \boldsymbol{j}\cdot\boldsymbol{E}$。
- 电路能量在导线外的电磁场中流动（不是沿导线内部）。
- **电磁动量密度**：$\boldsymbol{g} = \boldsymbol{S}/c^2$；辐射压 $p = I/c$（吸收）或 $2I/c$（反射）。

在下一节，我们研究电磁波的传播——**平面电磁波的传播及其在介质界面的反射与折射**。
