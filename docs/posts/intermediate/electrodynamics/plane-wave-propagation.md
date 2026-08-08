---
title: 平面电磁波的传播
date: 2026-08-07
---

# 平面电磁波的传播

<div class="epigraph">
<p>一束平面波，是电磁波最朴素的形象：电场与磁场横振，能量直线前进。</p>
<footer>—— 海因里希 · 赫兹（Heinrich Hertz）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第四章 §4.2 ｜ 2026-08-07</p>
</div>

## 为什么平面波是「一切的起点」

亥姆霍兹方程有无穷多解，但最基础、最常用的是**平面波**——波前是平面的解。它看似简单，却是理解一切电磁波的积木：球面波在远处就是平面波，任意波场可以分解成平面波（角谱方法），反射折射、波导、天线辐射的分析全都从平面波出发。吃透平面波的偏振、能量、色散，就拿到了整个电磁波理论的钥匙。<span class="marginnote">「平面波」是一种理想化：严格的平面波在空间无限延展、能量无限大。但真实中的远场——无线电波传到几百公里外、太阳光到达地球——波前曲率已可忽略，平面波是极好的近似。数学理想化 + 物理近似，这是电动力学一贯的做派。</span>

## 1 平面波解与偏振

**单色平面波**：设波沿 $z$ 方向传播，电场复振幅解为

$$\widetilde{\mathbf{E}} = \mathbf{E}_0 e^{ikz}$$

其中 $\mathbf{E}_0$ 是常矢量（振幅与偏振方向）。由 $\nabla\cdot\widetilde{\mathbf{E}} = 0$ 得 $ik\,\hat{\mathbf{z}}\cdot\mathbf{E}_0 = 0$——**电场垂直于传播方向**（横波）。磁场的解由法拉第定律的复形式给出：

$$\widetilde{\mathbf{B}} = \frac{\mathbf{k}\times\widetilde{\mathbf{E}}}{\omega} = \frac{1}{v}\hat{\mathbf{k}}\times\widetilde{\mathbf{E}}$$

磁场同样垂直于传播方向，且 $\mathbf{E}$、$\mathbf{B}$、$\mathbf{k}$ 三者互相垂直，构成右手系。**真空与线性介质中，电场与磁场同相位**，大小满足 $B = E/v = \sqrt{\mu\varepsilon}E$。<span class="marginnote">「电磁波是横波」不是假设，而是无源条件 $\nabla\cdot\mathbf{E} = 0$ 的必然推论：若有纵波分量，它必须满足 $\mathbf{k}\cdot\mathbf{E}_0 = 0$，即只能横振。相比之下，声波是纵波（介质密度扰动沿传播方向）——两类波的极化结构由场方程的类型决定。</span>

**偏振（polarization）**：电场矢量的取向随时间的变化。

- **线偏振**：$\mathbf{E}_0$ 固定方向，场沿一条直线振动。
- **圆偏振**：两个等幅、相位差 $90°$ 的线偏振叠加，电场矢量末端画圆。
- **椭圆偏振**：一般情形，两个正交线偏振分量振幅不等、相位差任意。

任意偏振都可分解为两个正交线偏振的叠加，或两个反向圆偏振的叠加——偏振是矢量叠加原理的直接体现。

## 2 相速度、波长与能量流

平面波的实场为 $\mathbf{E}(\mathbf{r},t) = \mathbf{E}_0\cos(\omega t - kz)$。定义关键量：

- **角频率** $\omega$（单位时间相位变化），**波数** $k = \omega/v$（单位长度相位变化）。
- **相速度** $v = \omega/k = 1/\sqrt{\mu\varepsilon}$：等相位面的移动速度。
- **波长** $\lambda = 2\pi/k$；**周期** $T = 2\pi/\omega$；$\lambda = vT$。

**能量与能流**：时间平均能量密度与坡印亭矢量为

$$\langle w \rangle = \frac{1}{2}\varepsilon E_0^2, \qquad \langle \mathbf{S} \rangle = \frac{1}{2}\sqrt{\frac{\varepsilon}{\mu}}\,E_0^2\,\hat{\mathbf{k}} = v\langle w\rangle\,\hat{\mathbf{k}}$$

——**能量以相速度沿传播方向流动**，光强 $I = \langle S \rangle = \frac{1}{2}\sqrt{\varepsilon/\mu}\,E_0^2$。<span class="marginnote">电场与磁场各贡献一半能量密度：$\langle w_E\rangle = \langle w_B\rangle = \frac{1}{4}\varepsilon E_0^2$。电磁波的能量在电场与磁场之间等分振荡——这是「电场磁场互锁」的能量学体现。光学里「光强 ∝ 振幅平方」就来自 $\langle w \rangle$。</span>

**真空中的数值**：真空中 $v = c = 3\times10^8\ \mathrm{m/s}$。按频率从低到高，电磁波谱依次为：无线电波、微波、红外、可见光、紫外、X 射线、γ 射线。可见光只是其中极窄的一段（约 $400$–$700\ \mathrm{nm}$）。

## 3 波包与群速度

严格单色波是理想化——真实信号总有有限带宽，是**波包（wave packet）**。波包由中心频率附近的若干平面波叠加而成。当介质**无色散**（$v$ 与频率无关）时，波包整体以相速度传播、形状不变；当介质**有色散**（$v = v(\omega)$）时，波包的包络以**群速度（group velocity）**传播：

$$v_g = \frac{\mathrm{d}\omega}{\mathrm{d}k}$$

而相速度 $v_p = \omega/k$。两者关系：$v_g = v_p + k\dfrac{\mathrm{d}v_p}{\mathrm{d}k}$。<span class="marginnote">信号携带的信息以群速度传播（不是相速度）——相速度描述单色波的相位面移动，群速度描述波包的能量与信息移动。在反常色散介质中 $v_p$ 可以超过光速，但 $v_g \le c$，信息速率不违反相对论。区分「相速度」与「群速度」是读懂色散文献的第一课。</span>

**波动方程的色散**：真空中 $\omega = ck$，无色散，$v_p = v_g = c$。介质中 $\varepsilon(\omega)$ 依赖频率，$\omega$ 与 $k$ 关系非线性，$v_p \neq v_g$。**群速度与色散关系直接相关**——$\omega(k)$ 曲线的斜率就是群速度。

## 4 公式解析：为什么 $\mathbf{B}$ 不是独立变量

平面波中磁场由电场唯一确定，这是整个平面波理论最核心的简化：

**第一步，从法拉第定律出发**：复形式 $\nabla\times\widetilde{\mathbf{E}} = i\omega\widetilde{\mathbf{B}}$。对平面波 $\widetilde{\mathbf{E}} = \mathbf{E}_0 e^{ikz}$，旋度 $\nabla\times\widetilde{\mathbf{E}} = ik\,\hat{\mathbf{z}}\times\widetilde{\mathbf{E}}$。代入得 $\widetilde{\mathbf{B}} = \dfrac{k}{\omega}\hat{\mathbf{z}}\times\widetilde{\mathbf{E}} = \dfrac{1}{v}\hat{\mathbf{k}}\times\widetilde{\mathbf{E}}$。
**第二步，读出横波结构**：叉乘 $\hat{\mathbf{k}}\times\widetilde{\mathbf{E}}$ 垂直于 $\hat{\mathbf{k}}$ 与 $\widetilde{\mathbf{E}}$——磁场自动垂直于传播方向与电场方向。**电磁波是三矢量互相垂直的右手系**：$\mathbf{E}$ 与 $\mathbf{B}$ 都横振，能量沿 $\mathbf{k}$ 流动。
**第三步，读出大小与相位**：$|\widetilde{\mathbf{B}}| = E/v$——磁场大小等于电场除以介质中的波速。真空或无损介质中 $\hat{\mathbf{k}}\times$ 不引入相位，所以 $\mathbf{E}$ 与 $\mathbf{B}$ **同相位**。若介质有损耗（$\varepsilon$ 复数），$v$ 变复数，$\mathbf{B}$ 相对 $\mathbf{E}$ 有相位差——这标志着场从「波」变为「衰减场」，是导体电磁波的入口（下一节）。<span class="marginnote">对照静电场：静电场中 $\mathbf{E}$ 与 $\mathbf{B}$ 互相独立（一个由电荷决定、一个由电流决定）；电磁波里两者被法拉第定律和位移电流锁成一体。从「静态解耦」到「波动耦合」，是电磁学从静到动的最深刻转变。</span>

**辨析｜易错点：** 平面波的电场方向（偏振）是**任意**的，只要垂直于传播方向。初学者常误以为 $\mathbf{E}$ 与 $\mathbf{B}$ 有固定方向——其实 $\mathbf{E}_0$ 可以指向垂直于 $\mathbf{k}$ 的任何方向，$\mathbf{B}$ 跟着自动确定。偏振自由度的存在，正是起偏器、偏振镜等光学器件能工作的前提。

## 5 平面波的电磁谱与日常

平面波理论统一解释了整个电磁谱：

| 波段 | 频率范围 | 产生/探测 | 典型应用 |
| --- | --- | --- | --- |
| 无线电波 | $3\ \mathrm{kHz}$–$300\ \mathrm{GHz}$ | 电子振荡/天线 | 广播、通信、雷达 |
| 微波 | $300\ \mathrm{MHz}$–$300\ \mathrm{GHz}$ | 微波管/半导体 | 微波炉、5G、卫星 |
| 红外 | $10^{12}$–$4\times10^{14}\ \mathrm{Hz}$ | 热辐射/热敏探测 | 遥感、夜视 |
| 可见光 | $4\times10^{14}$–$8\times10^{14}\ \mathrm{Hz}$ | 原子跃迁/光电 | 视觉、激光 |
| X 射线 | $10^{16}$–$10^{19}\ \mathrm{Hz}$ | 韧致辐射/闪烁体 | 医学影像、晶体分析 |

<span class="marginnote">整个电磁谱都是同一种物理——平面波在不同频率下的表现。频率越高，波长越短、能量光子越大、穿透/电离能力越强。从收音机到伽马刀，全部是「平面电磁波的传播」这一节内容在不同频段的应用。</span>

## 6 平面波的完整例题：圆偏振的分解

平面波理论里最训练直觉的操作是把复杂偏振分解成基本偏振。用一个例子把「分解与叠加」走通。

**问题**：一束沿 $z$ 传播的圆偏振波 $\widetilde{\mathbf{E}} = E_0(\hat{\mathbf{x}} + i\hat{\mathbf{y}})e^{ikz}$，把它分解为两个线偏振波的叠加。

**第一步，取实部看物理。** $\mathbf{E}(z,t) = \operatorname{Re}[E_0(\hat{\mathbf{x}} + i\hat{\mathbf{y}})e^{i(kz-\omega t)}]$。分别看两个分量：$E_x = E_0\cos(kz-\omega t)$，$E_y = E_0\cos(kz-\omega t - \pi/2) = E_0\sin(kz-\omega t)$。**$x$ 分量与 $y$ 分量幅度相等、相位差 90°**——这正是圆偏振的定义。

**第二步，看电场矢量的轨迹。** 在固定位置 $z$ 处，电场矢量 $\mathbf{E}(t) = E_0[\cos\omega t\,\hat{\mathbf{x}} + \sin\omega t\,\hat{\mathbf{y}}]$（已取 $kz = 0$）。随着时间推移，矢量末端画出一个**半径 $E_0$ 的圆**。若用 $e^{+i\omega t}$ 约定，则旋转方向相反——**左旋与右旋取决于约定与传播方向，物理上要小心定义**。

**第三步，反向分解**：任意椭圆偏振 = 两个正交线偏振（振幅不等、相位差任意）的叠加；或 = 左旋与右旋圆偏振的叠加。**分解基的选择是自由的**——这是矢量叠加原理的直接应用。

**为什么圆偏振重要**：圆偏振携带角动量（光子自旋），是光学操控、手性分子鉴别、以及量子信息（光子极化编码）的基础。磁光效应（法拉第旋转）、旋光介质都表现为「左右旋圆偏振相位不同」，最终体现为线偏振方向旋转。

**辨析｜易错点：** ① 「$\hat{\mathbf{x}} + i\hat{\mathbf{y}}$ 是圆偏振」里的 $i$ 表示 90° 相位差，**不是**「电场有虚部分量」——物理场总是实函数，复数只是运算工具。② 判断左旋/右旋必须约定「观察者面向光源」还是「顺着传播方向看」，两种约定旋向相反。③ 分解时两个分量的**相位关系**决定偏振形态：同相是线偏振，相位差 90° 是圆偏振，介于两者之间是椭圆偏振。

**电场与磁场的能量等分**：平面波中 $\langle w_E\rangle = \frac{1}{4}\varepsilon E_0^2$ 与 $\langle w_B\rangle = \frac{1}{4\mu}B_0^2$，而 $B_0 = E_0/v = \sqrt{\mu\varepsilon}E_0$，所以 $\langle w_B\rangle = \frac{1}{4\mu}\mu\varepsilon E_0^2 = \frac{1}{4}\varepsilon E_0^2 = \langle w_E\rangle$——**电磁波的能量在电场与磁场之间严格等分**。这个等分不是巧合，而是「$\mathbf{E}$ 与 $\mathbf{B}$ 互锁成波」在能量层面的体现。**在任何参考系、任何介质中都成立**，是电磁波能量的一个不变量特征。

**偏振与信息传输**：光纤通信里，光的偏振态是额外的信息维度——偏振复用（PDM）把两路正交偏振作为独立信道，让单根光纤的容量翻倍。**你在这节学的「线偏振分解为两正交分量、圆偏振是 90° 相位差的叠加」，正是偏振复用与偏振分集技术的数学基础**。偏振从「波动理论的习题」到「通信工程的饭碗」，只隔着一层工程实现。

**平面波的坐标自由**：选择沿 $z$ 传播只是坐标系取法。把 $e^{ikz}$ 换成一般的 $e^{i\mathbf{k}\cdot\mathbf{r}}$，所有结论自动推广到任意传播方向：$\widetilde{\mathbf{B}} = \dfrac{1}{\omega}\mathbf{k}\times\widetilde{\mathbf{E}}$、$\mathbf{k}\cdot\mathbf{E}_0 = 0$。**「取一个坐标轴对准传播方向」不是限制，而是自由度**——物理规律不依赖坐标取向，挑顺手的坐标系是解题的自由，不是近似。

## 7 小结

- **平面波** $\widetilde{\mathbf{E}} = \mathbf{E}_0 e^{ikz}$：横波（$\mathbf{k}\cdot\mathbf{E}_0 = 0$），$\mathbf{E},\mathbf{B},\mathbf{k}$ 构成右手系。
- $\mathbf{B} = \dfrac{1}{v}\hat{\mathbf{k}}\times\mathbf{E}$，无损介质中同相位，$B = E/v$。
- **偏振**：线、圆、椭圆三种，任意偏振可分解为两正交分量。
- **相速度** $v_p = \omega/k$，**群速度** $v_g = \mathrm{d}\omega/\mathrm{d}k$；信息以群速度传播。
- 能量密度与能流 $\langle\mathbf{S}\rangle = v\langle w\rangle\hat{\mathbf{k}}$，光强 $I = \frac{1}{2}\sqrt{\varepsilon/\mu}E_0^2$。

在下一节，平面波撞上两种介质的界面：透射、反射、全反射、布儒斯特角——**电磁波在界面的反射折射**。
