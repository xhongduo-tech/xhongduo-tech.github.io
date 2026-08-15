---
title: 静电场与恒定电场
date: 2026-08-07
---

# 静电场与恒定电场

<div class="epigraph">
<p>我渐渐发现，法拉第的思维方法也是数学的，尽管他没有用常规的数学符号来表达。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell）</footer>
</div>

<div class="article-byline">
<p>第六级 · 电磁场与电磁波（工程电磁场） ｜ David K. Cheng《Field and Wave Electromagnetics》第2版 §3-1～§3-7、§5-1～§5-5 ｜ 2026-08-07</p>
</div>

## 为什么从静电场开始

静电复印机的鼓面上吸附墨粉、高压输电线下方感应出电荷、手机触摸屏靠电场识别你的手指——这些都是**静电场（electrostatic field）**的日常版本：电荷静止，场不随时间变。<span class="marginnote">"静止"不是说电荷少，而是说宏观上电荷不宏观移动。静电场是第 1 篇梯度、散度、旋度的第一次实战：它将展示"无旋场可以写成势函数"这一整套方法，后面恒定磁场只是换了源。</span>本专题对标 David K. Cheng《Field and Wave Electromagnetics》，本篇连同"恒定电场"一起讲：两者共享同一套场论工具，又各自引出一个工程上天天用到的量——**电容**与**电阻**。

## 1 库仑定律与电场强度

1785 年库仑用扭秤测出两点电荷间的力：**两点电荷之间的作用力与电荷量的乘积成正比，与距离的平方成反比，方向沿连线**。写成矢量式，点电荷 $q_1$ 对 $q_2$ 的力

$$\mathbf{F}_{12} = \frac{q_1 q_2}{4\pi\varepsilon_0 R^2}\,\hat{\mathbf{R}}$$

其中 $R$ 是两电荷间距，$\hat{\mathbf{R}}$ 是 $q_1$ 指向 $q_2$ 的单位矢量，$\varepsilon_0 \approx 8.854\times10^{-12}\ \mathrm{F/m}$ 是真空介电常数。<span class="marginnote">力的"反平方"形式不是偶然：它是三维空间中"通量守恒"的必然结果——源发出的某种东西均匀铺满球面，密度随 $1/R^2$ 衰减。这个几何直觉在第 2 节高斯定律里会得到最漂亮的表达。</span>

**电场强度（electric field intensity）** $\mathbf{E}$ 定义为"单位正电荷所受的力"：把试探电荷 $q_t$ 放进场中，$\mathbf{E} = \mathbf{F}/q_t$，单位 V/m。点电荷在场点产生的场

$$\mathbf{E} = \frac{Q}{4\pi\varepsilon_0 R^2}\,\hat{\mathbf{R}}$$

多个电荷的总场由**叠加原理**逐点相加：$\mathbf{E} = \sum_i \mathbf{E}_i$。连续分布的电荷则把求和换成积分——线电荷、面电荷、体电荷分别对应线积分、面积分、体积分。

**辨析｜易错点：** 场强公式里的 $R$ 是"源到场点"的距离，$\hat{\mathbf{R}}$ 必须从源指向场点。方向写反是初学者最高频的错误，尤其对负电荷，$\mathbf{E}$ 方向指向电荷本身（负电荷是电场的"汇"）。

## 2 高斯定律与它的三种用法

计算对称分布的电场，积分往往比叠加更省力。把通量概念与散度概念结合，就得到**高斯定律（Gauss's law）**：穿过闭合曲面的电通量等于面内净电荷除以 $\varepsilon_0$。

$$\oint_S \mathbf{D}\cdot\mathrm{d}\mathbf{S} = Q_{\mathrm{enc}}, \qquad \nabla\cdot\mathbf{D} = \rho$$

其中 $\mathbf{D} = \varepsilon\mathbf{E}$ 是**电位移矢量（electric flux density）**，$\rho$ 是体电荷密度。<span class="marginnote">$\mathbf{D}$ 的引入把"介质极化产生的束缚电荷"吸收进 $\varepsilon$，使高斯定律只需面对自由电荷。这一步"把微观复杂性打包进材料参数"的手法，在磁学里再次出现（$\mathbf{H}$ 与磁化强度），是贯穿全书的方法论。</span>

高斯定律有三种用法：**对称积分**（球对称、轴对称、面对称的电荷分布，选高斯面直接算出 $\mathbf{E}$）；**微分形式**（给定 $\rho(x,y,z)$ 求散度，反解场）；**判据**（由通量分布判断源）。

**辨析｜易错点：** 高斯定律永远成立，但只有**电荷分布高度对称**时才能用它"反解"出 $\mathbf{E}$。对非对称分布，$\oint\mathbf{D}\cdot\mathrm{d}\mathbf{S} = Q$ 只能告诉你总通量，给不出场逐点的大小——拿它硬算任意形状带电体，是常见的方法误用。

## 3 电位与静电场的势

静电场无旋（$\nabla\times\mathbf{E} = 0$），于是存在**电位（electric potential）** $\varphi$ 使

$$\mathbf{E} = -\nabla\varphi$$

负号表示电场指向电位降落的方向——正电荷自发从高电位流向低电位。把电荷 $q$ 从 $A$ 移到 $B$，电场力做功 $W = q(\varphi_A - \varphi_B)$，与路径无关。点电荷的电位

$$\varphi = \frac{Q}{4\pi\varepsilon_0 R}$$

（取无穷远处为零电位参考）。<span class="marginnote">电位是标量，叠加起来只要做代数加法，而电场叠加要矢量加法——这是"先求 $\varphi$ 再取梯度"之所以省力的根本原因，也是工程上用等电位线画电场图（场线处处垂直等位面）的根据。</span>

**辨析｜易错点：** 电位是相对量，谈"某点的电位"必须先声明参考点；但**电位差**是绝对量，与参考点无关。工程上常把"地"取为零电位，于是"对地电压"成为习惯说法——这是参考点的选择，不代表电位本身有绝对意义。

## 4 静电场的边界条件

两种介质的分界面两侧，场量不是随便接上的，必须满足**边界条件**：

电场强度的切向分量连续：$\hat{\mathbf{n}}\times(\mathbf{E}_1 - \mathbf{E}_2) = 0$；
电位移的法向分量差等于自由面电荷密度：$\hat{\mathbf{n}}\cdot(\mathbf{D}_1 - \mathbf{D}_2) = \rho_s$。

前者来自无旋（电场沿分界面的环量为零），后者来自高斯定律（跨过面源的净通量即面电荷）。<span class="marginnote">导体内部静电场为零，于是导体表面外侧 $\mathbf{D}$ 的法向分量就是 $\rho_s$，切向分量为零——静电屏蔽、高压设备外壳均由此而来。</span>

**辨析｜易错点：** 当分界面上无自由面电荷时，$D$ 的法向连续但 $\varepsilon E$ 随之"跳变"，$\mathbf{E}$ 的法向不连续；同时 $\mathbf{E}$ 切向连续。许多人把"$\mathbf{E}$ 连续"当成普遍结论，正确说法是"$\mathbf{E}$ 的**切向分量**连续"。遇到折射问题（电场线斜穿界面），要先分清切向与法向再套公式。

## 5 电容与静电能

两导体带等量异号电荷 $+Q, -Q$，其间电位差 $V$，比值即**电容（capacitance）** $C = Q/V$，单位法拉（F）。平行板电容器的电容

$$C = \frac{\varepsilon S}{d}$$

其中 $S$ 是极板面积，$d$ 是板间距。<span class="marginnote">电容只由几何与介质决定，与是否充电无关——它是"这个导体结构装电荷的能力"的固有属性。这个概念在第六级《电路分析基础》里是三大基本元件之一，在《电力系统分析》里又是线路参数。</span>

静电场存储能量，能量密度

$$w_e = \frac{1}{2}\varepsilon E^2 = \frac{1}{2}\mathbf{D}\cdot\mathbf{E}$$

总静电能 $W_e = \int_V \frac{1}{2}\mathbf{D}\cdot\mathbf{E}\,\mathrm{d}v$。**公式解析**见第 7 节，这里先记住它的用法：已知场求能量，再借 $W_e = \frac{1}{2}CV^2$ 反求电容，是许多复杂电极结构求电容的省力捷径。

## 6 恒定电场：电荷的定常流动

上一节电荷静止；若电荷在导体里以**不随时间变化的密度**定向流动，场不随时间变，就是**恒定电场（steady electric current field）**。定义**电流密度（current density）** $\mathbf{J}$：垂直流过单位面积的电流，单位 A/m²，$I = \int_S \mathbf{J}\cdot\mathrm{d}\mathbf{S}$。

导体内的本构关系是**欧姆定律的场形式** $\mathbf{J} = \sigma\mathbf{E}$，$\sigma$ 是电导率（S/m）。恒定电场中电荷分布不随时间变，所以 $\nabla\cdot\mathbf{J} = 0$——电流线没有源与汇，形成闭合回路，这正是基尔霍夫电流定律（KCL）的场论根源。

恒定电场与静电场满足同一套数学结构：$\nabla\times\mathbf{E}=0$、$\nabla\cdot\mathbf{J}=0$，对应静电场的 $\nabla\times\mathbf{E}=0$、$\nabla\cdot\mathbf{D}=0$（无源区）。因此两者可以**静电类比**：$D\leftrightarrow J$、$\varepsilon\leftrightarrow\sigma$、$C\leftrightarrow G$（电导）。电阻 $R = V/I$ 与电导 $G = 1/R$ 完全可以用静电容类比的办法求。

**辨析｜易错点：** 导体内部静电场为零，但恒定电场不为零——恒定电流要靠电场"推"着电荷走，导体内部 $\mathbf{E} = \mathbf{J}/\sigma \neq 0$，只是导体内表面必有法向场把电流"导向"外表面。把"静电导体"的结论套到恒定电流上，是最常见的一类误判。

## 7 公式解析：用高斯定律求无限长线电荷的场

把抽象公式落到一个经典算例——**无限长均匀线电荷**，线密度 $\rho_l$。求它产生的电场。按三步走：

- **第一步，选高斯面**：电荷绕轴旋转对称，取同轴的圆柱面，半径 $\rho$、长 $l$。由对称性 $\mathbf{E}$ 只有径向分量 $E_\rho$，且同一半径上处处相等。
- **第二步，算通量**：圆柱侧面通量为 $E_\rho \cdot 2\pi\rho l$；上下底面无法向通量。面内净电荷 $Q_{\mathrm{enc}} = \rho_l l$。
- **第三步，联立求解**：$E_\rho \cdot 2\pi\rho l = \rho_l l/\varepsilon_0$，得

$$E_\rho = \frac{\rho_l}{2\pi\varepsilon_0 \rho}$$

**这个结果值得记住两个特征**：场随 $1/\rho$ 衰减（比点电荷的 $1/R^2$ 慢，因为二维空间通量铺在圆周上）；方向始终垂直于线电荷向外。配套电位（取 $\rho_0$ 处为零参考）为 $\varphi = (\rho_l/2\pi\varepsilon_0)\ln(\rho_0/\rho)$。<span class="marginnote">对数电位的出现说明：二维问题的"无穷远"不再是合法的零电位参考点——线电荷延伸到无穷远，能量发散。工程上处理架空输电线（近似无限长）时，电位参考都取有限距离而非无穷远。</span>

## 8 小结

- 库仑定律给出点电荷力，叠加原理推广到任意电荷分布；$\mathbf{E}$ 是"单位正电荷受力"。
- **高斯定律** $\nabla\cdot\mathbf{D}=\rho$：电荷是电场的源；只有对称分布才能用它反解 $\mathbf{E}$。
- 静电场无旋，所以有**电位** $\varphi$，$\mathbf{E}=-\nabla\varphi$；电位差与路径无关。
- **边界条件**：$\mathbf{E}$ 切向连续、$\mathbf{D}$ 法向差等于面电荷；导体内静电场为零。
- **电容** $C=Q/V$ 由几何与介质决定，静电能密度 $w_e=\frac{1}{2}\varepsilon E^2$。
- **恒定电场** $\mathbf{J}=\sigma\mathbf{E}$、$\nabla\cdot\mathbf{J}=0$，与静电场互为静电类比。
- 静电类比是"记忆放大器"：记住静电场的场方程、边界条件与储能公式，把 $D\leftrightarrow J$、$\varepsilon\leftrightarrow\sigma$