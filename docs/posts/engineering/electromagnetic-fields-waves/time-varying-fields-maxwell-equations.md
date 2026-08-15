---
title: 时变电磁场与 Maxwell 方程组
date: 2026-08-07
---

# 时变电磁场与 Maxwell 方程组

<div class="epigraph">
<p>我们几乎无法避免这样一种推断：光是在同一种介质中传播的横波，而这种介质正是电与磁现象的原因。</p>
<footer>—— 詹姆斯 · 克拉克 · 麦克斯韦（James Clerk Maxwell）</footer>
</div>

<div class="article-byline">
<p>第六级 · 电磁场与电磁波（工程电磁场） ｜ David K. Cheng《Field and Wave Electromagnetics》第2版 §7-1～§7-9 ｜ 2026-08-07</p>
</div>

## 为什么从 Maxwell 方程组开始

前两篇把电磁场切成两半：静电与静磁各自为政，互不相干。本篇把它们**缝合**起来，靠的是两条"时变"规律——法拉第的"变化的磁场产生电场"，与麦克斯韦补上的"变化的电场产生磁场"。这一针缝合的直接后果是：**电磁波的存在在数学上不可避免**。1873 年麦克斯韦写下这组方程时，方程自己告诉他光就是电磁波，波长覆盖从无线电到可见光到 X 射线的整个谱。<span class="marginnote">这个发现被爱因斯坦称为"自牛顿以来物理学最深刻的一场实在概念变革"。今天你手机里的每一格信号、Wi-Fi 的每一比特数据，追根溯源都在这组方程里。</span>本篇对标 David K. Cheng《Field and Wave Electromagnetics》第 7 章，建立电磁理论的"宪法"——Maxwell 方程组，并导出它的第一个伟大推论：波动方程。

## 1 法拉第电磁感应定律：变化的磁场造电场

1831 年法拉第发现：**穿过回路的磁通变化时，回路中会感应出电动势**。定量写成法拉第电磁感应定律

$$\oint_C \mathbf{E}\cdot\mathrm{d}\mathbf{l} = -\frac{\mathrm{d}}{\mathrm{d}t}\int_S \mathbf{B}\cdot\mathrm{d}\mathbf{S}$$

负号即**楞次定律**：感应电流的方向总是"对抗"引起它的磁通变化。<span class="marginnote">感应电动势的来源有两类：磁场随时间变化，或回路相对磁场运动（发电机转子转动）。Maxwell 方程组描述的是一般情形——场本身随时间变化，运动导体是它的特例。发电机、变压器、电磁炉全在这条定律的账下。</span>

用斯托克斯定理把环路积分换成面积分，就得到微分形式 $\nabla\times\mathbf{E} = -\partial\mathbf{B}/\partial t$。<strong>关键洞见：时变磁场的旋度不为零，电场不再无旋，也就不能再写成某标量位的梯度</strong>——上一节"静电无旋"的结论在时变情形下必须让位。

一个立刻的推论：时变情形下"电位"不再是全局概念，电路里的"电压"也只在准静态近似下才有严格意义。工程上当频率升高到信号波长可与电路尺寸相比时，必须放弃集总电路思维，回到场——这正是第 7 篇传输线理论的起点。所以法拉第定律不仅是电磁学的砖，也是**电路论向场论过渡的枢纽**。

## 2 位移电流：麦克斯韦补上的那块拼图

安培环路定律 $\nabla\times\mathbf{H}=\mathbf{J}$ 在时变情形下自相矛盾：对两边取散度，左边恒为零，右边 $\nabla\cdot\mathbf{J}$ 在电荷变化时不为零（连续性方程 $\nabla\cdot\mathbf{J}=-\partial\rho/\partial t$）。麦克斯韦的诊断：方程缺了一项，这一项在充电电容器两极板之间把"电流回路"续上。

$$\nabla\times\mathbf{H} = \mathbf{J} + \frac{\partial\mathbf{D}}{\partial t}$$

$\partial\mathbf{D}/\partial t$ 叫**位移电流密度（displacement current density）**。<span class="marginnote">在充电电容器的极板间，没有自由电荷流动（$\mathbf{J}=0$），但电场随时间变化（$\partial\mathbf{D}/\partial t\neq0$），位移电流恰好等于导线里的传导电流——"电流"被概念性地打通了。高频电路里位移电流还可以比传导电流大得多，这正是微波在介质中传播的机制。</span>

**辨析｜易错点：** 位移电流不是"电荷的流动"，它是**电位移矢量的时间变化率**，在真空中照样存在。把它误解为某种真实电荷流，是学电磁场最顽固的错误之一。真空里没有传导电流，但电磁波照样传播，靠的就是位移电流一环扣一环。

## 3 Maxwell 方程组：电磁理论的宪法

把四条规律并排放下，就是完整的**Maxwell 方程组**：

| 名称 | 微分形式 | 积分形式 | 物理含义 |
| --- | --- | --- | --- |
| 高斯电定律 | $\nabla\cdot\mathbf{D}=\rho$ | $\oint_S\mathbf{D}\cdot\mathrm{d}\mathbf{S}=Q$ | 电荷是电场的源 |
| 高斯磁定律 | $\nabla\cdot\mathbf{B}=0$ | $\oint_S\mathbf{B}\cdot\mathrm{d}\mathbf{S}=0$ | 无磁单极，磁力线闭合 |
| 法拉第定律 | $\nabla\times\mathbf{E}=-\partial\mathbf{B}/\partial t$ | $\oint_C\mathbf{E}\cdot\mathrm{d}\mathbf{l}=-\frac{\mathrm{d}}{\mathrm{d}t}\Phi$ | 变化磁场感生电场 |
| 安培-麦克斯韦定律 | $\nabla\times\mathbf{H}=\mathbf{J}+\partial\mathbf{D}/\partial t$ | $\oint_C\mathbf{H}\cdot\mathrm{d}\mathbf{l}=I+\frac{\mathrm{d}}{\mathrm{d}t}\Psi$ | 电流与变化电场感生磁场 |

再补上介质本构关系 $\mathbf{D}=\varepsilon\mathbf{E}$、$\mathbf{B}=\mu\mathbf{H}$、$\mathbf{J}=\sigma\mathbf{E}$，电磁场问题在原理上全部闭合。<span class="marginnote">注意对称性：法拉第定律说"变化磁场→电场"，安培-麦克斯韦定律说"变化电场→磁场"，一条负号、一条正号，恰好让能量在电场与磁场之间来回振荡——这就是波。</span>

**辨析｜易错点：** 四条方程中，静电场的两条（$\nabla\cdot\mathbf{D}=\rho$）与静磁的两条（$\nabla\cdot\mathbf{B}=0$）在时变下**不变**；变的只有两条旋度方程。把四条全当成"时变专属"或把两条散度方程也加上 $\partial/\partial t$，是常见的过度改写。

## 4 时变场的边界条件与电荷守恒

静态场的边界条件在时变下依然成立，只是两边可以再叠加时变项：

- $\hat{\mathbf{n}}\times(\mathbf{E}_1-\mathbf{E}_2)=0$：电场切向分量连续；
- $\hat{\mathbf{n}}\times(\mathbf{H}_1-\mathbf{H}_2)=\mathbf{K}_s$：磁场切向分量差等于自由面电流；
- $\hat{\mathbf{n}}\cdot(\mathbf{D}_1-\mathbf{D}_2)=\rho_s$、$\hat{\mathbf{n}}\cdot(\mathbf{B}_1-\mathbf{B}_2)=0$：法向条件与静态相同。

**辨析｜易错点：** 理想导体的表面，时变电场切向为零（否则会驱动无限大电流）；但表面法向场可以不为零，且表面的时变电荷、时变面电流把场"关"在导体外——这就是第 8 篇波导边界条件的来源。把"导体表面电场为零"不加区分地当"整个切向法向都为零"，是推导波导模时最常翻车的地方。

## 5 坡印廷定理：电磁能量的流动

能量守恒在电磁场里的微分表述是**坡印廷定理**

$$-\frac{\partial w}{\partial t} = \nabla\cdot\mathbf{S} + \mathbf{J}\cdot\mathbf{E}, \qquad \mathbf{S} = \mathbf{E}\times\mathbf{H}$$

**坡印廷矢量（Poynting vector）** $\mathbf{S}$ 单位 W/m²，是电磁能量流动的"功率流密度"；$\mathbf{J}\cdot\mathbf{E}$ 是欧姆损耗。式子的意思是：某点电磁能量密度的减少，一部分变成流出该点的能流，一部分被焦耳热耗散。其中总能量密度

$$w = \frac{1}{2}\varepsilon E^2 + \frac{1}{2}\mu H^2 = w_e + w_m$$

把前两篇的静电能密度与磁能密度加在一起——时变场里能量在电场与磁场之间来回转移，$w$ 就是这场振荡的"总资本"。

<span class="marginnote">一个反直觉的结论：直流电路的能量不是沿着导线内部"运"过去的，而是以坡印廷能流的形式在导线<strong>外的空间</strong>流动，导线把能量"引"到负载。把电池、导线、负载周围的空间电场磁场画出来，会看到能流确实贴着导线表面流向负载。</span>

对时谐场，瞬态坡印廷矢量 $\mathbf{S}=\mathbf{E}\times\mathbf{H}$ 以 $2\omega$ 的频率快速振荡，工程上关心的是**时间平均能流密度**——"单位时间、单位面积真正流走的平均功率"

$$\langle\mathbf{S}\rangle = \frac{1}{2}\mathrm{Re}[\dot{\mathbf{E}}\times\dot{\mathbf{H}}^*]$$

它由复振幅直接算出，是天线辐射功率、无线链路预算、传输线功率的统一起点。注意有功功率只来自 $\dot{\mathbf{E}}$ 与 $\dot{\mathbf{H}}^*$ 中**同相**的分量，正交分量只做无功振荡、不带走功率——这与电路里"平均功率等于电压电流同相分量之积"是同一句话，只是换成了场的语言。下一篇算平面波功率流，用的正是这条式子。

## 6 时谐场与复数表示

工程上绝大多数信号是正弦稳态（时谐），于是引入**复数表示（phasor）**：$\mathbf{E}(t) = \mathrm{Re}[\dot{\mathbf{E}}\,e^{j\omega t}]$，其中 $\dot{\mathbf{E}}$ 是复振幅，只含空间变化与相位。<span class="marginnote">复数表示把一个偏微分方程在时间上的求导变成代数乘法 $\partial/\partial t \leftrightarrow j\omega$——这是后三篇（平面波、传输线、波导）所有计算的枢纽，也是射频工程的通用语言。</span>

时谐下 Maxwell 方程组化为

$$\nabla\times\dot{\mathbf{E}} = -j\omega\mu\dot{\mathbf{H}}, \qquad \nabla\times\dot{\mathbf{H}} = \dot{\mathbf{J}} + j\omega\varepsilon\dot{\mathbf{E}}$$

值得强调的是，复振幅 $\dot{\mathbf{E}}$ 是一个**只含空间变化的相量**，时间因子 $e^{j\omega t}$ 在书写中略去不写；实际的瞬时值等于取实部。这套记号牺牲了一点直观，换来的是把"对时间的导数"全部变成"乘 $j\omega$"，波动方程的推导因此变成纯代数运算。

两个方程四个未知分量，联立消元后每一组就只剩一个未知量，这就是下一节波动方程的求法。

## 7 公式解析：从 Maxwell 方程组到波动方程

把两条旋度方程消元，可以导出一组自洽的**波动方程**。以电场为例，三步走：

- **第一步，对法拉第定律取旋度**：$\nabla\times(\nabla\times\dot{\mathbf{E}}) = -j\omega\mu\,\nabla\times\dot{\mathbf{H}}$。
- **第二步，代入安培-麦克斯韦定律**：$\nabla\times\dot{\mathbf{H}} = j\omega\varepsilon\dot{\mathbf{E}}$（无源区 $\dot{\mathbf{J}}=0$），得 $\nabla\times(\nabla\times\dot{\mathbf{E}}) = \omega^2\mu\varepsilon\dot{\mathbf{E}}$。
- **第三步，用矢量恒等式展开**：$\nabla(\nabla\cdot\dot{\mathbf{E}}) - \nabla^2\dot{\mathbf{E}} = \omega^2\mu\varepsilon\dot{\mathbf{E}}$，无源区 $\nabla\cdot\dot{\mathbf{E}}=0$，于是

$$\nabla^2\dot{\mathbf{E}} + \omega^2\mu\varepsilon\dot{\mathbf{E}} = 0$$

这就是**亥姆霍兹方程**。变回时间域，它等价于 $\nabla^2\mathbf{E} = \mu\varepsilon\,\partial^2\mathbf{E}/\partial t^2$——标准的波动方程，传播速度

$$v = \frac{1}{\sqrt{\mu\varepsilon}}$$

**三个要点**：第一，波动方程是 Maxwell 方程组的**推论**，不是新假设；第二，速度由介质决定，真空里 $v = 1/\sqrt{\mu_0\varepsilon_0} = c \approx 3\times10^8\ \mathrm{m/s}$，等于光速——这正是麦克斯韦写下那篇名言的依据；第三，电场与磁场通过耦合方程互相"生成"，你推我、我推你，波就这样在无源真空里自己跑了下去。<span class="marginnote">"真空里没有电荷电流，却能有电磁波"常被当作魔法，其实关键在位移电流：时变电场产生时变磁场，时变磁场又产生时变电场，一环扣一环，能量在两者间换手前进。</span>

## 8 小结

- **法拉第定律** $\nabla\times\mathbf{E}=-\partial\mathbf{B}/\partial t$：变化磁场造电场，负号即楞次定律。
- **位移电流** $\partial\mathbf{D}/\partial t$：麦克斯韦的补丁，让"电流回路"在电容器极板间断开处续上，也保住了电荷守恒。
- **Maxwell 方程组**四条：两条散度（有源/无源）描述场形，两条旋度（时变）描述耦合；本构关系补齐介质。
- **边界条件**：$\mathbf{E}$ 切向连续、$\mathbf{H}$ 切向差为面电流，时变下不变。
- **坡印廷定理**：能流密度 $\mathbf{S}=\mathbf{E}\times\mathbf{H}$，电磁能量在场空间流动。
- **时谐复数表示**：$\partial/\partial t\leftrightarrow j\omega$，把偏微分方程代数化；由此导出波动方程 $v=1/\sqrt{\mu\varepsilon}=c$；时谐场时间平均能流由 $\langle\mathbf{S}\rangle=\frac{1}{2}\mathrm{Re}[\dot{\mathbf{E}}\times\dot{\mathbf{H}}^*]$ 给出。
- 边界条件、坡印廷定理、复数表示在时谐下仍成立，是全场论在射频工程里的通用外壳。

在下一节，我们将拿到波动方程的具体解，并算清它在理想介质与导电媒质里分别怎样奔跑与衰减——那就是**平面电磁波的传播**。