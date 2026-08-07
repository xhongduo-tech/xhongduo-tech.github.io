---
title: 时谐电磁波
date: 2026-08-07
---

# 时谐电磁波

<div class="epigraph">
<p>任何波都可以写成不同频率的简谐波的叠加——时谐波是波动世界的原子。</p>
<footer>—— 让-巴蒂斯特 · 约瑟夫 · 傅里叶（Jean-Baptiste Joseph Fourier）</footer>
</div>

<div class="article-byline">
<p>第二级 · 电动力学 ｜ 郭硕鸿《电动力学》第四章 §4.1 ｜ 2026-08-07</p>
</div>

## 为什么先研究「时谐」这种特殊波

麦克斯韦方程组在真空中给出波动方程，但波动方程的一般解包罗万象——任意波形都满足它。为了系统地研究电磁波，我们采用傅里叶的基本思想：**任何时间波形都能展开成不同频率的简谐波（时谐波）的叠加**。于是先吃透单色（单一频率）时谐波，再叠加还原任意波形。这一节建立时谐波的整套语言：复数表示、色散关系、亥姆霍兹方程——它们是后面平面波、反射折射、波导的公共地基。<span class="marginnote">时谐分析是「频率域思维」的体现：时间域的微分方程在频率域变成代数方程（$\partial/\partial t \to -i\omega$），复杂度骤降。这与信号处理里傅里叶变换把卷积变乘法的逻辑完全一致——电动力学只是先走了一步。</span>

## 1 时谐波的复数表示

**时谐电磁场（time-harmonic field）**：随时间按 $\cos\omega t$ 或 $\sin\omega t$ 变化的场。物理场是实函数，但计算用复数极为方便。对任意时谐场：

$$\mathbf{E}(\mathbf{r}, t) = \mathbf{E}_0(\mathbf{r})\cos(\omega t + \phi) \quad \longleftrightarrow \quad \widetilde{\mathbf{E}}(\mathbf{r}, t) = \mathbf{E}_0(\mathbf{r})e^{-i(\omega t + \phi)}$$

**约定**：实际场取复数场的实部，$\mathbf{E}(\mathbf{r},t) = \operatorname{Re}[\widetilde{\mathbf{E}}(\mathbf{r},t)]$。把时间因子 $e^{-i\omega t}$ 分离出去，记**复振幅**（相量）

$$\widetilde{\mathbf{E}}(\mathbf{r}) = \mathbf{E}_0(\mathbf{r})e^{-i\phi}$$

则 $\mathbf{E}(\mathbf{r},t) = \operatorname{Re}[\widetilde{\mathbf{E}}(\mathbf{r})e^{-i\omega t}]$。时间导数在复域中变为乘法：$\partial/\partial t \to -i\omega$。<span class="marginnote">相位约定（用 $e^{-i\omega t}$ 而非 $e^{+i\omega t}$）是各教材最容易不一致的地方——郭硕鸿用 $e^{-i\omega t}$，工程电磁学常用 $e^{j\omega t}$。选定一种约定后，折射率虚部、复介电常数的符号必须与之配套，否则会得出吸收还是增益的相反结论。考试与阅读时先确认约定，再读公式。</span>

**叠加的注意**：复数表示只对**线性**运算（加减、求导、积分）有效。两个时谐量相乘（如求能量密度 $w \propto E^2$、坡印亭矢量 $\mathbf{S} = \mathbf{E}\times\mathbf{H}$）时，**不能**直接乘复数再取实部——必须回到实函数乘，或使用下面的时间平均公式。

## 2 复振幅形式的麦克斯韦方程组与亥姆霍兹方程

把 $\partial/\partial t \to -i\omega$ 代入麦克斯韦方程组，得到复振幅形式：

$$\nabla\times\widetilde{\mathbf{E}} = i\omega\mathbf{B}, \qquad \nabla\times\widetilde{\mathbf{H}} = \mathbf{J} - i\omega\mathbf{D}$$

$$\nabla\cdot\widetilde{\mathbf{D}} = \rho, \qquad \nabla\cdot\widetilde{\mathbf{B}} = 0$$

在无源、线性均匀介质（$\mathbf{J}=0,\ \rho=0$，$\mathbf{D} = \varepsilon\mathbf{E},\ \mathbf{B} = \mu\mathbf{H}$）中，对旋度方程再取旋度，得到**亥姆霍兹方程（Helmholtz equation）**：

$$\nabla^2\widetilde{\mathbf{E}} + k^2\widetilde{\mathbf{E}} = 0, \qquad \nabla^2\widetilde{\mathbf{B}} + k^2\widetilde{\mathbf{B}} = 0$$

其中

$$k = \omega\sqrt{\mu\varepsilon} = \frac{\omega}{v}, \qquad v = \frac{1}{\sqrt{\mu\varepsilon}}$$

$k$ 是**波数**，$v$ 是介质中的**相速度**。<span class="marginnote">亥姆霍兹方程是「时间因子剥离后」的定态波动方程，比波动方程少一个时间维。它属于椭圆型方程，与拉普拉斯/泊松方程同族，求解方法（分离变量、本征函数）与静电边值问题一脉相承——波导、谐振腔、散射问题全在解亥姆霍兹方程。</span>

**色散关系（dispersion relation）**：$\omega$ 与 $k$ 的关系 $k = \omega\sqrt{\mu\varepsilon}$。对线性无色散介质，这是线性关系（$v$ 与频率无关）；有耗介质中 $\varepsilon$ 或 $\mu$ 变为复数量，$k$ 变为复数，波在传播中衰减——这是下一节与导体吸收的入口。

## 3 公式解析：为什么电场与磁场互锁成波

时谐电磁波最反直觉的特点是「电场与磁场互相激发、缺一不可」。从亥姆霍兹方程回到旋度方程看这个耦合：

- **第一步，解亥姆霍兹方程**：$\widetilde{\mathbf{E}}$ 满足 $\nabla^2\widetilde{\mathbf{E}} + k^2\widetilde{\mathbf{E}} = 0$。但 $\widetilde{\mathbf{E}}$ 的三个分量并不独立——还要满足 $\nabla\cdot\widetilde{\mathbf{E}} = 0$（无源条件）。这约束了波的偏振结构：电场必须垂直于传播方向（横波）。
- **第二步，用旋度方程耦合**：一旦知道 $\widetilde{\mathbf{E}}$，磁场由法拉第定律的复形式决定：$\widetilde{\mathbf{B}} = -\dfrac{i}{\omega}\nabla\times\widetilde{\mathbf{E}}$。也就是说**磁场不是独立的解，而是电场的旋度**——电场与磁场之间有确定的代数关系（对平面波是 $\mathbf{B} = \mathbf{k}\times\mathbf{E}/\omega$）。<span class="marginnote">把电磁场想成「一个场的两种表现」：解出 $\mathbf{E}$ 就自动有 $\mathbf{B}$（反之亦然）。在波里，$\mathbf{E}$ 与 $\mathbf{B}$ 同相位（无耗介质），且 $E = vB$。到反射折射与导体问题里，这个相位关系会被破坏，成为物理新效应的来源。</span>
- **第三步，能量循环**：坡印亭矢量 $\mathbf{S} = \mathbf{E}\times\mathbf{H}$ 沿传播方向，能量从电场「倒」进磁场、再倒回来，平均能流恒定。电磁波是电场与磁场能量互相转换、整体向前传输的过程——这正是「电磁波能脱离源独立传播」的能量学解释。

## 4 时谐场的能量与能流：时间平均值

由于瞬时能量密度 $w = \frac{1}{2}(\varepsilon E^2 + B^2/\mu)$ 以 $2\omega$ 频率振荡，实际关心的是**时间平均**。用复振幅表述的时间平均公式为：

$$\langle w \rangle = \frac{1}{4}\operatorname{Re}\left[\varepsilon\widetilde{\mathbf{E}}\cdot\widetilde{\mathbf{E}}^* + \frac{1}{\mu}\widetilde{\mathbf{B}}\cdot\widetilde{\mathbf{B}}^*\right]$$

$$\langle \mathbf{S} \rangle = \frac{1}{2}\operatorname{Re}\left[\widetilde{\mathbf{E}}\times\widetilde{\mathbf{H}}^*\right]$$

后一式定义**复坡印亭矢量** $\widetilde{\mathbf{S}} = \frac{1}{2}\widetilde{\mathbf{E}}\times\widetilde{\mathbf{H}}^*$，其实部是时间平均能流，虚部与储能有关。<span class="marginnote">公式里的 $\frac{1}{2}$ 与共轭：$\langle\cos^2\omega t\rangle = 1/2$ 提供 $1/2$ 因子，$\widetilde{\mathbf{H}}^*$ 的共轭来自「两个实量乘积的平均 = 复量乘共轭再取实部的一半」。工程中有的书不写 $1/2$（用峰值而非有效值），读公式前先对量纲。</span>

**辨析｜易错点：** 计算 $\langle w \rangle$ 或 $\langle \mathbf{S} \rangle$ 时，**不要**直接用 $\operatorname{Re}[\widetilde{\mathbf{E}}]\times\operatorname{Re}[\widetilde{\mathbf{H}}]$——那是瞬时值在振荡，不是平均值。必须用「复量 × 共轭」的标准公式。同理，求瞬时坡印亭矢量也不要直接取 $\frac{1}{2}$ 那套公式（那是平均的）。区分「瞬时」与「平均」两套公式，是时谐电磁波计算的第一个坑。

## 5 时谐电磁波的实用价值

- **通信**：电磁波按频率划分波段——广播、微波、光通信的本质都是不同频率的时谐波叠加（调制）。
- **电路分析**：交流电路、滤波器、天线阻抗全用相量法（时谐分析）计算。
- **数值方法**：有限元、矩量法解电磁问题几乎都在频域（时谐）进行，因为频率域方程是椭圆型的、便于离散求解。
- **材料表征**：通过测量介质对时谐波的反射/透射谱，反推材料的 $\varepsilon(\omega)$、$\mu(\omega)$——这是微波遥感与材料科学的标准手段。<span class="marginnote">时谐语言还通向「复介电常数 $\varepsilon(\omega) = \varepsilon' + i\varepsilon''$」：实部决定折射与相速，虚部决定吸收。材料的「色散」全部打包进 $\varepsilon(\omega)$ 的频率依赖里——从 X 射线到无线电，同一套框架描述所有电磁波与物质的相互作用。</span>

## 6 时谐波的一个完整计算：复数法解 RC 电路

时谐法的价值不在于「重算一遍电路」，而在于它把电路与场统一进同一套语言。用 RC 电路演示「$\partial/\partial t \to -i\omega$」如何把微分方程变成代数方程。

**问题**：RC 串联电路接时谐电源 $V(t) = V_0\cos\omega t$，求电流。

**第一步，写电路方程。** $V(t) = IR + \dfrac{Q}{C}$，其中 $I = \mathrm{d}Q/\mathrm{d}t$。对时谐量，$Q(t) = \operatorname{Re}[\widetilde{Q}e^{-i\omega t}]$，$I = \mathrm{d}Q/\mathrm{d}t = -i\omega Q$——**微分变成了乘以 $-i\omega$**。

**第二步，转成复代数方程。** $\widetilde{V} = \widetilde{I}R + \dfrac{\widetilde{Q}}{C} = \widetilde{I}R + \dfrac{\widetilde{I}}{-i\omega C} = \widetilde{I}\left(R + \dfrac{i}{\omega C}\right)$。于是

$$\widetilde{I} = \frac{\widetilde{V}}{R + i/(\omega C)}$$

**第三步，读出物理。** 分母 $Z = R + i/(\omega C)$ 是复阻抗：实部是电阻，虚部是容抗。电流与电压的相位差 $\phi = \arctan\left(\dfrac{1}{\omega CR}\right)$——**电容使电流领先电压**。取实部得 $I(t) = \dfrac{V_0}{|Z|}\cos(\omega t - \phi)$。

**这个例子的意义**：同一套「复数阻抗」语言，在电路里算的是电流，在波导里算的是传播常数，在天线里算的是输入阻抗——**时谐方法统一了「集总电路」与「分布电磁场」两套世界**。你在这里学的相量法，正是后面传输线、天线、微波工程全部计算的地基。

**辨析｜易错点：** ① 复阻抗的虚部正负号依赖相位约定（$e^{-i\omega t}$ 时容抗为 $+i/\omega C$，用 $e^{+i\omega t}$ 约定则变号）——**先统一约定再算**。② $\widetilde{I} = -i\omega\widetilde{Q}$ 里 $-i$ 因子对应 90° 相位延迟，丢了它电流与电压的相位关系全错。③ 复数的模与相位分开处理：$|Z|$ 决定幅度，$\arg Z$ 决定相移，两者必须同时取出。

**复功率与能量守恒**：时谐场的能量分析常用**复坡印亭定理**：

$$\nabla\cdot\widetilde{\mathbf{S}} = -\frac{1}{2}\widetilde{\mathbf{J}}^*\cdot\widetilde{\mathbf{E}} + 2i\omega\left(\frac{1}{4}\varepsilon|\widetilde{\mathbf{E}}|^2 - \frac{1}{4}\mu|\widetilde{\mathbf{H}}|^2\right)$$

左端复能流的散度，实部是有功功率（耗散），虚部是无功功率（储能振荡）。**这个定理统一了「能量守恒」与「相位」**：实部对应坡印亭定理的平均形式，虚部对应电场/磁场能量密度的差值。

**时谐法与频域的工程思维**：为什么整个射频工程都泡在频域？因为时域里「一段电缆 + 一段不连续」是微分方程，频域里变成了「阻抗匹配」的代数问题；时域里「调制信号」是复杂波形，频域里是「载波 + 边带」的清晰频谱。**「把微分变代数、把波形变频谱」是时谐分析送给工程师的礼物**——从手机射频到雷达信号处理，全是这套语言。

**时谐法与量子力学的相通**：时谐电磁场与量子力学共享同一套数学结构——$\partial/\partial t \to -i\omega$ 对应薛定谔方程的 $\partial/\partial t \to -iE/\hbar$，亥姆霍兹方程对应定态薛定谔方程。**「分离时间因子、求解空间定态」的套路，正是量子力学「定态」概念的经典翻版**。学通时谐电磁波，等于提前预习了量子力学的一半数学工具。

**辨析｜易错点：** ① 复振幅 $\widetilde{\mathbf{E}}$ 的相位是「空间相位」$e^{ikz}$，时间相位 $e^{-i\omega t}$ 已分离——两者不要混进同一个 $e$ 指数。② 复坡印亭矢量 $\widetilde{\mathbf{S}} = \frac{1}{2}\widetilde{\mathbf{E}}\times\widetilde{\mathbf{H}}^*$ 的实部是平均能流，虚部与无功功率对应；取实部前先确认用的是「共轭」而非原场。③ 对非线性介质或含非线性元件的系统，复数法失效（叠加原理不成立），必须回到时域。

## 7 小结

- 任意波 = 时谐波叠加（傅里叶）；时谐场用**复数表示**，$\partial/\partial t \to -i\omega$。
- 无源介质中 $\mathbf{E},\mathbf{B}$ 满足**亥姆霍兹方程** $\nabla^2\widetilde{\mathbf{E}} + k^2\widetilde{\mathbf{E}} = 0$，$k = \omega/v$。
- 电场与磁场**互锁**：$\widetilde{\mathbf{B}} = -i\nabla\times\widetilde{\mathbf{E}}/\omega$，能量经坡印亭矢量向前传输。
- 时间平均用**复量乘共轭**公式：$\langle\mathbf{S}\rangle = \frac{1}{2}\operatorname{Re}[\widetilde{\mathbf{E}}\times\widetilde{\mathbf{H}}^*]$。
- 时谐分析是通信、电路、数值电磁学的共同语言。

在下一节，我们研究最简单的时谐波解——**平面电磁波**：它的偏振、传播、能量流动，以及它如何与物质相互作用。
