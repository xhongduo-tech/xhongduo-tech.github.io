---
title: 平面电磁波在理想介质/导电媒质中的传播
date: 2026-08-07
---

# 平面电磁波在理想介质/导电媒质中的传播

<div class="epigraph">
<p>这毫无用处……但这只是个实验，它证明了马斯特罗·麦克斯韦是对的——我们只是拥有这些肉眼看不见的神秘电磁波，但它们确实在那里。</p>
<footer>—— 海因里希 · 赫兹（Heinrich Hertz）</footer>
</div>

<div class="article-byline">
<p>第六级 · 电磁场与电磁波（工程电磁场） ｜ David K. Cheng《Field and Wave Electromagnetics》第2版 §8-1～§8-6 ｜ 2026-08-07</p>
</div>

## 为什么从平面波开始

上一节从 Maxwell 方程组导出了波动方程，但没说解长什么样。本篇解出**最简单的一族解——均匀平面波（uniform plane wave）**：等相位面是平面，且面上场强处处相同。<span class="marginnote">为什么先学它？因为任意复杂电磁波都能用平面波的叠加（傅里叶）拼出来。理解一个沿单一方向传播的平面波，就理解了传播速度、波阻抗、衰减、极化这些全部概念——它们是传输线、波导、天线、光学共同的地基。</span>真实世界的波当然复杂得多，但工程师的第一课永远是把它拆成平面波来看。本篇对标 David K. Cheng《Field and Wave Electromagnetics》第 8 章：先看理想介质里平面波如何"无损耗"地奔跑，再看导电媒质里它如何被耗散、衰减。

## 1 无源介质中的波动方程与平面波解

在无源、无耗、均匀介质里，电场满足齐次亥姆霍兹方程 $\nabla^2\dot{\mathbf{E}} + k^2\dot{\mathbf{E}} = 0$，其中 $k = \omega\sqrt{\mu\varepsilon}$ 是**波数（wavenumber）**。<span class="marginnote">波数 $k$ 的物理含义：$2\pi$ 长度内有多少个波。它与波长 $\lambda$ 的关系 $k=2\pi/\lambda$，与频率的关系 $k=\omega/v$，是贯穿全部波动理论的三个换算键。</span>

设波沿 $+z$ 方向传播、电场只有 $x$ 分量，波动方程化为 $\mathrm{d}^2\dot{E}_x/\mathrm{d}z^2 + k^2\dot{E}_x = 0$，解为

$$\dot{E}_x = E_0 e^{-jkz}$$

瞬时值为 $E_x(z,t) = E_0\cos(\omega t - kz)$。这是一个以 $\omega t - kz$ 为相位的行波：固定某一相位点（$\omega t - kz = \text{常数}$），它沿 $+z$ 匀速推进。<strong>均匀平面波的三个"均匀"指标：等相位面是平面、面上幅度相同、极化方向不变</strong>——任何一个"均匀"被破坏，就成了非均匀波（后面波导里会见到）。

## 2 横波性与波阻抗

把 $\dot{E}_x$ 代回 Maxwell 方程组，可以解出磁场。电场沿 $x$、磁场沿 $y$、传播方向沿 $z$——三者两两垂直，电磁波是**横波（TEM 波）**：电场、磁场都垂直于传播方向。场幅值之比是一个常数，叫**波阻抗（intrinsic impedance）**

$$\eta = \frac{E_x}{H_y} = \sqrt{\frac{\mu}{\varepsilon}}$$

真空/空气中 $\eta_0 = \sqrt{\mu_0/\varepsilon_0} \approx 377\ \Omega$。<span class="marginnote">$\eta$ 的量纲是欧姆，但它不是电路的阻抗，而是"电场强度与磁场强度之比"——平面波的固有属性。记住 377Ω 这个数，射频工程里到处用它估算：知道电场幅值就知道磁场幅值，知道坡印廷能流也能反推场。</span>

**辨析｜易错点：** $\mathbf{E}\times\mathbf{H}$ 的方向必须等于传播方向（坡印廷能流指向传播方向）。若手推时发现 $\mathbf{E}\times\mathbf{H}$ 与 $k$ 反了，多半是叉积方向或旋度方程符号出错——这是自洽性检验，比任何记忆都可靠。

## 3 相速度、波长与色散

相位 $\omega t - kz$ 恒定给出 $z = (\omega/k)t$，于是**相速度（phase velocity）**

$$v_p = \frac{\omega}{k} = \frac{1}{\sqrt{\mu\varepsilon}}$$

**波长** $\lambda = 2\pi/k = v_p/f$。在无耗介质里 $v_p$ 与频率无关——所有频率的平面波跑一样快，波形在传播中不畸变。<span class="marginnote">一旦介质有损耗或几何有约束（波导、光纤），$k$ 与 $\omega$ 不再成正比，不同频率跑速不同，叫<strong>色散（dispersion）</strong>。色散导致脉冲信号展宽，是高速通信的天然敌人——这正是光纤通信里"色散补偿"要处理的问题，本博客《通信原理》有专门展开。</span>

**辨析｜易错点：** 相速度可以大于光速而不违反相对论，因为它描述的是"等相位面"的移动速度，不携带信息。携带信息的信号速度是**群速度** $v_g = \mathrm{d}\omega/\mathrm{d}k$，在色散介质中两者不同。把相速度当成信号速度，是常见的概念混淆。

## 4 平面波的能量与功率流

时谐场的坡印廷矢量随时间快速振荡，工程关心的是**时间平均能流密度（平均功率流）**

$$\langle\mathbf{S}\rangle = \frac{1}{2}\mathrm{Re}[\dot{\mathbf{E}}\times\dot{\mathbf{H}}^*] = \frac{|E_0|^2}{2\eta}\,\hat{\mathbf{z}}$$

平均能流密度是"单位面积、单位时间"流过的平均能量，单位 W/m²——这就是天线辐射强度、无线链路预算里"功率密度"的来源。<span class="marginnote">远场通信的经典估算：基站发射 50 W 功率，天线在目标方向增益 30 dB，距离 1 km 处的功率密度就由这条式子决定——比手机天线大多少倍、链路预算够不够，都从这里算起。本博客《信息与通信工程》专题会接着算。</span>

## 5 导电媒质中的平面波：衰减与相位

真实媒质有电导率 $\sigma$，传导电流 $\mathbf{J}=\sigma\mathbf{E}$ 参与耦合。把安培-麦克斯韦定律写成 $\nabla\times\dot{\mathbf{H}} = j\omega\varepsilon_c\dot{\mathbf{E}}$，等效复介电常数

$$\varepsilon_c = \varepsilon - j\frac{\sigma}{\omega}$$

于是波数变成复数 $\gamma = \alpha + j\beta$，其中 $\alpha$ 是**衰减常数（attenuation constant）**、$\beta$ 是**相位常数（phase constant）**。波解成为

$$\dot{E}_x = E_0 e^{-\alpha z}e^{-j\beta z}$$

$e^{-\alpha z}$ 项让振幅沿传播方向指数衰减。<strong>导电媒质里的波 = 无耗行波 + 指数衰减</strong>——能量被焦耳热一点一点吃掉。$\alpha$ 与 $\beta$ 的完整表达式由

$$\alpha = \omega\sqrt{\frac{\mu\varepsilon}{2}}\left[\sqrt{1+\left(\frac{\sigma}{\omega\varepsilon}\right)^2}-1\right]^{1/2}, \qquad \beta = \omega\sqrt{\frac{\mu\varepsilon}{2}}\left[\sqrt{1+\left(\frac{\sigma}{\omega\varepsilon}\right)^2}+1\right]^{1/2}$$

给出。其中 $\sigma/(\omega\varepsilon)$ 叫**损耗正切（loss tangent）**，是"媒质有多导电"的无量纲判据。<span class="marginnote">损耗正切 $\tan\delta = \sigma/(\omega\varepsilon)$：远小于 1 是良介质（微波、光学材料），远大于 1 是良导体（铜、铝）。同一块材料在低频可能是良导体、在太赫兹却接近介质——"导体还是介质"由频率说了算。</span>

损耗正切把媒质分档，是工程选材的第一眼判断：

| 媒质类型 | $\tan\delta = \sigma/(\omega\varepsilon)$ | 衰减行为 | 典型例子 |
| --- | --- | --- | --- |
| 良介质 | $\ll 1$ | 衰减极弱，近似无耗 | 聚乙烯、玻璃、云母 |
| 有耗介质 | 与 1 同量级 | 中等衰减，色散明显 | 潮湿土壤、生物组织 |
| 良导体 | $\gg 1$ | 波在极浅层衰减殆尽 | 铜、铝、金 |

微波电路板材、天线罩材料都要挑 $\tan\delta$ 小的介质，功率损耗才低；而电磁兼容、屏蔽设计恰恰相反，要利用良导体的"拒绝"能力。

## 6 趋肤效应：高频电流的"表皮生活"

在良导体（$\sigma/(\omega\varepsilon)\gg1$）中，衰减常数简化为

$$\alpha \approx \sqrt{\frac{\omega\mu\sigma}{2}} = \frac{1}{\delta}$$

**趋肤深度（skin depth）** $\delta = \sqrt{2/(\omega\mu\sigma)}$ 是波衰减到表面值的 $1/e \approx 37\%$ 的距离。<span class="marginnote">铜在 60 Hz 工频下 $\delta\approx8.5\ \mathrm{mm}$，在 1 GHz 微波下 $\delta\approx2\ \mu\mathrm{m}$——同一个导体，频率升高 7 个数量级，电流就被"挤"到表面薄薄一层。这是高频电路设计里绕不开的物理。</span>

趋肤效应有三个直接工程后果：**高频电阻增大**（有效截面积变小，交流电阻远大于直流电阻）；**电流趋向表面**（空心管在高频下与实心棒等效，因此高频导体用空心或镀银）；**屏蔽与衰减**（电磁波进不了导体深处，金属外壳天然屏蔽——微波炉门上的金属网孔小于趋肤深度对应的波长就能挡住微波）。

**辨析｜易错点：** 趋肤深度 $\delta$ 随频率升高而变薄、随电导率增大而变薄——"更好的导体反而让波进得更浅"。直觉上"导电越好屏蔽越强"是对的，但"导电越好衰减越快"要小心：$\alpha = 1/\delta$ 随 $\sigma$ 增大而增大，意味着波在好导体表面就衰减殆尽。两者是同一枚硬币的两面。

工程上还常用"表面电阻"来量化趋肤效应的影响：长度为 $l$、宽度为 $w$ 的导体，高频交流电阻近似 $R_{ac} \approx l/(w\sigma\delta)$，把趋肤深度当作"电流实际流过的厚度"。由此可知，把导体做厚到远超 $\delta$ 并不会降低高频电阻——厚度的收益在 $\delta$ 处已经饱和，这也是"镀银导体"与"实心银块"在高频下几乎等价的道理。

## 7 公式解析：良导体趋肤深度的数量级

把 $\delta = \sqrt{2/(\omega\mu\sigma)}$ 算一个具体数，感受它的力量。设铜：$\sigma \approx 5.8\times10^7\ \mathrm{S/m}$，$\mu\approx\mu_0 = 4\pi\times10^{-7}\ \mathrm{H/m}$。

**第一步，代入工频 50 Hz**：$\omega = 2\pi\times50 \approx 314\ \mathrm{rad/s}$，$\delta = \sqrt{2/(314\times4\pi\times10^{-7}\times5.8\times10^7)} \approx 9.3\times10^{-3}\ \mathrm{m} \approx 9.3\ \mathrm{mm}$。
**第二步，代入 Wi-Fi 2.4 GHz**：$\omega = 2\pi\times2.4\times10^9 \approx 1.5\times10^{10}$，$\delta = \sqrt{2/(1.5\times10^{10}\times4\pi\times10^{-7}\times5.8\times10^7)} \approx 1.3\times10^{-6}\ \mathrm{m} \approx 1.3\ \mu\mathrm{m}$。
**第三步，读出结论**：频率升高约 5 个数量级，$\delta$ 缩小约 3.5 个数量级——因为 $\delta\propto 1/\sqrt{f}$。所以电力传输线用粗铜线（低频，集肤损失小），而射频器件只需镀一层几微米的银就能承担全部电流。

**这个公式的工程配方**：$\delta = \sqrt{2/(\omega\mu\sigma)}$ 里，频率越高、磁导率越大、电导率越大，趋肤深度越浅。<span class="marginnote">顺带一提：这解释了为什么电镀、镀银、镀金高频导体那么贵——多出来的金属只是在趋肤深度内"有用"，其余全是浪费。懂了这个，再看射频器件的选材就有数了。</span>

## 8 小结

- **均匀平面波**是波动方程最简单解：等相位面为平面、面上幅度相同，可用平面波叠加出任意波。
- 电磁波是**横波**：$\mathbf{E}\perp\mathbf{H}\perp\hat{\mathbf{z}}$；波阻抗 $\eta=\sqrt{\mu/\varepsilon}$，真空约 377Ω。
- 相速度 $v_p=1/\sqrt{\mu\varepsilon}$，无耗介质无色散；色散介质里相速度与群速度分离。
- 平均能流密度 $\langle\mathbf{S}\rangle = |E_0|^2/(2\eta)$，是天线辐射与链路预算的起点。
- 导电媒质引入复介电常数与**衰减常数** $\alpha$，波以 $e^{-\alpha z}$ 指数衰减。
- **趋肤深度** $\delta=\sqrt{2/(\omega\mu\sigma)}$：高频电流被挤到表面，决定高频电阻、屏蔽与选材。
- 判断"导体还是介质"看损耗正切 $\tan\delta=\sigma/(\omega\varepsilon)$，同一材料随频率升降可以在两档间切换。
- 平面波的能量在电场与磁场间振荡，功率流密度 $\langle\mathbf{S}\rangle$