---
title: 地幔对流的流体动力学方程
date: 2026-08-07
---

# 地幔对流的流体动力学方程

<div class="epigraph">
<p>地幔不是一块冰冷坚固的石头，而是一锅以地质时间为尺度、缓缓翻涌的稠粥。</p>
<footer>—— 唐纳德 · 特科特（Donald Turcotte）</footer>
</div>

<div class="article-byline">
<p>第二级 · 地球动力学（Geodynamics） ｜ Turcotte & Schubert《Geodynamics》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从对流方程开始

前几节处理的都是「一维或弹性」的问题：热沿垂向传导、板块整块弯曲。但地球动力学最核心的引擎——**地幔对流**——是一个真正的流体动力学问题：软流层中的岩石在几百万年尺度上以每年厘米级的速度蠕动，把热量从地核搬运到地表，并拖动上方的板块。要用数学描述它，必须写出质量、动量、能量三组守恒方程，再根据地幔的特殊性质（黏度极高、流速极慢）大幅简化。这一节建立的方程组，是下一节瑞利数、以及对流形态讨论的物理基础；它也与第二级《流体力学》《偏微分方程》课程共享同一套语言。

读完本节后你会得到一个重要观念：**地幔对流的数学是「简化出来的」**——从完整的纳维-斯托克斯方程出发，删掉对地幔无意义的项（惯性），保留真正驱动流动的项（浮力、黏性），最后只剩下一个干净的线性方程。这种「从完整到极简」的建模思路，正是地球动力学与工程流体力学最大的不同。

## 1 控制方程：质量、动量与能量守恒

把地幔视为连续介质，速度场 $\mathbf{v}$、压力 $p$、温度 $T$、密度 $\rho$。三个基本方程如下。

**质量守恒（不可压缩近似）**：

$$\nabla \cdot \mathbf{v} = 0$$

**动量守恒（纳维-斯托克斯方程）**：

$$\rho \left( \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v}\cdot\nabla \mathbf{v} \right) = -\nabla p + \mu \nabla^2 \mathbf{v} + \rho \mathbf{g}$$

**能量守恒（温度方程）**：

$$\frac{\partial T}{\partial t} + \mathbf{v}\cdot\nabla T = \kappa \nabla^2 T + \frac{H}{\rho c}$$

这套方程看着与普通流体力学无异，但地幔的极端参数会让其中许多项自动消失——这正是下一节要做的「简化」工作。在着手简化前，先记住三个变量的典型值：$\mu \approx 10^{21}$ Pa·s、$\kappa \approx 10^{-6}$ m²/s、$\alpha \approx 3\times10^{-5}$ K⁻¹，它们是后面所有数量级估计的原料。<span class="marginnote">$\mu$ 是动力黏度（Pa·s），地幔典型值 $10^{21}$ Pa·s，比水的 $10^{-3}$ Pa·s 大 24 个数量级。$\kappa$ 是上一节的热扩散率约 $10^{-6}$ m²/s。</span>

## 2 三个关键近似：无穷普朗特数、布西内斯克、斯托克斯流

地幔对流之所以成为「简化版」流体力学，是因为三个近似层层递进。

**近似一：无穷普朗特数（infinite Prandtl number）。** 普朗特数定义为：

$$\mathrm{Pr} = \frac{\nu}{\kappa} = \frac{\mu}{\rho \kappa}$$

它衡量动量扩散对热量扩散的比值。水 $\mathrm{Pr}\approx7$，空气约 0.7，而地幔黏度 $10^{21}$ Pa·s、密度 4000 kg/m³、$\kappa=10^{-6}$ m²/s，得 $\mathrm{Pr} \approx 10^{23}$——动量方程中的惯性项 $\rho(\partial \mathbf{v}/\partial t + \mathbf{v}\cdot\nabla\mathbf{v})$ 相比黏性项 $\mu\nabla^2\mathbf{v}$ 完全可以忽略。<span class="marginnote">无穷普朗特数意味着：一旦浮力撤去，流动几乎立刻停止——地幔流没有「惯性余量」。这也让对流形态完全由边界条件与浮力分布决定，而非初始扰动。</span>

**近似二：布西内斯克近似（Boussinesq approximation）。** 密度只在浮力项里随温度变化，其他各处取常值：

$$\rho = \rho_0 [1 - \alpha (T - T_0)]$$

其中 $\alpha$ 是热膨胀系数（约 $3\times10^{-5}$ K⁻¹）。这样动量方程里的 $\rho \mathbf{g}$ 保留温度耦合，而惯性项与质量守恒里的密度都取常数。<span class="marginnote">布西内斯克近似把「热-流耦合」压缩到唯一一项：浮力。对地幔对流它是极好的近似，因为 $\alpha\Delta T \sim 3\times10^{-5}\times2000 \sim 0.06 \ll 1$，密度相对变化只有百分之几。</span>

**近似三：斯托克斯流（Stokes flow）。** 无穷普朗特数 + 布西内斯克后，动量方程退化为：

$$0 = -\nabla p + \mu \nabla^2 \mathbf{v} + \rho_0 \alpha (T - T_0) \mathbf{g}$$

这是**线性的斯托克斯方程**：惯性项消失，黏性力与压力梯度、浮力精确平衡。它的一大好处是**线性**——速度场与浮力成正比，解可以叠加，便于解析与数值求解。<span class="marginnote">斯托克斯流是流体力学里的经典极限情形，也出现在微型机器人的低雷诺数推进中。地幔流的雷诺数 $\mathrm{Re} \sim 10^{-20}$，是自然界最极端的斯托克斯流之一。</span>

## 3 二维流函数：把矢量方程降成标量方程

完整的三维矢量方程组对解析与计算都不友好。对二维问题引入**流函数（stream function）** $\psi$，令：

$$v_x = \frac{\partial \psi}{\partial z}, \qquad v_z = -\frac{\partial \psi}{\partial x}$$

质量守恒 $\partial v_x/\partial x + \partial v_z/\partial z = 0$ 自动满足。再对动量方程取旋度消去压力项，得到流函数形式的涡度方程：

$$\mu \nabla^4 \psi = \rho_0 \alpha g \frac{\partial T}{\partial x}$$

其中 $\nabla^4$ 是双调和算子。这个方程把「两个速度分量 + 压力 + 温度」的耦合系统，压缩成「一个流函数 + 温度」的**双调和-泊松耦合系统**，是地幔对流理论与数值模拟的基本工作形式。<span class="marginnote">取旋度消压力的技巧在第二级《流体力学》与《电动力学》里都很常见——它是把矢量场「内部旋转」提取出来的标准手段，涡度 $\omega = \nabla\times\mathbf{v}$ 是理解对流的天然语言。</span>

**边界条件的物理意义**：刚性上边界（地表）要求 $v_x = v_z = 0$，即 $\psi = \partial\psi/\partial z = 0$；而「自由滑移」边界只要求法向速度为零，$\psi = 0$、$\partial^2\psi/\partial z^2 = 0$。上边界是自由表面还是刚性表面，直接影响对流形态与板块运动——这是下一节讨论对流的几何边界前提。对温度而言，上边界取地表温度 $T=0$（或 $T_s$），下边界（核幔边界）取 $T_{CMB}$，两者之差 $\Delta T$ 正是瑞利数里的温差。

## 4 公式解析：涡度方程 ∇⁴ψ = (ραg/μ) ∂T/∂x

这条方程是地幔对流的「发动机方程」，分三步理解：

$$
\nabla^4 \psi = \frac{\rho_0 \alpha g}{\mu} \frac{\partial T}{\partial x}
$$

- **第一步，看驱动项 $\frac{\partial T}{\partial x}$**：水平温度梯度是浮力的来源——热的一侧密度小、向上浮，冷的一侧下沉。若温度只有垂向分层（$\partial T/\partial x = 0$），则方程齐次，无对流；**没有水平温差，就没有对流**。
- **第二步，看响应项 $\nabla^4 \psi$**：双调和算子描述黏性如何抵抗弯曲。流函数变化越剧烈，黏性耗散越大。$1/\mu$ 说明黏度越大，同样的浮力产生的流动越弱。
- **第三步，看比例系数 $\frac{\rho_0 \alpha g}{\mu}$**：它是「浮力产生流动」的效率。$\alpha$（热膨胀）越大、$g$ 越大、$\mu$ 越小，对流越强。把这个系数连同几何尺度、温度差组合成无量纲数，就得到下一节的主角——**瑞利数**。

取数感受：$\rho_0 = 4000$ kg/m³，$\alpha = 3\times10^{-5}$ K⁻¹，$g = 10$ m/s²，$\mu = 10^{21}$ Pa·s，得 $\rho_0\alpha g/\mu \approx 1.2\times10^{-18}$ (m²·s·K)⁻¹。这个数极小，但因为流函数量级也大（km²/s 量级），二者乘积仍然给出每秒厘米级的真实速度——**小系数配大尺度，正是地幔对流的数字特征**。<span class="marginnote">把真实尺度代入后会发现：速度 $u \sim 10^{-9}$ m/s ≈ 3 cm/yr，与 GPS 测得的板块运动速率完美吻合——这证明板块运动就是地幔对流的表面表达。</span>

## 5 从方程到无量纲：瑞利数的前奏

为了让方程不依赖具体尺度，引入无量纲变量。取特征长度 $d$（对流层厚度）、特征温差 $\Delta T$、特征时间 $d^2/\kappa$，能量方程化为：

$$\frac{\partial T'}{\partial t'} + \mathbf{v'}\cdot\nabla' T' = \nabla'^2 T'$$

动量方程则冒出无量纲组合：

$$\mathrm{Ra} = \frac{\rho_0 \alpha g \Delta T d^3}{\mu \kappa}$$

**瑞利数（Rayleigh number, $\mathrm{Ra}$）** 是浮力驱动与黏性、热扩散阻力之比，是地幔对流唯一的控制参数。$\mathrm{Ra}$ 超过临界值对流才会启动；对内部加热或底部加热的层，临界值不同。<span class="marginnote">地幔整体的 $\mathrm{Ra}$ 估计在 $10^6$–$10^8$ 量级，远超临界值——这意味着地幔对流高度湍流化、形态复杂。下一节将专门讨论瑞利数与对流形态的关系。</span>

代入地幔真实参数感受量级：$\rho_0 = 4000$ kg/m³，$\alpha = 3\times10^{-5}$ K⁻¹，$g = 10$ m/s²，$\Delta T = 2000$ K，$d = 2900$ km，$\mu = 10^{21}$ Pa·s，$\kappa = 10^{-6}$ m²/s：

$$\mathrm{Ra} = \frac{4000 \times 3\times10^{-5} \times 10 \times 2000 \times (2.9\times10^6)^3}{10^{21}\times10^{-6}} \approx 2\times10^7$$

**$10^7$ 量级的 $\mathrm{Ra}$ 意味着对流是强烈、非线性的**：流动不会是规则的层流胞元，而是动态演化的边界层+上升流+下沉流的复杂图案。但正因为 $\mathrm{Ra}$ 远超临界值（约 10³ 量级），我们才敢断言：**地幔一定在剧烈对流，板块运动只是这锅对流粥的表面泡沫**。<span class="marginnote">注意 $d^3$ 的敏感性：对流层厚度翻倍，$\mathrm{Ra}$ 翻 8 倍。这正是「下地幔若整体对流则 $\mathrm{Ra}$ 巨大、若分层对流则各层独立」争论的数值根源。</span>

**辨析｜易错点：** 别把斯托克斯流当成「无流动」。斯托克斯流是「无惯性但有流动」，速度场照样存在且非零；消去的是 $\partial\mathbf{v}/\partial t$ 与 $\mathbf{v}\cdot\nabla\mathbf{v}$，而不是 $\mathbf{v}$ 本身。另一个易错点是把 $\nabla^4$ 当成普通高阶导数——它包含 $\partial^4/\partial x^4 + 2\partial^4/(\partial x^2\partial z^2) + \partial^4/\partial z^4$ 四项，混偏导项常常被漏写，导致数值实现出错。<span class="marginnote">在第二级《偏微分方程数值解》中，$\nabla^4 \psi$ 需要五点差分格式离散，交叉导数 $\partial^4/\partial x^2\partial z^2$ 是稳定性的关键。</span>

## 6 小结

- 地幔对流由**质量、动量、能量三方程**描述，经三个近似简化：无穷普朗特数、布西内斯克、斯托克斯流。
- 普朗特数 $\mathrm{Pr} = \nu/\kappa \approx 10^{23}$，惯性项可忽略。
- 布西内斯克近似把密度变化压缩进浮力项：$\rho = \rho_0[1-\alpha(T-T_0)]$。
- 二维流函数 $\psi$ 满足涡度方程 $\mu\nabla^4\psi = \rho_0\alpha g\,\partial T/\partial x$，是解析与数值的基本形式。
- 无量纲化后唯一的控制参数是**瑞利数** $\mathrm{Ra} = \rho_0\alpha g\Delta T d^3/(\mu\kappa)$。
- 地幔整体 $\mathrm{Ra} \approx 10^7$