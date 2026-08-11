---
title: 湍流简介与 Reynolds 应力
date: 2026-08-11
---

# 湍流简介与 Reynolds 应力

<div class="epigraph">
<p>我如今已老。待我死后升入天堂，有两件事我希望得到启示：其一是量子电动力学，其二是流体的湍流运动。</p>
<footer>—— 贺拉斯 · 兰姆（Horace Lamb，1932）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 流体力学 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么湍流是"最后的大奖"

大多数真实流动是湍流：飞机尾流、河流、大气、发动机燃烧室。它如此普遍，却如此顽固——兰姆的感叹（1932）到今天仍没有完全兑现。物理学家费曼称湍流为"经典物理学中最后一个未解决的大问题"。而这一切的起点，是一套惊人的观察：**Navier-Stokes 方程在雷诺数足够大时，解变得不唯一、混沌、对初值敏感**——确定性方程产生了不可预测的运动。<span class="marginnote">1895 年雷诺用染料在管中实验直观展示了层流到湍流的转捩，同年他写下 RANS 方程（平均 Navier-Stokes），把湍流变成"平均流 + 脉动"的统计理论——这是人类第一次系统地把"混沌运动"变成"可算的平均量"。湍流研究从"看得见的迷宫"变成"可逼近的统计"，全拜这套分解所赐。</span>

本章不求"解决湍流"，而是建立理解它的工具语言：**Reynolds 分解、Reynolds 应力、能量级联**。这三样东西是读懂一切湍流文献（包括 CFD 里的 RANS/LES，见《计算流体力学》）的钥匙。

## 1 湍流的本质：三个支柱

**核心概念：** 湍流（turbulence）有三个支柱性特征：

**多尺度**：同一时刻流场里挤满了从大到小的旋涡，大涡（与流场同尺度）到小涡（粘性可耗散的 Kolmogorov 尺度），跨度可达 4 个量级。
- **耗散**：湍流强烈耗散能量——它的动能必须靠外部持续供能（压差、剪切）来维持，一旦停供就迅速衰亡。
- **混沌**：对初始条件极端敏感，逐点精确预测无望，只能做统计描述。<span class="marginnote">"湍流是 Navier-Stokes 方程的解"这句话既对又无解——数学上"光滑解存在性"至今是千禧年问题（见《Navier-Stokes 方程》）。物理上，混沌把解的信息炸成统计噪声，这正是 Reynolds 平均的理论动机：不要逐点追踪，只求平均场的演化。</span>

这三点决定了研究范式：**湍流不是"随机噪声"，而是"确定性方程的统计涌现"**。它可被统计地刻画，却不可被逐点预言。

## 2 Reynolds 分解：把场拆成"平均 + 脉动"

Reynolds 分解（Reynolds decomposition）把任意量写成时间/系综平均与脉动之和：

$$\boldsymbol{v} = \bar{\boldsymbol{v}} + \boldsymbol{v}', \qquad p = \bar{p} + p', \qquad \overline{\boldsymbol{v}'} = 0$$

把分解代入 Navier-Stokes 后做平均（注意平均与导数可交换，但平均与乘积不可交换），得到**Reynolds 平均 Navier-Stokes 方程（RANS）**：

$$\rho\frac{\partial\bar{v}_i}{\partial t} + \rho\bar{v}_j\frac{\partial\bar{v}_i}{\partial x_j} = -\frac{\partial\bar{p}}{\partial x_i} + \frac{\partial}{\partial x_j}\left(\mu\frac{\partial\bar{v}_i}{\partial x_j} - \rho\overline{v_i' v_j'}\right)$$

**核心概念：平均后的方程与 Navier-Stokes 长得一样，只是多了最后一项** $-\rho\overline{v_i'v_j'}$。它来自非线性的"乘积平均 ≠ 平均乘积"：

$$\overline{v_i v_j} = \bar{v}_i\bar{v}_j + \overline{v_i'v_j'}$$

**Reynolds 应力（Reynolds stress）** $\tau_{ij}^{(R)} = -\rho\overline{v_i'v_j'}$ 就是"脉动场对平均场施加的额外应力"。<span class="marginnote">物理图像：湍流脉动像一群"活蹦乱跳的搬运工"，把动量从高速区搬到低速区（$\overline{v'u'}<0$ 时），等效于一层额外"湍流粘度"。它是平均流的"内在摩擦"——远比分子粘度大（湍流粘度可比分子粘度大几个量级），这就是为什么湍流混合极快、湍流边界层更抗分离（第 6 章）。</span>

**重点：RANS 方程不封闭。** 未知数超过方程数——解 $\bar{\boldsymbol{v}}$ 需要知道 $\overline{v_i'v_j'}$，而后者又需要更高阶矩……这就是**湍流封闭问题（closure problem）**：方程的链式结构永远差一环。所有湍流模型（Boussinesq 涡粘假设、$k$–$\varepsilon$、雷诺应力模型）都是"如何切断这条链"的工程抉择。

## 3 能量级联与 Kolmogorov 标度

湍流为什么必然是"多尺度 + 耗散"？答案是**能量级联（energy cascade）**。大尺度的涡（惯性区）从平均流吸收能量，通过涡拉伸（第 8 章）不断分裂成更小的涡，直到小到粘性尺度，才被 $\nu\nabla^2\boldsymbol{\omega}$ 耗散成热。

**核心概念：Kolmogorov（1941）假定，在充分发展湍流的惯性子区内，能谱只由耗散率 $\varepsilon$ 与波数 $k$ 决定**（量纲分析）得：

$$E(k) = C_K \varepsilon^{2/3} k^{-5/3}$$

$k^{-5/3}$ 谱是湍流理论最著名的定量预言，也是实验里被反复验证的"湍流签名"。<span class="marginnote">K41 的量纲论证只有三行：$E(k)$ 量纲 $\mathrm{m^3/s^2}$，$\varepsilon$ 量纲 $\mathrm{m^2/s^3}$，$k$ 量纲 $1/\mathrm{m}$，唯一组合就是 $E=C_K\varepsilon^{2/3}k^{-5/3}$。它把"湍流能谱长什么样"的难题压缩成一个通用常数 $C_K\approx1.5$。后来人们发现间歇性修正（$E\propto k^{-5/3-\mu/9}$）让问题更深，但 $k^{-5/3}$ 仍是湍流的象征。</span>从小尺度 $\eta=(\nu^3/\varepsilon)^{1/4}$（Kolmogorov 尺度）到大尺度 $L$，中间隔着 $\sim Re^{3/4}$ 个尺度的"链"。$Re=10^6$ 意味着链上有约 3 万级——直接数值模拟（DNS）要分辨最小涡，网格量 $\sim Re^{9/4}$，这就是湍流计算的"诅咒"（见《计算流体力学》）。

## 4 壁湍流：对数律与摩擦

靠近壁面的湍流（壁湍流，wall-bounded turbulence）有惊人普适的**分层结构**：最贴壁是粘性底层（$y^+<5$，分子粘性主导），然后是缓冲层，再是**对数律层**（$30<y^+<300$）：

$$u^+ = \frac{1}{\kappa}\ln y^+ + B, \qquad u^+=\frac{\bar u}{u_\tau},\; y^+=\frac{yu_\tau}{\nu}$$

$u_\tau=\sqrt{\tau_w/\rho}$ 是**摩擦速度**，$\kappa\approx0.41$ 是卡门常数。<span class="marginnote">对数律的普适性使它成为壁面湍流理论与工程的核心锚点：CFD 里的"壁面函数"、气象学的近地风廓线、管道摩阻公式全部建立在它上面。有趣的是，对数律至今没有严格推导，它是一个"实验+理论半猜"的普适结果——湍流里最成功的公式，恰恰是最不严谨的公式。</span>

**辨析｜易错点：** 湍流边界层的剖面**不是**抛物线，而是"粘性底层线性 + 对数层 + 外层尾迹律"的复合。把 Poiseuille 流抛物线剖面的直觉搬到湍流管流，会得到完全错误的摩擦阻力。湍流管流的平均剖面接近幂律/对数律，"平坦得多"。

## 5 公式解析：Reynolds 应力从哪冒出来

$$\tau_{ij}^{(R)} = -\rho\overline{v_i' v_j'}$$

- **第一步，把速度写成平均+脉动**：$v_i=\bar v_i+v_i'$，$v_j=\bar v_j+v_j'$。
- **第二步，展开乘积**：$v_iv_j = \bar v_i\bar v_j + \bar v_iv_j' + v_i'\bar v_j + v_i'v_j'$。交叉项在平均后消失（因为 $\overline{v'}=0$），只剩 $\bar v_i\bar v_j+\overline{v_i'v_j'}$。
- **第三步，进入动量方程**：Navier-Stokes 里的对流项 $\rho v_j\partial v_i/\partial x_j$ 平均后多出 $-\rho\overline{v_i'v_j'}$，它对平均流起"应力"作用。
- **第四步，为什么是"应力"**：一个流体质点以脉动速度 $v_j'$ 穿过面元时，携带的 $x_i$ 向动量通量就是 $v_i'v_j'$——这正是"动量通量"的定义，与分子输运（粘度）同构。**湍流应力 = 脉动场对平均场的动量输运**。<span class="marginnote">类比第 13 章：分子输运系数 $\mu\sim\rho\ell v_{\mathrm{th}}$；湍流"输运系数"$\mu_T\sim\rho\ell_T u'$，其中 $\ell_T$ 是混合长度。普朗特 1925 年据此提出"混合长度理论"，把 Reynolds 应力模型化为 $\overline{u'v'}\approx-\ell_m^2|\partial\bar u/\partial y|(\partial\bar u/\partial y)$——一把粗糙却极其有用的工程钥匙，至今还在边界层程序里服役。</span>

## 6 小结

- 湍流三大支柱：**多尺度、强耗散、混沌**；它是"确定性方程的统计涌现"。
- **Reynolds 分解** $\boldsymbol{v}=\bar{\boldsymbol{v}}+\boldsymbol{v}'$ 把方程拆成平均流 + 脉动，RANS 中多出 **Reynolds 应力** $-\rho\overline{v_i'v_j'}$。
- **封闭问题**：脉动的高阶矩永远缺一环，湍流模型都是"如何切断链条"的抉择。
- **能量级联**与 Kolmogorov 标度 $E(k)\propto k^{-5/3}$：能量从大涡到小涡传递，最终被粘性耗散。
- 壁湍流有普适的**对数律**分层；湍流剖面不是抛物线，摩擦阻力估算不能借用层流直觉。

在下一节，我们回答一个更根本的问题：湍流是从哪里"长出来"的？当层流失稳、扰动增长，流动从有序走向混沌——这就是**流体稳定性**。
