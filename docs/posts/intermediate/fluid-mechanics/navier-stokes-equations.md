---
title: Navier-Stokes 方程
date: 2026-08-07
---

# Navier-Stokes 方程

<div class="epigraph">
<p>宇宙这部大书是用数学语言写成的。</p>
<footer>—— 伽利略 · 伽利莱（Galileo Galilei，《试金者》，1623）</footer>
</div>

<div class="article-byline">
<p>第二级 · 流体力学 ｜ Batchelor Ch. 5 ｜ 2026-08-07</p>
</div>

## 为什么 Navier-Stokes 方程是全场主角

有了运动学（怎么动）和应力张量（什么力），把它们用牛顿第二定律拼起来，得到的就是**Navier-Stokes 方程**——流体力学里最重要的一个方程。它描述从茶杯里的旋转到大气环流、从血液流动到机翼升力的几乎一切真实流体运动。它同时也是著名的千禧年问题之一：其光滑解的存在性至今悬而未决，被 Clay 研究所悬赏一百万美元。<span class="marginnote">2000 年 Clay 研究所把"Navier-Stokes 方程解的存在性与光滑性"列为七大千禧年问题之一。三百年过去，我们连它的基本数学性质都未能完全证明——这恰说明这个方程同时是工程学的骄傲与数学的深井。</span>

本章的目标是把它**推导出来、读懂每一项、并学会无量纲化**。因为后面所有的专题——精确解、边界层、湍流、CFD——都是在这一个方程的各个"极端工况"里做文章。

## 1 从牛顿第二定律到动量方程

对流体中一个物质微团，牛顿第二定律 $\rho\, \frac{D\boldsymbol{v}}{Dt} = \boldsymbol{f}$ 中，作用力包括：体积力（重力 $\rho\boldsymbol{g}$）与表面力（应力张量散度 $\nabla\cdot\boldsymbol{\sigma}$）。于是：

$$\rho \frac{D\boldsymbol{v}}{Dt} = \rho\boldsymbol{g} + \nabla\cdot\boldsymbol{\sigma}$$

这是**动量方程（momentum equation）的通用形式**，对任何连续介质（流体、固体、软物质）都成立，尚未掺入任何流体特有的假设。<span class="marginnote">注意我们用的是物质导数——方程左侧跟随的是"同一团流体"，而不是"同一空间点"。这正是上一章物质导数的用武之地：$\rho\frac{D\boldsymbol{v}}{Dt}$ 是牛顿第二定律的正确翻译。</span>接下来把上一章的应力分解与本构关系代入，就得到具体形式。

## 2 不可压缩 Navier-Stokes 方程

对不可压缩牛顿流体（$\nabla\cdot\boldsymbol{v}=0$，$\mu$ 常数），代入 $\sigma_{ij}=-p\delta_{ij}+2\mu e_{ij}$ 后整理，得到：

$$\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{v}\cdot\nabla)\boldsymbol{v}\right) = -\nabla p + \mu\nabla^2\boldsymbol{v} + \rho\boldsymbol{g}, \qquad \nabla\cdot\boldsymbol{v}=0$$

这是流体力学中最著名的方程组，写成分量形式为 $\rho\big(\frac{\partial v_i}{\partial t}+v_j\frac{\partial v_i}{\partial x_j}\big) = -\frac{\partial p}{\partial x_i}+\mu\nabla^2 v_i + \rho g_i$。<span class="marginnote">方程左侧是惯性（当地项+非线性迁移项），右侧依次是压强梯度、粘性扩散、重力。注意粘性项只有 $v_i$ 的拉普拉斯，没有 $p$——压强在不可压问题里由 $\nabla\cdot\boldsymbol{v}=0$ 这个约束"反推"出来，像一个拉格朗日乘子，这是数值求解的大麻烦（见《计算流体力学》）。</span>

**核心概念：** 四个物理效应——惯性、压强、粘性、外力——在此相互平衡。方程由纳维（C.-L. Navier，1822，从分子模型）与斯托克斯（G. G. Stokes，1845，从连续介质假设）先后独立导出，因此得名。

## 3 欧拉方程：粘性消失的极限

令 $\mu=0$，得到**欧拉方程（Euler equations）**：

$$\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + (\boldsymbol{v}\cdot\nabla)\boldsymbol{v}\right) = -\nabla p + \rho\boldsymbol{g}$$

无粘流（理想流体）由它描述。<span class="marginnote">欧拉 1755 年就写下了这个方程，比 Navier-Stokes 早近百年。但无粘流有个致命悖论：按它计算绕过圆柱的阻力恒为零（达朗贝尔佯谬，见《不可压缩无粘流动与势流理论》）——真实流体的阻力全靠粘性，欧拉方程因此无法自圆其说。</span>

**重点：欧拉方程不是"简化了的 Navier-Stokes"，而是"另一种极限"。** 数学上粘性项是二阶导数（$k^2$ 在傅里叶空间放大），去掉它方程组从抛物型变成双曲型，解的性质完全不同；物理上无粘流允许滑移边界，真实流必须满足无滑移条件。把两者混淆，是初学阶段最常见的错误。

## 4 无量纲化：雷诺数的登场

Navier-Stokes 方程最深刻的结构，在把它写成**无量纲形式**时才显露。取特征速度 $U$、特征长度 $L$，令 $\hat{\boldsymbol{x}}=\boldsymbol{x}/L$、$\hat{\boldsymbol{v}}=\boldsymbol{v}/U$、$\hat{t}=tU/L$、$\hat{p}=p/(\rho U^2)$，代入方程得：

$$\frac{\partial \hat{\boldsymbol{v}}}{\partial \hat{t}} + (\hat{\boldsymbol{v}}\cdot\hat{\nabla})\hat{\boldsymbol{v}} = -\hat{\nabla}\hat{p} + \frac{1}{Re}\,\hat{\nabla}^2\hat{\boldsymbol{v}}$$

其中唯一留下的参数是**雷诺数（Reynolds number）**：

$$Re = \frac{\rho U L}{\mu} = \frac{UL}{\nu}$$

$\nu=\mu/\rho$ 称为**运动粘度**。<span class="marginnote">雷诺数衡量"惯性力与粘性力之比"。Re 很小的流（如细菌游动）粘性统治，惯性项可丢（见《低雷诺数流动（Stokes 流）》）；Re 很大的流（如飞机、河流）惯性统治，粘性只在极薄的边界层内起作用（见《粘性流动与边界层》）。同一个方程，两个极端，两套世界观。</span>

**重点：无量纲化的意义是"同样的方程描述不同的物理"**。几何相似（同一 $Re$）的两组流动，即使介质、尺寸、速度完全不同，无量纲解也完全相同——这就是风洞实验、模型试验的理论依据，也是流体力学从个别问题走向普适规律的钥匙。

## 5 能量与耗散

把动量方程点乘 $\boldsymbol{v}$ 并积分，得到能量平衡：动能的损失率为

$$\Phi = 2\mu\, e_{ij}e_{ij}$$

这是**粘性耗散率（viscous dissipation）**——单位体积流体的机械能不可逆地转化为热。<span class="marginnote">$\Phi>0$ 恒成立（应变率张量平方的正定性），体现了热力学第二定律：粘性只能耗能，不能供能。这正是湍流最终"衰亡"的物理原因——大涡把动能级联到小涡，小涡最后靠粘性耗散成热，见《湍流简介与 Reynolds 应力》。</span>机械能与热能的这个耦合，是流体能量方程中"源项"的物理内容，也是《流体中的热传导与扩散》的出发点。

## 6 公式解析：Navier-Stokes 方程一项一项读

$$\rho\left(\frac{\partial \boldsymbol{v}}{\partial t} + \underbrace{(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}}_{\text{非线性项}}\right) = \underbrace{-\nabla p}_{\text{压强}} + \underbrace{\mu\nabla^2\boldsymbol{v}}_{\text{粘性}} + \rho\boldsymbol{g}$$

- **第一步，左侧的 $\rho\partial\boldsymbol{v}/\partial t$**：当地惯性——固定点的流速随时间的变化，相当于"这里的水在加速"。
- **第二步，左侧的 $\rho(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}$**：迁移惯性——流体流进高压/低压区或弯道时，即使流场定常，"跟着走的质点"也会加速。这就是为什么在文丘里管收缩处流速自动加快（连续性方程）并导致压强降低（伯努利，见势流章节）。
- **第三步，右侧的 $-\nabla p$**：压强梯度力，流体总是从高压被推向低压；$\nabla p$ 前的负号与"压"的方向一致。
- **第四步，右侧的 $\mu\nabla^2\boldsymbol{v}$**：粘性扩散——速度差在分子层面被抹平，等价于 $\nu$ 个"速度场"的温度扩散方程 $\partial_t\boldsymbol{v}=\nu\nabla^2\boldsymbol{v}$。粘性让相邻流层"团结一致"。

**直觉总纲：这个方程说的是——"一団流体所受的净力 = 它自身惯性改变的理由"。** 所有流体力学问题，本质都是这四个词在不同 $Re$ 下的角力。

## 7 数值算例：雷诺数一把尺，量出三个世界

雷诺数 $Re=\rho UL/\mu=UL/\nu$ 的价值在于：它让你用**一个数**判断一场流动的"性格"。随手算三组真实流动：

| 流动 | 特征速度 $U$ | 特征长度 $L$ | $Re$ |
| --- | --- | --- | --- |
| 游泳的细菌 | $30\,\mathrm{\mu m/s}$ | $1\,\mathrm{\mu m}$ | $\sim10^{-5}$ |
| 输油管道 | $1\,\mathrm{m/s}$ | $0.1\,\mathrm{m}$ | $\sim10^{3}$ |
| 客机巡航 | $250\,\mathrm{m/s}$ | $10\,\mathrm{m}$ | $\sim10^{8}$ |

同一套 Navier-Stokes 方程，$Re$ 从 $10^{-5}$ 到 $10^{8}$ 跨了十三个数量级，物理却彻底不同：细菌靠粘性"游"（见《低雷诺数流动》），油管里有层流/湍流之分，客机的粘性只蜷缩在边界层里逞强（见《粘性流动与边界层》）。**把 $Re$ 先算出来、定性格、再选工具——这是流体力学解决问题的第一动作。**

**辨析｜易错点：** 雷诺数必须用**特征尺度**来算，同一场流动换了尺度结论可以相反。人体血液在主动脉里 $Re\sim10^3$、在毛细血管里 $Re\sim10^{-2}$——"这个流动是不是低雷诺数"没有全局答案，必须说清"以什么尺度衡量"。另一个常见错误是把 $L$ 选错：管流取管径而不是管长，绕流取特征宽度而不是展长。

再把"选错特征量"的代价量化一遍。估算一辆汽车（$L=5\,\mathrm{m}$，$U=30\,\mathrm{m/s}$）在空气中的雷诺数：$\nu=1.5\times10^{-5}\,\mathrm{m^2/s}$，$Re=UL/\nu=30\times5/1.5\times10^{-5}=10^{7}$——于是边界层厚度 $\delta\sim L/\sqrt{Re}\approx5/3000\approx1.6\,\mathrm{mm}$。这 1.6 毫米的薄层，决定了整辆车的摩擦阻力与是否分离，这正是"高 $Re$ 不等于无粘"最直接的数字证明。

## 8 无量纲数速查表

无量纲数是流体力学的"指纹"，每个数回答一个问题。记住这张表，读文献时就有了翻译器：

| 无量纲数 | 定义 | 回答的问题 | 出场章节 |
| --- | --- | --- | --- |
| 雷诺数 $Re$ | $\frac{UL}{\nu}$ | 惯性 vs 粘性 | 本章、边界层 |
| 弗劳德数 $Fr$ | $\frac{U}{\sqrt{gL}}$ | 惯性 vs 重力 | 表面波 |
| 马赫数 $Ma$ | $\frac{U}{c_s}$ | 流速 vs 声速 | 可压缩流 |
| 佩克莱数 $Pe$ | $\frac{UL}{D}$ | 对流 vs 扩散 | 热传导与扩散 |
| 斯特劳哈尔数 $St$ | $\frac{fL}{U}$ | 非定常 vs 对流 | 涡脱落 |
| 韦伯数 $We$ | $\frac{\rho U^2 L}{\sigma}$ | 惯性 vs 表面张力 | 毛细波 |

## 9 常见误区对照表

| 误区说法 | 正确理解 |
| --- | --- |
| "欧拉方程是 Navier-Stokes 的近似" | 是"无粘极限"，方程类型与边界条件都变 |
| "不可压就是密度处处为常数" | 是"沿质点密度不变"，等价于 $\nabla\cdot\boldsymbol{v}=0$ |
| "$Re$ 大则粘性可忽略" | 粘性只在边界层内起作用，不能全场忽略 |
| "压强由状态方程给出" | 不可压流中压强由约束 $\nabla\cdot\boldsymbol{v}=0$ 反推 |
| "耗散率 $\Phi$ 可正可负" | $\Phi=2\mu e_{ij}e_{ij}\ge0$ 恒成立，只能耗能 |

## 10 小结

- **动量方程通用式** $\rho\frac{D\boldsymbol{v}}{Dt}=\rho\boldsymbol{g}+\nabla\cdot\boldsymbol{\sigma}$ 对任何连续介质成立；代入牛顿本构得 Navier-Stokes。
- **不可压缩 Navier-Stokes**：$\rho(\partial_t\boldsymbol{v}+\boldsymbol{v}\cdot\nabla\boldsymbol{v})=-\nabla p+\mu\nabla^2\boldsymbol{v}+\rho\boldsymbol{g}$，外加 $\nabla\cdot\boldsymbol{v}=0$。
- **欧拉方程**是无粘极限（$\mu=0$），抛物型变双曲型，边界条件从无滑移变滑移，勿与 Navier-Stokes 混用。
- **无量纲化后方程只含一个参数**：雷诺数 $Re=\rho UL/\mu$，它决定流动的所有定性行为。
- **粘性耗散率** $\Phi=2\mu e_{ij}e_{ij}\geq 0$ 恒成立，机械能单向损失。

在下一节，我们把方程放进最简单的几何里：平行板与圆管中的定常流动，粘性项独大、非线性和压强都有显式解——这就是**Couette 与 Poiseuille 精确解**。
