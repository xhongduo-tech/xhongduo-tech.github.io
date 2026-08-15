---
title: 粘滞流动与流体力学输运
date: 2026-08-07
---

# 粘滞流动与流体力学输运

<div class="epigraph">
<p>水是自然界一切动力的源泉。</p>
<footer>—— 列奥纳多 · 达 · 芬奇（Leonardo da Vinci）</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ de Groot &amp; Mazur《Non-Equilibrium Thermodynamics》第12章 ｜ 2026-08-07</p>
</div>

## 为什么从粘滞流动开始

前几讲处理的是**标量**与**矢量**输运：热量与物质沿梯度方向流动。但动量还有第三种输运方式——它沿**速度梯度**方向流动，表现为流体内部的「内摩擦」，也就是**粘滞（viscosity）**。当两层流体相对滑动时，快速层拖拽慢速层、慢速层牵制快速层，动量被一层层「扩散」下去。

粘滞流动是熵产生的又一重要来源，也是连接非平衡热力学与流体力学（Navier-Stokes 方程）的桥梁。本讲将展示：从动量平衡方程出发，加上「粘滞应力正比于形变率」的牛顿粘滞定律，就能完整推出流体力学方程组；而 Curie 原理（第3篇）精确限定了粘滞系数只有两个独立分量。

## 1 压力张量与粘滞应力

第1讲里动量平衡方程含压力张量 $\mathbf{P}$，其对角项是流体静压强 $p$，非对角与偏离部分构成**粘滞应力张量** $\mathbf{\Pi}$：

$$
\mathbf{P} = p\mathbf{1} + \mathbf{\Pi}
$$

在局部转动下，$\mathbf{\Pi}$ 必须与形变率（速度梯度的对称部分）相联系。对流体的**形变率张量**：

$$
\nabla\mathbf{v}^{\,s} = \frac{1}{2}\left(\nabla\mathbf{v} + (\nabla\mathbf{v})^T\right)
$$

牛顿粘滞定律（线性区）断言粘滞应力正比于形变率：$\mathbf{\Pi} = -2\eta\,\text{dev}(\nabla\mathbf{v}^s) - \eta_v(\nabla\cdot\mathbf{v})\mathbf{1}$，其中 $\eta$ 是**剪切粘滞系数**，$\eta_v$ 是**体粘滞系数**。<span class="marginnote">体粘滞描述流体整体压缩/膨胀时的内耗（体积变化中动能的不可逆耗散），剪切粘滞描述无体积变化下的形变耗散。对单原子气体，体粘滞理论预言为零，实验近似成立；对多原子气体（有内部自由度弛豫）体粘滞显著不为零。</span>

## 2 Curie 原理裁剪系数

回顾第3篇的居里原理：只有**同张量阶**的流与力才耦合。粘滞应力是二阶张量，其共轭力是形变率 $\nabla\mathbf{v}$。于是粘滞这一行只能与二阶张量的力耦合——而各向同性介质中唯一可用的二阶各向同性张量是单位张量 $\mathbf{1}$。

由此立刻得到两个结论：

1. 粘滞应力只能由形变率的**对称无迹部分**与**迹（散度）**两部分组成——前者配 $\eta$，后者配 $\eta_v$。
2. **独立的粘滞系数只有两个**：$\eta$ 与 $\eta_v$。任何更一般的各向同性线性响应都自动落进这两个参数。

这就是居里原理在流体力学里留下的指纹：它**不允许**粘滞应力与温度梯度（矢量）耦合，所以「剪切流直接生热」只能通过耗散发生，而不能通过一种「流致温差」的线性耦合发生。<span class="marginnote">这也解释了为什么 Navier-Stokes 方程只有两个粘滞参数，而非对称张量最一般的情形有三个（多出一个与转动耦合的项）。流体微观上是各向同性且无内部转动自由度的理想化，才让 Curie 原理干净地裁剪。</span>

## 3 熵产生与耗散函数

把粘滞项写进熵产生（第2篇）：

$$
\sigma = -\frac{1}{T}\,\mathbf{\Pi}:\nabla\mathbf{v} = \frac{1}{T}\left[2\eta\,\text{dev}(\nabla\mathbf{v}^s):\nabla\mathbf{v} + \eta_v(\nabla\cdot\mathbf{v})^2\right]
$$

展开后每一项都是正定的：剪切耗散 $2\eta(\text{dev}\,\nabla\mathbf{v}^s)^2 \ge 0$ 与体耗散 $\eta_v(\nabla\cdot\mathbf{v})^2 \ge 0$ 都要求 $\eta \ge 0$、$\eta_v \ge 0$。习惯上定义**耗散函数（dissipation function）** $\Phi = T\sigma = -\mathbf{\Pi}:\nabla\mathbf{v} \ge 0$，它度量单位体积内动能转化为内能的速率——流体力学的能量方程里，动能经 $\Phi$ 被「摩擦生热」。

**辨析｜易错点：** $\Phi$ 用的是形变率 $\nabla\mathbf{v}$ 而不是旋度 $\nabla\times\mathbf{v}$。纯刚体转动（无形变）不产生粘滞耗散——搅动一杯水使其整体匀速旋转，并不会因为它旋转而发热。粘滞只对**形变**做功，不对**转动**做功。

## 4 公式解析：牛顿粘滞定律

把牛顿粘滞定律完整写出并逐项拆解：

$$
\mathbf{\Pi} = -2\eta\,\left[\frac{1}{2}(\nabla\mathbf{v} + (\nabla\mathbf{v})^T) - \frac{1}{3}(\nabla\cdot\mathbf{v})\mathbf{1}\right] - \eta_v(\nabla\cdot\mathbf{v})\mathbf{1}
$$

- **$\nabla\mathbf{v} + (\nabla\mathbf{v})^T$**：形变率的原始量，对称部分捕捉「拉伸/剪切」，反对称部分对应转动（这里被消除）。
- **$-\frac{1}{3}(\nabla\cdot\mathbf{v})\mathbf{1}$**：减去迹，保证方括号内是无迹张量——纯膨胀的体积变化不应出现在剪切项里。$-\nabla\cdot\mathbf{v}$ 是局部膨胀率（压缩为负）。
- **第一项系数 $-2\eta$**：剪切粘滞系数乘形变率无迹部分。负号保证「流得快的一层被慢层拖慢」，即粘滞应力总是反抗形变。
- **第二项系数 $-\eta_v$**：体粘滞乘膨胀率。它只在 $\nabla\cdot\mathbf{v} \neq 0$（压缩或膨胀）时起作用。
- **物理含义**：粘滞应力的作用是把速度场的**不均匀性**转化成内能——这正是熵产生项 $-\mathbf{\Pi}:\nabla\mathbf{v}/T$ 的微观来源。

## 5 从平衡方程到 Navier-Stokes

把牛顿粘滞定律代回动量平衡方程（第1讲）：

$$
\rho\frac{\partial\mathbf{v}}{\partial t} + \rho(\mathbf{v}\cdot\nabla)\mathbf{v} = -\nabla p + \eta\nabla^2\mathbf{v} + \left(\frac{\eta}{3} + \eta_v\right)\nabla(\nabla\cdot\mathbf{v}) + \mathbf{F}
$$

这就是**Navier-Stokes 方程**——不可压缩情形（$\nabla\cdot\mathbf{v}=0$）下退化为 $\rho D\mathbf{v}/Dt = -\nabla p + \eta\nabla^2\mathbf{v} + \mathbf{F}$。它的每个部分都能回溯到非平衡热力学的构件：$-\nabla p$ 来自压力张量的各向同性部分，$\eta\nabla^2\mathbf{v}$ 来自剪切粘滞，多出的散度项来自体粘滞。<span class="marginnote">有趣的是：非平衡热力学并不是「从牛顿定律推出流体力学」，而是「从热力学第二定律的框架里，用 Curie 原理 + 线性响应把流体力学的形式唯一地定出来」。输运系数 $\eta,\eta_v$ 的具体数值，则要交给动理学理论（本专题第3篇玻尔兹曼方程）去计算。</span>

由此，热传导、扩散、粘滞三种输运在**同一个熵产生框架**下统一：各自对应一组流与力，各有各的输运系数，而系数的微观计算由统计力学提供。这也是为什么把流体力学放进「非平衡统计物理」这门课——它只是线性响应理论的一个宏观侧面。

## 6 从熵产生看流体力学的耗散结构

粘滞耗散不是抽象的数学项，它直接决定流体力学中的能量预算。把耗散函数 $\Phi = -\mathbf{\Pi}:\nabla\mathbf{v}$ 用到几个典型流动，就能看清「熵产生在何处发生」：

**库埃特流（Couette flow）**：两平行板相距 $h$，上板以速度 $U$ 运动。速度线性分布 $\mathbf{v} = (Uy/h)\hat{x}$，形变率常数，耗散函数：

$$
\Phi = \eta\left(\frac{U}{h}\right)^2
$$

**泊肃叶流（Poiseuille flow）**：圆管中的压力驱动流，速度是抛物线分布 $v_z(r) = v_{max}(1 - r^2/R^2)$。耗散不是均匀的——壁面附近速度梯度最大，熵产生率在壁面处最大、管心为零：

$$
\Phi(r) = 4\eta\left(\frac{v_{max}r}{R^2}\right)^2
$$

把 $\Phi$ 对管截面积分，得到单位长度管道的总耗散功率 $\propto \eta v_{max}^2$——这正是流体力学中「压降 × 流量」的能量损失。**粘滞耗散把流动的机械能不可逆地变成热**，这是管道输送、轴承润滑、血液流动中能量损耗的统一来源。

| 流动 | 速度分布 | 耗散位置 |
| --- | --- | --- |
| 库埃特流 | 线性 | 全空间均匀 |
| 泊肃叶流 | 抛物线 | 壁面集中 |
| 刚体转动 | 线性，但无剪切 | 零 |

**辨析｜易错点：** 刚体转动（$\mathbf{v} = \boldsymbol{\Omega}\times\mathbf{r}$）虽然速度不均匀，但形变率张量 $\nabla\mathbf{v}$ 的对称部分为零，粘滞耗散为零——**匀速旋转的流体不耗散**。判断是否耗散，看的是「形变率」而非「速度差」。

## 7 小结

- **粘滞应力** $\mathbf{\Pi}$ 是压力张量偏离各向同性压强 $p$ 的部分，是动量输运的不可逆源头。
- Curie 原理保证各向同性流体只有**两个**独立粘滞系数：剪切 $\eta$ 与体粘滞 $\eta_v$。
- 熵产生中的粘滞项 $-\mathbf{\Pi}:\nabla\mathbf{v}/T \ge 0$ 正定，导出耗散函数 $\Phi = -\mathbf{\Pi}:\nabla\mathbf{v}$。
- **牛顿粘滞定律**把 $\mathbf{\Pi}$ 与形变率线性相连：$\mathbf{\Pi} = -2\eta\,\text{dev}(\nabla\mathbf{v}^s) - \eta_v(\nabla\cdot\mathbf{v})\mathbf{1}$。
- 把牛顿粘滞定律代回动量平衡方程即得 **Navier-Stokes 方程**——热传导、扩散、粘滞三种输运在同一个熵产生框架下统一。
- 耗散函数 $\Phi = -\mathbf{\Pi}:\nabla\mathbf{v}\ge 0$ 度量动能转化为内能的速率；库埃特流全空间均匀耗散，泊肃叶流壁面集中耗散。
- **辨析**：匀速转动的流体不耗散——判断耗散看「形变率」而非「速度差」或旋度。

在下一节，我们把输运的唯象框架带到化学反应——一个不需要任何空间梯度的纯标量过程，看亲和势如何驱动系统向平衡弛豫，即**化学反应与弛豫现象**。