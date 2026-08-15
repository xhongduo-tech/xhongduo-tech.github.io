---
title: 结构动力学有限元（模态/瞬态）
date: 2026-08-07
---

# 结构动力学有限元（模态/瞬态）

<div class="epigraph">
<p>自然界的振动，是最古老也最深刻的音乐。</p>
<footer>—— 伽利略 · 伽利雷（Galileo Galilei，关于摆的等时性）</footer>
</div>

<div class="article-byline">
<p>第六级 · 计算力学与有限元方法 ｜ Bathe《Finite Element Procedures》第9–11章 ｜ 2026-08-07</p>
</div>

## 为什么从结构动力学开始

前面的所有单元都在解决同一件事：结构在**静力**下的响应。但真实世界从不静止——地震晃动建筑，风振撼动桥梁，冲击波击中舰船。当载荷随时间快速变化时，**惯性力**不能再被忽略，平衡方程升级为动力学方程。有限元在动力学里要回答两类问题：结构**以什么频率、什么形态振动**（模态分析），以及结构在给定载荷历史下**如何随时间响应**（瞬态分析）。这一节你会看到：静态有限元的全部积累（$\boldsymbol{K}$、$\boldsymbol{M}$ 的组装）在这里直接复用，而新加入的只有两件事——质量矩阵与时间积分。

## 1 动力方程：惯性力进场

在静力平衡 $\boldsymbol{K}\boldsymbol{u} = \boldsymbol{F}$ 中加入惯性力（质量 × 加速度）与阻尼力，得到**结构动力方程**：

$$
\boldsymbol{M} \ddot{\boldsymbol{u}}(t) + \boldsymbol{C} \dot{\boldsymbol{u}}(t) + \boldsymbol{K} \boldsymbol{u}(t) = \boldsymbol{F}(t)
$$

其中 $\boldsymbol{M}$ 是**质量矩阵**，$\boldsymbol{C}$ 是**阻尼矩阵**，$\boldsymbol{K}$ 就是我们熟知的刚度矩阵。<span class="marginnote">动力方程是牛顿第二定律 $\boldsymbol{F} = m\boldsymbol{a}$ 的离散化身：每个节点上的合力 = 节点质量 × 节点加速度。静力只是 $\ddot{\boldsymbol{u}} = \dot{\boldsymbol{u}} = 0$ 的特例——这样看，静力分析是动力学方程被「冻结」后的一个快照。</span>

**关键结论：动力学引入了时间维，未知量从「一个位移向量」变成「一条位移时间历程」。** 解法的核心不再是单纯解代数方程，而是处理「半离散」问题：空间已由有限元离散，时间仍是连续变量，需要专门的**时间积分**来处理。

## 2 质量矩阵：一致质量与集中质量

$\boldsymbol{K}$ 的组装我们已熟，$\boldsymbol{M}$ 的组装同样由形函数完成。单元质量矩阵：

$$
\boldsymbol{m}^e = \int_{\Omega^e} \rho\, \boldsymbol{N}^{\mathsf{T}} \boldsymbol{N} \, d\Omega
$$

其中 $\boldsymbol{N}$ 是形函数矩阵，$\rho$ 是密度。这种由同一套形函数构造的质量矩阵叫**一致质量矩阵（consistent mass matrix）**——质量与刚度「同源」，能量关系最自洽。<span class="marginnote">「一致质量 = 用形函数把质量分布到节点上」——它与刚度的组装方式完全平行，所以程序实现几乎零额外成本。但一致质量矩阵是非对角的，给显式时间积分带来不便。</span>

工程上更常用**集中质量矩阵（lumped mass matrix）**：把单元总质量按一定规则分配到对角线节点上，得到一个**对角矩阵**。对角质量矩阵在显式时间积分里价值巨大——解 $\boldsymbol{M}\ddot{\boldsymbol{u}}$ 不需要解方程组，逐点相除即可。<span class="marginnote">「对角 ⇒ 无需求逆 ⇒ 显式时间积分每步只做矩阵乘法」是集中质量最核心的价值。代价是高频模态的精度略降，但工程上常可接受——这是「效率换一点精度」的典型权衡。</span>

## 3 模态分析：特征值问题

**模态分析（modal analysis）** 回答：结构自由振动（无外载、无阻尼）的固有频率与振型。设 $\boldsymbol{u}(t) = \boldsymbol{\phi} e^{i\omega t}$ 代入无阻尼自由振动方程，得**广义特征值问题**：

$$
\boldsymbol{K} \boldsymbol{\phi} = \omega^2 \boldsymbol{M} \boldsymbol{\phi}
$$

特征值 $\lambda_i = \omega_i^2$ 给出第 $i$ 阶**固有频率** $\omega_i$（或 $f_i = \omega_i/2\pi$），特征向量 $\boldsymbol{\phi}_i$ 给出第 $i$ 阶**振型**。<span class="marginnote">模态分析是动力学里最重要的「体检」：它告诉你结构「天生爱以什么频率振动」。工程设计最怕的就是外载频率恰好接近某阶固有频率——共振。汽车避开发动机转速、桥梁避开风振、楼宇避开地震主频，本质都在做同一件事：让固有频率远离激励频率。</span>

特征向量还满足**正交性**：$\boldsymbol{\phi}_i^{\mathsf{T}} \boldsymbol{M} \boldsymbol{\phi}_j = \delta_{ij}$（质量归一化），$\boldsymbol{\phi}_i^{\mathsf{T}} \boldsymbol{K} \boldsymbol{\phi}_j = \omega_i^2 \delta_{ij}$。正交性使振型可以作为一组「基」，把高维动力方程解耦——这就是**振型叠加法（modal superposition）**的原理。

## 4 瞬态分析：时间积分

**瞬态分析（transient analysis）** 直接数值求解动力方程，得到完整的时程响应 $\boldsymbol{u}(t)$。核心工具是**时间积分（time integration）**：把连续时间离散成步长 $\Delta t$，逐时刻推进。最著名的两类：

**Newmark 方法（隐式）**：

$$
\boldsymbol{u}_{n+1} = \boldsymbol{u}_n + \Delta t\, \dot{\boldsymbol{u}}_n + \frac{\Delta t^2}{2}\left[(1-2\beta)\ddot{\boldsymbol{u}}_n + 2\beta \ddot{\boldsymbol{u}}_{n+1}\right]
$$

当 $\beta \ge 1/4$ 时**无条件稳定**——$\Delta t$ 可以取大，适合低频主导的长时程问题（地震响应）。代价是每一步要解方程组。<span class="marginnote">Newmark 的 $\gamma, \beta$ 两个参数控制稳定与精度：$\gamma = 1/2$ 保证二阶精度、$\beta = 1/4$ 得平均加速度法（无条件稳定）。Bathe 版教材对这套参数的血缘关系讲得最透——它本质是一族「隐式梯形」的推广。</span>

**中心差分法（显式）**：

$$
\ddot{\boldsymbol{u}}_n = \frac{\boldsymbol{u}_{n+1} - 2\boldsymbol{u}_n + \boldsymbol{u}_{n-1}}{\Delta t^2}
$$

把加速度代入动力方程后，只要 $\boldsymbol{M}$ 是对角矩阵，$\boldsymbol{u}_{n+1}$ 就能**直接解出而无需解方程组**。但显式是**条件稳定**：$\Delta t \le \Delta t_{\text{cr}}$，临界步长与最小单元尺寸正相关。冲击、碰撞这类「高频 + 大量小单元」的问题，显式方法虽步数多，但每步极便宜，仍是首选。<span class="marginnote">「隐式解大步长、显式解小步长但每步便宜」——选择隐式还是显式，是动力学计算最重要的战略决策。Abaqus/Standard 是隐式、Abaqus/Explicit 是显式，软件的分家就是这条路线的工程固化。</span>

## 5 公式解析：两自由度弹簧质量系统的固有频率

一个两自由度系统：两个质量 $m$ 串联两个刚度同为 $k$ 的弹簧（左侧固定）。质量矩阵与刚度矩阵：

$$
\boldsymbol{M} = m\begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}, \qquad \boldsymbol{K} = k\begin{bmatrix}2 & -1\\-1 & 1\end{bmatrix}
$$

**第一步，列特征值方程**：$\det(\boldsymbol{K} - \omega^2 \boldsymbol{M}) = 0$：

$$
\det \begin{bmatrix} 2k - m\omega^2 & -k \\ -k & k - m\omega^2 \end{bmatrix} = 0
$$

**第二步，展开求频率**：$(2k - m\omega^2)(k - m\omega^2) - k^2 = 0$，整理得 $m^2\omega^4 - 3mk\omega^2 + k^2 = 0$，解得：

$$
\omega_1^2 = \frac{(3 - \sqrt{5})k}{2m}, \qquad \omega_2^2 = \frac{(3 + \sqrt{5})k}{2m}
$$

数值上 $\omega_1 \approx 0.618\sqrt{k/m}$，$\omega_2 \approx 1.618\sqrt{k/m}$——比值恰是黄金分割的倒数和它本身，振型分别对应「同向缓摆」与「反向急振」。<span class="marginnote">这个两自由度手算题的价值在于示范「特征值 → 频率 → 振型」的完整链条。真实结构动辄上万自由度，$\boldsymbol{K},\boldsymbol{M}$ 巨型化后不能直接展开行列式，必须用迭代法（子空间迭代、Lanczos）只求前若干阶——但原理与这个 $2\times2$ 例子完全一致。</span>

**第三步，验振型**：把 $\omega_1^2$ 代回 $(\boldsymbol{K} - \omega^2\boldsymbol{M})\boldsymbol{\phi} = 0$，得第一阶振型 $\boldsymbol{\phi}_1 \approx [0.618, 1]^{\mathsf{T}}$——两质量同向位移，靠近墙的质量动得少。第二阶 $\boldsymbol{\phi}_2 \approx [-1.618, 1]^{\mathsf{T}}$，反向运动。

## 6 动力学方法选型

**阻尼矩阵从哪来**：结构阻尼的物理机制复杂（材料内耗、连接摩擦、空气阻尼），工程上几乎从不从单元层面构造 $\boldsymbol{C}$，而是用两种等效手段：**瑞利阻尼** $\boldsymbol{C} = \alpha\boldsymbol{M} + \beta\boldsymbol{K}$（用两个系数配出给定两阶阻尼比），或直接指定各阶模态的阻尼比（振型叠加时逐阶施加）。理解「阻尼是模型化的、不是推导出来的」，是动力学建模的重要心态。

**模态截断的工程准则**：实际结构的模态是无穷阶的，计算只能取前若干阶。工程经验：取到「激励频率范围内的所有模态」，且参与质量（effective mass）累计达到总质量的 90% 以上——这两条是判断「模态取够了没有」的标准答案。

**显式方法步长为什么卡在网格上**：显式方法的临界步长 $\Delta t_{\text{cr}}$ 与「应力波穿过最小单元所需的时间」同量级，$\Delta t_{\text{cr}} \approx L_{\min}/c$（$c$ 为波速）。所以最小单元尺寸 $L_{\min}$ 越小，步长越短，总步数越多——「加密一个局部的网格，会拖慢整个显式分析」，这是显式方法最反直觉的成本特征。

**静力、模态、瞬态：什么时候用哪个**：

| 问题类型 | 分析方法 | 关注量 |
| --- | --- | --- |
| 稳态承载 | 静力分析 | 应力、变形 |
| 固有特性 | 模态分析 | 频率、振型 |
| 地震/冲击响应 | 瞬态分析（隐式） | 时程位移、应力 |
| 高速碰撞 | 瞬态分析（显式） | 波传播、塑性区 |
| 周期载荷 | 谐响应分析 | 稳态振幅、共振 |
| 随机载荷 | 随机振动（PSD） | 统计响应 |

**「共振」的工程翻译**：激励频率等于某阶固有频率时，无阻尼系统振幅趋于无穷——这是理想化的数学结论。真实结构有阻尼，振幅被限制在一个有限大值，但可能比静力响应大几十上百倍。所以工程准则不是「避开共振」（那不可能），而是「让共振幅值足够小」——靠阻尼、靠质量分布、靠避开激励主频。

## 7 小结

- **动力方程** $\boldsymbol{M}\ddot{\boldsymbol{u}} + \boldsymbol{C}\dot{\boldsymbol{u}} + \boldsymbol{K}\boldsymbol{u} = \boldsymbol{F}(t)$ 是静力平衡的动力学推广。
- **质量矩阵**：一致质量（非对角、能量自洽）与集中质量（对角、显式友好）两种构造。
- **模态分析**：求解 $\boldsymbol{K}\boldsymbol{\phi} = \omega^2\boldsymbol{M}\boldsymbol{\phi}$