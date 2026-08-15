---
title: 拉格朗日方程
date: 2026-08-07
---

# 拉格朗日方程

<div class="epigraph">
<p>从动能减势能，构造一个函数，它的一条方程就等价于牛顿的全部方程——分析力学的魔法，在这一节显形。</p>
<footer>—— 约瑟夫-路易 · 拉格朗日（Joseph-Louis Lagrange），1788</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 周衍柏《理论力学》分析力学部分 ｜ 2026-08-07</p>
</div>

## 为什么从拉格朗日方程开始

分析力学的目标是把牛顿方程改写为「只与能量有关、且对任何广义坐标都成立」的方程。**拉格朗日方程（Lagrange's equations）**实现这一点：定义**拉格朗日量** $L = T - V$（动能减势能），则系统运动由 $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = 0$ 决定。它比牛顿方程优雅：不显含约束力、不依赖坐标选择、自动适应对称性。这一节从达朗贝尔原理推导拉格朗日方程，并用两个例子演练。

## 1 拉格朗日方程的推导

从达朗贝尔原理出发（理想约束、完整约束、主动力有势）：

$$\sum_i(\boldsymbol{F}_i - m_i\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0$$

用广义坐标 $\delta\boldsymbol{r}_i = \sum_j\frac{\partial\boldsymbol{r}_i}{\partial q_j}\delta q_j$ 展开，经过「惯性力项化为动能导数、主动力项化为势能导数」的运算，得到**拉格朗日方程**：

$$\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = 0, \qquad j = 1, 2, \dots, s$$

其中 **拉格朗日量（Lagrangian）** $L = T - V$（动能 − 势能），$s$ 是自由度。

**重点：拉格朗日方程是「每个广义坐标一条方程」——方程数 = 自由度，不含约束力。** 只要写出系统的 $T$ 与 $V$，代入方程即可得到运动方程，无需画受力图、不用处理约束力。

## 2 用拉格朗日方程求解的步骤

解拉格朗日问题的四步：

1. **选广义坐标** $q_j$（自由度个数）；
2. **写动能 $T$ 与势能 $V$**（用广义坐标与广义速度表达）；
3. **构造拉格朗日量** $L = T - V$；
4. **代入拉格朗日方程**，得到运动微分方程。

**重点：拉格朗日方法的「能量化」流程——选坐标 → 写 $T$、$V$ → $L = T-V$ → 方程。** 全程不需要受力分析，约束力从未出现。<span class="marginnote">「为什么能绕开约束力」：理想约束力不做虚功（上节），被达朗贝尔原理剔除；拉格朗日量只含动能与势能（主动力有势时）。光滑约束、绳、铰链的力都不进入方程——需要求约束力时，再用拉格朗日乘子法（非完整约束/待定约束力的进阶方法）补回。</span>

## 3 公式解析：单摆的拉格朗日方程

用拉格朗日方法求平面单摆的运动方程（摆长 $l$、摆锤质量 $m$）。

$$
L = T - V = \frac{1}{2}ml^2\dot{\theta}^2 - mgl(1-\cos\theta)
$$

- **第一步，选广义坐标**：角度 $\theta$（自由度 1）。
- **第二步，写能量**：动能 $T = \frac{1}{2}ml^2\dot{\theta}^2$；势能 $V = mgl(1-\cos\theta)$（取最低点为零点）。
- **第三步，构造 $L$**：$L = \frac{1}{2}ml^2\dot{\theta}^2 - mgl(1-\cos\theta)$。
- **第四步，代入拉格朗日方程**：$\frac{\partial L}{\partial\dot{\theta}} = ml^2\dot{\theta}$，$\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{\theta}} = ml^2\ddot{\theta}$；$\frac{\partial L}{\partial\theta} = -mgl\sin\theta$。于是 $ml^2\ddot{\theta} + mgl\sin\theta = 0$，即 $\ddot{\theta} + \frac{g}{l}\sin\theta = 0$——单摆运动方程。

**辨析｜易错点：**$\frac{\partial L}{\partial\dot{q}}$ 是对广义速度求偏导（把 $\dot{q}$ 当独立变量），$\frac{\mathrm{d}}{\mathrm{d}t}$ 是对时间全导数。$T$ 若含 $q$（如转动惯量随坐标变化），$\frac{\partial L}{\partial q}$ 里要包含 $T$ 对 $q$ 的项。势能 $V$ 只与 $q$ 有关（不含 $\dot{q}$），所以 $\frac{\partial L}{\partial\dot{q}} = \frac{\partial T}{\partial\dot{q}}$。

## 4 拉格朗日方程的优点

- **不依赖坐标选择**：任何广义坐标下形式相同——选最方便的坐标即可；
- **自动含约束**：约束被坐标吸收，方程中无约束力；
- **统一处理复杂系统**：多质点、多自由度系统只需写能量；
- **势能语言**：只关心势能函数，天然衔接守恒律与对称性；
- **推广性**：从经典力学到电磁场（电荷在电磁场中的 $L$）、量子力学（路径积分）、相对论、规范场论——拉格朗日语言是整个理论物理的通用工具。

**重点：拉格朗日方程把力学问题「标准化」为写能量 + 代方程——适合任意坐标、任意自由度的系统。** 它是理论物理的「母语」：量子场论、广义相对论的方程都由拉格朗日量导出。<span class="marginnote">「拉格朗日语言是现代物理的母语」：麦克斯韦方程组可由电磁场的 $L$ 导出、薛定谔方程对应量子场论的 $L$、广义相对论由爱因斯坦-希尔伯特作用量导出——所有基本理论都写成「拉格朗日量 + 变分」的形式。「从极限到大模型」的整个现代物理，都是拉格朗日语法的产物。</span>

## 5 有势力与非保守力

若主动力不含势（如摩擦力、耗散力），拉格朗日方程需修正：

$$\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = Q_j^{\text{非保守}}$$

右边是**非保守广义力**（如摩擦力对应的广义力）。耗散系统也可用**瑞利耗散函数**处理。

**辨析｜易错点：**拉格朗日方程的标准形式只适用于「主动力有势」的系统。含摩擦、阻尼（非保守力）时，右边要加非保守广义力 $Q_j$——不能直接省略。判断「能否直接用标准拉格朗日方程」先看有没有非保守力做功。

## 6 数值算例：弹簧振子的拉格朗日方程

用拉格朗日方法求水平弹簧振子（质量 $m$、劲度系数 $k$）的运动方程，并与牛顿法对比。

$$

L = \frac{1}{2}m\dot{x}^2 - \frac{1}{2}kx^2, \qquad \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{x}} - \frac{\partial L}{\partial x} = m\ddot{x} + kx = 0

$$

- **第一步，写能量**：$T = \frac12 m\dot{x}^2$，$V = \frac12 kx^2$。
- **第二步，构造 $L$**：$L = T - V = \frac12 m\dot{x}^2 - \frac12 kx^2$。
- **第三步，代入拉格朗日方程**：$\frac{\partial L}{\partial\dot{x}} = m\dot{x}$、$\frac{\partial L}{\partial x} = -kx$，得 $m\ddot{x} + kx = 0$。
- **第四步，对比牛顿法**：$F = -kx = ma$，同样得 $m\ddot{x} + kx = 0$——两种方法殊途同归，但拉格朗日法只写了能量、没有画受力图。<span class="marginnote">「拉格朗日法的省力之处」：对弹簧振子，两种方法几乎一样省力；但对复杂系统（多质点、约束、转动的耦合），拉格朗日法只写能量的优势就显现了——受力分析往往要列多个方向的方程，而 $L = T - V$ 一写到底。这正是分析力学在机器人逆动力学、航天器姿态控制中成为标准方法的原因。</span>

## 7 循环坐标与广义动量守恒

若拉格朗日量不含某广义坐标 $q_j$（$\frac{\partial L}{\partial q_j} = 0$），该坐标叫**循环坐标（cyclic coordinate）**，其共轭广义动量守恒：

$$

p_j = \frac{\partial L}{\partial\dot{q}_j} = \text{常量}, \qquad \frac{\mathrm{d}p_j}{\mathrm{d}t} = \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} = \frac{\partial L}{\partial q_j} = 0

$$

- **第一步，读循环坐标定义**：$L$ 不含 $q_j$ ⟹ $\partial L/\partial q_j = 0$。
- **第二步，写守恒**：拉格朗日方程给 $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} = 0$，即 $p_j = \partial L/\partial\dot{q}_j$ 守恒。
- **第三步，举例**：自由粒子 $L = \frac12 m\dot{x}^2$ 不含 $x$ ⟹ $p_x = m\dot{x}$ 守恒（动量守恒）；有心力场 $L$ 不含 $\phi$ ⟹ $p_\phi = mr^2\dot{\phi}$ 守恒（角动量守恒）。
- **第四步，体会**：循环坐标 ⟹ 守恒量——这是诺特定理在拉格朗日框架里的直接表现。找守恒量不必解方程，只看 $L$ 缺哪个坐标即可。<span class="marginnote">「对称性 → 缺坐标 → 守恒」：$L$ 不含 $q_j$ 说明系统在 $q_j$ 方向上对称（平移对称 → 动量守恒、旋转对称 → 角动量守恒）。分析力学把「守恒律」从「解方程后验证」变成「看 $L$ 就知」——这是它比牛顿法深刻的又一例证。这个「循环坐标 → 守恒」的判别法，是理论力学与量子力学（守恒量子数）共同的语言。</span>

**辨析｜易错点：**广义动量 $p_j = \partial L/\partial\dot{q}_j$ 不一定是「质量 × 速度」——若坐标是角度，$p_\phi = mr^2\dot\phi$ 是角动量；若坐标是电荷（电磁场中），$p_j$ 含矢势项。写 $p_j$ 时直接用定义求导，别默认「动量 = mv」。循环坐标的判断标准是「$L$ 不含 $q_j$」，不是「$L$ 不含 $\dot{q}_j$」。

## 8 术语速查表

| 术语 | 公式 | 要点 |
| --- | --- | --- |
| 拉格朗日量 | $L = T - V$ | 动能减势能 |
| 拉格朗日方程 | $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot q_j} - \frac{\partial L}{\partial q_j} = 0$ | 每个坐标一条 |
| 广义坐标 | $q_j$（$s$ 个） | 自由度 |
| 广义动量 | $p_j = \partial L/\partial\dot{q}_j$ | 不一定是 mv |
| 循环坐标 | $\partial L/\partial q_j = 0$ | ⟹ 动量守恒 |
| 广义力 | $Q_j$ | 非保守力修正 |

拉格朗日方程把力学标准化为「写能量 + 代方程」：$L = T - V$、每个广义坐标一条方程、不含约束力。循环坐标还让守恒量「一看便知」——对称性与守恒律在拉格朗日框架里直接对话。这一节的方法，是分析力学（哈密顿、最小作用量）与量子场论的共同起点。下一节我们转向哈密顿的表述——**哈密顿正则方程**。

## 9 小结

- **拉格朗日量**：$L = T - V$（动能减势能）。
- **拉格朗日方程**：$\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = 0$，每个广义坐标一条。
- 四步法：选坐标 → 写 $T$、$V$ → 构造 $L$ → 代方程。
- 优点：无约束力、坐标无关、多自由度统一、理论物理通用语言。
- 非保守力：方程右边加广义力 $Q_j$ 修正。

在下一节，我们转向哈密顿的表述——**哈密顿正则方程**。
