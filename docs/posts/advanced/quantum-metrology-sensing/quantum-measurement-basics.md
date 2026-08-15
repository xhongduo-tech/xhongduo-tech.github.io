---
title: 量子测量的基本概念与投影测量
date: 2026-08-07
---

# 量子测量的基本概念与投影测量

<div class="epigraph">
<p>我们所观察到的并不是自然本身，而是自然暴露于我们的提问方式之下的面貌。</p>
<footer>—— 维尔纳 · 海森堡（Werner Heisenberg, *Physics and Philosophy\*, 1958）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子精密测量与量子传感 ｜ Wiseman & Milburn, *Quantum Measurement and Control\* §2 ｜ 2026-08-07</p>
</div>

## 为什么从测量开始

量子精密测量要做的事，一句话就能说清：**用少数量子系统去估计一个经典参数，并让误差逼近量子力学允许的极限。** 但「测量」本身恰恰是量子力学里最微妙的一环——它既是信息的读出，也是态的破坏。要谈散粒噪声、海森堡极限、压缩态，第一步必须先建立「量子测量」的严格语言：观测量是什么、读出概率怎么算、测量之后态变成了什么。<span class="marginnote">这一篇是全专题的地基。后面的标准量子极限、Cramér–Rao 界、压缩态干涉仪全部建立在本篇的三件套之上：算符、投影、Born 规则。</span>

## 1 观测量与算符

在经典物理里，一次测量对应一个确定的数：温度计读出 $T$，天平读出 $m$。在量子力学里，情况完全不同——**每个可测量的物理量（observable）对应一个厄米算符（Hermitian operator）** $A$，而测量这个量，就是在读算符的「谱」。

厄米算符有两个性质支撑了整个测量理论：

**本征值是实数**：$A|\psi_n\rangle = a_n|\psi_n\rangle$，其中 $a_n \in \mathbb{R}$。这保证了测量结果一定是实数，与实验一致。
**本征态构成完备正交基**：$\sum_n |\psi_n\rangle\langle\psi_n| = I$，任意态都能在这组基下展开。

**厄米性 $\hat{A}^\dagger = \hat{A}$ 不是技术细节，而是「可观测量」的定义本身。** 一个非厄米算符的本征值可能是复数，无法对应任何仪器的读数。<span class="marginnote">在量子信息里我们常把观测量简写成 $\hat{Z}$（对应自旋 $z$ 分量）或 $\hat{X}$，它们的本征值都是 $\pm1$。这两类算符在第二级《量子力学》与第四级《量子计算》里反复出场。</span>

具体例子随手可得：位置算符 $\hat{x}$ 的本征值覆盖整个实数轴，动量算符 $\hat{p} = -i\hbar\,\partial_x$ 与之构成一对非对易算符；哈密顿量 $\hat{H}$ 的本征值正是可测的能量。**一套可观测量的本征值谱，就是这台「量子仪器」能给出的所有读数的清单**——谱越丰富，能测的信息越多，这也是为什么量子计量学总爱用量子比特（谱只有 $\pm1$）以外的连续变量系统来追求更丰富的读出。

## 2 投影测量与 Born 规则

测量过程的标准模型叫**投影测量（projective measurement）**，也叫 von Neumann 测量。设待测观测量 $\hat{A}$ 的谱分解为

$$
\hat{A} = \sum_m a_m \hat{P}_m, \qquad \hat{P}_m = |m\rangle\langle m|
$$

其中 $\hat{P}_m$ 是**投影算符**，满足 $\hat{P}_m \hat{P}_n = \delta_{mn}\hat{P}_m$ 与 $\sum_m \hat{P}_m = I$。

**Born 规则**给出读出结果 $a_m$ 的概率：

$$
p(m) = \langle\psi|\hat{P}_m|\psi\rangle = \operatorname{Tr}\big(\rho\,\hat{P}_m\big)
$$

对混态 $\rho$，写成迹的形式。**结果是离散的：哪怕初态是叠加态，一次测量也只能得到某一个本征值，而概率由 $|\langle m|\psi\rangle|^2$ 给出。**<span class="marginnote">Born 规则是量子力学里唯一把「波函数」翻译成「观测到的计数」的桥梁。把它翻译成实验语言：同一实验重复 $N$ 次，$N\to\infty$ 时得到 $a_m$ 的次数占比趋于 $p(m)$——这正是后一篇「散粒噪声」的出发点。</span>

举个最熟悉的例子：测量自旋-$\tfrac12$ 粒子的 $z$ 分量 $\hat{Z}$。设初态为 $|\psi\rangle = (\ket{\uparrow} + \ket{\downarrow})/\sqrt{2}$，$\hat{Z}$ 的本征值为 $\pm1$，对应本征态 $\ket{\uparrow},\ket{\downarrow}$。Born 规则给出 $p(\uparrow) = |\langle\uparrow|\psi\rangle|^2 = 1/2$，$p(\downarrow) = 1/2$。**注意：叠加态并没有「把一半自旋放在上、一半放在下」，而是「测量时随机落到上或下，各占一半概率」。** 这个区分是量子测量里最容易搞混、也最本质的一点。

## 3 测量对态的影响：坍缩与可重复性

测量不只读出信息，它还**改变**系统的态。**若测量得到结果 $a_m$，测量后的态（未归一化）为**

$$
\rho \;\longrightarrow\; \hat{P}_m \rho \hat{P}_m
$$

对纯态 $|\psi\rangle$，就是 $\rho \mapsto |m\rangle\langle m|$——系统「坍缩」到本征态 $|m\rangle$ 上。这一规则叫 **Lüders 规则（Lüders rule）**。

由它立刻推出一个关键性质：**可重复性（repeatability）**。同一观测量对坍缩后的态再测一次，得到相同结果的概率是 1。这个性质是**非破坏性测量（QND, quantum non-demolition）** 的雏形——精密测量之所以能对同一个原子反复提问，靠的就是它。<span class="marginnote">注意「坍缩」只是表象规则，它不解释物理机制。测量与环境的退相干关系，我们在第四级《量子信息基础》与《量子光学》里用主方程给出更自洽的描述；本篇只需把「测量＝投影＋概率」当作公设使用。</span>

一个直接推论：**连续两次测量同一个可观测量，得到同一个结果。** 这听起来平凡，却支撑了计量学里最常见的实验循环——「初始化 → 演化 → 读出 → 再次初始化」。原子钟的 Ramsey 序列、NV 色心的 ODMR、超导量子比特的读出，全部建立在这条可重复性之上。测量在这里不是「消耗态」的破坏性动作，而是可以反复执行的程序化步骤。

## 4 公式解析：投影算符与读出概率

把核心公式拆开，每一步都有对应物：

$$
p(m) = \langle\psi|\hat{P}_m|\psi\rangle = \langle\psi|m\rangle\langle m|\psi\rangle = \big|\langle m|\psi\rangle\big|^2
$$

- **第一步，中间插入**：$\hat{P}_m = |m\rangle\langle m|$ 是一个秩一投影，代入后出现两个内积的乘积。
- **第二步，取模方**：$\langle\psi|m\rangle$ 与 $\langle m|\psi\rangle$ 互为共轭，相乘正是振幅的模平方 $|\langle m|\psi\rangle|^2$。
- **第三步，物理含义**：$|\langle m|\psi\rangle|^2$ 是「初态在 $|m\rangle$ 方向上的投影长度平方」——这正是量子力学里概率的几何图像。

做一个数字检验：设 $|\psi\rangle = (\ket{\uparrow} + \sqrt{3}\ket{\downarrow})/2$，测 $\hat{Z}$。则 $p(\uparrow) = |\langle\uparrow|\psi\rangle|^2 = 1/4$，$p(\downarrow) = 3/4$，且 $1/4 + 3/4 = 1$ 自动满足——归一化的波函数保证概率和为 1。**检验概率归一，是任何测量计算的第一步**，也是写进程序里的第一个断言。

**辨析｜易错点：** $\operatorname{Tr}(\rho\hat{P}_m)$ 里的迹运算不能与「测量后求期望」混淆。期望值是 $\langle\hat{A}\rangle = \sum_m a_m p(m) = \operatorname{Tr}(\rho\hat{A})$，它是所有结果的加权平均；而 $p(m)$ 是单次结果的概率。一次实验只能得一个 $a_m$，反复实验才得到 $\langle\hat{A}\rangle$——这条区分在计量学里对应「单次读数 vs 统计估计」两种视角。顺带一提，对纯态 $\rho = |\psi\rangle\langle\psi|$，迹运算约化为内积 $\langle\psi|\hat{P}_m|\psi\rangle$；对混态则必须保留完整的迹——这正是混态与纯态在测量层面最直观的差别。

## 5 辨析：投影测量、POVM 与弱测量

投影测量不是全部。计量学里另两类测量同样重要：

**POVM（positive operator-valued measure）**：把投影 $\hat{P}_m$ 换成一般的半正定算符 $\hat{E}_m$，只要求 $\sum_m \hat{E}_m = I$，$\hat{E}_m \ge 0$。POVM 允许结果数超过 Hilbert 空间维数，也不保证可重复性。<span class="marginnote">投影测量是 POVM 的特例。在纠缠辅助的测量（如把系统与辅助比特耦合后再测）里，POVM 是自然语言——标准量子极限那一篇会用到它。</span>

**弱测量（weak measurement）**：测量与系统的作用强度放弱，让态只被扰动一点点，代价是单次携带的信息量也变少。弱测量是连续测量与量子反馈控制的基础，也是引力波探测器里连续读出思想的雏形。弱测量的极端情形是**连续测量**：以很小的时间步长反复弱测，就能在不大幅扰动系统的前提下，把参数的演化轨迹「录」下来——这为量子反馈、量子误差消除提供了实验上可行的窗口。

**辨析｜易错点：** 不要把「坍缩」理解成「知道结果才坍缩」。投影发生在物理作用层面，与观察者是否看见读数无关；「未读出结果」时的态是混合态 $\rho' = \sum_m \hat{P}_m\rho\hat{P}_m$，而不是某个确定的本征态。

### 核心概念速查表

把本篇的可观测量与测量语言汇总成一张表，供后续所有篇章随时查阅：

| 概念 | 记号/公式 | 一句话含义 |
| --- | --- | --- |
| 观测量（observable） | 厄米算符 $\hat{A} = \hat{A}^\dagger$ | 每个可测物理量对应一个厄米算符 |
| 谱分解 | $\hat{A} = \sum_m a_m \hat{P}_m$ | 本征值 $a_m$ 是读数，本征态张成空间 |
| 投影算符 | $\hat{P}_m = \|m\rangle\langle m\|$ | 满足 $\hat{P}_m^2 = \hat{P}_m$、$\sum_m \hat{P}_m = I$ |
| Born 规则 | $p(m) = \operatorname{Tr}(\rho \hat{P}_m)$ | 读出概率等于投影后的迹 |
| Lüders 规则 | $\rho \mapsto \hat{P}_m\rho\hat{P}_m / p(m)$ | 测量后态的更新规则 |
| 期望值 | $\langle\hat{A}\rangle = \operatorname{Tr}(\rho\hat{A})$ | 全部结果的加权平均，与单次读数不同 |
| POVM | $\{\hat{E}_m\}$，$\sum_m \hat{E}_m = I$ | 投影测量的推广，允许超维度结果 |
| 可重复性 | 再测一次结果不变 | 投影测量可重复，QND 的雏形 |

这张表里的前六行是量子信息与量子计量的「公共词表」——第四级《量子信息基础》里测量一章会原样沿用它们，只是把单自旋换成多体系统。

## 6 小结

- 每个可观测物理量对应一个**厄米算符**，本征值为实数，本征态构成完备正交基。
- **投影测量**由投影算符族 $\{\hat{P}_m\}$ 描述，**Born 规则**给出概率 $p(m)=\operatorname{Tr}(\rho\hat{P}_m)$。
- 测量后态按 **Lüders 规则**更新为 $\hat{P}_m\rho\hat{P}_m/p(m)$，投影测量因此**可重复**——QND 测量的雏形。
- 更一般的测量是 **POVM**，结果个数可超维度、不必可重复；**弱测量**以扰动为代价换取连续读出。
- 单次结果 $a_m$ 与期望 $\langle\hat{A}\rangle$ 是两种不同对象，计量学关心前者在重复实验下的统计分布。
- **可重复性**是投影测量的天然属性，支撑「初始化→演化→读出」的实验循环，是 QND 测量的起点。
- 弱测量与连续测量以扰动为代价换取演化轨迹，是反馈控制与量子误差消除的入口。

在下一节，我们将问一个计量学最根本的问题：重复测量 $N$ 次，估计精度如何随 $N$