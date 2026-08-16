---
title: 量子力学公理体系（希尔伯特空间、算符、测量）
date: 2026-08-07
---

# 量子力学公理体系（希尔伯特空间、算符、测量）

<div class="epigraph">
<p>我们所观察到的并不是自然本身，而是自然暴露在我们的追问方法之下的那一面。</p>
<footer>—— 维尔纳 · 海森堡（Werner Heisenberg），《物理学与哲学》，1958</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 曾谨言《量子力学》第一～四章 ｜ 2026-08-07</p>
</div>

## 为什么从量子力学公理体系开始

在「第七篇 四大力学入门 · 第二十三章 量子力学」里，我们已经建立了态叠加原理、算符与平均值、本征值问题这些章节——那是把量子力学的「操作手册」摊开。而本专题进入「第1篇」深挖课程，这一节要把量子力学**从经验规则升格为公理体系**：几个公设（态是希尔伯特空间矢量、可观测量是厄米算符、测量结果是本征值并引起坍缩、态按薛定谔方程演化）就足够推出整个量子力学。这组公设是整个「从极限到大模型」课程里最值得认真对待的思想节点——现代物理的几乎所有分支（量子场论、凝聚态、量子信息）都在同一套公理上重建。立稳公理，后面几篇（典型系统、近似方法）才有共同地基。

## 1 态矢与希尔伯特空间

量子力学的第一个公设：**物理系统的状态由希尔伯特空间中的一个单位矢量（态矢）$|\psi\rangle$ 完全描述。**

**希尔伯特空间（Hilbert space）**：完备的内积空间，即「定义了内积、且对极限封闭的线性空间」。量子力学里常用的希尔伯特空间包括：自旋的空间（二维）、谐振子的能级空间（无限维）、粒子的位置空间（$L^2(\mathbb{R}^3)$）。<span class="marginnote"><strong>为什么必须是线性空间</strong>：态叠加原理要求「$|\psi_1\rangle$、$|\psi_2\rangle$ 是可能的态，则 $a|\psi_1\rangle + b|\psi_2\rangle$ 也是可能的态」——这正是线性空间的定义。叠加性是量子力学区别于经典力学的第一特征，见《态叠加原理与量子力学的基本假设》。</span>

**内积与概率**：内积 $\langle\phi|\psi\rangle$ 的模平方 $|\langle\phi|\psi\rangle|^2$ 给出「系统处于 $|\psi\rangle$ 时，测得它处于 $|\phi\rangle$ 的概率」——这是玻恩（Max Born）1926 年给出的概率诠释。态矢归一化 $\langle\psi|\psi\rangle = 1$ 保证总概率为 1。<span class="marginnote"><strong>狄拉克记号</strong>：$|\psi\rangle$ 叫 ket，$\langle\phi|$ 叫 bra，内积写成 $\langle\phi|\psi\rangle$——这个记号由狄拉克 1939 年引入，把「矢量」与「对偶」的区分变成一件顺手的外衣。后面我们会反复用到。</span>

## 2 力学量与算符

第二个公设：**每个可观测的力学量 $A$ 对应一个厄米算符 $\hat A$，它在希尔伯特空间上线性作用，其本征值即测量可能得到的结果。**

**厄米算符（Hermitian operator）**：满足 $\langle\phi|\hat A\psi\rangle = \langle\hat A\phi|\psi\rangle$（记作 $\hat A^\dagger = \hat A$）的算符。厄米性保证：**本征值是实数**（可测量必须给出实数）、**不同本征值对应的本征态正交**。

核心算符的对应关系：位置 $\hat x$、动量 $\hat p = -i\hbar\nabla$、能量（哈密顿）$\hat H = \frac{\hat p^2}{2m} + V(\hat x)$、角动量 $\hat L = \hat r\times\hat p$。**对应原理**：经典力学量换成厄米算符即得量子力学。<span class="marginnote"><strong>为什么动量是 $-i\hbar\nabla$</strong>：要求平移对称性（空间的均匀性），态矢在平移下变化，其生成元就一定是动量算符——与经典力学里「平移对称 → 动量守恒」是同一套诺特思想，只是从「守恒量」变成了「算符」。见《哈密顿正则方程》与《中心力场与角动量》。</span>

## 3 测量公设与波函数坍缩

第三个公设（测量）：**对处于 $|\psi\rangle$ 的系统测量 $\hat A$，只能得到某个本征值 $a_n$，概率为 $|\langle a_n|\psi\rangle|^2$；测量后系统立即「跳」到对应的本征态 $|a_n\rangle$。**

这个「跳到本征态」的过程叫**波函数坍缩（wavefunction collapse）**，是量子力学里最受争议、也最实用的公设——它把「测量」从旁观提升为改变系统的基本操作。若测量后立即再测同量，必得同一本征值：测量把系统「制备」进了本征态。<span class="marginnote"><strong>历史之争</strong>：坍缩公设让爱因斯坦问出「难道月亮只有在被看的时候才存在？」；薛定谔的猫、EPR 佯谬都围绕它展开。现代量子信息把「测量 = 制备 + 信息获取」作为第一性操作，量子比特的读取、量子隐形传态都以坍缩为工具——测量不再是恼人的副作用，而是可编程的资源。</span>

**测量投影的数学**：设本征态族 $\{|a_n\rangle\}$ 构成正交完备基，则 $|\psi\rangle = \sum_n c_n |a_n\rangle$，其中 $c_n = \langle a_n|\psi\rangle$。测得 $a_n$ 的概率 $p_n = |c_n|^2$，且 $\sum_n p_n = 1$（归一化）。测量后系统处于 $|a_n\rangle$。

## 4 公式解析：期望值与不确定关系

第四个公设给出态的演化（薛定谔方程），而测量结果的**平均**由期望值给出。对态 $|\psi\rangle$，力学量 $A$ 的期望值：

$$
\langle A\rangle = \langle\psi|\hat A|\psi\rangle = \sum_n a_n |c_n|^2
$$

- **第一步，展开态**：$|\psi\rangle = \sum_n c_n|a_n\rangle$，$c_n = \langle a_n|\psi\rangle$ 是投影系数。
- **第二步，代内积**：$\langle\psi|\hat A|\psi\rangle = \sum_{m,n} c_m^* c_n \langle a_m|\hat A|a_n\rangle = \sum_{m,n} c_m^* c_n a_n \delta_{mn} = \sum_n a_n |c_n|^2$。
- **第三步，概率诠释**：期望值 = 各本征值按概率 $|c_n|^2$ 加权平均——与经典统计「平均值 = 各可能值按概率加权」完全同构。
- **第四步，涨落**：方差 $(\Delta A)^2 = \langle(\hat A - \langle A\rangle)^2\rangle = \langle\hat A^2\rangle - \langle A\rangle^2$ 度量不确定性。

两个不对易的力学量存在**不确定关系（uncertainty relation）**：若 $[\hat A, \hat B] \neq 0$，则 $\Delta A\,\Delta B \ge \frac{1}{2}|\langle[\hat A,\hat B]\rangle|$。对位置与动量，$[\hat x, \hat p] = i\hbar$，于是

$$
\Delta x\,\Delta p \ge \frac{\hbar}{2}
$$

**重点：不确定关系不是「测量技术不够好」，而是系统本身的性质——位置与动量在量子力学里没有共同本征态，任何态都不可能同时精确定义两者。** 它是经典力学「同时给定位置与速度」这个基本前提的直接否定。<span class="marginnote"><strong>数值算例</strong>：电子（$m_e \approx 9.11\times10^{-31}\ \mathrm{kg}$）被约束在原子尺度 $\Delta x \sim 10^{-10}\ \mathrm{m}$ 内，则 $\Delta p \ge \hbar/(2\Delta x) \sim 5\times10^{-25}\ \mathrm{kg\cdot m/s}$，对应速度不确定度约 $5\times10^5\ \mathrm{m/s}$——原子中电子无法静止，这正是「电子为什么不会坍缩进原子核」的经典解释之一。见《不确定关系》。</span>

## 5 对易关系与共同本征态

算符是否对易决定了能否同时精确测量。**对易子（commutator）**：$[\hat A, \hat B] = \hat A\hat B - \hat B\hat A$。

**定理：$\hat A$、$\hat B$ 具有一组共同完备本征态 $\Leftrightarrow$ $[\hat A, \hat B] = 0$。** 可同时测量的一组力学量叫**相容力学量**。例如位置三分量 $[\hat x_i, \hat x_j] = 0$，可同时确定；角动量分量 $[\hat L_x, \hat L_y] = i\hbar\hat L_z \neq 0$，不能同时精确确定——这正是「量子化的方向」（见《中心力场与角动量》）。<span class="marginnote"><strong>完备集的概念</strong>：找出「最大的两两对易算符集合」（如氢原子的 $H, L^2, L_z$），它们的共同本征值就完全标定每个量子态——量子数 $(n, l, m)$ 就是这么来的。这个「找完备对易集 → 取共同本征态 → 用量子数标定」的程序，是解一切量子系统的总纲。</span>

**数值算例（对易与测量）**：自旋 1/2 系统，$\hat S_z$ 本征态 $|\uparrow\rangle, |\downarrow\rangle$。若先测 $\hat S_x$ 得 $+\hbar/2$，再测 $\hat S_z$，结果各 $50\%$——因为 $[\hat S_x, \hat S_z] = i\hbar\hat S_y \neq 0$，第一次测量破坏了对 $\hat S_z$ 的确定性。这组实验（Stern-Gerlach 装置）是量子测量公设最直观的演示。

## 6 数值算例：量子比特与测量

公理体系在现代量子信息里的第一个应用对象就是**量子比特（qubit）**：一个两态量子系统（自旋 1/2、原子两能级、光子偏振），其状态是二维希尔伯特空间里的单位矢量

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \qquad |\alpha|^2 + |\beta|^2 = 1
$$

- **第一步，读系数**：$\alpha = \langle0|\psi\rangle$、$\beta = \langle1|\psi\rangle$ 是两个基态的振幅，其模平方即测得 0/1 的概率。
- **第二步，体会「叠加」**：量子比特可以「同时」处于 $|0\rangle$ 与 $|1\rangle$ 的叠加，而经典比特只能是 0 或 1——这是量子计算加速的源泉（$n$ 个量子比特有 $2^n$ 个振幅）。
- **第三步，测量即坍缩**：对 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ 测量，结果 0 概率 $|\alpha|^2$，结果 1 概率 $|\beta|^2$；测量后态变为 $|0\rangle$ 或 $|1\rangle$——振幅信息在测量时不可逆地丢失。
- **第四步，连线量子算法**：量子门（如 Hadamard 门把 $|0\rangle$ 变到 $\frac{|0\rangle+|1\rangle}{\sqrt2}$）都是酉变换，保归一化；只有最后读出时测量公设才登场。<span class="marginnote"><strong>为什么不等于「随机数」</strong>：叠加态不是「知道一半概率」的经典随机——它是确定性的量子态，只是测量结果具有概率性。贝尔不等式的实验（Aspect 1982、Hensen 2015）证明这种区别是真实的，量子信息正是利用这种「非经典相关性」实现保密通信与加速计算。见《态叠加原理》与《现代物理前沿专题》。</span>

## 7 核心对比表：经典力学 vs 量子力学

| 维度 | 经典力学 | 量子力学 |
| --- | --- | --- |
| 状态 | 相空间点 $(q,p)$ | 希尔伯特空间态矢 $|\psi\rangle$ |
| 物理量 | 实数函数 $A(q,p)$ | 厄米算符 $\hat A$ |
| 可预测性 | 初始态决定一切 | 结果有概率分布 |
| 测量 | 不扰动系统 | 引起坍缩、改变系统 |
| 叠加 | 无 | 核心特征 |
| 统计不确定 | 仅测量误差 | 原理性的 $\Delta x\Delta p\ge\hbar/2$ |
| 对易 | 处处交换 | $[\hat x,\hat p]=i\hbar$ |

**重点：量子力学不是「经典力学加一点修正」，而是一套全新的描述语言——经典力学是其 $\hbar\to0$ 的极限。** 把期望值方程与经典力学对照，所有经典方程都出现在期望值层面（埃伦费斯特定理 $\frac{\mathrm{d}}{\mathrm{d}t}\langle x\rangle = \langle p\rangle/m$），量子修正项（如 $\frac{\hbar^2}{8m}$ 量级的量子势）在宏观极限下消失。<span class="marginnote"><strong>对应原理的历史角色</strong>：玻尔用「量子结果在大量子数极限退回经典」来检验早期量子论；现代观点则把经典极限当作「退相干」的结果——环境噪声摧毁叠加，让宏观世界看起来像经典的。这套「量子公理 + 退相干 → 经典」的图景，是理解「微观量子、宏观经典」为何不矛盾的关键。</span>

## 8 术语速查表

| 概念 | 数学表述 | 要点 |
| --- | --- | --- |
| 态矢 | $|\psi\rangle \in \mathcal{H}$ | 归一化 $\langle\psi|\psi\rangle=1$ |
| 可观测量 | $\hat A^\dagger = \hat A$ | 本征值为实数 |
| 测量 | 概率 $p_n=|\langle a_n|\psi\rangle|^2$ | 测量后坍缩到 $|a_n\rangle$ |
| 期望值 | $\langle A\rangle=\langle\psi|\hat A|\psi\rangle$ | 概率加权平均 |
| 不确定关系 | $\Delta A\,\Delta B \ge \frac12|\langle[\hat A,\hat B]\rangle|$ | 不对易 → 无法同时确定 |
| 对易子 | $[\hat A,\hat B]$ | 为 0 ⇔ 有共同本征态 |

## 9 小结

- **量子力学五公设**：态是希尔伯特空间单位矢量；可观测量是厄米算符；测量得本征值并坍缩；态按薛定谔方程演化；全同粒子服从对称化规则（见《自旋与全同粒子体系》）。
- **希尔伯特空间** + 内积给出概率诠释：$|\langle\phi|\psi\rangle|^2$ 是过渡概率。
- **厄米算符**保证实本征值、正交本征态；测量即「向本征态投影 + 得到本征值」。
- **不确定关系** $\Delta x\Delta p \ge \hbar/2$ 是系统性质而非测量误差，来自不对易。
- **对易 = 可同时测量**：找完备对易集、用量子数标定态，是解一切量子系统的总纲。

在下一节，我们将把公理体系应用到几个最重要、也最常用的系统上——谐振子、角动量、氢原子：**量子力学典型系统**。
