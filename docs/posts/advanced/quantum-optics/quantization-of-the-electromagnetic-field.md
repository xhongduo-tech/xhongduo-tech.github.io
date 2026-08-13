---
title: 电磁场的量子化
date: 2026-08-07
---

# 电磁场的量子化

<div class="epigraph">
<p>上帝掷骰子吗？</p>
<footer>—— 阿尔伯特·爱因斯坦（Albert Einstein），1926 年致马克斯·玻恩</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ R. Loudon, The Quantum Theory of Light 第1章 ｜ 2026-08-07</p>
</div>

## 为什么从电磁场的量子化开始

量子光学的一切都从这里出发。**把电磁场量子化**，
就是把光的描述从「连续的波」升级为「能一份一份存在的东西」——光子。
1900 
年普朗克为解释黑体辐射而假设能量量子 $\hbar\omega$，
1905 年爱因斯坦用光量子解释光电效应，但直到 1927 
年狄拉克（P. A. M. 
Dirac）才把**量子力学与电磁场**统一成一套自洽的理论：
场的每个振动模式都是一台量子谐振子。
这台「谐振子」的激发数就是光子数。理解这一步，你才能往下走：
相干态、压缩态、腔 QED 全部是「谐振子语言」的不同讲法。
它也与你已学的第一级《基础物理》的光学、
第二级《量子力学》的谐振子直接接榫。<span class="marginnote">本专题是「从极限到大模型」第四级高阶物理的一部分，
与同级的《量子信息基础》《量子计算》共用同一套 Fock 
态与产生湮灭算符语言。</span>

## 1 把电磁场装进盒子：模式分解

自由空间的电磁场有无穷多个自由度，直接量子化无从下手。
第一步是**把空间限制在一个有限体积的盒子**里，
让场展开成驻波模式。取一个边长 $L$ 的立方腔，
满足周期性边界条件，则波矢 $k$ 只能取分立值

$$k_i = \frac{2\pi n_i}{L}, \qquad n_i = 0, \pm 1, \pm 2, \ldots$$

每个模式 $(k, s)$（$s$ 
为偏振）对应一个简正频率 $\omega_k = c|\vec{k}|$。
在库仑规范下，矢量势可展开为

$$\vec{A}(\vec{r}, t) = \sum_{k, s} \sqrt{\frac{\hbar}{2\epsilon_0 \omega_k V}} \, \vec{e}_{ks} \left[ a_{ks}(t) e^{i\vec{k}\cdot\vec{r}} + a_{ks}^*(t) e^{-i\vec{k}\cdot\vec{r}} \right]$$

其中 $V = L^3$ 
是腔体积，$\vec{e}_{ks}$ 
是偏振单位矢量。**关键句：每一个 $(k, s)$ 模式都携带自己的复振幅 $a_{ks}(t)$，它随时间简谐振荡。** <span class="marginnote">周期性边界条件是「把无穷大装进盒子」的数学手段，
代价是引入假想的盒子；取 $V \to \infty$ 
极限时所有物理可观测量都回到连续极限。</span>

## 2 模式即谐振子：一次严格的类比

现在出现一个漂亮的对应。写出单个模式的电磁能量

$$H_{ks} = \epsilon_0 V \left( |\dot{A}_{ks}|^2 + \omega_k^2 |A_{ks}|^2 \right)$$

这正是一台**角频率 $\omega_k$ 的经典谐振子**的能量表达式 $H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 q^2$。
于是每一台「光谐振子」都有坐标与动量：

$$q_{ks} = \sqrt{\frac{\epsilon_0 V}{\omega_k^2}} (A_{ks} + A_{ks}^*), \qquad p_{ks} = -i\sqrt{\epsilon_0 V} (A_{ks} - A_{ks}^*)$$

**电磁场的每个简正模式，物理上等价于一台质量为 $m = \epsilon_0 V$ 的量子谐振子。** 
这就是量子化的全部依据——场的动力学与谐振子完全同构，
谐振子的量子力学在第二级《量子力学》里你已经学过。

## 3 对易关系与产生湮灭算符

把经典正则变量 $q, p$ 提升为算符，并施加正则量子化条件：

$$[\hat{q}_{ks}, \hat{p}_{k's'}] = i\hbar \,\delta_{kk'} \delta_{ss'}$$

定义无量纲的产生湮灭算符

$$\hat{a}_{ks} = \sqrt{\frac{\omega_k}{2\hbar}} \left( \hat{q}_{ks} + \frac{i}{m\omega_k} \hat{p}_{ks} \right), \qquad \hat{a}_{ks}^\dagger = \sqrt{\frac{\omega_k}{2\hbar}} \left( \hat{q}_{ks} - \frac{i}{m\omega_k} \hat{p}_{ks} \right)$$

它们满足玻色对易关系

$$[\hat{a}_{ks}, \hat{a}_{k's'}^\dagger] = \delta_{kk'} \delta_{ss'}, \qquad [\hat{a}_{ks}, \hat{a}_{k's'}] = 0$$

算符 $\hat{a}_{ks}^\dagger$ 
在模式 $(k,s)$ 
中**产生一个光子**，$\hat{a}_{ks}$ 
则**湮灭一个光子**；光子是玻色子，同一模式可以堆叠任意多个。
哈密顿量求和后得到

$$\hat{H} = \sum_{k,s} \hbar\omega_k \left( \hat{a}_{ks}^\dagger \hat{a}_{ks} + \frac{1}{2} \right)$$

## 4 公式解析：场哈密顿量 $\hat{H} = \sum_{k,s} \hbar\omega_k(\hat{a}^\dagger_{ks}\hat{a}_{ks} + \frac{1}{2})$

这条式子是整个量子光学的「能量账本」，拆成三步读：

- **第一步，认识数算符 $\hat{N}_{ks} = \hat{a}_{ks}^\dagger \hat{a}_{ks}$**：它测量该模式中的光子数。本征态 $|n_{ks}\rangle$ 满足 $\hat{N}_{ks}|n_{ks}\rangle = n_{ks}|n_{ks}\rangle$，$n_{ks} = 0, 1, 2, \ldots$ 是光子数本征值。
- **第二步，理解能量结构**：每个模式贡献 $n\hbar\omega_k$ 的激发能量，外加常数 $\frac{1}{2}\hbar\omega_k$。这说明电磁场的能量是**量子化的**——增量为 $\hbar\omega_k$，这正是「光子能量」的来源。
- **第三步，直面零点能 $\sum_{k,s}\frac{1}{2}\hbar\omega_k$**：即使所有模式都没有光子，真空仍有无穷大的能量。这个无穷大在经典物理里是灾难，在量子场论里则被重正化吸收；但在量子光学里，它的**物理后果**是真实的——真空涨落驱动自发辐射、产生 Casimir 力、引发 Lamb 位移。零点能不是病，而是光与物质相互作用的引擎。

## 5 量子化带来的三个世界观转变

**电磁场不再是「波或粒子」，而是「粒子 + 波的完备统一」。** 
量子化之后，场算符本身携带波的相位与传播结构，
而它的激发数 $n$ 携带粒子性。

**辨析｜易错点：** 光子数态 $|n\rangle$ 
描述的是**场**的状态，
不是某个光子「在第几个位置」——光子不可分辨，
同一模式里所有光子是全同的。
这与第一级化学里的「电子不可分辨」精神一致：
量子统计的关键不在「对象不同」，而在「模式里有多少个」。

**真空态 $|0\rangle$ 不是「什么都没有」。** 
它的能量零点处处涨落，
平均场 $\langle 0|\hat{E}|0\rangle = 0$ 
但方差 $\langle 0|\hat{E}^2|0\rangle \neq 0$。
真空是活跃的、
可被探测的——第四级《量子场论》里会用虚光子图重新讲这件事，
这里先记住「真空 = 
零点涨落」即可。<span class="marginnote">下一节我们就坐上这台谐振子，
数一数它各阶激发的性质：<strong>光子数态与 Fock 
态</strong>。</span>

## 6 小结

- 电磁场量子化的配方：**有限体积腔 → 模式展开 → 每模式 = 谐振子 → 正则量子化**。
- 产生湮灭算符满足 $[\hat{a},\hat{a}^\dagger] = 1$，$\hat{N} = \hat{a}^\dagger\hat{a}$ 是光子数算符。
- 场哈密顿量 $\hat{H} = \sum_{k,s}\hbar\omega_k(\hat{N}_{ks} + \frac{1}{2})$