---
title: 相干态及其性质
date: 2026-08-07
---

# 相干态及其性质

<div class="epigraph">
<p>激光的量子本质，就在于它的相位像波浪一样确定，而能量像粒子一样计数。</p>
<footer>—— 罗伊·格劳伯（Roy J. Glauber），2005 年诺贝尔物理学奖演讲</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ M. O. Scully & M. S. Zubairy, Quantum Optics 第2章 ｜ 2026-08-07</p>
</div>

## 为什么从相干态开始

上一节我们认识了 Fock 态——光子数精确但相位全无。
可现实中的激光器输出的是**有明确相位、且强度稳定**的光，
它既不是「正好 n 个光子」，也不像热光那样混乱。1963 年，
格劳伯给出了答案：
激光的量子态是**相干态（coherent state）**，
记作 $|\alpha\rangle$。它是量子光学最核心的工作马：
是唯一「尽可能接近经典波」的量子态，也是量子力学「最小不确定态」的典范。
理解它，激光物理、干涉测量、量子密钥分发的安全证明，
全都由此展开。<span class="marginnote">格劳伯因这项贡献获 
2005 年诺贝尔物理学奖。相干态概念此前在薛定谔 1926 
年关于「最接近经典运动的波包」的论文中已现雏形。</span>

## 1 相干态的三条等价定义

相干态最优雅的地方在于它有**三条彼此等价**的定义，每条给出一种直觉：

**定义一（湮灭算符本征态）**：$|\alpha\rangle$ 
是湮灭算符的本征态

$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle, \qquad \alpha \in \mathbb{C}$$

复数 $\alpha = |\alpha|e^{i\theta}$ 
编码光场的振幅（$|\alpha|$）与相位（$\theta$）。


**定义二（真空位移）**：用位移算符把真空态平移

$$|\alpha\rangle = \hat{D}(\alpha)|0\rangle, \qquad \hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$$

**定义三（Fock 态展开）**：按光子数态展开，系数是泊松分布

$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}}\,|n\rangle$$

<span class="marginnote">位移算符 $\hat{D}(\alpha)$ 
是相干态家族的李群结构：$|\alpha\rangle$ 
可以看作把真空态在相空间里平移一个复矢量 $\alpha$，平移量越大，
场越「强」。</span>

## 2 光子数分布：泊松分布

把相干态投影到 Fock 态，得到光子数分布

$$P(n) = |\langle n|\alpha\rangle|^2 = \frac{\bar{n}^n e^{-\bar{n}}}{n!}, \qquad \bar{n} = |\alpha|^2$$

这正是**泊松分布**。它的两个关键性质：

- **均值等于方差**：$\langle n\rangle = |\alpha|^2 = \mathrm{Var}(n)$。平均光子数与涨落量级相同。
- **相对涨落随强度衰减**：$\frac{\Delta n}{\bar{n}} = \frac{1}{\sqrt{\bar{n}}}$——强激光（$\bar{n}$ 大）相对涨落趋近于零，宏观上表现为「稳定的经典波」。

**重点：泊松分布是「光子的最小涨落」在强度上的表现，但它允许光子数有任意值。** 
相干态在相位上最确定、在光子数上仍是分布的——它和 Fock 
态正好站在不确定关系的两端。

## 3 最小不确定态与相空间意义

计算正交分量 $X_1 = \frac{1}{2}(\hat{a} + \hat{a}^\dagger)$、$X_2 = \frac{1}{2i}(\hat{a} - \hat{a}^\dagger)$ 
的涨落：

$$\Delta X_1^2 = \Delta X_2^2 = \frac{1}{4}, \qquad \Delta X_1 \Delta X_2 = \frac{1}{4}$$

满足 Heisenberg 
不确定关系 $\Delta X_1\Delta X_2 \geq \frac{1}{4}$（$\hbar = 1$ 
单位）的**等号**，且两个方向的涨落相等。
这意味着相干态在相空间中是一个**对称的圆斑**，
圆心在 $\alpha$，
半径固定。<span class="marginnote">相空间里这一圆斑就是它的 
Wigner 函数轮廓，下一节《量子相空间》会把圆斑、
椭圆斑（压缩态）放在一起比较。</span>

**辨析｜易错点：** 
不确定关系写 $\Delta X_1\Delta X_2 \geq \frac{1}{4}$ 
还是 $\geq \frac{\hbar}{2}$？
取决于是否把 $\hbar$ 归一。若保留 $\hbar$，
则 $\Delta X_1\Delta X_2 \geq \hbar/2$，
相干态取等号。
做数值题前先确认单位约定——这是量子光学文献里最常见的「差一个 
2」之源。

## 4 公式解析：$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_n \frac{\alpha^n}{\sqrt{n!}}|n\rangle$

这条展开式是相干态的「身份证」，拆成三步：

**第一步，系数来自泊松分布**：$c_n = \langle n|\alpha\rangle = e^{-|\alpha|^2/2}\alpha^n/\sqrt{n!}$。模方 $|c_n|^2$ 给出泊松分布 $P(n) = \bar{n}^n e^{-\bar{n}}/n!$，其中 $\bar{n} = |\alpha|^2$。
**第二步，相位藏在 $\alpha^n$ 里**：$\alpha = |\alpha|e^{i\theta}$，于是 $c_n \propto e^{in\theta}$——不同 Fock 态带上递增的相位 $n\theta$，正是这些相邻态的相对相位构造出确定的场振荡。
- **第三步，归一化**：$\sum_n |c_n|^2 = e^{-|\alpha|^2}\sum_n |\alpha|^{2n}/n! = e^{-|\alpha|^2}e^{|\alpha|^2} = 1$，恰好归一，不需要额外系数。

**补充**：一个深刻的性质——相干态是**过完备**的。
不同 $\alpha$ 
的相干态并不正交：$\langle \beta|\alpha\rangle = e^{-|\alpha-\beta|^2/2}$，
但它们满足完备性关系 $\frac{1}{\pi}\int d^2\alpha\,|\alpha\rangle\langle\alpha| = \mathbb{1}$。
过完备性让相干态成为强大的计算工具（P 表示、Gaussian 
态运算），也是连续变量量子计算的基础。

## 5 相干态：激光的量子描述

为什么激光器输出相干态？物理直觉：
受激辐射的玻色增强效应让相位不断被克隆到新光子，
光场趋向于一个**经典相位确定、强度稳定的状态**；数学上，
激光阈值以上单模场在无损耗近似下就是 $|\alpha\rangle$。
相干态也因此成为量子光学与经典光学的「接缝」——它是量子态，
但行为几乎完全经典。这也解释了为什么日常光学（干涉、
衍射）不需要量子理论：
所有相干叠加都已被相干态的平均场 $\langle \hat{E}\rangle \neq 0$ 
捕获。<span class="marginnote">这条「相干态 
≈ 经典光」的桥梁，
正是「从极限到大模型」主线里经典—量子接口的又一实例：
宏观世界从微观量子态中涌现，但保留了相干的秩序。</span>

## 6 小结

- 相干态是湮灭算符本征态 $\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$，等价于真空位移 $|\alpha\rangle = \hat{D}(\alpha)|0\rangle$。
- 光子数服从**泊松分布**：均值 = 方差 = $|\alpha|^2$，相对涨落 $\propto 1/\sqrt{\bar{n}}$。
- 相干态是**最小不确定态**：$\Delta X_1\Delta X_2 = 1/4$（$\hbar=1$），涨落对称分布。
- 相干态**过完备**：$\langle\beta|\alpha\rangle = e^{-|\alpha-\beta|^2/2}$