---
title: 算符与力学量的平均值
date: 2026-08-07
---

# 算符与力学量的平均值

<div class="epigraph">
<p>在量子世界，每一个力学量背后站着一个算符；算符作用在态上，给出这个量的全部可能值与它们的概率。</p>
<footer>—— 量子力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 曾谨言《量子力学》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从算符开始

上一节假设二说「力学量对应厄米算符」，这一节把它展开：**算符（operator）**是量子力学的「力学量工厂」——位置、动量、能量、角动量各有其算符；测量力学量的平均值由「算符的期望值」给出。理解算符与平均值，才能进行具体的量子计算（本征值问题下节）。这一节建立算符语言：基本算符、厄米算符的性质、期望值与对易子。

## 1 基本算符

**力学量算符**：位置、动量、能量对应的算符（位置表象）：

$$\hat{x} = x, \qquad \hat{p} = -i\hbar\frac{\partial}{\partial x}, \qquad \hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(\boldsymbol{r})$$

**对应规则（正则量子化）**：经典量 $A(x, p)$ → 算符 $\hat{A} = A(\hat{x}, \hat{p})$——把 $p$ 换成 $-i\hbar\partial/\partial x$（$x$ 换成 $x$）。例如动能 $\hat{T} = \frac{\hat{p}^2}{2m} = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}$。

**重点：位置、动量、能量的算符——$\hat{x} = x$、$\hat{p} = -i\hbar\partial/\partial x$、$\hat{H} = \hat{T} + \hat{V}$。** 正则量子化（第 118 节）把经典力学量变成算符：$p \to -i\hbar\partial/\partial x$ 是最基本的对应。哈密顿算符 $\hat{H}$ 是薛定谔方程的「引擎」。

## 2 厄米算符与平均值

**厄米算符（Hermitian operator）**：满足 $\int\psi^*(\hat{A}\phi)\mathrm{d}x = \int(\hat{A}\psi)^*\phi\,\mathrm{d}x$（即 $\hat{A}^\dagger = \hat{A}$）的算符。

厄米算符的性质：

- **本征值为实数**（可观测量的测量值必须实数）；
- 不同本征值的本征态正交。

**力学量的平均值（期望值）**：系统处于态 $|\psi\rangle$（归一化）时，力学量 $A$ 的平均值：

$$\langle A\rangle = \langle\psi|\hat{A}|\psi\rangle = \int\psi^*\hat{A}\psi\,\mathrm{d}x$$

**重点：力学量的平均值 $\langle A\rangle = \langle\psi|\hat{A}|\psi\rangle$——由态与算符的内积给出。** 厄米性保证平均值是实数（可观测量的必要条件）。平均值 = 各本征值按概率加权：$\langle A\rangle = \sum_n a_n|c_n|^2$（本征值 $a_n$、概率 $|c_n|^2$）。<span class="marginnote">「平均值的两种算法」：① $\langle A\rangle = \int\psi^*\hat{A}\psi\,\mathrm{d}x$（用态与算符）；② $\langle A\rangle = \sum_n a_n|c_n|^2$（用本征展开系数）。两者等价。期望值不是「测量结果」，而是「多次测量的统计平均」——单次测量得到的是某个本征值（随机）。</span>

## 3 对易子与不确定关系

**对易子（commutator）**：

$$[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}$$

**基本对易关系**（位置与动量）：

$$[\hat{x}, \hat{p}] = i\hbar$$

**重点：基本对易关系 $[\hat{x}, \hat{p}] = i\hbar$——位置与动量不对易，这是不确定关系的代数根源。** 第 101 节的不确定关系 $\Delta x\Delta p \ge \hbar/2$ 可由对易子严格推出（一般形式 $\Delta A\Delta B \ge \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$）。两个可同时精确测量的量必须对易（$[\hat{A},\hat{B}] = 0$）——它们有共同本征态。

**辨析｜易错点：**对易子 $[\hat{x},\hat{p}] = i\hbar$ 中「$\hat{x}\hat{p}$」与「$\hat{p}\hat{x}$」不同（算符作用顺序不可交换）——$\hat{x}\hat{p}\psi = x(-i\hbar\psi')$，$\hat{p}\hat{x}\psi = -i\hbar(x\psi)' = -i\hbar\psi - i\hbar x\psi'$，差 $i\hbar\psi$。这就是 $i\hbar$ 的来源。计算对易子时注意算符的作用次序。

## 4 公式解析：谐振子的平均值

一维谐振子基态波函数 $\psi_0(x) = (\frac{m\omega}{\pi\hbar})^{1/4}e^{-m\omega x^2/(2\hbar)}$。求位置与动量的平均值与方差。

$$
\langle x\rangle = \int_{-\infty}^{\infty}x|\psi_0|^2\mathrm{d}x = 0, \qquad \langle p\rangle = 0
$$

- **第一步，求 $\langle x\rangle$**：$|\psi_0|^2$ 是偶函数，$x|\psi_0|^2$ 是奇函数，积分为零——$\langle x\rangle = 0$（基态对称，平均位置在原点）。
- **第二步，求 $\langle p\rangle$**：$\langle p\rangle = \int\psi_0^*(-i\hbar\frac{\mathrm{d}}{\mathrm{d}x})\psi_0\mathrm{d}x = 0$（$\psi_0$ 为实偶函数，其导数为奇函数）。
- **第三步，求方差**：$\langle x^2\rangle = \frac{\hbar}{2m\omega}$、$\langle p^2\rangle = \frac{m\omega\hbar}{2}$（高斯积分），$\Delta x = \sqrt{\frac{\hbar}{2m\omega}}$、$\Delta p = \sqrt{\frac{m\omega\hbar}{2}}$。
- **第四步，验证不确定关系**：$\Delta x\Delta p = \sqrt{\frac{\hbar}{2m\omega}\cdot\frac{m\omega\hbar}{2}} = \frac{\hbar}{2}$——谐振子基态**达到**不确定关系的下限（最小不确定态）。高斯波包是最小不确定态。

**辨析｜易错点：**$\langle x\rangle = 0$ 不代表「位置确定」（方差 $\Delta x \neq 0$）。平均值是「统计中心」，方差是「统计宽度」——量子态由两者共同刻画。基态谐振子 $\Delta x\Delta p = \hbar/2$ 恰好达到下限，这是高斯态（相干态）的特殊性质。

## 5 算符语言的地位

- **力学量的统一表示**：位置、动量、能量、角动量、自旋都写成算符——量子力学的统一语言；
- **守恒律**：若 $[\hat{A}, \hat{H}] = 0$ 且 $\hat{A}$ 不显含时间，则 $\langle A\rangle$ 守恒——量子守恒律的对易子判据（对应经典泊松括号）；
- **对称性**：哈密顿量的对称性 ⟹ 守恒量（诺特定理的量子版本）；
- **测量理论**：本征值 + 概率 + 塌缩——测量由算符框架描述。

**重点：算符是量子力学的「力学量语言」——一切可观测量的平均值、本征值、守恒性都由算符运算给出。** 量子守恒律的判据 $[\hat{A}, \hat{H}] = 0$ 与经典 $\{A, H\} = 0$（第 118 节）完全对应——算符语言是经典力学哈密顿框架的量子延续。<span class="marginnote">「算符 ↔ 经典量的对应」：$\hat{x}$、$\hat{p}$、$\hat{H}$、$\hat{L}$（角动量）都有经典对应；对易子对应泊松括号（$[\hat{A},\hat{B}] = i\hbar\{A,B\}$）。量子力学不是抛弃经典力学，而是把经典力学（哈密顿、泊松括号）「提升」为算符语言——又一次「旧理论是极限」（$\hbar \to 0$）。</span>

## 6 小结

- **基本算符**：$\hat{x} = x$、$\hat{p} = -i\hbar\partial/\partial x$、$\hat{H} = \hat{T} + \hat{V}$；正则量子化 $p \to -i\hbar\partial/\partial x$。
- **厄米算符**：本征值实数、本征态正交——可观测量的条件。
- **平均值**：$\langle A\rangle = \langle\psi|\hat{A}|\psi\rangle = \sum a_n|c_n|^2$。
- **对易子**：$[\hat{x},\hat{p}] = i\hbar$——不确定关系的代数根源；$[\hat{A},\hat{H}] = 0$ ⟹ $\langle A\rangle$ 守恒。
- 谐振子基态：$\Delta x\Delta p = \hbar/2$（最小不确定态）。

在下一节，我们研究算符的本征值问题——**本征值问题与力学量的本征态**。
