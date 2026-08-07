---
title: 线性谐振子
date: 2026-08-07
---

# 线性谐振子

<div class="epigraph">
<p>当我发现量子谐振子的本征值可以用一个简单的递推关系求出时，我的心情是难以描述的。</p>
<footer>—— 保罗 · 狄拉克（Paul Dirac）</footer>
</div>

<div class="article-byline">
<p>第二级 · 量子力学 ｜ 曾谨言《量子力学》卷一 第3章 / Griffiths《量子力学概论》§2.3 ｜ 2026-08-07</p>
</div>

## 为什么从线性谐振子开始

无限深势阱是一次「硬碰硬」的量子化，而现实世界里更常见的是**光滑的势能**。任何平衡点附近的势能曲线，泰勒展开后最低阶近似都是抛物线 $V(x) \propto x^2$——这就是谐振子势。弹簧、摆、分子内原子振动、晶格振动、电磁场的每个模式，全都落在谐振子的框架里。<span class="marginnote">谐振子是量子力学的「万能近似」：固体比热、激光、量子场论、甚至超导的 BCS 理论，都从量子谐振子出发。费曼曾统计，物理学里一大半问题最终都归结为谐振子。它是继势阱之后第二个必须掌握的精确可解模型。</span>更妙的是，它能被两种完全不同的方法解出——解析法与代数法，后者将引出一整套贯穿量子力学的**升降算符**技术。

## 1 势能与定态方程

一维谐振子的势能为：

$$
V(x) = \frac{1}{2}m\omega^2 x^2
$$

其中 $\omega$ 是经典角频率（由劲度系数 $k$ 与质量 $m$ 决定：$\omega = \sqrt{k/m}$）。势能是抛物线，在 $x=0$ 处有最小值。经典粒子在抛物线势场中来回振动，能量可连续取任意非负值。

定态薛定谔方程：

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2 x^2\,\psi = E\psi
$$

<span class="marginnote">与无限深势阱不同，谐振子的势能处处有限，粒子可以「隧穿」到经典允许区域之外——波函数在 $|x|$ 很大处指数衰减但永不严格为零。这意味着「粒子出现在经典不能到达的地方」是谐振子的常态，这一点在经典极限下才逐渐消失。</span>这个方程有两个参数 $m$、$\omega$，加上 $\hbar$，可以组合出**特征长度**：

$$
\xi = \sqrt{\frac{m\omega}{\hbar}}\,x, \qquad \varepsilon = \frac{2E}{\hbar\omega}
$$

做变量代换后，方程化成无参数的标准形式，求解将变得纯粹。

## 2 解析法：厄米多项式

引入无量纲变量 $\xi = \sqrt{\frac{m\omega}{\hbar}}x$，方程化为：

$$
\frac{d^2\psi}{d\xi^2} + (\varepsilon - \xi^2)\psi = 0
$$

标准的求解思路是「先剥出大 $|\xi|$ 处的渐近行为，再展开幂级数」：

- **渐近解**：当 $|\xi| \to \infty$，方程近似为 $\psi'' - \xi^2\psi = 0$，物理上可接受的解为 $\psi \sim e^{-\xi^2/2}$（另一个 $e^{+\xi^2/2}$ 发散，舍去）。
- **代回**：设 $\psi(\xi) = u(\xi)e^{-\xi^2/2}$，代入后 $u$ 满足厄米方程。
- **级数收敛条件**：$u$ 的幂级数必须在中途截断，否则会指数发散。截断条件恰好给出量子化能量。

截断条件导致 $\varepsilon = 2n + 1$，即：

$$
E_n = \left(n + \frac{1}{2}\right)\hbar\omega, \qquad n = 0, 1, 2, \dots
$$

对应的波函数是高斯函数乘**厄米多项式（Hermite polynomials）** $H_n(\xi)$：

$$
\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4}\frac{1}{\sqrt{2^n n!}}\,H_n(\xi)\,e^{-\xi^2/2}
$$

前几个厄米多项式：$H_0 = 1$，$H_1 = 2\xi$，$H_2 = 4\xi^2 - 2$，$H_3 = 8\xi^3 - 12\xi$。<span class="marginnote">厄米多项式是正交多项式家族的一员，同族还有勒让德多项式（角动量部分用）、拉盖尔多项式（氢原子径向部分用）。量子力学里三大可解模型——谐振子、角动量、氢原子——恰好对应三套特殊函数，这是「特殊函数论」在物理里最集中的一次亮相。</span>

## 3 代数法：升降算符

解析法直来直去但计算繁重。狄拉克给出了更优雅的代数方法。定义两个**阶梯算符**：

$$
\hat{a} = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} + \frac{i}{m\omega}\hat{p}\right), \qquad
\hat{a}^\dagger = \sqrt{\frac{m\omega}{2\hbar}}\left(\hat{x} - \frac{i}{m\omega}\hat{p}\right)
$$

$\hat{a}$ 叫**湮灭算符（lowering / annihilation operator）**，$\hat{a}^\dagger$ 叫**产生算符（raising / creation operator）**。它们的作用是改变量子数：

$$
\hat{a}^\dagger\psi_n = \sqrt{n+1}\,\psi_{n+1}, \qquad \hat{a}\,\psi_n = \sqrt{n}\,\psi_{n-1}
$$

即 $\hat{a}^\dagger$ 把能级升一档，$\hat{a}$ 把能级降一档——这是「量子」一词最形象的体现：能量只能跳整数档。进一步定义**粒子数算符** $\hat{N} = \hat{a}^\dagger\hat{a}$，哈密顿量可以写成：

$$
\hat{H} = \hbar\omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\left(\hat{N} + \frac{1}{2}\right)
$$

<span class="marginnote">$\hat{N}$ 的本征值是非负整数 $n$，它「数出」量子数。这套「产生/湮灭 + 粒子数」的语言，在量子场论里被推广为真正的粒子产生与湮灭算符——那里的「$n$ 个粒子」与谐振子的「$n$ 个激发量子」共享同一套数学。学通谐振子的代数法，等于预习了量子场论的第一课。</span>

代数法的威力在于：不需要解微分方程，只需利用 $\hat{N}$ 的正定性（本征值不能为负）就能推出能级公式和整个谱结构。

## 4 公式解析：从对易关系到能级

把代数法的逻辑链完整拆开：

$$
[\hat{a}, \hat{a}^\dagger] = 1, \qquad \hat{H} = \hbar\omega\!\left(\hat{N} + \frac{1}{2}\right), \qquad E_n = \left(n+\frac{1}{2}\right)\hbar\omega
$$

- **第一步，对易关系 $[\hat{a}, \hat{a}^\dagger] = \hat{a}\hat{a}^\dagger - \hat{a}^\dagger\hat{a} = 1$**：这是整条链的起点。它来自 $[\hat{x}, \hat{p}] = i\hbar$。$\hat{a}$ 与 $\hat{a}^\dagger$ **不对易**——正是不对易造成了能级之间的等差结构。
- **第二步，$\hat{N} = \hat{a}^\dagger\hat{a}$ 的作用**：用对易关系可以证明 $\hat{N}\psi_n = n\psi_n$。若 $\psi_n$ 是 $\hat{N}$ 的本征态，则 $\hat{N}(\hat{a}^\dagger\psi_n) = (n+1)(\hat{a}^\dagger\psi_n)$——$\hat{a}^\dagger\psi_n$ 仍是 $\hat{N}$ 的本征态，本征值加一。这就是「阶梯」的机制。
- **第三步，正定性约束**：$n = \langle \psi_n | \hat{N} | \psi_n \rangle = \|\hat{a}\psi_n\|^2 \ge 0$。粒子数本征值非负。若 $n < 0$ 可以无休止下降，但下降次数被正定性封顶——最低态 $\psi_0$ 满足 $\hat{a}\psi_0 = 0$。
- **第四步，零基态**：解 $\hat{a}\psi_0 = 0$ 得 $\psi_0 \propto e^{-m\omega x^2/2\hbar}$，对应的能量 $E_0 = \frac12\hbar\omega$。从 $\psi_0$ 出发逐级上升 $\psi_n = \frac{1}{\sqrt{n!}}(\hat{a}^\dagger)^n\psi_0$，能级公式 $E_n = (n+\frac12)\hbar\omega$ 全部浮现。

**关键结论：基态能量不是零，而是 $\frac12\hbar\omega$**——这是**零点能（zero-point energy）**，与不确定度关系 $\Delta x\,\Delta p \ge \hbar/2$ 严格对应：粒子不能同时停在 $x=0$ 且动量确定为零，最低能量必须保留 $\frac12\hbar\omega$。

## 5 谐振子的应用

- **等间距能级**：$E_n = (n+\frac12)\hbar\omega$，相邻能级间距恒为 $\hbar\omega$。这使谐振子成了「量子」的天然字典——吸收一个光子 $\hbar\omega$ 升一档，放出一个光子 $\hbar\omega$ 降一档。电磁场模式的量子化（光子）正是谐振子能级量子化的翻版。<span class="marginnote">普朗克当年假设「振子能量只能取 $nh\nu$」，却不含零点能 $\frac12h\nu$——差一个常数不影响辐射谱推导，但物理上零点能至关重要：它解释了为什么液氦在常压下不凝固（零点振动能足以阻止晶格冻结）、以及卡西米尔效应等量子现象。</span>
- **能级结构与谱**：谐振子谱只有一条谱线频率 $\omega$（因为等间距），这在分子振动光谱里表现为一个特征吸收峰——红外光谱识别化学键的基础。
- **量子涨落**：基态波函数 $\psi_0 \propto e^{-\xi^2/2}$ 是高斯型，位置分布 $\Delta x = \sqrt{\hbar/2m\omega}$，动量分布同样高斯。这是「最小不确定态」——量子信息与量子光学里相干态的地基。

## 6 小结

- **谐振子势** $V(x) = \frac12 m\omega^2 x^2$ 是任何光滑势能在平衡点附近的普适近似。
- 解析法给出**厄米多项式解** $\psi_n \propto H_n(\xi)e^{-\xi^2/2}$，能级等间距 $E_n = (n+\frac12)\hbar\omega$。
- **代数法**用升降算符 $\hat{a}^\dagger, \hat{a}$ 和粒子数算符 $\hat{N}$ 重构整个谱，对易关系 $[\hat{a},\hat{a}^\dagger] = 1$ 是机制核心。
- **零点能** $\frac12\hbar\omega$：基态能量不为零，由不确定度关系保证，是普朗克能量子 $nh\nu$ 未含的常数修正。
- 谐振子框架贯穿量子场论、固体物理、量子光学，是「量子化的通用模板」。

在下一节，我们转向一个「量子味道」更浓的奇观：粒子能穿过能量比它更高的势垒——**隧道效应**。它在核衰变、扫描隧道显微镜、闪存芯片里无处不在。
