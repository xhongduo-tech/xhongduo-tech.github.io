---
title: 奇异积分方程
date: 2026-08-07
---

# 奇异积分方程

<div class="epigraph">
<p>当核的奇点恰好在积分路径上，积分就必须重新定义——而正是这个重新定义，引出整个复分析最优雅的一章。</p>
<footer>—— 尤利乌斯 · 普莱梅利（Josip Plemelj）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第十二章 ｜ 2026-08-07</p>
</div>

## 为什么需要「重新定义积分」

到目前为止，我们处理的核都是有界函数（或至多有 Abel 型弱奇点）。但有一大类物理问题——**机翼理论、接触力学、裂纹扩展、电磁散射**——导出的核在积分路径上出现**一阶极点**：

$$K(x,t) = \frac{1}{t - x}$$

当 $t = x$ 时核发散，而 $\int_a^b f(t)/(t-x)\,dt$ 按普通 Riemann 或 Lebesgue 积分**不存在**。这类方程称为**奇异积分方程（singular integral equation）**。处理它的第一步，是把那个「不存在的积分」赋予意义——这就是 **Cauchy 主值（Cauchy principal value, CPV）**。<span class="marginnote">机翼升力理论里，环形涡分布沿翼展的诱导速度恰好是 Cauchy 型积分；弦线理论与机翼理论（Prandtl、Glauert、Theodorsen）的核心方程都是奇异积分方程。这不是数学家的猎奇，而是空气动力学的日常。</span>

**Cauchy 主值**：关于 $x$ 对称地抠掉奇点，再取极限

$$\text{P.V.}\int_{a}^{b} \frac{f(t)}{t-x}\, dt := \lim_{\varepsilon \to 0^+} \left[\int_{a}^{x-\varepsilon} + \int_{x+\varepsilon}^{b}\right] \frac{f(t)}{t-x}\, dt$$

对 Hölder 连续的 $f$，这个极限存在。关键在于「对称抠除」：奇点两侧各自发散的部分（对数发散）恰好相互抵消，留下有限主值。**主值改变了积分的「可求和方式」，但不改变被积函数——这是理解一切奇异积分方程的第一课。**

## 1 特征方程：奇异积分方程的主角

**特征方程（characteristic equation）** 是最典型的奇异积分方程：

$$a(x)\, y(x) + \frac{b(x)}{\pi i} \int_{a}^{b} \frac{y(t)}{t-x}\, dt = f(x)$$

这里 $\int$ 一律取 Cauchy 主值。系数 $a(x), b(x)$ 已知，$y$ 是未知函数。<span class="marginnote">之所以叫「特征」，因为更一般的奇异积分方程（带正则核叠加）总可以「剥掉」正则部分，剩下这个最本质的算子 $\left(a(x) + \dfrac{b(x)}{\pi i}S\right)$，其中 $S$ 是 Hilbert 奇异算子 $(Sf)(x) = \int f(t)/(t-x)dt$。解出特征方程，一般方程就是在它身上叠加一个 Fredholm 型扰动。</span>

把算子记作 $\mathcal{K} y = a y + (b/\pi i) S y$。它比 Fredholm 算子多了「乘 $a(x)$」这一项，谱结构也随之改变——**特征方程的指标（index）不再恒为 0，而是由系数的绕数决定**，这是它与 Fredholm 理论最本质的分野。

## 2 Plemelj 公式：把积分方程翻译成解析函数

奇异积分方程与**边界值问题**之间有一座桥，就是 Cauchy 型积分：

$$\Phi(z) = \frac{1}{2\pi i} \int_{a}^{b} \frac{f(t)}{t - z}\, dt, \qquad z \notin [a,b]$$

$\Phi$ 在去掉割线 $[a,b]$ 的平面上解析。让 $z$ 从割线上方（$z = x + i0$）或下方（$z = x - i0$）趋近割线，$\Phi$ 的两个边界值 $\Phi^+(x), \Phi^-(x)$ 满足 **Plemelj–Sokhotski 公式**：

$$\Phi^+(x) - \Phi^-(x) = f(x)$$

$$\Phi^+(x) + \Phi^-(x) = \frac{1}{\pi i}\, \text{P.V.}\int_{a}^{b} \frac{f(t)}{t-x}\, dt$$

第二条公式的右边正是特征方程里的奇异积分。**Plemelj 公式把「奇异积分」与「解析函数在割线两侧的跳变」划上等号**——于是解奇异积分方程，等价于构造一个在割线两侧有指定跳变的解析函数，这正是 Riemann–Hilbert 问题。<span class="marginnote">Plemelj 1908 年给出这两条公式；Sokhotski 更早（1873 年）已在俄国文献中叙述。它们是奇异积分方程理论的操作基石：所有解奇异方程的程序，第一步都是「把它翻译成解析函数问题」。</span>

## 3 公式解析：Plemelj 公式为什么成立

把第一条公式的证明拆开，看主值的魔法在哪里：

$$
\Phi^+(x) - \Phi^-(x) = \lim_{\varepsilon\to0^+} \frac{1}{2\pi i}\left[\int_{a}^{b}\frac{f(t)}{t-x-i\varepsilon}dt - \int_{a}^{b}\frac{f(t)}{t-x+i\varepsilon}dt\right]
$$

- **第一步，合并被积函数**：两个积分相减，被积函数合成

$$\frac{1}{2\pi i}\left(\frac{1}{t-x-i\varepsilon} - \frac{1}{t-x+i\varepsilon}\right) f(t) = \frac{\varepsilon/\pi}{(t-x)^2 + \varepsilon^2}\, f(t)$$

分子分母同乘后，括号里变成著名的 **Poisson 核** $P_\varepsilon(u) = \dfrac{\varepsilon/\pi}{u^2 + \varepsilon^2}$。
- **第二步，认出场次**：$\varepsilon \to 0$ 时 Poisson 核趋向 **Dirac δ 函数**——它把几乎全部质量压到 $u = 0$ 附近，积分 $\int P_\varepsilon(t-x) f(t)dt \to f(x)$。**「跳变等于 $f(x)$」不是魔术，而是 δ 函数的逼近性质。**
- **第三步，看第二条公式**：两边求和时 Poisson 核的奇偶性让 δ 部分抵消，剩下的恰是主值积分——因为 $\int$ 的对称抠除恰好与「上下极限对称取」一致。
- **第四步，记住用法**：对任何要解的奇异方程，把奇异算子 $S$ 换成一侧的 $\Phi^\pm$ 差或和，方程就变成**纯解析函数条件**，可用因式分解求解——这就是下一节的操作。

## 4 特征方程的解：规范化函数与指标

解特征方程的标准程序，是把算子 $\mathcal{K}$ 写成边界值算子。定义

$$D(x) := \frac{a(x) - b(x)}{a(x) + b(x)}$$

并设 $a(x) \neq \pm b(x)$。核心结论：**特征方程的指标为**

$$\kappa = \text{Ind}\, D = \frac{1}{2\pi} \int_{a}^{b} d\arg D(x)$$

即 $D$ 沿区间走一圈的**绕数**（winding number）。$\kappa$ 是一个整数，它决定齐次方程独立解的数量与可解性条件数——**这是「择一」的推广：障碍数与自由度数的差不再恒为 0，而是等于 $\kappa$**。<span class="marginnote">对比 Fredholm 情形（指标恒为 0），奇异方程的指标可以不为零——比如空气动力学里，$\kappa = 1$ 的情形对应「Kutta 条件」：翼型后缘要加一个额外条件才能选出物理上合理的解。指标理论把「择一」升级成了「指标 + 若干边界条件」的完整计数。</span>

解的构造引入**规范化函数（canonical function）**

$$X(z) = \exp\left[\frac{1}{2\pi i}\int_{a}^{b} \frac{\log D(t)}{t-z}\, dt\right]$$

$X$ 在割线两侧的边界值满足 $X^+(x) = D(x)X^-(x)$，把「$D$ 造成的跳变」吸收干净。把 $y$ 表示成 $X$ 的边界值，特征方程就化为普通的 Riemann–Hilbert 问题，再用 Cauchy 积分公式解出。**整个过程可以概括为：主值公式（建立桥）→ 指标（计数自由度）→ 规范化函数（吸收跳变）→ Cauchy 积分（显式解出）。**

实际遇到的方程往往比特征方程多一项**正则核扰动**：

$$a(x)y(x) + \frac{b(x)}{\pi i}\int_{a}^{b}\frac{y(t)}{t-x}dt + \int_{a}^{b} N(x,t)\,y(t)\,dt = f(x)$$

其中 $N$ 是连续核。处理它的标准策略是把特征部分**当作主算子**，把正则部分当作小扰动，通过「先解特征方程、再迭代修正」逐步逼近。**奇异部分决定方程的本质（指标、解的结构），正则部分决定修正量**——这个主次分明的拆解，是解一切奇异方程的通用心法。

## 5 从弱奇异到强奇异：一张对照表

奇异积分方程内部还有层级，初学者最易混淆。把 Abel 型与 Cauchy 型并排：

| | Abel 型（弱奇异） | Cauchy 型（强奇异） |
| --- | --- | --- |
| 核 | $(t-x)^{-\alpha}$，$0\lt \alpha\lt 1$ | $(t-x)^{-1}$ |
| 积分含义 | 普通（Lebesgue）积分 | Cauchy 主值 |
| 典型方程 | $\int_0^x \frac{y(t)}{(x-t)^\alpha}dt = f(x)$ | $a(x)y(x) + \frac{b(x)}{\pi i}\int\frac{y(t)}{t-x}dt = f(x)$ |
| 解的工具 | Abel 反演 / 分数阶微积分 | Plemelj 公式 / Riemann–Hilbert |
| 指标现象 | 无指标 | 有指标 $\kappa$，绕数决定 |
| 物理来源 | 等时曲线 | 机翼理论、接触力学 |

**辨析｜易错点：** 强弱奇异的分界在于**积分是否在普通意义下收敛**。$(t-x)^{-\alpha}$（$\alpha\lt 1$）的积分收敛，只是被积函数无界；$(t-x)^{-1}$ 的积分发散，必须靠主值「对称抠除」拯救。**把主值当普通积分、或在 Abel 型上硬套主值，都会得到错误答案。**

## 6 小结

- **奇异积分方程**的核在路径上有一阶极点，积分须按 **Cauchy 主值**理解：$\text{P.V.}\int = \lim_{\varepsilon\to0}\left[\int_a^{x-\varepsilon}+\int_{x+\varepsilon}^b\right]$。
- **特征方程** $ay + \frac{b}{\pi i}Sy = f$ 是理论的主角，指标 $\kappa = \text{Ind}\,D$ 由系数绕数决定，不再恒为 0。
- **Plemelj 公式**把奇异积分与解析函数的跳变相连：$\Phi^+ - \Phi^- = f$，$\Phi^+ + \Phi^- = \frac{1}{\pi i}\text{P.V.}\int\frac{f(t)}{t-x}dt$——证明的实质是 **Poisson 核逼近 δ 函数**。
- 解特征方程的四步：**主值公式（建立桥）→ 指标计数（数自由度）→ 规范化函数 $X$（吸收跳变）→ Cauchy 积分（显式解出）**。
- 指标 $\kappa = \text{Ind}\,D$ 是「择一」的推广：障碍数与自由度数之差不再恒为 0；空气动力学里 $\kappa = 1$ 对应的正是 **Kutta 条件**这个物理补丁。
- 一般奇异方程 = 特征部分 + 正则核扰动；奇异部分决定本质（指标与解的结构），正则部分决定修正量。
- 强弱奇异之分在于普通积分是否收敛：Abel 型 $(x-t)^{-\alpha}$（$\alpha<1$）弱奇异，Cauchy 型 $(t-x)^{-1}$ 必须用主值。

在下一节，我们把奇异积分方程送上 Fourier 舞台：当区域退化成半直线、核退化成位移核，Plemelj 公式就翻面成 **Wiener–Hopf 方法的分解与延拓**。