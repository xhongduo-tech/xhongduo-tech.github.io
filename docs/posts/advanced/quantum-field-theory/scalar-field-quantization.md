---
title: 标量场量子化
date: 2026-08-07
---

# 标量场量子化

<div class="epigraph">
<p>量子力学就是统计诠释的力学；场论则是把这种诠释贯彻到无穷多自由度的力学。</p>
<footer>—— 帕斯夸尔 · 约当（Pascual Jordan），引自其场论工作</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §2.3 ｜ 2026-08-07</p>
</div>

## 为什么从谐振子开始量子化

上一节的克莱因-戈登场 $\phi(\boldsymbol{x},t)$ 还是经典对象——一个在时空里取值、服从欧拉-拉格朗日方程的「数」。
要把它变成量子对象，P&S 采用的方法既不是「猜波函数」也不是「猜算符」，而是**取经典力学的正则量子化（canonical quantization）程序，原封不动地搬到无穷多自由度**：把场 $\phi$ 与它的共轭动量 $\pi$ 提升为算符，并施加对易关系。
而这一切的技术起点，是量子力学里最古老的玩具——**谐振子**。<span class="marginnote">为什么谐振子如此万能？因为它是最简单的二次型势能 $V = \frac12\omega^2 q^2$，其量子化给出均匀的能级间隔，而任何场在平面波分解后都是无穷多个独立谐振子的叠加。<strong>学懂了谐振子，就学懂了整个自由场论。</strong></span>

## 1 正则量子化的纲领

在量子力学里，正则量子化三步走：找出正则坐标 $q$ 与其共轭动量 $p = \partial L/\partial\dot{q}$；把两者提升为算符；施加**正则对易关系** $[q, p] = i$。对场，这一步几乎照抄，只是 $q$ 换成场 $\phi(\boldsymbol{x})$，$p$ 换成共轭动量密度：

$$\pi(\boldsymbol{x}) = \frac{\partial \mathcal{L}}{\partial \dot{\phi}(\boldsymbol{x})} = \dot{\phi}(\boldsymbol{x})$$

对实标量场 $\mathcal{L} = \frac12 \dot\phi^2 - \frac12(\nabla\phi)^2 - \frac12 m^2\phi^2$，共轭动量恰好就是 $\dot\phi$。**等时正则对易关系（equal-time commutation relations）**为：

$$[\phi(\boldsymbol{x}), \pi(\boldsymbol{y})] = i\,\delta^{(3)}(\boldsymbol{x} - \boldsymbol{y}), \qquad [\phi,\phi] = [\pi,\pi] = 0$$

这里 $\delta^{(3)}$ 是三维狄拉克 $\delta$ 函数，它取代了量子力学里克罗内克 $\delta_{ij}$——因为场有连续无穷多个自由度，每个时空点 $\boldsymbol{x}$ 都是一台「独立的振子」。<span class="marginnote">同一时刻、不同空间点的两个场算符对易（$[\phi(\boldsymbol{x}),\phi(\boldsymbol{y})]=0$），这是相对论因果性的体现：类空分离的两点没有任何物理量可以即时通讯，测量互不干扰。</span>

## 2 平面波展开：把场拆成无穷多个谐振子

自由场满足线性方程 $(\Box + m^2)\phi = 0$，所以最自然的做法是把 $\phi$ 按平面波展开。因为场算符是厄米的（$\phi^\dagger = \phi$，实标量场），展开必须把正频与负频部分配对，写成：

$$\phi(\boldsymbol{x}, t) = \int \frac{d^3p}{(2\pi)^3} \frac{1}{\sqrt{2E_{\boldsymbol{p}}}}\left( a_{\boldsymbol{p}} e^{-ip\cdot x} + a_{\boldsymbol{p}}^\dagger e^{ip\cdot x} \right)$$

其中 $E_{\boldsymbol{p}} = \sqrt{|\boldsymbol{p}|^2 + m^2}$ 是单个量子的能量，$p\cdot x = E_{\boldsymbol{p}} t - \boldsymbol{p}\cdot\boldsymbol{x}$。<span class="marginnote">归一化系数 $1/\sqrt{2E_{\boldsymbol{p}}}$ 与 $(2\pi)^{-3}$ 是「洛伦兹不变的动量空间测度」的配套选择，后面算散射截面时它会在相空间里再次出场。现在只需知道：这么选让所有公式保持洛伦兹协变。</span>

**核心解读：$a_{\boldsymbol{p}}$ 与 $a_{\boldsymbol{p}}^\dagger$ 不再是普通系数，而是算符。** 把展开式代入对易关系，用平面波的正交性，得到**升降算符代数**：

$$[a_{\boldsymbol{p}}, a_{\boldsymbol{q}}^\dagger] = (2\pi)^3 \delta^{(3)}(\boldsymbol{p} - \boldsymbol{q}), \qquad [a_{\boldsymbol{p}}, a_{\boldsymbol{q}}] = [a_{\boldsymbol{p}}^\dagger, a_{\boldsymbol{q}}^\dagger] = 0$$

这正是谐振子升降算符 $[a,a^\dagger]=1$ 的「连续版」——对每个动量 $\boldsymbol{p}$ 都有一台独立的谐振子。

## 3 哈密顿量与多粒子态

把场算符展开代入哈密顿量 $H = \int d^3x\, \mathcal{H}$，利用对易关系化简，得到：

$$H = \int \frac{d^3p}{(2\pi)^3} E_{\boldsymbol{p}}\left( a_{\boldsymbol{p}}^\dagger a_{\boldsymbol{p}} + \tfrac12 [a_{\boldsymbol{p}}, a_{\boldsymbol{p}}^\dagger] \right)$$

忽略那个发散的常数项 $\frac12 \delta^{(3)}(0)$（零能标定问题，见下节辨析），哈密顿量就是每台谐振子的能量 $E_{\boldsymbol{p}}$ 乘粒子数。
定义**数算符** $N_{\boldsymbol{p}} = a_{\boldsymbol{p}}^\dagger a_{\boldsymbol{p}}$，它数出动量 $\boldsymbol{p}$ 的量子有多少个。
于是：

- **真空态（vacuum）** $|0\rangle$：被所有 $a_{\boldsymbol{p}}$ 湮灭，$a_{\boldsymbol{p}}|0\rangle = 0$，能量为 0。
- **单粒子态** $|\boldsymbol{p}\rangle = a_{\boldsymbol{p}}^\dagger|0\rangle$：能量 $E_{\boldsymbol{p}} = \sqrt{\boldsymbol{p}^2 + m^2}$，正是相对论质壳条件——**量子化的场自动给出相对论粒子**。
- **多粒子态** $a_{\boldsymbol{p}}^\dagger a_{\boldsymbol{q}}^\dagger|0\rangle$：两个粒子，因为算符可交换，交换两个粒子得到同一状态——**标量场自动满足玻色统计**。

这一节的全部惊讶浓缩成一句话：**我们从来没有「放入」粒子，只量子化了一个场，粒子就自己冒出来了。**<span class="marginnote">粒子是场的量子激发，不是独立实体——这是 QFT 相对量子力学的本体论跃迁。同一场可以容纳任意多个粒子（因为 $a^\dagger$ 可反复作用），粒子数不再守恒。守恒的只剩总能量与总动量。</span>

## 4 公式解析：场算符的平面波展开

**平面波展开是正则量子化的「交接仪式」：它把经典场的每个傅里叶分量翻译成一台量子谐振子。** 我们拆解三步：

$$
\phi(\boldsymbol{x}, t) = \int \frac{d^3p}{(2\pi)^3} \frac{1}{\sqrt{2E_{\boldsymbol{p}}}}\left( a_{\boldsymbol{p}} e^{-ip\cdot x} + a_{\boldsymbol{p}}^\dagger e^{ip\cdot x} \right)
$$

- **第一步，为什么正负频配对**：实场满足 $\phi^\dagger = \phi$。傅里叶分量 $e^{-ip\cdot x}$ 与 $e^{+ip\cdot x}$ 是共轭关系，所以系数必须成对出现为 $a_{\boldsymbol{p}}$ 与 $a_{\boldsymbol{p}}^\dagger$——单有 $a_{\boldsymbol{p}} e^{-ip\cdot x}$ 的场不是厄米的。这正对应着「一个粒子 + 一个反粒子」的成对出现。
- **第二步，读出 $e^{-ip\cdot x}$ 的物理**：$e^{-ip\cdot x} = e^{-i(E_{\boldsymbol{p}}t - \boldsymbol{p}\cdot\boldsymbol{x})}$ 是正频平面波，能量为 $+E_{\boldsymbol{p}}$。$a_{\boldsymbol{p}}$ 乘上它 = 从态中**移除**一个能量 $E_{\boldsymbol{p}}$ 的量子（湮灭）；$a_{\boldsymbol{p}}^\dagger$ 乘上 $e^{+ip\cdot x}$ = **注入**一个量子（产生）。频率的正负与算符的产生/湮灭一一对应。
- **第三步，归一化因子 $1/\sqrt{2E_{\boldsymbol{p}}}$**：它来自对易关系的自洽要求。把展开式代入 $[\phi,\pi]=i\delta^{(3)}$，会推出 $[a_{\boldsymbol{p}},a_{\boldsymbol{q}}^\dagger] = (2\pi)^3 2E_{\boldsymbol{p}}\,\delta^{(3)}(\boldsymbol{p}-\boldsymbol{q})$ 才对；若去掉 $1/\sqrt{2E_{\boldsymbol{p}}}$，对易关系会多出 $2E_{\boldsymbol{p}}$ 因子。P&S 选择把它「预支」在展开系数里，让 $a$ 代数的形式最干净。

## 5 辨析｜易错点

- **$a_{\boldsymbol{p}}$ 是湮灭算符还是升降算符**：在谐振子语境里叫「降算符」，在场论里叫「湮灭算符」——同一数学对象，语义不同：前者降低量子数，后者消灭粒子。**不要混淆 $a_{\boldsymbol{p}}$（消灭动量 $\boldsymbol{p}$ 的量子）与 $\phi$（场算符本身，同时含产生与湮灭两部分）。**
- **真空不是「空无一物」**：$|0\rangle$ 被 $a_{\boldsymbol{p}}$ 湮灭，但场的真空涨落（零点能、虚粒子）依然存在。说「真空=没有粒子」是准的，说「真空=什么都没有」就不准了。<span class="marginnote">$H$ 里的 $\frac12[a_{\boldsymbol{p}},a_{\boldsymbol{p}}^\dagger]$ 项给出 $\int \frac12 E_{\boldsymbol{p}}\delta^{(3)}(0)$，是无穷大的零点能。物理里我们永远只测量<strong>能量差</strong>，所以把它丢进「重新标定零点」的筐里——这正是第三章重正化的第一个预兆。</span>
**把 $\phi(\boldsymbol{x})$ 当成「位置」**：场算符不是坐标算符，它的「本征值」没有位置的意义。说「粒子在 $\boldsymbol{x}$ 处」要用局域场算符 $\phi(\boldsymbol{x})|0\rangle$ 构造，但这一状态并非位置本征态——粒子的位置概念在 QFT 里是模糊的，这与量子力学里 $\hat{x}$ 的位置本征态截然不同。

## 6 延伸：谐振子语言的统一威力

这一章的全部技术可以压缩成一句话：**相对论场论 = 无穷多台谐振子，各自动量标签 $\boldsymbol{p}$，共享一个真空。** 谐振子之所以万能，是因为它抓住了一切二次型系统的共同谱：等间距能级、升降算符、数算符。自由场正是二次型（$\phi^2$ 型拉格朗日量），所以它分解成独立谐振子；相互作用（$\phi^4$ 等）打破了「独立」，才需要微扰论。

从谐振子视角，三个结论几乎是「免费」的：

**玻色统计**：$[a,a^\dagger]=1$ 允许一个模式装任意多量子——谐振子的「多光子占据」图像。
**真空零点能**：$H = \sum E_{\boldsymbol{p}}(a^\dagger a + \frac12)$ 里的 $\frac12$ 是每台谐振子的零点能，全模式求和发散——重正化的第一个入口。
**产生算符的物理**：$a_{\boldsymbol{p}}^\dagger|0\rangle$ 就是「把动量 $\boldsymbol{p}$ 的那台谐振子激发到第一能级」，它的能量 $E_{\boldsymbol{p}}$ 正是相对论质壳。

这套语言往后的每一次出场（光子、胶子、希格斯、以及凝聚态里的声子、磁振子）都是「换汤不换药」。

### 自测清单

[ ] 能写出等时对易关系 $[\phi(\boldsymbol{x}),\pi(\boldsymbol{y})] = i\delta^{(3)}$ 并解释 $\delta$ 函数为何出现。
[ ] 能从展开式推出 $[a_{\boldsymbol{p}},a_{\boldsymbol{q}}^\dagger] = (2\pi)^3\delta^{(3)}$。
[ ] 能解释真空 $|0\rangle$ 的定义与零点能的「重新标定」。
[ ] 能说出单粒子态 $a^\dagger|0\rangle$ 的能量为何是质壳关系。
[ ] 能区分「场算符 $\phi$」与「湮灭算符 $a_{\boldsymbol{p}}$」。

<span class="marginnote">把「每台谐振子」的图像刻进脑子：之后读费曼图、算传播子、理解真空涨落，都是在给这台无限维的谐振子系统记账。</span>

### 延伸阅读指引

- 深化推导：P&S §2.3 的标量场量子化、§2.4 的传播子；想理解「谐振子分解」的普适性可对比量子力学谐振子章节。
- 实践：把 $H$ 用 $a^\dagger a$ 重写并验证零点能；用 $[a_p,a_q^\dagger]$ 检查对易关系自洽。
- 联系主线：场 = 无穷谐振子是「把一个连续对象分解成离散模」的范本——与《信号处理》里「傅里叶分解」、以及《推荐系统》里「矩阵分解」是同一数学动作。

## 7 小结

- 正则量子化把 $\phi$ 与 $\pi$ 提升为算符，满足等时对易关系 $[\phi(\boldsymbol{x}),\pi(\boldsymbol{y})] = i\delta^{(3)}(\boldsymbol{x}-\boldsymbol{y})$。
- 平面波展开把场拆成无穷多谐振子，系数 $a_{\boldsymbol{p}}, a_{\boldsymbol{p}}^\dagger$ 成为产生/湮灭算符。
- 哈密顿量 $H = \int \frac{d^3p}{(2\pi)^3}E_{\boldsymbol{p}}\, a_{\boldsymbol{p}}^\dagger a_{\boldsymbol{p}}$（加零点能常数），真空被所有 $a$ 湮灭。
- **粒子 = 场的量子激发**：$a_{\boldsymbol{p}}^\dagger|0\rangle$ 自动给出相对论质壳粒子 $E = \sqrt{\boldsymbol{p}^2 + m^2}$，多粒子态自动满足玻色统计。
- 无穷大的零点能是重正化的第一个预兆。

在下一节，我们把实标量场推广到**复标量场**，看「带电」如何出现——场内部的相位对称性如何制造出「粒子-反粒子」对与守恒电荷。


