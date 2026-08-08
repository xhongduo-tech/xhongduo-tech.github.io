---
title: 实标量场与复标量场
date: 2026-08-07
---

# 实标量场与复标量场

<div class="epigraph">
<p>对称性规定着相互作用的形式；一个复场的相位就是它带电的记号。</p>
<footer>—— 赫尔曼 · 外尔（Hermann Weyl），对称性思想之集大成</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §2.2, §2.5 ｜ 2026-08-07</p>
</div>

## 为什么区分实场与复场

上一节的实标量场已经造出了「一个动量一个粒子」的玻色子，但它只能描述**中性粒子**——$\pi^0$ 介子、希格斯玻色子，它们与自己的反粒子是同一个。
可自然界还有带电的玻色子：$\pi^\pm$ 介子、$W^\pm$ 玻色子，它们有明确的「粒子-反粒子」之别。
把实场换成**复标量场**，只做一个小手术——让场取复数值——就同时获得了两样新东西：**电荷**与**粒子-反粒子对**。<span class="marginnote">这场「手术」的实质是给场增加一个内部自由度（相位），并让拉格朗日量在这个相位旋转下不变。微观里多了一个 $U(1)$ 对称性，宏观里就多了一条守恒律——电荷。这又是诺特定理在还债。</span>

## 1 实标量场的「中性」品格

先看清楚实标量场的边界。实场满足厄米条件 $\phi^\dagger = \phi$，它的平面波展开里 $a_{\boldsymbol{p}}$ 与 $a_{\boldsymbol{p}}^\dagger$ 共轭成对，产生一个量子与湮灭一个量子用的是同一个算符。这意味着：

- **粒子 = 反粒子**：场只对应一种量子，不存在「反粒子」这个独立概念。
- **无电荷**：场的整体相位没有意义（实数相位只有 $\pm 1$），没有连续的内部对称性，诺特电荷恒为零。
- 能标量：$\pi^0$、Higgs、真实的赝标量介子，都属于这一类。

实场是「最省料的场」——一个洛伦兹标量、一个自由度、一种粒子。它的拉格朗日量我们已经见过：

$$\mathcal{L} = \tfrac12 (\partial_\mu\phi)(\partial^\mu\phi) - \tfrac12 m^2\phi^2$$

## 2 构造复标量场：两个实场叠加

**复标量场（complex scalar field）**：把两个独立的实场 $\phi_1, \phi_2$ 拼成一个复数：

$$\phi = \frac{1}{\sqrt{2}}(\phi_1 + i\phi_2), \qquad \phi^\dagger = \frac{1}{\sqrt{2}}(\phi_1 - i\phi_2)$$

$1/\sqrt{2}$ 保证两个实场的动能项 $\frac12(\partial\phi_1)^2 + \frac12(\partial\phi_2)^2$ 恰好拼成一个干净的形式 $\partial_\mu\phi^\dagger \partial^\mu\phi$。复场的拉格朗日量取：

$$\mathcal{L} = (\partial_\mu\phi^\dagger)(\partial^\mu\phi) - m^2 \phi^\dagger\phi$$

把 $\phi = (\phi_1+i\phi_2)/\sqrt2$ 代进去，展开后正好还原成两个独立实标量场的拉格朗日量之和——**复场 = 两个质量相同的实场**，自由度从 1 变成 2。<span class="marginnote">两种读法等价：数学上它是两个实场，物理上它是「一个带正电荷的粒子 + 一个带负电荷的反粒子」。自由度的总数不变（2 个实自由度），只是重组了描述方式。</span>

## 3 整体 U(1) 对称性与守恒电荷

复场最关键的属性是它的相位对称性：把场整体转一个相位角 $\alpha$，

$$\phi \to e^{i\alpha}\phi, \qquad \phi^\dagger \to e^{-i\alpha}\phi^\dagger$$

拉格朗日量每一项都含 $\phi^\dagger\phi$ 或 $|\partial\phi|^2$，相位恰好抵消——**拉格朗日量在整体 $U(1)$ 旋转下不变**。由诺特定理，存在守恒流与守恒荷：

$$j^\mu = i(\phi^\dagger \partial^\mu \phi - \phi\,\partial^\mu\phi^\dagger), \qquad Q = \int d^3x\, j^0$$

把平面波展开代进去算 $Q$，会得到一件漂亮的事：

$$Q = \int \frac{d^3p}{(2\pi)^3}\left( a_{\boldsymbol{p}}^\dagger a_{\boldsymbol{p}} - b_{\boldsymbol{p}}^\dagger b_{\boldsymbol{p}} \right)$$

复场的展开需要**两套算符**：$\phi$ 里的正频项系数是 $a_{\boldsymbol{p}}$（对应「粒子」），负频项系数是 $b_{\boldsymbol{p}}^\dagger$（对应「反粒子」）。
于是 $Q$ 数的是「粒子数减反粒子数」——**电荷 = 粒子数 − 反粒子数**。正负电荷因此成对出现、守恒、可由能量凭空创生。<span class="marginnote">「正反粒子对凭空创生」正是 $e^+e^-$ 对撞机里 $\gamma \to e^+e^-$ 的机制：能量转成一对电荷相反、总电荷为零的粒子。电荷守恒是整体 $U(1)$ 对称性的直接后果。</span>

## 4 公式解析：复标量场拉格朗日量与诺特流

**复标量场拉格朗日量是「最小作用量原理 + 内部对称性」完美合作的标本**，拆三步：

$$
\mathcal{L} = (\partial_\mu\phi^\dagger)(\partial^\mu\phi) - m^2\phi^\dagger\phi, \qquad j^\mu = i\left(\phi^\dagger\partial^\mu\phi - \phi\,\partial^\mu\phi^\dagger\right)
$$

- **第一步，为什么没有 $1/2$ 因子**：实场的 $\frac12(\partial\phi)^2$ 展开后给出两个实场的动能，而复场写成 $\partial\phi^\dagger\partial\phi$ 时已直接等于 $\frac12[(\partial\phi_1)^2 + (\partial\phi_2)^2]$，所以不再需要 $1/2$。系数是为「两个自由度」付出的对价。
- **第二步，读出 $\phi^\dagger\phi$ 项**：$m^2\phi^\dagger\phi$ 是质量项。它对 $\phi^\dagger$ 求导时按 $\partial\mathcal{L}/\partial\phi^\dagger = -m^2\phi$ 处理；对 $\partial_\mu\phi^\dagger$ 求导得到 $\partial^\mu\phi$。正是这两项代入欧拉-拉格朗日方程，消出 $(\Box + m^2)\phi = 0$——复场每个分量都满足克莱因-戈登方程。
- **第三步，读诺特流的符号结构**：$j^\mu = i(\phi^\dagger\partial^\mu\phi - \phi\,\partial^\mu\phi^\dagger)$。$\partial^\mu$ 在中间、$\phi^\dagger$ 在左，保证 $j^\mu$ 是实数。对 $\phi \to e^{i\alpha}\phi$ 这个对称性，无穷小变换 $\Delta\phi = i\phi$，代入一般诺特流公式 $j^\mu = \frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)}\Delta\phi + \text{cc}$ 即得——**流的方向由相位的「旋转方向」决定，正电荷与负电荷因此互为镜像**。

## 5 辨析｜易错点

- **复场不是「两个场」而是「两种粒子」**：在量子化的语境下，$\phi_1$ 与 $\phi_2$ 是数学拆分，物理上它们是同一个 $U(1)$ 多重态里的粒子与反粒子。用 $\phi_1, \phi_2$ 语言时，$U(1)$ 对称性「隐身」了；用 $\phi, \phi^\dagger$ 语言时它显形。<span class="marginnote"><strong>中性 vs 带电玻色子</strong>：$\pi^0$ 用实场描述（粒子=反粒子）；$\pi^+,\pi^-$ 用复场描述（互为反粒子、电荷相反）。同一种粒子不能同时既是实场又是复场——这是「哪种场」由荷谱决定的例子。</span>
**把 $a^\dagger$ 与 $b^\dagger$ 混淆**：$a^\dagger$ 产生粒子（带正电）、$b^\dagger$ 产生反粒子（带负电）。若误把两者当成同一个，电荷算符会变成「总粒子数」而非「粒子数减反粒子数」，电荷守恒就消失了。
**整体 $U(1)$ 不是规范对称性**：这里 $\alpha$ 是常数（与时空点无关）。只有让 $\alpha(x)$ 随点变化、并引入光子场去「补救」，才是规范对称性——那是第四章《非阿贝尔规范场》的主题，现在还早。

## 6 延伸：带电与中性的两种场，一处哲学

实场与复场的选择不是技术细节，而是**「粒子是否与自己的反粒子相同」的场论编码**。对照表：

| 性质 | 实标量场 | 复标量场 |
| --- | --- | --- |
| 场条件 | $\phi^\dagger = \phi$ | $\phi^\dagger \neq \phi$ |
| 自由度 | 1 个实场 | 2 个实场（$\phi_1,\phi_2$） |
| 粒子/反粒子 | 同一 | 成对（$a^\dagger$ vs $b^\dagger$） |
| 内部对称 | 无连续对称 | 整体 $U(1)$ |
| 守恒荷 | 无 | $Q = N_+ - N_-$ |
| 物理例子 | $\pi^0$、Higgs | $\pi^\pm$、$W^\pm$、带电标量 |

深层教训：**「有没有电荷」取决于场有没有「能被相位旋转的方向」**。一个场若只在一个「实方向」取值，就没有相位可转，也就没有电荷。电荷守恒不是「物质的性质」，而是「场结构的对称性」的投影——这正是诺特定理最漂亮的回响。

另外提醒：复场的 $U(1)$ 现在还是**整体**的（$\alpha$ 常数）。把它升级成**定域**的（$\alpha(x)$ 逐点变），就必须引入光子——这正是第四章规范场论的入场方式。当前章与第四章之间，隔着一个「把对称性从整体变局域」的思维跃迁。

### 自测清单

[ ] 能写出复标量场拉格朗日量并验证整体 $U(1)$ 不变。
[ ] 能推出诺特流 $j^\mu = i(\phi^\dagger\partial^\mu\phi - \phi\partial^\mu\phi^\dagger)$。
[ ] 能解释为什么 $Q$ 数「粒子数减反粒子数」。
[ ] 能说出「复场 = 两个同质量实场」与「两种粒子」的等价性。

<span class="marginnote">记住这个跃迁的预告：<strong>整体 $U(1)$ 守恒是电荷，定域 $U(1)$ 是电磁</strong>。一个对称性的「局域化」会把守恒荷变成相互作用——这是规范原理的心脏。</span>

### 延伸阅读指引

- 深化推导：P&S §2.2 的复标量场、§2.5 的守恒荷；电荷守恒与 $U(1)$ 对称的严格对应见诺特定理章节。
- 实践：写出复场展开里的 $a, b$ 两套算符并验证 $Q$ 数粒子数减反粒子数。
- 联系主线：「实场 ↔ 中性、复场 ↔ 带电」是「自由度结构决定守恒荷」的第一课——与《线性代数》里「对称性决定可对角化结构」同源。

## 7 小结

- **实标量场** $\phi^\dagger = \phi$：中性粒子，粒子 = 反粒子，无电荷，1 个自由度。
- **复标量场** $\phi = (\phi_1 + i\phi_2)/\sqrt2$：2 个实自由度，等价于两个同质量实场。
- 复场的拉格朗日量 $\mathcal{L} = \partial_\mu\phi^\dagger\partial^\mu\phi - m^2\phi^\dagger\phi$ 在**整体 $U(1)$ 旋转**下不变。
- 诺特荷 $Q = \int d^3x\, j^0$ 数出**电荷 = 粒子数 − 反粒子数**，是「粒子-反粒子对」的记账本。
- 复场需要两套产生/湮灭算符 $a, b$，对应粒子与反粒子。

在下一节，我们从标量场转向**旋量场**，处理自旋 $\frac12$ 的物质粒子——那将是狄拉克方程的完整场论化，也是费米统计第一次登场的地方。


