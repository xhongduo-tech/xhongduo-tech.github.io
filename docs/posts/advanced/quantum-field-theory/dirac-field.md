---
title: 狄拉克场
date: 2026-08-07
---

# 狄拉克场

<div class="epigraph">
<p>一个方程，竟同时装下了电子的自旋、负能量海的谜题，与一个未知的新粒子——正电子。</p>
<footer>—— 亚伯拉罕 · 派斯（Abraham Pais），科学史家</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §3.1–3.5 ｜ 2026-08-07</p>
</div>

## 为什么需要狄拉克场

上一节的标量场描述了自旋 0 的玻色子。可构成物质的电子、夸克都是**自旋 $\frac12$ 的费米子**——它们不服从克莱因-戈登方程，而服从**狄拉克方程**。
更重要的是，费米子服从**泡利不相容原理**：同一状态至多一个粒子。正则量子化里，这个原理不是额外假设，而是**反对易关系**自动吐出来的结论。
<span class="marginnote">标量场的产生算符满足 $[a^\dagger,a^\dagger]=0$ 没关系；费米子若也用对易关系，同一状态能塞进无限多粒子，会违背泡利原理。把对易换成分层反对易 $\{,\}$，泡利原理立刻从代数里长出来。</span>本节要完成三件事：重述狄拉克方程，理解旋量的洛伦兹性质，并用**反对易子**把它量子化成费米子场。

## 1 狄拉克方程与 $\gamma$ 矩阵

**狄拉克方程（Dirac equation）**：对 4 分量旋量 $\psi(t,\boldsymbol{x})$ 的运动方程

$$(i\gamma^\mu \partial_\mu - m)\psi = 0$$

其中 $\gamma^\mu$（$\mu=0,1,2,3$）是 $4\times4$ 矩阵，满足**克利福德代数**：

$$\{\gamma^\mu, \gamma^\nu\} = \gamma^\mu\gamma^\nu + \gamma^\nu\gamma^\mu = 2g^{\mu\nu}\mathbb{1}_4$$

$g^{\mu\nu} = \text{diag}(1,-1,-1,-1)$ 是闵可夫斯基度规。<span class="marginnote">为什么必须是 $4\times4$ 矩阵？因为我们要找一个「一阶微分方程」来模拟 $E^2 = p^2 + m^2$，而 $\sqrt{E^2}$ 无法直接写成普通数的线性组合——只有矩阵代数才能「开方」。$4\times4$ 是最小维度，恰好容纳自旋 ±1/2 的两个取向 × 粒子/反粒子两个分支。</span>共轭旋量 $\bar\psi \equiv \psi^\dagger \gamma^0$，它服从 $\bar\psi(i\gamma^\mu\overset{\leftarrow}{\partial}_\mu + m) = 0$。狄拉克场的拉格朗日量：

$$\mathcal{L} = \bar\psi(i\gamma^\mu\partial_\mu - m)\psi$$

对 $\bar\psi$ 变分即得狄拉克方程。注意：$\mathcal{L}$ 对 $\psi$ 与 $\bar\psi$ 是**双线性**的，而标量场是 $\phi^2$——自旋 $\frac12$ 场必须是「一次方」结构，这将在下节直接导致反对易。

## 2 旋量：洛伦兹群的双值表示

旋量 $\psi$ 按洛伦兹变换 $\Lambda$ 的方式与标量完全不同。标量变换是 $\phi'(x) = \phi(\Lambda^{-1}x)$；旋量则多一个**旋量表示矩阵**：

$$\psi'(x') = S(\Lambda)\,\psi(x), \qquad S(\Lambda) = \exp\left(-\frac{i}{4}\omega_{\mu\nu}\sigma^{\mu\nu}\right), \quad \sigma^{\mu\nu} = \frac{i}{2}[\gamma^\mu,\gamma^\nu]$$

$S(\Lambda)$ 有两个关键性质：**不是酉矩阵**（$S^\dagger \neq S^{-1}$，但 $\gamma^0 S^\dagger \gamma^0 = S^{-1}$），且 $S(\Lambda)$ 对 $2\pi$ 旋转给出 $-1$——旋量转一圈回到自身时差一个负号。<span class="marginnote">「转 360° 变负号」在数学上叫洛伦兹群的<b>双值表示</b>。这在经典世界里不可观测（任何物理量都含 $\bar\psi\psi$，负号成对抵消），但在量子力学干涉实验里是真实的——中子的 $2\pi$ 旋转干涉实验证实了这一点。</span>

旋量的**平面波解**有两族：正能解 $u^s(p)$ 与负能解 $v^s(p)$，$s = \pm\frac12$ 标记自旋取向。它们满足：

$$(\not p - m)u^s(p) = 0, \qquad (\not p + m)v^s(p) = 0$$

其中 $\not p \equiv \gamma^\mu p_\mu$ 是费曼斜杠记号。
<span class="marginnote">费曼斜杠 $\not p = \gamma^\mu p_\mu$ 会伴随全书：它在费曼规则里是传播子的分子、是顶点里的动量「内接线」。</span>归一化取 $u^{s\dagger}u^r = 2E_{\boldsymbol{p}}\delta^{sr}$、$v^{s\dagger}v^r = 2E_{\boldsymbol{p}}\delta^{sr}$，让后续流与截断公式保持洛伦兹协变。

## 3 正则量子化：反对易关系与费米子

现在把 $\psi$ 提升为算符。与标量场的唯一区别是：**对易子换成反对易子**。**等时反对易关系**为：

$$\{\psi_a(\boldsymbol{x}), \psi_b^\dagger(\boldsymbol{y})\} = \delta^{(3)}(\boldsymbol{x}-\boldsymbol{y})\,\delta_{ab}, \qquad \{\psi,\psi\} = \{\psi^\dagger,\psi^\dagger\} = 0$$

$\,a,b$ 是旋量分量指标。场算符展开为：

$$\psi(x) = \int \frac{d^3p}{(2\pi)^3}\frac{1}{\sqrt{2E_{\boldsymbol{p}}}}\sum_{s=\pm}\left( a_{\boldsymbol{p}}^s u^s(p)e^{-ip\cdot x} + b_{\boldsymbol{p}}^{s\dagger} v^s(p)e^{ip\cdot x} \right)$$

代入反对易关系，得到产生/湮灭算符的代数：

$$\{a_{\boldsymbol{p}}^r, a_{\boldsymbol{q}}^{s\dagger}\} = (2\pi)^3\delta^{rs}\delta^{(3)}(\boldsymbol{p}-\boldsymbol{q}), \qquad \{b_{\boldsymbol{p}}^r, b_{\boldsymbol{q}}^{s\dagger}\} = \text{同形}$$

对费米子，$a^{s\dagger}_{\boldsymbol{p}}$ 产生一个自旋 $s$ 动量为 $\boldsymbol{p}$ 的**电子**，$b^{s\dagger}_{\boldsymbol{p}}$ 产生一个**正电子**。
关键推论：$\{a^\dagger,a^\dagger\}=0$ 意味着 $a^{s\dagger}_{\boldsymbol{p}}a^{s\dagger}_{\boldsymbol{p}}|0\rangle = 0$——**同一状态塞不进两个费米子，泡利不相容原理自动成立**。<span class="marginnote">费米子的 $a^\dagger$ 自己反对易，所以 $(a^\dagger)^2 = 0$。这就是「一个萝卜一个坑」的代数根源。对称性（反对易）与统计（费米-狄拉克）在这里是同一件事——自旋-统计定理的雏形。</span>

## 4 公式解析：狄拉克场的哈密顿量与负能海

**狄拉克场量子化的真正考验是哈密顿量不再「从上到下」，而是靠反对易把它扶正。** 先写出（含常数修正前的）哈密顿量：

$$
H = \int \frac{d^3p}{(2\pi)^3}\sum_s E_{\boldsymbol{p}}\left( a_{\boldsymbol{p}}^{s\dagger} a_{\boldsymbol{p}}^s - b_{\boldsymbol{p}}^{s\dagger} b_{\boldsymbol{p}}^s \right)
$$

三步拆解：

- **第一步，负号从哪来**：$H = \int d^3x\,\bar\psi(i\gamma^i\partial_i + m)\psi$ 代入展开式时，正能部分（$a^\dagger a$）带 $+E$，负能部分（$b^\dagger b$）因 $v^s(p)$ 解的能量为负而带 $-E$。**如果 $\{,\}$ 用成 $[,]$，这个 $-E$ 项将让能量无下限——系统会一路掉进负能深渊。**
- **第二步，反对易如何救场**：对费米子，$b_{\boldsymbol{p}}^{s\dagger}b_{\boldsymbol{p}}^s$ 是数算符（取值 0 或 1）。在 $H$ 里补上「换序常数」$b b^\dagger \to -b^\dagger b + 1$（利用 $\{b,b^\dagger\}=1$），$-E\,b^\dagger b$ 变成 $+E\,b^\dagger b$ 减去一个常数。**负能项被反对易关系「翻转」成正能量**——正电子因此有正能量，物理系统的能量从下方有界。
- **第三步，泡利原理与正能量是同一个要求的两次体现**：狄拉克 1928 年用「负能海填满电子 + 空穴即正电子」来修复负能量问题，费米子要求每个态至多一个。场论的正则量子化把这个曲折的「海」故事替换成一条干净的代数事实：**费米子必须反对易，反对易自动给出正能量与泡利原理**。

## 5 辨析｜易错点

- **把 $\{,\}$ 与 $[,]$ 混用**：玻色子（标量、光子、引力子）用对易子 $[a,a^\dagger]=1$，费米子（电子、夸克、中微子）用反对易子 $\{a,a^\dagger\}=1$。写费曼规则时两者会各自给出**不同的传播子符号约定**——这是第 5 篇《费曼规则》最容易错的地方。<span class="marginnote"><strong>自旋-统计定理</strong>：整数自旋 → 玻色（对易）；半整数自旋 → 费米（反对易）。这是相对论量子场论的定理（泡利证明），不是经验拟合。违反它会直接制造负概率或负能量。</span>
- **$\gamma^0 = \gamma^{0\dagger}$ 但 $\gamma^i = -\gamma^{i\dagger}$**：只有 $\gamma^0$ 是厄米的。所以 $\bar\psi = \psi^\dagger\gamma^0$ 里的 $\gamma^0$ 不是装饰——它负责把「旋量内积」修正成洛伦兹标量。写 $\psi^\dagger\psi$ 想当标量用是错的。
- **$u^s(p)$ 是「粒子」解还是「正能」解**：两者对——对电子 $u$ 是正能解（对应 $a$，湮灭），$v$ 是负能解（对应 $b^\dagger$，产生反粒子）。但「正能」是相对的：正电子的 $u$ 解在电荷共轭下对应 $v$ 解。

## 6 延伸：螺旋度、手性与中微子

狄拉克场的两个「度」极易混淆，值得单独钉死：

- **螺旋度（helicity）** $h = \hat{\boldsymbol{p}}\cdot\boldsymbol{S}$：自旋沿运动方向的投影。无质量粒子它是洛伦兹不变量；有质量粒子在高速变换下可翻转。
- **手性（chirality）**：$\psi_{L,R} = \frac{1\mp\gamma^5}{2}\psi$ 的投影。它是洛伦兹不变量，永远不变。
- 两者只在**无质量极限**下重合：$m \to 0$ 时，左手手性 = 负螺旋度、右手 = 正螺旋度。

中微子（近乎无质量）的左手性直接导致弱作用的宇称破坏：只有左手中微子参与弱过程，右手中微子（若存在）是惰性单态。这套「手性 = 弱作用的语法」是第四章电弱理论的出发点——$SU(2)_L$ 的下标 L 就是这个手性。

另外提醒一个常被问到的点：狄拉克方程的解 $u^s(p), v^s(p)$ 并不对应「粒子/反粒子」的先验标签——是量子化（$a$ 湮灭粒子、$b^\dagger$ 产生反粒子）赋予了它们物理意义。**方程本身只提供数学解，场论赋予它们身份。**

### 自测清单

- [ ] 能写出克利福德代数并说明 $\gamma$ 为何是 $4\times4$。
- [ ] 能写出狄拉克场展开并说明 $u, v$ 各对应什么。
- [ ] 能解释反对易关系如何实现泡利不相容。
- [ ] 能说明负能解如何被重新解释为正电子。
- [ ] 能区分手性与螺旋度，并说出两者何时重合。

<span class="marginnote">中微子的例子是「手性真正有物理」的活教材：<strong>实验发现弱作用只认左手</strong>——这不是理论家的自由选择，而是对自然律的忠实记录。</span>

## 7 小结

- 狄拉克方程 $(i\gamma^\mu\partial_\mu - m)\psi = 0$，$\gamma$ 矩阵满足克利福德代数 $\{\gamma^\mu,\gamma^\nu\} = 2g^{\mu\nu}$。
- 旋量是洛伦兹群的双值表示：$S(\Lambda)$ 非酉、$2\pi$ 旋转变号；平面波解 $u^s(p), v^s(p)$。
- **费米子用反对易关系量子化**：$\{\psi_a,\psi_b^\dagger\} = \delta^{(3)}\delta_{ab}$，$(a^\dagger)^2 = 0$ 自动实现泡利不相容。
- 反对易关系把负能项「翻转」成正能量，彻底解决狄拉克的负能海问题。
- 自旋-统计：半整数自旋必为费米子，整数自旋必为玻色子。

在下一节，我们处理最后一种自由场——**电磁场**。它自带规范对称性、有 4 个分量却只有 2 个物理自由度，量子化它需要新的技巧。



