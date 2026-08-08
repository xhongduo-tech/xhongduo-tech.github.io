---
title: S矩阵与散射截面
date: 2026-08-07
---

# S矩阵与散射截面

<div class="epigraph">
<p>量子场论与实验的对话，永远以同一个问题开场：这个过程的截面是多少？</p>
<footer>—— 自加速器物理传统（为本文所作）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §4.5–4.6 ｜ 2026-08-07</p>
</div>

## 为什么振幅不等于实验

费曼规则给出的是**振幅** $i\mathcal{M}$——一个复数，藏在波函数里，实验测不到。
对撞机报给世界的是一串数字：**截面**（cross section，单位面积）、**衰变率**（decay rate，单位时间）。
把 $i\mathcal{M}$ 翻译成截面，需要三个环节：S 矩阵编码「初态→末态」的概率幅；**相空间**数出末态可用的「跑道」；**金规则**把振幅平方、乘相空间、除流因子，得到截面。
本节就是这条翻译链。<span class="marginnote">一句话脉络：<strong>振幅 $i\mathcal{M}$（理论家的产出）$\to$ $|\mathcal{M}|^2$（概率）$\to$ × 相空间（末态自由度）$\to$ ÷ 入射流（归一化）$=$ 截面（实验家的读数）</strong>。</span>

## 1 S 矩阵与减缩公式

**S 矩阵（S-matrix / scattering matrix）** 把「初态 $|\boldsymbol{p}_A\boldsymbol{p}_B\rangle$ 演化为末态 $|\boldsymbol{p}_1\cdots\boldsymbol{p}_n\rangle$」的概率幅打包：

$$\langle \boldsymbol{p}_1\cdots\boldsymbol{p}_n | S | \boldsymbol{p}_A\boldsymbol{p}_B \rangle = \underbrace{\langle \cdots\rangle_{\text{无散射}}}_{\text{单位阵部分 } 1} + \underbrace{(2\pi)^4\delta^{(4)}\!\left(\sum p_{\text{in}}-\sum p_{\text{out}}\right) i\mathcal{M}}_{\text{真正的散射}}$$

$S = 1 + iT$ 分解：「$1$」是没散射的部分，「$iT$」是相互作用的部分。$i\mathcal{M}$ 正是费曼规则算出的那个「约化矩阵元」。<span class="marginnote">为什么外腿的波函数因子（$u^s,\bar u^s$ 等）出现在费曼规则里？因为 S 矩阵元要把场算符与单粒子渐近态「对接」——这正是 LSZ 减缩公式的内容：<strong>把外腿「钉」到渐近单粒子态上，外线变波函数</strong>。</span>

散射振幅 $i\mathcal{M}$ 是**洛伦兹不变的**（好的振幅设计）。而截面不是洛伦兹标量——它依赖观察者的流与参考系。所以翻译时要小心选取参考系。

## 2 相空间：末态的自由度

一个 $n$ 体末态的体积元（**洛伦兹不变相空间**）：

$$d\Pi = (2\pi)^4 \delta^{(4)}\!\left(p_A + p_B - \sum_{f} p_f\right) \prod_{f=1}^{n} \frac{d^3p_f}{(2\pi)^3}\frac{1}{2E_{\boldsymbol{p}_f}}$$

- $\delta^{(4)}$：总四动量守恒（由 S 矩阵分解里的 $\delta$ 而来，此处显式写出）。
- $\frac{d^3p_f}{(2\pi)^3}\frac{1}{2E_{\boldsymbol{p}_f}}$：每个末态粒子的洛伦兹不变测度。

这个测度的关键性质：$\int \frac{d^3p}{(2\pi)^3}\frac{1}{2E_{\boldsymbol{p}}}$ 是洛伦兹不变的（因为 $d^3p/E_{\boldsymbol{p}}$ 不变）。<span class="marginnote">洛伦兹不变测度来自「质壳条件」：$d^4p\,\delta(p^2-m^2)\theta(p^0) = \frac{d^3p}{2E_{\boldsymbol{p}}}$。它把四动量积分「压」到质壳上——末态粒子必须满足 $E^2 = \boldsymbol{p}^2 + m^2$。</span>

## 3 散射截面公式

对一个 $2 \to n$ 过程，**微分截面**：

$$d\sigma = \frac{1}{2E_A\, 2E_B\, |v_A - v_B|}\, |\mathcal{M}|^2\, d\Pi$$

分子：$|\mathcal{M}|^2$（振幅平方 = 概率）× $d\Pi$（末态体积）。分母：$2E_A\,2E_B\,|v_A-v_B|$ 是**入射流因子**（两束粒子的相对速度 × 能量归一化），保证截面是「每个入射粒子单位面积上的有效靶大小」。**衰变率**结构相同，只是没有入射流，分母换成 $2E_A$：

$$d\Gamma = \frac{1}{2E_A}\,|\mathcal{M}|^2\, d\Pi$$

对 $2\to 2$ 散射在质心系里，公式化成最常用的形式：

$$\frac{d\sigma}{d\Omega}\Big|_{\text{CM}} = \frac{1}{64\pi^2 s}\frac{|\boldsymbol{p}_f|}{|\boldsymbol{p}_i|}\,|\mathcal{M}|^2$$

其中 $s = (p_A+p_B)^2$ 是曼德尔斯塔姆变量，$|\boldsymbol{p}_i|, |\boldsymbol{p}_f|$ 是初末态粒子的质心动量大小。<span class="marginnote">$64\pi^2 s$ 的来源：相空间两个粒子的积分 $+$ 立体角 $d\Omega$ 的归化。它不含任何动力学——是「两体相空间」的纯几何量，动力学全部在 $|\mathcal{M}|^2$ 里。</span>

## 4 公式解析：$2\to2$ 微分截面

**把「振幅 → 截面」的完整链条压缩进一条公式。** 拆解三步：

$$
\frac{d\sigma}{d\Omega} = \frac{1}{64\pi^2 s}\frac{|\boldsymbol{p}_f|}{|\boldsymbol{p}_i|}|\mathcal{M}|^2
$$

- **第一步，$1/(64\pi^2 s)$ 从哪来**：两体相空间积分 $\int d\Pi_2$ 在质心系里可解析算出：$\int d\Pi_2 = \frac{1}{16\pi}\frac{|\boldsymbol{p}_f|}{\sqrt{s}}\,d\Omega$（立体角部分）。再除以入射流 $2E_A\,2E_B|v_A-v_B| = 2\sqrt{s}|\boldsymbol{p}_i|$（质心系），合并得到 $1/(64\pi^2 s)\cdot |\boldsymbol{p}_f|/|\boldsymbol{p}_i|$。**全部是几何，没有动力学。**
- **第二步，$|\boldsymbol{p}_f|/|\boldsymbol{p}_i|$ 的物理**：末态动量相对初态动量的比。弹性散射时两者相等（$=1$）；非弹性时它反映「质量差」如何压缩或放宽相空间。$|\boldsymbol{p}_f| \to 0$（阈值附近）时截面被相空间压制。
- **第三步，$|\mathcal{M}|^2$ 是唯一的「物理」**：所有耦合、传播子、自旋求和都在这里。对 QED $e^-\mu^-$ 散射，$|\mathcal{M}|^2$ 在极端相对论极限下给出著名的 $\frac{d\sigma}{d\Omega}\propto \frac{1}{s}\frac{1+\cos^2\theta}{2}$ 型角分布——Rutherford/Mott 公式的场论版。**测角分布 = 测振幅的平方的角依赖 = 测理论**。

## 5 辨析｜易错点

- **$\delta^{(4)}$ 不能直接「消掉」**：S 矩阵元里有一个总守恒 $\delta$，但截面公式里它已经被「挪进」相空间、并通过对末态动量积分处理掉了。**不要在截面公式里再手动加 $\delta$**——那是双算守恒。
- **入射流因子的参考系依赖**：$2E_A2E_B|v_A-v_B|$ 是质心系的形式；在实验室系要换公式。截面是洛伦兹标量，但**中间步骤**（流、相空间拆解）随参考系变。计算时自始至终钉在一个系里。<span class="marginnote"><strong>快查：$i\mathcal{M}$ 的共轭</strong>：$|\mathcal{M}|^2 = \mathcal{M}^*\mathcal{M}$。对费米子，$\mathcal{M}$ 是旋量链的缩并，共轭要翻转 $\gamma$ 矩阵顺序、取转置与 $\gamma^0$——P&S §5.1 的「Casimir 技巧」就是干这个的。</span>
- **把「截面」与「衰变率」混淆**：衰变 $1\to n$ 无入射流，分母 $2E_A$；散射 $2\to n$ 有入射流，分母 $2E_A2E_B|v_A-v_B|$。公式结构相同但物理与量纲不同。

## 6 延伸：截面与实验的接口

「振幅 → 截面 → 实验」这条链上，有几个工程细节决定理论能否被检验：

- **自旋与极化的平均/求和**：初态自旋不可控，所以要平均；末态自旋可测（如 $Z$ 的极化分析），可以分别求和。这一惯例是「实验能测什么」直接决定的。
- **相空间的数值积分**：高末态粒子数（$n \ge 5$）时相空间积分无法解析，用 Monte Carlo 数值积分。LHC 的每个截面数字背后都是数百万次随机采样。
- **事件生成器**：把理论截面与「喷注簇射、强子化、探测器响应」接起来的软件层。理论家算 $\sigma$，实验家看事件——事件生成器是中间的翻译器。

一个常被忽视的要点：**截面公式里「几何因子 $1/(64\pi^2 s)$」不含任何物理**，但它决定了实验对「大 $\theta$ 角还是小 $\theta$ 角灵敏」。设计实验探测器时，物理学家先看理论的角分布决定在哪放探测器——这又是「理论 → 实验」的反馈。

回到本主题的意义：这一章把「场论的产出」接到「对撞机的事实」。之后读任何唯象学论文，你都要能回答：他们算的 $|\mathcal{M}|^2$ 怎么变成报给世界的截面数字的。

### 自测清单

- [ ] 能写出 S 矩阵的 $1 + iT$ 分解与约化振幅的定义。
- [ ] 能写出洛伦兹不变相空间测度。
- [ ] 能写出 $2\to n$ 截面公式与入射流因子。
- [ ] 能默写质心系 $2\to2$ 微分截面公式。

<span class="marginnote">理解「相空间是跑道、振幅是车、截面是成绩」这个三件套，你就掌握了从理论到实验的完整翻译链。</span>

### 延伸阅读指引

- 深化推导：P&S §4.5 的 LSZ 减缩公式、§4.6 的截面公式推导；想理解「外腿钉到渐近态」可精读 §7.2 的散射矩阵元。
- 实践：用 $e^+e^-\to\mu^+\mu^-$ 的完整截面对照 PDG 数据；试算 $W$ 衰变宽度体会「衰变率 = 振幅² × 相空间」。
- 联系主线：截面是「理论产出与实验世界的接口」——正如《信息论》里的信道容量是「编码方案与物理信道的接口」。理解接口比理解两端更重要。

### 本节记忆锚点

- S 矩阵：$S = 1 + iT$，$\langle f|T|i\rangle = (2\pi)^4\delta^{(4)}i\mathcal{M}$。
- 相空间：$d\Pi = (2\pi)^4\delta^{(4)}\prod\frac{d^3p}{(2\pi)^3 2E}$，洛伦兹不变。
- 截面：$d\sigma = \frac{|\mathcal{M}|^2}{2E_A2E_B|v_A-v_B|}d\Pi$；衰变率无流因子。
- $2\to2$ 质心系：$\frac{d\sigma}{d\Omega} = \frac{1}{64\pi^2 s}\frac{|\boldsymbol{p}_f|}{|\boldsymbol{p}_i|}|\mathcal{M}|^2$。
- 交叉引用：与《粒子物理》的散射实验、第四级《天体物理》的截面对照。

## 7 小结

- S 矩阵 $S = 1 + iT$；$i\mathcal{M}$ 是去掉守恒 $\delta$ 后的约化振幅，洛伦兹不变。
- 相空间 $d\Pi$ 含质壳测度 $\frac{d^3p}{(2\pi)^3 2E_{\boldsymbol{p}}}$ 与总守恒 $\delta$。
- 截面 $d\sigma = \frac{|\mathcal{M}|^2}{2E_A2E_B|v_A-v_B|}d\Pi$；衰变率 $d\Gamma = \frac{|\mathcal{M}|^2}{2E_A}d\Pi$。
- 质心系 $2\to2$：$\frac{d\sigma}{d\Omega} = \frac{1}{64\pi^2 s}\frac{|\boldsymbol{p}_f|}{|\boldsymbol{p}_i|}|\mathcal{M}|^2$。
- 几何全在 $1/(64\pi^2s)$ 里，物理全在 $|\mathcal{M}|^2$ 里。

在下一节，我们用这些规则与公式**实际动手算**——从最简单的 $e^+e^-\to\mu^+\mu^-$ 到含自旋求和的完整费曼图计算。


