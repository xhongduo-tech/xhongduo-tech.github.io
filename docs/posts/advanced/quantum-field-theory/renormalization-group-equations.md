---
title: 重整化群方程
date: 2026-08-07
---

# 重整化群方程

<div class="epigraph">
<p>参数随尺子的变化，本身就是理论的内容——它不是噪声，而是定律。</p>
<footer>—— 肯尼斯 · 威尔逊（Kenneth G. Wilson），标度不变性与重整化群</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §12.1–12.3 ｜ 2026-08-07</p>
</div>

## 为什么参数会「跑」

上一节我们留了一个尾巴：$\overline{\text{MS}}$ 方案里引入的质量标度 $\mu$ 是任意的，但圈图修正里出现了 $\log(\Delta/\mu^2)$——**物理量悄悄依赖 $\mu$**。
物理不能依赖我们选的尺子，于是「物理量不依赖 $\mu$」变成一条方程。
把它展开，就得到**重整化群方程（renormalization group equation）**——它描述耦合常数、质量、场强如何随 $\mu$ 变化，也就是**参数随能量标度「跑动」**。
这一节的主题是：为什么「尺子不同，参数不同」，以及 β 函数如何统治这种跑动。<span class="marginnote">直觉先行：测力常数时用低能尺子（$\mu$ 小）和用高能尺子（$\mu$ 大），「裸耦合」不同，但物理观测（散射截面）一样。重正化群就是「换尺子时参数怎么变」的精确答案——威尔逊把它推广成「标度变分」的现代语言。</span>

## 1 μ 依赖从哪来

物理量 $G^{(n)}$（$n$ 点关联函数）由裸量计算时不含 $\mu$——但换到重正化量后，参数 $\lambda(\mu), m(\mu)$ 与场 $Z_\phi^{1/2}(\mu)\phi$ 都带 $\mu$ 依赖。
**$G^{(n)}$ 作为裸量的函数不依赖 $\mu$，作为重正化量的函数则必须「参数跟着 $\mu$ 跑」来保持物理不变**。
把「$\mu$ 无关」写成方程：对任意物理量 $G^{(n)}$，

$$\frac{d}{d\log\mu}G^{(n)} = 0 \quad\Rightarrow\quad \mu\frac{\partial G^{(n)}}{\partial\mu} + \mu\frac{d\lambda}{d\mu}\frac{\partial G^{(n)}}{\partial\lambda} + \mu\frac{dm}{d\mu}\frac{\partial G^{(n)}}{\partial m} + n\gamma\, G^{(n)} = 0$$

这还不是完整形式——场的重正化让关联函数多一个「标度因子」$\gamma$，即**异常维数（anomalous dimension）**。<span class="marginnote">异常维数名字里的「异常」：经典维度分析说场按「长度$^{-1}$」标度；圈图修正让场的实际标度多了一个 $\gamma$——不是整数，是「异常」的。它衡量「量子涨落改变了场算符的标度性质」。</span>

## 2 Callan–Symanzik 方程

把上式写整齐，就是 **Callan–Symanzik 方程**（P&S §12.3）：

$$\left[\mu\frac{\partial}{\partial\mu} + \beta(\lambda)\frac{\partial}{\partial\lambda} + m\gamma_m\frac{\partial}{\partial m} + n\gamma\right]G^{(n)}(x_1,\dots,x_n) = 0$$

各系数是理论的「指纹」：

- **β 函数（beta function）**：$\beta(\lambda) \equiv \mu\frac{d\lambda}{d\mu}$，描述耦合常数随标度的变化率。
- **质量异常维数**：$\gamma_m \equiv -\frac{\mu}{m}\frac{dm}{d\mu}$，描述质量随标度的变化率。
- **场异常维数**：$\gamma$，描述场的标度因子。

对 $\phi^4$ 理论，单圈计算给出：

$$\beta(\lambda) = \frac{3\lambda^2}{16\pi^2} + \mathcal{O}(\lambda^3), \qquad \gamma = \frac{\lambda^2}{12(16\pi^2)} + \cdots, \qquad \gamma_m = \frac{\lambda}{16\pi^2} + \cdots$$

$\beta > 0$：耦合常数随能量升高而**增大**——$\phi^4$ 是「红外自由、紫外爆发」的理论。<span class="marginnote">β 的符号是理论的「性格」：$\beta>0$ 高能强耦合（$\phi^4$、QED），$\beta<0$ 高能弱耦合（QCD，见《渐进自由》）。它直接决定「高能下能不能用微扰论」。</span>

## 3 求解 RG 方程：跑动耦合

RG 方程是可解的——用**特征线方法**。设 $\bar\lambda(t)$ 是「跑动耦合」：

$$\frac{d\bar\lambda}{dt} = \beta(\bar\lambda), \qquad t = \log\frac{\mu}{\mu_0}, \qquad \bar\lambda(0) = \lambda(\mu_0)$$

它的解告诉我们：**在标度 $\mu$ 处「有效耦合」是 $\bar\lambda(\mu)$**，而不是裸的或低能的常数。对 $\phi^4$（单圈）：

$$\bar\lambda(\mu) = \frac{\lambda(\mu_0)}{1 - \frac{3\lambda(\mu_0)}{16\pi^2}\log\frac{\mu}{\mu_0}}$$

注意分母：$\mu$ 增大时，若 $\lambda(\mu_0)>0$，分母减小，$\bar\lambda$ 增大——在标度 $\mu = \mu_0 e^{16\pi^2/3\lambda}$ 处**发散**，这就是 **Landau 极点**（Landaupole）：纯 $\phi^4$/QED 在高能处耦合爆炸，理论自身崩溃。<span class="marginnote">Landau 极点不是一个「物理预言」，而是「这个理论在某个超高能标度失效」的信号。它告诉我们要么有新的物理（截断），要么理论是近似的。对 $\phi^4$ 这是「平凡性问题」；对 QED 这是「低能有效理论」的判决。</span>

RG 方程最实用的结论是**增强微扰论的适用性**：即便 $\lambda(\mu_0)$ 不够小，只要跑动耦合 $\bar\lambda(\mu)$ 在关心能区仍小，微扰论就能用——反之则不能。这就是为什么 QCD 在低能微扰失效、高能却微扰成功的理论根据。

## 4 公式解析：Callan–Symanzik 方程

**CS 方程是「物理量不依赖 μ」的精确编码。** 拆解四步：

$$
\left[\mu\frac{\partial}{\partial\mu} + \beta\frac{\partial}{\partial\lambda} + m\gamma_m\frac{\partial}{\partial m} + n\gamma\right]G^{(n)} = 0
$$

- **第一步，第一项 $\mu\partial_\mu$**：直接对 $\mu$ 的偏导——重正化参数显式依赖 $\mu$ 的部分。圈图里 $\log(\Delta/\mu^2)$ 的 $\mu$ 就住在这。
- **第二步，β 与 $\gamma_m$ 项**：$\mu\frac{d\lambda}{d\mu} = \beta$ 与 $\mu\frac{dm}{d\mu} = -m\gamma_m$ 进入链式法则——参数随 $\mu$ 变化时关联函数的响应。**它们把「耦合/质量随尺子变化」这一动力学写进了方程。**
- **第三步，$n\gamma$ 项**：$n$ 个外腿，每个贡献 $\gamma$ 的标度因子。来自 $G^{(n)} = Z_\phi^{-n/2}G^{(n)}_0$ 的场重正化。$n\gamma G^{(n)}$ 是「场标度变化的整体补偿」。
- **第四步，为什么是 0**：左边就是「物理量对 $\mu$ 的**全**导数」。物理量不依赖 $\mu$——这条守恒律把方程定格为 0。**求解它 = 在 (μ, λ, m) 空间中沿「特征流」移动，同时保证 G 不变。**

## 5 辨析｜易错点

- **把「耦合常数」当常数**：在 QFT 里耦合常数是**标度依赖的**——$\alpha(m_e) \neq \alpha(m_Z)$。说「精细结构常数 = 1/137」只在低能近似下成立。跑动是理论结构，不是实验误差。<span class="marginnote"><strong>β 的符号别记反</strong>：$\beta(\lambda) = \frac{3\lambda^2}{16\pi^2}$ 的「+」号是 $\phi^4$ 的招牌。QED 的 β 也是正（$\beta \propto \alpha^2 > 0$），QCD 的 β 是负（$\beta \propto -g^3$）。正 β = 高能变强，负 β = 高能变弱。</span>
**把「跑动耦合」当「真实的耦合」**：$\bar\lambda(\mu)$ 是「在标度 $\mu$ 处做实验时有效看到的耦合」，不是「耦合的真实值」——耦合没有唯一真实值。可测的是**物理量**，它们对 $\mu$ 不变，只是「用哪个参数展开微扰」会变。
**Landau 极点不是预言「宇宙爆炸」**：它是理论自洽性失效的标度，不是物理事件。标准做法是承认理论在该标度前有新的物理进入（新粒子、新对称性）。

## 6 延伸：RG 作为「标度物理」的通用语言

重整化群远远超出微扰场论——它是现代物理理解「不同尺度的理论」的通用框架：

**统计力学与相变**：威尔逊的重整化群把「临界指数」解释成 RG 不动点的标度性质。不同材料（铁磁、液体-气体、合金）在临界点表现出相同的临界指数，因为它们的 RG 流收敛到同一个不动点——这是「普适性」的机制。
**有效场论**：所有理论都是某个能标以下的有效理论。RG 告诉你「低能下哪些算子存活」：可重正化算子在低能主导，不可重正化的被 $1/\Lambda^2$ 压低。这就是为什么标准模型（维数 4 算子）是低能世界的语法。
**凝聚态**：BCS 超导、费米液体理论都靠 RG 在低能重标度下识别「什么是重要的相互作用」。

「从极限到大模型」的主线在这里再次现身：**理解一个复杂系统，常常不是直接解它，而是看它在不同标度下如何「重标度自身」**。RG 就是这套「标度自相似」的精确语言——它是连接量子场论与统计力学的桥梁，也是理解「为什么不同系统看起来相似」的钥匙。

### 自测清单

[ ] 能写出 Callan–Symanzik 方程并解释每项来源。
[ ] 能定义 β 函数、$\gamma_m$、$\gamma$ 并说出符号的含义。
[ ] 能解单圈跑动方程并读出 Landau 极点。
[ ] 能解释「跑动耦合决定微扰论适用范围」。

<span class="marginnote">把 RG 当「缩放镜」：<strong>每次放大/缩小镜头（改变 $\mu$），参数就重新标定一次</strong>——但照片里的物理（物理量）不变。这是整章的哲学。</span>

### 延伸阅读指引

- 深化推导：P&S §12.1 的 Callan–Symanzik 方程、§12.3 的求解；想理解 RG 的统计力学根源可读 Wilson 1971 年综述。
- 实践：解 $\phi^4$ 单圈 RG 方程，画出 $\bar\lambda(\mu)$ 曲线并标出 Landau 极点。
- 联系主线：RG 是「标度变化下的自相似」语言——与《小波分析》里「多分辨率」、以及《深度学习》里「不同层学到不同尺度的特征」是同一思想。

## 7 小结

- 物理量不依赖 $\mu$ → **Callan–Symanzik 方程**：$[\mu\partial_\mu + \beta\partial_\lambda + m\gamma_m\partial_m + n\gamma]G^{(n)} = 0$。
- **β 函数** $\beta = \mu\frac{d\lambda}{d\mu}$ 决定耦合跑动方向；$\gamma_m$、$\gamma$ 决定质量与场的标度行为。
- 单圈 $\phi^4$：$\beta = \frac{3\lambda^2}{16\pi^2} > 0$，高能耦合增大。
- 跑动耦合 $\bar\lambda(\mu)$ 的 Landau 极点 = 理论失效标度，非物理预言。
- RG 增强微扰论：微扰展开的判据是**跑动耦合**而非裸耦合。

在下一节，我们把重正化的全套机器用在第一个真实理论——**QED 重正化**：电子自能、光子自能、顶点修正，以及它们如何被 Ward 恒等式统一。


