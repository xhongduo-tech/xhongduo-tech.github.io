---
title: 全变差距离
date: 2026-08-07
---

# 全变差距离

<div class="epigraph">
<p>概率是思想的量尺。</p>
<footer>—— 皮埃尔-西蒙 · 拉普拉斯（Pierre-Simon Laplace）</footer>
</div>

<div class="article-byline">
<p>第二级 · 马尔可夫链与混合时间 ｜ Levin, Peres &amp; Wilmer《Markov Chains and Mixing Times》Ch. 4 ｜ 2026-08-07</p>
</div>

## 为什么从全变差距离开始

前面三节回答了「链的结构是什么、极限分布是什么」，但还缺一个度量：**两个概率分布到底差多少？** 平稳分布 $\pi$ 就在那里，初值分布 $\mu$ 经过 $n$ 步变成 $\mu P^n$，它离 $\pi$ 还有多远？这个问题不解决，「混合时间」就无从谈起。**全变差距离（total variation distance）**给出了最自然、也最常用的答案：它是两个分布在所有事件上概率差的最大值，是统计里「最坏情况判别误差」的精确表达。<span class="marginnote">全变差距离如此重要，是因为它与「观察者能否区分两个分布」绑定：距离等于 $\varepsilon$，意味着任何基于观测的判别器面对两个分布都至少以 $1-\varepsilon$ 的置信度犯错或无法区分。它是统计检验与信息论都绕不开的度量。</span>

## 1 定义

设 $\mu, \nu$ 是有限空间 $\Omega$ 上的两个概率分布，它们的**全变差距离**定义为

$$
\lVert \mu - \nu \rVert_{\mathrm{TV}} = \max_{A \subseteq \Omega} \left| \mu(A) - \nu(A) \right|
$$

即「在任意事件 $A$ 上，两个分布给出的概率之差的最大值」。若存在一个事件让两个分布的概率差到 $0.5$，说明它们很容易被区分；若对任何事件都差不到 $0.01$，说明它们几乎不可区分。<span class="marginnote">注意这是个<strong>距离</strong>：对称、非负、满足三角不等式，且 $\lVert\mu-\nu\rVert_{\mathrm{TV}}=0$ 当且仅当 $\mu=\nu$。它让「分布的收敛」有了度量空间的语法。</span>

全变差距离有两个常用的等价形式，分别服务于计算与直觉：

$$
\lVert \mu - \nu \rVert_{\mathrm{TV}} = \frac{1}{2} \sum_{x \in \Omega} |\mu(x) - \nu(x)| = \frac{1}{2} \lVert \mu - \nu \rVert_1
$$

**这就是著名的「一半 $L^1$ 距离」**：两个分布全变差距离等于逐点差的绝对值之和的一半。<span class="marginnote">为什么是 $1/2$？因为 $\sum_x |\mu(x)-\nu(x)|$ 里正项之和恰好等于负项绝对值之和（都等于 $\sum_x (\mu(x)-\nu(x))_+$，即所有「超额质量」），而正项之和本身是一个事件上的差，恰好就是全变差。所以 $L^1/2$ 自动不超过定义式，且取到等号。</span>

## 2 公式解析：三种等价形式怎么互相转化

把三个表达式放在一起逐项拆解：

$$
\lVert \mu - \nu \rVert_{\mathrm{TV}} = \max_{A \subseteq \Omega} |\mu(A) - \nu(A)| = \frac{1}{2}\sum_{x} |\mu(x) - \nu(x)| = 1 - \sum_{x} \min(\mu(x), \nu(x))
$$

- **第一步，从最大化到求和**：记 $B = \{x : \mu(x) \geq \nu(x)\}$ 为「$\mu$ 超重的状态集」。对任意 $A$，$|\mu(A)-\nu(A)| \leq \mu(B)-\nu(B)$，所以最大值在 $B$ 上取得，且 $\mu(B)-\nu(B) = \tfrac12 \sum_x |\mu(x)-\nu(x)|$。
- **第二步，从求和到最小值**：注意恒等式 $\sum_x \min(\mu,\nu) = 1 - \tfrac12\sum_x |\mu-\nu|$——两个分布重叠的质量是 $1$ 减去差异的一半。
- **第三步，读出直观**：第三式是「**两个分布能够重叠的质量的上界**」，即若把两个分布的「质量」同时摆上状态空间，最多能重合多少。$1 - \sum \min(\mu,\nu)$ 越小，两个分布越相似。
- **第四步，联系耦合**：$\sum_x \min(\mu(x),\nu(x))$ 恰是两个分布作为「饼图」重叠的面积，而耦合理论会证明「最大重叠」恰等于最优耦合的成功概率——这是下一节耦合方法的伏笔。

## 3 马尔可夫链收敛定理

全变差距离给了收敛一个精确语言。设 $P$ 不可约非周期，$\pi$ 为唯一平稳分布，则

$$
\lim_{n \to \infty} \lVert \mu_0 P^n - \pi \rVert_{\mathrm{TV}} = 0, \qquad \forall \mu_0
$$

并且对可逆链，全变差距离**随 $n$ 单调不增**。<span class="marginnote">单调性不是显然的：分布越来越接近平稳分布，且不会「冲过头」。对可逆链成立，对一般链则可能非单调（但上极限仍收敛）。这在后面的「截断现象（cutoff）」讨论里很重要。</span>这条收敛定理把上一节的「存在性」升级为「确实会到」，接下来就只剩一个问题：**多快**？

为此引入**混合时间（mixing time）**：对 $0 \lt  \varepsilon \lt  1$，

$$
t_{\mathrm{mix}}(\varepsilon) = \min\left\{ n : \max_{\mu_0} \lVert \mu_0 P^n - \pi \rVert_{\mathrm{TV}} \leq \varepsilon \right\}
$$

即「从最坏的初始分布出发，把与平稳分布的全变差距离压到 $\varepsilon$ 以下所需的最少步数」。最常用的是 $t_{\mathrm{mix}} = t_{\mathrm{mix}}(1/4)$。<span class="marginnote">为什么取 $1/4$ 而不是更小？经验约定：从 $1/4$ 到 $\varepsilon$ 只需再走 $O(\log(1/\varepsilon))$ 步（亚几何收敛下），所以 $1/4$ 是能反映量级又不必纠缠小常数的最简选择。所有关于混合时间的「上界/下界」都围绕这个量展开。</span>

## 4 例子：两状态链的混合时间

回到最简例子 $\Omega=\{0,1\}$，转移矩阵 $P = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}$。

用全变差距离计算混合时间，分四步：

- **第一步，求平稳分布**：解 $\pi P = \pi$ 得 $\pi(0) = \beta/(\alpha+\beta)$，$\pi(1) = \alpha/(\alpha+\beta)$。
- **第二步，写 $P^n$ 的谱分解**：$P$ 有特征值 $\lambda_0 = 1$ 与 $\lambda_1 = 1 - \alpha - \beta$。故 $P^n = \pi^T \mathbf{1} + \lambda_1^n R$，其中 $R$ 是固定的秩 1 修正项。
- **第三步，代回全变差距离**：从初值 $\mu_0 = (1, 0)$ 出发，$\mu_0 P^n(1) = \pi(1)(1 - \lambda_1^n)$，于是

$$
\lVert \mu_0 P^n - \pi \rVert_{\mathrm{TV}} = |\mu_0 P^n(1) - \pi(1)| = \pi(1)\,|\lambda_1|^n
$$

- **第四步，解混合时间**：要求 $\pi(1)|\lambda_1|^n \leq 1/4$，取对数得 $n \geq \frac{\log(4\pi(1))}{|\log|\lambda_1||}$。当 $\alpha=\beta$ 时 $\pi(1)=1/2$，混合时间 $\approx \log 2 / |\log|\lambda_1||$——**混合速度由第二大特征值的模决定**，$|\lambda_1|$ 越接近 1，混合越慢。

**数值验证**：取 $\alpha = \beta = 0.1$，则 $\lambda_1 = 0.8$、$\pi(1) = 1/2$，$n$ 步距离为 $0.5 \times 0.8^n$。压到 $1/4$ 需 $0.8^n \leq 0.5$，解得 $n \geq \log 0.5 / \log 0.8 \approx 3.1$，故 $t_{\mathrm{mix}}(1/4) = 4$。再取 $\alpha = \beta = 0.5$，$\lambda_1 = 0$，一步后分布即为 $\pi$，距离归零——两状态链几乎没有「空间扩散」瓶颈，混合完全由特征值衰减主导。

这个例子的公式 $\lVert \mu P^n - \pi \rVert_{\mathrm{TV}} \approx C |\lambda_1|^n$ 是「谱隙方法」的雏形，第 6 篇会把它推广到一般可逆链。

## 5 全变差距离的统计解释

全变差距离不只是一个抽象范数，它有清晰的统计含义。设 $X \sim \mu$ 或 $X \sim \nu$ 各以 $1/2$ 概率发生，观察者看到 $X$ 后要猜它来自哪个分布。最优判别器（Neyman–Pearson 意义下）的**最小犯错概率**满足

$$
\mathbb{P}_{\text{error}} = \frac{1}{2}\left(1 - \lVert \mu - \nu \rVert_{\mathrm{TV}}\right)
$$

所以 $\lVert\mu-\nu\rVert_{\mathrm{TV}} = 0$ 时猜中率只有 $50\%$（等同瞎猜），$\lVert\mu-\nu\rVert_{\mathrm{TV}} = 1$ 时能完全正确区分（$100\%$），中间的距离把成功率从 $50\%$ 线性抬升到 $100\%$。下表给出几个典型档位：

| 全变差距离 | 最优判别成功率 | 解读 |
| --- | --- | --- |
| $0$ | $50\%$ | 分布相同，等同瞎猜 |
| $0.25$ | $62.5\%$ | 部分可区分 |
| $0.5$ | $75\%$ | 明显可区分 |
| $0.75$ | $87.5\%$ | 高度可区分 |
| $1$ | $100\%$ | 支撑不相交，完全可区分 |

<span class="marginnote">这个联系让「混合时间」有了操作意义：$t_{\mathrm{mix}}(\varepsilon)$ 就是「跑这么多步之后，任何观察者都无法以超过 $1-\varepsilon$ 的胜率分辨链的状态与平稳采样」。这在大模型采样、MCMC 输出诊断里都有直接应用。</span>它把抽象的距离翻译成「能不能靠观测赢过掷硬币」，是统计可分辨性的精确语言。

**辨析｜易错点：** 全变差距离与「逐点相对误差」不是一回事。$\lVert \mu-\nu\rVert_{\mathrm{TV}} = 1/2 \sum|\mu(x)-\nu(x)|$ 允许某个状态上的相对误差极大，只要它在总质量里占比小。若应用里关心「稀有状态的命中率」（如 MCMC 采样稀有事件），全变差距离会过于乐观，需要改用其他度量（如分离距离）。

## 6 小结

- **全变差距离** $\lVert\mu-\nu\rVert_{\mathrm{TV}} = \max_A |\mu(A)-\nu(A)| = \tfrac12\sum_x|\mu(x)-\nu(x)|$，是分布间最坏事件误差，也是 $L^1$ 距离的一半。
- **第三个等价形式** $1 - \sum_x \min(\mu(x),\nu(x))$ 给出「最优耦合重叠」的直觉，是下一节耦合方法的入口。
- **收敛定理**：不可约非周期链从任意初值出发，全变差距离趋于 0；可逆链下随 $n$ 单调不增。
- **混合时间** $t_{\mathrm{mix}}(\varepsilon)$ 是最坏初值下把距离压到 $\varepsilon$ 的最少步数，$t_{\mathrm{mix}} = t_{\mathrm{mix}}(1/4)$ 是默认研究对象。
- 两状态链给出显式公式：距离 $\approx \pi(1)|\lambda_1|^n$