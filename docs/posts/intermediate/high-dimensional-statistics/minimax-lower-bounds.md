---
title: 高维 minimax 下界与信息论下界
date: 2026-08-11
---

# 高维 minimax 下界与信息论下界

<div class="epigraph">
<p>天下没有免费的午餐。</p>
<footer>—— 英文谚语；沃珀特与麦克里迪（Wolpert & Macready）1997 年以「No Free Lunch」定理将其精确化</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 高维统计分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 minimax 下界开始

第 5 篇我们学会了「样本复杂性」：一个算法要多少样本。当时我们留了一个悬案——那些界（比如 Lasso 的 $\sigma\sqrt{s\log d/n}$）是不是**最优**的？会不会存在一个我们没想到的算法，只需要少得多的样本？

minimax 下界把悬案画上句号。它要证明的是：**没有任何估计算法**（无论多聪明）能在最坏情形下把风险压到某个水平之下。这不只是理论洁癖——它告诉你 Lasso 的误差界「已经到底了」，再改进算法只能在常数与对数因子上打转。<span class="marginnote">「最坏情形」是 minimax 的灵魂：风险 $\sup_{\theta}$ 取遍所有可能的参数，是对「算法对每个问题的最坏表现」负责。这与贝叶斯风险（对参数取先验期望）是两种哲学——minimax 是对手博弈，贝叶斯是平均主义。高维统计的主体是前者。</span>

minimax 下界与信息论血脉相连：样本能传递的信息量有限，而参数空间「太大」，两相挤压，必有一批参数无法被分辨——这正是**「信息论下界」**四个字的由来。本节我们聚焦最常用的工具：Fano 不等式方法，辅以 Le Cam 方法与 Assouad 引理。

## 1 上界与下界：统计最优性的两面

设参数空间 $\Theta$，损失 $\ell(\cdot, \cdot)$，观测来自 $P_\theta$。**估计量**是数据的函数 $\hat\theta: \mathcal{X}^n \to \Theta$。定义

$$
\mathfrak{M}_n(\Theta) \;=\; \inf_{\hat\theta}\; \sup_{\theta \in \Theta}\; \mathbb{E}_\theta\big[\ell(\hat\theta, \theta)\big]
$$

这是 **minimax 风险**：先让对手挑最难的 $\theta$，再看最好的估计量能承受多少损失。「上界」是「我们的算法达到了 $U_n$」，即 $\mathfrak{M}_n \le U_n$；「下界」是「任何算法至少损失 $L_n$」，即 $\mathfrak{M}_n \ge L_n$。两者吻合（至多差常数/对数因子）时，问题就被**精确刻画**了。<span class="marginnote">这套「上界 + 下界 = 精确刻画」的叙事在本专题反复出现：Lasso 的上界（第 3 篇）、压缩感知的 RIP 保证（第 4 篇）、补全的样本数（第 6 篇），都等着下界来「验明正身」。minimax 下界是这整条论证链的最后一环。</span>

下界难在「任何算法」：你不能假设算法长什么样。对策是把问题**降维成一个有限假设检验**——找 $M$ 个「互相分离、又难分辨」的参数，任何估计器面对它们都只能靠猜。

## 2 三种经典方法

**Le Cam 方法（两点）**：找一对参数 $\theta^0, \theta^1$，距离 $d(\theta^0,\theta^1) \ge 2\delta$，但对应的分布 $P_0, P_1$ 几乎不可分辨（$\mathrm{TV}(P_0, P_1)$ 很小）。任何估计器必须在这两个「几乎一样」的分布下选一个方向，错误概率被总变差界住：

$$
\mathfrak{M}_n \;\ge\; \frac{\delta}{2}\big(1 - \mathrm{TV}(P_0, P_1)\big)
$$

**Assouad 引理（超立方体）**：取 $d$ 个「坐标」构成的超立方体 $\{-1,1\}^d$ 上的参数族，每个坐标给一个扰动。误差的 $\ell_2$ 范数下界变成各坐标「猜对方向」概率之和的下界。它擅长导出 $\sqrt{d/n}$ 型下界——每个自由坐标都要花样本。

**Fano 方法（多假设）**：取 $M$ 个候选参数，它们构成一个**$\delta$-分离的打包（packing）**——两两距离 $\ge 2\delta$。Fano 不等式从信息论角度限制「从样本里猜出是哪一个」的正确率。<span class="marginnote">三种方法各有主场：Le Cam 最省事但只能给两点的结论；Assouad 擅长超立方体与 $\ell_2$ 误差；Fano 最通用，能处理「组合数巨大」的稀疏/低秩参数空间——因为它只需要估计平均 KL 散度，而对数的基数正是 $\log M$。Fano 是本节的主角。</span>

先看一个 Le Cam 方法的微型演示，感受下界的「对手博弈」味。设 $X \sim N(\mu, \sigma^2)$，只关心「$\mu$ 靠近 $0$ 还是靠近 $\delta$」，取 $\theta^0 = 0$、$\theta^1 = \delta$。两个分布的总变差距离 $\mathrm{TV} \approx \sqrt{n\delta^2/\sigma^2}$ 量级，Le Cam 方法给出

$$
\inf_{\hat\mu} \max\big\{|\hat\mu - 0|,\; |\hat\mu - \delta|\big\} \;\ge\; \frac{\delta}{2}\left(1 - \sqrt{\frac{n\delta^2}{\sigma^2}}\right)
$$

取 $\delta \asymp \sigma/\sqrt n$ 时括号项保持常数，下界为 $\sigma/\sqrt n$ 量级——一维均值估计的 $\sigma/\sqrt n$ 速率被两句话钉死。这个微型例子的全部价值在于展示套路：**挑两个难分的参数 → 算总变差/KL → 套定理 → 反解 $\delta$**。Fano 方法只是把「两个参数」扩成「指数多个参数」，让组合爆炸替下界出力。

## 3 Fano 不等式：信息论的黑箱

**Fano 不等式**（通信理论的经典）：若 $I$ 是 $\{1,\dots,M\}$ 上的均匀随机指标，$\hat I$ 是 $I$ 的估计，则

$$
\mathbb{P}[\hat I \neq I] \;\ge\; 1 - \frac{H(I \mid \hat I) + 1}{\log M}
$$

直觉：要从输出 $\hat I$ 反推 $I$，条件熵 $H(I|\hat I)$ 是「反推的困难度」；若两者接近独立（$H(I|\hat I) \approx H(I) = \log M$），错误率就接近 $1 - 1/\log M \to 1$。<span class="marginnote">Fano 不等式原为信道编码理论里的工具（Claude Shannon 的同事 R. M. Fano 于 1952 年前后提出），它把「信息量不足 ⟹ 不能无错通信」精确化。统计学家借用它：把「从样本估计参数」视作「从观测解码消息」——样本是信道输出，参数是输入消息。</span>

把它接到估计问题：把参数族 $\{\theta^1, \dots, \theta^M\}$ 视为消息。由数据处理不等式，$H(I \mid \hat\theta) \ge H(I \mid \hat I)$，而互信息 $I(I; \hat\theta) \le \frac{1}{M}\sum_j \mathrm{KL}(P_j \| P_{\text{avg}})$。用凸性把平均换成**两两 KL 的平均**，得到统计版 Fano：

$$
\inf_{\hat\theta} \max_{j} \mathbb{P}\big[d(\hat\theta, \theta^j) \ge \delta\big]
\;\ge\; 1 - \frac{\bar D + \log 2}{\log M}, \qquad \bar D = \frac{1}{M^2}\sum_{j,k}\mathrm{KL}(P_{\theta^j} \| P_{\theta^k})
$$

其中要求打包是 $\delta$-分离的（两两距离 $\ge 2\delta$）。**只要假设族两两 KL 的平均 $\bar D$ 被控制住，下界就成立——KL 散度是信息论的货币，样本的「分辨力」以它为限。**

## 4 公式解析：Fano 方法应用到稀疏估计

把 Fano 方法用于 $s$-稀疏高斯均值估计（$\theta \in \mathbb{R}^d$，$\theta$ 有至多 $s$ 个非零元），可以得到

$$
\mathfrak{M}_n \;\ge\; c\,\sigma\sqrt{\frac{s}{n}\,\log\frac{d}{s}}
$$

（在平方 $\ell_2$ 风险下为 $\ge c\, \sigma^2 \frac{s}{n}\log\frac{d}{s}$）。四步拆解：

- **第一步，选打包**：在 $s$-稀疏向量空间里选 $M$ 个两两相距 $\ge 2\delta$ 的候选。一个高效构造：随机取 $s$ 个坐标位置，每个位置放 $\pm \delta$。组合计数给出 $M \approx \binom{d}{s} \approx \exp\big(s\log(d/s)\big)$，于是 $\log M \approx s\log(d/s)$——**打包的大小由支撑集组合数决定**，这是 $s\log(d/s)$ 的出处。
- **第二步，算 KL**：高斯观测 $Y_i = \theta + W_i$，$W_i \sim N(0,\sigma^2 I)$，两两 KL 为 $n\|\theta^j - \theta^k\|_2^2/(2\sigma^2) \le 2n\delta^2/\sigma^2$（由 $\delta$-分离）。取 $\delta \asymp \sigma\sqrt{s\log(d/s)/n}$ 可使 $\bar D \le \alpha \log M$（$\alpha$ 为小常数）。
- **第三步，套 Fano**：把 $\bar D \le \alpha\log M$ 代入，得错误概率 $\ge 1 - \frac{\alpha\log M + \log 2}{\log M}$——只要 $\alpha$ 足够小，这个概率不低于某个常数（如 $1/2$）。
- **第四步，把概率换成风险**：若估计器在 $\ge 1/2$ 概率下偏离 $\ge \delta$，则期望风险 $\ge \frac12 \cdot \delta = c\sigma\sqrt{s\log(d/s)/n}$。**风险下界成立——且它只用了「支撑集组合数」与「KL 代价」两条信息，估计器的任何巧妙结构都在打包选择面前失效。**

**辨析｜易错点：** 三个常踩的坑——其一，**打包（packing）不是覆盖（covering）**：覆盖要求「包住」，打包要求「互斥」，两者方向相反，混用会得出错误界；其二，**下界是 max，不是点态**：它说「存在某个 $\theta$ 使得误差大」，不是「所有 $\theta$ 都难」——把一个好估计的界误读成「对每个 $\theta$ 都成立」是最常见的错误；其三，KL 必须对**观测分布**（$P_\theta$ 的 $n$ 次幂）算，而不是对参数算——忘记乘 $n$ 会得到荒谬的「不依赖样本量」的下界。

## 5 应用：Lasso 的最优性与维数诅咒

把下界与前面的上界对齐，得到本专题最重要的一组结论：

- **稀疏回归**：minimax 下界 $\sigma\sqrt{s\log(d/s)/n}$（$\ell_2$）与 Lasso 上界 $\sigma\sqrt{s\log d/n}$ 吻合到对数因子——**Lasso 是（至多对数因子意义下的）最优估计器**。残余的 $\sqrt{\log s}$ 缺口来自 Lasso 的 $\ell_1$ 收缩偏差，可用阈值化/adaptive 修正补上。
- **矩阵补全**：自由参数 $r(d_1+d_2)$ 给出下界 $m \gtrsim r(d_1+d_2)$，与核范数方法的上界在 $(\log d)^2$ 内吻合——**补全同样接近最优**。
- **非参数回归**：$\alpha$-光滑函数在 $d$ 维上的 minimax 速率是 $n^{-2\alpha/(2\alpha+d)}$，指数里有 $d$——**维数诅咒从下界来看是不可避免的**，任何算法都无法逃过它，这就是「信息论下界」最深刻的教训：不是我们的算法笨，而是样本能携带的信息就只有这么多。<span class="marginnote">把这些结论放到一起，你会看到一个完整的图景：高维统计的「最优性」不是一个口号，而是一个可证明的事实——上界（算法）与下界（信息）两面夹击，把每个经典问题的速率定死在某个阶上。这正是「从极限到大模型」主线里统计与信息论交汇的高光时刻。</span>

## 6 小结

- **minimax 风险** $\inf_{\hat\theta}\sup_\theta \mathbb{E}\ell(\hat\theta,\theta)$ 是最坏情形下的最优风险；**下界**证明任何算法都无法超越它。
- 三种经典方法：**Le Cam**（两点）、**Assouad**（超立方体）、**Fano**（多假设打包），各有主场。
- **Fano 不等式**把估计问题翻译成「从样本解码消息」：错误率下界由**平均 KL 散度**与 $\log M$（打包大小）之比控制。
- 稀疏估计的下界 $\sigma\sqrt{s\log(d/s)/n}$ 来自「支撑集组合数 $\binom{d}{s}$」与高斯 KL 的对账。
- 上下界吻合证明 **Lasso、矩阵补全等算法最优（至多对数因子）**；非参数的维数诅咒（$n^{-2\alpha/(2\alpha+d)}$）是信息论必然。

在下一节，我们将把视线从「稀疏结构」移向「整个协方差结构」：当 $d$ 与 $n$ 同阶时，样本协方差矩阵本身会发生什么？——这是**协方差矩阵估计与随机矩阵理论**的领地。
