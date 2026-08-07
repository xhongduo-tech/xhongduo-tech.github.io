---
title: 全期望公式与条件方差公式
date: 2026-08-07
---

# 全期望公式与条件方差公式

<div class="epigraph">
<p>做数学的艺术，在于找到那个包含全部普遍性萌芽的特例。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》第1章 ｜ 2026-08-07</p>
</div>

## 为什么从全期望公式开始

上一批我们花四篇篇幅把条件期望从「定义」一路建到了「性质」。但随机过程真正天天在用的，其实是两条**衍生公式**：全期望公式（law of total expectation）与条件方差公式（law of total variance）。它们不引入新概念，却把条件期望变成了可以反复套用的**计算引擎**。

为什么说它们是引擎？因为随机过程里几乎没有一次期望是能直接算的——泊松过程里「一段随机时间内来了多少个事件」、更新过程里「到第 $n$ 次更新要等多久」、复合泊松里「随机数目的随机变量之和」，全都长成「内层随机、外层再随机」的嵌套结构。<span class="marginnote">「内层随机、外层随机」正是本专题第二篇泊松过程、第三篇更新过程的常态：比如复合泊松 $S = \sum_{i=1}^{N} X_i$，外层 $N$ 随机，内层每个 $X_i$ 也随机。</span>凡是这种结构，第一步永远是：先固定外层，算内层；再用全期望公式把外层平均掉。这一篇就把这条「先分层、再平均」的套路练到肌肉记忆。

这一篇也顺带把上一篇《条件期望的性质》里最后一行提到的条件方差公式展开成完整定理，并给出它在统计与机器学习里的一个著名投影——**偏差-方差分解**。

## 1 全期望公式：先分组，再平均

最朴素的版本来自全概率公式。设 $A_1, A_2, \dots, A_n$ 是样本空间的一个**划分**（两两互斥且并为全集），则

$$
\mathbb{E}[X] = \sum_{i=1}^{n} P(A_i) \, \mathbb{E}[X \mid A_i]
$$

读法很直白：**把样本空间切成分组，每一组内的平均按该组概率加权，加权平均就是总平均。** 这几乎就是日常「班级平均分 = 各班平均分的加权平均」的数学化。

更常用的写法是让分组由另一个随机变量 $Y$ 决定。若 $Y$ 取有限或可数个值 $y_1, y_2, \dots$，则

$$
\mathbb{E}[X] = \sum_{j} P(Y = y_j) \, \mathbb{E}[X \mid Y = y_j]
$$

**核心直觉：一个随机量在信息不完整时的最佳猜测，等于先按每个可能的中间信息做猜测、再对中间信息本身求平均。**<span class="marginnote">把「信息」拟人化：你只知道 $Y$，你对 $X$ 的猜测是 $\mathbb{E}[X\mid Y]$；再把 $Y$ 也当成随机的平均掉，就得到一无所知时的猜测 $\mathbb{E}[X]$。中间那层信息被「稀释」成它的期望。</span>

用记号写，全期望公式就是一行：

$$
\mathbb{E}[X] = \mathbb{E}\bigl[\mathbb{E}[X \mid Y]\bigr]
$$

这个三重的写法值得盯十秒钟：内层 $\mathbb{E}[X \mid Y]$ 是 $Y$ 的函数（随机变量），外层是对 $Y$ 取期望。

**一个立刻见效的例子**：某路口车辆数 $N \sim \text{Poisson}(\lambda)$，每辆车载人数 $X_i$ 独立同分布，期望 $E[X_i] = \mu$。问车上总人数的期望 $\mathbb{E}\left[\sum_{i=1}^{N} X_i\right]$。没人能直接算——$N$ 还随机着呢。但先固定 $N=n$：此时和是 $n$ 个独立同分布变量之和，$\mathbb{E}\left[\sum_{i=1}^{n} X_i\right] = n\mu$。于是

$$
\mathbb{E}\left[\sum_{i=1}^{N} X_i\right] = \mathbb{E}\left[\mathbb{E}\left[\sum_{i=1}^{N} X_i \;\middle|\; N\right]\right] = \mathbb{E}[N\mu] = \lambda \mu
$$

这就是**复合分布的均值公式**：外层个数期望乘以内层单项期望。整个计算只有两步——先固定 $N$，再对 $N$ 平均。

## 2 塔性质的视角：全期望公式是它的特例

上一篇我们把塔性质写成：对 $\mathcal{H} \subset \mathcal{G}$，

$$
\mathbb{E}\bigl[\mathbb{E}[X \mid \mathcal{G}] \mid \mathcal{H}\bigr] = \mathbb{E}[X \mid \mathcal{H}]
$$

取 $\mathcal{G} = \sigma(Y)$、$\mathcal{H} = \{\emptyset, \Omega\}$（平凡 σ-代数，不含任何信息），则 $\mathbb{E}[X \mid \mathcal{H}] = \mathbb{E}[X]$（常数），于是

$$
\mathbb{E}\bigl[\mathbb{E}[X \mid Y]\bigr] = \mathbb{E}[X]
$$

**全期望公式正是塔性质在「先按 $Y$ 压缩、再压到空信息」时的特例。** 这解释了为什么两者读起来那么像——它们是同一条「信息只能变少」原理的两种讲法。<span class="marginnote">这也是上一篇强调塔性质的原因：随机过程里所有「先条件化再平均」的步骤，本质上都是在重复这一条性质。你不需要记住许多公式，记住「信息只能变少」就够了。</span>

反过来，塔性质也提醒我们一个常常被忽略的方向问题：$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\mid Y]]$ 只能「先细后粗」。若先取无条件期望再条件化，$\mathbb{E}[\mathbb{E}[X] \mid Y] = \mathbb{E}[X]$，什么都没变——信息一旦在第一步就丢光，后面再也找不回来。

## 3 条件方差公式：方差分解

期望只有一条公式，方差则多出一条对偶的、同样重要的**方差分解公式（law of total variance）**，也叫 EVE 律：

$$
\operatorname{Var}(X) = \mathbb{E}\bigl[\operatorname{Var}(X \mid Y)\bigr] + \operatorname{Var}\bigl(\mathbb{E}[X \mid Y]\bigr)
$$

其中 $\operatorname{Var}(X \mid Y) = \mathbb{E}[X^2 \mid Y] - \bigl(\mathbb{E}[X \mid Y]\bigr)^2$ 是**条件方差**——固定 $Y$ 后 $X$ 的方差。

这条公式值得用中文读出来：**总方差 = 组内方差的平均 + 组间均值的方差。** 两组信息各管一块：第一项 $\mathbb{E}[\operatorname{Var}(X\mid Y)]$ 衡量「即使知道 $Y$，$X$ 仍有的波动」；第二项 $\operatorname{Var}(\mathbb{E}[X\mid Y])$ 衡量「$Y$ 的不同取值让均值本身发生多大的漂移」。

**易错点先打预防针**：只取 $\mathbb{E}[\operatorname{Var}(X\mid Y)]$ 是**错**的——它丢掉了第二项。直觉上，若 $Y$ 能强烈改变 $X$ 的均值（比如 $Y$ 是「下雨/晴天」，$X$ 是降水量），那么「知道 $Y$」本身就解释了大量方差，这部分必须用第二项记下。

回到路口例子，若再给每辆车载人数的方差 $\sigma^2$，则总人数 $S = \sum_{i=1}^{N} X_i$ 的方差是

$$
\operatorname{Var}(S) = \mathbb{E}\bigl[\operatorname{Var}(S \mid N)\bigr] + \operatorname{Var}\bigl(\mathbb{E}[S \mid N]\bigr) = \mathbb{E}[N \sigma^2] + \operatorname{Var}(N \mu) = \lambda\sigma^2 + \mu^2 \lambda
$$

合并成 $\lambda(\sigma^2 + \mu^2) = \lambda \mathbb{E}[X_1^2]$——这是复合泊松的方差公式，第二篇泊松过程还要正式登场。

## 4 公式解析：方差分解为什么成立

把 $\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X\mid Y)] + \operatorname{Var}(\mathbb{E}[X\mid Y])$ 拆成四步，你会发现它只是「期望平方减平方期望」这条定义被应用了三次。

- **第一步，写条件方差的定义**：$\operatorname{Var}(X \mid Y) = \mathbb{E}[X^2 \mid Y] - \bigl(\mathbb{E}[X \mid Y]\bigr)^2$。这是「平方的平均减平均的平方」在条件化版本下的复制。
- **第二步，对两边取期望**，并利用全期望公式：

$$\mathbb{E}\bigl[\operatorname{Var}(X \mid Y)\bigr] = \mathbb{E}\bigl[\mathbb{E}[X^2 \mid Y]\bigr] - \mathbb{E}\Bigl[\bigl(\mathbb{E}[X \mid Y]\bigr)^2\Bigr] = \mathbb{E}[X^2] - \mathbb{E}\Bigl[\bigl(\mathbb{E}[X \mid Y]\bigr)^2\Bigr]$$

- **第三步，算组间项**：$\operatorname{Var}\bigl(\mathbb{E}[X\mid Y]\bigr) = \mathbb{E}\Bigl[\bigl(\mathbb{E}[X\mid Y]\bigr)^2\Bigr] - \bigl(\mathbb{E}[\mathbb{E}[X\mid Y]]\bigr)^2$，其中外层期望用全期望公式得 $\mathbb{E}[\mathbb{E}[X\mid Y]] = \mathbb{E}[X]$，所以

$$\operatorname{Var}\bigl(\mathbb{E}[X\mid Y]\bigr) = \mathbb{E}\Bigl[\bigl(\mathbb{E}[X\mid Y]\bigr)^2\Bigr] - \bigl(\mathbb{E}[X]\bigr)^2$$

- **第四步，相加**：注意第二步与第三步中 $\mathbb{E}\bigl[\bigl(\mathbb{E}[X\mid Y]\bigr)^2\bigr]$ 一负一正正好抵消，剩下

$$\mathbb{E}[\operatorname{Var}(X\mid Y)] + \operatorname{Var}(\mathbb{E}[X\mid Y]) = \mathbb{E}[X^2] - \bigl(\mathbb{E}[X]\bigr)^2 = \operatorname{Var}(X)$$

**整个证明的机关在第四步**：那个讨厌的中间项 $\bigl(\mathbb{E}[X\mid Y]\bigr)^2$ 被两边「共享」，相加时抵消。这是方差分解公式的隐藏结构——它把「均值漂移造成的方差」与「组内剩余方差」分得干干净净。

## 5 应用一：混合分布的均值与方差

**混合分布（mixture distribution）**是「先抽一个类别，再从该类别里抽值」的一族分布，写法为 $X \sim \sum_{k=1}^{K} \pi_k \, F_k$。这类分布是统计建模里描述**异质性**的标配：顾客分成几类、每个类内部又各有波动。

设 $K=2$，$\pi_1 = \pi_2 = 0.5$，类别 $C=1$ 时 $X \sim \mathcal{N}(0, 1)$，类别 $C=2$ 时 $X \sim \mathcal{N}(5, 1)$。均值是平凡的全期望：

$$
\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid C]] = 0.5 \times 0 + 0.5 \times 5 = 2.5
$$

方差用分解公式：

$$
\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X \mid C)] + \operatorname{Var}(\mathbb{E}[X \mid C]) = 1 + \bigl[(0-2.5)^2 \times 0.5 + (5-2.5)^2 \times 0.5\bigr] = 1 + 6.25 = 7.25
$$

注意这比单个高斯分布的方差 $1$ 大得多——**混合把分布的方差「撑开」了**。第二项 $6.25$ 纯粹来自「两个类别中心相距 5」，这部分方差是类别结构贡献的，单看类内方差 $1$ 完全看不出来。高斯混合模型（GMM）之所以是聚类与密度估计的经典工具，正是因为它显式地建模了这第二项。

## 6 应用二：偏差-方差分解与随机过程预演

方差分解公式在统计学习里有一个著名的化身——**偏差-方差分解**。设 $f$ 是真实回归函数，$\hat{f}$ 是估计器，$X$ 为输入，噪声 $\varepsilon$ 均值为零，那么均方误差

$$
\mathbb{E}\bigl[(\hat{f}(x) - f(x))^2\bigr] = \underbrace{\bigl(\mathbb{E}[\hat{f}(x)] - f(x)\bigr)^2}_{\text{偏差}^2} \;+\; \underbrace{\mathbb{E}\bigl[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\bigr]}_{\text{方差}}
$$

把 $Y$ 换成「训练数据 $\mathcal{D}$」就一目了然：总误差 = 系统性的偏（偏差）+ 数据波动引起的抖（方差）。这与 $\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X\mid Y)] + \operatorname{Var}(\mathbb{E}[X\mid Y])$ 是同一个骨架——**训练数据扮演 $Y$ 的角色，它既通过均值漂移贡献方差，又在固定时留下残余误差**。这将在第三级《机器学习》中反复出现；而到了本专题第四篇马尔可夫链、第五篇连续时间链，条件方差公式还会在「给定过去、预测未来」的波动分析里再次登场。<span class="marginnote">偏差-方差是机器学习里最著名的「方差分解」实例之一，见第三级《机器学习》专题；而它的数学根源就是本节这条 EVE 律——这也是本博客「从极限到大模型」主线里条件期望工具第一次在 AI 语境落地。</span>

## 7 辨析：三个高频错误

**辨析｜易错点：** 第一，**漏掉组间项**。求混合分布方差时只写 $\mathbb{E}[\operatorname{Var}(X\mid Y)]$ 忘记 $\operatorname{Var}(\mathbb{E}[X\mid Y])$，是出现频率最高的错误；只要 $Y$ 对 $X$ 的均值有影响，第二项就不可省略。第二，**把 $\mathbb{E}[X \mid Y]$ 当常数**。它是 $Y$ 的函数、是随机变量，取方差必须按随机变量处理，不能丢进 $\operatorname{Var}$ 后仍写 $\operatorname{Var}(\mathbb{E}[X\mid Y]) = 0$。第三，**混淆条件方差与无条件方差**：$\operatorname{Var}(X \mid Y = y)$ 是**一个数**，$\operatorname{Var}(X \mid Y)$ 是**一个随机变量**（随 $Y$ 变），$\operatorname{Var}(X)$ 是**一个常数**——三个对象性质完全不同，公式中不可互换。

## 8 小结

- **全期望公式**：$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\mid Y]] = \sum_j P(Y=y_j)\,\mathbb{E}[X\mid Y=y_j]$，是塔性质在平凡 σ-代数下的特例，是「嵌套随机」的第一步计算工具。
- **条件方差公式（EVE 律）**：$\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X\mid Y)] + \operatorname{Var}(\mathbb{E}[X\mid Y])$，总方差 = 组内方差的平均 + 组间均值的方差。
- 复合结构（随机个随机变量之和）的均值、方差都用这两条公式：$E[S_N] = E[N]E[X]$，$\operatorname{Var}(S_N) = E[N]\operatorname{Var}(X) + \operatorname{Var}(N)(E[X])^2$。
- 混合分布的方差被类别结构「撑开」，第二项不可省；偏差-方差分解是 EVE 律在机器学习中的化身。

在下一节，我们将回答「如何用一个函数装下随机变量的全部矩」——这就是**矩母函数与特征函数**：它们把分布编码成生成函数，让「独立和的卷积」变成一次乘法。
