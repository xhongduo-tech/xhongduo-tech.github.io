---
title: Girsanov 定理与测度变换
date: 2026-08-11
---

# Girsanov 定理与测度变换

<div class="epigraph">
<p>同一个随机过程，换一把概率尺子来量，漂移就消失了——金融定价的秘密藏在这一换之中。</p>
<footer>—— 伊戈尔 · 吉拉萨诺夫（Igor Girsanov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 随机分析（Itô 微积分） ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从测度变换开始

前几节我们在**固定的一把概率尺子**（概率测度 $P$）下工作。但随机分析有一个近乎哲学级的武器：**概率测度本身是可以换的**。同一个随机过程 $\{X_t\}$，在测度 $P$ 下是带漂移的扩散，换到等价的测度 $Q$ 下可能就变成鞅。测度不变的是**轨道**（哪些路径可能、哪些不可能），改变的是**每条路径的权重**。

Girsanov 定理精确回答：**如何在换测度之后，让一个带漂移的布朗运动重新变回（新的）标准布朗运动。** 这之所以震撼，是因为它对金融理论是地基性的——衍生品定价的「风险中性测度」正是从这里诞生的；它也是辐射到统计（EM 算法、偏差校正）与机器学习（重要性采样、重加权）的核心思想。<span class="marginnote">对标 Karatzas &amp; Shreve §3.5 与 Protter 第三章。测度变换的直觉可追溯到英国精算师<strong>Cameron 与 Martin</strong> 对漂移的早期工作，Girsanov 把它推广到随机漂移。</span>

## 1 等价测度与 Radon–Nikodym 导数

两个测度 $P, Q$ 若对同一事件给零概率的集合完全一致，就称它们**等价（equivalent）**。等价测度之间由**Radon–Nikodym 导数**连接：

$$Z = \frac{dQ}{dP}, \qquad Q(A) = \int_A Z\,dP, \qquad E_P[\,Z\,] = 1, \; Z > 0 \text{ a.s.}$$

直观：$Z$ 给每一条样本路径一个「新权重」，把 $P$ 下对路径的看重程度重新分配成 $Q$。<span class="marginnote">等价测度改变的是「每条路径有多重」，不是「哪些路径可能」。<strong>如果两个测度不等价（比如给某条路径正概率而另一个给零），它们描述的就是不同的随机世界</strong>。</span>

**重点：换测度后，随机变量的期望按 $E_Q[X] = E_P[Z X]$ 换算。** 一切「换尺子」的运算，本质都是乘以 $Z$。

## 2 随机指数与 Girsanov 的核心构造

给定适应过程 $\theta_t$（想象它是对噪声的「再定价权重」），定义

$$\mathcal{E}(\theta)_t := \exp\Big(\int_0^t \theta_s\,dB_s - \frac12 \int_0^t \theta_s^2\,ds\Big).$$

这是**随机指数（stochastic exponential）**。由 Itô 公式可得 $d\mathcal{E}(\theta)_t = \mathcal{E}(\theta)_t \,\theta_t\,dB_t$，所以 $\mathcal{E}(\theta)$ 是一个鞅（至少是局部鞅）。<span class="marginnote">随机指数是鞅这个事实，靠的是它把 $\theta dB$ 的「能量修正」$-\tfrac12 \theta^2 ds$ 也一并算进了指数里——<strong>这正是 Itô 公式给出的凸性修正</strong>。</span>

为保证全局的鞅性（而非仅局部鞅），需要 **Novikov 条件**：

$$E\Big[\exp\Big(\frac12 \int_0^T \theta_s^2\,ds\Big)\Big] < \infty,$$

它在绝大多数应用中自动满足。

## 3 Girsanov 定理：漂移的消失术

**Girsanov 定理**：设 $\theta$ 满足 Novikov 条件，定义测度

$$\frac{dQ}{dP}\Big|_{\mathcal{F}_T} = \mathcal{E}(\theta)_T = \exp\Big(\int_0^T \theta_s\,dB_s - \frac12\int_0^T \theta_s^2\,ds\Big),$$

则过程

$$\widetilde B_t := B_t - \int_0^t \theta_s\,ds$$

在测度 $Q$ 下是标准布朗运动。

**重点：换测度把「$P$ 下的带漂移过程 $B$」变成「$Q$ 下的标准布朗运动 $\widetilde B$」。** 反过来读更常用：若在 $P$ 下 $\widetilde B$ 是标准布朗运动，那么 $B_t = \widetilde B_t + \int_0^t \theta_s\,ds$ 在 $Q$ 下是有漂移的布朗运动。方向可以自由选，这是整个定价理论的开关。<span class="marginnote">为什么漂移会消失？关键在于 $Q$ 改变了测度后，Lévy 特征定理检查的「二次变差」在等价测度下不变：$\langle B\rangle^Q_t = \langle B\rangle^P_t = t$，而 $B$ 在 $Q$ 下仍是连续局部鞅——于是 Lévy 特征自动判定它是 $Q$-布朗运动。<strong>这正是上一节铺垫 Lévy 特征的回报。</strong></span>

## 4 公式解析：$\frac{dQ}{dP}$ 的每一项各司其职

把核心公式拆成三步读：

- **第一步，认识 $\int_0^t \theta_s\,dB_s$**：这是「重新定价」的线性部分——每条路径按 $\theta$ 对噪声的暴露程度被加权。$\theta$ 大，走「被噪声推得远」的路径在 $Q$ 下就更可能。
- **第二步，认识 $-\frac12\int_0^t \theta_s^2\,ds$**：这是「归一化」的二次修正。没有它，$E_P[\exp(\int\theta\,dB)] \ne 1$，$Q$ 就不是概率测度。它确保 $Z$ 的 $P$-期望恰为 1——而这个「恰为 1」，正是由鞅性 + Novikov 条件担保的。
- **第三步，整体读出**：$Z_t = \mathcal{E}(\theta)_t$ 是密度过程，$dQ = Z_T\,dP$；在 $Q$ 下 $B_t$ 的漂移项 $\int_0^t \theta_s\,ds$ 被「吃掉」，$\widetilde B$ 重新成为标准布朗运动。

**一句话：$\int\theta\,dB$ 给路径重定价，$-\tfrac12\int\theta^2\,ds$ 保证测度归一，漂移消失是这笔交易的余额。**

## 5 应用预告：把漂移从 SDE 里抹掉

经典操作：考虑带漂移的 SDE $dX_t = \mu(X_t)\,dt + \sigma\,dB_t$。想让它变成鞅（对定价极有价值），取 $\theta_t = -\mu(X_t)/\sigma(X_t)$，则

$$\widetilde B_t := B_t + \int_0^t \frac{\mu(X_s)}{\sigma(X_s)}\,ds$$

在 $Q$ 下是标准布朗运动，于是

$$dX_t = \sigma(X_t)\,d\widetilde B_t$$

——**漂移被测度吸收，只剩下扩散**。这个「风险中性测度」下的鞅表示，是下一节 Black-Scholes 定价的直接铺垫。<span class="marginnote">注意一个优雅的分工：<strong>Lévy 特征保证二次变差不变，Girsanov 只改漂移；两者拼起来，说明「漂移不是轨道的客观属性，而是测度的观点」</strong>——这个思想也贯穿统计力学与信息几何。</span>

**辨析｜易错点：** Girsanov 能删掉漂移，**删不掉扩散系数**。$\sigma$ 是由二次变差决定的「客观量」，换等价测度不改变它；若想改变 $\sigma$，只能去改变时间（重新参数化），那已经超出等价格测度变换的范畴。

**一个记忆锚点**：Girsanov 与 Lévy 特征是一对镜像——Lévy 说「二次变差 $= t$ 的连续局部鞅是布朗运动」，Girsanov 说「我可以在不动二次变差的前提下改写漂移」。**把两个定理放一起读，你会得到一幅干净图景：布朗运动的身份（Lévy）与它头上的漂移（Girsanov）是两件可以独立处理的事。**

## 6 例：显式密度与一次漂移消除

设 $P$ 下 $B$ 是标准布朗运动，取**常数** $\theta$，则 $Q = \mathcal{E}(\theta)_T \cdot P$，即

$$\frac{dQ}{dP} = \exp\big(\theta B_T - \tfrac12 \theta^2 T\big).$$

$B_T \sim \mathcal{N}(0, T)$ 在 $P$ 下，于是用矩生成函数直接核对归一化：

$$E_P\Big[\frac{dQ}{dP}\Big] = E_P\big[e^{\theta B_T}\big] e^{-\frac12\theta^2 T} = e^{\frac12\theta^2 T} e^{-\frac12\theta^2 T} = 1.$$

**看，归一化自动成立——因为 $e^{\theta B_T}$ 的矩生成函数恰为 $e^{\theta^2 T/2}$，指数里的 $-\tfrac12\theta^2T$ 正是为「吃掉」这个矩而设。** 而 $Q$ 下 $B_t$ 的分布变为 $\mathcal{N}(\theta t, t)$（对 $t \le T$）——一个带漂移 $\theta t$ 的「伪布朗运动」。这就是经典的 **Cameron–Martin 定理**：对确定性漂移，测度变换的密度是简单的指数因子。

**反向读法（Girsanov 更常用的方向）**：若 $Q$ 下 $\widetilde B$ 是标准布朗运动，则 $P$ 下 $B_t = \widetilde B_t + \theta t$ 是带漂移的布朗运动。**漂移从来不是轨道的内在属性，而是「用什么尺子去量」的产物。** 这一条几乎成了现代随机分析的第一直觉。

**应用：重要性采样与偏差校正。** 若 $E_P[\phi(B)]$ 不好算，可改用 $E_Q[\phi(B)/Z]$——只要 $Z$ 好采样、$\phi/Z$ 在 $Q$ 下方差更小。机器学习里的重加权、统计里的偏差校正、粒子滤波里的重采样，都是同一个「换测度」动作。

## 7 等价测度在统计与机器学习里的身影

测度变换不只在金融。**重要性采样**：想把 $E_P[\phi(X)]$ 用 $Q$ 下的样本估计，就写 $E_P[\phi] = E_Q[\phi \cdot dP/dQ]$——密度比就是「重新加权」。**偏差校正**：拒绝采样、EM 算法的 E 步、粒子滤波的重采样，都在显式或隐式地维护一个 Radon–Nikodym 导数。

**重点：Girsanov 定理的不可替代之处，在于它给出了这个密度比的「闭式表达式」**——对连续时间过程，密度比就是随机指数，而随机指数是可模拟、可求梯度、可微分估计的。机器学习里的重加权风险最小化、因果推断里的倾向得分加权，都是这张表上的常客。<span class="marginnote">工程直觉：<strong>换测度 = 给样本换权重；Girsanov 告诉你权重是「随机指数的函数」</strong>——密度比的闭式，让「在不同世界之间来回搬期望」变得可计算。</span>

（尾声：下一次你听到「换个角度看看」，请想起 Girsanov——有些问题不是算不出来，而是站错了概率测度。换一把尺子，漂移可能就消失了。而下一节要回答的问题是：换完尺子之后，那个过程在更大的过程家族里住在哪——这正是局部鞅与半鞅的地图。）

（再补一句：等价测度关系构成一个「等价类」——所有与 $P$ 等价的测度是一个集合，Girsanov 说我们可以在这个集合里自由移动，只要随机指数是鞅。测度论里的「等价」，在金融里就是「定价一致」的同义反复。）

## 8 小结

- **等价测度**不改变轨道的可能集合，只改变权重；换算工具是 Radon–Nikodym 导数 $Z = dQ/dP$。
- **随机指数** $\mathcal{E}(\theta)_t = \exp(\int\theta\,dB - \tfrac12\int\theta^2\,ds)$ 是鞅（局部鞅 + Novikov 条件保全局鞅性）。
- **Girsanov 定理**：换测度后 $B_t - \int\theta_s\,ds$ 成为新测度下的标准布朗运动；漂移是测度的观点，不是轨道的属性。
- **作用机制**：$\int\theta\,dB$ 重定价，$-\tfrac12\int\theta^2\,ds$ 保归一，Lévy 特征保二次变差。
- **标准应用**：用 $\theta = -\mu/\sigma$ 把带漂移 SDE 变成测度下的鞅，是风险中性定价的引擎。

在下一节，我们将正式介绍一个比鞅更大的过程家族——**局部鞅与半鞅理论**，它是 Itô 积分真正的生活环境，也是前面所有「局部化」手法的理论归宿。
