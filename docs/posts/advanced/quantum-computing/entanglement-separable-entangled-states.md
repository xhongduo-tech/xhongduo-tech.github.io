---
title: 纠缠的定义：可分态与纠缠态
date: 2026-08-07
---

# 纠缠的定义：可分态与纠缠态

<div class="epigraph">
<p>我无法相信量子纠缠这种可怕的机制……我不会相信上帝会掷骰子。</p>
<footer>—— 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§2.5 ｜ 2026-08-07</p>
</div>

## 为什么从纠缠的定义开始

前几篇反复用到「纠缠」这个词：量子并行靠纠缠、隐形传态靠纠缠、Shor 算法也靠纠缠。但**到底什么样的态才算纠缠？** 直觉上「两个比特之间有关联」并不必然等于纠缠——经典世界里两个比特也可以完全相关。真正区分「经典可解释的关联」与「量子独有的纠缠」的，是**可分态（separable state）**与**纠缠态（entangled state）**的精确划分。<span class="marginnote">纠缠是量子信息区别于经典信息的核心资源，它既不是简单的「相关」，也不是「一个态能同时取两个值」——这一节把概念彻底掰清，为贝尔不等式、纠缠度量、量子纠错全部后续内容奠基。</span>

## 1 可分态：可写成积的态

设系统分为两部分 $A$ 与 $B$（各为一个或多个比特），联合系统的状态空间是张量积 $\mathcal{H}_A \otimes \mathcal{H}_B$。一个纯态 $\lvert\psi\rangle$ 是**可分态（separable / product state）**，若存在 $\lvert\psi_A\rangle \in \mathcal{H}_A$ 与 $\lvert\psi_B\rangle \in \mathcal{H}_B$ 使得

$$
\lvert\psi\rangle = \lvert\psi_A\rangle \otimes \lvert\psi_B\rangle
$$

可分态能写成两个子系统态的**积**。例如 $\lvert0\rangle \otimes \lvert1\rangle$、$\frac{1}{\sqrt2}(\lvert0\rangle + \lvert1\rangle) \otimes \lvert+\rangle$ 都是可分态。<span class="marginnote">可分态的语义：两个子系统的状态完全独立地被指定，各自演化、互不干涉。测量 $A$ 不会对 $B$ 的态造成任何「不可预知」的影响——$B$ 的约化态一直是 $\lvert\psi_B\rangle$，确定不变。</span>

反过来，若一个纯态**不能**写成任何积的形式，就称为**纠缠态（entangled state）**。经典例子是贝尔态 $\lvert\Phi^+\rangle = \frac{1}{\sqrt2}(\lvert00\rangle + \lvert11\rangle)$——它无法写成 $\lvert\psi_A\rangle \otimes \lvert\psi_B\rangle$，因为写成积意味着两个比特的振幅可分离为两数之积，而这里 $\lvert00\rangle$ 与 $\lvert11\rangle$ 的系数不允许。

## 2 混合态情形：可分态的定义要更小心

对纯态，「写成积」与否干净利落；但对**混合态**（用密度算符描述，见《密度算符：混合态与部分迹》一篇），「可分」的定义变成**混合的积**：

$$
\rho = \sum_i p_i \, \rho_i^A \otimes \rho_i^B, \qquad p_i \ge 0,\; \sum_i p_i = 1
$$

即：可分混合态是一组「积态」的**概率混合**。<span class="marginnote">直觉：可分混合态可以被理解为「以概率 $p_i$ 制备积态 $\rho_i^A \otimes \rho_i^B$」——整个制备过程可以由经典随机性（掷骰子）驱动，不需要任何量子相干。</span>任何不满足这一形式的密度算符都是纠缠的。

**辨析｜易错点：** 混合态可分 ≠ 子系统不相关。可分混合态 $\rho = \frac12 \lvert00\rangle\langle00\rvert + \frac12 \lvert11\rangle\langle11\rvert$ 的两个比特**统计上完全相关**（测 $A$ 得 0，$B$ 必为 0），但它依然是可分的——因为它只是「两个确定积态的随机混合」，是经典相关。**纠缠必须是「量子相干叠加导致的、无法用经典概率解释的关联」**。这是本节最重要的一句话。

## 3 公式解析：为什么贝尔态不可分

用具体例子演示「为什么写不成积」。设 $\lvert\psi_A\rangle = a\lvert0\rangle + b\lvert1\rangle$、$\lvert\psi_B\rangle = c\lvert0\rangle + d\lvert1\rangle$，它们的积为

$$
\lvert\psi_A\rangle \otimes \lvert\psi_B\rangle = ac\lvert00\rangle + ad\lvert01\rangle + bc\lvert10\rangle + bd\lvert11\rangle
$$

而 $\lvert\Phi^+\rangle = \frac{1}{\sqrt2}\lvert00\rangle + \frac{1}{\sqrt2}\lvert11\rangle$。三步拆解：

- **第一步，对齐系数**：若 $\lvert\Phi^+\rangle$ 可写成积，则必须有 $ac = \frac{1}{\sqrt2}$、$ad = 0$、$bc = 0$、$bd = \frac{1}{\sqrt2}$。
- **第二步，导出矛盾**：由 $ad = 0$ 知 $a=0$ 或 $d=0$。若 $a=0$，则 $ac=0 \ne \frac{1}{\sqrt2}$，矛盾；若 $d=0$，则 $bd=0 \ne \frac{1}{\sqrt2}$，同样矛盾。
- **第三步，结论**：不存在任何 $a,b,c,d$ 满足四个方程，故 $\lvert\Phi^+\rangle$ 不可分，必为纠缠态。<span class="marginnote">这个「凑积不成」的论证是判定纯态纠缠的通用方法：把目标态按基展开，看系数矩阵的秩——纯态可分当且仅当系数矩阵秩为 1。$\lvert\Phi^+\rangle$ 的系数矩阵是 $\frac{1}{\sqrt2}\begin{pmatrix}1&0\\0&1\end{pmatrix}$，秩为 2。</span>

## 4 一个深刻的判据：约化态的纯度

有没有更省力的判据？有。**可分纯态的子系统约化态是纯态**；反之，纠缠纯态的子系统约化态必为混合态。于是可以用**冯·诺依曼熵**判定：

$$
S(\rho_A) = -\operatorname{tr}(\rho_A \log \rho_A) \; \begin{cases} = 0 & \lvert\psi\rangle \text{ 可分}\\ > 0 & \lvert\psi\rangle \text{ 纠缠} \end{cases}
$$

其中 $\rho_A = \operatorname{tr}_B(\lvert\psi\rangle\langle\psi\rvert)$ 是 $A$ 的约化密度算符。<span class="marginnote">直觉：若整体是积态，$A$ 的「视角」里 $B$ 完全不存在，$A$ 自己保持纯态；若整体纠缠，$A$ 丢失了与 $B$ 共享的关联信息，看起来就是混合的。对纯态，$S(\rho_A) = S(\rho_B)$ 永远成立——纠缠是「对称」的。</span>这个判据把「纠缠」翻译成「子系统有多混乱」，是后面《纠缠的度量：并发度与纠缠熵》一篇的核心起点。

**辨析｜易错点：** 上面的熵判据**只对纯态**有效。对混合态，$S(\rho_A) > 0$ 并不能推出 $\rho$ 纠缠——经典相关的混合态（上一节的例子）同样给出 $S(\rho_A) = 1 > 0$。混合态的纠缠判定需要**可分性判据**（如 PPT 判据、纠缠见证），是量子信息里最难的问题之一。

## 5 小结

- 纯态可分 = 可写成两个子系统态的**张量积**；否则为**纠缠态**。
- 混合态可分 = 可写成积态的**概率混合**；「经典相关」不等于「纠缠」。
- 贝尔态 $\lvert\Phi^+\rangle$ 不可分，因为系数无法分解为两数之积（系数矩阵秩为 2）。
- **熵判据**：纯态纠缠当且仅当子系统约化态为混合态（$S(\rho_A)>0$）；对混合态不适用。

在下一节，我们认识最著名的四个纠缠态——**贝尔态（Bell states）**，并学习如何把任意态「做成」贝尔态、如何对它们做测量。
