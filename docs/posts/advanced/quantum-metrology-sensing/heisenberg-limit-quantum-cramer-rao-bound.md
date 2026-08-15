---
title: 海森堡极限与量子 Cramér–Rao 界
date: 2026-08-07
---

# 海森堡极限与量子 Cramér–Rao 界

<div class="epigraph">
<p>位置测定得越精确，这一瞬间动量就测定得越不精确，反之亦然。</p>
<footer>—— 维尔纳 · 海森堡（Werner Heisenberg, *Zeitschrift für Physik\*, 1927）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子精密测量与量子传感 ｜ Giovannetti, Lloyd & Maccone, *Advances in quantum metrology\* §II ｜ 2026-08-07</p>
</div>

## 为什么需要一条原理性的界线

上一篇的标准量子极限 $1/\sqrt{n}$ 来自「粒子互不相关」的假设。但它不是原理性的——没有哪条定律规定粒子不能相关。那么问题来了：**给定 $n$ 个粒子，在量子力学允许的一切策略（包括任意纠缠、任意测量）下，估计精度到底有没有一个不可逾越的下界？** 答案是有的，它由**量子 Cramér–Rao 界（quantum Cramér–Rao bound, QCRB）**给出，而纠缠辅助下能达到的最优标度就是**海森堡极限（Heisenberg limit, HL）** $\Delta\varphi \sim 1/n$。这一篇要建立的，是连接「信息」「纠缠」「精度」三者的完整链条。

## 1 经典 Cramér–Rao 界与 Fisher 信息

先退到经典统计。假设概率分布 $p(x|\theta)$ 依赖未知参数 $\theta$，我们从数据 $x$ 中估计 $\theta$。**经典 Cramér–Rao 界**说：任何无偏估计量 $\hat{\theta}$ 的方差都不小于 Fisher 信息的倒数：

$$
\operatorname{Var}(\hat\theta) \ge \frac{1}{N\, F(\theta)}, \qquad
F(\theta) = \int \Big(\frac{\partial \ln p(x|\theta)}{\partial \theta}\Big)^2 p(x|\theta)\, dx
$$

**Fisher 信息 $F(\theta)$ 度量的是：分布 $p(x|\theta)$ 对 $\theta$ 的小扰动有多敏感。** 分布越「陡峭」，一次测量能带出的信息越多，精度上限越高。<span class="marginnote">Fisher 信息是信息论与统计学的核心量，与相对熵的关系是 $F = 2\lim_{\epsilon\to0} D_{\mathrm{KL}}(p_\theta \| p_{\theta+\epsilon})/\epsilon^2$。它度量分布的「曲率」，而曲率越大，参数越容易被分辨。</span>

当 Fisher 信息随测量次数线性增长 $F \propto N$ 时，就回到 $1/\sqrt{N}$ 的 SQL——**Cramér–Rao 界是 SQL 的严格版**。

## 2 量子 Fisher 信息与 QCRB

经典 Fisher 信息依赖具体的测量方案：同一个态，选不同的测量基，得到的 $F(\theta)$ 不同。**量子 Cramér–Rao 界**把这条界线推到最底——对所有可能的测量方案取最大：

$$
\Delta\theta \ge \frac{1}{\sqrt{N\, F_Q(\rho_\theta)}}
$$

其中 $F_Q$ 是**量子 Fisher 信息（quantum Fisher information, QFI）**，只依赖生成态 $\rho_\theta$，与测量方案无关。**QFI 是纯量子对象：它把「参数对态的影响」编码成信息量，纠缠的作用就体现在 $F_Q$ 身上。**<span class="marginnote">QFI 与 Bures 距离的关系是 $F_Q = 4\, d_{\mathrm{Bures}}^2(\rho_\theta, \rho_{\theta+d\theta})/d\theta^2$——它度量两个邻近量子态的可分辨性。Bures 距离是量子态空间的「度量」，相当于经典 Fisher 信息的量子对应物。</span>

对纯态 $|\psi_\theta\rangle$，QFI 有一个漂亮的封闭式：

$$
F_Q = 4\Big(\langle\dot\psi_\theta|\dot\psi_\theta\rangle - \big|\langle\psi_\theta|\dot\psi_\theta\rangle\big|^2\Big), \qquad |\dot\psi_\theta\rangle = \partial_\theta|\psi_\theta\rangle
$$

当演化由厄米生成元 $\hat{H}$ 给出，即 $|\psi_\theta\rangle = e^{-i\theta\hat{H}}|\psi_0\rangle$ 时，它简化为

$$
F_Q = 4\,(\Delta \hat{H})^2
$$

也就是 $4$ 倍生成元的方差。<span class="marginnote">这个「$F_Q = 4(\Delta H)^2$」是整篇的枢纽：想提高精度，要么增大生成元方差（用高能态、更多光子），要么把方差做成超线性标度——纠缠态恰恰能办到。</span>

## 3 海森堡极限：纠缠如何把 $\sqrt{n}$ 变成 $n$

现在看 $n$ 个粒子。若它们互不相关（直积态），总生成元 $\hat{H} = \sum_i \hat{h}_i$ 的方差是单粒子方差之和，$(\Delta H)^2 \propto n$，于是

$$
\Delta\theta \ge \frac{1}{\sqrt{4(\Delta H)^2}} \propto \frac{1}{\sqrt{n}} \quad\longleftarrow\quad \text{SQL}
$$

但若 $n$ 个粒子被制备成**纠缠态**，生成元方差可以做到 $(\Delta H)^2 \propto n^2$，于是

$$
\Delta\theta \propto \frac{1}{n} \quad\longleftarrow\quad \text{海森堡极限}
$$

**重点：纠缠态的本质贡献，是把「每个粒子独立贡献噪声」改成「全体粒子协同贡献信号」。** 一个经典的例子是 NOON 态 $|N{:}0{:}0{:}N\rangle = (|N,0\rangle + |0,N\rangle)/\sqrt{2}$：所有 $n$ 个光子同时处于一条臂或另一条臂，相位演化给出 $e^{-in\varphi}$ 的相位因子，有效相位被放大 $n$ 倍——单光子的波长被「压缩」成了 $n$ 光子叠加的短波长。<span class="marginnote">NOON 态名字来自它的形式：$N$ 个光子在路径 A、$0$ 个在 B，叠加 $0$ 个在 A、$N$ 个在 B。它是最早被提出、也最常被引用的纠缠测量态。</span>

做个数字对比：$n = 100$ 个光子测相位。相干态方案给 $\Delta\varphi \sim 1/\sqrt{100} = 0.1$；NOON 态方案给 $\Delta\varphi \sim 1/100 = 0.01$。**同样的光子数，纠缠把精度提高了 10 倍。** 若把 $n$ 推到 $10^4$，SQL 给 $0.01$，HL 给 $10^{-4}$——差距扩大到 100 倍。纠缠的收益随 $n$ 增大而增大，这正是「平方根 vs 线性」两条标度律的差距在资源维度上的指数放大。

**海森堡极限（HL）**就是这条标度律的名字：$1/n$。注意它并不违反不确定性原理——不确定性原理约束的是「单次测量的两个互补量」，而这里是「$n$ 个粒子协同估计一个参数」，二者不冲突。

## 4 公式解析：从 $F_Q = 4(\Delta H)^2$ 推出海森堡极限

把核心公式拆开：

$$
F_Q = 4(\Delta \hat{H})^2 = 4\Big(\langle\psi_0|\hat{H}^2|\psi_0\rangle - \langle\psi_0|\hat{H}|\psi_0\rangle^2\Big)
$$

- **第一步，量子 Fisher 信息的纯态公式**：对幺正参数化 $|\psi_\theta\rangle = e^{-i\theta\hat H}|\psi_0\rangle$，QFI 恰好是生成元方差的 $4$ 倍。这一公式把「态对参数的敏感度」完全编码进 $\hat{H}$ 的统计性质。
- **第二步，把方差拆开**：$\langle\hat{H}^2\rangle - \langle\hat{H}\rangle^2$。对直积态 $\hat{H}=\sum_i\hat{h}_i$，方差线性相加，$(\Delta H)^2 \propto n$。
- **第三步，纠缠改变标度**：若 $|\psi_0\rangle$ 是 $n$ 粒子纠缠态，$\hat{H}$ 的方差可以做到 $O(n^2)$。为什么？因为纠缠态里 $\langle\hat{h}_i\hat{h}_j\rangle$ 的关联项不为零，交叉项贡献了额外的 $n(n-1)$ 量级。**关联就是信息**——这正是纠缠超越 SQL 的数学根源。

**辨析｜易错点：** 海森堡极限 $1/n$ 不是无条件可达的。现实里光子损耗、退相干、非幺正演化都会把标度拉回 $1/\sqrt{n}$。严格地说，QCRB 给出的是「理想相干情形」的下界，在有损情形下即使纠缠也无法恢复 $1/n$——这一点在谈工程化的第十篇还要回来。

## 5 辨析：SQL 与 HL 的本质区别

| 性质 | 标准量子极限 (SQL) | 海森堡极限 (HL) |
| --- | --- | --- |
| 标度 | $1/\sqrt{n}$ | $1/n$ |
| 粒子状态 | 独立、无关联（直积态/相干态） | 纠缠态（NOON、GHZ 等） |
| 量子 Fisher 信息 | $F_Q \propto n$ | $F_Q \propto n^2$ |
| 数学来源 | 中心极限定理 / 泊松涨落 | 纠缠关联项 $O(n^2)$ |
| 是否原理性 | 否（可被纠缠突破） | 是（在幺正演化假设下不可突破） |

**辨析｜易错点：** 不要以为「HL 就是测不准原理的另一种说法」。测不准原理约束互补观测量（如位置与动量）的乘积；HL 约束的是「$n$ 个粒子对一个参数的估计精度」。两者都涉及「量子极限」这个词，但一个是单粒子双量、一个是多粒子单量，物理完全不同。

### 核心概念速查表

| 概念 | 记号/公式 | 一句话含义 |
| --- | --- | --- |
| Fisher 信息 | $F(\theta) = \int (\partial_\theta \ln p)^2 p\, dx$ | 分布对参数的敏感度，经典计量学的信息量 |
| 经典 Cramér–Rao 界 | $\operatorname{Var}(\hat\theta) \ge 1/(NF)$ | 任何无偏估计的方差下界 |
| 量子 Fisher 信息 | $F_Q$ | 对所有测量取最大的最优信息量，只依赖态 |
| 量子 Cramér–Rao 界 | $\Delta\theta \ge 1/\sqrt{NF_Q}$ | 量子测量精度的原理性下界 |
| 纯态 QFI | $F_Q = 4(\Delta\hat{H})^2$ | 幺正演化下等于 4 倍生成元方差 |
| NOON 态 | $\|N{:}0{:}0{:}N\rangle$ | 全部光子叠加在一条臂或另一条臂 |
| 标准量子极限 | $\Delta \propto 1/\sqrt{n}$ | 直积态，$F_Q \propto n$ |
| 海森堡极限 | $\Delta \propto 1/n$ | 纠缠态，$F_Q \propto n^2$ |

这张表是「极限」三篇的收束：第一篇给了测量公设，第二篇给了 SQL，本篇给了原理性的 QCRB 与 HL。压缩态篇会看到，实验上更可行的路径往往是在「全纠缠」与「独立」之间折中。

## 6 小结

- **经典 Cramér–Rao 界**：$\operatorname{Var}(\hat\theta) \ge 1/(N F(\theta))$，$F$ 是 Fisher 信息。
- **量子 Fisher 信息 $F_Q$**：对所有测量取最大的最优信息量，只依赖生成态，是纯量子对象。
- 纯态幺正演化下 **$F_Q = 4(\Delta\hat{H})^2$**，把精度问题变成「生成元方差多大」的问题。
- **标准量子极限 $1/\sqrt{n}$** 来自直积态的 $(\Delta H)^2 \propto n$；**海森堡极限 $1/n$** 来自纠缠态的 $(\Delta H)^2 \propto n^2$。
- NOON 态把有效相位放大 $n$