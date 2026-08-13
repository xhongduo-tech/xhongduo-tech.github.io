---
title: 光子数态与 Fock 态
date: 2026-08-07
---

# 光子数态与 Fock 态

<div class="epigraph">
<p>自然喜欢简单，而不喜欢炫耀多余的原因。</p>
<footer>—— 艾萨克·牛顿（Isaac Newton）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子光学 ｜ R. Loudon, The Quantum Theory of Light 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从光子数态开始

上一节我们把电磁场写成了无穷多谐振子，
但还没有仔细看它的**本征态**。光子数态（Fock 
态）就是这台谐振子的能量本征态：$|n\rangle$ 
表示该模式**恰好有 $n$ 个光子**。
它是最「量子」的光场——因为它没有经典对应物。一台经典激光器可以输出 
0.5 个单位的平均能量，但「正好半个光子」在 Fock 
态语言里没有意义；光子数必须是整数。理解 Fock 态，
是后面理解相干态、压缩态、
反聚束和单光子源的基石。<span class="marginnote">Fock 
态以苏联物理学家弗拉基米尔·福克（Vladimir Fock）命名，
他在 1932 
年提出了这种编号态空间——它也是量子场论里「占据数表示」的原型。</span>

## 1 Fock 态的定义与基本性质

**Fock 态（光子数态）**：
数算符 $\hat{N} = \hat{a}^\dagger\hat{a}$ 
的本征态，满足

$$\hat{N}|n\rangle = n|n\rangle, \qquad n = 0, 1, 2, \ldots$$

Fock 态构成完备正交基，
任意单模光场态 $|\psi\rangle$ 
都可以展开为 $|\psi\rangle = \sum_n c_n |n\rangle$。
正交归一性是

$$\langle n | m \rangle = \delta_{nm}, \qquad \sum_{n=0}^{\infty} |n\rangle\langle n| = \mathbb{1}$$

**重点：Fock 态的光子数是完全确定的，因此它的相位完全不确定。** 
这就像一枚硬币：确定了「有几个」，就丢了「在哪儿摆」的信息。
下一节《量子相位》将专门讨论这条互补关系。

## 2 产生与湮灭算符的作用

产生湮灭算符在 Fock 
态上的作用遵循一条严格规则——这是全篇最需要背熟的一条链：


$$\hat{a}|n\rangle = \sqrt{n}\,|n-1\rangle, \qquad \hat{a}^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle$$

对真空态连续作用产生算符 $n$ 次，就能「爬」到任意 Fock 态：

$$|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}\,|0\rangle$$

系数 $\sqrt{n}$ 和 $\sqrt{n+1}$ 不是装饰，
它保证数算符的本征值恰好是 $n$：$\hat{a}^\dagger\hat{a}|n\rangle = \hat{a}^\dagger\sqrt{n}|n-1\rangle = \sqrt{n}\cdot\sqrt{n}|n\rangle = n|n\rangle$。<span class="marginnote">这些系数正是谐振子量子力学里「阶梯算符」的老朋友：
算符 $a^\dagger$ 是升算符，$a$ 是降算符，
系数 $\sqrt{n+1}$ 
是量子统计「玻色增强」的代数来源。</span>

**辨析｜易错点：** 
把 $\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$ 
错记成 $\sqrt{n+1}$ 是高频错误。记忆法：
湮灭算符「拿掉一个光子」，所以系数跟**当前**光子数 $n$ 走；
产生算符「放回一个光子」，系数跟**放回后**的 $n+1$ 走。

## 3 光子数态的时间演化与相位模糊

在自由哈密顿量 $\hat{H}_0 = \hbar\omega\hat{a}^\dagger\hat{a}$ 
下，Fock 态只获得一个整体相位：

$$|n(t)\rangle = e^{-i\hat{H}_0 t/\hbar}|n(0)\rangle = e^{-in\omega t}|n(0)\rangle$$

整体相位不出现在任何可观测量里，
所以 **Fock 态在自由演化下不变**——它是个「定态」。
要得到时间依赖的电场，必须叠加多个 $n$：

$$\hat{E}(t) = \sum_n c_n e^{-in\omega t} \,\hat{E}_{n+1,n}|n\rangle\langle n+1| + \text{h.c.}$$

**这意味着 Fock 态的期望电场恒为零**，$\langle n|\hat{E}|n\rangle = 0$：
因为 $\langle n|\hat{E}|n\rangle$ 
只含 $\langle n|\hat{a}|n\rangle$ 
与 $\langle n|\hat{a}^\dagger|n\rangle$，
而它们都等于零。一个「正好有 $n$ 个光子」的场，电场平均值为零，
只存在涨落——这是 Fock 态最反直觉的性质，
也是它与「经典正弦波」的根本鸿沟。<span class="marginnote">想要电场有确定的振荡相位，
就必须把相邻 Fock 
态相干叠加——这正是下一节<strong>相干态</strong>要做的事：
它牺牲了光子数的确定性，换来了确定的相位。</span>

## 4 公式解析：$|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$

这条式子是 Fock 态的「构造公式」，拆成三步：

**第一步，从真空出发**：$|0\rangle$ 是零光子态，$\hat{a}|0\rangle = 0$。它是所有 Fock 态的「地基」。
**第二步，连续作用产生算符**：每作用一次 $\hat{a}^\dagger$，光子数加一。作用 $n$ 次得到 $(n!)^{-1/2}(\hat{a}^\dagger)^n|0\rangle$——分母的 $\sqrt{n!}$ 来自每步 $\sqrt{k+1}$ 系数的连乘：$\sqrt{1}\sqrt{2}\cdots\sqrt{n} = \sqrt{n!}$。
- **第三步，归一化**：因为 $\langle 0|(\hat{a})^n(\hat{a}^\dagger)^n|0\rangle = n!$，除以 $\sqrt{n!}$ 恰好把态归一。这条公式也是量子场论、凝聚态里「占据数表示」的标准写法——同样的结构会出现在第四级《量子多体理论》的二次量子化里。

## 5 光子数态的实验地位

理论干净的 Fock 态，实验上最难制备。原因正是第 3 节说的：
Fock 态的电场平均为零、相位完全不确定，普通光源（激光、
热灯）都给不出「恰好 n 个光子」的状态。目前单光子 Fock 
态 $|1\rangle$ 
已可由**单光子源**稳定产生（见本专题《光子反聚束与单光子源》），$|2\rangle$ 
及更高光子数态则需腔 QED 或参量下转换后测量。
光子数分辨探测器（transition-edge 
sensor）的出现让 $n$ 的直接读出成为可能。
这条「从理论态到实验态」的路径，
也正是量子光学从教科书走向量子技术的缩影。<span class="marginnote">光子数态是后续连续变量量子信息的基础资源之一，
与第五级《量子信息》里讨论的光子数编码直接相关。</span>

## 6 小结

- Fock 态 $|n\rangle$ 是数算符本征态，光子数**精确确定**、相位**完全不确定**。
- 产生湮灭规则：$\hat{a}|n\rangle = \sqrt{n}|n-1\rangle$，$\hat{a}^\dagger|n\rangle = \sqrt{n+1}|n+1\rangle$。
- 构造公式 $|n\rangle = (\hat{a}^\dagger)^n|0\rangle/\sqrt{n!}$