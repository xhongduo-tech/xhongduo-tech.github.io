---
title: GHZ 态与 W 态：多体纠缠的两种类型
date: 2026-08-07
---

# GHZ 态与 W 态：多体纠缠的两种类型

<div class="epigraph">
<p>三个粒子的纠缠所呈现出的现象，是两粒子纠缠中完全看不到的。</p>
<footer>—— 格林伯格（Daniel Greenberger）、霍恩（Michael Horne）与蔡林格（Anton Zeilinger）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§2.5.3 ｜ 2026-08-07</p>
</div>

## 为什么从多体纠缠开始

前面的纠缠度量都是**两体**的：两比特分成 A、B，看它们纠缠得多深。但真实世界（和真实量子设备）有三个以上粒子：量子纠错的稳定子态、量子计算的初态、多量子比特实验，动辄几十上百个比特。多体纠缠的世界比两体丰富得多——它不再是「一根尺子量到底」，而是分出**不同类型**的纠缠结构。最著名的一对代表就是 **GHZ 态**与 **W 态**。<span class="marginnote">GHZ 由 Greenberger、Horne、Zeilinger 于 1989 年提出；W 态是 Dür、Vidal、Cirac 在 2000 年分类多体纠缠时发现的。两者同为「最大纠缠」，却互不可转化——用 LOCC 谁也变不成谁。</span>理解它们，是理解量子纠错稳定子态、以及「多体纠缠的分类」的入口。

## 1 GHZ 态：全或无的相干

三比特 **GHZ 态**定义为

$$
\lvert GHZ\rangle = \frac{1}{\sqrt2}(\lvert000\rangle + \lvert111\rangle)
$$

它直观上是「三比特版本的 $\lvert\Phi^+\rangle$」：要么全部为 0、要么全部为 1 的相干叠加。<span class="marginnote">一般 $n$ 比特 GHZ 态是 $\frac{1}{\sqrt2}(\lvert0\cdots0\rangle + \lvert1\cdots1\rangle)$。它的特征：任意丢掉一个比特，剩下的两比特就变成完全混合态——纠缠「藏在整个系统里」，任何一个子部分都看不到。</span>最惊人的性质：**对 GHZ 态，单比特测量后的关联可以用「确定性」表述**。若每个比特都沿 $X$ 方向测量，三个结果之积 $\langle XXX\rangle = -1$；若沿 $X,Y,Y$ 等混合方向，也有确定关联——这类**多体关联不等式**（GHZ 矛盾）比 CHSH 更尖锐：它无需统计违背，一个单次测量就与定域实在论冲突。

## 2 W 态：单个激发的分布

三比特 **W 态**定义为

$$
\lvert W\rangle = \frac{1}{\sqrt3}(\lvert100\rangle + \lvert010\rangle + \lvert001\rangle)
$$

它直观上是「一个 1 均匀分布在三个比特上」：三个比特中恰有一个为 1 的所有情形的等幅叠加。<span class="marginnote">一般 $n$ 比特 W 态是 $\frac{1}{\sqrt n}\sum_k \lvert\text{第 }k\text{ 位为 1}\rangle$。与 GHZ 相反，W 态丢掉一个比特后，剩下两比特<strong>仍然纠缠</strong>——纠缠「分布得更散」，不那么集中。</span>W 态在噪声下更鲁棒：部分退相干后仍保有纠缠，而 GHZ 态在任一部分丢失后立即完全退化为混合态。

## 3 GHZ 与 W：两种不可互化的纠缠

Dür–Vidal–Cirac（2000）证明了一个深刻的分类学结论：**GHZ 态与 W 态在 LOCC（局域操作与经典通信）下不可互相转化。** 也就是说，不存在任何「各自局域操作 + 经典交流」的方案能把 GHZ 变成 W（或反之）。<span class="marginnote">LOCC 是量子信息里「自由操作」的标准：每方只能在自己的比特上操作、互相通电话，不能做跨体的量子操作。在这个框架下等价的两个态被视为「同一种纠缠」；GHZ 与 W 分属不同等价类，是多体纠缠分类的第一道分水岭。</span>这一点与两体形成鲜明对比——两体纯态在 LOCC 下「纠缠一样多就等价」，而三体纯态有**无穷多**个 LOCC 不等价类。

**辨析｜易错点：** GHZ 态与 W 态**都是**最大纠缠吗？这取决于「最大」的定义。若用「两体约化熵」衡量，两者子系统约化熵不同：GHZ 的任意单比特约化态是 $I/2$（熵 1，最大混合），W 的单比特约化态是 $\frac23\lvert0\rangle\langle0\rvert+\frac13\lvert1\rangle\langle1\rvert$（熵 $< 1$）。但从 LOCC 角度看，两者都「无法从更弱纠缠的态制备出来」，都算「不可再分解」的最大纠缠——只是**类型不同**。

## 4 公式解析：GHZ 的单比特约化态

计算 $\lvert GHZ\rangle$ 对某个比特的部分迹，验证「纠缠藏在整个系统里」：

$$
\rho_1 = \operatorname{tr}_{23}(\lvert GHZ\rangle\langle GHZ\rvert) = \frac{1}{2}\lvert0\rangle\langle0\rvert + \frac{1}{2}\lvert1\rangle\langle1\rvert = \frac{I}{2}
$$

- **第一步，展开**：$\lvert GHZ\rangle\langle GHZ\rvert = \frac12(\lvert000\rangle\langle000\rvert + \lvert000\rangle\langle111\rvert + \lvert111\rangle\langle000\rvert + \lvert111\rangle\langle111\rvert)$。
- **第二步，对 2、3 取迹**：交叉项 $\lvert000\rangle\langle111\rvert$ 里 2、3 的指标对不上（$00$ vs $11$），对它们取迹得 0，只剩两个对角项。
- **第三步，归一**：$\rho_1 = \frac12(\lvert0\rangle\langle0\rvert + \lvert1\rangle\langle1\rvert)$，即 $I/2$。<span class="marginnote">对比 W 态：$\operatorname{tr}_{23}(\lvert W\rangle\langle W\rvert) = \frac23\lvert0\rangle\langle0\rvert + \frac13\lvert1\rangle\langle1\rvert \neq I/2$。这个差别可测：在 $Z$ 基下 GHZ 单比特测量结果完全随机（各 $\tfrac12$），W 态则是 0 的概率 $\frac23$、1 的概率 $\frac13$。</span>

## 5 多体纠缠为什么难度量

两体有并发度、纠缠熵；多体却没有统一标尺。原因：**多体纠缠的结构远比「一个数」丰富**。有的纠缠存在于「任意两部分之间」（W 类），有的存在于「整体相干」（GHZ 类），还有各种更复杂的分层结构。<span class="marginnote">量子信息里多体纠缠的度量仍是开放问题：有纠缠目击者（entanglement witness）、有 $\pi$-concurrence、有几何度量，但没有任何一个量能唯一刻画「多体纠缠」。这个问题与量子计算的硬件命运直接相关——纠错码的稳定子态就是特定的多体纠缠态。</span>好在量子计算最常用的多体纠缠态是**稳定子态**（stabilizer states），它们由 Pauli 群的代数结构控制，比一般多体态「规整」得多——第八篇《稳定子形式体系》会给出它们的完整刻画。

## 6 小结

- **GHZ 态** $\frac{1}{\sqrt2}(\lvert000\rangle+\lvert111\rangle)$：全或无相干，丢掉任一比特即完全失去纠缠，具有确定性的多体关联矛盾。
- **W 态** $\frac{1}{\sqrt3}(\lvert100\rangle+\lvert010\rangle+\lvert001\rangle)$：单激发均匀分布，鲁棒性更强，丢比特后其余仍纠缠。
- **GHZ 与 W 在 LOCC 下不可互化**，属于不同的纠缠等价类——多体纠缠分类的第一道分水岭。
- 多体纠缠**没有统一标尺**；但稳定子态（量子纠错的主角）有完整的代数刻画。

在下一节，我们进入**第五篇 量子算法基础**，从最抽象的复杂度模型讲起：**量子查询复杂度与黑盒模型**。
