---
title: 三比特比特翻转码（bit-flip code）
date: 2026-08-07
---

# 三比特比特翻转码（bit-flip code）

<div class="epigraph">
<p>最简单的量子纠错码，藏着量子纠错的一切原理。</p>
<footer>—— 尼尔森（Michael Nielsen）与庄（Isaac Chuang）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§10.2 ｜ 2026-08-07</p>
</div>

## 为什么从比特翻转码开始

上一节立了三大障碍与解法，现在是「亲手搭码」的时刻。**三比特比特翻转码（bit-flip code）** 是量子纠错里最简单的码，它只处理一类错误（比特翻转 $X$），但它的每个部件——编码线路、综合征测量、纠错门——都是所有更复杂码的微缩模型。<span class="marginnote">类比学外语先学音标：比特翻转码把「纠缠编码 + 非破坏测量 + 条件修复」这三件事以小见大地做一遍，让你真正看懂 Shor 码、表面码的每一行。</span>本节把它从头到尾走一遍，并讲清它「能纠什么、不能纠什么」。

## 1 编码：把 1 个逻辑比特藏进 3 个物理比特

定义逻辑码字：

$$
\lvert0_L\rangle = \lvert000\rangle, \qquad \lvert1_L\rangle = \lvert111\rangle
$$

编码线路：$\lvert0\rangle\lvert0\rangle\lvert0\rangle \to$（两个 CNOT 把第一个比特「拷贝」到后两个）$\to \alpha\lvert000\rangle + \beta\lvert111\rangle$。<span class="marginnote">注意这<strong>不是</strong>复制（不违背不可克隆定理）：输入是叠加态 $\alpha\lvert0\rangle+\beta\lvert1\rangle$ 时，线性性给出 $\alpha\lvert000\rangle+\beta\lvert111\rangle$，三个物理比特是纠缠的，不各自独立。</span>编码空间的维度是 2（一个逻辑比特），嵌在 $2^3=8$ 维物理空间里，剩余 6 维是「错误空间」——错误会把态从编码空间「踢」到错误空间，综合征测量的任务就是探测这种「踢出」。

## 2 错误模型与编码空间结构

设噪声是「每比特以概率 $p$ 发生 $X$ 翻转」。编码态 $\lvert\psi_L\rangle = \alpha\lvert000\rangle+\beta\lvert111\rangle$ 可能被一位翻转：第 1 位翻转得 $\alpha\lvert100\rangle+\beta\lvert011\rangle$，等等。错误后的态落在 8 维空间里不同的 2 维子空间（「扇区」）：

| 错误 | 态（扇区） | 正交性 |
| --- | --- | --- |
| 无（$I$） | $\alpha\lvert000\rangle+\beta\lvert111\rangle$ | 与其它扇区正交 |
| 第 1 位翻转（$X_1$） | $\alpha\lvert100\rangle+\beta\lvert011\rangle$ | 与其它扇区正交 |
| 第 2 位翻转（$X_2$） | $\alpha\lvert010\rangle+\beta\lvert101\rangle$ | 与其它扇区正交 |
| 第 3 位翻转（$X_3$） | $\alpha\lvert001\rangle+\beta\lvert110\rangle$ | 与其它扇区正交 |

关键性质：**四个扇区两两正交**。这保证综合征测量能把它们无歧义地分开——测到哪个扇区，就知道错误是哪类。<span class="marginnote">这个「错误扇区互相正交」的性质正是 Knill–Laflamme 纠错条件在比特翻转码上的具体体现：不同的 Pauli 错误把编码空间映到互相正交的子空间，于是可被无歧义纠正。</span>

## 3 综合征测量：只测错误，不碰数据

综合征测量用两个校验算符 $Z_1 Z_2$ 与 $Z_2 Z_3$（它们作用在编码空间上为恒等）。测量过程用「辅助比特 + CNOT」实现：把 $Z_1Z_2$ 的本征值（$\pm1$）转移到辅助比特上，读出结果。<span class="marginnote">经典实现：$Z_i Z_j$ 测量可以通过「两个 CNOT + 辅助比特 + $H$」线路完成，读出结果 0/1 对应本征值 $+1/-1$。两个辅助比特各给一位综合征，合成两位二进制（见上一节的表）。</span>

**为什么不破坏叠加？** 因为辅助比特记录的是「$Z_1Z_2$ 是否 $=-1$」，这个信息在编码态上本来是确定的值（$+1$），只有错误才把它变成 $-1$。测量结果不携带 $\alpha$、$\beta$ 的任何信息——数据叠加态安然无恙，只是「错误状态」被读了出来。

## 4 公式解析：编码线路的逐步作用

编码线路 = $CNOT_{1\to2}\, CNOT_{1\to3}\, (H\otimes I\otimes I)$ 的一部分。展开单比特情形：

$$
\alpha\lvert0\rangle+\beta\lvert1\rangle \xrightarrow{CNOT_{1\to2}} \alpha\lvert00\rangle+\beta\lvert11\rangle \xrightarrow{CNOT_{1\to3}} \alpha\lvert000\rangle+\beta\lvert111\rangle
$$

- **第一步，第一个 CNOT**：控制位是第一个比特。控制位 0 时目标不变、控制位 1 时目标翻转，于是 $\lvert0\rangle\to\lvert00\rangle$、$\lvert1\rangle\to\lvert11\rangle$。
- **第二步，第二个 CNOT**：同样的机制把信息「广播」到第三个比特，$\lvert00\rangle\to\lvert000\rangle$、$\lvert11\rangle\to\lvert111\rangle$。
- **第三步，线性性**：叠加输入逐项映射，得到 $\alpha\lvert000\rangle+\beta\lvert111\rangle$。<span class="marginnote">这一步与贝尔态生成的 CNOT 技巧一脉相承（第四篇《贝尔态》）：CNOT 把「单比特叠加」扩展成「多比特纠缠」。编码的本质就是「制造一个特定的多比特纠缠态」来承载逻辑信息。</span>

## 5 纠错：根据综合征施门

得到两位综合征后，查表施加修复门：

| 综合征 $(s_1, s_2)$ | 推断错误 | 修复门 |
| --- | --- | --- |
| $(0,0)$ | 无 | 无 |
| $(1,0)$ | 第 1 位翻转 | $X_1$ |
| $(1,1)$ | 第 2 位翻转 | $X_2$ |
| $(0,1)$ | 第 3 位翻转 | $X_3$ |

修复门 $X_i$ 作用后，态回到 $\alpha\lvert000\rangle+\beta\lvert111\rangle$。<span class="marginnote">整个流程闭环：编码 → 噪声 → 综合征 → 解码纠错 → 还原逻辑态。比特翻转码把「保护一个逻辑比特」实现为「检测并修复单个 $X$ 错误」。真实硬件上这个过程由量子电路自动完成，不需要「知道」 $\alpha,\beta$。</span>

**辨析｜易错点：** 比特翻转码**不能**纠相位错误 $Z$。因为 $Z$ 作用在 $\lvert000\rangle$ 与 $\lvert111\rangle$ 上只改变整体相位（$Z\lvert0\rangle=\lvert0\rangle$、$Z\lvert1\rangle=-\lvert1\rangle$，$\lvert111\rangle$ 的三个相位乘起来是 $-1$），对两个码字都只是整体相位因子——在 $Z_1Z_2$、$Z_2Z_3$ 的测量下检测不到。要同时处理两类错误，需要相位翻转码（下节）或 Shor 码。

## 6 小结

- **编码**：$\lvert0_L\rangle=\lvert000\rangle$、$\lvert1_L\rangle=\lvert111\rangle$，用两个 CNOT 实现，是「制造多比特纠缠」而非复制。
- **错误扇区**：$I, X_1, X_2, X_3$ 把编码空间映到四个两两正交的 2 维子空间。
- **综合征**：测 $Z_1Z_2$、$Z_2Z_3$，只获取错误信息、不破坏叠加。
- **纠错**：按综合征表施 $X_i$ 修复。
- **局限**：只能纠单比特翻转，不能纠相位错误——那是下一节《相位翻转码》的任务。

在下一节，我们把「翻转」换成「相位」——**三比特相位翻转码（phase-flip code）**，并发现它与比特翻转码之间有一条漂亮的对称关系。
