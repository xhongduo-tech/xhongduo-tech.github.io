---
title: 时序逻辑 LTL/CTL
date: 2026-08-11
---

# 时序逻辑 LTL/CTL

<div class="epigraph">
<p>时间是最古老、也最顽固的模态词；把时间形式化，人类才算真正开始讨论「必然」与「可能」之外的第三种模态。</p>
<footer>—— 亚瑟 · 普莱尔（Arthur Prior），现代时态逻辑之父</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 形式化方法 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么在逻辑里加入时间

上一篇我们用命题逻辑与谓词逻辑描述「系统此刻的状态」。但一个系统是否**正确**，几乎从不看单帧快照，而看它在时间里的行为：「电梯永远不会在门未关时移动」「某个进程最终一定进入临界区」「两个进程永远不会同时进入临界区」。这些性质都是关于**时间演化的断言**。要做到机械地验证它们，就必须把「时间」本身形式化——这就是**时序逻辑（temporal logic）**的任务。<span class="marginnote">普莱尔（Arthur Prior）1950 年代在研究希腊哲人的时间哲学时发明了时态逻辑，把「曾经」「将要」变成模态算子。埃米尔森 (Emerson) 与哈雷尔 (Harel)、帕努利 (Pnueli) 1977 年起把它引入程序验证，Pnueli 因此在 1996 年获得图灵奖。</span>

本专题第 3 篇要讲的**模型检测**，正是拿时序逻辑公式去描述系统必须满足的性质，再自动检查系统模型是否满足它。可以说：**时序逻辑是「要验证什么」的语言，模型检测是「怎么验证」的算法。** 今天这一课把前者打牢。

## 1 系统模型：克里普克结构

在谈时序算子之前，得先给「系统随时间演化」一个数学模型。本专题采用**克里普克结构（Kripke structure）** $M = (S, S_0, R, L)$：<span class="marginnote">克里普克（Saul Kripke）1963 年在模态逻辑的语义学中提出这套结构。它是「可能世界」模型的计算机版：每个世界是一个状态，变迁关系连接相邻世界，标注函数告诉我们每个世界「哪些原子命题为真」。</span>

$S$：状态（state）的集合；
- $S_0 \subseteq S$：初始状态集合；
- $R \subseteq S \times S$：变迁关系（transition relation），要求每个状态至少有一个后继（时间不终止）；
- $L: S \to 2^{AP}$：标注函数（labeling），把每个状态映射到该状态上为真的原子命题集合。

一条**路径（path）**是 $S$ 中无穷序列 $\pi = s_0 s_1 s_2 \cdots$，满足每相邻两步在 $R$ 中。系统的性质就写在路径上。例如一台「两个进程抢一把锁」的互斥系统，原子命题可设为 $\{ \mathrm{crit}_1, \mathrm{crit}_2, \mathrm{try}_1, \dots \}$，「两进程永不同时在临界区」就是「在任意路径的任意时刻，$\mathrm{crit}_1$ 与 $\mathrm{crit}_2$ 不同时为真」——这句话马上就要被翻译成时序公式。

## 2 线性时序逻辑 LTL

**线性时序逻辑（Linear Temporal Logic, LTL）**把时间看成一条线：**每一条路径单独构成一个世界**，公式在路径上求值。LTL 的语法在命题逻辑之上加了四个时序算子：

| 算子 | 读法 | 直觉（在路径当前位置） |
| --- | --- | --- |
| $\mathrm{X}\,\phi$ | 下一个（next） | 下一时刻 $\phi$ 成立 |
| $\mathrm{F}\,\phi$ | 最终（eventually） | 未来某时刻 $\phi$ 成立 |
| $\mathrm{G}\,\phi$ | 恒久（globally） | 从现在起永远 $\phi$ 成立 |
| $\phi\,\mathrm{U}\,\psi$ | 直到（until） | $\phi$ 一直成立，直到 $\psi$ 成立 |

LTL 的语义在路径 $\pi = s_0 s_1 \cdots$ 上归纳定义，记 $\pi \models \phi$。例如：$\pi \models \mathrm{G}\,\phi$ 当且仅当**路径的每一后缀**都满足 $\phi$；$\pi \models \phi\,\mathrm{U}\,\psi$ 当且仅当存在 $i \geq 0$ 使 $s_i \models \psi$，且对所有 $j < i$ 有 $s_j \models \phi$。而「系统 $M$ 满足 $\phi$」（记 $M \models \phi$）定义为**从所有初始状态出发的所有路径**都满足 $\phi$。<span class="marginnote">$\mathrm{F}$ 与 $\mathrm{G}$ 可用 $\mathrm{U}$ 定义：$\mathrm{F}\,\phi \equiv \mathrm{T}\,\mathrm{U}\,\phi$（真直到 $\phi$），$\mathrm{G}\,\phi \equiv \neg \mathrm{F}\,\neg \phi$。所以「核心算子只有 $\mathrm{X}$ 和 $\mathrm{U}$」，其余都是派生——这与程序设计语言里「少数原语 + 语法糖」的思路同构。</span>

**辨析｜易错点：** 「$\mathrm{F}\,\phi$」只是说「未来**某个**时刻 $\phi$ 成立」，**不承诺「最终是否一定发生」之外的任何时序**；而「$\mathrm{G}\,\mathrm{F}\,\phi$」（无限经常成立）要求 $\phi$ 在路径上**出现无穷多次**，是「活锁必被打破」「某请求最终会再次被服务」这类更苛刻性质的正确表达。把「最终发生一次」和「无限经常发生」分清楚，是写 LTL 性质的第一道坎。

## 3 分支时序逻辑 CTL

LTL 站在单条路径上看世界。但很多性质是关于「**选择**」的：在某个状态，**存在**一条路径能避开故障；或**所有**路径都保证进入恢复。这需要**分支时序逻辑（Computation Tree Logic, CTL）**，它把系统的演化看成一棵以初始状态为根的**计算树**，用两个路径量词（path quantifiers）来谈论「所有路径」或「存在路径」：

$\mathrm{A}$（All）：所有从当前状态出发的路径；
- $\mathrm{E}$（Exists）：至少存在一条这样的路径。

路径量词必须与一个时序算子**成对出现**，组成八个时态组合：$\mathrm{AX}, \mathrm{EX}, \mathrm{AG}, \mathrm{EG}, \mathrm{AF}, \mathrm{EF}, \mathrm{AU}, \mathrm{EU}$。<span class="marginnote">CTL 的「A/E 必须紧跟 X/G/F/U 之一」是一条硬性语法约束。违反它（例如写 $\mathrm{AG}(p \to \mathrm{F}\,q)$，把 A 与 G 拆开、中间夹了别的东西）得到的公式叫「CTL\*」，语义更强大但模型检测算法更贵。教材 Huth §3 主要讲语法受限的 CTL。</span>

用 CTL 表达典型性质：

- $\mathrm{AG}\,\neg(\mathrm{crit}_1 \land \mathrm{crit}_2)$：**一切路径的一切时刻**，两进程不同时在临界区（互斥，安全性）。
- $\mathrm{AF}\,\mathrm{crit}_1$：每条路径最终都能让进程 1 进入临界区（响应性/活性的雏形）。
- $\mathrm{AG}\,(p \to \mathrm{AF}\,q)$：无论何时 $p$ 成立，此后 $q$ 终将成立（反应性）。

**重点：CTL 与 LTL 是两套不同的逻辑，表达力互有覆盖、互不包含。** 有些性质 LTL 能写而 CTL 不能（如 $\mathrm{GF}\,p$），有些 CTL 能写而 LTL 不能（如 $\mathrm{AG\,EF}\,p$）。工程上选择哪套，取决于性质类型与模型检测算法（CTL 有 $\mathcal{O}(|M| \cdot |\phi|)$ 的标签算法，LTL 要用自动机乘积，见第 3 篇）。

## 4 公式解析：AG 到底在断言什么

CTL 公式里最常用、也最容易被想浅的，是 $\mathrm{AG}\,\phi$。把它拆开：

$$
M \models \mathrm{AG}\,\phi
$$

- **第一步，读路径量词**：最外层没有路径量词时，CTL 语义约定为「从所有初始状态出发的所有路径」——等价于写成 $\mathrm{AG}$ 其实是「A + G」两个符号在说话。
- **第二步，读 A**：$\mathrm{A}$ 是「所有路径」。于是 $\mathrm{AG}\,\phi$ 承诺了计算树上**每一个分支**。
- **第三步，读 G**：$\mathrm{G}\,\phi$ 说「对路径上的每一个状态，$\phi$ 在该状态成立」。与 A 合起来：**从任一初始状态沿任一可执行路径走下去，任何一个到达的状态都必须满足 $\phi$**。
- **第四步，看成不动点**：$\mathrm{AG}\,\phi$ 恰是所有满足「自身状态满足 $\phi$，且所有后继也满足 AG 条件」的状态集——即集合方程 $Z = \phi \land \mathrm{EX}\,Z$ 的**最大不动点**。这一条在第 3 篇会变成算法：模型检测就是算不动点。

对互斥例子：$M \models \mathrm{AG}\,\neg(\mathrm{crit}_1 \land \mathrm{crit}_2)$ 断言「无论执行轨迹多长、无论调度如何选择，两进程同时在临界区的状态**永远不会**被到达」。安全性（safety）性质大多长成 $\mathrm{AG}$ 的样子——**它们说的是「坏状态不可达」**，靠最大不动点去检查。

## 5 小结

- **克里普克结构** $M=(S,S_0,R,L)$ 是系统的语义模型；性质写在状态标注与变迁关系之上。
- **LTL** 是线性时序逻辑：公式在单条路径上求值，核心算子是 $\mathrm{X}$ 与 $\mathrm{U}$，$\mathrm{F}, \mathrm{G}$ 是派生；「最终一次」$\mathrm{F}$ 与「无限经常」$\mathrm{GF}$ 必须分清。
- **CTL** 是分支时序逻辑：路径量词 $\mathrm{A}/\mathrm{E}$ 必须紧跟时序算子，组成 AX/EX/AG/EG/AF/EF/AU/EU 八种组合；适合表达「所有路径」或「存在路径」下的性质。
- **安全性**性质形如 $\mathrm{AG}\,\neg(\text{坏状态})$，说的是不可达；**活性**性质形如 $\mathrm{AF}\,\phi$，说的是终将发生。
- $\mathrm{AG}\,\phi$ 是不动点方程 $Z = \phi \land \mathrm{EX}\,Z$ 的最大不动点——这条桥梁把「性质」直接接上「算法」。

在下一节，我们将把时序逻辑落到工程：给定克里普克结构 $M$ 与 CTL 公式 $\phi$，**模型检测**如何自动判定 $M \models \phi$，以及为什么状态空间会指数爆炸、算法又该如何应对。
