---
title: 稳定子（stabilizer）形式体系
date: 2026-08-07
---

# 稳定子（stabilizer）形式体系

<div class="epigraph">
<p>稳定子给了我们一本描述和构造量子纠错码的词典。</p>
<footer>—— 丹尼尔 · 戈特斯曼（Daniel Gottesman）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen &amp; Chuang《量子计算与量子信息》§10.5 ｜ 2026-08-07</p>
</div>

## 为什么从稳定子形式体系开始

前面的比特翻转码、Shor 码，每个都是手工设计、逐个验证。有没有一种**统一语言**，能描述、构造、分析所有「好」的量子纠错码？答案是 **稳定子形式体系（stabilizer formalism）**。它由 Gottesman 在 1997 年系统建立，是量子纠错的「标准坐标」——表面码、Steane 码、拓扑码，全都在这个框架里。<span class="marginnote">稳定子形式体系的基本思想：<strong>用一个 Abel 群（稳定子群）来「定义」码空间</strong>——码空间是所有被这个群逐点固定的态。这个定义方式让编码、综合征、纠错全部变成群论运算，甚至可以经典模拟（Gottesman-Knill 定理）。</span>理解稳定子，就是理解现代量子纠错的全部语法。

## 1 Pauli 群与稳定子群

**Pauli 群** $\mathcal{G}_n$ 是 $n$ 比特 Pauli 算子组成的群：

$$
\mathcal{G}_n = \{\pm I, \pm iI\} \times \{I, X, Y, Z\}^{\otimes n}
$$

（乘以整体相位 $\pm1, \pm i$ 保证群封闭）。Pauli 群的关键性质：任意两个元素**要么对易、要么反对易**——$P, Q \in \mathcal{G}_n$，要么 $PQ = QP$，要么 $PQ = -QP$。<span class="marginnote">这个「非对易即反对易」的二分性是稳定子理论的基石：正是它让「测量某 Pauli」成为「探测错误」的可靠工具。$n$ 比特 Pauli 全体张满 $2^n\times2^n$ 矩阵空间，所以用 Pauli 语言就能描述任何可纠正错误（见上一节离散化）。</span>

**稳定子群** $S$ 是 $\mathcal{G}_n$ 的一个 Abel 子群（所有元素两两对易），且不含 $-I$。<span class="marginnote">要求 Abel 且不含 $-I$ 的理由：我们要找的码空间 $\mathcal{C}_S$ 是 $S$ 中所有算符的 $+1$ 公共本征空间；若 $S$ 含 $-I$，公共本征空间必为空（$-I$ 没有 $+1$ 本征态），码空间不存在。</span>

## 2 稳定子码的定义

**稳定子码（stabilizer code）**：给定 Abel 群 $S = \langle g_1, \dots, g_{n-k}\rangle$（由 $n-k$ 个独立生成元生成），码空间定义为

$$
\mathcal{C}_S = \{\lvert\psi\rangle : g_i \lvert\psi\rangle = \lvert\psi\rangle, \; \forall i\}
$$

即「同时被所有生成元逐点固定」的态。<span class="marginnote">编码参数：$n$ 个物理比特，$n-k$ 个独立稳定子生成元，码空间维数 $2^k$——所以记作 $[[n,k]]$ 码。每个生成元把空间「砍」一半，$n-k$ 个独立生成元把 $2^n$ 维砍成 $2^k$ 维。</span>每个生成元 $g_i$ 都是可测量的（因为对易），测量 $g_i$ 给出 $\pm1$，$+1$ 表示「无错误」、$-1$ 表示「该稳定子被破坏」。

以比特翻转码为例：$S = \langle Z_1 Z_2, Z_2 Z_3\rangle$，码空间 $\{\alpha\lvert000\rangle+\beta\lvert111\rangle\}$ 正是 $Z_1Z_2$、$Z_2Z_3$ 的 $+1$ 公共本征空间。<span class="marginnote">回顾上一节的综合征表：测 $Z_1Z_2$、$Z_2Z_3$ 得到 $(+1,+1)$ 表示无错，$(-1,\cdots)$ 表示某位翻转——这正是「测量稳定子生成元」的标准化表述。稳定子把 Syndrome 表翻译成「生成元本征值向量」。</span>

## 3 公式解析：$[[n,k]]$ 码的参数

为什么 $n-k$ 个生成元给出 $2^k$ 维码空间？

$$
\dim \mathcal{C}_S = \frac{2^n}{2^{n-k}} = 2^k
$$

- **第一步，每个生成元的作用**：$g_i$ 是 Pauli 群元素，非 $\pm I$，有 $\pm1$ 两个本征值，各对应维数 $2^{n-1}$ 的本征子空间。
- **第二步，公共本征空间**：所有生成元对易，可同时对角化；取「全部 $+1」的交集，每取一个生成元维数减半。
- **第三步，结果**：$n-k$ 个独立生成元把 $2^n$ 维砍到 $2^k$ 维。<span class="marginnote">这解释了记号的来由：$[[n,k,d]]$ 中 $d$ 是「距离」（最小错误权），$k$ 是逻辑比特数。稳定子码的码率是 $k/n$，距离 $d$ 由「不能被稳定子群检测的最小 Pauli 错误权」决定——$d$ 与生成元结构之间的平衡，就是码设计的全部艺术。</span>

## 4 公式解析：错误如何被检测

设错误 $E$ 作用在码空间上。$E$ 能被检测当且仅当 $E$ 与**所有**稳定子生成元要么对易（本征值不变）、要么反对易（本征值翻转，可检测）：

$$
E \in N(S) \setminus S \Rightarrow \text{可检测}; \qquad E \in S \Rightarrow \text{不改变码字（无害）}; \qquad E \notin N(S) \Rightarrow \text{不可检测}
$$

其中 $N(S)$ 是 $S$ 的正规化子（与 $S$ 中所有元素对易的 Pauli 集合）。三步拆解：

- **第一步，反对易即翻转**：若 $Eg_i = -g_i E$，则 $g_i(E\lvert\psi\rangle) = -E(g_i\lvert\psi\rangle) = -E\lvert\psi\rangle$——错误把码字变成「$g_i$ 的 $-1$ 本征态」，测量 $g_i$ 得到 $-1$，错误被发现。
- **第二步，对易即检测不到**：若 $E$ 与所有 $g_i$ 对易，则 $E\lvert\psi\rangle$ 仍是所有 $g_i$ 的 $+1$ 本征态——错误把码字留在码空间内，任何生成元测量都看不到（除非 $E$ 恰在 $S$ 内，此时它只是「逻辑操作」）。
- **第三步，距离的定义**：码的距离 $d$ = 最小权重的「不可检测错误」权重。能纠正 $t = \lfloor (d-1)/2\rfloor$ 位错误。<span class="marginnote">这套「对易/反对易 → 检测/不检测」的判据是稳定子理论的精髓：<strong>码的纠错能力完全由 Pauli 群的交换关系决定</strong>。设计码 = 挑一个 Abel 子群 $S$，让「不可检测错误」尽量「重」——这就是 CSS 码、表面码设计的代数骨架。</span>

**辨析｜易错点：** 稳定子生成元**不是**「编码算符」。生成元定义码空间、用于检测错误；逻辑算符（如 $\bar X$、$\bar Z$）是另一个集合（$N(S)\setminus S$ 里「等价类」的代表）。初学者常把「测量生成元」与「执行逻辑门」混为一谈——前者是纠错，后者是计算。

## 5 稳定子的力量：Gottesman-Knill 与经典模拟

稳定子形式体系最惊艳的推论是 **Gottesman-Knill 定理**：只含 Clifford 门（$H, S, CNOT$）与稳定子初态、稳定子测量的量子线路，可以用经典计算机**多项式时间精确模拟**。<span class="marginnote">证明思路：Clifford 门共轭地把 Pauli 映到 Pauli，所以稳定子群在 Clifford 演算下保持「Pauli 描述」，用 $O(n^2)$ 个比特就能存下整个状态，运算变成 Pauli 乘法。这是「Clifford 量子计算不超越经典」的严格证明，也是「量子优势必须靠非 Clifford 门（T 门）」的代数根源。</span>这个定理的双重意义：

- **理论**：划出了「可经典模拟」与「必须量子」的边界。
- **工程**：量子纠错的综合征模拟、解码器设计，全在这个框架里做——你不必真的在量子计算机上「试错」。

## 6 小结

- **Pauli 群** $\mathcal{G}_n$：任意两元素要么对易要么反对易；张满全部矩阵空间。
- **稳定子码**：$S = \langle g_1,\dots,g_{n-k}\rangle$ 的 $+1$ 公共本征空间，记作 $[[n,k]]$；比特翻转码、Shor 码都是它的实例。
- **参数**：$\dim\mathcal{C}_S = 2^k$；距离 $d$ = 最小不可检测错误权重，纠 $t=\lfloor(d-1)/2\rfloor$ 位错。
- **检测判据**：$E$ 与生成元反对易 ⇔ 可检测；对易 ⇔ 检测不到。
- **Gottesman-Knill**：Clifford 线路可经典模拟，量子优势必须靠非 Clifford 门。

在下一节，我们把稳定子思想落地到一类重要的构造——**CSS 码与 Steane 码**，看如何用「经典码对」直接造出量子码。
