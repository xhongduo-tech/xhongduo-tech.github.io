---
title: 极大理想与素理想
date: 2026-08-07
---

# 极大理想与素理想

<div class="epigraph">
<p>商环是域还是整环，取决于理想是极大还是素——域与整环从理想的世界里重新长出来。</p>
<footer>—— 自 题（理想课堂笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§8.5 ｜ 2026-08-07</p>
</div>

## 为什么从极大理想与素理想开始

上一节的对应定理告诉我们：商环 $R/I$ 的性质完全由理想 $I$ 决定。本节把「$R/I$ 是域吗」「$R/I$ 是整环吗」翻译成理想本身的两种属性——**极大理想（maximal ideal）**与**素理想（prime ideal）**。

- **极大理想** $M$：夹在 $M$ 与 $R$ 之间没有其他理想 ⟺ $R/M$ 是**域**；
- **素理想** $P$：$ab \in P \Rightarrow a \in P$ 或 $b \in P$ ⟺ $R/P$ 是**整环**。

这两条等价关系是环论的枢纽：它让「造域」变成「找一个极大理想」，让「有限域的构造」（第十篇）有了精确的代数配方。本节把极大理想与素理想的定义、等价定理、以及它们在 $\mathbb{Z}$ 与多项式环中的具体形态讲透。

## 1 极大理想的定义与判定

**极大理想（maximal ideal）**：设 $R$ 是含幺环，$M \ne R$ 是 $R$ 的理想。若不存在理想 $I$ 满足 $M \subsetneq I \subsetneq R$，则称 $M$ 是**极大理想**。

直观：$M$ 是「在包含序下尽可能大」的真理想——它上面没有任何中间层，再往上一步就是整个 $R$。

**判定（对应定理的推论）：** $M$ 是 $R$ 的极大理想 ⟺ $R/M$ 是**域**。

- 若 $M$ 极大，则 $R/M$ 无真非零理想（对应定理），故非零元都可逆（有限情形）……对一般情形：$R/M$ 的每个非零元素 $\bar a \ne 0$，$a \notin M$，$\langle a \rangle + M \supsetneq M$，极大性逼得 $\langle a \rangle + M = R$，故存在 $r$ 使 $ra + m = 1$，$\bar r \bar a = \bar 1$——$\bar a$ 可逆，$R/M$ 是域；
- 反之若 $R/M$ 是域，域只有 $\{0\}$ 与自身两个理想，对应定理逼出 $M$ 极大。$\blacksquare$<span class="marginnote">「极大理想 ⟺ 商环是域」是环论最实用的等价：想要一个域，找极大理想商掉即可。$\mathbb{R}[x]$ 中 $\langle x^2 + 1 \rangle$ 是极大理想（$x^2 + 1$ 不可约），故 $\mathbb{R}[x]/\langle x^2 + 1\rangle \cong \mathbb{C}$ 是域——上一节的构造在此被追认为「极大理想造域」。</span>

**例：**
- $\mathbb{Z}$ 中极大理想恰是 $\langle p \rangle$（$p$ 素数）：$p\mathbb{Z}$ 之上没有真中间理想，$\mathbb{Z}/p\mathbb{Z} = \mathbb{F}_p$ 是域；
- $\mathbb{R}[x]$ 中极大理想恰是 $\langle x - a \rangle$（$a \in \mathbb{R}$）与 $\langle x^2 + bx + c \rangle$（无实根的二次式）——它们对应的商环分别是 $\mathbb{R}$ 与 $\mathbb{C}$；
- $\mathbb{Z}$ 中 $\langle 0 \rangle$ 不是极大（$\mathbb{Z}$ 不是域），$\langle 4 \rangle$ 不是极大（$\langle 2 \rangle$ 夹在中间）。

## 2 素理想的定义与判定

**素理想（prime ideal）**：设 $R$ 是交换含幺环，$P \ne R$ 是 $R$ 的理想。若 $ab \in P$ 蕴含 $a \in P$ 或 $b \in P$，则称 $P$ 是**素理想**。

素理想的名字来自「素数」的整除性质：$p \mid ab \Rightarrow p \mid a$ 或 $p \mid b$（$p$ 素数）。素理想把这套「素性」抽象到理想上。

**判定：** $P$ 是 $R$ 的素理想 ⟺ $R/P$ 是**整环**。

- 若 $R/P$ 是整环且 $\bar a \bar b = \bar 0$，则 $\bar a = \bar 0$ 或 $\bar b = \bar 0$，即 $a \in P$ 或 $b \in P$——素性成立；
- 反之若 $P$ 是素理想，$R/P$ 无零因子（$\bar a \bar b = 0 \Rightarrow ab \in P \Rightarrow a \in P$ 或 $b \in P$）。$\blacksquare$<span class="marginnote">素理想 ⟺ 商环是整环，与「极大 ⟺ 商环是域」并排：因为域 = 整环 + 全可逆，所以<strong>极大理想必是素理想</strong>（$R/M$ 是域必是整环）。反过来素理想不必极大：$\mathbb{Z}$ 中 $\langle 0 \rangle$ 是素理想（$\mathbb{Z}$ 是整环）但非极大；$\mathbb{Z}[x]$ 中 $\langle x \rangle$ 是素理想（$\mathbb{Z}[x]/\langle x\rangle \cong \mathbb{Z}$ 是整环）但非极大。</span>

**例：**
- $\mathbb{Z}$ 中素理想：$\langle 0 \rangle$（$\mathbb{Z}$ 整环）与 $\langle p \rangle$（$p$ 素数，$\mathbb{F}_p$ 域）；$\langle 4 \rangle$ 不是素（$2 \cdot 2 \in \langle 4 \rangle$ 但 $2 \notin \langle 4 \rangle$）；
- $\mathbb{R}[x]$ 中素理想：$\langle 0 \rangle$、$\langle x - a \rangle$、$\langle f \rangle$（$f$ 不可约）；
- 商环例子：$\mathbb{Z}/4\mathbb{Z}$ 有零因子（$2 \cdot 2 = 0$），故 $\langle 4 \rangle$ 非素。

## 3 极大与素：一张对照表

把两类理想并排，关系与差异一目了然。

| 属性 | 定义 | 商环 | 例（$\mathbb{Z}$） | 例（$\mathbb{R}[x]$） |
| --- | --- | --- | --- | --- |
| 极大 | 无真中间理想 | 域 | $\langle p \rangle$ | $\langle x-a\rangle$、$\langle x^2+1\rangle$ |
| 素 | $ab \in P \Rightarrow a \in P$ 或 $b \in P$ | 整环 | $\langle 0\rangle$、$\langle p\rangle$ | $\langle 0\rangle$、$\langle f\rangle$（不可约） |

**关键关系：极大 ⟹ 素**（域 ⟹ 整环），但素 ⇏ 极大（$\langle 0 \rangle$ 素非极大）。<span class="marginnote">「极大 ⊆ 素」是包含关系：所有极大理想都是素理想，但素理想（如 $\langle 0 \rangle$、$\langle x\rangle \subseteq \mathbb{Z}[x]$）可以不是极大。有限维情形下两者会重合：$\mathbb{F}_p[x]$ 里每个非零素理想都是极大（因为 $F[x]$ 是 PID 且 $F[x]/\langle f\rangle$ 有限维），这解释了为什么有限域的分类如此干净。</span>

**定理（有限维/PID 里的重合）：** 在 PID 中，非零素理想都是极大理想。证明：$R$ 是 PID，$P = \langle p \rangle$ 非零素理想，$p$ 是素元（素理想 ⟺ 生成元素元，第九篇）；素元在 PID 里即不可约，$\langle p \rangle$ 极大（第九篇证明「不可约生成元 ⟹ 极大」）。$\blacksquare$

## 4 公式解析：R/M 是域 ⟺ M 极大

把「极大理想造域」的机制完整拆解，它是有限域构造的地基。

- **第一步，方向（⟸）：$R/M$ 是域 ⟹ $M$ 极大。** 假设 $M \subsetneq I$ 是真包含 $M$ 的理想。由对应定理，$I/M$ 是 $R/M$ 的非零理想。$R/M$ 是域，非零理想只能是整个商环，故 $I/M = R/M$，$I = R$。所以 $M$ 之上没有真中间理想，$M$ 极大。

- **第二步，方向（⟹）：$M$ 极大 ⟹ $R/M$ 是域。** 取 $\bar a \ne \bar 0$（$a \notin M$）。$\langle a \rangle + M$ 是含 $M$ 且真包含 $M$ 的理想（$a \notin M$），由极大性 $\langle a \rangle + M = R$，故存在 $r, m$ 使 $ra + m = 1$，即 $\bar r \bar a = \bar 1$。$\bar a$ 可逆，$R/M$ 是域。

- **第三步，两向合的机关。** 两个方向分别是「对应定理」与「极大性 ⟹ 生成元组合出 1」。**极大性保证「任何非 $M$ 元素都能与 $M$ 组合出单位元」——这正是「非零元可逆」的环论翻译。**

- **第四步，应用。** 造域的标准配方：找一个极大理想 $M$，商环 $R/M$ 就是域。$F[x]$ 中 $M = \langle f \rangle$（$f$ 不可约）是极大，$F[x]/\langle f\rangle$ 是域——**第十篇有限域 $\mathbb{F}_{p^n}$ 的全部构造都从这里出发。**

## 5 例子：多项式环中的极大理想与素理想

在多项式环里，极大/素理想与「不可约」紧密相连，是第九篇的前奏。

**$\mathbb{R}[x]$ 的极大理想**：$\langle x - a \rangle$（$a \in \mathbb{R}$）——商环 $\mathbb{R}[x]/\langle x-a\rangle \cong \mathbb{R}$ 是域；$\langle x^2 + 1 \rangle$——商环 $\cong \mathbb{C}$ 是域。**$\mathbb{R}[x]$ 的极大理想 = 不可约多项式生成的主理想。**

**$\mathbb{R}[x]$ 的素理想**：$\langle 0 \rangle$（整环）、$\langle x - a \rangle$、$\langle x^2 + 1 \rangle$、$\langle x^2 + x + 1 \rangle$ 等不可约生成者。

**非极大素理想的例子**：$\mathbb{Z}[x]$ 中 $\langle x \rangle$ 是素理想（$\mathbb{Z}[x]/\langle x\rangle \cong \mathbb{Z}$ 是整环）但非极大（$\mathbb{Z}$ 不是域）——**「素非极大」在多项式环里俯拾即是**。<span class="marginnote">$\mathbb{Z}[x]/\langle x\rangle \cong \mathbb{Z}$ 说明「素非极大」的典型来源：$\langle x\rangle$ 的商环是整环 $\mathbb{Z}$，但 $\mathbb{Z}$ 不是域，所以 $\langle x\rangle$ 不是极大。更一般地，$k[x_1,\dots,x_n]$ 中 $\langle x_1, \dots, x_m\rangle$（$m < n$）是素非极大——素理想的「几何」是代数簇的子簇，极大理想对应点，这正是代数几何的起点。</span>

## 6 小结

- **极大理想**：无真中间理想；⟺ 商环是域。$\mathbb{Z}$ 中极大理想 = $\langle p \rangle$。
- **素理想**：$ab \in P \Rightarrow a \in P$ 或 $b \in P$；⟺ 商环是整环。$\mathbb{Z}$ 中素理想 = $\langle 0\rangle$、$\langle p\rangle$。
- **极大 ⟹ 素**，但素 ⇏ 极大；PID 中非零素理想都是极大。
- **造域配方**：$F[x]/\langle f\rangle$ 在 $f$ 不可约时是域（$f$ 生成极大理想）。
- 素理想与不可约多项式（第九篇）相连；极大/素是代数几何「点/簇」的代数雏形。

在下一节，我们收割极大/素理想的一个应用：**中国剩余定理（Chinese Remainder Theorem）**。互素的理想之和为全环，商环随之分裂成直积。
