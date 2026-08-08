---
title: 群的外直积与内直积
date: 2026-08-07
---

# 群的外直积与内直积

<div class="epigraph">
<p>把小群装配成大群，是群论的加法；把大群拆回小群，是群论的减法。</p>
<footer>—— 自 题（直积课堂笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§6.1 ｜ 2026-08-07</p>
</div>

## 为什么从直积开始

前五篇我们把一个群拆开看（子群、陪集、商群），第六篇换一个方向：**把小群「乘」成大群。** 这个乘法操作叫**直积（direct product）**。它分两种面貌——从外面搭积木叫**外直积**，从里面认结构叫**内直积**——但本质是同一件事的两端。

直积是有限生成阿贝尔群基本定理的核心工具：**任何有限生成阿贝尔群都是循环群的直积。** 理解直积，就理解了「大群 = 小群的组合」如何运作；也理解了什么情况下一个大群**真的可以**拆成两个小群的乘积（内直积），什么情况下不能。本节先把外直积与内直积的定义、判定与互化彻底弄清。

## 1 外直积：从外部搭积木

**外直积（external direct product）**：设 $G_1, \dots, G_k$ 是群，它们的（外）直积定义为集合

$$
G_1 \times \cdots \times G_k = \{ (g_1, \dots, g_k) \mid g_i \in G_i \}
$$

配上逐分量运算：

$$
(g_1, \dots, g_k)(h_1, \dots, h_k) = (g_1h_1, \dots, g_kh_k)
$$

**定理：** $G_1 \times \cdots \times G_k$ 构成群，单位元 $(e_1, \dots, e_k)$，逆元 $(g_1,\dots,g_k)^{-1} = (g_1^{-1}, \dots, g_k^{-1})$。运算按分量独立进行——各分量互不干扰，这正是「直积」的直觉：**每个分量自由地活在自己的群里，互不牵连。**<span class="marginnote">直积的阶：$|G_1 \times \cdots \times G_k| = |G_1| \cdots |G_k|$（有限时），这是「乘法」这个名字的由来——群的大小真的相乘了。$\mathbb{Z}_2 \times \mathbb{Z}_2$ 有 4 个元素（克莱因四元群 $V_4$），$\mathbb{Z}_2 \times \mathbb{Z}_3$ 有 6 个元素且 $\cong \mathbb{Z}_6$。</span>

**例：**
$\mathbb{Z}_2 \times \mathbb{Z}_2 = V_4$（克莱因四元群），每个非单位元阶为 2；
$\mathbb{Z}_2 \times \mathbb{Z}_3 \cong \mathbb{Z}_6$（因为 $\gcd(2,3) = 1$，下一节定理）；
$\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$（向量空间的加法群）；
$GL_n(\mathbb{R}) \times GL_m(\mathbb{R})$：分块对角矩阵群。

## 2 内直积：从内部认出「乘积结构」

**内直积（internal direct product）**：设 $N_1, \dots, N_k$ 是 $G$ 的正规子群。若

1. $G = N_1 N_2 \cdots N_k$（每个 $g \in G$ 都能写成 $g = n_1 n_2 \cdots n_k$，$n_i \in N_i$）；
2. 且这种写法唯一（$n_1\cdots n_k = n_1'\cdots n_k'$ 推出 $n_i = n_i'$ 对一切 $i$），

则称 $G$ 是 $N_1, \dots, N_k$ 的**内直积**，记作 $G = N_1 \times N_2 \times \cdots \times N_k$（内）。

**内直积的判定**：对两个正规子群 $N_1, N_2 \trianglelefteq G$，$G$ 是 $N_1 \times N_2$ 的内直积 ⟺ ① $G = N_1 N_2$；② $N_1 \cap N_2 = \{e\}$。<span class="marginnote">「交为 $\{e\}$」正是「写法唯一」的等价条件：若 $n_1n_2 = n_1'n_2'$，则 $n_1^{-1}n_1' = n_2 n_2'^{-1} \in N_1 \cap N_2 = \{e\}$，于是 $n_1 = n_1'$、$n_2 = n_2'$。当 $N_1 \cap N_2 = \{e\}$ 时，$N_1$ 与 $N_2$ 的元素还自动互相交换（$n_1n_2 = n_2n_1$，因为 $n_1n_2n_1^{-1}n_2^{-1} \in N_1 \cap N_2$）——内直积的两个因子「交叉换位」。</span>

**例（分解 vs 不分解）：**
- $S_3$ **不能**写成内直积 $A_3 \times \langle (12)\rangle$：$A_3$ 正规但 $\langle (12)\rangle$ 不正规（内直积要求两个因子都正规）。
- $V_4 = \mathbb{Z}_2 \times \mathbb{Z}_2$（内）：三个 2 阶正规子群任取两个都是内直积分解。
- $S_3 \cong \mathbb{Z}_3 \rtimes \mathbb{Z}_2$ 是**半直积**而非直积：$\mathbb{Z}_2$ 部分不正交（不正规），它的作用不是「自由分量」而是「旋转置换」。

**内直积与外直积的关系**：**内直积本质上是外直积的「内部实现」。** 若 $G$ 是 $N_1, \dots, N_k$ 的内直积，则映射

$$
N_1 \times \cdots \times N_k \longrightarrow G, \qquad (n_1, \dots, n_k) \mapsto n_1 \cdots n_k
$$

是同构（外直积 $\cong$ 内直积）。反之，外直积 $G_1 \times \cdots \times G_k$ 是子群 $\tilde{G}_i = \{ (e, \dots, g_i, \dots, e) \}$ 的内直积。**外与内是同一个结构的两副面孔**。

## 3 直积与交换群：Z_m × Z_n 何时等于 Z_mn

直积最常用的交换群事实是「互素时直积可合并为循环群」。

**定理：** 若 $\gcd(m, n) = 1$，则

$$
\mathbb{Z}_m \times \mathbb{Z}_n \ \cong \ \mathbb{Z}_{mn}
$$

**证明：** 考虑 $\mathbb{Z}_m \times \mathbb{Z}_n$ 中元素 $(\bar{1}, \bar{1})$ 的阶。$k(\bar 1, \bar 1) = (\bar k, \bar k) = (\bar 0, \bar 0)$ 当且仅当 $m \mid k$ 且 $n \mid k$，当且仅当 $\mathrm{lcm}(m,n) = mn \mid k$（互素时 lcm = 乘积）。故 $o((\bar 1, \bar 1)) = mn = |\mathbb{Z}_m \times \mathbb{Z}_n|$，该元素生成整个群，$G$ 是 $mn$ 阶循环群 $\cong \mathbb{Z}_{mn}$。$\blacksquare$<span class="marginnote">「互素 ⟹ 直积成循环」是中国剩余定理的群论形态。反过来，若 $\gcd(m,n) > 1$，则 $\mathbb{Z}_m \times \mathbb{Z}_n$ 不是循环群（如 $\mathbb{Z}_2 \times \mathbb{Z}_2$ 非循环）。「拆循环群为直积」与「中国剩余定理拆分同余」是同一枚硬币——第八篇我们会用环论语言重温。</span>

**例：** $\mathbb{Z}_6 \cong \mathbb{Z}_2 \times \mathbb{Z}_3$；$\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$（$4, 3$ 互素），但不 $\cong \mathbb{Z}_2 \times \mathbb{Z}_6$（后者阶 12 但非循环）。**同一个阶的交换群可以有不同分解方式，这正是有限阿贝尔群分类要解决的问题。**

## 4 公式解析：内直积判定 N1 ∩ N2 = {e}

把「内直积」最常用的判定条件拆开。

- **第一步，条件是什么。** 对 $N_1, N_2 \trianglelefteq G$，$G$ 是内直积 $N_1 \times N_2$ 当且仅当：① $G = N_1N_2$；② $N_1 \cap N_2 = \{e\}$。

- **第二步，为什么要两个条件。** ① 保证「覆盖」：每个元素都能分解；② 保证「唯一」：分解没有歧义。覆盖 + 唯一 = 直积分解。缺一不可：$V_4$ 中 $\mathbb{Z}_2 \times \mathbb{Z}_2$ 的三个 2 阶子群两两满足①且②；而 $S_3$ 中 $A_3$ 与 $\langle(12)\rangle$ 满足①（$A_3\langle(12)\rangle = S_3$）但不满足②的「正规」前提（$\langle(12)\rangle$ 不正规）。

- **第三步，为什么交为 $\{e\}$ 蕴含交换。** 若 $n_1 \in N_1$、$n_2 \in N_2$，则换位子 $n_1n_2n_1^{-1}n_2^{-1} = (n_1n_2n_1^{-1})n_2^{-1}$。由 $N_2$ 正规，$n_1n_2n_1^{-1} \in N_2$，整体 $\in N_2$；又由 $N_1$ 正规，整体也 $\in N_1$，故换位子 $\in N_1 \cap N_2 = \{e\}$，$n_1n_2 = n_2n_1$。**直积因子之间互相交换**——这就是为什么直积如此好驾驭。

- **第四步，阶的验证。** $|N_1 \times N_2| = \frac{|N_1||N_2|}{|N_1 \cap N_2|}$（一般子群乘积公式），交为 $\{e\}$ 时恰为 $|N_1||N_2|$。若 $|N_1||N_2| = |G|$ 且 $N_1, N_2$ 正规，条件②自动给条件①——**计数有时可以省掉覆盖验证**。

## 5 例子：把有限阿贝尔群「拆开」

直积的实战是「拆交换群」。用「互素合并」的逆操作，把循环群拆成素数幂循环群的直积。

**例：** $\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$（拆成 $2^2$ 与 $3$ 的部分）；$\mathbb{Z}_{36} \cong \mathbb{Z}_4 \times \mathbb{Z}_9$。一般地，对 $n = p_1^{e_1} \cdots p_k^{e_k}$，**中国剩余定理**给出

$$
\mathbb{Z}_n \cong \mathbb{Z}_{p_1^{e_1}} \times \cdots \times \mathbb{Z}_{p_k^{e_k}}
$$

「循环群拆成素数幂分量的直积」是有限阿贝尔群分解的基础——任何有限阿贝尔群先拆成素数幂部分，再对每个素数幂部分做进一步分解（第六篇接下来的两节）。

**例：** 有限阿贝尔群 $\mathbb{Z}_{12} \times \mathbb{Z}_{18}$ 先各自拆：$\mathbb{Z}_{12} \cong \mathbb{Z}_4 \times \mathbb{Z}_3$，$\mathbb{Z}_{18} \cong \mathbb{Z}_2 \times \mathbb{Z}_9$，于是

$$
\mathbb{Z}_{12} \times \mathbb{Z}_{18} \cong \mathbb{Z}_4 \times \mathbb{Z}_9 \times \mathbb{Z}_2 \times \mathbb{Z}_3 \cong \mathbb{Z}_4 \times \mathbb{Z}_9 \times \mathbb{Z}_6
$$

其中 $2, 3$ 合并回 $\mathbb{Z}_6$。这套「拆素因子 → 重新拼装」的代数，是第六篇「不变量分解」的日常操作。<span class="marginnote">「$\mathbb{Z}_4 \times \mathbb{Z}_9 \times \mathbb{Z}_6$」与「$\mathbb{Z}_4 \times \mathbb{Z}_3 \times \mathbb{Z}_2 \times \mathbb{Z}_3$」是同一个群，但前一种写法（按素数分别列出最大幂）是「规范形」——下一节有限阿贝尔群基本定理会说明为什么规范形存在且唯一。</span>

## 6 小结

- **外直积**：逐分量运算的集合乘积 $G_1 \times \cdots \times G_k$，阶相乘。
- **内直积**：正规子群 $N_i$ 满足「覆盖 + 唯一」；两个因子时等价于「$G = N_1N_2$ 且 $N_1 \cap N_2 = \{e\}$」。
- **外内一致**：内直积 = 外直积的内部实现，映射 $(n_i) \mapsto n_1\cdots n_k$ 是同构。
- **互素合并**：$\gcd(m,n) = 1 \Rightarrow \mathbb{Z}_m \times \mathbb{Z}_n \cong \mathbb{Z}_{mn}$；中国剩余定理是它的推广。
- 直积因子互相交换；内直积要求因子皆正规（半直积只要求其一正规）。

在下一节，我们收获直积理论的王冠：**有限生成阿贝尔群基本定理**。它将宣布：每个有限生成阿贝尔群都是循环群的直积，且分解方式唯一。
