---
title: 集合的基本运算
date: 2026-08-05
---

# 集合的基本运算

<div class="epigraph">
<p>集合的并、交、补，是数学里最古老也最年轻的三种运算——它们与逻辑的「或、且、非」一一对应。</p>
<footer>—— 论布尔代数（George Boole）的遗产（意译）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第一册 §1.3 ｜ 2026-08-05</p>
</div>

## 为什么要给集合做运算

实数有加减乘除，集合也需要自己的运算。上一节我们回答了「集合之间如何比较大小」，这一节要回答「集合之间如何组合、如何取补」。**并、交、补**这三种运算构成了集合的代数骨架——它们与逻辑里的「或、且、非」完全同构，也正是布尔代数、数理逻辑、数据库查询、电路设计共同的地基。<span class="marginnote">布尔（1815—1864）把逻辑推理化成代数演算，写下了《思维规律的研究》。他那套「真/假 1/0、与或非」的代数，一百年后变成了计算机的开关电路。今天我们把集合运算与逻辑运算对照着学，等于同时预习了第三级《数字逻辑》与第一级《逻辑学》。</span>

## 1 并集

**定义**：由所有属于集合 $A$ **或**属于集合 $B$ 的元素组成的集合，称为 $A$ 与 $B$ 的**并集（union）**，记作 $A \cup B$。

$$
A \cup B = \{x \mid x \in A \ \text{或}\ x \in B\}
$$

关键词是**「或」**——元素只要来自 $A$、来自 $B$、或两者兼有，统统收进来。并集的直观意义是「把两个集合合并，去掉重复」。<span class="marginnote">注意这里的「或」是「可兼或」（inclusive or）：两者都满足也算。中文的「或」有时也含「只能二选一」的意思（可兼或的对立「异或」），但集合论里的 $\cup$ 永远是可兼的。逻辑学里 $\vee$ 与此同义，见《逻辑学》命题联结词一节。</span>

**例**：$A = \{1, 2, 3\}$，$B = \{3, 4, 5\}$，则 $A \cup B = \{1, 2, 3, 4, 5\}$。

**基本性质**（可自证）：

$$
A \cup A = A, \qquad A \cup \emptyset = A, \qquad A \subseteq A \cup B, \qquad B \subseteq A \cup B
$$

## 2 交集

**定义**：由所有属于集合 $A$ **且**属于集合 $B$ 的元素组成的集合，称为 $A$ 与 $B$ 的**交集（intersection）**，记作 $A \cap B$。

$$
A \cap B = \{x \mid x \in A \ \text{且}\ x \in B\}
$$

关键词是**「且」**——元素必须同时属于 $A$ 和 $B$。若 $A \cap B = \emptyset$，称 $A$ 与 $B$ **不相交（disjoint）**。<span class="marginnote">「不相交」是一个常用术语：两个集合没有公共元素。注意「不相交」不是说「没有关系」，而恰恰说明它们可以被并成一个更大的集合——全集被无重叠地切分成若干块的「划分」概念由此萌芽。</span>

**例**：$A = \{1, 2, 3\}$，$B = \{3, 4, 5\}$，则 $A \cap B = \{3\}$。

**基本性质**：

$$
A \cap A = A, \qquad A \cap \emptyset = \emptyset, \qquad A \cap B \subseteq A, \qquad A \cap B \subseteq B
$$

## 3 补集

「补集」需要一个**讨论范围**。当我们研究的问题限定在某个集合 $U$ 内时，$U$ 称为**全集（universal set）**。

**定义**：设 $U$ 为全集，$A \subseteq U$，由 $U$ 中**不属于 $A$** 的所有元素组成的集合，称为 $A$ 相对于 $U$ 的**补集（complement）**，记作 $\complement_U A$。

$$
\complement_U A = \{x \mid x \in U \ \text{且}\ x \notin A\}
$$

当全集 $U$ 在上下文里明确时，常简写为 $\complement A$ 或 $\overline{A}$。<span class="marginnote">「补集」必须有全集作参照：同样是「非正数」的集合，在全集 $\mathbb{R}$ 里补集是 $(-\infty, 0)$，在全集 $\mathbb{Z}$ 里补集就只剩负整数了。写 $\overline{A}$ 时务必先确认讨论的全集是谁——这是考试里最常挖坑的地方。</span>

**例**：全集 $U = \{1, 2, 3, 4, 5\}$，$A = \{1, 2\}$，则 $\complement_U A = \{3, 4, 5\}$。

**基本性质**（即「双重否定」与「排除中律」的集合版）：

$$
\complement_U (\complement_U A) = A, \qquad A \cup \complement_U A = U, \qquad A \cap \complement_U A = \emptyset
$$

## 4 运算律与德摩根定律

并、交、补三者在代数结构上服从一整套漂亮的**运算律**，用表格收拢在一起：

| 名称 | 并的版本 | 交的版本 |
| --- | --- | --- |
| 交换律 | $A \cup B = B \cup A$ | $A \cap B = B \cap A$ |
| 结合律 | $A \cup (B \cup C) = (A \cup B) \cup C$ | $A \cap (B \cap C) = (A \cap B) \cap C$ |
| 分配律 | $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$ | $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ |
| 吸收律 | $A \cup (A \cap B) = A$ | $A \cap (A \cup B) = A$ |

其中最重要的是**德摩根定律（De Morgan's laws）**——它把「取补」与「并、交」联系起来，是唯一一条同时涉及全部三种运算的法则：

$$
\overline{A \cup B} = \overline{A} \cap \overline{B}, \qquad \overline{A \cap B} = \overline{A} \cup \overline{B}
$$

用语言说：**「既不是 A 也不是 B」等于「既非 A 又非 B」；「不同时是 A 和 B」等于「不是 A 或不是 B」**。<span class="marginnote">德摩根定律在逻辑里长这样：$\neg(p \vee q) = \neg p \wedge \neg q$、$\neg(p \wedge q) = \neg p \vee \neg q$——把集合换成命题，「或/且」换成 $\vee/\wedge$，一字不差。它还是「把否定号穿过括号时翻转联结词」的口诀来源，在《逻辑学》与后续所有证明里都会被反复调用。</span>

**证明思路**（以第一条为例）：要证两个集合相等，用上一节的**双向包含**——先证 $\overline{A \cup B} \subseteq \overline{A} \cap \overline{B}$：任取 $x \in \overline{A \cup B}$，则 $x \notin A \cup B$，即 $x \notin A$ 且 $x \notin B$，故 $x \in \overline{A}$ 且 $x \in \overline{B}$，即 $x \in \overline{A} \cap \overline{B}$；反向同理。

## 5 容斥原理：有限集的计数

并、交、补与计数的结合，产出了一条威力巨大的公式——**容斥原理（inclusion–exclusion principle）**。对两个有限集合：

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

符号 $|S|$ 表示集合 $S$ 中元素的个数（**基数**）。**公式解析**：先数 $A$ 再数 $B$，公共部分 $A \cap B$ 被数了两次，所以要减掉一次。对三个集合，需要更精细的补补丁：

$$
|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|
$$

「加多、减多、再加回」——依次是单个、两两交集、三三交集，奇加偶减，层层抵扣。<span class="marginnote">容斥原理是「文氏图数数」的公式化，它和概率里的加法公式 $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ 结构完全一致——概率论里那套公式的集合版本就是它。第二级《概率论与数理统计》还会给出它的严格推导。</span>

**例**：全班 45 人，其中 30 人喜欢数学，28 人喜欢物理，既喜欢数学又喜欢物理的 22 人。问至少喜欢一科的有几人？

设 $A$ 为喜欢数学者的集合，$B$ 为喜欢物理者的集合，则 $|A|=30, |B|=28, |A\cap B|=22$，代入公式：

$$
|A \cup B| = 30 + 28 - 22 = 36
$$

所以至少喜欢一科的有 36 人，剩下的 $45 - 36 = 9$ 人两科都不喜欢。<span class="marginnote">最后一步用到了补集：$|U \setminus (A \cup B)| = |U| - |A \cup B|$。这提醒我们，集合题的常规套路就是「先算并集，再补一次集」。把「至少」「至多」「都不」这类词先翻译成集合语言，题目就成功了一半。</span>

## 6 集合运算与逻辑运算、程序语言

把三种运算与日常判断对齐，是理解它们的捷径：

| 集合 | 逻辑 | 语言 | 程序/数据库 |
| --- | --- | --- | --- |
| $A \cup B$ | 「或」$p \vee q$ | 「…或…」 | 位或 `\|`、SQL `UNION` |
| $A \cap B$ | 「且」$p \wedge q$ | 「…且…」 | 位与 `&`、SQL `INTERSECT`、`WHERE` 多条件 |
| $\complement A$ | 「非」$\neg p$ | 「不…」 | 位非 `~`、SQL `NOT IN` |

这套对照不是巧合，而是同一套结构（布尔代数）的三个化身。<span class="marginnote">SQL 里 `UNION` 去重、`UNION ALL` 不去重，对应集合的并集与多重集；Python 的 `set` 直接提供 `|`、`&`、`-` 运算符，与本节公式一一对应。等你学到第三级《数据库》，会惊讶地发现查询优化器正是在做「集合运算」的等价改写——包括德摩根定律的布尔形式。</span>

## 7 小结

- **并集** $A \cup B$：元素属于 $A$ **或** $B$；**交集** $A \cap B$：元素属于 $A$ **且** $B$；**补集** $\complement_U A$：$U$ 中去掉 $A$。
- **运算律**：交换、结合、分配、吸收；**德摩根定律** $\overline{A \cup B} = \overline{A} \cap \overline{B}$ 沟通全部三种运算。
- **容斥原理**：$|A \cup B| = |A| + |B| - |A \cap B|$，三集合时奇加偶减。
- 集合运算与**逻辑「或且非」**、**程序位运算**同构，是布尔代数的一条主线。

集合的语法已经齐了——概念、关系、运算。下一节开始，我们要用集合去理解另一件事：**命题之间的逻辑关系**，也就是**充分条件与必要条件**。
