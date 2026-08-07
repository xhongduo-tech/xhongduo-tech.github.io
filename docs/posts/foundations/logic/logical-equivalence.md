---
title: 逻辑等值与常见的等值式
date: 2026-08-07
---

# 逻辑等值与常见的等值式

<div class="epigraph">
<p>两个命题只有在所有情况下同真同假，才能说它们同一。</p>
<footer>—— 逻辑等价的一种通行表述</footer>
</div>

<div class="article-byline">
<p>第一级 · 逻辑学 ｜ 陈波《逻辑学导论》第2章 §2.5 ｜ 2026-08-07</p>
</div>

## 为什么从逻辑等值开始

在《真值表》一课，我们见到 $(\neg P \lor Q)$ 与 $(P \to Q)$ 在每一行取值都相同。
这不是巧合，而是一类重要关系的标本：**逻辑等值（logical equivalence）**。
两个命题逻辑等值，意味着它们说的是同一件事、在任何语境下可以互相替换而不改变真值。
本课把「长得一样的真值表」变成一套代数工具——德摩根律、分配律、双重否定等——这些等值式是化简命题、改写条件式、设计电路的通用规则，也是下一课讨论「实质蕴涵怪论」的语言基础。

## 1 定义与记号

**逻辑等值（logical equivalence）**：公式 $A$ 与 $B$ 逻辑等值，当且仅当 $A \leftrightarrow B$ 是重言式，记作 $A \equiv B$，读作「$A$ 等值于 $B$」。

检验等值的程序与检验有效论证完全同构：展开 $A$ 与 $B$ 的真值表，看两列是否逐行相同。完全相同即等值，任何一行不同即不等值。<span class="marginnote">注意 $\equiv$ 是<strong>元语言</strong>记号——它不在命题逻辑内部，而是描述「两式在所有赋值下同真同假」这个外部事实。命题内部的「等值」联结词是 $\leftrightarrow$，两者层级不同。</span>

**辨析｜易错点：** $A \equiv B$ 与 $A \leftrightarrow B$ 的区别要分清：$A \leftrightarrow B$ 是对象语言里的一个复合命题（可能真也可能假）；$A \equiv B$ 是元语言里的断言，它说「$A \leftrightarrow B$ 是重言式」。前者是被谈论的式子，后者是谈论的结果。

## 2 基本等值式表

命题逻辑的等值式可以列成一张「代数表」，与算术里的分配律、交换律遥相呼应：

| 等值式 | 名称 | 直觉 |
| --- | --- | --- |
| $P \lor Q \equiv Q \lor P$ | 析取交换律 | 顺序无关 |
| $P \land Q \equiv Q \land P$ | 合取交换律 | 顺序无关 |
| $P \lor (Q \lor R) \equiv (P \lor Q) \lor R$ | 析取结合律 | 括号无关 |
| $P \land (Q \land R) \equiv (P \land Q) \land R$ | 合取结合律 | 括号无关 |
| $P \land (Q \lor R) \equiv (P \land Q) \lor (P \land R)$ | 分配律 | 乘法分配律的类比 |
| $P \lor (Q \land R) \equiv (P \lor Q) \land (P \lor R)$ | 分配律 | 或对与也分配 |
| $\neg(\neg P) \equiv P$ | 双重否定律 | 负负得正 |
| $\neg(P \land Q) \equiv \neg P \lor \neg Q$ | 德摩根律 | 否定进入并翻转 |
| $\neg(P \lor Q) \equiv \neg P \land \neg Q$ | 德摩根律 | 否定进入并翻转 |
| $P \to Q \equiv \neg P \lor Q$ | 蕴涵改写 | 「如果不P，则Q」 |
| $P \to Q \equiv \neg Q \to \neg P$ | 逆否等价 | 与原式同真假 |
| $P \leftrightarrow Q \equiv (P \to Q) \land (Q \to P)$ | 等值展开 | 双向蕴涵 |

这张表不是死记硬背的清单，而是**改写工具**：任何复杂的命题公式，都可以用这些规则逐步替换，变成等价而更简单的形式。

## 3 德摩根律：否定号如何穿过

德摩根律（De Morgan's laws）是全部等值式中最重要的一条，它回答「否定号放在括号前时会发生什么」：

$$
\neg(P \land Q) \equiv \neg P \lor \neg Q, \qquad \neg(P \lor Q)
 \equiv \neg P \land \neg Q
$$

用自然语言理解：「并非（$P$ 且 $Q$）」=「非 $P$ 或 非 $Q$」——要让「且」整体为假，只需至少一边为假；「并非（$P$ 或 $Q$）」=「非 $P$ 且 非 $Q$」——要让「或」整体为假，必须两边都假。<span class="marginnote">德摩根律的名字来自英国数学家奥古斯塔斯·德摩根（Augustus De Morgan，1806—1871），他是布尔的同时代人，也是形式逻辑代数化的先驱。这条律在编程里无处不在：`!(a && b)` 等价于 `!a || !b`——你写的每一行条件判断都在用它。</span>

口诀：**否定号穿进括号，$\land$ 与 $\lor$ 互换，各支取否定。
** 德摩根律在谓词逻辑中还有量词版本（「并非所有人都」=「有人不」），那是第三篇《量词的否定与对偶关系》的核心。

## 4 公式解析：用等值式化简一条复合命题

等值式的价值在「化简」。看这条式子：

$$
\neg(P \to Q) \lor (\neg P \land Q)
$$

- **第一步，改写蕴涵**：$P \to Q \equiv \neg P \lor Q$，于是 $\neg(P \to Q) \equiv \neg(\neg P \lor Q)$。
- **第二步，德摩根律**：$\neg(\neg P \lor Q) \equiv \neg(\neg P) \land \neg Q \equiv P \land \neg Q$。整个式子变成 $(P \land \neg Q) \lor (\neg P \land Q)$。
- **第三步，观察**：$(P \land \neg Q) \lor (\neg P \land Q)$ 正是「$P$ 与 $Q$ 恰有一个为真」——这就是**异或（exclusive or）**。于是原式化简为一个紧凑的描述。

每一行改写都保持了真值不变，所以最终的异或表达式与原式**逻辑等值**。这就是等值式作为「演算规则」的用法：像代数化简一样，把一个表达式改写到更清晰的形式。<span class="marginnote">异或 $P \oplus Q \equiv (P \land \neg Q) \lor (\neg P \land Q)$ 不是基本联结词，但可用基本联结词定义——这预告了第十课《主范式与联结词的完备集》：只要具备足够的表达能力，任何真值函项都能用基本联结词搭出来。</span>

## 5 例题演练

**例 1**：用等值式化简 $\neg(P \lor Q) \lor (\neg P \land \neg Q)$。

- 先德摩根：$\neg(P \lor Q) \equiv \neg P \land \neg Q$。于是原式 $= (\neg P \land \neg Q) \lor (\neg P \land \neg Q)$——自析取，仍等于 $\neg P \land \neg Q$。化简结果与「非 P 且 非 Q」等值。

**例 2**：证明 $P \to (Q \to R) \equiv (P \land Q) \to R$（输出/输入等价）。

- 左边：$\neg P \lor (\neg Q \lor R) \equiv (\neg P \lor \neg Q) \lor R \equiv \neg(P \land Q) \lor R \equiv (P \land Q) \to R$。这条等值式在自然演绎里对应「条件证明可以打包展开」。

**例 3**：$P \to Q$ 与 $\neg P \to \neg Q$ 等值吗？

- 不等值。$P$ 真 $Q$ 假时 $P \to Q$ 假，而 $\neg P \to \neg Q$ 假前件为真——一个反例就断交。初学者常把「逆否」（等值）与「逆命题」（不等值）混淆。

**例 4**：为什么「逆否等价」是唯一可反向的蕴涵操作？

- $P\to Q\equiv\neg Q\to\neg P$ 是重言式；而「逆命题」$Q\to P$ 与「否命题」$\neg P\to\neg Q$ 都不与 $P\to Q$ 等值。反向推导只能走逆否——这是防止肯定后件、否定前件谬误的根源。

**延伸思考**：等值式表是「改写工具箱」——化简、证明两式同义、重构代码，全靠「真值不变」这条底线。

**例 5**：为什么「等值」是比「同真」更强的概念？

- 两个命题「碰巧同真」（如「北京下雨」与「伦敦下雨」）不是等值——等值要求「所有赋值下同真同假」。等值是结构性的，同真是偶然的。

**延伸思考**：检验等值用真值表逐行比对，或逐步等值式改写——两条路殊途同归。

**一句话记忆**：等值式是改写不是事实——$P\equiv Q$ 说的是「所有赋值下同真同假」，不是「碰巧都为真」。

**本节要点自检**：等值式是改写工具——化简、证明两式同义、定义新联结词，全靠「真值不变」这条底线。

## 6 小结

- **逻辑等值** $A \equiv B$：$A \leftrightarrow B$ 是重言式，两式在任何赋值下同真同假。
- 检验方法：真值表逐行比对，或逐步用等值式改写。
- 核心等值式：**交换律、结合律、分配律、双重否定、德摩根律、蕴涵改写、逆否等价**。
- **德摩根律**的实质：否定号穿进括号，$\land$ 与 $\lor$ 互换，各支取否定。
- 等值式是**改写工具**：化简命题、定义新联结词（如异或）、为后续范式化与电路设计铺路。

在下一节，我们直面蕴涵那个最刺眼的真值表行——**前件为假时 $P \to Q$ 恒真**。
为什么逻辑学家甘愿接受这种「怪论」？
