---
title: 集合恒等式与文氏图
date: 2026-08-07
---

# 集合恒等式与文氏图

<div class="epigraph">
<p>逻辑必须照料好自己。</p>
<footer>—— 路德维希 · 维特根斯坦（Ludwig Wittgenstein），《逻辑哲学论》</footer>
</div>

<div class="article-byline">
<p>第二级 · 离散数学 ｜ Rosen《离散数学》§2.2 ｜ 2026-08-07</p>
</div>

## 为什么从集合恒等式开始

上一节我们定义了五种集合运算。但会算还不够——离散数学真正的力量在于**化简**：把一条长得吓人的集合表达式，一步步改写成一个极简形式，就像中学代数里把多项式展开再合并同类项。完成这个任务的工具，就是**集合恒等式（set identity）**——两个集合表达式对任意集合都相等的式子。

集合恒等式的意义有三层。第一，它们是**证明的弹药库**：判断 $A \cup (B \cap C)$ 与 $(A \cup B) \cap (A \cup C)$ 是否相等，不用每个元素去验，查表即知。第二，它们是**布尔代数与逻辑电路的地基**：第十三篇《布尔代数》里，同一批恒等式会以「与或非」的形态原样重现，而计算机的加法器、比较器本质就是这些恒等式的硅片实现。第三，它们在工程里被无声使用：数据库查询优化器把 $A \cup (B \cap C)$ 条件改写成等价但更快的形式，靠的正是集合（关系）恒等式。<span class="marginnote">本节是「恒等式 × 证明方法」的双线并进：一条线是「有哪些恒等式」，另一条线是「怎么证明/验证它们」。后一条线——元素法、文氏图、逻辑等价——才是本次真正要带走的技能。</span>

## 1 主要恒等式清单

罗森书中把最基本的恒等式整理成一张「公式表」（§2.2 的表 1），共十组。设 $A, B, C$ 为全集 $U$ 的任意子集：

| 恒等式 | 名称 |
| --- | --- |
| $A \cup \emptyset = A$，$A \cap U = A$ | 同一律（Identity laws） |
| $A \cup U = U$，$A \cap \emptyset = \emptyset$ | 支配律（Domination laws） |
| $A \cup A = A$，$A \cap A = A$ | 幂等律（Idempotent laws） |
| $\overline{\overline{A}} = A$ | 双重否定律（Complementation law） |
| $A \cup B = B \cup A$，$A \cap B = B \cap A$ | 交换律（Commutative laws） |
| $(A \cup B) \cup C = A \cup (B \cup C)$，$(A \cap B) \cap C = A \cap (B \cap C)$ | 结合律（Associative laws） |
| $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$，$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ | 分配律（Distributive laws） |
| $A \cup (A \cap B) = A$，$A \cap (A \cup B) = A$ | 吸收律（Absorption laws） |
| $A \cup \overline{A} = U$，$A \cap \overline{A} = \emptyset$ | 互补律（Complement laws） |
| $\overline{A \cup B} = \overline{A} \cap \overline{B}$，$\overline{A \cap B} = \overline{A} \cup \overline{B}$ | 德摩根律（De Morgan's laws） |

**重点：这张表是成对出现的。** 把任一恒等式里的 $\cup$ 与 $\cap$ 互换、$\emptyset$ 与 $U$ 互换，就会得到同组的另一条——这叫**对偶原理**。例如同一律里 $A \cup \emptyset = A$ 对偶成 $A \cap U = A$。记住对偶原理，十组二十条恒等式，实际只需记十条。

另外还有两条「工具箱」式的恒等式，来自上一节的定义，证明时会反复调用：

$$A - B = A \cap \overline{B}, \qquad A \oplus B = (A \cup B) - (A \cap B)$$

<span class="marginnote">为什么这套体系叫「布尔代数」？19 世纪布尔（George Boole）正是观察到逻辑联结词与这套运算共享同一批定律，才建立了一套「用代数算逻辑」的系统。第十一篇之后我们将看到它进化成逻辑电路中的与门、或门、非门。</span>

## 2 用文氏图验证恒等式

拿到一条恒等式，第一反应是**画文氏图验货**：两侧分别着色，若着色区域完全一致，则高度可信；若不一致，则该等式是假命题，且差异区域就是反例所在。

以分配律 $A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ 为例：左侧先画 $B \cup C$，再与 $A$ 取交，得到的着色区域是「$A$ 与 $B \cup C$ 的重叠」；右侧先分别画 $A \cap B$ 与 $A \cap C$，再取并，得到的同样是「既在 $A$ 中、又在 $B$ 或 $C$ 中」的区域。两幅图一致，命题可信。

文氏图的真正威力在于**快速找到反例**。试看一个貌似合理的假命题：

$$A - (B \cup C) \overset{?}{=} (A - B) \cup C$$

画图会立刻发现：右侧包含了「$C$ 中不属于 $A$ 的元素」，而左侧是「在 $A$ 中但不在 $B$、$C$ 中」，两者差出一整块 $C - A$。取 $A = \{1\}$、$B = \{2\}$、$C = \{2\}$，左侧是 $\{1\}$，右侧是 $\{1, 2\}$，反例成立。

**辨析｜易错点：文氏图不能当严格证明。** 它只对「画得出来的那些情形」负责，而两个集合的包含关系可能隐藏着画图时没考虑到的边界情形（比如空集、全集、部分重叠）。正确分工是：**文氏图用于发现与验证，元素法用于严格证明。**

## 3 元素法：把集合问题翻译成逻辑问题

**元素法（element proof）** 是最严谨的证明手段，原理一句话：**证明集合相等，就证明两边的元素集合相同**。标准流程是「双向包含」——先证 $A \subseteq B$，再证 $B \subseteq A$。

以分配律 $A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$ 为例，证 $A \cup (B \cap C) \subseteq (A \cup B) \cap (A \cup C)$：

任取 $x \in A \cup (B \cap C)$。由并集定义，$x \in A$ 或 $x \in B \cap C$。

- 若 $x \in A$，则 $x \in A \cup B$ 且 $x \in A \cup C$，故 $x \in (A \cup B) \cap (A \cup C)$；
- 若 $x \in B \cap C$，则 $x \in B$ 且 $x \in C$，于是 $x \in A \cup B$ 且 $x \in A \cup C$，同样得到 $x \in (A \cup B) \cap (A \cup C)$。

这就证完了左侧包含于右侧。反向的 $B \subseteq A$ 类似，把推理倒过来走一遍即可。**两个方向合起来，分配律得证。**

这个「任取一个元素 → 用定义展开 → 用逻辑规则重组 → 收束回目标」的套路，几乎能机械地套用到所有集合恒等式上——因为集合运算的定义本就是「用逻辑联结词写成的」。

## 4 公式解析：德摩根律的证明

德摩根律是全部恒等式中最能体现「集合与逻辑同构」的一条：

$$
\overline{A \cup B} = \overline{A} \cap \overline{B}
$$

它说的是：**「$A$ 或 $B$」之外的东西，恰是「$A$ 之外」与「$B$ 之外」的交集。** 我们来逐步证明，并在每一步同时写下对应的逻辑等价式：

- **第一步，展开补集定义**：$x \in \overline{A \cup B}$ 当且仅当 $x \notin A \cup B$，即 $\neg(x \in A \cup B)$。
- **第二步，展开并集定义**：$x \in A \cup B$ 的意思是 $x \in A \lor x \in B$，所以 $\neg(x \in A \cup B)$ 即 $\neg(x \in A \lor x \in B)$。
- **第三步，用逻辑上的德摩根律**：第一篇讲过 $\neg(p \lor q) \equiv (\neg p) \land (\neg q)$。把 $p = x \in A$、$q = x \in B$ 代入，得 $\neg(x \in A) \land \neg(x \in B)$。
- **第四步，翻译回集合语言**：$\neg(x \in A)$ 即 $x \in \overline{A}$，$\neg(x \in B)$ 即 $x \in \overline{B}$，二者同时成立即 $x \in \overline{A} \cap \overline{B}$。

每一步都是「定义 ↔ 逻辑联结词」的来回翻译，最终得到 $\overline{A \cup B} = \overline{A} \cap \overline{B}$。另一条 $\overline{A \cap B} = \overline{A} \cup \overline{B}$ 由对偶原理或同一流程（把 $\lor$ 换成 $\land$）得到。

**重点：这一串翻译之所以每一步都合法，是因为我们预先掌握了对应的逻辑等价式。** 换句话说，集合恒等式的证明库 = 集合定义 + 逻辑等价式库。第一篇学的逻辑等价，在这里真正「投产」。

## 5 应用：化简集合表达式

恒等式最大的用处是**化简**。看这条算式：

$$(A \cup B) \cap (A \cup \overline{B})$$

硬算很繁琐，但套用分配律（$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$ 的反向）：

$$
\begin{aligned}
(A \cup B) \cap (A \cup \overline{B})
&= A \cup (B \cap \overline{B}) & &\text{（分配律）} \\
&= A \cup \emptyset & &\text{（互补律）} \\
&= A & &\text{（同一律）}
\end{aligned}
$$

三步就把一条复杂表达式化简成了 $A$。<span class="marginnote">这个例子不是编出来的玩具：数据库查询优化器做的事与它同构——把 $A$ 这类条件集合改写成等价的 $A$，让扫描量从「两个表」降为「一个表」，正是现代 SQL 引擎每秒都在做的集合恒等式化简。</span>

另一种「化简/验证」手段是**成员表（membership table）**：枚举元素 $x$ 属于各集合的所有 8 种组合（$x \in A$ 或不在、$x \in B$ 或不在、$x \in C$ 或不在），逐列计算两侧的真值。它与逻辑里的真值表完全同构，适合验证那些画图容易出错的恒等式，也适合当程序员的脑内「单元测试」。

## 6 小结

- **十组集合恒等式**（同一、支配、幂等、双重否定、交换、结合、分配、吸收、互补、德摩根）是化简与证明的公式表，且成对出现、服从**对偶原理**。
- **文氏图**适合验证与找反例，但**不能作为严格证明**。
- **元素法**通过「双向包含」证明集合相等；一切集合证明都可归结为「集合定义 + 逻辑等价式」的翻译。
- **德摩根律** $\overline{A \cup B} = \overline{A} \cap \overline{B}$ 是集合与逻辑同构的最佳样板。
- 化简集合表达式（如 $(A \cup B) \cap (A \cup \overline{B}) = A$）是查询优化等工程应用的地基。

在下一节，我们将给集合加上两种「放大镜」式的结构——**幂集与笛卡尔积**：前者把集合变成「所有子集组成的集合」，后者把集合拼成「有序对的全体」，它们将直接为第十篇《关系》铺路。
