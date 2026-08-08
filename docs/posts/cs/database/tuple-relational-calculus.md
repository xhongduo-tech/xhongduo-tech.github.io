---
title: 元组关系演算
date: 2026-08-07
---

# 元组关系演算

<div class="epigraph">
<p>告诉我你要什么，而不是怎么去拿——声明式查询的精髓。</p>
<footer>—— 埃德加 · 科德（E. F. Codd，关系模型之父）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第6章 形式化关系查询语言 ｜ 2026-08-07</p>
</div>

## 为什么从关系演算开始

前面学的关系代数是**过程式（procedural）**的：你要明确写出「先投影，再选择，再连接」的每一步。SQL 却是**声明式（declarative）**的——你只描述「想要什么样的结果」，执行步骤交给优化器。SQL 的语义根基不在关系代数，而在**关系演算（relational calculus）**：一种用逻辑公式描述结果的查询语言。本篇讲**元组关系演算（tuple relational calculus, TRC）**——SQL 的 WHERE 子句正是 TRC 公式的直接翻译。理解了 TRC，你才真正明白 WHERE 子句在逻辑上意味着什么。

## 1 从逻辑公式到查询

元组关系演算的基本查询形式是：

$$
\{ t \mid P(t) \}
$$

读作「所有满足谓词 $P$ 的元组 $t$ 的集合」。$t$ 是一个**元组变量**，遍历数据库中的所有关系；$P(t)$ 是谓词，筛出符合条件的元组。<span class="marginnote">这个形式与高中数学的<strong>描述法</strong> $\{x \mid x \text{ 具有性质}\}$ 完全同构——集合论语言在数据库里的又一次现身。回顾第一级《集合的概念》，你就会发现 TRC 就是「把关系当成集合，把查询写成描述法」。</span>

一个具体例子，找出薪资超过 80000 的教师：

$$
\{ t \mid t \in instructor \land t[{\text{salary}}] > 80000 \}
$$

$t \in instructor$ 声明 $t$ 遍历 instructor 关系，$t[{\text{salary}}]$ 取元组的属性值。**关系代数问「怎么算」，关系演算问「是什么」**——两种哲学的分界，从这里开始。

## 2 谓词的组合：原子公式与逻辑连接词

TRC 的谓词由**原子公式**通过逻辑连接词组合而成。原子公式有三类：

- $s \in r$：元组 $s$ 是关系 $r$ 的成员。
- $s[A] \;\theta\; u[B]$：两个元组变量的属性比较，$\theta \in \{=, \ne, <, \le, >, \ge\}$。
- $s[A] \;\theta\; c$：元组属性与常量比较。

原子公式用**逻辑连接词**组合成复合谓词：

- **合取** $\land$（AND）：两个公式同时为真
- **析取** $\lor$（OR）
- **否定** $\neg$（NOT）
- **量词**：存在量词 $\exists$ 与全称量词 $\forall$

例如「找出所有在其所在系开了课程、且薪资超过 70000 的教师」：

$$
\{ t \mid t \in instructor \land t[{\text{salary}}] > 70000 \land \exists\, s \in teaches \; (s[{\text{ID}}] = t[{\text{ID}}]) \}
$$

**辨析｜易错点：** 量词的辖域（scope）是它后面括号里的公式。$\exists$ 读作「存在一个……使得」，$\forall$ 读作「对所有的……都有」。初学者最常犯的错是把 $\exists$ 与 $\forall$ 写反，导致「存在某门课」与「所有课程」完全不同的语义。

## 3 自由变量与查询结果的形状

**自由变量（free variable）**：在公式中**不被量词约束**的变量。TRC 查询的结果，就是「自由变量的所有取值组合」。<span class="marginnote">一个公式可以有多个自由变量，查询结果就是这些变量的组合构成的表。SQL 的 SELECT 列表与 TRC 的自由变量一一对应——这也解释了为什么 SQL 每个查询都返回一个关系。</span>

对比：

- $\{ t \mid t \in instructor \land t[salary] > 80000 \}$：$t$ 自由，结果是完整元组。
- $\{ \langle t[{\text{name}}] \rangle \mid t \in instructor \land t[salary] > 80000 \}$：只投影出 name 一个属性，结果是单列表。

**辨析｜易错点：** 所有自由变量都必须满足**安全约束**——变量的取值必须限定在某个关系的范围内。否则「所有不满足性质 X 的元组」这类 $\neg$ 公式会把「数据库里不存在的东西」也算进去，导致无穷结果。安全的 TRC 要求 $\neg$ 与 $\forall$ 只出现在「有限范围内」。

## 4 公式解析：全称量词表达「所有」

用 TRC 表达「找到所有选修了全部课程的学生」这类**全称**查询，是全称量词的标准用法：

$$
\{ t \mid t \in student \land \forall\, u \in course \; ( \exists\, v \in takes \; ( v[{\text{ID}}] = t[{\text{ID}}] \land v[{\text{course\_id}}] = u[{\text{course\_id}}] ) ) \}
$$

- **第一步，读懂嵌套**：外层的 $\forall u \in course$ 说「对每一门课程 $u$」，内层 $\exists v \in takes$ 说「都存在一条选课记录 $v$ 把它与当前学生 $t$ 连起来」。
- **第二步，翻译成中文**：对每一门课，都有一位学生选过它——即该学生选修了**全部**课程。
- **第三步，对照关系代数**：这个查询需要**除法**运算。关系演算的 $\forall$ 与关系代数的除法 $\div$ 表达的是同一个语义，但逻辑语言表达得更自然。
- **第四步，注意否定陷阱**：把 $\forall$ 换成「不存在的反例」可以改写成 $\neg \exists$。SQL 里的 NOT EXISTS 与 NOT EXISTS（嵌套）正是 $\forall$ / $\neg \exists$ 的实现。

## 5 与 SQL 的对应关系

**SQL 的 WHERE 子句几乎就是 TRC 公式的线性化。** 三句对应：

| TRC 成分 | SQL 对应 | 说明 |
| --- | --- | --- |
| $t \in r$ | `FROM r` | 元组变量遍历关系 |
| $t[A] > c$ | `WHERE t.A > c` | 属性比较 |
| $\exists\, s \in r$ | `EXISTS (SELECT ...)` | 相关子查询 |
| $\forall\, s \in r$ | `NOT EXISTS (SELECT ...)` | 全称化为否定存在 |
| 自由变量 | `SELECT` 列表 | 输出哪些属性 |

**辨析｜易错点：** TRC 与 SQL 的对应不是机械的。SQL 的 DISTINCT 去重对应 TRC 的**集合**语义（无重复元组）；若不写 DISTINCT，SQL 默认是多重集。逻辑语言天然是集合，SQL 为了工程效率引入了多重集——这个差异在查询等价性讨论中反复出现。

## 6 小结

- 元组关系演算用 $\{t \mid P(t)\}$ 声明式地描述结果，是 SQL WHERE 子句的**逻辑根基**。
- 谓词由原子公式经 $\land, \lor, \neg, \exists, \forall$ 组合而成。
- **自由变量**决定输出形状；安全约束保证结果有限。
- $\forall$ 与关系代数的**除法**等价，SQL 用 NOT EXISTS 实现。
- TRC 是集合语义，SQL 默认多重集，去重靠 DISTINCT。

在下一节，我们将换一种风格——把变量从「整行元组」换成「单个属性值」，这就是**域关系演算**。
