---
title: 连接表达式：内连接、外连接与自然连接
date: 2026-08-07
---

# 连接表达式：内连接、外连接与自然连接

<div class="epigraph">
<p>整体大于部分之和。</p>
<footer>—— 亚里士多德（Aristotle，《形而上学》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从连接表达式开始

在第3章的 FROM 子句里，我们已经用逗号写出了连接：`FROM instructor, teaches WHERE instructor.ID = teaches.ID`。这其实是一种**内连接**——只保留两侧都匹配得上的行。但真实业务里有一个高频问题它回答不了：**「没有匹配的行怎么办？」**

举个最常见的例子：「列出所有课程，以及它们的先修课」。有先修课的课程自然匹配得上；但那些**没有先修课**的课程呢？用第3章的内连接，它们会从结果里悄悄消失——用户看到的是一份「缺了课程」的清单，却不知道少了谁。**外连接（outer join）**专门解决这个问题：它保留一侧（或两侧）未匹配的行，缺失的一侧用 NULL 补齐。这一节我们把内连接、外连接、自然连接三兄弟讲透，它们正是第2章关系代数里**外连接**这个扩展运算在 SQL 中的落地<span class="marginnote">关系代数里有左外、右外、全外连接三种扩展运算，符号分别是 $\bowtie_{\text{左}}$ 等；当时只是抽象的集合表达式，现在我们要把它们翻译成能真正跑起来的 SQL 语法。</span>。

## 1 四种连接：一张表看清区别

设两个关系 $R$ 与 $S$，用「匹配」表示满足连接条件的元组对，四种连接的关系是：

**内连接（inner join）**：$R \bowtie S$，只保留匹配的元组对。
**左外连接（left outer join）**：内连接的结果，**加上 $R$ 中所有未匹配的元组**，它们右侧的 $S$ 属性填 NULL。
**右外连接（right outer join）**：内连接的结果，**加上 $S$ 中所有未匹配的元组**，它们左侧的 $R$ 属性填 NULL。
**全外连接（full outer join）**：内连接的结果，**加上两侧所有未匹配的元组**，对面属性一律填 NULL。

直觉上：内连接是「只留成双成对的」，外连接是「成双的留下，落单的也要登场，只是空着半边座位」。SQL 里它们写作：

```sql
-- 内连接：只保留匹配的行
SELECT * FROM course INNER JOIN prereq
  ON course.course_id = prereq.course_id;
-- 左外连接：再补上左表未匹配的行
SELECT * FROM course LEFT OUTER JOIN prereq
  ON course.course_id = prereq.course_id;
-- 右外连接：再补上右表未匹配的行
SELECT * FROM course RIGHT OUTER JOIN prereq
  ON course.course_id = prereq.course_id;
-- 全外连接：两侧未匹配的都补上
SELECT * FROM course FULL OUTER JOIN prereq
  ON course.course_id = prereq.course_id;
```

`OUTER` 关键字可以省略，`INNER JOIN` 与 `JOIN` 是同一回事。<span class="marginnote">兼容性提醒：三种外连接里，LEFT 与 RIGHT 几乎人人支持；FULL OUTER JOIN 在 MySQL 里至今没有直接实现，需要用 LEFT JOIN UNION RIGHT JOIN 手工拼。</span>

## 2 左外连接的实战：找出没有先修课的课程

Silberschatz 的经典例子是「列出所有课程及其先修课，没有先修课的课程也要出现，先修课列填 NULL」：

```sql
SELECT course.course_id, title, prereq_id
FROM course LEFT OUTER JOIN prereq
  ON course.course_id = prereq.course_id;
```

这里 `course` 在共同属性 `course_id` 上做左外连接，再筛出 `prereq_id` 为 NULL 的行——它们正是**没有任何先修课的课程**。若换成内连接，这些行在第一阶段就被吞掉了，`IS NULL` 将一条也查不出。

**辨析｜易错点：外连接 + WHERE 的经典陷阱。** 看这条「想列出所有课程，只要 CS-101 的先修信息」的错误写法：

```sql
SELECT course.course_id, title, prereq_id
FROM course LEFT OUTER JOIN prereq
  ON course.course_id = prereq.course_id
WHERE prereq.course_id = 'CS-101';
```

看起来是左外连接，但 **WHERE 在连接完成之后才执行**。先左外连接补齐了 NULL，再被 `WHERE` 一筛，所有补了 NULL 的行（NULL = 'CS-101' 是 UNKNOWN，不满足）全被删掉——结果退化成内连接，未匹配的课程依旧消失。**判断条件是「连接条件」还是「筛选条件」，决定了它该进 ON 还是 WHERE**。想要「右表只按 CS-101 匹配、但左表行一个不少」，条件必须写进 ON：

```sql
SELECT course.course_id, title, prereq_id
FROM course LEFT OUTER JOIN prereq
  ON course.course_id = prereq.course_id
 AND prereq.course_id = 'CS-101';
```

## 3 自然连接与它的隐患

**自然连接（natural join）**：在两个关系的**所有共同属性**（同名且类型相容）上做等值连接，且结果里每个共同属性只保留一份。它写起来最省事：

```sql
SELECT name, course_id
FROM instructor NATURAL JOIN teaches;
```

`instructor` 与 `teaches` 的共同属性是 ID，于是自动在 `ID` 上连接——结果与显式 `JOIN ... USING (ID)` 一致。

**但省事正是危险的来源。** 自然连接的连接条件由**属性名**决定，而不是由人明确指定。一旦某张表的模式多出一个与对面同名的属性，连接条件就悄悄变了。设想你随手写出：

```sql
SELECT name, course_id
FROM instructor NATURAL JOIN teaches NATURAL JOIN course;
```

`instructor` 与 `course` 的共同属性是 `dept_name`，于是这条查询返回的是「**同一系**的教师与课程的笛卡儿积式组合」——这很可能根本不是用户想要的东西，但它**不报错、不警告**，静默地给出一个庞大而错误的答案。<span class="marginnote">正是出于这种「schema 一改、语义就漂移」的不确定性，工程规范普遍要求：生产代码禁用 NATURAL JOIN，一律显式写 JOIN ... ON 或 JOIN ... USING。可读性、可维护性、正确性三方面它都吃亏。</span>

## 4 公式解析：左外连接的集合定义

要真正理解外连接，最好回到集合语言。设 $R$、$S$ 是两个关系，$\bowtie$ 表示按连接条件 $\theta$ 的内连接，则**左外连接**可定义为：

$$
R \;\; \text{⟕}\;\; S \;=\; \underbrace{(R \bowtie_{\theta} S)}_{\text{匹配的元组对}} \;\cup\; \underbrace{\big\{\, t \cdot (\text{NULL},\dots,\text{NULL}) \mid t \in R,\; t \text{ 在 } S \text{ 中无匹配}\,\big\}}_{\text{左表落单的元组，右侧补 NULL}}
$$

逐项拆解：

- **第一项 $R \bowtie_{\theta} S$**：与内连接完全相同，是全部「成对」的元组，两边的属性合并在同一行里。
- **第二项是集合描述法**：取 $R$ 中「在 $S$ 里找不到任何匹配」的元组 $t$，把 $t$ 的所有属性原样保留，再在右侧拼接一组全 NULL 的 $S$ 属性。$\cdot$ 表示元组拼接。
- **并集 $\cup$**：两项拼到一起，就是左外连接——匹配的成对出现，落单的孤行也出现。

用一个两行的小例子验证。设 `course` 有 3 门课，`prereq` 只有 1 条先修关系：

| course | prereq |
| --- | --- |
| CS-101, Intro to CS | CS-301 ← CS-101 |
| CS-301, Algorithms | |
| MUS-101, Music Appreciation | |

左外连接 `course` 的结果有 **3 行**：CS-301 与 prereq 匹配上，得到 `(CS-301, CS-101)` 这一行；CS-101 与 MUS-101 在 prereq 里找不到以自己为 course_id 的行，于是右侧补 NULL。若换成内连接，结果只有 1 行——**多出来的 2 行，正是外连接「一个都不能少」的承诺**。

## 5 辨析：NATURAL JOIN、USING 与 ON 三选一

三种语法对应三种对连接条件的控制粒度：

| 语法 | 连接条件由谁决定 | 共同属性在结果中 |
| --- | --- | --- |
| `NATURAL JOIN` | 所有共同属性，自动 | 只保留一份 |
| `JOIN ... USING (A)` | 显式指定的属性 A | 只保留一份 |
| `JOIN ... ON` | 完全由你写 | 两个表各一份（需前缀区分） |

`USING` 是折中：连接条件由人指定，避免了自然连接的「模式漂移」，结果又像自然连接那样合并共同属性。`ON` 最灵活，可以写任意条件（包括非等值连接、复合条件），代价是两个表的同名属性都要带前缀访问。

**辨析｜易错点：** `ON` 里如果两个表恰好都有同名属性，结果会同时出现 `course.course_id` 与 `prereq.course_id` 两列；此时 SELECT 里写裸 `course_id` 会报「列引用不明确」。这是从「逗号连接 + WHERE」迁移到显式 JOIN 语法时最常见的编译错误——给列加上表前缀即可。

## 6 小结

- **内连接**只保留两侧匹配的行；**外连接**额外保留落单的行并用 NULL 补齐（左外保左、右外保右、全外全保）。
- 外连接的经典用途：找「没有对应项」的数据（如没有先修课的课程），先外连接再 `IS NULL` 过滤。
- **WHERE 在连接之后执行**：把右表的筛选条件写进 WHERE 会把外连接退化成内连接，应写进 ON。
- **NATURAL JOIN 由属性名自动决定连接条件**，schema 一变语义就漂移，生产环境慎用。
- 控制粒度排序：NATURAL JOIN（最省事）＜ USING（指定列）＜ ON（任意条件，最灵活）。
- 外连接的关系代数定义 = 内连接 ∪ 落单元组补 NULL，这一节就是第2章关系代数外连接的 SQL 化。

在下一节，我们处理另一种「想要一个虚拟的关系却不想真的建表」的需求——**视图**：如何把一段复杂的查询包装成一个可以反复引用的名字，以及这个虚拟关系能否被更新。
