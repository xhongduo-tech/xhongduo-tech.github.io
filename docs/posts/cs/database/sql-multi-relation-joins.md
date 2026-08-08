---
title: 多关系查询：连接与笛卡儿积
date: 2026-08-07
---

# 多关系查询：连接与笛卡儿积

<div class="epigraph">
<p>关系模型的能力来自于把信息分解到多张表中，再由查询把它们重新缝合起来。</p>
<footer>—— 埃德加 · 科德（E. F. Codd），《大型共享数据银行的关系模型》（1970）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.3–3.4 ｜ 2026-08-07</p>
</div>

## 为什么从多关系查询开始

单关系查询解决的是「对一张表问问题」。但现实数据库的价值恰恰在于**数据被拆分到多张表里**：教师的基本信息存在 `instructor`，谁教哪门课存在 `teaches`，课程信息存在 `course`。要回答「计算机系每个教师各教哪几门课」，就必须把两张表**缝合**起来。这个缝合动作，就是连接（join）。

这一节把 SQL 里的多关系查询拆到底。你会看到三件套如何被推广到 `FROM` 后接多张表，看到那条看似昂贵的笛卡儿积如何与 `WHERE` 配合出「连接」语义，还会认识 `NATURAL JOIN` 与 `USING` 两个语法糖。**连接是 SQL 全篇最核心的运算**——后面几节的子查询、聚集、第4章的外连接，全都建立在「多表怎么并起来」这一课上。<span class="marginnote">科德 1970 年那篇论文的名字就叫「大而共享的数据银行的关系模型」，核心论点是：与其把数据塞进一张巨型表，不如分解成规范化的多张表，再用查询语言重组。连接就是重组的手术刀。</span>

## 1 FROM 里出现多张表：笛卡儿积

单关系查询的 `FROM` 子句只有一张表。多关系查询把它扩展成：

```sql
SELECT name, course_id
FROM instructor, teaches
WHERE instructor.ID = teaches.ID;
```

`FROM` 子句的意思是：先把两张表做**笛卡儿积（Cartesian product）**——每一行与另一张表的每一行配对——得到一张巨大的中间表；`WHERE` 再在这张中间表上筛行；`SELECT` 最后投影选列。

**重点：`FROM` 逗号 = 笛卡儿积。** 关系代数里写作 $instructor \times teaches$。若 $|instructor| = n_1$、$|teaches| = n_2$，则中间表有 $n_1 \times n_2$ 行。<span class="marginnote">还记得关系代数那节我们说过 $\times$ 是基本运算吗？SQL 的 `FROM` 逗号就是它的直接翻译。第11章我们会算它的代价：数据量一大，笛卡儿积的膨胀是灾难性的，这正是优化器要拼命消除它的原因。</span>

## 2 用 WHERE 写连接条件：等值连接

光有积还不够——`WHERE` 这一行是灵魂。它叫**连接条件（join condition）**，把「同一名教师的教师信息与其授课记录」匹配起来。这种「按某属性相等缝合两表」的连接叫**等值连接（equi-join）**。

**辨析｜易错点：列名歧义。** 两张表都有 `ID` 列，直接写 `ID` 会因**属性名歧义**而报错。必须用**关系名限定**：`instructor.ID` 与 `teaches.ID`。当属性只出现在一张表时（如 `name`），可以省略限定；一旦有歧义，就必须写全。

```sql
SELECT name, course_id
FROM instructor, teaches
WHERE instructor.ID = teaches.ID
  AND dept_name = 'Music';
```

这里把**连接条件与选择条件混在同一个 `WHERE` 里**。SQL 允许，初学者也爱这么写，但语义上要分清：`instructor.ID = teaches.ID` 是「连接」，`dept_name = 'Music'` 是「选择」。两者混写不报错，却容易让代价分析（第12章）失灵——优化器得自己费力把它们重新分开。关系代数里，它们本就是两个独立的算子。

## 3 公式解析：多关系查询的三步拆解

把上面的查询翻译成关系代数，整条链路就透明了：

$$
\Pi_{name,\ course\_id}\Big(\sigma_{instructor.ID = teaches.ID \ \wedge \ dept\_name = \text{'Music'}}(instructor \times teaches)\Big)
$$

**第一步，笛卡儿积**：$instructor \times teaches$。假如 `instructor` 有 1400 行、`teaches` 有 13000 行，则中间表有 $1400 \times 13000 = 18{,}200{,}000$ 行——一千八百万行，绝大多数是无意义的错配。

**第二步，选择**：$\sigma_{instructor.ID = teaches.ID \wedge dept\_name = \text{'Music'}}$。先按 `ID` 相等把错配丢掉（假设每行 `teaches` 恰好对应一名教师，剩下约 13000 行），再按 `dept_name` 只留音乐系（约 40 行）。

**第三步，投影**：$\Pi_{name,\ course\_id}$。只留下 `name` 与 `course_id` 两列，得到最终结果。

**关键直觉：连接 = 「积」后「选」。** 数学上，等值连接就是笛卡儿积加一个选择，两者没有新增表达能力。但**工程上**两者天差地别：先积再选要物化一千八百万行中间结果；而真正实现时，数据库会边读边匹配，只产生约 40 行的最终结果。第11章的连接算法（嵌套循环、归并连接、哈希连接）全部在做同一件事——**避免真的把笛卡儿积物化出来**。先记住这条公式，后面代价分析时它是最重要的起点。

## 4 语法糖：NATURAL JOIN 与 USING

`WHERE` 里手写连接条件太啰嗦，且容易把连接条件与选择条件混成一团。SQL 提供了更清晰的写法：

```sql
SELECT name, course_id
FROM instructor NATURAL JOIN teaches;
```

**自然连接（natural join）**：自动匹配两表中**名字相同的所有属性**，且结果里相同属性只保留一列。`instructor` 与 `teaches` 共有 `ID`，于是它自动按 `ID` 等值连接——比手写 `WHERE` 更短。

**辨析｜易错点：NATURAL JOIN 的危险在于「自动匹配所有同名属性」。** 若两表除 `ID` 外还恰好都有 `created_at`，自然连接会**隐式**要求两边的 `created_at` 也相等——这常常不是你想要的条件，结果莫名其妙地少掉很多行。<span class="marginnote">著名的生产事故：某系统给用户表与订单表都加了 `created_at` 列，`NATURAL JOIN` 就悄悄按 `created_at` 也连了一遍，导致大量订单「丢失」。很多团队因此明令禁用 `NATURAL JOIN`。</span>

更可控的折中是 `USING`，显式指定按哪些列连接：

```sql
SELECT name, course_id
FROM instructor JOIN teaches USING (ID);
```

它只在 `ID` 上做等值连接，语义透明。若要完全自定义连接条件（包括非等值），用 `ON`：

```sql
SELECT name, course_id
FROM instructor JOIN teaches
  ON instructor.ID = teaches.ID;
```

`ON` 与 `WHERE` 写法在这里等价，但职责更清晰：**`ON` 里放连接条件，`WHERE` 里放选择条件**。第4章讲外连接时，两者的区别会变得性命攸关。

## 5 三张以上的表：连接的可结合性

多关系查询不限于两张表。「列出计算机系每个教师教的每门课的课程名」要三张表：

```sql
SELECT name, title
FROM instructor, teaches, course
WHERE instructor.ID = teaches.ID
  AND teaches.course_id = course.course_id
  AND dept_name = 'CS';
```

三张表的笛卡儿积是 $n_1 \times n_2 \times n_3$，`WHERE` 里两条等值条件把无效配对逐层滤掉。等价的关系代数表达式可以写成两种结合方式，结果相同——这正是后面第12章**连接顺序优化**的数学依据：

$$
\Pi_{name,title}\Big(\sigma_{dept\_name=\text{'CS'}}\big((instructor \bowtie teaches) \bowtie course\big)\Big)
$$

**重点：连接的结合律与交换律成立。** $(A \bowtie B) \bowtie C$ 与 $A \bowtie (B \bowtie C)$ 语义相同，但**执行代价**可能差几个数量级——优化器的全部工作，就是在这群「语义等价」的候选里挑「执行最便宜」的那一个。

## 6 易错辨析：多关系查询的四个坑

- **忘记连接条件 = 笛卡儿积爆炸**：`FROM` 里多张表却不写 `WHERE` 连接条件，你会得到一千八百万行垃圾。这是初学者最经典的「查询把数据库卡死」事故。<span class="marginnote">有些数据库提供 `CROSS JOIN` 显式表示笛卡儿积，语义与逗号相同，但更醒目——用 `CROSS JOIN` 就是在对读者说：「我确实要积」。</span>
- **同名属性必须限定**：两表共有 `ID`，就得写 `instructor.ID`。省略限定符在大多数数据库里直接报「ambiguous column」。
- **`NATURAL JOIN` 可能多做连接**：自动按所有同名属性连接，`created_at`、`updated_at` 这类意外同名列会悄悄改变语义。
- **`WHERE` 里混写连接与选择**：不报错，但代价估算与索引选择会受影响。养成习惯：`ON` 管连接，`WHERE` 管选择。

## 7 小结

- 多关系查询的骨架：`FROM` 后多张表 = **笛卡儿积**，`WHERE` 里写**连接条件**，`SELECT` 投影。
- 连接 = 积 + 选：$\sigma_{cond}(r \times s)$ 与等值连接语义等价，但真正实现绝不物化笛卡儿积。
- **等值连接（equi-join）**按某属性相等缝合两表；同名属性要用**关系名限定**消歧义。
- `NATURAL JOIN` 自动匹配所有同名属性、结果去重列，简洁但危险；`USING` 显式指定列；`ON` 完全自定义连接条件。
- 连接满足**结合律与交换律**，语义等价但执行代价迥异——这是查询优化的数学起点。

在下一节，我们把「多表缝合」换成「多结果合并」：当你要的答案分布在两次查询的结果里，如何用**集合运算**——并、交、差——把它们拼起来，以及 SQL 的 `UNION`/`INTERSECT`/`EXCEPT` 与关系代数相比多了什么。
