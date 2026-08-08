---
title: 集合运算：并、交、差
date: 2026-08-07
---

# 集合运算：并、交、差

<div class="epigraph">
<p>把两个集合合并起来的方法只有有限几种，而理解它们正是理解关系查询的捷径。</p>
<footer>—— 埃德加 · 科德（E. F. Codd），《关系完整性与规范化》</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.5 ｜ 2026-08-07</p>
</div>

## 为什么从集合运算开始

上一节我们学会了把多张表**横向缝合**（连接）。但有一类问题靠缝合解决不了：「2017 秋或 2018 春开过的课」——这是**两个查询结果之间的纵向合并**。要合并，就得回到关系模型的老家：**集合论**。

第一级《集合》里你已经见过 $\cup$、$\cap$、$-$ 三个运算，SQL 把它们翻译成 UNION、INTERSECT、EXCEPT。这节要做的，是看清翻译过程中的三件事：**并集相容条件**、**集合语义与多重集语义的又一次碰撞**、以及 ALL 版本在性能与语义上的双重意义。<span class="marginnote">从第一级数学里的集合运算到这里的 SQL，你会发现「运算」本身没变，变的是实现：数学集合保证互异，SQL 表却默认允许多重集——这套「集合 vs 多重集」的二重奏，是整个第3章反复响起的主题。</span>

## 1 并：UNION

经典问题：「找出 2017 年秋季学期或 2018 年春季学期开过的所有课程。」把两个单关系查询的结果合并起来：

```sql
(SELECT course_id FROM section WHERE semester = 'Fall'   AND year = 2017)
UNION
(SELECT course_id FROM section WHERE semester = 'Spring' AND year = 2018);
```

两个括号里是两条独立的查询，UNION 把它们的**结果行**合并成一张结果表。

**重点：UNION 默认去重，返回严格集合。** 假设 2017 秋有课 CS-101 开了 3 个教学班，第一条子查询本身会返回 3 行 CS-101（SQL 默认多重集）；但 UNION 会把跨两边、甚至同一遍内的重复全部消掉，最终 CS-101 只出现一次。这正是数学并集的互异性。

若你**想要**重复（例如统计每个教学班的记录都要保留），用 `UNION ALL`：

```sql
(SELECT course_id FROM section WHERE semester = 'Fall'   AND year = 2017)
UNION ALL
(SELECT course_id FROM section WHERE semester = 'Spring' AND year = 2018);
```

**辨析｜易错点：ALL 不是「可有可无」的修饰，而是语义开关。** 不带 ALL 是集合语义（去重），带 ALL 是多重集语义（保留重复）。去重需要排序或哈希，很贵——你明确知道两边无重复、或重复无意义时，写 `UNION ALL` 既能保留语义又能省掉一大笔开销。<span class="marginnote">性能直觉：UNION 要先去重再合并，代价近似「排序两个结果集」；UNION ALL 只是把两段结果拼在一起。第11章排序、第12章代价估算时，这个差异会被精确量化。很多线上慢查询就是「该用 UNION ALL 却写了 UNION」。</span>

## 2 交：INTERSECT

「找出 2017 年秋**和** 2018 年春都开过的课程」——两边的**公共部分**：

```sql
(SELECT course_id FROM section WHERE semester = 'Fall'   AND year = 2017)
INTERSECT
(SELECT course_id FROM section WHERE semester = 'Spring' AND year = 2018);
```

INTERSECT 同样默认去重。**辨析｜易错点：INTERSECT 不是「两个条件 AND 在一起」。** 新手常写成 `WHERE semester = 'Fall' AND year = 2017 AND semester = 'Spring' AND year = 2018`——这一行永远为假，因为 semester 不可能同时等于两个值。「秋开过**且**春开过」要求的是**同一门课在两个不同时间段各出现一次**，这在单张表的单行上无法表达，必须把时间维度拆成两个结果集再相交。这个「跨维度 AND 需要集合运算」的觉悟，是本周期的关键跳跃。

## 3 差：EXCEPT

「找出 2017 年秋开设、但 2018 年春未开设的课程」——从左边**减去**右边：

```sql
(SELECT course_id FROM section WHERE semester = 'Fall'   AND year = 2017)
EXCEPT
(SELECT course_id FROM section WHERE semester = 'Spring' AND year = 2018);
```

EXCEPT（甲骨文/Oracle 方言里叫 MINUS）输出「属于左边、不属于右边」的行。它与后面子查询里的 NOT IN 语义相近，但**在空值处理上截然不同**——EXCEPT 走集合逻辑，NOT IN 踩三值逻辑的坑，这个雷我们到《嵌套子查询》一节专门排。

## 4 公式解析：三个运算的一个等式家族

把三个运算并排写成关系代数，结构立刻清晰。设
$F = \Pi_{course\_id}\big(\sigma_{semester=\text{'Fall'} \wedge year=2017}(section)\big)$，
$S = \Pi_{course\_id}\big(\sigma_{semester=\text{'Spring'} \wedge year=2018}(section)\big)$，则：

$$
F \cup S,\qquad F \cap S,\qquad F - S
$$

用集合的谓词定义逐个拆解：

- **并集** $F \cup S = \{x \mid x \in F \vee x \in S\}$：属于 F **或**属于 S。SQL 的 UNION。
- **交集** $F \cap S = \{x \mid x \in F \wedge x \in S\}$：同时属于两边。SQL 的 INTERSECT。
- **差集** $F - S = \{x \mid x \in F \wedge x \notin S\}$：属于 F 但不属于 S。SQL 的 EXCEPT。

**第一步**，两个 $\Pi$ 投影分别把 course_id 取出；注意 section 里同一课程同一学期可能有多个教学班，所以**投影后仍是多重集**——course_id 可能重复出现。

**第二步**，集合运算按**集合语义**执行：无论输入是否含重复，输出都自动去重。这就是为什么 SQL 的 UNION 不带 ALL 时，能把两条含重复的子查询结果并成一个干净的集合。

**第三步**，关于结果属性名：并、交、差的**结果列名取自第一个查询**，两侧列数必须相同、对应列类型必须兼容（**并集相容性**，union compatibility）。若第一个查询写 `SELECT course_id`，结果列就叫 course_id；两个查询的列名不同没关系，只要求**数量与类型**匹配。<span class="marginnote">并集相容性在关系代数里是 $\cup$ 成立的前提：关系 $r$ 与 $s$ 可并，当且仅当它们有相同的属性个数，且对应属性域兼容。违反这条，SQL 直接报错；关系代数里则是「未定义」。</span>

## 5 集合运算的工程侧面

- **`ORDER BY` 只能放在整个集合表达式末尾**，不能放在单个子查询里——先并、再排序。

```sql
(SELECT course_id FROM section WHERE semester = 'Fall'   AND year = 2017
 UNION
 SELECT course_id FROM section WHERE semester = 'Spring' AND year = 2018)
ORDER BY course_id;
```

- **方言差异**：标准 SQL 用 `EXCEPT`，Oracle 用 `MINUS`；MySQL 长期只支持 `UNION`，到 8.0.31（2022 年）才补上 `INTERSECT` 与 `EXCEPT`。老项目里用 `NOT IN` 或 `LEFT JOIN ... IS NULL` / `IN` 模拟交、差是常见做法——那些写法在下一节和《嵌套子查询》里会逐一登场。

**重点：集合运算与连接分工明确。** 连接是**横向**缝合（加宽列），集合运算是**纵向**合并（加长行）。「并」加行且去重，「积」加列且不配对——两种方向的混淆，是初学 SQL 最大的语义迷障之一。

## 6 易错辨析：集合运算的四个坑

- **忘了 ALL 是语义开关**：UNION 去重、UNION ALL 不去重。别把「去重太贵」归咎于数据库——那是你没写 `UNION ALL`。
- **INTERSECT 不是两个条件 AND**：「秋开且春开」必须拆成两个结果集相交，单表单行写不出跨时间的 AND 条件。
- **并集相容性被破坏**：列数不同、或对应列类型不兼容（如 `VARCHAR` 对 `INTEGER`），直接报错。先投影出同样的列、把类型对齐后再并。
- **`ORDER BY` 位置**：写进单个子查询内部多半无效或报错，必须放在整个并/交/差表达式的最后。

## 7 小结

- SQL 三个集合运算：UNION（并 $\cup$）、INTERSECT（交 $\cap$）、EXCEPT（差 $-$，Oracle 为 MINUS）。
- 三者**默认去重**（集合语义）；带 ALL 的版本（`UNION ALL`、`INTERSECT ALL`、`EXCEPT ALL` 等）保留重复（多重集语义），且性能更优。
- **并集相容性**：两侧列数相同、对应域兼容；结果属性名取第一个查询。
- INTERSECT 与「单行内 AND」的本质区别——跨维度比较必须靠集合运算。
- 连接加宽列、集合运算加长行，两个方向不可混淆。

在下一节，我们回答「聚合」问题：当我们不再想要每一行，而是想要**每组的统计值**——总人数、平均工资、每个系的门数——GROUP BY 与 HAVING 如何把「先分组、再聚合、后过滤」这条流水线翻译成 SQL。
