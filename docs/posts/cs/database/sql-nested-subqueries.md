---
title: 嵌套子查询与集合成员比较
date: 2026-08-07
---

# 嵌套子查询与集合成员比较

<div class="epigraph">
<p>一个查询的复杂，不在它的长度，而在它嵌套了几层对世界的判断。</p>
<footer>—— 埃德加 · 科德（E. F. Codd），《关系数据库：面向可移植性的务实观点》</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.8 ｜ 2026-08-07</p>
</div>

## 为什么从嵌套子查询开始

前三节我们处理的问题，都能在**一层查询**里解决。但真正的世界是分层的：「找出工资高于生物系某位教师的所有教师」——「生物系某位教师」本身就是一个查询；「找出所有没在 2018 春开课的课程」——「在 2018 春开课」又是另一个查询。

把一条查询**嵌进另一条查询的 WHERE 子句里**，就是**嵌套子查询（nested subquery）**。它把「判断条件需要先查一次」这件事变成了 SQL 的一等公民，也让 SQL 的表达能力一下子逼近了关系代数的极限。这一节我们逐个认识五件工具：**IN（成员）、SOME/ALL（集合比较）、EXISTS（存在性）、关联子查询、FROM 里的子查询**，并在最后直面它们与**三值逻辑**相遇时挖下的坑——那是下一节《空值与三值逻辑》的预告，也是全篇最阴险的雷区之一。<span class="marginnote">子查询让 SQL 从「一层流水线」升级为「可以递归地构造条件」。它与前面学的关系代数有个深刻区别：关系代数算子是扁平的组合，而子查询引入了<strong>嵌套作用域</strong>——内层查询可以引用外层查询的列，这是「关联子查询」的种子。</span>

## 1 集合成员：IN 与 NOT IN

子查询最基本的用法，是构造一个**值的集合**，再用 IN 判断某值是否属于它。上一节用 INTERSECT 写的「2017 秋与 2018 春都开过的课」，这里用 IN 换一种写法：

```sql
SELECT DISTINCT course_id
FROM section
WHERE semester = 'Fall' AND year = 2017
  AND course_id IN (SELECT course_id
                    FROM section
                    WHERE semester = 'Spring' AND year = 2018);
```

**重点：IN 就是数学里的「属于 $\in$」。** 外层查询逐行判断 course_id 是否落进内层子查询返回的集合；NOT IN 则是「不属于 $\notin$」。子查询结果为空集合时：IN 为假、NOT IN 为真——这与集合论完全一致。

## 2 集合比较：SOME 与 ALL

IN 只回答「在不在」。有时你要的是「比集合里**某个**大」或「比集合里**所有**都大」——这需要带量词的集合比较。

「找出工资高于**生物系任意一名**教师的教师」：

```sql
SELECT name
FROM instructor
WHERE salary > SOME (SELECT salary
                     FROM instructor
                     WHERE dept_name = 'Biology');
```

`salary > SOME (…)` 读作「大于某个」：只要大于生物系教师工资的**至少一个**就成立。若想「大于**所有**生物系教师」，把 SOME 换成 ALL：

```sql
SELECT name
FROM instructor
WHERE salary > ALL (SELECT salary
                    FROM instructor
                    WHERE dept_name = 'Biology');
```

**辨析｜易错点：SOME 与 ALL 在空集合上的行为相反。** 若生物系一名教师都没有：SOME 为**假**（没有可比较的对象），ALL 为**真**（「大于所有元素」对空集是平凡成立）。SQL 里的 `SOME` 等价于 `ANY`（标准曾用 ANY，现推荐 SOME），SOME 与 ANY 语义相同——前提是没有 NULL，一旦有 NULL，两个都要踩三值逻辑的坑，见第 5 节。

## 3 存在性：EXISTS 与 NOT EXISTS

IN 比较的是「列值」，而有时你要判断的是「**有没有这样一行**」。EXISTS 接受一个子查询，子查询返回**非空**即为真：

```sql
SELECT course_id
FROM section AS S
WHERE semester = 'Fall' AND year = 2017
  AND EXISTS (SELECT 1
              FROM section AS T
              WHERE T.semester = 'Spring' AND T.year = 2018
                AND T.course_id = S.course_id);
```

注意内层 WHERE 子句里的 `T.course_id = S.course_id`——它引用了**外层**查询的列 `S.course_id`。这种「内层引用外层」的子查询叫**关联子查询（correlated subquery）**，执行方式不是「先算一次子查询再用」，而是**外层每来一行，就用这一行的值去执行一次内层查询**。<span class="marginnote">代价直觉：非关联子查询只算一次，关联子查询要对外层每一行都算一次——若外层 1 万行，内层就执行 1 万次。优化器经常能把它们改写成连接（第12章），但写 SQL 时要有这根弦：关联子查询贵。</span>

**NOT EXISTS + 关联子查询是实现「差集」的通用写法。** 「找出 2017 秋开过、但 2018 春没开过的课程」：

```sql
SELECT DISTINCT course_id
FROM section AS S
WHERE semester = 'Fall' AND year = 2017
  AND NOT EXISTS (SELECT 1
                  FROM section AS T
                  WHERE T.semester = 'Spring' AND T.year = 2018
                    AND T.course_id = S.course_id);
```

## 4 子查询的其他落点：FROM 与 WITH

子查询不只住在 WHERE 子句里，还能出现在 FROM 子句（作为**派生关系**）与 WITH 子句（作为**公共表表达式**）中。

```sql
SELECT dept_name, avg_salary
FROM (SELECT dept_name, AVG(salary) AS avg_salary
      FROM instructor
      GROUP BY dept_name) AS dept_avg
WHERE avg_salary > 42000;
```

FROM 后的子查询必须先算出来，然后**像一张表一样**被外层查询读——相当于「先物化一张临时表」。更清晰的写法是用 WITH 给派生关系起名：

```sql
WITH dept_avg AS (
    SELECT dept_name, AVG(salary) AS avg_salary
    FROM instructor
    GROUP BY dept_name
)
SELECT dept_name, avg_salary
FROM dept_avg
WHERE avg_salary > 42000;
```

**重点：WITH 是给子查询起名字的语法糖，却改变了可读性的量级。** 多层嵌套的 FROM 子查询几乎不可读；拆成 WITH 命名块后，每个块只做一件事，像函数一样被后续查询复用。第5章的递归查询 `WITH RECURSIVE` 也建立在这条语法之上。

## 5 公式解析：NOT IN 与三值逻辑的陷阱

这是全篇最值得逐字拆解的陷阱。回想第一级《集合》，$x \notin A$ 的定义是：

$$
x \notin A \quad \Longleftrightarrow \quad x \neq a_1 \wedge x \neq a_2 \wedge \cdots \wedge x \neq a_n
$$

**第一步，把 NOT IN 展开成「一连串 `<>` 的不相等判断」**。例如：

```sql
SELECT name
FROM instructor
WHERE dept_name NOT IN (SELECT dept_name
                        FROM instructor
                        WHERE dept_name <> 'Biology');
```

子查询返回生物系之外所有出现过、且 dept_name 非空的系名。若结果恰好含一个 NULL，展开式就带上了一个未知项。

**第二步，代入三值逻辑**。SQL 里**与 NULL 比较**的结果是**未知（unknown）**，不是真、也不是假。于是对任何一个 dept_name 不为 NULL 的教师，他的判断式是：

$$
(\text{dept\_name} \neq \text{'CS'}) \wedge (\text{dept\_name} \neq \text{'Music'}) \wedge \cdots \wedge (\text{dept\_name} \neq \text{NULL})
$$

最后一项恒为 unknown。**真 ∧ unknown = unknown**，整个条件既不是真也不是假。

**第三步，WHERE 只留「真」的行**。unknown 的行被丢弃——于是这位教师的行**不会**出现在结果里。换句话说：**只要子查询结果里有一个 NULL，NOT IN 就会吞掉本应返回的所有行**，查询「神秘地」返回空集。

**关键结论：NOT IN 在子查询可能含 NULL 时是危险品。** 三条出路：① 用 NOT EXISTS 替代（它是布尔判断，NULL 陷阱少得多）；② 在内层用 `WHERE dept_name IS NOT NULL` 过滤；③ 确认该列有 `NOT NULL` 约束。这一节开头预告的「EXISTS / EXCEPT 走集合逻辑、NOT IN 踩三值逻辑」，就在这里应验——EXCEPT 是集合差，天然处理 NULL，不会犯这个错。

## 6 小结

- 子查询把「判断条件需要先查一次」变成 SQL 的一等公民；WHERE、FROM、WITH 都能落脚。
- IN / NOT IN 对应集合成员 $\in$ / $\notin$；SOME（ANY）对应「∃」、ALL 对应「∀」，空集上行为相反。
- EXISTS / NOT EXISTS 判断子查询**是否非空**；关联子查询引用外层列，外层每行执行一次，代价更高。
- FROM 子查询是派生关系，WITH 给它命名，显著提升可读性，是递归查询的起点。
- **NOT IN 遇到 NULL 会返回空集**（三值逻辑）；NOT EXISTS、过滤 NULL、或非空约束是三条出路。

在下一节，我们终于正面处理这节反复预警的那头大象——**空值与三值逻辑**：NULL 如何把真假二值撑成三值，比较运算在三值上如何求值，以及它如何渗透进比较、聚集、连接与子查询的每一个角落。
