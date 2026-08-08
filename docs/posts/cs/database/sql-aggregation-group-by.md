---
title: 聚集函数与分组（GROUP BY / HAVING）
date: 2026-08-07
---

# 聚集函数与分组（GROUP BY / HAVING）

<div class="epigraph">
<p>数据单独来看只是记录，被聚合之后才成为知识。</p>
<footer>—— 佚名，数据库工程师们的共识</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.7 ｜ 2026-08-07</p>
</div>

## 为什么从聚集函数开始

到目前为止，每条查询返回的都是**表中的某几行**。但大多数真实问题问的不是行，而是**统计**：「全系教师平均工资多少」「每个系各有多少名教师」「2017 秋开了几门课」。这类问题要把许多行**压成一个数**，或把一张表**切成几组**、每组压成一个数。

压缩行成数，靠**聚集函数（aggregate function）**；把表切成组再分别压缩，靠 **GROUP BY**；组压完之后还想筛组，靠 **HAVING**。这三件套是 SQL 从「取数工具」升级为「数据分析工具」的分水岭——也是从第一级《统计》的平均数、计数，到第五篇 OLAP 数据立方体的中间一站。<span class="marginnote">统计学里的「均值」「频数」在这里找到了 SQL 肉身：<strong>AVG、COUNT</strong>`。学会聚集，你就具备了用数据库做描述性统计的能力；而 <strong>GROUP BY</strong>` 按某个维度切组，正是后面《OLAP 操作：CUBE、ROLLUP》里「维度下钻」的雏形。</span>

## 1 五个基本聚集函数

SQL 提供五个标准聚集函数，全部接受一个**表达式**（通常是列名）作参数：

| 函数 | 含义 | 对 NULL |
| --- | --- | --- |
| AVG | 平均值 | 忽略 NULL |
| SUM | 求和 | 忽略 NULL |
| COUNT | 计数（非 NULL 的行数） | 忽略 NULL |
| MAX | 最大值 | 忽略 NULL |
| MIN | 最小值 | 忽略 NULL |

```sql
SELECT AVG(salary) AS avg_salary
FROM instructor;
```

**重点：除了 COUNT(*)，所有聚集函数都忽略 NULL。** 数学上，$AVG(\{100, \text{NULL}, 200\})$ 的标准答案是 $150$（对非空值求平均），而不是**把 NULL 当作一个普通数值参与运算**、也不是把 NULL 当 0 算出的 $100$。**辨析｜易错点：COUNT(*) 数「行」，COUNT(salary) 数「该列非 NULL 的值」。** 一张表 100 行、**salary 列**有 3 行是 NULL：**COUNT(\*)** = 100，**COUNT(salary)** = 97，**COUNT(DISTINCT salary)** = 去重后的非 NULL 个数。

## 2 GROUP BY：把表切成组

**不带 GROUP BY 的 AVG(salary)** 回答「全体平均」。要回答「**每个系**的平均工资」，就得先按 **dept_name** 把表切成若干组，再对每组分别求平均：

```sql
SELECT dept_name, AVG(salary)
FROM instructor
GROUP BY dept_name;
```

执行流程是流水线式的四步：

- **第一步，WHERE + GROUP BY**：先筛出参与分组的行（**WHERE** 先于分组执行——这条顺序将反复救你）。
- **第二步，GROUP BY**：按 **dept_name** 把行分成组，同系的行聚成一组。
- **第三步，聚集**：每组各自计算 **AVG(salary)**，每组输出一行。
- **第四步，SELECT**：输出 **dept_name** 与算好的平均值。

**重点：分组后，SELECT 子句里只能出现「分组属性」或「聚集函数」。** 因为每组被压成一行后，组内非分组、非聚集的列**没有一个唯一的代表值**。下面的查询是**非法的**（MySQL 关掉 **ONLY_FULL_GROUP_BY** 时能跑，但结果无意义）：

```sql
SELECT dept_name, name, AVG(salary)
FROM instructor
GROUP BY dept_name;   -- 非法：name 既非分组属性也非聚集函数
```

**name（姓名）** 在每个系里有很多个，压成一行后该显示谁？数据库没法回答——这正是 SQL 报错的理由。

## 3 公式解析：分组聚合的一条完整流水线

「找出教师人数不少于 2、且平均工资高于 42000 的系及其平均工资。」SQL：

```sql
SELECT dept_name, COUNT(*) AS cnt, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING COUNT(*) >= 2 AND AVG(salary) > 42000;
```

关系代数里，分组聚合写作**聚集运算** $\gamma$：

$$
\gamma_{dept\_name,\ COUNT(*) \rightarrow cnt,\ AVG(salary) \rightarrow avg\_salary}(instructor)
$$

逐步拆解：

- **第一步，投影分组属性并聚集**：$instructor$ 按 **dept_name** 分成 $k$ 组，每组算 **COUNT(\*)** 与 **AVG(salary)**，得到 $k$ 行，每行 $(dept\_name, cnt, avg\_salary)$。
- **第二步，HAVING = 对组结果再做一次选择**：在上面 $k$ 行里，选出满足 $cnt \ge 2 \wedge avg\_salary > 42000$ 的组。它相当于
  $$
  \sigma_{cnt \ge 2 \wedge avg\_salary > 42000}\big(\gamma_{dept\_name,\ COUNT(*) \rightarrow cnt,\ AVG(salary) \rightarrow avg\_salary}(instructor)\big)
  $$
- **第三步，SELECT**：只输出 **dept_name** 与 **avg_salary** 两列。

**关键直觉：WHERE 管行、HAVING 管组。** 如果把 **HAVING** 误写成 **WHERE**——**`WHERE AVG(salary) > 42000`**——会直接报错：因为 **WHERE** 在 **分组** 之前执行，那时根本还没有「组」、更谈不上 **对组过滤**。**求值顺序的完整链条是 FROM → WHERE → GROUP BY → HAVING → SELECT**，比单关系查询多出的两环正是分组的生命线。

## 4 HAVING 与 WHERE 的分工再辨析

「每个系里工资大于 80000 的教师人数」与「教师人数大于 2 的系」，两条查询考验你对顺序的理解：

```sql
-- ① 每个系里工资 > 80000 的教师人数（WHERE 先滤行）
SELECT dept_name, COUNT(*)
FROM instructor
WHERE salary > 80000
GROUP BY dept_name;

-- ② 教师人数 > 2 的系（HAVING 后滤组）
SELECT dept_name, COUNT(*)
FROM instructor
GROUP BY dept_name
HAVING COUNT(*) > 2;
```

查询 ① 里，工资 ≤ 80000 的行在分组前就被丢弃，**COUNT(\*)** 数的是「高薪教师数」；查询 ② 里，所有行先分组，**HAVING** 筛的是「组的大小」。**同一个 COUNT(\*)，WHERE 在前就数「筛后的行」，HAVING 在后就数「全组行」**——数字可能差很多，语义天差地别。<span class="marginnote">一个经典面试题：「每个系高薪教师数大于 2 的系」。正确答案是两条子句都要：先 <strong>WHERE</strong>` 把高薪行筛出来分组计数，再 <strong>HAVING</strong>` 只留计数大于 2 的系。删掉任何一条，答案都不对。</span>

## 5 聚集与空组、NULL 的细节

- **空组返回 NULL**：**SUM、AVG、MAX、MIN** 作用在一个空组（没有行）上返回 **NULL**，而 **COUNT** 返回 **0**。这不对称常被忽略。
- **NULL 分组**：NULL 值自成一族——按 **dept_name** 分组时，所有 **dept_name 为 NULL** 的行会聚成一组。
- **对聚集结果去重**：**COUNT(DISTINCT dept_name)** 数「出现的不同系名个数」，**AVG(DISTINCT salary)** 对去重后的工资求平均。**DISTINCT** 与聚集函数可以组合，代价同样是排序/哈希。

**辨析｜易错点：WHERE 里不能出现聚集函数。** **`WHERE COUNT(*) > 2`** 是语法错误——**WHERE** 逐行判断，而 **COUNT(\*)** 需要整组数据。凡是想「按统计结果过滤」，思考路径都应是：**先 GROUP BY 形成统计，再用 HAVING 过滤**。

## 6 小结

- 五个聚集函数 **AVG / SUM / COUNT / MAX / MIN**；除 **COUNT(\*)** 外**忽略 NULL**，**COUNT(\*)** 数行、**COUNT(col)** 数非空值。
- **GROUP BY** 把行分成组、每组输出一行；**SELECT 子句**里只能出现**分组属性或聚集函数**。
- **HAVING** 在分组**之后**过滤组，**WHERE** 在分组**之前**过滤行——两者不可互换。
- 完整求值顺序：**FROM → WHERE → GROUP BY → HAVING → SELECT**。
- 空组上 **SUM/AVG** 为 **NULL**、**COUNT** 为 **0**；**DISTINCT** 可与聚集组合。

在下一节，我们面对 SQL 最强大、也最容易写错的一类查询——**嵌套子查询**：当「判断条件」本身需要一次查询的结果（集合成员、存在性、比较）时，**IN、EXISTS、ANY/ALL、比较运算符** 如何在子查询内外建立联系，以及那条贯穿全篇的三值逻辑如何在这些操作符里设下陷阱。
