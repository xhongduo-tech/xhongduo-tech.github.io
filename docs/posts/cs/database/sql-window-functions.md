---
title: 高级聚集：窗口函数、排名与分桶
date: 2026-08-07
---

# 高级聚集：窗口函数、排名与分桶

<div class="epigraph">
<p>数据本身并不说话，是我们向它提问的方式塑造了答案。</p>
<footer>—— 汉斯 · 罗斯林（Hans Rosling，数据可视化先驱）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第5章 高级 SQL ｜ 2026-08-07</p>
</div>

## 为什么从窗口函数开始

前面学过的 `GROUP BY` 聚合有一个先天限制：**一旦分组，每一行就被折叠成一个组级结果**，你想「既要每一行的明细，又要它所在组的统计值」，普通聚合就做不到了。窗口函数（window function）正是为这个问题而生的：它在不折叠行的前提下，对每一行计算其「窗口」内的聚合值。在 LeetCode 的数据库题里，超过一半的高频题（连续登录、分组 TopN、同比环比）都要靠窗口函数，它因此被称为 SQL 的「分水岭」——会与不会，直接决定你处理报表型查询的能力。本篇补齐第 5 章高级 SQL 的最后一块核心拼图。

## 1 窗口函数的基本形态

窗口函数在 `SELECT` 子句中调用，`OVER` 子句划出**窗口（window）**：行的一个子集，函数在这个子集上计算，但每一行仍保留自己的身份。

```sql
SELECT ID, name, dept_name, salary,
       AVG(salary) OVER (PARTITION BY dept_name) AS dept_avg
FROM instructor;
```

这里的 `OVER` 子句：`PARTITION BY dept_name` 把行按系别分成若干「分区」，`AVG(salary)` 在每个分区内单独求均值，结果作为新列拼回到每一行上。<span class="marginnote">对比普通聚合：`GROUP BY` 输出的是每个系一行；窗口函数输出的行数与输入完全一致。窗口是在「保留明细」之上的分组计算。</span>

**窗口函数区别于普通聚合函数的核心：函数值所在的行仍然保留，且可以同时看到它前后相邻的行。** 窗口内没有 `GROUP BY` 的折叠语义，但可以有排序、有边界。

## 2 排名函数：ROW_NUMBER、RANK 与 DENSE_RANK

排名是窗口函数最经典的应用。三种排名函数都按 `ORDER BY` 对分区内行排序并编号，区别在于**并列（tie）的处理**：

| 函数 | 并列时 | 下一个排名 | 示例（分数 100, 90, 90, 80） |
| --- | --- | --- | --- |
| `ROW_NUMBER()` | 强制区分，乱序编号 | 顺延 1 | 1, 2, 3, 4 |
| `RANK()` | 并列同号 | 跳过 | 1, 2, 2, 4 |
| `DENSE_RANK()` | 并列同号 | 不跳过 | 1, 2, 2, 3 |

用 `RANK() OVER (ORDER BY salary DESC)` 给每位教师按薪资排名，再套一层子查询，就能拿到「每个系薪资前三」——这是面试高频的「分组 TopN」问题。

**辨析｜易错点：** `ORDER BY` 在窗口内排序的默认边界是「到当前行为止」，即累计窗口；不写 `ORDER BY` 时窗口才是整个分区。这个区别直接决定 `SUM()` 是「累计和」还是「总和」。

## 3 滑动窗口与聚合窗口函数

除了排名，任何聚合函数（`COUNT`、`SUM`、`AVG`、`MIN`、`MAX`）都可以放进 `OVER`，再配合**窗口框（frame）**指定边界：

```sql
SELECT date, sales,
       AVG(sales) OVER (ORDER BY date
                        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7
FROM daily_sales;
```

`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` 把窗口框成「当前行及之前 6 行」。`ROWS` 按物理行数框定，`RANGE` 按值框定（如 `RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW`）。<span class="marginnote">滚动窗口是金融、零售报表的基础：移动平均、滚动求和、7 日留存都建立在这条语法上。</span>

## 4 分桶：NTILE 与百分位

`NTILE(n)` 把分区内有序的行**尽量均匀地分成 n 桶**，返回每行所属的桶号 1..n。这是「分位数」的直接实现：`NTILE(4)` 相当于四分位，`NTILE(100)` 相当于百分位。

```sql
SELECT ID, name, salary,
       NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM instructor;
```

**辨析｜易错点：** 分桶与排名不同——排名关心「第几名」，分桶关心「属于哪一档」。`NTILE(n)` 在不能整除时前若干桶多一行，行为与 `CUME_DIST()`、`PERCENT_RANK()` 这类分布函数相互补充。

## 5 公式解析：累计比例与移动平均的数学本质

窗口函数背后的数学是一次「有序前缀聚合」。设第 $i$ 行为 $v_i$，则：

$$
\text{CUM}_i = \sum_{k=1}^{i} v_k, \qquad
\text{MA}_i^{(m)} = \frac{1}{m}\sum_{k=i-m+1}^{i} v_k
$$

- **第一步，认清求和范围**：$\text{CUM}_i$ 的下标从 1 到 $i$，这正是 `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`（累计窗口）的语义——从分区开头到当前行。
- **第二步，理解窗口大小**：$\text{MA}_i^{(m)}$ 只取最近 $m$ 项，即 `ROWS BETWEEN (m-1) PRECEDING AND CURRENT ROW` 的窗口框；$m$ 越大曲线越平滑、滞后越大。
- **第三步，看到 O(n) 结构**：两种指标都可以在**一趟顺序扫描**内维护——这就是窗口函数效率上优于「自连接 + 分组」的根本原因。

## 6 窗口函数与普通聚合的对照

**窗口函数不是普通聚合的替代品，而是它无法表达的查询的补充。** 核心对照：

| 维度 | 普通 `GROUP BY` 聚合 | 窗口函数 |
| --- | --- | --- |
| 输出行数 | 每个分组一行 | 每行一行（不折叠） |
| 可见范围 | 组内全体 | 由 `OVER` 指定的窗口 |
| 可访问相邻行 | 否 | 是（前后 N 行） |
| 与明细共存 | 需 `JOIN` 回去 | 直接并列输出 |
| 典型场景 | 汇总报表 | 排名、滚动值、行内占比 |

**辨析｜易错点：** 窗口函数**不能**出现在 `WHERE`、`GROUP BY`、`HAVING` 中，只能出现在 `SELECT` 与 `ORDER BY` 里——因为窗口的划分依赖于分组与过滤都完成之后的中间结果。要想过滤窗口计算的结果，必须套一层子查询。

## 7 窗口函数的数值算例与术语速查

**把「分组 TopN」的三种写法算一遍，理解窗口函数为什么是正解。** 设查询「每个系薪资前三的教师」。

- **写法 A（相关子查询 + 计数）**：对每个教师，数「同系且薪资 ≥ 自己」的人数 ≤ 3——每行一个相关子查询，复杂度 $O(n^2)$，大表慢。
- **写法 B（窗口函数）**：`SELECT * FROM (SELECT *, RANK() OVER (PARTITION BY dept_name ORDER BY salary DESC) rk FROM instructor) t WHERE rk <= 3`——一趟排序 + 一趟过滤，**快一个量级**。
- **写法 C（GROUP BY + 自连接）**：先 GROUP BY 找各系 Top3 薪资，再连接回来——逻辑绕、易错。
- **结论**：窗口函数让「分组内排名」从「绕路的 SQL」变成「直观的一行」——**这是它成为面试高频的原因**。

**数值算例：移动平均的窗口大小** 设日销售额，算 7 日与 30 日移动平均。

- 窗口大小 $m$ 越大，曲线越平滑、对拐点越迟钝（滞后 $m/2$ 天）。
- 7 日 MA 适合「周内波动」的平滑；30 日 MA 适合「月度趋势」的观察。
- 实现上两种都是一趟前缀维护——**窗口函数让「滚动统计」从「逐日自连接」变成「一条 SQL」**。

**辨析｜易错点：** 窗口函数的 `ORDER BY` 与 `PARTITION BY` 缺一不可的误解——`PARTITION BY` 可省（全表一个窗口），但**没有 `ORDER BY` 时默认窗口是整个分区**，聚合函数给出的是「分区总和」而非「累计」。**要「累计」必须写 `ORDER BY`（默认边界到当前行）**——这是最隐蔽的坑。

<span class="marginnote">窗口函数与第 11 章「排序 vs 哈希」执行模型呼应：<strong>窗口计算依赖「分区 + 排序」——数据库用排序实现 PARTITION BY + ORDER BY，用哈希实现无排序窗口</strong>。理解执行层怎么算窗口，就不会对「为什么窗口查询有时走 Sort」感到意外。</span>

### 术语速查

| 术语 | 含义 |
| --- | --- |
| OVER | 划出窗口的子句 |
| PARTITION BY | 窗口分区（类 GROUP BY） |
| ROW_NUMBER | 强制编号 |
| RANK / DENSE_RANK | 并列同号 / 跳过 vs 不跳 |
| 窗口框 | ROWS/RANGE 边界 |
| NTILE | 分桶/分位 |

## 8 小结

- 窗口函数在**不折叠行**的前提下按 `OVER` 划分的窗口做计算，是「明细 + 组统计」的唯一直接写法。
- 三种排名函数 `ROW_NUMBER()`、`RANK()`、`DENSE_RANK()` 的差别只在**并列处理**。
- 聚合函数 + `ROWS` 窗口框实现**滚动聚合**（移动平均、累计和）。
- `NTILE(n)` 实现分桶/分位；窗口计算本质是**有序前缀聚合**，一趟扫描即可完成。
- 窗口函数只能出现在 `SELECT` 与 `ORDER BY`，过滤结果必须套子查询。

在下一节，我们将从窗口函数走向**多维聚合**——`ROLLUP`、`CUBE` 与数据立方体，回答「如何一次算出所有粒度的汇总」。
