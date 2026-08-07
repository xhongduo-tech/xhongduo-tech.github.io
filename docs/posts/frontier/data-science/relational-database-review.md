---
title: 关系数据库回顾
date: 2026-08-07
---

# 关系数据库回顾

<div class="epigraph">
<p>SQL 是数据科学家的第一语言——它把「取数据」这件事变成一件可靠、可复现的事。</p>
<footer>—— 对 SQL 在数据生态中地位的一句话概括（数据工程界共识）</footer>
</div>

<div class="article-byline">
<p>第九级 · 数据科学 ｜ 《数据库系统概念》 第1-3章 ｜ 2026-08-07</p>
</div>

## 为什么从数据库回顾开始

进入第三篇「数据工程与系统」，第一站是数据科学的数据地基——**关系数据库**。<span class="marginnote">第三级《数据库》已系统讲过关系模型、SQL 与事务；这里是从数据科学视角的「回顾 + 补位」：重点不在数据库内部实现，而在「数据科学家怎么用 SQL 高效、安全地取数」。若你已熟练掌握 JOIN 与聚合，本节的连接点会让你更快进入数据仓库与数据湖。</span>数据科学的上游是数据，而企业数据 90% 存在关系数据库里——不会 SQL，等于不会取数。

## 1 关系模型：二维表的世界

关系数据库建立在**关系模型**之上：数据组织成一张张**表（relation/table）**，每行是一条记录，每列是一个字段。

关系模型的三个基本概念：

- **表**：同构记录的集合，如 `users` 表存放所有用户。
- **主键（primary key）**：唯一标识一行的一列或列组合——同一张表里主键不可重复、不可为空。
- **外键（foreign key）**：本表中指向另一张表主键的列，用来表达表之间的关联。<span class="marginnote">主键与外键构成了关系模型的「实体-联系」骨架：`orders` 表里的 `user_id` 外键指向 `users` 表的 `user_id` 主键，就表达了一条订单属于一个用户的关系。理解外键，是理解 JOIN 的前提。</span>

**重点：关系模型之所以长盛不衰，在于它的范式化设计消除了冗余。** 一个用户的信息只存一份，订单表只存 `user_id` 而非重复拷贝用户全部信息——修改用户姓名只需改一处。这套「数据只存一份」的纪律，正是数据质量的第一道防线。

## 2 SQL 的核心操作：SELECT 的五件套

SQL 的绝大多数取数需求，都能用 SELECT 语句及其五个子句完成：

```sql
SELECT user_id, SUM(amount) AS total_spend
FROM orders
WHERE created_at >= '2026-01-01'
GROUP BY user_id
HAVING SUM(amount) > 1000
ORDER BY total_spend DESC
LIMIT 10;
```

逐条拆解执行顺序（注意**书写顺序 ≠ 执行顺序**）：

1. **FROM**：先确定从哪张表取。
2. **WHERE**：过滤行（逐行条件），这里过滤出 2026 年后的订单。
3. **GROUP BY**：按用户分组，为聚合做准备。
4. **HAVING**：过滤分组（对聚合结果的条件，不能用 WHERE）。
5. **SELECT / ORDER BY / LIMIT**：选列、排序、限量。

**辨析｜易错点：** WHERE 与 HAVING 的差别是 SQL 新手第一坑。**WHERE 在分组之前过滤行，HAVING 在分组之后过滤组**。想筛「订单金额 > 100 的订单」用 WHERE；想筛「总消费 > 1000 的用户」必须用 HAVING。混用二者，轻则报错，重则结果错误——把行级条件写进 HAVING 不但多余，还会在无 GROUP BY 时语义混乱。

## 3 JOIN：把多张表拼成一张

数据分析几乎总是跨表：用户行为在 `events` 表，用户属性在 `users` 表，要一起分析就得**JOIN**。四种主要连接方式：

| JOIN 类型 | 结果 | 记忆口诀 |
| --- | --- | --- |
| INNER JOIN | 两边都匹配的行 | 交集 |
| LEFT JOIN | 左表全部 + 右表匹配，无匹配补 NULL | 左表说了算 |
| RIGHT JOIN | 右表全部 + 左表匹配 | 右表说了算 |
| FULL OUTER JOIN | 两边全部，各自无匹配补 NULL | 并集 |

```sql
SELECT u.user_id, COUNT(e.event_id) AS event_cnt
FROM users u
LEFT JOIN events e ON u.user_id = e.user_id
GROUP BY u.user_id;
```

**重点：LEFT JOIN 是数据分析里最常用的连接——它保证「左表的每一行都在结果里」。** 查「每个用户的活跃度」时，没产生行为的用户也必须出现（事件数为 0 或 NULL），LEFT JOIN 恰好满足。<span class="marginnote">一个隐蔽的坑：LEFT JOIN 后对右表做 WHERE 过滤，会悄悄把 LEFT 变成 INNER。因为 `WHERE e.event_id IS NOT NULL` 会删掉「无匹配」的行。要「保留左表全部」，过滤条件应放在 ON 子句里而非 WHERE——这是 SQL 里最经典的语义陷阱之一。</span>

## 4 聚合与窗口函数：从行级到群体级

**聚合（aggregation）** 把多行压缩成一行：`COUNT`、`SUM`、`AVG`、`MIN`、`MAX`，常配 `GROUP BY` 使用。聚合是「从明细到汇总」的语法表达，对应特征工程里的「聚合造特征」（第16篇）。

**窗口函数（window functions）** 是聚合的升级：它**不合并行**，而是在每一行旁边附加「它所在组的统计量」。典型用途：

```sql
SELECT
  user_id,
  created_at,
  amount,
  RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rnk
FROM orders;
```

窗口函数在**不丢失行级明细**的前提下给出组内排名、累计值、滑动均值——正是时间序列特征、排行榜、复购分析的主力工具。<span class="marginnote">窗口函数与普通聚合的核心区别：`GROUP BY` 会把组内行折叠成一行，`OVER (PARTITION BY ...)` 保留每一行、同时把组内计算的结果附上来。理解了这个「折叠 vs 附加」的分野，窗口函数就从天书变成顺手工具。</span>

## 5 从数据库到数据科学：取数的三条纪律

数据科学家用数据库时，三条纪律能避免大量返工：

1. **能用 SQL 做的就别搬到 Python**：聚合、过滤在数据库里做（「下推」），只把已经减负的结果拉回内存。几千亿行不可能全量拉进 pandas。
2. **取数即留痕**：记录 SQL 与取数时间，让「这份数据怎么来的」可追溯——元数据纪律（第3篇）的数据库版。
3. **先小样本再全量**：写 SQL 先 `LIMIT` 验正确性，跑通后再放开全量——避免一条错误 SQL 浪费一小时。

**辨析｜易错点：** 「把数据库当 CSV 全量拉出来再在 pandas 里过滤」是新手最常见的低效模式。数据库引擎的索引、并行与统计信息远强于单机 pandas。**让过滤尽量发生在数据库端**，是数据科学工程化的第一课。

## 6 一个完整的取数示例：构造「用户月度活跃度」表

把本节的语法全部串进一个真实任务：**造一张「用户月度活跃度」表，供后续分析用**。

业务定义（第25篇会用到）：月度活跃 = 该月至少有一次行为的用户。

**第一步，行为明细在 `events` 表**。每条记录有 `user_id`、`event_time`、`event_type`。

**第二步，按「年-月」分组计数**：

```sql
SELECT
  user_id,
  DATE_FORMAT(event_time, '%Y-%m') AS month,
  COUNT(DISTINCT event_type) AS type_cnt,
  COUNT(*) AS event_cnt
FROM events
WHERE event_time >= '2026-01-01'
GROUP BY user_id, DATE_FORMAT(event_time, '%Y-%m');
```

注意用 `COUNT(DISTINCT event_type)` 而非 `COUNT(event_type)`——后者只数「非空值」的行，前者才真正数「不同的类型数」。一个用户一天点 100 次同一按钮，`COUNT(*)` 是 100，但「类型数」可能只是 1——**选错聚合函数，指标含义就变了**。

**第三步，定义「活跃」并生成标志列**：

```sql
SELECT
  user_id,
  month,
  event_cnt,
  CASE WHEN event_cnt > 0 THEN 1 ELSE 0 END AS is_active
FROM monthly_events;
```

**第四步，与用户表 LEFT JOIN 补全属性**：把 `users` 表的注册时间、渠道 LEFT JOIN 进来，供后续按渠道下钻（第25篇商业分析的分组透视）。

**辨析｜易错点：** 这个例子最典型的两处错误是：① 用 `COUNT(event_type)` 当「不同类型数」；② 忘了 `WHERE` 的时间过滤导致数据量爆炸。**每一步 SQL 都要问「这个数字的业务含义到底是什么」**——SQL 语法简单，难的是「写出来的数对不对」，而这要靠对业务定义的清晰把握。

**重点：取数的本质是「把业务定义翻译成 SQL」。** 「月度活跃」「近 30 天留存」这些指标，写 SQL 前先写清楚定义（怎么算活跃、算哪段时间），SQL 只是定义的执行。第23篇《数据治理》里的「指标口径统一」，说的就是让这套「定义 → SQL」的翻译在全公司一致。

## 7 小结

- 关系模型以**表**组织数据，**主键**唯一标识、**外键**表达关联，范式化消灭冗余。
- SELECT 五件套执行顺序：**FROM → WHERE → GROUP BY → HAVING → SELECT/ORDER/LIMIT**。
- **WHERE 过滤行、HAVING 过滤组**，二者不可混用。
- JOIN 四种：**INNER/LEFT/RIGHT/FULL**，数据分析最常用 LEFT JOIN，注意 ON 与 WHERE 的语义陷阱。
- **窗口函数**在保留明细的同时附加组内统计，是分析型取数的主力。
- 取数三纪律：**下推、留痕、先小样本**。

在下一节，数据从「关系数据库」走向「分析平台」：企业级的数据存储如何分层——这就是**数据仓库与数据湖**。
