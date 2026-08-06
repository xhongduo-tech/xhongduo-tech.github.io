---
title: 单关系查询：SELECT 基本结构
date: 2026-08-07
---

# 单关系查询：SELECT 基本结构

<div class="epigraph">
<p>用户应当能够描述自己要什么，而不必规定它如何被计算。</p>
<footer>—— 埃德加 · 科德（E. F. Codd），《大型共享数据银行的关系模型》（1970）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.3 ｜ 2026-08-07</p>
</div>

## 为什么从单关系查询开始

上一节我们用 DDL 把 `instructor` 表真正「造」了出来。现在到了最激动人心的一刻：**从这张表里问问题**。单关系查询是所有 SQL 的地基——多表连接、子查询、聚合，全都建立在「怎么对一张表做投影与选择」之上。把这三条子句吃透，后面几百个复杂查询都只是它的组合。

而单关系查询最值得深挖的，是它与关系代数的**第一次分歧**。上一节预告过：**关系代数 = 集合（去重、无序），SQL = 多重集（默认不去重）**。这一节你会亲眼看到这条裂缝如何在 `SELECT` 里暴露，以及 SQL 用什么关键字缝补它。<span class="marginnote">科德 1970 年那句「描述要什么，不规定怎么算」，正是 SQL 查询子句的哲学：`SELECT-FROM-WHERE` 是声明式的，求值顺序由系统决定——这与关系代数「一步步操作」的过程式风格形成鲜明对照。</span>

## 1 SELECT-FROM-WHERE 三件套

一条最简单的单关系查询长这样：

```sql
SELECT name, salary
FROM   instructor
WHERE  dept_name = 'CS';
```

三个子句，三个职责，与关系代数逐词对应：

- `FROM instructor`：指定输入关系——「要对哪张表操作」。
- `WHERE dept_name = 'CS'`：**选择（σ）**——筛行。只保留满足条件的行。
- `SELECT name, salary`：**投影（Π）**——选列。只输出列出的属性。

**辨析｜易错点：`SELECT` 是误称。** 名字叫「选择」，干的却是投影（选列）；真正的「选择」（选行）在 `WHERE` 里。这是初学者第一大坑——把两者都叫「选」，就分不清行筛选与列筛选了。<span class="marginnote">历史的误会：SQL 的 `SELECT` 沿用自 SEQUEL（Structured English QUEry Language），当时用它泛指「取数据」。到今天它已被全世界接受，但想真正理解 SQL，必须记住「SELECT = Π，WHERE = σ」这层解剖图。</span>

## 2 WHERE 子句的谓词

`WHERE` 接受一个**布尔表达式**，逐行判断真假，为真的行留下。可用的比较与组合：

- **比较运算**：`=`, `<>`（不等于）, `<`, `>`, `<=`, `>=`。
- **逻辑组合**：`AND`, `OR`, `NOT`，配合括号改变优先级。
- **区间判断**：`BETWEEN x AND y`，等价于 `>= x AND <= y`。
- **字符串匹配**：`LIKE` 配合通配符 `%`（任意串）与 `_`（单个字符）。
- **空值判断**：`IS NULL` / `IS NOT NULL`。

```sql
SELECT name
FROM   instructor
WHERE  dept_name = 'CS' AND salary > 70000;
```

**重点：`LIKE` 用 `%` 与 `_`，不是正则表达式。** `LIKE 'CS%'` 匹配「CS 开头」；`LIKE '_CS'` 匹配「任意一个字符 + CS」；要匹配字面的 `%` 需用 `ESCAPE` 转义。不同数据库对大小写敏感度不一——MySQL 默认不敏感，PostgreSQL 默认敏感，这是常见的跨库移植坑。

## 3 SELECT 子句：表达式与 DISTINCT

`SELECT` 后面不只能写属性名，还能写**表达式**：

```sql
SELECT name, salary * 1.1
FROM   instructor
WHERE  dept_name = 'CS';
```

结果第二列是 `salary * 1.1`，一个计算出来的值。`SELECT *` 表示「所有列」；`SELECT DISTINCT` 则显式去重：

```sql
SELECT DISTINCT dept_name
FROM   instructor;
```

不加 `DISTINCT`，同一系出现几次就返回几行（多重集）；加上后，结果才是严格集合——这正是关系代数 $\Pi$ 的行为。

**辨析｜易错点：默认不去重是「特性」，不是「bug」。** 对「每个 CS 教师各拿多少工资」这类查询，保留重复行是语义正确的；只有对「有哪些系」这类集合式提问，才需要 `DISTINCT`。**代价视角**：去重需要排序或哈希，很贵——SQL 把去重的开关交给用户，是对性能的务实让步，第12章会把它量化。<span class="marginnote">从第一级《集合》的互异性到这里：数学集合天生无重复，SQL 表默认允许多重集。SQL 的选择是「默认性能优先、需要时再付去重费」，这条 trade-off 会反复出现在整个数据库专题。</span>

## 4 ORDER BY：输出层的排序

关系的元组**天生无序**，但人看结果时需要顺序。`ORDER BY` 就是那个「输出层」的出口：

```sql
SELECT name, salary
FROM   instructor
WHERE  dept_name = 'CS'
ORDER BY salary DESC;
```

`ASC`（默认）升序、`DESC` 降序；多个关键字按字典序依次排序：`ORDER BY dept_name ASC, salary DESC` 先按系升序，同系内按工资降序。

**辨析｜易错点：`ORDER BY` 不改关系，只改输出。** 关系代数里没有排序——因为关系是集合。`ORDER BY` 是 SQL 为「人机交互」增加的附加特性，它发生在查询的最后一步，也不影响后续运算（子查询里 `ORDER BY` 基本无意义）。把「排序是输出层的修饰」刻在脑子里，就不会写出「先排序再筛选」的无效代码。

## 5 公式解析：一条单关系查询的完整求值

把一条查询拆成「系统实际执行的步骤」，是理解 SQL 语义的钥匙。考虑

```sql
SELECT name, salary * 1.1
FROM   instructor
WHERE  dept_name = 'CS' AND salary > 70000
ORDER BY salary DESC;
```

其求值顺序是固定的四步：

- **第一步，FROM**：取输入关系 `instructor` 的全部元组。
- **第二步，WHERE**：逐行判断 `dept_name = 'CS' AND salary > 70000`，为真的行进入下一步。这一步等价于关系代数的选择
  $$\sigma_{dept\_name = \text{'CS'} \wedge salary > 70000}(instructor)$$
- **第三步，SELECT**：对留下的行，逐行计算 `salary * 1.1`，只输出 `name` 与结果列。等价于广义投影
  $$\Pi_{name,\ salary \times 1.1}\big(\sigma_{dept\_name = \text{'CS'} \wedge salary > 70000}(instructor)\big)$$
- **第四步，ORDER BY**：对输出结果按 `salary DESC` 排序。

**重点：SQL 的求值顺序是「先 FROM，再 WHERE，后 SELECT」**——这解释了一个经典困惑：为什么 `WHERE` 里不能用 `SELECT` 里起的别名？因为 `WHERE` 在第3步才出现的别名**还不存在**。同理，第3章后面讲 `HAVING` 与 `GROUP BY` 时，这条顺序会继续支配你的判断。

再补一个微妙的推导：比较两个都用了聚合的系平均工资，「找平均工资高于全系平均的系」这类查询为何必须子查询？因为 `WHERE` 在 `SELECT` 之前执行，而聚合（第3章第5节）发生在更后面——**单关系查询的单层结构根本装不下「先聚合再比较」**。这种结构性局限，正是嵌套子查询存在的理由。

## 6 易错辨析：单关系查询的四个坑

- **`NULL` 不能比较**：`salary > 70000` 遇到 `salary IS NULL` 时，结果为**未知（unknown）**，该行被 `WHERE` 排除。判断空值必须用 `IS NULL`，写 `= NULL` 永远不成立——三值逻辑第3章《空值与三值逻辑》专节展开。<span class="marginnote">记住：`NULL = NULL` 的结果是「未知」，不是「真」。这是 SQL 里违反直觉、却必须接受的第一个事实。</span>
- **字符串字面量用单引号**：`'CS'`，不是双引号。双引号在多数数据库里表示标识符，混用会报错。
- **`<>` 才是「不等于」**：有些方言也认 `!=`，但标准写法是 `<>`。别在面试里写 `SELECT ... WHERE x != NULL`——既错在 `!=`，更错在 `NULL`。
- **`BETWEEN` 是闭区间**：`salary BETWEEN 70000 AND 80000` 含两端。若想开区间，老老实实写 `> 70000 AND < 80000`。

## 7 小结

- 单关系查询三件套：`FROM`（输入）、`WHERE`（选择 σ）、`SELECT`（投影 Π）；`SELECT` 是误称，真正的行筛选在 `WHERE`。
- 谓词支持比较、`AND/OR/NOT`、`BETWEEN`、`LIKE`（`%`/`_`）、`IS NULL`。
- SQL 默认**不去重**（多重集语义），`SELECT DISTINCT` 才回到集合语义；`SELECT` 后可写算术表达式。
- `ORDER BY` 只作用于**输出层**，不改变关系本身；多关键字按字典序排序。
- 求值顺序：**FROM → WHERE → SELECT → ORDER BY**——这解释了为什么 `WHERE` 不能用 `SELECT` 的别名。
- 空值判断必须 `IS NULL`；`NULL` 参与比较产生「未知」，与真假并列为三值。

在下一节，我们进入多关系查询：把两张甚至更多表连起来——**连接与笛卡儿积**在 SQL 里怎么写、`NATURAL JOIN` 与 `ON` 怎么选，以及那条「先积后选」的代数推导如何变成一行优雅的 `JOIN`。
