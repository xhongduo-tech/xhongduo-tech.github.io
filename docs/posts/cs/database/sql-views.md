---
title: 视图：定义、查询与更新
date: 2026-08-07
---

# 视图：定义、查询与更新

<div class="epigraph">
<p>所有模型都是错的，但有些是有用的。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从视图开始

前几节我们积累了不少「很长、很常用」的查询——比如「物理系 2009 年秋季学期的开课清单」，写起来要连接两张表、带四个条件。每次用到都重抄一遍，既啰嗦又容易抄错。数据库的回答是：**把这段查询存起来，起个名字，以后像用普通表一样用它**。这个「有名字的查询」就叫**视图（view）**。

但视图又和真正的表有本质区别：**它不占物理存储**，不复制数据，只是把一段查询定义记住。于是出现了一个微妙的问题：既然视图是「查询的别名」，那对它做 INSERT 到底意味着什么？所有视图都能更新吗？这一节把视图的定义、展开、更新三条线讲清楚。它同时是第1章**数据抽象**里「外模式」概念的落地——视图正是用户看到的那一层<span class="marginnote">在第1章《数据视图》里我们讲过三层抽象：物理层、逻辑层、视图层。今天几乎所有数据库的「视图层」都是由 SQL 视图实现的：每个用户看到的是经过裁剪、包装过的逻辑数据，而底层表可以随心调整。</span>。

## 1 视图定义：CREATE VIEW 与虚关系

**视图（view）**：用一个查询表达式定义、但并不实际存储的关系，SQL 中称为**虚关系（virtual relation）**。定义视图的语句是：

```sql
CREATE VIEW faculty AS
    SELECT ID, name, dept_name
    FROM instructor;
```

定义之后，`faculty` 就可以像普通表一样出现在任何查询的 FROM 里。这个视图的典型用途是**隐藏敏感列**：教师关系里有 salary（工资），财务以外的人不该看到它，于是对外只暴露 `ID, name, dept_name` 三列。

更复杂的视图可以是「连接 + 筛选」的产物：

```sql
CREATE VIEW physics_fall_2009 AS
    SELECT course.course_id, sec_id, building, room_number
    FROM course, section
    WHERE course.course_id = section.course_id
      AND course.dept_name = 'Physics'
      AND section.semester = 'Fall'
      AND section.year = 2009;
```

此后任何用户查询物理系秋季课程，都只需 `SELECT * FROM physics_fall_2009`——背后的两张表、四个条件被封装进了名字里。<span class="marginnote">注意 CREATE VIEW 是 DDL 不是 DML：它在系统目录（数据字典）里登记一条「视图定义」，但不触碰任何业务数据。与表不同，视图建立时不会真的执行那条 SELECT。</span>不用了可以 `DROP VIEW physics_fall_2009;`，只删定义、不动底层表。

## 2 视图的查询展开：它是怎么跑起来的

视图本身没有数据，那 `SELECT * FROM faculty` 是怎么返回结果的？答案是**展开（expansion）**：数据库在收到这条查询时，把 FROM 里的视图名替换成它的定义查询，然后对合并后的完整查询执行计划。

$$\text{用户查询}\;\; Q(\text{faculty}) \;\;\longrightarrow\;\; Q\Big(\pi_{\text{ID},\,\text{name},\,\text{dept\_name}}(\text{instructor})\Big)$$

比如用户执行：

```sql
SELECT name FROM faculty WHERE dept_name = 'Music';
```

它先被展开成作用在视图定义上的查询，再经过优化器等价改写，最终等价于直接对底层表的查询：

$$
\pi_{\text{name}}\Big(\sigma_{\text{dept\_name}='Music'}\big(\pi_{\text{ID,name,dept\_name}}(\text{instructor})\big)\Big)
\;=\;
\pi_{\text{name}}\big(\sigma_{\text{dept\_name}='Music'}(\text{instructor})\big)
$$

**辨析｜易错点：** 很多人以为「视图会预先算好结果存起来」。对**普通视图（虚视图）**，这完全错误——它每次查询都现场展开、现场执行，底层数据变了，视图结果立刻跟着变，这恰恰是它「实时」的优点。视图带来的便利是**逻辑层面**的封装，不是物理层面的加速。真正「预先算好、存下来」的另有其物，见第 5 节的物化视图。

把这一条和查询优化联系起来：展开后的查询交给优化器，等价改写（如把 $\sigma$ 下推到 $\pi$ 之下）与代价估算正是第12章《查询优化》的主战场——视图在这里不过是「查询文本的自动替换」，替换完之后，优化器一视同仁。

## 3 视图的更新：哪些能写，哪些不能写

视图能查，但**不一定能改**。直觉上这是对的：对 `dept_tot_salary` 这种「按系汇总工资」的视图，用户执行 `INSERT` 要插进哪个底层元组？语义根本不存在。Silberschatz 给出了**视图可更新（updatable）**的充分条件：

- FROM 子句**只含一个关系**；
- SELECT 子句**只含该关系的属性名**，不含表达式、聚集函数、DISTINCT；
- **没有 GROUP BY 与 HAVING**。

于是下面这个视图满足条件：

```sql
CREATE VIEW instructor_info AS
    SELECT ID, name, dept_name
    FROM instructor
    WHERE dept_name = 'Music';
```

对它做插入是合法的——系统把插入翻译到 `instructor` 表上，未出现在视图里的列（如 salary）填默认值或 NULL：

```sql
INSERT INTO instructor_info VALUES ('69987', 'White', 'Music');
```

这条插入会在 `instructor` 里生成一个音乐系的教师元组，它恰好落在视图的范围内，因此能在这个视图里被看见。

**但这里有个隐藏的语义漏洞。** 若插入的是：

```sql
INSERT INTO instructor_info VALUES ('69987', 'White', 'Finance');
```

系统同样会插入到 `instructor`——生成的元组 `dept_name='Finance'` **不满足视图的 WHERE 条件**，于是在这个视图里永远看不到自己。用户插了一行「合法的教师」，却在自己的视图里查不到，会造成严重困惑。

**WITH CHECK OPTION** 正是堵这个洞的：定义视图时带上它，系统就会强制校验「插入 / 更新后的元组必须仍满足视图条件」：

```sql
CREATE VIEW instructor_info AS
    SELECT ID, name, dept_name
    FROM instructor
    WHERE dept_name = 'Music'
    WITH CHECK OPTION;
```

此后 `INSERT ... ('Finance')` 会被**整条拒绝**，因为新元组不符合视图的 WHERE 条件。

**辨析｜易错点：** 不满足可更新条件的视图，对其 UPDATE / DELETE 在标准里同样受限。对聚集视图（GROUP BY）、连接视图（多个 FROM 关系）做更新，多数系统直接报错，个别系统则按「尽力而为」翻译，语义难以预期。**工程铁律：别指望视图层替你写数据，视图是读的抽象，写请落到真正的表上。**

## 4 公式解析：视图的可更新性判定

把上一节的判定条件翻译成一组「如果 … 就 …」的判定链，判断一个视图能不能安全更新：

$$
\text{可更新}
\;=\;
\underbrace{\big(|\text{FROM}| = 1\big)}_{\text{单表}}
\;\wedge\;
\underbrace{\big(\text{SELECT 只含属性名}\big)}_{\text{无表达式 / 聚集 / DISTINCT}}
\;\wedge\;
\underbrace{\big(\text{无 GROUP BY, HAVING}\big)}_{\text{无分组筛选}}
$$

逐项拆解每条规则的动机：

- **单表（$|\text{FROM}|=1$）**：插入视图的一行必须能翻译成某个底层关系的一行。若视图来自两张表的连接，新行该拆成两个元组塞进两张表？拆分没有唯一答案，系统无从下手。
- **SELECT 只含属性名**：视图若含 `salary * 1.1` 这类表达式，反向翻译时无法从新值反推旧值（乘 1.1 是单射，但 `SUM`、`AVG`、`COUNT` 这类聚集完全不可逆）；含 DISTINCT 意味着「视图行 ≠ 底层行」，也无法回写。
- **无 GROUP BY / HAVING**：分组后每一行代表一组元组的汇总，不再是任何单个底层元组，回写无意义。

这三条不是繁琐的教条，而是同一个原则的三次现身：**可更新 ⇔ 视图中的每一行都能无歧义地对应到底层的一个真实元组**。反过来，只要某处破坏了「一一对应」，更新就失去了翻译基础。这也是为什么把上面判定式里的任意一条改为假，视图就会从「可更新」翻转为「只读」。

## 5 物化视图与视图的真正价值

普通视图每次查询都现场计算。若底层数据变化不频繁、而查询结果被反复使用，更好的策略是**把视图结果真的存下来**——这就是**物化视图（materialized view）**。它建立时执行一次查询把结果落盘，此后需要**维护**：底层表每次增删改，视图的存储副本都要同步更新（或按约定定时刷新）。物化视图换来了查询时的快，付出的是存储开销与维护复杂度，它是数据库里「用空间换时间」的典型——第12章我们还会见到它在查询优化中充当「预计算缓存」的角色。

最后给视图的存在意义一个收束。视图真正解决的问题，是**让不同角色看到不同的数据形状**：

- 隐藏敏感列（`faculty` 不暴露 salary），是安全与授权的轻量实现，第4章《授权》会从权限模型角度再讲一遍。
- 把复杂连接封装成「假表」，让不熟悉表结构的业务方直接使用，降低使用门槛。
- 作为外模式隔离底层 schema 变更：底层表拆了、加了列，只要视图定义不变，应用代码一行不用改——这正是数据抽象与逻辑独立性的价值。

**辨析｜易错点：** 视图能「简化查询」，但不能「加速查询」。一段慢查询包成视图后依旧慢；相反，视图展开可能让优化器看不全全局信息，个别场景还更慢。想要性能，靠索引与优化器，不靠视图。

## 6 小结

- **视图**：由查询定义、不实际存储的虚关系，`CREATE VIEW v AS <查询>` 创建，`DROP VIEW` 删除。
- **查询展开**：使用视图的查询会把视图名替换为定义，展开后的查询交给优化器；普通视图**每次查询现场执行**，实时反映底层数据。
- **可更新视图**需满足：FROM 单表、SELECT 只含属性名、无 GROUP BY / HAVING——本质是「视图行与底层元组一一对应」。
- **WITH CHECK OPTION** 强制「插入 / 更新后的元组仍满足视图条件」，堵住「插入却看不见」的漏洞。
- **物化视图**把结果存下来换取查询速度，代价是存储与维护同步；它区别于虚视图的关键是「存没存」。
- 视图是外模式与逻辑独立性的实现载体，也承担隐藏敏感信息的安全职责，但它不加速查询。

在下一节，我们将进入 SQL 的另一个支柱——**事务的 SQL 语义**：把多条语句打包成原子单元，理解提交与回滚，以及为什么「连接要趁早、提交要果断」。
