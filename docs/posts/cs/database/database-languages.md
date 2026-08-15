---
title: 数据库语言：DDL 与 DML
date: 2026-08-07
---

# 数据库语言：DDL 与 DML

<div class="epigraph">
<p>我的语言的边界，就是我世界的边界。</p>
<footer>—— 路德维希 · 维特根斯坦（Ludwig Wittgenstein）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》§1.4 ｜ 2026-08-07</p>
</div>

## 为什么从数据库语言开始

上一节我们有了数据的"视图"：物理层、逻辑层、视图层。但用户怎么把"我想要什么结构"告诉 DBMS？怎么增删改查数据？这就要靠数据库语言。几乎市面上所有主流数据库都讲同一种"方言"——SQL。这一节先厘清数据库语言的两大板块：**数据定义语言（DDL）** 与**数据操纵语言（DML）**，为第 3 章真正动手写 SQL 铺路。

## 1 数据定义语言（DDL）

**数据定义语言（data-definition language, DDL）**：用于定义数据库模式的语言。它回答的问题是"数据库长什么样"。

例如，要定义一张教师表，可以写这样的 SQL 语句：

```sql
CREATE TABLE instructor (
    ID        varchar(5),
    name      varchar(20) NOT NULL,
    dept_name varchar(20),
    salary    numeric(8,2),
    PRIMARY KEY (ID)
);
```

这条语句做了三件事：声明每个属性的名字与类型；声明 name 不能为空（NOT NULL）；声明 ID 是主码。DDL 不只是建表，它还能定义索引、完整性约束、视图等。<span class="marginnote">主码（primary key）用最小的属性集合唯一区分每一行，是下一章《关系模型》的核心概念，这里先混个脸熟。完整性约束也会在第 4 章《完整性约束》专门展开。</span>

DDL 语句交给 **DDL 编译器** 处理后，会把模式信息写入**数据字典（data dictionary）**——一张专门存放"关于数据的数据"（元数据）的表。<span class="marginnote">数据字典也叫系统目录（system catalog）：它自己也是表，记录着有哪些表、每张表有哪些列、主码是谁、有哪些索引。查询优化器全靠它来估算代价，第 12 章《查询优化》会用到。</span>

**辨析｜易错点：ALTER 与 DROP 同样是 DDL。** 初学者常把 DDL 只理解成 CREATE，其实修改与删除模式也是 DDL：ALTER TABLE 给表加一列，DROP TABLE 把整张表连数据一起删掉。理解了"DDL 动的是模式"，就会明白为什么 DROP TABLE 如此危险——它删的不是几行数据，而是骨架本身。

## 2 数据操纵语言（DML）

**数据操纵语言（data-manipulation language, DML）**：用于对数据进行查询与增删改的语言。它回答的问题是"数据库里有什么、我该怎么动它"。

SQL 的 DML 四件套长这样：

```sql
INSERT INTO instructor VALUES ('10211', 'Smith', 'CS', 80000);
SELECT * FROM instructor WHERE dept_name = 'CS';
UPDATE instructor SET salary = 90000 WHERE ID = '10211';
DELETE FROM instructor WHERE ID = '10211';
```

其中**查询（query）** 指检索信息的语句，是 DML 里信息检索的部分。数据库理论按"用户要说多细"把 DML 分成两类，这是理解整个数据库体系的分水岭：

**过程式（procedural）DML**：用户不但说"要什么"，还要说"怎么取"。关系代数就是过程式语言的代表——先做选择，再做投影，一步步拼出结果。
**声明式（declarative / nonprocedural）DML**：用户只说"要什么"，由系统决定"怎么取"。SQL 属于此类。<span class="marginnote">声明式的威力在于"怎么取"由查询优化器决定——这正是数据库能把性能做好的关键，也是本专题第 11、12 章《查询处理与优化》的主题。用户声明意图，系统负责手艺。</span>

同一个问题，两种 DML 的写法差异很直观。查"CS 系教师的姓名"，过程式的关系代数要自己编排步骤：

$$\Pi_{\text{name}}\big(\sigma_{\text{dept\_name} = \text{'CS'}}(\text{instructor})\big)$$

先选择、再投影，步骤写死在表达式里；而声明式的 SQL 只描述结果：

```sql
SELECT name
FROM instructor
WHERE dept_name = 'CS';
```

**"怎么取"的步骤去哪了？被优化器接管了。** 这是声明式语言的根本交易：用户放弃对执行细节的控制，换取更简单的表达与系统级优化的空间。一个数据库系统能不能把同样的 SQL 跑得足够快，全看第 12 章要讲的查询优化器。

## 3 辨析｜易错点：DDL 与 DML 的分工

很多初学者把"SQL"当成一个浑然一体的东西，其实它内部有明确分工：

- DDL 操作的是**模式**（结构、骨架）；DML 操作的是**实例**（数据、血肉）。
- DDL 的典型动词是 CREATE/ALTER/DROP；DML 的典型动词是 SELECT/INSERT/UPDATE/DELETE。
- 执行 DDL 会修改数据字典；执行 DML 只修改用户数据本身。
- 把两者混在一起想，就理解不了"为什么 DROP TABLE 比 DELETE 危险得多"——后者只删数据，前者连模式带数据一起没了。

另外要澄清一个易错点：**SQL 并不止 DDL 与 DML 两件套**。它还包括**数据控制语言（DCL）**，比如 GRANT 与 REVOKE 用于授权——这对应第 4 章《授权：权限、角色与收回》；以及事务控制语句 COMMIT / ROLLBACK。把 SQL 想成"DDL + DML"会漏掉它作为一门完整语言的其他部分。

## 4 公式解析：一条 SELECT 的"语义执行顺序"

SQL 是声明式的：**你书写的顺序 ≠ 执行的顺序**。理解这一点，是写对复杂查询的前提。考虑这条查询：

```sql
SELECT dept_name, AVG(salary) AS avg_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > 80000;
```

把它的语义按"逻辑执行顺序"拆成五步：

- **第一步 FROM**：先确定数据来源——扫描 instructor 这张表。
- **第二步 WHERE**：逐行过滤，丢掉不满足条件的行。<span class="marginnote">WHERE 里不能直接用聚集函数（如 AVG），因为这一步发生在"分组"之前，此时还没有组的概念。新手在此处报错是第 3 章最常见的疑问之一。</span>
- **第三步 GROUP BY**：把剩余的行按系别分组，一组输出一行。
- **第四步 HAVING**：对分组后的每组做过滤，只保留平均工资高于 8 万的组。HAVING 可以出现聚集函数，因为它发生在分组之后。
- **第五步 SELECT**：最后才投影出要的列，并为平均工资起别名 avg_salary。

**写 SELECT 时从"结果想要什么"出发，理解它时从"数据怎么流动"出发。** 这个逻辑顺序（FROM → WHERE → GROUP BY → HAVING → SELECT）比书写顺序更能解释 WHERE 与 HAVING 的差别，也会在第 3 章《聚集函数与分组》被再次严格化。它背后的集合操作，与第一级《集合》里"先筛元素、再分组聚合"的思路完全同构。

## 5 数据库语言的数值算例与术语速查

**把「过程式 vs 声明式」用同一个查询的两种写法放大。** 查「CS 系教师的姓名」。

- **过程式（关系代数）**：$\Pi_{\text{name}}(\sigma_{\text{dept\_name='CS'}}(\text{instructor}))$——执行步骤写死在表达式里：先扫全表、再过滤、再投影。**用户指定步骤**。
- **声明式（SQL）**：`SELECT name FROM instructor WHERE dept_name = 'CS'`——只说结果，**步骤交给优化器**。
- **收益差异**：若表有 1 亿行、CS 系 1 万行——过程式写死「先扫再滤」；声明式让优化器可能「先走 dept_name 索引再投影」——**同样结果，执行快百倍**。
- **结论**：声明式的价值不在「写法省字」，而在「**把优化空间留给系统**」——这是第 11、12 章全部优化的前提。

**数值算例：WHERE 与 HAVING 的分工** 设按系分组查平均工资。

- `WHERE` 在分组前过滤行；`HAVING` 在分组后过滤组——**`WHERE` 不能写 AVG()（此时无组），`HAVING` 可以**。
- 例：`SELECT dept_name FROM instructor GROUP BY dept_name HAVING AVG(salary) > 80000`——HAVING 里的 AVG 合法。
- **执行顺序**：FROM → WHERE → GROUP BY → HAVING → SELECT——**记住这个顺序，WHERE/HAVING 的差别自动解开**。

**辨析｜易错点：** 「SQL 是声明式」不意味着「没有执行顺序」——**它没有「书写执行顺序」，但有「逻辑语义顺序」**。书写顺序（SELECT 在前）与逻辑顺序（FROM 在前）相反，这是新手最容易搞混的。**理解逻辑顺序，就理解了 WHERE/HAVING、别名可用范围、聚集函数位置的规则**。

<span class="marginnote">把「声明式」与第一级《数理逻辑》的「命题描述」对照：<strong>声明式查询 = 用逻辑公式描述结果集合，优化器 = 把公式翻译成高效求值程序</strong>——SQL 的 WHERE 子句本质是逻辑谓词。这也是第 6 章「关系演算」（逻辑语言描述查询）的主题，那里你会看到 SQL 的完整逻辑根基。</span>

### 术语速查

| 术语 | 含义 |
| --- | --- |
| DDL | 数据定义语言（模式） |
| DML | 数据操纵语言（数据） |
| DCL | 数据控制语言（授权） |
| 过程式 DML | 用户指定执行步骤 |
| 声明式 DML | 用户描述结果 |
| 逻辑执行顺序 | FROM→WHERE→GROUP BY→HAVING→SELECT |

## 6 小结

- 数据库语言分两大板块：**DDL** 定义模式，**DML** 操纵数据。
- DDL 的输出进入**数据字典**；DML 分为**过程式**与**声明式**，SQL 属于声明式。
- SQL 不止 DDL 与 DML，还含 **DCL**（授权）与事务控制语句。
- 一条 SELECT 的逻辑执行顺序是 **FROM → WHERE → GROUP BY → HAVING → SELECT**，与书写顺序不同。
- 声明式的代价与回报：用户放弃"怎么取"的控制，换取简洁表达与优化器带来的性能空间。
- DDL 编译器把模式写入**数据字典**，这是"元数据"的第一次正式登场，后续章节会反复用到。

在下一节，我们将把目光从"语言"转向"数据结构"——看看**关系数据库**到底怎么用"表"承载这一切，以及设计一张好表要考虑什么。
