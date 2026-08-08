---
title: SQL 查询语言概览与数据定义（DDL）
date: 2026-08-07
---

# SQL 查询语言概览与数据定义（DDL）

<div class="epigraph">
<p>标准的好处在于，它们给了你太多可供选择的东西。</p>
<footer>—— 安德鲁 · 塔嫩鲍姆（Andrew S. Tanenbaum）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 §3.1–3.2 ｜ 2026-08-07</p>
</div>

## 为什么从 SQL 的全貌开始

关系代数是干净的数学，但现实世界没有人在终端里敲 $\sigma$ 和 $\Pi$。这门真正的商业语言，就是 **SQL（Structured Query Language）**。前两节我们把关系代数打磨成了「底层语法图」，现在开始图与真实语言的对译——这一步要写好几篇。

但先别急着写第一条查询。这一节做两件事：**第一，给 SQL 画一张全貌图**——它不止「查询」，而是由数据定义、数据操纵、完整性、视图、事务、授权等好几块拼成的一门完整语言；**第二，吃透其中最先出场的 DDL（数据定义语言）**——用 `CREATE TABLE` 把「关系模式」这一路学来的抽象，落成一张真实存在、带约束的表。<span class="marginnote">塔嫩鲍姆调侃标准太多，SQL 就是活例子：SQL-86、SQL-92、SQL:1999、SQL:2003……各厂商还各有方言。好在核心语句三十年如一日，学通用内核，方言只是外围。</span>

## 1 SQL 全貌：一门语言，五个部件

**SQL**：由几部分子系统拼成的一门完整数据库语言。按功能划分：

**数据定义语言（DDL）**：定义关系模式、删除关系、修改模式。对应我们学的「关系模式 $R(A_1,\dots,A_n)$」。
**数据操纵语言（DML）**：查询与增删改。`SELECT`、`INSERT`、`UPDATE`、`DELETE`。
**完整性（integrity）**：定义约束，保证数据合法（主码、外码、`CHECK`）。
**视图定义（view definition）**：把查询存成「虚关系」。
**事务控制（transaction control）**：`COMMIT`、`ROLLBACK`，保证并发与故障下的正确性。
**嵌入式 / 动态 SQL**：把 SQL 嵌进 C、Java、Python 等宿主语言。
**授权（authorization）**：`GRANT`、`REVOKE`，控制谁能做什么。

**重点：SQL 是「关系语言」，不是纯声明式。** 它的查询部分是声明式的（你说要什么，系统决定怎么做），但 DDL、事务控制又带着命令式基因。上一节那张「要什么 vs 怎么做」的二分图，在 SQL 内部是**混合**存在的。

SQL 与纯关系模型还有一个根本差异：**SQL 的表基于多重集（bag）语义，允许重复行**；关系代数基于集合语义，天然去重。这条裂缝会贯穿整个第3章。

## 2 数据定义与 CREATE TABLE

**DDL**：定义「数据库的结构」——创建/删除/修改表与约束的语言。核心语句是 `CREATE TABLE`，它把关系模式翻译成真实定义：

```sql
CREATE TABLE instructor (
  ID        CHAR(5)      NOT NULL,
  name      VARCHAR(20)  NOT NULL,
  dept_name VARCHAR(20),
  salary    NUMERIC(8, 2),
  PRIMARY KEY (ID)
);
```

`instructor` 是关系名，括号里是属性名 + 域（类型）+ 约束。**基本域（basic domain）** 是建表的砖块，常用的有：

| 类型 | 含义 | 说明 |
| --- | --- | --- |
| `CHAR(n)` | 定长字符串 | 不足 n 补空格 |
| `VARCHAR(n)` | 变长字符串 | 最多 n 字符 |
| `SMALLINT` / `INTEGER` | 整数 | `INTEGER` 通常 32 位 |
| `NUMERIC(p, d)` | 定点数 | p 位精度，d 位小数，如 `NUMERIC(8, 2)` |
| `REAL` / `DOUBLE PRECISION` | 浮点数 | 近似，慎用于金额 |
| `FLOAT(n)` | 浮点数 | 精度至少 n 位 |
| `DATE` / `TIME` / `TIMESTAMP` | 日期时间 | `DATE` 含年月日 |
| `INTERVAL` | 时间段 | 可与日期做加减 |

**辨析｜易错点：金额用 `NUMERIC`，不用 `REAL`/`FLOAT`。** 浮点数是近似值，`0.1` 在二进制里不精确；账目必须用定点数 `NUMERIC`，这是金融系统的铁律。<span class="marginnote">`VARCHAR` 只存实际长度，`CHAR` 固定占 n 字符，查询时还要去掉尾随空格——「定长 vs 变长」的选择是数据库面试的经典送分题，也是第9章《存储与文件组织》的伏笔。</span>

## 3 完整性约束：建表时把规则写死

建表的同时可以声明三类最常用的完整性约束：

**NOT NULL**：该列不允许空值。

**主码（PRIMARY KEY）**：唯一标识每一行，隐含 `NOT NULL` 与唯一。可以声明在列后（列约束）或表后（表约束）：

```sql
CREATE TABLE instructor (
  ID   CHAR(5) PRIMARY KEY,
  name VARCHAR(20)
);
```

等价于在表尾写 `PRIMARY KEY (ID)`。

**外码（FOREIGN KEY）**：声明本表某列引用另一表的主码，保证「引用完整性」：

```sql
CREATE TABLE teaches (
  ID        CHAR(5),
  course_id VARCHAR(8),
  PRIMARY KEY (ID, course_id),
  FOREIGN KEY (ID) REFERENCES instructor
);
```

**重点：外码约束保证「引用的对象必须存在」。** 插入一条 `teaches` 记录时，若 `ID` 在 `instructor` 中不存在，数据库拒绝写入。<span class="marginnote">这就是第2章《码》那一节讲外码时埋的坑：码的定义是结构性的，但外码在 SQL 里被实现为一条强制规则——约束（constraint）把「设计意图」变成了「数据库行为」。</span>

**CHECK**：对值域加自定义条件：

```sql
CREATE TABLE instructor (
  ID     CHAR(5),
  salary NUMERIC(8, 2),
  CHECK (salary >= 0)
);
```

**辨析｜易错点：外码列被引用方被删除怎么办？** 若 `instructor` 里某教师被删，而 `teaches` 还引用它，数据库必须有个交代：默认行为是**拒绝删除**（RESTRICT）；也可以声明 `CASCADE`（级联删除该教师的所有 `teaches` 记录）或 `SET NULL`（把引用置空）。三种行为对应三种业务语义，选错会在运行期暴露。

## 4 修改模式：DROP 与 ALTER

**删除关系**：`DROP TABLE R`。默认行为是**拒绝**在还有引用时删除（RESTRICT），声明 `CASCADE` 则连带删除所有引用它的对象。

**修改模式**：

```sql
ALTER TABLE instructor ADD COLUMN phone VARCHAR(15);
```

`ALTER TABLE` 增减列。注意**删除**一列可能违反已有数据——若该列是外码或被其他约束引用，数据库会报错。

**辨析｜易错点：`DROP TABLE` 与 `DELETE FROM` 完全不同。** 前者删除**整个表的结构与数据**（schema 没了），后者只删**数据行**（结构还在）。「把表删了」和「把表清空」是两个量级的操作，误用 `DROP TABLE` 无法通过事务回滚。<span class="marginnote">MySQL 里这句口诀传得很广：「`DROP TABLE` 是连房子一起拆，`DELETE FROM` 是只扔家具。」后面第15章《恢复系统》会看到，日志能恢复 `DELETE`，却很难恢复 `DROP TABLE`。</span>

## 5 索引的创建

`CREATE INDEX` 不属于标准 SQL 核心，但被所有主流数据库支持，因为它直接决定查询速度：

```sql
CREATE INDEX idx_salary ON instructor (salary);
```

在 `salary` 上建索引后，查询不用扫描整张表，而是走索引（通常是 B+ 树）直达目标。**辨析｜易错点：索引是「读写权衡」——加速查询，拖慢写入**，因为每次 `INSERT` / `UPDATE` 都要同步维护索引结构。索引不是建得越多越好，第10章《索引与哈希》会专门研究它的结构与代价。

## 6 公式解析：把关系模式翻译成 CREATE TABLE

DDL 的全部工作可以浓缩成一个「翻译函数」：**关系模式 $\to$ CREATE TABLE 语句**。设模式

$$R(A_1, A_2, \dots, A_n), \qquad K = \text{主码}, \qquad F = \{(B_1 \to S_1), \dots\}$$

翻译分四步：

- **第一步，写表头**：`CREATE TABLE R (`，把模式名变成表名。
- **第二步，逐属性翻译**：每个属性 $A_i$ 写 `A_i 类型 [约束]`，域来自第2节的类型表，约束来自第3节。原子性规则在此落成「一列一种类型」。
- **第三步，翻译码**：主码 $K$ 写 `PRIMARY KEY (K)`；每个外码 $B_j \to S_j$ 写 `FOREIGN KEY (B_j) REFERENCES S_j`。
- **第四步，收尾**：以右括号与分号 `);` 结束，数据库随后为该表建立存储、系统目录条目与默认索引。

把这条映射的完整结果写出来对照：

```sql
CREATE TABLE instructor (
  ID        CHAR(5)       NOT NULL,
  name      VARCHAR(20)   NOT NULL,
  dept_name VARCHAR(20),
  salary    NUMERIC(8, 2),
  PRIMARY KEY (ID),
  FOREIGN KEY (dept_name) REFERENCES department
);
```

**这条映射的价值，是让「抽象的模式」第一次获得「物理的生命」**：从第2章《关系结构》到现在的所有概念——域、原子性、主码、外码——全部在一个语句里汇合。这也解释了为什么 DDL 通常要由 DBA 而非应用代码随意执行：它改的是数据库的**骨架**。

## 7 小结

- SQL 由 DDL、DML、完整性、视图、事务、授权等部件组成；查询部分是声明式的，但整体是**混合范式**。
- SQL 基于**多重集**语义（允许重复行），与关系代数的集合语义不同。
- DDL 用 `CREATE TABLE` 定义模式：基本域（`CHAR`/`VARCHAR`/`INT`）+ 约束（`NOT NULL`、`PRIMARY KEY`、`FOREIGN KEY`、`CHECK`）。
- 外码约束是**引用完整性**的执行者；被引用行删除时的默认行为是拒绝，`CASCADE`/`SET NULL` 是备选。
- `DROP TABLE` 删结构，`DELETE FROM` 删数据；`ALTER TABLE` 修改模式；`CREATE INDEX` 以写换读。
- 关系模式 $\to$ CREATE TABLE 的四步翻译，是 DDL 的浓缩公式。

在下一节，我们终于开始查询：**单关系查询的 SELECT 基本结构**——三条子句如何从一张表里「投影 + 选择」出你要的答案，以及 SQL 与关系代数在那里出现的第一次分歧。
