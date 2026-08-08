---
title: 数据库修改：INSERT、UPDATE、DELETE
date: 2026-08-07
---

# 数据库修改：INSERT、UPDATE、DELETE

<div class="epigraph">
<p>万物流变，无物常驻。</p>
<footer>—— 赫拉克利特（Heraclitus，πάντα ῥεῖ）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从修改开始

到上一节为止，我们一直把数据库当作一本「只读」的书：SELECT 把它翻开、投影、筛选、连接，但从不往上写字。可数据库存在的全部意义，恰恰在于它**能被写入**——新学生注册、教师涨薪、课程下架，每一项都是对关系的增、删、改。这一节讲 SQL 的三条写命令：**INSERT（插入）、DELETE（删除）、UPDATE（更新）**。

写操作比读操作危险得多：一条 UPDATE 可能改动成千上万行，一条 DELETE 若忘了 WHERE 会把整张表清空。更微妙的是「更新时机」——更新一个元组时，新值到底按旧值算还是按已算出的新值算？SQL 标准对这个问题有一个确定的答案，而各数据库的实现却并不完全一致。这一节把这套规则讲清，也为后面第13章《事务》的原子性与一致性埋下伏笔<span class="marginnote">写操作天然带着「改变」的语义，因此它与事务是同一枚硬币的两面：没有事务，一条写到一半的 UPDATE 在系统崩溃后就说不清状态了。到事务那一章，你会发现 DML 语句就是事务最小的执行单元。</span>。

## 1 INSERT：插入元组与插入查询结果

**INSERT** 用于向关系中添加元组，最直接的形式是指定属性值：

```sql
INSERT INTO instructor (ID, name, dept_name, salary)
VALUES ('10211', 'Smith', 'Economics', 66000);
```

属性名列表可省略，此时**值的顺序必须与模式中的属性顺序完全一致**；省略列表、依赖列顺序的写法可读性差，且一旦模式调整就出错，工程上不推荐。
未列出的属性自动取默认值；没有默认值且允许 NULL 时取 NULL，若该属性是主码的一部分或非空约束，则这条插入会**违反完整性约束而被拒绝**。

更强大的形式是**把一条查询的结果整批插入**——「从另一个关系筛选出符合条件的数据，灌进当前关系」：

```sql
INSERT INTO instructor (ID, name, dept_name, salary)
    SELECT ID, name, dept_name, 18000
    FROM student
    WHERE dept_name = 'Music';
```

这条语句把音乐系每位学生的学号、姓名、系别复制为教师，工资统一给 18000。<span class="marginnote">SELECT 子句里写常量 18000 是合法的：查询的每个输出列既可以是表列，也可以是表达式与常量。这展示了 INSERT 与 SELECT 的无缝衔接——查询结果本质就是一个关系，而关系就可以被插入。</span>

**辨析｜易错点：** 若被插入的表与查询来源是同一张表，SQL 标准规定先算完整个查询、再执行插入（快照语义），因此通常不会出乱子；但个别方言对「边查边插」的处理有差异，生产环境要避免这种自引用写法。

## 2 DELETE：删除元组

**DELETE** 按条件删除元组，语法与 SELECT 的 WHERE 一脉相承：

```sql
DELETE FROM instructor
WHERE dept_name = 'Finance';
```

- 不带 WHERE 的 DELETE 会**删除所有元组**——表还在，只是空了。这经常是误操作的来源：手一抖删了整张表，与 TRUNCATE 的行为需仔细区分。
- DELETE 删除的是元组，不是关系本身；关系模式、索引、约束都原样保留。
- 与 INSERT 一样，DELETE 可能违反外码约束：若其他表的外码引用了待删元组，删除可能被拒绝，或触发级联删除——这取决于约束的定义方式，第4章《完整性约束》会专门展开。

WHERE 里可以使用子查询，比如「删除在 Watson 大楼开设课程的所有教师」：

```sql
DELETE FROM instructor
WHERE dept_name IN (SELECT dept_name
                    FROM department
                    WHERE building = 'Watson');
```

这里的子查询先在当前状态下求值，再对每条元组应用删除条件——标准语义同样基于快照。

## 3 UPDATE：更新元组

**UPDATE** 修改已有元组的一个或多个属性，由 SET 指定新值、WHERE 限定范围：

```sql
UPDATE instructor
SET salary = salary * 1.05
WHERE dept_name = 'Physics';
```

- SET 右侧可以引用**该元组的旧值**，`salary = salary * 1.05` 意为「在旧工资基础上涨 5%」——这是 UPDATE 最常用的形态。
- 不带 WHERE 的 UPDATE 会作用到所有元组：`UPDATE instructor SET salary = salary * 1.05` 给全体教师涨薪 5%。

要表达「按条件给不同的人涨不同幅度」，就需要 CASE 表达式——它把「分情况赋值」写进一条 UPDATE。

## 4 公式解析：UPDATE 的求值时机与 CASE 分档

先看 Silberschatz 的经典例子：给教师涨薪，**10 万以下涨 5%，10 万及以上涨 3%**：

```sql
UPDATE instructor
SET salary = CASE
    WHEN salary <= 100000 THEN salary * 1.05
    ELSE salary * 1.03
END;
```

这条语句的求值可以拆成三步：

- **第一步，逐元组读取旧值**：UPDATE 隐式地对每个元组执行一次循环，循环体内的 `salary` 在 SET 与 CASE 中引用的都是**该元组的旧值**。
- **第二步，CASE 分档**：CASE 从第一个 WHEN 开始顺序判断，命中第一个为 TRUE 的分支即停止。`WHEN salary <= 100000` 对某个元组为 TRUE，就取 `salary * 1.05`；否则落到 ELSE 取 `salary * 1.03`。
- **第三步，整体写回**：对每个元组算出新值后，再统一把新值写回该元组。于是**每一行的新值都只依赖它自己的旧值**，与遍历顺序无关：

$$v_{\text{new}}(t_i) \;=\; f\big(v_{\text{old}}(t_i)\big)$$

这条式子是 SQL 标准对 UPDATE 的核心约定：**SET 表达式一律基于旧行求值**。它的直接推论是，写一条「交换两列」的语句 `UPDATE instructor SET A = B, B = A` 在标准语义下能够正确交换；但**部分数据库（如 MySQL 默认模式）从左到右依次赋值**，`A = B` 先执行后，`B = A` 拿到的是 A 的新值，结果便不是交换——同一份 SQL 在两个系统里行为不同。这是标准与实现分歧的著名实例，也是「读写要分清方言」的教训。

**辨析｜易错点：** 再问一次三值逻辑——若某个元组的 salary 为 NULL，CASE 的 WHEN 条件得到 UNKNOWN，UNKNOWN 不等于 TRUE，于是落入 ELSE，算出 ELSE 分支的值（仍是 NULL），工资依旧是 NULL。**CASE 把 UNKNOWN 当「不满足」处理**，因此 NULL 值在条件更新中会被静默跳过，绝无异常提示。

## 5 辨析：写操作与约束、事务

三条写命令有一个共同特点：它们都可能**违反完整性约束**，而系统对此的态度是「拒绝整条语句」。常见的失败场景：

- 主码冲突：插入一个主码已存在的元组。
- 非空约束：插入或更新后某属性为 NULL，而该属性不允许 NULL。
- 外码约束：插入了引用不存在的外码，或删除了被外码引用的元组。

**辨析｜易错点：** UPDATE 里 WHERE 不满足任何行、INSERT 里值合法却一个都不插——这些语句「成功执行但影响 0 行」，SQL 不报错。判断是否真的改到了数据，要看命令返回的「受影响行数」。许多 ORM 与数据同步脚本的 bug，都源于把「语句没报错」误当成「数据被改了」。

最后，写操作不是孤立事件：它们在一个**事务（transaction）**里执行，可以整体提交（commit）或整体回滚（rollback）。这也是为什么 INSERT/UPDATE/DELETE 叫 **DML（数据操纵语言）**——它们操纵数据，而 DDL 操纵模式，两者的职责边界从第1章的《数据库语言》延续至今。到第13章，我们将回答：多条写操作怎样才能打包成一个「要么全发生、要么全不发生」的原子单元。

## 6 小结

- **INSERT**：插入指定元组，或把一条 SELECT 的结果整批插入；未列出的属性取默认值 / NULL。
- **DELETE**：按 WHERE 删除元组；不带 WHERE 会清空整张表，但关系模式保留。
- **UPDATE**：用 SET 修改元组，SET 右侧可引用该元组的旧值。
- 更新时机：SQL 标准规定 **SET 表达式一律基于旧行求值**，与遍历顺序无关；但部分数据库从左到右赋值，行为有别。
- CASE 在条件更新里做分档赋值，UNKNOWN 被当作「不满足」，NULL 因此会被静默跳过。
- 三条命令都可能违反完整性约束而被整条拒绝；「成功执行」不等于「真的改了数据」，要看受影响行数。
- 写操作运行在事务之中，可提交可回滚——事务语义在第13章展开。

在下一节，我们将进入中级 SQL：前面的连接都是「只保留匹配行」的内连接，接下来要处理「不匹配的行怎么办」——这就是 **连接表达式**：内连接、外连接与自然连接。
