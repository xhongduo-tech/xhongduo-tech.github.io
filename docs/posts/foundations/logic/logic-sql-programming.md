---
title: 命题逻辑与 SQL 查询、程序中的条件判断
date: 2026-08-07
---

# 命题逻辑与 SQL 查询、程序中的条件判断

<div class="epigraph">
<p>你写的每一行条件判断，都是命题逻辑的一个公式。</p>
<footer>—— 逻辑与编程的日常相遇</footer>
</div>

<div class="article-byline">
<p>第一级 · 逻辑学 ｜ 陈波《逻辑学导论》第10章 §10.7 ｜ 2026-08-07</p>
</div>

## 为什么从条件判断开始

布尔代数通向了硬件，本课转向**软件**：命题逻辑是程序中一切条件判断（`if`、`while`、`for`、`switch`）的数学骨架，也是数据库 SQL 查询的语义核心。<span class="marginnote">程序里的 `&&`/`||` 就是合取/析取，短路求值是真值表「一项决定整体」的运行时优化——逻辑没变，求值时机变了。</span>
读懂一段复杂条件、化简一个表达式、写出正确 WHERE 子句——这些程序员日常，本质都是命题逻辑的运算。
本课把第二篇的等值式、真值表搬到代码与 SQL 里，看逻辑如何直接变成可运行的程序。

## 1 程序中的布尔条件

程序里的一切判断都是**布尔表达式**——它就是命题逻辑公式的编程形态：

```python
# Python 中的布尔条件
if age >= 18 and has_ticket:
    print("welcome")
else:
    print("access denied")
```

对应关系：

`&&`（Python 里 `and`）↔ 合取 $\land$；`||`（Python 里 `or`）↔ 析取 $\lor$；`!`（Python 里 `not`）↔ 否定 $\neg$。
`age >= 18` 这样的比较表达式是一个原子命题（它的真假由数据决定）。
整个条件表达式是一个复合命题。

**德摩根律在代码里**（第二篇的等值式直接应用）：

```python
# 德摩根律：not (A and B) == (not A) or (not B)
if not (age >= 18 and has_id):
    print("rejected")
# 等价改写为：
if age < 18 or not has_id:
    print("rejected")
```

「不是（A 且 B）」等价于「（非 A）或（非 B）」——在重构代码、简化负逻辑时，德摩根律是最常用的工具。

## 2 短路求值：逻辑优化的实现

现代编程语言的逻辑运算符采用**短路求值（short-circuit evaluation）**：一旦结果确定，不再计算后面的操作数。

```python
# 短路求值：若 user 为 None，第二项根本不会求值，从而避免空指针异常
if user is not None and user.is_admin():
    print("admin panel")
```

当 `user is None` 时，`user is not None` 为假，`user.is_admin()` **根本不会被求值**——避免了空指针异常。
这正是「合取中一项为假，整体即假」的真值表特性在运行时的利用。

**短路求值的逻辑本质**：程序利用「$A \land B$ 在 $A$ 假时必假」「$A \lor B$ 在 $A$ 真时必真」的**提前终止**规则——它是逻辑运算顺序的工程优化。程序员用它防错（空值检查放前面），也用它提速（代价高的检查放后面）。

**辨析｜易错点：** 短路求值不是「逻辑变了」，而是「求值时机变了」。表达式 $A \land B$ 的逻辑值永远与真值表一致；短路只改变**计算过程**（B 何时被求值）。依赖副作用的代码要小心短路——`A or B` 中，`A` 为真时 `B` 根本不会执行——这既是特性也是陷阱。

## 3 SQL 查询：WHERE 子句就是逻辑公式

SQL 的 `WHERE` 子句本质是一个**逻辑公式**——它选出「使这个公式为真的所有行」：

```sql
SELECT name, age, department
FROM employees
WHERE age >= 18 AND department = 'IT';
```

`age >= 18`、`department = 'IT'` 是原子命题（对每一行求值）。
`AND`、`OR`、`NOT` 是逻辑联结词。
查询结果 = 真值表中「公式为真」的那些行（数据行当「赋值」）。

**SQL 查询优化 = 逻辑化简**：数据库优化器用逻辑等值式重写查询，让它跑得更快。例如：

```sql
-- 原查询：NOT (paid = 1 AND shipped = 1)
SELECT * FROM orders
WHERE NOT (paid = 1 AND shipped = 1);

-- 等价重写（德摩根律）：(NOT a) OR (NOT b)
SELECT * FROM orders
WHERE paid <> 1 OR shipped <> 1;
```

优化器把复杂条件重写成等价的简单形式——这正是第二篇等值式的工程应用。<span class="marginnote">数据库查询优化器本质上是一台「布尔表达式化简器」：它把 WHERE 子句转成某种规范形式（如合取范式）、用等值式寻找更优写法、利用索引推断谓词的蕴含关系。你在第二篇学的每一条等值式，数据库引擎都在偷偷用。</span>

## 4 公式解析：SQL 的 NULL 与三值逻辑

SQL 里有一个逻辑学熟悉的陷阱：**NULL 与三值逻辑**。`NULL` 表示「未知」，SQL 的布尔逻辑于是有第三个真值——UNKNOWN：

| 运算符 | 结果 |
| --- | --- |
| `NULL = NULL` | UNKNOWN（NULL 不等于 NULL！） |
| `NULL = 5` | UNKNOWN |
| `NULL < 5` | UNKNOWN |
| `UNKNOWN AND TRUE` | UNKNOWN |
| `UNKNOWN AND FALSE` | FALSE |
| `UNKNOWN OR TRUE` | TRUE |

逐项拆解这个三值表：

- **NULL = NULL 是 UNKNOWN**：NULL 不是「一个值」，而是「缺失」——两个缺失不能相等。这是 SQL 新手最大的坑。
- **UNKNOWN 的合取**：$UNKNOWN \land FALSE = FALSE$（一项为假整体即假）；$UNKNOWN \land TRUE = UNKNOWN$（真值不确定）。
- **UNKNOWN 的析取**：$UNKNOWN \lor TRUE = TRUE$（一项为真整体即真）。
- **WHERE 的筛选**：`WHERE` 只保留结果为 `TRUE` 的行——`FALSE` 与 `UNKNOWN` 一样被排除。

**这正是第六篇的三值逻辑（克莱尼）在数据库里的实战**：SQL 的 NULL 逻辑就是三值逻辑的工程实现。写查询时遇到「查不到带 NULL 的行」的困惑，根源就在此——`col = NULL` 永远匹配不到行，必须用 `IS NULL`。

## 5 逻辑与程序设计的实践

命题逻辑在编程中的应用不止条件判断：

- **单元测试的判定**：测试断言（assertion）就是「这个条件必须为真」的逻辑公式。
- **不变式（invariant）**：循环不变量、对象不变量——程序正确性的逻辑约束（第九篇数学归纳法的应用）。
- **程序验证**：静态分析、符号执行把程序条件抽象成逻辑公式来检查（下一课模型检测）。
- **代码重构**：用逻辑等值式改写条件，保持行为不变——重构的安全性由等值保证。

**辨析｜易错点：** 编程里的逻辑容易踩的坑：**把 `or`（`||`）当日常「或者」**（日常或常是排他性的，程序里的 `or` 是相容的）、**否定条件化简出错**（忘了德摩根律的翻转）、**SQL 的 `NULL` 三值语义**。这些坑的根源都在逻辑学的「精确语义」——程序设计逼着你把日常语言翻译成严格逻辑。

## 6 小结

- 程序的 `if`/`while` 条件 = **布尔表达式** = 命题逻辑公式。
- **德摩根律**在代码里用于负逻辑化简；**短路求值**是合取/析取真值表的运行时优化。
- SQL 的 `WHERE` 子句 = 逻辑公式，查询结果 = 公式为真的行。
- **SQL 的 NULL = 三值逻辑的 UNKNOWN**——NULL = NULL 不成立，WHERE 只保留 TRUE。
- 逻辑等值式贯穿重构、测试断言、不变式与程序验证——命题逻辑就是程序逻辑。

在下一节，我们把一阶逻辑变成可执行的语言——**逻辑编程与 Prolog**。
