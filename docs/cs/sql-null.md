---
title: 空值与三值逻辑
date: 2026-09-08
section: cs
---

# 空值与三值逻辑

<div class="epigraph">
<p>NULL 不是域里的值；谓词遇上它得到 unknown，WHERE 只留真，于是同一表达式不再总服从集合代数的直觉。</p>
<footer>—— 据 Date 对缺失信息的批评；SQL 标准三值逻辑；Codd 后来的标记方案对照</footer>
</div>

[上一课](/cs/sql-declarative)把查询写成声明：含义应对准代数。本课不重写 SELECT。缺口是 SQL 允许缺失：外键置空、未填列。若把 NULL 当普通值，相等与连接会把「不知道」当成匹配或当成不等，两种都会错。三值逻辑是工程合同，不是关系模型 1970 的一部分。

## 问题

[关系模型](/cs/relational-model) 的域里每个属性有值。SQL 增加 NULL。比较结果是 true / false / unknown；`WHERE` 与 `HAVING` 保留 true。`NOT unknown` 仍是 unknown。连接在 NULL 上不匹配。缺口是**声明语义相对集合代数的缝**，以便后课连接算法不要把 NULL 当哈希键上的普通字节就完事。

<span class="marginnote">Date 主张避免 NULL。主干承认 SQL 有它，并用三值把坑钉死：`x = x` 在 x 为 NULL 时不是真；`UNIQUE` 对多 NULL 的行为按标准版本而异。</span>

<span class="marginnote">直觉类比：NULL 是「考卷没交」，不是「0 分」。问「他及格了吗」答案不是「否」而是「无从判断」——所以 `age \gt 18` 与 `NOT (age \gt 18)` 在 age 缺失时都留不下该行，缺值的记录两头都进不去。</span>

## 方法

求值：算术与比较遇 NULL 变 unknown（除少数 `IS NULL`）。聚合默认跳过 NULL，`COUNT(*)` 例外。外连接用 NULL 填不成配的一侧——这是有意的标记，不是「零」。本课不把全部标准边角写成百科。

```mermaid
flowchart TD
  P["谓词"] --> TV["真 / 假 / 未知"]
  TV --> W["WHERE 只留真"]
  NULL["NULL 参与比较"] --> UNK["未知"]
```

## 机制

三值使优化器的某些代数律失效：选择下推遇到 NULL 与外连接时要检查。哈希连接把 NULL 当不相等，嵌套循环同样。完整性：主键不允许 NULL；外键列可空是[参照动作](/cs/fk-actions) 置空的前提。

```mermaid
flowchart TD
  N["同一个 NULL"] --> C["比较: NULL = NULL 得 unknown<br/>WHERE 一行都不留"]
  N --> J["连接: NULL 不与任何键匹配<br/>内连接丢行，外连接补 NULL"]
  N --> AG1["COUNT(*): 数行, NULL 也算"]
  N --> AG2["COUNT(col) / SUM: 跳过 NULL<br/>全 NULL 时 SUM 为 NULL 非 0"]
  N --> IS["IS NULL: 显式判缺失<br/>唯一拿到 true 的写法"]
```

<span class="marginnote">数字实例：10 行表里 3 行 salary 为 NULL——`COUNT(*)` 得 10，`COUNT(salary)` 得 7，`SUM(salary)` 只加 7 个数。对账时把这两个 COUNT 混用，差额恰好是缺失行数。</span>

<span class="marginnote">常见误区：初学者容易写 `WHERE col = NULL`，永远查不到任何行（结果是 unknown 不是 true）。判缺失只有一种写法：`WHERE col IS NULL`，判非缺失用 `IS NOT NULL`。</span>

## 边界

本课不引入第四种逻辑或 Codd 的 A/I 标记当主干。物理上连接怎么扫表，下一课连接算法总览。

后课默认：SQL 结果按三值过滤。同一逻辑连接仍有多种物理算法。

## 小结

- NULL 产生 unknown；WHERE 不留未知。
- 部分代数律在 SQL 里要加条件。
- 连接的物理实现下一课。
- 出处：SQL 三值逻辑；Date；对照 Codd。
