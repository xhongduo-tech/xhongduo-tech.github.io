---
title: 空值与三值逻辑
date: 2026-08-07
---

# 空值与三值逻辑

<div class="epigraph">
<p>我唯一知道的，就是我一无所知。</p>
<footer>—— 苏格拉底（Socrates，转引自柏拉图《申辩篇》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ Silberschatz《数据库系统概念》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从空值开始

前面的章节里，我们见到的关系都是「干净」的：每一格都有值，每个谓词都判真或假。可现实世界的数据没有这么客气——某位顾客没有填写邮箱，某个新上架的商品还没有定价，某台传感器那次没有上报读数。**这些「缺了一个值」的格子，在 SQL 里用特殊记号 NULL 表示**，中文叫**空值**。

NULL 一进来，我们最熟悉的两值逻辑（真 / 假）就崩溃了：$5 > 3$ 是真，那 $5 > \text{NULL}$ 算什么？它既不是真也不是假。SQL 因此引入了一个独立的第三真值 UNKNOWN，于是整个查询的求值规则都要重写。这一节我们把「数据缺失」这件事讲透——它是后续中级 SQL、乃至事务与隔离级别里许多诡异行为的根源。它与第二级《离散数学》里只有真、假二值的命题逻辑形成鲜明对照，也是「把世界装进关系」时不得不做出的妥协<span class="marginnote">数学家早就遇见过类似困境：集合论里「未知」靠空集表达，但那意味着「确实没有」；而数据库里 NULL 要同时表达「确实没有」与「暂时还不知道」，所以 SQL 干脆给了它第三种身份。</span>。

## 1 空值：三种「缺失」并不相同

**空值（null value）**：表示某个属性值**未知（unknown）**或**不存在（does not exist）**的特殊记号，SQL 中用关键字 NULL 书写。

先给直觉：一个格子里写着 NULL，可能出自三种完全不同的现实情况。

- **值不存在**：一位从未登录过系统的顾客，其「上次登录时间」这个属性对他没有意义。
- **值存在但未知**：顾客确实有邮箱，但客服录入时还没收集到。
- **值未确定**：一件商品的定价还在审批中，最终值「尚未发生」。

这三种情况在现实里要区别对待，SQL 却一律塞进同一个 NULL——这是它的便利，也是它的模糊性来源。**NULL 不是 0，不是空字符串 ''，更不是空格**：0 是一个数，'' 是一个零长度的字符串，它们都是「有值」，而 NULL 是「没有值」。

**辨析｜易错点：** 很多初学者把 NULL 当成「空的东西」——于是试图用 `salary = NULL` 来筛选，结果一条都查不出来。原因正是它根本不在两值逻辑里。判断一个值是否为 NULL，只能使用专门的谓词 `IS NULL` 与 `IS NOT NULL`。

## 2 三值逻辑：TRUE / FALSE / UNKNOWN

**三值逻辑（three-valued logic）**：SQL 的谓词求值在 TRUE、FALSE 之外引入第三个真值 UNKNOWN，任何与 NULL 的比较、以及以 NULL 为操作数的运算，其真值一律是 UNKNOWN。

这一定义可以推出一连串结论：

- 任何比较 `x OP y`（OP 是 $=, <, >, \le, \ge, \ne$ 之一），只要 $x$ 或 $y$ 中有 NULL，结果就是 UNKNOWN。特别注意：**`NULL = NULL` 也是 UNKNOWN**，因为两个未知的东西无法断言相等。
- 算术表达式只要碰到 NULL，整个结果就是 NULL：`5 + NULL` 得到 NULL。
- 谓词求值的结果落在三值集合 $\{\text{TRUE}, \text{FALSE}, \text{UNKNOWN}\}$ 中，而 **WHERE 子句只保留真值为 TRUE 的行，FALSE 与 UNKNOWN 一律丢弃**——这句话是本节最重要的执行规则。

这三个真值之间，AND、OR、NOT 的运算规则由下面的真值表定义。

## 3 公式解析：三值逻辑的真值表与一条查询的求值

这一节把三值逻辑「算」清楚。AND、OR 的规则完全由两个原则确定：

> AND：有一个 FALSE 就 FALSE，否则有一个 UNKNOWN 就 UNKNOWN，否则 TRUE。
> OR：有一个 TRUE 就 TRUE，否则有一个 UNKNOWN 就 UNKNOWN，否则 FALSE。

写成真值表（左侧为第一个操作数，表头为第二个）：

$$
\begin{array}{c|ccc}
\text{AND} & \text{T} & \text{F} & \text{U} \\ \hline
\text{T} & \text{T} & \text{F} & \text{U} \\
\text{F} & \text{F} & \text{F} & \text{F} \\
\text{U} & \text{U} & \text{F} & \text{U}
\end{array}
\qquad
\begin{array}{c|ccc}
\text{OR} & \text{T} & \text{F} & \text{U} \\ \hline
\text{T} & \text{T} & \text{T} & \text{T} \\
\text{F} & \text{T} & \text{F} & \text{U} \\
\text{U} & \text{T} & \text{U} & \text{U}
\end{array}
$$

而 $\text{NOT}$ 则简单得多：$\text{NOT T}=\text{F}$，$\text{NOT F}=\text{T}$，**$\text{NOT U}=\text{U}$**。

现在看一条真实查询。设教师关系 `instructor`，其中三行数据如下：

| ID | name | dept_name | salary |
|----|------|-----------|--------|
| 10101 | Srinivasan | Comp. Sci. | 65000 |
| 12121 | Wu | Finance | 90000 |
| 15151 | Mozart | Music | NULL |

执行查询「找出薪资超过 90000 或属于音乐系的教师」：

```sql
SELECT ID, name
FROM instructor
WHERE salary > 90000 OR dept_name = 'Music';
```

逐行求值：

- **第 1 行**：$65000>90000$ 是 F，`dept_name = 'Music'` 是 F，`F OR F = F`，丢弃。
- **第 2 行**：$90000>90000$ 是 F，`dept_name = 'Music'` 是 F，`F OR F = F`，丢弃。
- **第 3 行**：`salary > 90000`（NULL 比较）是 U，`dept_name = 'Music'` 是 **T**，`U OR T = T`，**保留**。

答案只有 Mozart——即便他的薪资未知，OR 的「一真即真」原则救了他。反过来，若把条件换成 `AND`，Mozart 这一行会得到 UNKNOWN，被丢弃。**同样的数据，换一个连接词，未知行就可能被「救回」或「误杀」**——这正是三值逻辑在现实查询里的直接后果。

## 4 与空值共处：聚集、去重与唯一约束

WHERE 只是起点。NULL 的「传染性」在聚集与约束里同样明显：

- **聚集函数忽略 NULL**：`SUM`、`AVG` 只对非空值求和与平均，NULL 不参与计算。若被聚集的列全是 NULL，`SUM` 返回 NULL 而 `COUNT` 返回 0。
- **`COUNT` 例外**：它数的是行数，和列值无关，NULL 不影响。
- **去重与 UNION**：SQL 把两个 NULL 视为「相等」，`DISTINCT` 与 `UNION` 会把多个 NULL 合并成一个。
- **UNIQUE 约束**：允许出现多个 NULL——某列有 UNIQUE 约束时，两行都为 NULL 并不违反，因为 NULL 之间「不相等」。但**主码列不允许 NULL**，因为主码的职责就是唯一标识一行。

**辨析｜易错点：** `AVG` 忽略 NULL 常常造成「平均数偏了」的错觉。求「所有教师的平均薪资」时，薪资未知的教师根本不计入分母——这更接近「已知数据的平均」，而非「全体教师的平均」。若想让未知值按 0 参与，必须先用 `COALESCE` 显式转换<span class="marginnote">COALESCE 是 SQL 标准里的「取第一个非空值」函数：COALESCE(salary, 0) 表示 salary 为 NULL 时用 0 代替。它是处理 NULL 最常用的工具，在第4章的数据类型与模式中还会再见到它的兄弟函数 NULLIF。</span>。

## 5 辨析：NULL、空字符串与数学空集

把三样容易混淆的东西放一起看：

| 记号 | 含义 | 判断方式 | 参与比较 |
| --- | --- | --- | --- |
| NULL | 没有值（未知 / 不存在） | `IS NULL` | 产生 UNKNOWN |
| `''`（空字符串） | 长度为 0 的字符串，是「一个值」 | `= ''` | 正常比较，TRUE / FALSE |
| $\emptyset$（空集） | 集合论里不含任何元素 | 数学语言 | 不在 SQL 中出现 |

关键区别：**空字符串是一个有值的内容，NULL 是「没有内容」本身**。把两者画上等号，是写 SQL 时最常见的隐性 bug 之一——许多数据库默认空字符串与 NULL 不同，却又有一些工具把空字符串「归一化」成 NULL，导致同样的查询在不同环境结果不同。<span class="marginnote">这与第一级数学里 $\emptyset \neq \{0\}$ 的辨析异曲同工：$\emptyset$ 是「没有元素的集合」，$\{0\}$ 是「装着一个 0 的集合」，NULL 与空串的关系在语义上正是这种「有没有内容」的对立。</span>

## 6 小结

- **空值 NULL** 表示「未知」或「不存在」，它不是 0、不是空串、不是空集。
- SQL 用**三值逻辑**：TRUE / FALSE / UNKNOWN；任何与 NULL 的比较都得到 UNKNOWN。
- **WHERE 只保留 TRUE**，FALSE 与 UNKNOWN 都被丢弃——这是最容易出 bug 的执行规则。
- 真值表口诀：AND 有 FALSE 即 FALSE，OR 有 TRUE 即 TRUE，NOT 不改变 UNKNOWN。
- 判断 NULL 只能用 `IS NULL` / `IS NOT NULL`；聚集函数忽略 NULL，但 `COUNT` 除外。
- 主码不允许 NULL；UNIQUE 与 DISTINCT 视多个 NULL 为相等。

在下一节，我们将离开「读」，转向「写」：如何用 **INSERT、UPDATE、DELETE** 真正地修改数据库中的元组——那里会遇到三值逻辑的另一处现身，以及「更新时究竟按旧值还是新值计算」的经典陷阱。
