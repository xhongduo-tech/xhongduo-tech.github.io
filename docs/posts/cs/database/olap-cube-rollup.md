---
title: OLAP 操作：CUBE、ROLLUP 与数据立方体
date: 2026-08-07
---

# OLAP 操作：CUBE、ROLLUP 与数据立方体

<div class="epigraph">
<p>当你只能向上看时，你也在向下看——聚合的每一层都在回答不同的问题。</p>
<footer>—— 埃德加 · 科德（E. F. Codd，关系模型之父）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第5章 高级 SQL ｜ 2026-08-07</p>
</div>

## 为什么从数据立方体开始

假设你是一家连锁书店的经理，想知道销售额：按门店、按月份、按图书类别、按三者任意组合。逐条写 GROUP BY 查询要写 8 条不同的 SQL；更糟的是，业务上「总计 / 小计」这种多层次汇总，普通 GROUP BY 根本表达不出来——它要么给最细粒度，要么给单一粒度。SQL:1999 引入的 ROLLUP 与 CUBE 正是为**多维分析（OLAP）**而生的聚合算子：一次查询，把多个粒度的汇总全部算出来。本篇把第 5 章高级 SQL 收尾，也为第 20 章 OLAP 与数据仓库埋下伏笔。

## 1 从 GROUP BY 到多维聚合

GROUP BY 按一组属性分组，输出每个组合一行。多维聚合则把「按哪些属性分组」本身当成一个可枚举的集合：

```sql
SELECT branch, month, category, SUM(amount)
FROM sales
GROUP BY branch, month, category;
```

这只给了最细粒度（三者全分）。**分析报表真正需要的是一棵「从细到粗」的汇总树**：按 (branch, month, category) 的明细组，按 (branch, month) 的小计，按 (branch) 的中计，以及整张表的总计。

**辨析｜易错点：** 多维聚合结果的「小计行」里，未被分组的属性填的是 NULL。这个 NULL 与数据里真正的空值语义不同——它是**聚合占位符**，读报表时靠 GROUPING 函数区分，而非直接判断 NULL。

## 2 ROLLUP：下钻维度上的层级小计

ROLLUP 生成按 $(a,b,c)$、$(a,b)$、$(a)$、$()$ 四种粒度的分组并分别聚合，$()$ 表示全局总计：

```sql
SELECT branch, month, category, SUM(amount)
FROM sales
GROUP BY ROLLUP (branch, month, category);
```

结果里既有按 (branch, month) 的明细汇总，又有每个 branch 跨月的**月度小计**，还有全表**总计**。<span class="marginnote">ROLLUP 的名字来自「向上卷起」：它沿着分组属性从最细一层逐级「卷」向最粗一层。对时间维度做 ROLLUP 是报表系统最常见的用法——年 → 季 → 月 → 日，每卷一级少一个维度。</span>

**ROLLUP 的方向性**：它假设分组属性有自然的**层级顺序**（如 time、region、category），小计只沿这个顺序产生，不会交叉。

## 3 CUBE：所有组合的交叉小计

CUBE 生成**所有 8 种**属性组合的分组：$2^3 = 8$ 个分组，从 $(a,b,c)$ 到 $()$ 全覆盖。对于上面的连锁书店查询：

```sql
SELECT branch, month, category, SUM(amount)
FROM sales
GROUP BY CUBE (branch, month, category);
```

CUBE 对 $n$ 个分组属性会生成 $2^n$ 个分组。$n=3$ 时只有 8 个，$n=10$ 时就是 1024 个——**CUBE 的行数随维度指数爆炸**，这是多维聚合最核心的成本特征。

**辨析｜易错点：** ROLLUP 是 CUBE 的子集。对同一组属性，CUBE 的结果包含 ROLLUP 的全部层级，再加所有交叉组合。如果你的需求只是「时间逐级小计」，用 ROLLUP 即可；要「任意维度两两交叉」才需要 CUBE。

## 4 公式解析：多维聚合的规模

设分组属性个数为 $n$，则各算子的分组数为：

$$
N_{\text{GROUP BY}} = 1, \qquad
N_{\text{ROLLUP}} = n + 1, \qquad
N_{\text{CUBE}} = 2^n
$$

- **第一步，看 GROUP BY**：只有 $(a_1, \dots, a_n)$ 一种粒度，分组数恒为 1。
- **第二步，看 ROLLUP**：沿层级依次去掉末尾属性，得到 $n+1$ 种粒度，线性增长。
- **第三步，看 CUBE**：每个属性「参与 / 不参与」分组是二选一，共 $2^n$ 种组合，**指数增长**。
- **第四步，落到工程**：$n$ 大时 CUBE 结果膨胀，所以数据仓库常在**物化视图**里预计算常用组合，而不是每次查询现场算全立方体——这就是后面第 12 章物化视图、第 20 章数据仓库的接缝。

## 5 数据立方体与多维数据分析

**数据立方体（data cube）**是 OLAP 的概念模型：以多个维度为轴、以度量值为格点的多维数组。SQL 的 CUBE 只是立方体的一种物化形态。<span class="marginnote">OLAP 的四个标准操作：上卷（roll-up，减维度）、下钻（drill-down，加维度）、切片（slice，固定一个维度的某个值）、切块（dice，固定多个维度的取值区间）。CUBE 一次性预计算所有切片，正是为了这些操作能即时响应。</span>

- **切片（slice）**：固定一个维度的某个取值，如只看「北京门店」。
- **下钻（drill-down）**：从「月份」下钻到「日」，增加维度层次。
- **上卷（roll-up）**：反过来，从「日」汇总到「月」。
- **旋转（pivot）**：交换维度轴的角色，行变列。

**连接点：** 普通 GROUP BY 是你手写每一格，CUBE 是**一次把整座立方体算好**。理解这个「预计算 vs 按需计算」的对立，是后面学习物化视图（第 12 章）与数据仓库（第 20 章）的思维起点。

## 6 多维聚合的数值算例与术语速查

**用一个具体销售数据把 ROLLUP 与 CUBE 的行数算出来。** 设 3 个门店 × 4 个月 × 5 个类别 = 60 条明细。

- **GROUP BY (branch, month, category)**：60 行（最细粒度）。
- **ROLLUP**：$n+1 = 4$ 种粒度——(branch,month,category)=60 行 + (branch,month)=12 行 + (branch)=3 行 + ()=1 行 = **76 行**。
- **CUBE**：$2^3 = 8$ 种粒度，行数 ≈ 60 + 12 + 15 + 20 + 3 + 4 + 5 + 1 = **120 行**。
- **结论**：CUBE 行数约为 ROLLUP 的 1.6 倍（此例），但**维度增加到 6 个时，CUBE 是 ROLLUP 的 8 倍以上**——指数 vs 线性的差距随维度放大。

**数值算例：GROUPING 函数的使用** 设 ROLLUP 结果里有一行 `(北京, 3月, NULL, 5000)`。

- 这个 NULL 是「类别维度的汇总占位」，还是「类别真为 NULL 的商品」？
- 用 `GROUPING(category)` 区分：返回 1 表示聚合占位，0 表示真实数据——**报表层用 GROUPING 过滤/标注小计行**。
- 工程实践：ETL 把多维聚合结果写入事实表时，用 `GROUPING` 生成 `is_total` 标记列，避免下游把汇总行当明细。

**辨析｜易错点：** CUBE 与 ROLLUP 的「层级顺序」语义不同——ROLLUP 只沿给定顺序做前缀小计，CUBE 做所有组合。**`GROUP BY ROLLUP(a, b, c)` ≠ `GROUP BY CUBE(a, b, c)` 的前 4 层**：CUBE 的 8 层包含 ROLLUP 的 4 层 + 4 个交叉层（如 (a,c)、(b,c) 等）。

<span class="marginnote">多维聚合是「数据立方体」的 SQL 入口：<strong>CUBE 预计算整座立方体，OLAP 工具的上卷/下钻只是「读取立方体的不同切片」</strong>——这就是为什么数据仓库要预聚合（第 20 章），而不是每次报表现场算。</span>

### 术语速查

| 术语 | 含义 |
| --- | --- |
| ROLLUP | 沿层级顺序产生 n+1 级小计 |
| CUBE | 全部 2^n 种组合的分组 |
| 数据立方体 | 多维度 + 度量的多维数组 |
| 切片/切块 | 固定一个/多个维度的取值 |
| 下钻/上卷 | 增加/减少维度层次 |
| GROUPING | 区分聚合占位与真实 NULL |

## 7 小结

- ROLLUP 沿分组属性的**层级顺序**产生 $n+1$ 级小计；CUBE 产生全部 $2^n$ 种组合。
- 多维聚合的小计行用 NULL 占位未分组属性，须用 GROUPING 区分真正的空值。
- CUBE 结果**指数膨胀**，是 OLAP 查询的核心成本，需靠预计算物化视图缓解。
- 数据立方体把多维汇总建模为**轴 + 度量**，支撑切片、下钻、上卷、旋转四种分析操作。

在下一节，我们将进入第 6 章——**形式化关系查询语言**，用逻辑语言（而不是 SQL）重新表达「查询」，回答「SQL 到底能算什么、不能算什么」。
