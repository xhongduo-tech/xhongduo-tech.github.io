---
title: Spark SQL 与 DataFrame
date: 2026-08-07
---

# Spark SQL 与 DataFrame

<div class="epigraph">
<p>让分析人员用熟悉的语言（SQL）做大规模计算，是数据系统走向大众的分水岭。</p>
<footer>—— 马泰 · 扎哈里亚（Matei Zaharia），《Spark 权威指南》第 9 章</footer>
</div>

<div class="article-byline">
<p>第三级 · 大数据系统 ｜ 《Spark 权威指南》（Bill Chambers & Matei Zaharia）Ch.9 ｜ 2026-08-07</p>
</div>

## 为什么从 Spark SQL 与 DataFrame 开始

上一节我们看到了 RDD 的强大，但它有个现实问题：**RDD 的 API 是过程式的**——你得像写程序一样描述"每一步做什么"。而数据分析师、SQL 用户更习惯**声明式**：只说我"要什么"（`SELECT ... WHERE ... GROUP BY ...`），怎么算交给系统。Spark SQL 把这两种世界连接了起来：**DataFrame 是一张带有 schema 的分布式表，既能像 RDD 一样跑在集群上，又能被 SQL 直接查询，还能被一个自研优化器自动调优。**<span class="marginnote">Spark SQL 的野心不止于"给 Spark 加个 SQL 接口"：它要成为统一的数据访问层——让 SQL、DataFrame API、甚至机器学习管道共享同一套逻辑计划与优化器。</span>它与上一节的 RDD、第二级《数据库》的 SQL 查询优化、第一级《逻辑》的谓词逻辑一脉相承。

## 1 DataFrame：带 schema 的分布式表

**DataFrame**：一个**分布式的、带列名与类型（schema）的**数据集，逻辑上就是一张数据库表。

```scala
val users = spark.read.parquet("users.parquet")   // 从列式文件读入
users.where("age > 30").groupBy("city").count().show()
```

相比 RDD 的三个升级：

**有 schema**：每列的类型已知，系统可以据此做类型检查与优化。
- **有优化的余地**：SQL 表达式可分析、可重排、可合并——"自由"反而给了优化器施展的空间。
- **跨语言统一**：同一份逻辑可以用 SQL、Python、Scala、Java 表达，编译到同一个物理计划。

**重点：RDD 管"怎么算"，DataFrame 管"算什么"；前者给足自由，后者换来优化。** 让出部分自由度、换取系统替你优化，是声明式编程在分布式世界的又一次胜利。

## 2 Catalyst 优化器：把 SQL 变成最优物理计划

DataFrame / SQL 的查询进入 Spark 后，经 **Catalyst 优化器**处理，四步流水线：

1. **分析（Analysis）**：解析出逻辑计划（逻辑树），解析列名、类型、函数，生成带解析的表达式。
2. **逻辑优化（Logical Optimization）**：在**逻辑计划**上做等价变换——谓词下推（先过滤再 join）、列裁剪（只取需要的列）、常量折叠（`2+3` 提前算成 5）。这些变换保证"语义不变、代价变小"。
3. **物理计划（Physical Planning）**：为逻辑计划选择物理实现——用什么 join 算法（broadcast join、sort-merge join）、怎么分区、怎么扫描。
4. **生成代码（Codegen）**：把物理计划编译成 **Java 字节码**，避免每行数据都走一次通用解释器。

<span class="marginnote">逻辑优化是"规则驱动"的：一条条等价变换（filter 下推、join 重排）像定理一样被套用，直到收敛。这与你学过的第一级《逻辑》里的等价变形是同构的——只不过这里的对象从命题换成了查询树。</span>

Catalyst 的经典优化项及各自收益，值得单独列表（它们也是面试与调优的高频话题）：

| 优化 | 做的事 | 量化收益 |
| --- | --- | --- |
| 谓词下推 | 过滤条件尽量提前 | 连接量从 $N^2$ 降到 $qN^2$（见公式解析） |
| 列裁剪 | 只保留需要的列 | IO 与内存按 $\frac{K}{C}$ 收缩 |
| 常量折叠 | `1 + 1` 预计算 | 消除表达式重复求值 |
| 广播 join | 小表复制到各节点 | 消除 shuffle，IO 归零 |
| 阶段合并 | 相邻算子合并 | 减少任务数与调度开销 |

**实践要点**：想被优化器"看见"，就要把逻辑写"规整"——`WHERE` 条件尽量靠近读取处、join 键保持一致、避免在过滤前做大范围 `DISTINCT`。**优化器是"规则引擎"而非"魔法"：它只在你的逻辑恰好能套用等价变换时出手，把查询写成"能触发的规则越多"，收益越大。**

## 3 Tungsten：把 Java 对象踢出热路径

优化器的另一半功劳来自 **Tungsten** 项目。它的三个动作直击 JVM 的痛点：

**直接操作二进制**：数据按紧凑的二进制格式驻留内存，不经 Java 对象，省去对象头与 GC 压力。
- **缓存友好布局**：列式排列使 CPU 缓存命中率大幅提升。
- **全阶段代码生成（whole-stage codegen）**：把一个 stage 内所有操作（scan → filter → aggregate）融合成一段循环，消除虚函数调用。

**"Catalyst 决定'做什么'，Tungsten 决定'怎么做得快'。"** 两者结合，是 Spark SQL 在多数场景下比手写 RDD 更快的根本原因——这在直觉上反直觉，却正是"放弃控制换优化"的典型收益。

**GC 是 JVM 大数据的隐形杀手**：对象越多、存活越久，Full GC 越频繁。Tungsten 用"行外二进制"绕开对象堆，让千万行数据不再以数百万个 Java 对象驻留——这是它能支撑单个任务上百 GB 内存吞吐的底气。

## 4 SQL 与 DataFrame 的等价性

同一个查询，SQL 与 DataFrame API 完全等价：

```sql
SELECT city, COUNT(*) FROM users WHERE age > 30 GROUP BY city;
```

```scala
users.where($"age" > 30).groupBy("city").count()
```

两者最终生成**同一棵逻辑计划**。这意味着：SQL 分析人员与 DataFrame 工程师可以共存于同一系统；Spark SQL 甚至可以读写 Hive 表、连接 JDBC 数据源，成为异构数据湖的统一查询入口。<span class="marginnote">DataFrame 之上还有强类型的 <strong>Dataset</strong>（编译期类型安全），它是"类型安全的 DataFrame"。在本系列里我们专注于 DataFrame 这一层，它覆盖了绝大多数生产场景。</span>

**辨析｜易错点：** DataFrame 并不保证"比你手写的 RDD 快"——它保证的是"多数情况下更快且写起来更短"。当你用到 RDD 特有的复杂 UDF 或自定义分区逻辑时，优化器无法介入，这时手写的 RDD 可能与 DataFrame 打平。**优化的前提是"声明式、可分析"**：把逻辑表达得越规整，Catalyst 能替你做的越多。

## 5 公式解析：为什么"先过滤再连接"更快

谓词下推是 Catalyst 最经典的优化，值得精确地算一遍收益。设表 $A$、$B$ 各有 $N$ 行，连接条件为等值连接，过滤条件"只保留 $A$ 中满足 $P$ 的行"，命中率为 $q$（$0<q<1$）。

**不下推**：先做 $N \times N$ 的连接再过滤，计算量
$$
C_{\text{no-push}} = N^2
$$

**下推**：先过滤 $A$（得 $qN$ 行）再连接，计算量
$$
C_{\text{push}} = (qN) \cdot N = qN^2
$$

- **$q$（过滤命中率）**：谓词选中的行占全表的比例，例如"只留近 30 天订单"可能 $q=0.05$。
- **收益**：连接计算量降为原来的 $q$。配合**列裁剪**（连接前只保留 join 键与要输出的列），IO 与内存同步收缩。
- **更关键的是 broadcast join 的解锁**：当过滤后的小表小于阈值（默认 10 MB）时，Spark 把它复制到每个执行节点做内存哈希连接，**完全消除 shuffle**——一次"过滤下推"能同时省下计算、IO 与网络三笔账。

**这就是为什么写 SQL 时"能过滤的尽早过滤、能裁剪的尽早裁剪"是一条放之四海而皆准的规则。**

**数字算例**：设订单表 2 亿行、维度表"城市表"仅 200 行（约 40 KB，远小于默认 10 MB 的广播阈值）。不广播时，两表 join 需要把城市表 shuffle 到每个 reducer，一次全量网络传输；开启 broadcast join 后，城市表被复制到每个执行节点，join 完全在本地内存哈希表里完成，shuffle 的 IO 与网络开销归零。**Spark 3.0 的 AQE（自适应查询执行）甚至会在运行中动态判断小表尺寸、自动切换广播策略**——把"优化最后一公里"也交给了引擎，而不是依赖 DBA 手工加 `hint`。

## 6 小结

- **DataFrame** 是带 schema 的分布式表，声明式、可优化、跨语言统一；RDD 是过程式、给自由。
- **Catalyst** 四步：分析 → 逻辑优化 → 物理计划 → 代码生成，核心变换是谓词下推与列裁剪。
- **Tungsten** 提供二进制内存布局与 whole-stage codegen，把数据与代码都推向极限。
- SQL 与 DataFrame 最终生成**同一棵逻辑计划**，可统一访问 Hive 等异构源。
- 优化收益可量化：谓词下推把连接量从 $N^2$ 降到 $qN^2$，甚至解锁无 shuffle 的 broadcast join。
- **优化的前提是"声明式、可分析"**：逻辑写得越规整，Catalyst 能触发的规则越多；复杂的自定义 UDF 会绕过优化器，退回手写水平。
- **AQE 是"运行时的第四阶段"**：Spark 3.0 起在执行中动态重排 join、合并小分区，让优化从"写时"延伸到"跑时"。

在下一节，我们从"批量"的世界迈向"实时"的世界——看 **Structured Streaming 如何把流处理写成和批处理一模一样的代码**。
