---
title: Spark RDD 与 DAG
date: 2026-08-07
---

# Spark RDD 与 DAG

<div class="epigraph">
<p>持久化到内存的数据结构，让迭代计算第一次有了"在线"的体验。</p>
<footer>—— 马泰 · 扎哈里亚（Matei Zaharia），《Spark 权威指南》第 2 章</footer>
</div>

<div class="article-byline">
<p>第三级 · 大数据系统 ｜ 《Spark 权威指南》（Bill Chambers & Matei Zaharia）Ch.2 ｜ 2026-08-07</p>
</div>

## 为什么从 RDD 与 DAG 开始

MapReduce 把"计算"拆成了固定两段（map、reduce），模型简单，代价却是**每次计算都要落盘、每个中间结果都要重新生成**。机器学习、图计算这类**迭代式算法**要反复读同一份数据几十上百次，MapReduce 会在每次迭代间把数据写回磁盘又读出来，慢得让人难以忍受。<span class="marginnote">2009 年伯克利 AMPLab 的 Spark 项目就是为解决这个问题而生：核心洞察是——数据放内存、中间过程不落盘，迭代速度可以快上 100 倍。</span>RDD（Resilient Distributed Dataset，弹性分布式数据集）与 DAG（Directed Acyclic Graph，有向无环图）就是 Spark 实现这一目标的底层抽象。理解它们，就等于拿到了整个 Spark 的钥匙，也为你理解 Spark SQL、Structured Streaming 提供了地基。

## 1 RDD：一个被持久化的分布式集合

**RDD（弹性分布式数据集）**：一个**只读的、分区的**数据集合，它分布在集群的多台机器上，并且可以显式地**缓存在内存**中供反复使用。

拆开这三个词：

**分布式（Distributed）**：数据被切成若干个**分区（partition）**，分散在各节点上，每个分区可以在不同机器上并行处理。
- **数据集（Dataset）**：它像编程语言里的一个集合，支持 `map`、`filter`、`reduceByKey` 这类熟悉的操作，把分布式计算包装成"操作一个集合"的体验。
- **弹性（Resilient）**：节点故障时，RDD 能通过**血缘（lineage）**——记住自己是"怎么从源头算出来的"——重算丢失的分区，而不必依赖复杂的状态备份。

**重点：RDD 不是"存着数据的仓库"，而是"怎样从源头算出数据的一份食谱"（lazy 的食谱）。** 它默认不落盘，需要时按食谱重算；显式调用 `cache()` 或 `persist()` 才把分区留在内存。

## 2 转换与行动：惰性求值

RDD 上的操作分两类，这也是理解 Spark 性能的起点：

- **转换（transformation）**：如 `map`、`filter`、`flatMap`、`reduceByKey`。它返回一个新 RDD，**惰性（lazy）**——不立即计算，只是记录"变换食谱"。
- **行动（action）**：如 `count`、`collect`、`saveAsTextFile`。它才真正触发计算，把结果带回驱动端或写入存储。

<span class="marginnote">惰性求值并非 Spark 独创——Haskell 等函数式语言把它用到了极致。你可以在第一级《函数》与第二级《函数式编程》里看到同一思想：先描述"怎么算"，等到真正需要结果才执行，从而有机会把整条计算链整体优化。</span>**惰性求值让 Spark 能在触发计算前看到"整张计算图"，从而合并不必要的步骤、跳过用不到的中间结果。**

`map` 与 `reduceByKey` 的组合是理解这条链路最经典的例子——把词频统计写成一段 RDD 链：

```scala
val lines = sc.textFile("hdfs://.../news.txt")   // 读取：产出一个 RDD
val words = lines.flatMap(_.split(" "))          // 转换：拍平成一个个词
val pairs = words.map(w => (w, 1))               // 转换：每个词记上 1
val counts = pairs.reduceByKey(_ + _)            // 转换：按词累加（宽依赖）
counts.collect()                                  // 行动：真正触发计算
```

前四行只是"记食谱"，`collect()` 这一下才让 Spark 回溯整张 DAG 并真正执行。把这段代码贴进 spark-shell 亲手跑一遍，观察 driver 日志里"从源头读 HDFS → 逐级变换 → 归约"的舞台切分，是理解惰性求值最快的方式。

## 3 DAG：把计算画成一张有向无环图

驱动端把 RDD 之间的依赖关系组织成一棵**DAG（有向无环图）**——每个节点是一次转换，每条边是数据的依赖。DAG 被进一步切分为若干**阶段（stage）**：

- **宽依赖（wide / shuffle dependency）**：父分区的数据要跨节点重组后才能继续（如 `reduceByKey` 需要把相同 key 从四面八方收拢），必须**shuffle**，DAG 在这里被切开成新阶段。
- **窄依赖（narrow dependency）**：每个父分区只被一个子分区消费（如 `map`、`filter`），可**在同一个阶段内流水线式**执行，无需网络传输。

**辨析｜易错点：** 并不是"宽依赖更慢所以要避免"这么简单。宽依赖触发 shuffle（落盘 + 网络传输），确实昂贵；但它也是 `reduceByKey`、`join`、`distinct` 这类"需要全局按 key 聚合"的操作所必需的。真正要避免的是**无谓的 shuffle**——例如在 `reduceByKey` 之后立即再做一次 `groupByKey`，两次合并完全可以合成一次。

## 4 血缘与故障恢复：弹性从哪来

每个 RDD 都记录着自己的**血缘（lineage）**：它由哪个父 RDD、经过哪个转换产生，参数是什么。当某个节点宕机导致分区丢失时：

1. Spark 用 DAG 定位丢失的分区。
2. 从它的**最原始源头**（读 HDFS 或 cache 中的快照）按血缘逐步重算。
3. 若中间某处有 `persist()` 的检查点，则从最近的可缓存点恢复，避免全链路重算。

**Spark 的容错代价与 RDD 的物化程度成反比：缓存得越密，故障恢复越快；缓存得越疏，恢复时重算的开销越大。** 这是"空间换时间"在分布式容错里的又一次现身——与 HDFS 用副本换可靠性（$P = p^R$）是同一思想的不同实现。

`persist()` 提供了多档存储级别，从最"便宜"到最"稳"的选择逻辑值得单独理清：

- **MEMORY_ONLY**：只存内存，放不下就重算。适合"重算便宜、内存够"的场景——最省空间。
- **MEMORY_AND_DISK**：内存放不下的分区写磁盘，不重算。适合"重算贵"的场景——最稳妥的默认。
- **MEMORY_ONLY_SER**：序列化后存内存，省内存但 CPU 序列化有开销。
- **DISK_ONLY**：只落盘，相当于"便宜的物化"。
- **CHECKPOINT（检查点）**：把 RDD **连同其血缘一起切断**写入存储。与 `persist` 不同，检查点不保留血缘，故障后只能从检查点恢复——**适合血缘链极长、每次重算都要从源头跑很久的 RDD**，是"血缘重算"与"物化缓存"之外的第三条容错路径。

**辨析｜易错点：** `cache()` 只等于 `persist(StorageLevel.MEMORY_ONLY)`，且**惰性**——触发 action 时才会真正缓存，无 action 的一切缓存声明都不生效。调试"缓存为什么没生效"时，先检查是否真的跑了一次 action。

## 5 公式解析：为什么 Spark 比 MapReduce 快

两种模型处理同一份数据的成本可粗略建模。设数据规模 $N$，迭代 $I$ 轮，每轮计算所需数据传输量为 $T$，磁盘与网络带宽同为 $B$，且网络传输主导耗时。MapReduce 每轮结束把中间结果写磁盘、下一轮再读回，**每轮两次穿越带宽**；Spark 用内存缓存中间结果，只在 shuffle 时穿越一次：

$$
\text{MR 总耗时} \approx \frac{2 \cdot I \cdot N}{B}, \qquad
\text{Spark 总耗时} \approx \frac{I \cdot N}{B} + \frac{N}{B_{\text{mem}}}
$$

其中 $B_{\text{mem}}$ 是内存带宽，比磁盘/网络带宽高约一个数量级。

- **MR 的 $2 \cdot I \cdot N$**：每轮迭代写盘 + 读盘两次全量穿越。
- **Spark 的 $I \cdot N$**：每轮只在 shuffle 时穿越一次；内存缓存的读写几乎不产生跨设备开销。
- **结论**：迭代轮数 $I$ 越大，差距越显著——**这解释了为什么机器学习（几十上百轮迭代）是 Spark 诞生时的第一个杀手级场景**，也是"内存计算"这一口号的数量化含义。

**数字算例**：设 $N=100\text{ GB}$、磁盘带宽 $B=100\text{ MB/s}$、迭代 $I=50$ 轮。按上式，MapReduce 总耗时 $\approx \frac{2 \times 50 \times 100\text{ GB}}{100\text{ MB/s}} = 10^5$ 秒（约 28 小时）；Spark 若把中间结果全部缓存于内存，$\approx \frac{50 \times 100\text{ GB}}{100\text{ MB/s}} + \frac{N}{B_{\text{mem}}} \approx 5\times 10^4$ 秒（约 14 小时）。**粗模型只算出 2 倍，但真实差距远大于此**——因为磁盘的机械寻道与随机 IO 才是主要成本，而粗模型把读写理想化成连续带宽。这正是"内存计算快上百倍"这句话在真实负载里的来源。

## 6 术语速查表

| 术语 | 英文 | 一句话解释 |
| --- | --- | --- |
| 分区 | partition | RDD 被切成的并行处理单元 |
| 转换 | transformation | 惰性返回新 RDD 的操作 |
| 行动 | action | 真正触发计算的求值操作 |
| 惰性求值 | lazy evaluation | 先描述计算图，用到结果才执行 |
| 宽依赖 | wide dependency | 需跨节点重组数据，触发 shuffle |
| 窄依赖 | narrow dependency | 父分区只在节点内被消费，无需网络 |
| shuffle | shuffle | 按 key 跨节点重新分布数据 |
| 血缘 | lineage | 记录 RDD 由谁、如何计算而来 |
| 检查点 | checkpoint | 切断血缘、物化到稳定存储 |
| 持久化 | persistence | 把 RDD 缓存到内存/磁盘避免重算 |
| 存储级别 | storage level | persist() 指定的缓存方式与位置 |

## 7 小结

- **RDD** 是只读、分区、可缓存的分布式集合，默认惰性、可依赖血缘重算。
- 操作分**转换（惰性）与行动（触发）**，触发时才执行整张 DAG。
- **DAG 按依赖类型切分阶段**：窄依赖流水线执行，宽依赖触发 shuffle。
- 容错靠**血缘 + 持久化检查点**，重算路径与缓存密度成反比。
- 相比 MapReduce，内存缓存把每轮迭代的带宽穿越从 2 次降到 1 次（甚至 0 次），迭代场景快一个数量级。

在下一节，我们把 RDD 的"食谱式"编程升级为"声明式"的 SQL 体验——这就是 **Spark SQL 与 DataFrame**。
