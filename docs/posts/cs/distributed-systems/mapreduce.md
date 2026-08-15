---
title: MapReduce：编程模型与容错机制
date: 2026-08-07
---

# MapReduce：编程模型与容错机制

<div class="epigraph">
<p>用户只需写 Map 和 Reduce 两个函数，系统负责把计算分布到上千台机器、并扛住机器的随时失败。</p>
<footer>—— 杰弗里 · 迪恩与桑贾伊 · 格玛沃特（Dean & Ghemawat），MapReduce: Simplified Data Processing on Large Clusters，OSDI 2004</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ MIT 6.824 第1讲 / MapReduce 论文 2004 ｜ 2026-08-07</p>
</div>

## 为什么从 MapReduce 开始

GFS 解决了「数据存哪」，MapReduce 解决「数据怎么算」——它把「大规模分布式计算」抽象成两个函数（Map 与 Reduce），让用户不用管并行、分布、容错。它是「大数据」运动的第一块基石，也是理解「在故障常态下做确定性计算」的最佳教材。<span class="marginnote">MapReduce 由 Dean & Ghemawat 在 OSDI 2004 发表，2000 年代中期统治了 Google 的索引构建、日志分析等大规模计算。它借鉴了函数式编程的 `map`/`reduce`，但真正的创新在<strong>分布式执行与容错</strong>——用户代码只需写纯函数，系统的复杂性全部藏进框架。</span>

## 1 编程模型：两个函数解决一切

MapReduce 让用户只写两个函数：

**Map**：`map(in_key, in_value) -> [(out_key, out_value)]`——把一条输入记录映射成若干中间键值对。并行、无依赖、可任意分布。
**Reduce**：`reduce(out_key, [out_value]) -> [out_value]`——把同一中间键的所有值归约成结果。

经典例子：单词计数。Map 对每行文本输出 `(word, 1)`；Reduce 对每个 word 的所有 1 求和，输出 `(word, count)`。系统把整个流程（切分输入、调度、shuffle、容错）全部接管——**用户代码与分布式执行完全解耦**。

为什么这两个函数够用？因为**大量数据处理可以表达为「并行映射 + 按键归约」**：索引构建、排序、逆索引、分布式 grep、URL 统计——都是 map-reduce 结构。模型简单，但覆盖极广。

### 数值算例：单词计数的 Map/Reduce 展开

输入 3 个分片，各含若干行文本：

- **Map（分片 1）**：`"the cat", "the dog"` → `(the,1),(cat,1),(the,1),(dog,1)`。
- **Map（分片 2）**：`"the bird"` → `(the,1),(bird,1)`。
- **Shuffle 按键分组**：`the:[1,1,1]`、`cat:[1]`、`dog:[1]`、`bird:[1]`。
- **Reduce**：`the → 3`、`cat → 1`、`dog → 1`、`bird → 1`。

**关键**：Map 之间完全并行无依赖，Reduce 只对「同一键」归约——shuffle 把键聚合，Reduce 才能按组求和。模型看似简单，但「并行 Map + 按键归约」恰好覆盖了大量数据处理模式。

## 2 执行流程：六步流水线

一次 MapReduce 作业的执行：

1. **切分输入**：把输入文件切成 M 个分片，分片是 Map 任务的输入单元。
2. **Master 调度**：一个 master 节点把 M 个 Map 任务、R 个 Reduce 任务分派给空闲 worker。
3. **Map 执行**：worker 读分片，调用 Map 函数，把中间输出按「中间键的哈希」写入本地磁盘的 R 个分区（**中间结果在本地，不跨网络**）。
4. **Shuffle（洗牌）**：Map 完成后，master 通知 Reduce worker 去各 Map worker 拉取自己负责分区的中间数据——**跨网络传输**。
5. **Reduce 执行**：Reduce worker 对拉到的中间数据按键排序、分组，调用 Reduce 函数，把结果写入输出文件。
6. **完成**：master 汇总所有输出文件，作业完成。

**关键设计**：Map 的中间结果**写本地磁盘**而不是网络（减少网络流量）；Reduce 才跨网络拉数据（shuffle）。这个「本地计算、远程归约」的结构是 MapReduce 性能的核心。<span class="marginnote"><strong>辨析｜易错点：</strong>「Map 输出写本地」常被误解为「MapReduce 不 shuffle」。其实 shuffle 是 MapReduce 的固有步骤（第 4 步），只是发生在「Map 完成后、Reduce 开始前」——中间结果先落本地，再由 Reduce 端主动拉取。数据本地性（data locality）优化的对象是「Map 读输入」：尽量让 Map worker 读本机 GFS 上的数据，避免输入跨网络。</span>

## 3 容错机制：重执行 + 备份任务

MapReduce 的容错哲学是**「失败就重做」**——因为它处理的是确定性计算，重做结果相同，所以容错可以极简单：

**Worker 失败**：master 定期 ping worker；ping 超时即判定失败。该 worker 上**已完成的 Map 任务**需要重执行（它的中间结果在本地，worker 挂了就丢了）；已完成的 Reduce 任务不用（结果在 GFS 上）。master 把失效任务重新调度给其他 worker。
**Master 失败**：整个作业中止，由用户重跑——当时 Google 的做法（master 单点，失败概率低，重跑可接受）。
**备份任务（backup task / straggler mitigation）**：作业接近完成时，master 把「还在跑的任务」在空闲 worker 上**再复制一份**——谁先完成用谁。这解决**掉队者（straggler）**问题：一个慢 worker（磁盘慢、CPU 被占）拖住整个作业，备份任务让慢任务有「竞争者」。

**容错为什么这么简单就够**：因为 Map 和 Reduce 都是**确定性纯函数**——同一个输入在任何 worker 上重跑，输出相同。重执行不会产生「重复计算」问题（对输出文件而言是幂等的）。这就是「确定性计算 → 简单容错」的正反馈：**计算模型越确定，容错越廉价**。<span class="marginnote">备份任务是「用冗余换延迟」的经典：掉队者的成因多样（慢磁盘、坏网络、CPU 竞争），无法精确预测，那就「并行跑两份、用快的那份」。它在实践中能显著缩短作业延迟（据论文约 44% 的作业受益），也是后来 Spark、Flink 处理「倾斜任务」的思想源头。</span>

## 4 公式解析：数据本地性如何决定性能

把「Map 任务的数据本地性」量化。设 $N$ 个 Map 任务，其中 $L$ 个在「持有输入副本的机器」上执行：

$$
\text{local fraction} = \frac{L}{N}, \qquad
\text{作业 Map 阶段的网络流量} \propto (1 - \text{local fraction})
$$

拆解：

- 输入是 GFS 上的 3 副本文件；每个 Map 任务「读某个分片」——若调度到的 worker 恰好有该分片的一个副本，则**本地读**（无网络）；否则跨网络拉。
- **本地化率（local fraction）**：在「有本地副本」的机器上执行的比例。GFS 副本数为 3 时，理论本地化率可达 2/3 以上（master 优先调度到有副本的机器）。
- 网络流量与「非本地比例」成正比——本地化率越低，shuffle 前的输入读取越贵。

这条式子的工程含义：**「把计算搬到数据旁边」比「把数据搬到计算旁边」便宜得多**。MapReduce 的 master 调度策略（优先本地化）、GFS 的 3 副本（提供本地读的机会）、以及后来的 Spark「数据本地性级别」都是这个原理的不同实现——**数据本地性是分布式计算性能的第一杠杆**。<span class="marginnote">这也是「存储与计算同机」架构（GFS+MapReduce、HDFS+Spark）的理论根基：如果存储与计算分离（如对象存储 + 独立计算集群），每次 Map 都要跨网络读输入，本地化率归零——现代数据湖（S3 + Spark）正是这个代价的承受者，靠「缓存 + 列存裁剪」弥补。</span>

## 5 MapReduce 的局限与遗产

MapReduce 的成功也暴露了局限，这些局限催生了后续框架：

- **中间结果反复落盘**：每步 Map→shuffle→Reduce 都写磁盘——迭代式计算（机器学习）被磁盘 IO 拖死。
- **无流式/增量**：作业是「批」——不适用于需要低延迟的流处理。
- **表达力受限**：图计算、复杂 DAG 依赖要用多轮作业拼接。

遗产：**Spark** 把中间结果放内存（RDD）解决迭代；**Flink** 用流处理统一批流；**Dataflow/Beam** 抽象出统一的批流模型。但「Map/Reduce 抽象 + 数据本地性 + 失败重执行」的思想，贯穿所有现代大数据框架。

### 术语速查表

| 术语 | 一句话含义 |
| --- | --- |
| Map | 输入记录 → 若干中间键值对 |
| Reduce | 同一键的所有值 → 归约结果 |
| Shuffle | 把中间键值按键拉取归组 |
| Master | 调度 M 个 Map 与 R 个 Reduce |
| 数据本地性 | 在持有输入副本的机器上执行 Map |
| 备份任务 | 接近完成时复制慢任务，用快者 |

## 6 对照表：MapReduce 与后续框架

把 MapReduce 放进「大数据框架谱系」里定位，看清每代解决什么问题：

| 维度 | MapReduce | Spark | Flink |
| --- | --- | --- | --- |
| 中间结果 | 落磁盘 | 内存（RDD） | 内存 + 状态后端 |
| 计算模型 | 批（Map/Reduce） | 批 + 微批 | 流（批是特例） |
| 迭代计算 | 每轮落盘，慢 | 内存迭代，快 | 流式迭代 |
| 流式处理 | 无 | 微批近似 | 原生流 |
| 容错 | 重执行 + 备份 | 血统（lineage）重算 | 检查点 + 重放 |

**一句话读懂谱系**：MapReduce 定义了「分布式批处理」的骨架，Spark 用内存把迭代算快，Flink 用流把批流统一——每一代都在「减少落盘、降低延迟」这条线上前进，但「并行 Map + 按键归约 + 失败重执行」的底子没变。

**辨析｜易错点：**「Spark/Flink 淘汰了 MapReduce」只说对一半——MapReduce 的「容错 = 重执行」思想在 Spark（lineage）、Flink（checkpoint 重放）里原样继承。<strong>被淘汰的是「每步落盘的实现」，不是「确定性计算 + 重执行」的范式</strong>。理解范式 vs 实现的区别，才不会被「新框架取代旧框架」的叙事带偏。<span class="marginnote">6.824 的教学意义：MapReduce 论文是最适合「第一次学分布式系统」的材料——它把所有问题（调度、容错、数据本地性、掉队者）都放在一个极简模型里。<strong>先把 MapReduce 的每一层搞透，再学 Spark/Flink 就是在「往已懂的骨架上贴肌肉」</strong>。</span>

## 7 小结

- **MapReduce** 用 Map（并行映射）与 Reduce（按键归约）两个函数抽象大规模分布式计算。
- 六步流程：切分 → 调度 → Map（本地写中间结果）→ Shuffle（跨网络拉取）→ Reduce → 完成。
- **容错 = 失败重执行 + 备份任务**：确定性纯函数让重做结果相同，容错因此极廉价。
- **数据本地性**是性能第一杠杆：把计算调度到持有输入副本的机器上，避免跨网络读。
- 局限：中间结果反复落盘、无流式增量——催生了 Spark（内存迭代）、Flink（批流统一）。
- 遗产：Map/Reduce 抽象与「确定性计算 → 简单容错」的思想贯穿现代大数据框架。
- 备份任务用冗余换延迟：掉队者并行跑两份、用快的那份，约 44% 作业受益。
- 数据本地性是第一杠杆：把计算调度到持有输入副本的机器上，避免跨网络读。
- Map 中间结果写本地、Reduce 才跨网络拉——「本地计算、远程归约」是性能核心。

在下一节，我们看 GFS + MapReduce 之上的结构化存储——**Bigtable：数据模型与 SSTable**。
