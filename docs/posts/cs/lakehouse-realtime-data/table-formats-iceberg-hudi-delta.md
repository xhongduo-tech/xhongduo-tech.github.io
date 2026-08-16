---
title: 表格式（Iceberg/Hudi/Delta 的 ACID 与时间旅行）
date: 2026-08-07
---

# 表格式（Iceberg/Hudi/Delta 的 ACID 与时间旅行）

<div class="epigraph">
<p>简洁是可依赖性的前提。</p>
<footer>—— 艾兹格 · 迪杰斯特拉（Edsger W. Dijkstra），图灵奖得主</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据湖仓与实时流处理 ｜ Kleppmann《Designing Data-Intensive Applications》Ch.3 & Ch.7 ｜ 2026-08-07</p>
</div>

## 为什么从表格式开始

上一节我们在存储引擎层面回答了「单个文件怎么组织」。

但湖仓的真相是：**一张表 = 成千上万个 Parquet 文件 + 一套关于它们的元数据**。谁来保证这堆文件对外表现为一张「原子、可回溯、schema 可演进」的表？答案就是**表格式（table format）**。

Kleppmann 在第 3 章讲过「数据仓库/数据湖」对事务与索引的取舍，第 7 章则系统解释了 ACID 事务为什么难——而 Iceberg、Hudi、Delta Lake 正是把第 7 章的答案工程化到了数据湖上。

**这是湖仓一体的「心脏」**，也是后续 CDC 入湖、时间旅行、流批一体赖以成立的基础设施。

## 1 为什么文件之上需要一层事务

Hive 时代的悲剧很具体：一张表在 HDFS 上是一堆按分区组织的文件。

写入方 A 覆盖写一个分区、写入方 B 同时读这个分区——**读方可能看到写了一半的目录，写方之间互相覆盖**。

Kleppmann 在第 7 章指出事务要解决的四大性质（ACID）：

- **原子性**：要么全成，要么全无。
- **一致性**：约束不被破坏。
- **隔离性**：并发不互相污染。
- **持久性**：宕机不丢。

把 ACID 搬到「一堆廉价文件」上，核心难题是：**文件系统没有原子目录操作**，无法保证「提交成功」与「文件可见」是同一瞬间。<span class="marginnote">Hive 用「目录改名」模拟提交，但改名不是原子的、也没有并发控制；Spark 的 Hive 写曾因「先写临时目录再 move」在断电时留下孤儿文件——这不是工程偷懒，而是文件系统层面的根本局限，逼出了表格式。</span>

## 2 Iceberg：元数据即真相

**Iceberg**（Netflix 2018 年开源）的回答最「元数据化」：**表的每一份数据都是一张不可变的快照（snapshot）**。

一个快照由**清单（manifest）**描述「这张快照包含哪些数据文件」，最顶层用一个**指向当前快照的指针**代表「表的最新状态」。元数据层级从上到下是：

- **表元数据（table metadata）**：记录当前快照指针、schema、分区规格。
- **快照清单列表（manifest list）**：记录某快照包含哪些 manifest。
- **清单（manifest）**：记录实际数据文件与它们的分区、统计信息。

提交 = 原子地移动最顶层的快照指针：

- **写**：新数据文件写为新的 manifest，全部就绪后，把表指针从旧快照切到新快照——**一次原子切换**。
- **读**：拿到指针，就拿到整张表的完整文件清单，永不看到半成品。
- **时间旅行**：指针历史上每个位置都对应一个快照，`SELECT ... AS OF <snapshot-id>` 即可查任意历史版本。<span class="marginnote">Iceberg 的快照指针与 Git 的 commit 指针、ZFS 的快照几乎同构——「把状态机做成不可变快照 + 指针切换」，这套思路也贯穿了下一节《消息系统》里 Kafka 的日志与 Flink 的 Checkpoint。</span>

Iceberg 的另一个优势是**隐藏分区（hidden partitioning）**：用户写分区表达式（如 `date_trunc('day', ts)`），引擎自动生成分区与谓词下推，用户无需手工维护分区目录，这与 Hive 的「手动建分区目录」形成鲜明对比。

### 并发控制：乐观锁与快照隔离

Iceberg 用**乐观并发**处理多写者：提交时校验「我要基于的快照是否还是当前快照」，若是则原子切换，否则失败重试。

这实现了**快照隔离**——每个读方看到自己开始时的快照，写方互不阻塞。Kleppmann 第 7 章讲的隔离级别，在这里以「文件级的 CAS（compare-and-swap）」落地，比行级锁更粗、但足够湖仓的写模型使用。

## 3 Hudi 与 Delta：两条互补的路线

**Apache Hudi**（Uber 2017 年开源，Hadoop Upserts Deletes and Incrementals）主打**增量更新与 Upsert**：

- 它维护一张**记录级索引**（Bloom 过滤器 + 文件级记录 key 映射），支持对已有记录做更新与删除。
- 这对「上游有 CDC 更新」的场景极其关键。
- Hudi 用**文件组（file group）**组织数据，更新落到文件组内。
- 通过**合并（merge-on-read）**或**写时复制（copy-on-write）**两种模式实现：merge-on-read 把读放大换成写快，copy-on-write 相反。

**Delta Lake**（Databricks 2020 年开源）则把事务日志（transaction log）显式地做成一张**append-only 的 JSON 日志表**：

- 每次提交在日志末尾追加一条「本次改了哪些文件」的记录。
- 表状态 = 日志重放的结果。
- Delta 与 Spark 绑定最深，因此在 Databricks 生态内体验最好。<span class="marginnote">Delta 的「日志即真相」与事件溯源（event sourcing）完全同构——Kleppmann 在第 11 章把它列为数据库与流融合的两大范式之一。日志里每一条记录都可重放、可审计，天然支持时间旅行与 schema 演进。</span>

## 4 核心对比表：三大表格式

| 维度 | Iceberg | Hudi | Delta Lake |
| --- | --- | --- | --- |
| 开源方/时间 | Netflix，2018 | Uber，2017 | Databricks，2020 |
| 元数据模型 | 快照指针 + manifest | 文件组 + 记录级索引 | append-only JSON 事务日志 |
| 强项 | 快照隔离、隐藏分区、多引擎 | Upsert/增量、CDC 友好 | 与 Spark/Databricks 深度集成 |
| 更新删除 | 支持（copy-on-write） | 原生 Upsert，两种合并模式 | 支持 |
| 引擎支持 | Flink/Spark/Trino/StarRocks 等 | Flink/Spark 等 | Spark/Databricks 为主 |
| 典型选型 | 开放湖仓、多引擎共存 | 高频更新、增量同步 | Databricks 生态、数据中台 |

**辨析｜易错点：** 表格式不是「数据库」，它不提供查询执行，只提供「表语义 + 事务元数据」；查询仍由 Flink/Spark/Trino 这类引擎完成。

选型也不是「谁更好」，而是「你的更新频率、引擎栈、多引擎需求」落在哪：偏开放与多引擎选 Iceberg，偏高频更新与增量同步选 Hudi，偏 Databricks 闭环选 Delta。

### 术语速查表

| 术语 | 英文 | 一句话含义 |
| --- | --- | --- |
| 表格式 | table format | 描述「文件集合 + 元数据」构成一张表的规范 |
| 快照 | snapshot | 表在某个时刻的不可变完整状态 |
| 时间旅行 | time travel | 按快照 ID / 时间查询历史版本 |
| 隐藏分区 | hidden partitioning | 引擎按分区表达式自动生成分区与裁剪 |
| 乐观并发 | optimistic concurrency | 提交前校验版本，冲突则重试 |
| Upsert | upsert | 更新已存在记录、插入新记录

## 5 时间旅行与 schema 演进为什么值钱

时间旅行的价值常被低估，它其实是三件事的合体：

- **审计**：改坏了数据能回滚到事故前快照。
- **可复现实验**：算法与报表基于同一时刻数据，结果可对账。
- **增量读取**：消费两次快照之间的差异，正是 CDC 入湖与增量物化的基础。

schema 演进（加列、改列名、放宽类型）在 Hive 时代要重写整张表，在表格式里只是一次新的快照提交——旧快照与旧 schema 并存、互不影响。<span class="marginnote">时间旅行也带来<strong>数据治理责任</strong>：旧快照占据存储且不能随意删除，湖仓要做「快照过期清理」与「孤儿文件回收」——这是《数据编排与治理》篇会接续的话题。</span>

## 6 小结

- **表格式**解决「一堆文件如何对外表现为一张 ACID 表」：原子提交、快照隔离、时间旅行、schema 演进。
- **Iceberg** 用「快照指针 + manifest」实现原子切换与隐藏分区，是开放多引擎的默认选择。
- **Hudi** 用「文件组 + 记录级索引」原生支持 Upsert 与增量读取，CDC 场景友好。
- **Delta Lake** 用「append-only 事务日志」记录一切提交，与 Spark/Databricks 集成最深。
- 三大格式与 Kleppmann 讲的 ACID（第 7 章）与「数据库与流」（第 11 章）思想一一呼应——快照、日志、事件溯源是共同的语言。

在下一节，我们转向「数据怎么被传进来」——湖仓的上游是**消息系统**。我们从 Kafka 的日志抽象、ISR 副本协议讲起，一直讲到精确一次语义。
