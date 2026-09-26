---
title: 湖仓与开放表格式
date: 2026-09-08
section: cs
---

# 湖仓与开放表格式

<div class="epigraph">
<p>对象存储上的 Parquet 不是表：没有事务、没有隔离、没有可靠的 schema 演进。表格式把数据湖补成可被引擎共享的关系。</p>
<footer>—— 据 Armbrust et al. 对 lakehouse；Delta Lake / Apache Iceberg / Apache Hudi 的公开设计</footer>
</div>

[上一课](/cs/olap-cube)假定有一张可维护的事实表。缺口是文件堆：数仓一体机贵，HDFS 上的目录又没有 ACID。本课钉湖仓与开放表格式；MapReduce 下一课才讲批计算模型。不把对象存储当新的 B+ 树重讲。

## 问题

湖：廉价存储 + schema-on-read，丢了主键、并发写、时间旅行。仓：强事务，锁进专有格式。表格式（Iceberg/Delta/Hudi）在对象存储上加：快照隔离、分区演进、列投影、可重放的提交日志。缺口是**元数据如何成为真值**，不是再夸一次列存。

<span class="marginnote">多引擎（Spark、Trino、Flink）共用同一快照，才叫开放。只给一个引擎写的「湖」仍是专有仓。</span>

<span class="marginnote">直觉类比：每次提交像一次 git commit——数据文件是只增不删的工作区，元数据日志是提交历史，所谓「时间旅行」不过是 checkout 回某个旧提交再看表。</span>

## 方法

读一次提交：manifest / transaction log 指向不可变文件。写：copy-on-write 或 merge-on-read。compaction 把小文件合并。与立方：物化视图可以挂在表快照上。与流：changelog 表是后课 exactly-once 的落点。

```mermaid
flowchart TD
  OBJ["对象存储文件"] --> META["快照 / 日志"]
  META --> ENG["多引擎只读同一版本"]
```

## 机制

不可变文件 + 原子换根指针 ≈ [影子分页](/cs/shadow-paging) 在对象存储上的亲戚，也像 LSM 切根。隔离来自「读一个快照 ID」，不是行锁；跨表 2PC 通常没有或很弱，不要当 Spanner。小文件问题来自列出与打开成本，compaction 合并——表级 LSM。delete file / merge-on-read 让读时合并删除位，读放大上升。

并发写：条件 put 或锁服务提交元数据，冲突则 abort 重试。时间旅行等于选旧快照。schema 演进写进元数据，读侧按字段投影，与 [模式迁移](/cs/schema-migration) 的文件版对齐。TDE 在对象存储侧，密钥合同单独写。

```mermaid
flowchart TD
  W["写入新的不可变数据文件"] --> META["写新元数据: 新快照指向新旧文件集"]
  META --> CAS["原子换根指针: 条件 put 或锁服务"]
  CAS -->|"成功"| NEW["后续读者拿新快照 ID"]
  CAS -->|"被抢先"| ABORT["提交作废, 整体重试"]
```

<span class="marginnote">数字实例：一个每 5 分钟 flush 一次的管道，一天能写出 288 个小 Parquet，对象存储光「列目录+开文件头」就能拖垮查询规划。compaction 把它们并成几个几百 MB 的文件，规划期元数据开销直降一个数量级——这就是「表级 LSM」的日常账。</span>

<span class="marginnote">常见误区：初学者容易以为湖仓能顶替 OLTP 数据库。表格式给的是快照隔离与秒级提交，没有行锁、没有毫秒级事务；高并发单行更新的场景（扣库存、改余额）仍然要回 OLTP，湖仓只管分析侧。</span>

## 边界

本课不比较 Iceberg / Delta / Hudi 的每一条 SQL。下一课 MapReduce 是计算模型，不是表。不要把湖仓写成已经取代 OLTP：提交延迟与行锁粒度都不在同一档。流写入必须控制批次，否则小文件打爆元数据。

后课默认：分析表用开放格式快照；跨引擎共享的是文件集，不是缓冲池。

## 小结

- 表格式把对象存储补上快照与演进，才成为多引擎的表。
- 元数据日志是真值；数据文件只追加；compaction 管小文件。
- MapReduce 下一课。
- 出处：Armbrust 等 lakehouse；Iceberg / Delta / Hudi 设计文档。
