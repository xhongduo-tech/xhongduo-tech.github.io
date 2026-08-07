---
title: TiDB：架构、TiKV 与分布式 SQL 执行
date: 2026-08-07
---

# TiDB：架构、TiKV 与分布式 SQL 执行

<div class="epigraph">
<p>把 MySQL 的体验、Spanner 的架构、LSM 的性能装进一个系统——这就是 TiDB。</p>
<footer>—— 平凯星辰（PingCAP，TiDB 开发者）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第19章 NoSQL 与 NewSQL ｜ 2026-08-07</p>
</div>

## 为什么从 TiDB 开始

Spanner 是 Google 内部系统，**TiDB** 是开源的国产 NewSQL，把 Spanner 的「分片 + 共识 + 分布式事务」思想落成可用系统，且**兼容 MySQL 协议**。它是理解「NewSQL 如何工程化」的最佳样本：**TiDB（计算层） + TiKV（存储层） + PD（元数据）**三层架构、Raft 复制的 region、分布式 SQL 执行（算子下推、并行聚合）。这一节把 TiDB 的架构拆开——**它是第 16、17 章全部理论的「毕业设计」**。

## 1 三层架构

TiDB 的三个核心组件：

- **TiDB Server（计算层）**：无状态 SQL 引擎——解析 SQL、生成执行计划、协调分布式执行。可水平扩展（加 TiDB = 加查询能力）。
- **TiKV（存储层）**：分布式 KV 存储——数据按行键有序，region 分片 + Raft 复制。LSM 树存储引擎。
- **PD（Placement Driver，元数据层）**：管理 region 的分布与调度、分配全局时间戳（TSO）、Raft 选主。

**核心要点：TiDB = 计算与存储分离的 NewSQL。** 计算层无状态（随便扩）、存储层分片 + 共识（数据可靠）、元数据层调度（region 均衡）——三层各司其职，互不耦合。

## 2 TiKV：Raft 复制的 KV 层

**TiKV** 是底层 KV 存储，架构：

- **数据 = KV**：把表的行编码成 `(table_id, row_id)` 为键的有序 KV——**关系表落到 KV 上**。
- **Region（分片）**：KV 按键范围切成 **region**（默认约 96MB）——每个 region 是一个** Raft 组**（第 16 章）。
- **Raft 复制**：每 region 多个副本（默认 3），Raft 保证一致——**region 分裂自动再平衡**。
- **存储引擎**：**RocksDB（LSM 树）**——写走 memtable，顺序刷盘（第 10 章）。

**公式解析：Region 分裂与扩展**

region 超过阈值时**分裂**（类似 B+ 树节点分裂）：

$$
\text{region 大小} > S_{max} \Rightarrow \text{分裂成两个}，\text{PD 调度到不同节点}
$$

- **第一步，分裂**：region 数据量超阈值，按键范围一分为二。
- **第二步，调度**：PD 把新 region 迁移到负载低的节点。
- **第三步，线性扩展**：加节点 → 更多 region 位置 → 数据与写分散——**存储与写吞吐水平扩展**。
- **第四步，与 HBase 对比**：HBase region 由 HMaster 管理；TiKV 用 PD + Raft——**无单点 + 自动均衡**。

## 3 分布式 SQL 执行

TiDB 的 SQL 执行流程：

1. **TiDB 解析 SQL** → 逻辑计划 → 物理计划（第 12 章优化）。
2. **计划拆分**：把算子分配到各 region——**执行下推（pushdown）**。
3. **各 TiKV 并行执行**：本地扫描、过滤（`cop` 任务）。
4. **结果汇总**：TiDB 层做连接、聚合、排序。

**核心要点：TiDB 用「算子下推」把计算推到数据所在地。** 让每个 TiKV 在**本地**做过滤/聚合，只把**小结果**回传 TiDB——避免把海量数据传到中心再算。这是「数据本地性（data locality）」的核心思想。

**公式解析：算子下推的收益**

设 region $i$ 有 $n_i$ 行，谓词选择率 $s$，下推 vs 不下推的传输量：

$$
\text{传输}_{\text{下推}} = \sum_i n_i \cdot s, \qquad
\text{传输}_{\text{不下推}} = \sum_i n_i
$$

- **第一步，下推**：各 region 本地过滤（$n_i \cdot s$ 行），只传结果。
- **第二步，不下推**：全量数据传回 TiDB 再过滤（$n_i$ 行）。
- **第三步，收益**：$s$ 越小（选择性越高），下推省得越多——**过滤越狠，下推越值**。
- **第四步，更广**：不只是过滤，**聚合、甚至部分连接**也能下推（TiFlash 支持向量化聚合下推）——**「把计算移到数据」是分布式 SQL 的核心原则**。

## 4 TiDB 的分布式事务

TiDB 采用 **Percolator 模型**（Google Percolator 的事务模型）：

- **时间戳**：PD 分配**全局单调时间戳（TSO）**——所有事务拿全局序。
- **2PC 变体**：事务跨 region 用**两阶段提交**（第 17 章），但用**时间戳 + 锁**而非锁表。
- **MVCC**：TiKV 行级多版本（第 14 章）——快照读。
- **隔离级别**：默认**可重复读（快照隔离）**；可串行化可选。

**核心要点：TiDB 用「TSO + Percolator 2PC + MVCC」实现分布式事务。** 它比 Spanner 便宜（TSO 单点分配 vs TrueTime 原子钟），牺牲一点「外部一致性」（TSO 是逻辑时钟，不保证实时序）换实现简单——**TiDB 是「工程可行的 Spanner 近似」**。

**辨析｜易错点：** **TSO 的全局单调**保证「事务有全局序」，但不保证「与真实时间一致」——所以 TiDB 默认快照隔离而非外部一致。**「要外部一致」选 Spanner 系，「要 MySQL 兼容 + 可扩展」选 TiDB**——一致性档位是选型的关键维度。

## 5 小结

- TiDB 三层：**TiDB（计算）+ TiKV（存储）+ PD（元数据）**——计算存储分离。
- TiKV：KV 有序 + **region 分片 + Raft 复制** + RocksDB（LSM）。
- 分布式 SQL：**算子下推**到 region 本地执行——数据本地性。
- 分布式事务：**TSO 全局时间戳 + Percolator 2PC + MVCC**——快照隔离。
- TiDB 是「工程可行的 Spanner 近似」——MySQL 兼容 + 水平扩展。

在下一节，我们进入第 20 章——**列存与 OLAP**，先看工作负载差异。
