---
title: 宽列存储：BigTable 与 Cassandra
date: 2026-08-07
---

# 宽列存储：BigTable 与 Cassandra

<div class="epigraph">
<p>稀疏的行、动态的列、海量的写——宽列存储是为「写不完的数据」而生的。</p>
<footer>—— 杰夫 · 迪恩（Jeff Dean，BigTable 作者）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第19章 NoSQL 与 NewSQL ｜ 2026-08-07</p>
</div>

## 为什么从宽列存储开始

**宽列存储（wide-column store）**介于「文档」与「关系」之间：行有行键，但列按**列族（column family）**分组且**可动态变化**——每行的列可以不同（稀疏表）。它继承关系库的「行列」心智，却提供 NoSQL 的海量写扩展。这一节讲宽列的数据模型（行键、列族、列限定符、时间戳），以及两大代表的架构差异：**BigTable/HBase**（主从 + 范围分区 + LSM）与 **Cassandra**（对等 + 一致性哈希 + 可调一致）。宽列存储是「海量写入」场景的答案。

## 1 宽列的数据模型

宽列存储的层次：

```
表 (Table)
  └─ 行键 (row key)：行的唯一标识，按序/哈希排序
      └─ 列族 (column family)：逻辑分组的列集合（预先定义）
          └─ 列限定符 (column qualifier)：列名（可动态添加）
              └─ 值 + 时间戳（多版本）
```

**核心要点：行键定位行，列族分组列，列限定符动态扩展。** 与关系表「固定列」不同——**每行的列集可以不同**（稀疏表），加列不改表结构。<span class="marginnote">直觉对照：宽列表像「一张每行都不同的表」——行键是主键，列族是「预定义的分组」，组内的列可以随行变化。它比文档「更结构化」（有行键、列族），比关系表「更灵活」（列动态）。</span>

**BigTable 示例**：

```
"com.example.www" → 列族 "anchor"：列 "cnnsi.com" = "CNN"
                  → 列族 "contents"：列 "" = "<html>..."
```

## 2 BigTable / HBase：主从架构

**BigTable（Google）**与开源版 **HBase** 的架构：

- **主从结构**：一个 HMaster 管理分区（region），多个 RegionServer 服务数据。
- **数据分区**：按**行键范围**分成 **region**（表被切成一串有序 region）——类似 B+ 树叶子。
- **存储引擎**：**LSM 树**（第 10 章）——写走 memtable + WAL，读查多级 SSTable。
- **列族存储**：每个列族单独存储（类似列存）——按列族优化压缩与读。

**核心要点：BigTable/HBase 是「主从 + 范围分区 + LSM」。** 它擅长**大范围的顺序扫描**（按行键扫）、海量写入；单点 HMaster 是可用性短板（需要 ZooKeeper 协调）。

**公式解析：LSM 在 HBase 的读路径**

按行键 $k$ 读，查多级结构：

$$
T_{\text{read}} = \text{memtable 查} + \sum_{\text{每层 SSTable}} \text{二分}
$$

- **第一步，memtable 优先**：最近写可能在内存，先查。
- **第二步，多级 SSTable**：逐层二分找键（Bloom filter 跳过不存在层）。
- **第三步，写快读慢**：LSM 写路径顺序追加（快），读要查多层（慢）——**HBase 优化写、读用 Bloom filter + 缓存补**。
- **第四步，与 B+ 树对比**：B+ 树读快写慢；LSM 写快读慢——**HBase 选 LSM 因为「写海量数据」是主诉求**。

## 3 Cassandra：对等架构

**Cassandra**（源自 Dynamo 论文）的架构与 HBase 截然不同：

- **对等（peer-to-peer）**：无主节点——所有节点同等，任何节点可服务任何请求。
- **分区**：**一致性哈希**（含虚拟节点）——数据均匀分布。
- **复制**：每行复制到 $N$ 个节点（按复制因子），**可调一致性**（W/R 参数，第 18 章）。
- **存储**：LSM 树（SSTable + memtable），**Compaction 策略**可配（Size-Tiered/Leveled）。

**核心要点：Cassandra 是「无单点 + 可调一致 + 高可用」的宽列存储。** 它没有 HMaster 单点——任何节点故障，请求路由到其他副本；分区时可用性优先（AP），靠读修复与提示移交（hinted handoff）最终一致。

**辨析｜易错点：** **Cassandra 的「主键」决定分区与排序**：`PRIMARY KEY (partition_key, clustering_key)`——分区键决定「数据放哪个节点」，聚簇键决定「分区内的排序」。**查询必须带分区键**（否则全集群扫描）——这是 Cassandra 建模与关系库最大的心智差异。

## 4 HBase vs Cassandra：对比

| 维度 | HBase（BigTable） | Cassandra |
| --- | --- | --- |
| 架构 | 主从（HMaster + RegionServer） | **对等（无主）** |
| 分区 | 行键范围（有序 region） | **一致性哈希** |
| 一致性 | 单主强一致（默认） | **可调（W/R）** |
| 可用性 | 主节点单点风险 | 高（无单点） |
| 范围扫描 | ✅（行键有序） | ⚠️（分区内有序） |
| 写模型 | LSM（主写） | LSM + 对等写 |
| 故障处理 | ZooKeeper 协调 | 提示移交 + 读修复 |
| 代表场景 | 海量扫描 + 稀疏表 | 海量写入 + 高可用 |

**核心要点：HBase 偏「强一致 + 范围扫」，Cassandra 偏「高可用 + 写扩展」。** 两者都基于 LSM 处理海量写，但在「一致性与可用性」的谱系上站了不同位置——HBase 偏 CP、Cassandra 偏 AP（第 18 章）。

## 5 小结

- 宽列模型：**行键 + 列族 + 动态列限定符 + 时间戳**——稀疏行、灵活列。
- **BigTable/HBase**：主从 + 范围分区 + LSM，擅长顺序扫描、写海量，单主可用性短板。
- **Cassandra**：对等 + 一致性哈希 + 可调一致，无单点高可用，偏 AP。
- Cassandra 主键 = 分区键 + 聚簇键——**查询必须带分区键**。
- 两者都用 LSM（写快读慢），但在一致性谱系上站位不同。

在下一节，我们看「关系优先」的 NoSQL——**图数据库与图查询**。
