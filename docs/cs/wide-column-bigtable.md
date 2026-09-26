---
title: 宽列 Bigtable
date: 2026-09-08
section: cs
---

# 宽列 Bigtable

<div class="epigraph">
<p>行键、列族、时间戳三维：稀疏、按行键范围切 tablet。不是 SQL 表，也不是 JSON 树；访问路径是行键设计。</p>
<footer>—— 据 Chang et al., Bigtable, OSDI 2006；Percolator 课的底盘</footer>
</div>

[上一课](/cs/document-db)是嵌套文档。本课是稀疏宽表：Bigtable 模型。缺口是键设计决定局部性——与分片键课同构，粒度到列族。Percolator 已用此底盘做事务；本课模型本身：单行原子、跨行无（除非上层）。

## 问题

表：行键排序，列族分别存（独立 compaction），单元格可多版本时间戳。读一行一列族顺序 I/O。扫描前缀。缺口是**把查询写成行键**：倒置索引、连接预编码进键，否则全表扫。与关系 schema 相反，这里 schema 主要是族与键约定。

tablet：行键范围分片，再平衡拆范围。底层 SSTable+memtable，即 LSM。

<span class="marginnote">直觉类比：行键像图书馆的书架编号——想找「某用户 9 月的订单」，就把用户 ID 和年月编进书号开头，一次走对书架；若书号开头是入库日期，同一个用户的订单就散落全馆，只能一层层翻。</span>

<span class="marginnote">Chang, Dean, Ghemawat et al. OSDI 2006。HBase、Cassandra 宽列变体（Cassandra 另有分区键/聚簇列）。本课 Bigtable 三维。</span>

## 方法

设计行键避免热点（不要纯时间戳开头）。列族按访问共现分，不要一列一族。GC：按版本数或年龄，水位像 MVCC。与 SQL 层：Spanner 在类似存储上加 SQL，键仍要懂。

布隆、zone：SST 级已有。R 树不在此模型核心。

<span class="marginnote">常见误区：拿自增时间戳当行键开头，新写入永远落在最后一个 tablet 上——单机先写满、其余节点闲着，这就是热点；正解是对键哈希撒盐或用反向时间戳，把写压力摊开。</span>

```mermaid
flowchart TD
  RK["行键有序"] --> TAB["tablet 范围"]
  CF["列族"] --> SST["独立 LSM"]
  TS["时间戳版本"] --> GC["版本 GC"]
```

## 机制

局部性：相邻行键一起扫。连接：应用层或预连接进同一行。shuffle 发生在上层 MR/Spark。事务：行内原子；Percolator 跨行。

<span class="marginnote">数字实例：GC 规则设「每单元格留 3 个版本」，同一单元格写第 4 次时最旧版本在 compaction 时被扔掉；在那之前读方仍能按时间戳点名取到旧版本。</span>

```mermaid
flowchart TD
  GET["按行键发起读"] --> MEM["先查 memtable"]
  MEM --> BLOOM{"布隆说各 SST 可能有此键？"}
  BLOOM -->|"否"| SKIP["跳过该 SST"]
  BLOOM -->|"是"| LOOK["SST 内定位键区间"]
  LOOK --> MERGE["合并多版本，取目标时间戳"]
  SKIP --> MERGE
  MERGE --> ANS["返回单元格"]
```

与文档：宽列扁平稀疏，路径用列名限定符，不是任意 JSON 深度。

## 边界

本课不讲属性图。也不把宽列当 Parquet。时序库常用宽列或倒排时间，后课专门压缩。

后课默认：宽列靠行键局部性；跨行事务要上层。图数据库：点边一等，多跳遍历是第一查询。

tablet 拆分就是范围分片，令牌环是另一放置。

## 小结

- Bigtable：有序行键、列族、版本；LSM tablet。
- 查询能力几乎等于键设计。
- 图数据库下一课。
- 出处：Chang et al., OSDI 2006；Percolator 对照。
