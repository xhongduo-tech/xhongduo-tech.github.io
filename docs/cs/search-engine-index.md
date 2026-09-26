---
title: 全文检索引擎
date: 2026-09-08
section: cs
---

# 全文检索引擎

<div class="epigraph">
<p>独立搜索引擎把倒排段、打分、分片做成集群：近实时可见靠刷新，不是 SQL 提交即搜。库内 GIN 是子集。</p>
<footer>—— 据 Lucene 架构；Manning IR；倒排课的集群落地</footer>
</div>

[上一课](/cs/vector-db-ann)可与全文混合。本课专搜索引擎：Elasticsearch/Lucene 一族。缺口是相对库内全文：刷新间隔、副本分片、评分相关性、与主库 CDC 对齐。事务课的 D 在这里常是「近实时」，RPO 按刷新与 translog。

## 问题

文档进缓冲，refresh 成可搜段，search 多段归并，merge 后台——LSM 同构。缺口是 **SQL 事务 vs 可搜**：未 refresh 的文档搜不到，应用若以为提交即搜会错。CDC 从库到引擎是异步，读己之写要等 refresh 或搜主库。<span class="marginnote">「refresh」翻译过来就是：把内存里攒着的新文档封成一个只读小段，让它们从「写了但搜不到」变成「搜得到」。它约每秒一次，本质是在「可见快」和「段太碎要频繁合并」之间做取舍。</span>

分片：文档 id 哈希， scatter 查询再合并打分——分布式 top-k。相关性：BM25 在引擎，SQL `ORDER BY` 不是同一回事。

<span class="marginnote">Lucene 段与 translog。本课集群合同。库内倒排课已有 posting。向量可作为 Lucene 字段。</span>

## 方法

索引模板、mapping（分析器）。查询 DSL。与关系：主键同步、死信重试。权限：字段级安全 vs 后课行级安全在库。

容量：倒排内存、doc values 列存给排序聚合——引擎内 HTAP 雏形。

```mermaid
flowchart TD
  IDX["index 缓冲"] --> REF["refresh 成段"]
  REF --> SCH["多段检索打分"]
  REF --> MRG["merge"]
  DB["主库提交"] --> CDC["异步到引擎"]
```

## 机制

一致性：副本等待可配，类似半同步。脑裂有选举。Jepsen 后课对搜索集群也测过丢失。优化：filter 上下文走位图缓存，query 上下文打分。

同一次查询里，两种上下文走的是完全不同的计算路径——过滤部分能命中缓存复用，打分部分必须逐文档算：

```mermaid
flowchart LR
  Q["查询请求"] --> S{"拆分上下文"}
  S -- "filter 子句" --> BM["位图匹配命中/不命中"]
  BM --> CA["查位图缓存"]
  CA --> IT["多条件位图求交集"]
  S -- "query 子句" --> SC["逐文档算 BM25 分"]
  IT --> TOP["按相关分排序取 top-k"]
  SC --> TOP
```

与 zone：段上的 doc 值 min/max 跳过。<span class="marginnote">数字实例：一亿文档、refresh 周期 1 秒，意味着主库提交后最坏要再等约 1 秒新文档才可搜——这就是「读己之写」时要么按 _id 直查（走实时变更记录，不等 refresh）、要么读主库的原因。</span>

## 边界

本课不讲 SQLite 页。也不把搜索当唯一真相库——主库仍权威。湖仓分析另一路。

后课默认：全文集群近实时；提交即搜要等或走库。SQLite 架构：嵌入式整库一份文件，另一极端。

搜索引擎是特化执行器+倒排存储，不是 Selinger 的通用 SQL。<span class="marginnote">常见误区：初学者容易把所有条件都写进 query 上下文，以为「打分越全越准」——实际上纯过滤条件（状态=已发布、时间在某区间）放 filter 上下文既不算分还能吃位图缓存，同样的条件放 query 里就是白白重复计算。</span>

## 小结

- 段+refresh+merge 近实时可搜；与 SQL 提交不同步除非等待。
- 分片 scatter 打分；CDC 对齐主库。
- SQLite 架构下一课。
- 出处：Lucene；Manning IR；LSM 对照。
