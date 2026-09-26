---
title: 逻辑复制与 CDC
date: 2026-09-08
section: cs
---

# 逻辑复制与 CDC

<div class="epigraph">
<p>从 WAL 解出行级变化流：下游可异构、可过滤表。这是逻辑日志的消费者，不是备库再应用生理 redo。</p>
<footer>—— 据 Postgres logical decoding；Gray；CDC 实践；日志粒度课落地</footer>
</div>

[上一课](/cs/backup-pitr)把 WAL 当 PITR 原料。本课把同一 WAL（或触发器）当成变化数据捕获（CDC）流。主干 [复制与日志传送](/cs/replication-log) 偏物理。进阶：逻辑复制解码成 `INSERT/UPDATE/DELETE` 行像，给搜索索引、数仓、缓存。半同步下一课仍偏物理耐久。

## 问题

物理复制：同引擎、同版本、全库页。逻辑：表子集、转换、多下游。缺口是**解码与槽**：保持解码所需的 WAL 与旧行像（undo/堆版本），槽不消费则磁盘涨——与 MVCC 水位同一类钉子。顺序：事务提交序要在流里可见，否则下游聚合错。

全量+增量：先快照再从 LSN 接流，避免漏。DDL：逻辑流对模式演化敏感，迁移课的兼容窗口在此显现。

<span class="marginnote">PostgreSQL logical decoding、MySQL binlog 行模式。Debezium 一类是生态。本课机制。无虚构论文号。</span>

<span class="marginnote">CDC（变化数据捕获）翻译一下：不再每天半夜全量导一遍表，而是像装了水表一样，把数据库的每一笔 INSERT/UPDATE/DELETE 变成一条流水记录，实时递给下游的搜索索引、缓存或数仓——「捕获」的就是这股持续的变化流。</span>

## 方法

输出插件：把事务变化写成 JSON/protobuf。下游幂等：用 LSN 或主键+版本去重。恰好一次后课流系统再谈；库侧至少 at-least-once 加幂等键。

过滤：只发某些表，减少流量。大对象可跳过或另通道。

<span class="marginnote">复制槽像小区报箱：下游宕机不消费，报（WAL）就全堆在报箱里——压垮的不是报箱而是磁盘。几百 GB 的库，几天滞留的 WAL 就能塞满剩余空间；所以「忘删槽会胀爆磁盘」是运维第一事故，不删不用的槽要当常态清理。</span>

```mermaid
flowchart TD
  WAL["生理 WAL"] --> DEC["逻辑解码"]
  DEC --> SLOT["复制槽水位"]
  SLOT --> DOWN["下游异构消费者"]
  DOWN --> ACK["推进槽"]
```

## 机制

下游第一次接入怎么做到不漏不重？关键在把「快照」与「接流」在同一个 LSN 上对齐。

```mermaid
flowchart TD
  S0["开始：记下当前 LSN"] --> SNAP["对表做一致性快照"]
  SNAP --> SLOT["建复制槽，从该 LSN 起留存 WAL"]
  SLOT --> STREAM["快照完成后从 LSN 开始消费"]
  STREAM --> DEDUP["用主键+版本或 LSN 去重重叠部分"]
  DEDUP --> OK["快照与增量无缝衔接"]
```

<span class="marginnote">初学者容易以为流里的变化是逐条实时发布的。实际按事务提交序发布：长事务跑一小时，它攒的一万条改动会憋到提交那刻才整包涌出——下游必须能承受这种突发，而不是假设流量永远平滑。</span>

性能：解码 CPU、undo 读旧像。长事务一个大包提交时下游突发。与 SSI/锁无关直接，但未提交变化不应出现在流（按提交）。

冲突：多源写入下游要合并策略，本课单源。双向复制是另一事故源。

## 边界

本课不讲半同步 ACK。也不把 CDC 当 2PC。搜索引擎索引常用 CDC 填倒排。

后课默认：异构下游用逻辑 CDC+槽；同构 HA 用物理。半同步复制：提交等备库确认，换 RPO。

槽是消费者的 xmin，忘记 drop slot 会胀爆磁盘。

## 小结

- 逻辑复制从 WAL 解码行流，槽钉水位。
- 接流要快照对齐；DDL 要合同。
- 半同步复制下一课。
- 出处：logical decoding；Gray；CDC 实践。
