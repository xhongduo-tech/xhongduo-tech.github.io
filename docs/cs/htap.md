---
title: HTAP
date: 2026-09-08
section: cs
---

# HTAP

<div class="epigraph">
<p>同一份（或可对齐快照的）数据既要短事务又要扫描分析。引擎用双格式、双集群或可更新列存，隔离的是资源与新鲜度合同。</p>
<footer>—— 据 Özcan et al. HTAP 综述；C-Store/HyPer 谱系；OLTP-OLAP 边界</footer>
</div>

[上一课](/cs/storage-compute-separation)让计算弹性。本课不挂盘。缺口是工作负载混合：OLTP 要点查与索引，OLAP 要列扫。传统做法 ETL 到数仓，新鲜度小时级。HTAP（Hybrid Transactional/Analytical Processing）要秒级或事务级新鲜。分布式序列接近收口，NewSQL 下一课归产品族。

## 问题

一条缓冲池、一种页格式难同时最优。路径：① 行存主 + 异步列副本（CDC）；② 可更新列存 + delta；③ 内存行 + 列批次（HyPer 热切）；④ 存算分离上两计算引擎读同一 WAL。缺口是**新鲜度、隔离、资源隔离**：分析扫打爆 OLTP 缓冲，2Q 抗扫描不够还要 QoS、WLM。

<span class="marginnote">术语翻译：OLTP 是「一笔一笔办业务」——收银、转账，每次碰几行、要求毫秒响应；OLAP 是「月底翻全年账本做盘点」——一次扫几亿行、只要吞吐不要即时。HTAP 就是让同一份数据同时支撑这两种活法，难点在两种活法的最优数据格式相反。</span>

快照：分析用只读快照不挡 OLTP 写，但 GC 水位或列副本延迟仍在。

<span class="marginnote">HTAP 一词工业界常用。HyPer 热切、TiFlash 列副本质是工程点。本课合同不是商标。</span>

## 方法

声明 SLA：分析可落后 $x$ 秒。路由：短查询走行，扫走列。物化视图在 HTAP 里当列侧预聚合。计划：同一 SQL 两套代价模型。

与 shuffle：列侧 MPP 仍要连接。事务写只打行侧，列侧追。

```mermaid
flowchart TD
  W["OLTP 写"] --> ROW["行引擎"]
  ROW --> CDC["复制 / delta"]
  CDC --> COL["列引擎"]
  Q["查询"] --> ROUTE["按形状路由"]
```

## 机制

一致性：最强是同一 MVCC 戳两引擎可读；常见是列落后。读己之写：刚写去列侧可能没有，要路由到行或等。TDE 两份。vacuum 与列 compaction 双重 GC。

校准：两套 I/O。计划回归：优化器突然把 OLTP 语句送到列扫。

刚写进行的数据，为什么在列侧可能查不到？按新鲜度路由看：

```mermaid
flowchart TD
  Q["查询到达"] --> SNAP["查列侧快照水位"]
  SNAP --> F{"要的数据已进列副本？"}
  F -->|是| COL["列引擎扫描"]
  F -->|"否（读己之写）"| R{"合同允许落后 x 秒？"}
  R -->|能等| COL
  R -->|不能等| ROW["回退行引擎查询"]
```

<span class="marginnote">数字实例：传统 ETL 到数仓，新鲜度是小时级；HTAP 的 CDC 列副本把它压到秒级。若合同写「分析可落后 5 秒」，那刚下的订单 3 秒后就跑报表，走列侧没问题；要求「下单一秒后必须查到」，这类查询就得路由回行侧。</span>

## 边界

本课不列举 NewSQL 商标列表——下一课分类。也不把 HTAP 当向量数据库。湖仓后课是分析侧文件，新鲜度更松。

后课默认：混合负载要双路径或明确落后。NewSQL：SQL+分布式事务+分片的产品标签。

没有资源隔离的「一个库扛 HTAP」会把 2Q 与 latch 热点一起打穿。

<span class="marginnote">常见误区：初学者容易以为「合到一个库就万事大吉」。实际上没有资源隔离时，一条全表扫描的分析查询能占满缓冲池与 I/O，让前台收银事务一起排队——就像让盘点员和收银员共用一张桌子，盘点一展开，结账全堵死。QoS、WLM 这类配额机制是 HTAP 的必需品，不是可选项。</span>

## 小结

- HTAP 用双格式或双引擎对齐新鲜度与资源。
- 分析快照与 OLTP 写的水位要合同。
- NewSQL 下一课。
- 出处：HTAP 综述；HyPer/C-Store；CDC。
