---
title: 列存与 PAX
date: 2026-09-08
section: cs
---

# 列存与 PAX

<div class="epigraph">
<p>列存按属性切文件，扫描只读需要的列；PAX 在页内按列摆放，保持页为 I/O 单位，减少行组装的跨页随机。</p>
<footer>—— 据 Copeland and Khoshafian DSM；Ailamaki et al. PAX, VLDB 2001；C-Store / MonetDB</footer>
</div>

[上一课](/cs/leveled-tiered-compaction)决定 LSM 文件如何分层。本课不调倍率。缺口是页/文件**内部布局**：行存一行连续，OLAP 扫描浪费带宽在无用列上。列存（DSM）每列一堆文件，延迟物化自然。PAX（partition attributes across）折中：一页里所有列的一段行，页间仍是行组。

## 问题

分析查询常 `SUM(a) WHERE b>`，只需两列。行页把整行带进缓存。纯列存：谓词列与度量列分别顺序扫，组装用位置对齐。跨列更新、点查、宽行插入在纯列上变随机或写放大。缺口是 **OLAP 布局 vs OLTP 行页**，不是新 SQL。

PAX：一个 row group 放进页，页内各列分别数组。缓存行上扫一列仍顺序，点查一行在同一页内可取齐列。C-Store、Parquet 的 row group 是同一思想的文件级版本。

<span class="marginnote">术语翻译：NSM（行存）像记笔记本——一个人的所有信息连续写在一起；DSM（列存）像 Excel——同一属性竖着排一整列。前者适合「取出这一整行」，后者适合「对某一列算总数」，PAX 则是每页内部按 Excel 摆、页与页之间仍按笔记本分章。</span>

<span class="marginnote">Ailamaki, DeWitt, Hill, Skounakis，VLDB 2001 PAX。Copeland and Khoshafian 分解存储模型。Stonebraker et al. C-Store。本课布局；压缩下一课。</span>

## 方法

行页：槽+记录。列文件：每列块+位置对齐或每列自有空值位图。PAX 页：页头、各列偏移、列数据。执行器向量化从 PAX 或列块直接填列向量，少经行中间态。

<span class="marginnote">数字实例：`SELECT SUM(a) FROM t WHERE b > 10`，表有 100 列、100 万行、每格 8 字节。行页要把整行读进内存，约 100 万 × 800 字节 = 800 MB；列存只读 a、b 两列，约 100 万 × 16 字节 = 16 MB——差 50 倍，这就是分析查询偏爱列存的根由。</span>

HTAP 后课用两份或可更新列存；本课只钉只读分析布局的动机。

```mermaid
flowchart TD
  NSM["行页"] --> WIDE["扫行带上无用列"]
  DSM["列文件"] --> NARROW["只读需要的列"]
  PAX["页内列"] --> MIX["I/O 单位仍是页"]
```

## 机制

更新：行页原地或 MVCC 新行；列存常写新列块或 delta 再合并（LSM+列）。缓冲池：列块缓存与行页缓存权重不同，替换策略仍要抗扫描污染。

```mermaid
flowchart TD
  UP["更新一行"] --> R["行页路径"]
  UP --> C["列存路径"]
  R --> MVCC["原地改或写 MVCC 新行"]
  C --> DELTA["先写 delta 缓冲区"]
  DELTA --> MERGE["后台合并成新列块"]
```

<span class="marginnote">为什么重要：列存改一行，理论上该行的每一列块都要动——所以引擎先写进 delta 区「记账」，后台再批量合并。初学者容易以为列存只是「省读」；实际上它把写一次的成本摊成了「写 delta + 后台合并」两段，这就是 HTAP 系统要专门解决的事。</span>

连接：列存上延迟物化用位置；shuffle 只发键列。与 exchange 课对齐。

## 边界

本课不讲字典/RLE。也不把宽列 Bigtable 当分析列存——宽列是稀疏行键模型，后课。PAX 不是「有了就不需要 Parquet」；文件格式课会把 row group 标准化。

后课默认：分析扫描用列或 PAX；点查可用行或覆盖索引。列压缩：编码使列块更小、跳过更狠。

布局服务访问模式。同一 SQL 在不同布局上计划常数不同，要重校准。

## 小结

- 列存减分析 I/O；PAX 在页内列化以免点查跨页。
- 更新与点查是列存的代价面。
- 列压缩下一课：字典、RLE、位打包。
- 出处：Copeland and Khoshafian；Ailamaki et al. PAX；C-Store。
