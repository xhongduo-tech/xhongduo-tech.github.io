---
title: Parquet / ORC
date: 2026-09-08
section: cs
---

# Parquet / ORC

<div class="epigraph">
<p>开放列存文件把 row group、列块、编码与页统计写成可交换合同，让引擎不必共享缓冲池也能共享数据。</p>
<footer>—— 据 Apache Parquet 格式；Apache ORC；Dremel / C-Store 谱系</footer>
</div>

[上一课](/cs/column-compression)给出块内编码。本课不选 RLE 阈值。缺口是**文件级标准**：数据湖与多引擎（Spark、专用 OLAP、入库工具）要读同一份字节。Parquet 与 ORC 把 PAX 思想放到文件：footer、row group、column chunk、page、min/max 与字典。

## 问题

专有表空间把数据锁在一种引擎。分析把表卸成文件后，格式必须自描述：哪些列、编码、压缩、统计、嵌套（Parquet 的 repetition/definition 来自 Dremel）。缺口不是「大数据产品课」，而是存储进阶里的开放页格式，相对 InnoDB 页只在一个缓冲池里。

谓词下推进文件：读 footer 的 zone 统计，跳过 row group；再读列块统计，跳过 page。这是后课 zone map 的文件实例。嵌套结构不是文档库：仍是列存的嵌套编码。

<span class="marginnote">「自描述」就是文件自带说明书：有哪些列、什么类型、怎么编码、每块的 min/max 统计，全写在文件尾部的 footer 里——不需要连上某个数据库查元数据表，任何引擎拿到字节就能读懂。这正是数据湖里 Spark 和 Presto 能共读一份文件的原因。</span>

<span class="marginnote">Parquet 源自 Google Dremel 的列式嵌套；ORC 来自 Hive。本课对照合同，不背所有 version 字段。</span>

## 方法

写：按 row group 切（行数或字节阈值），每列一块，选编码与可选压缩器（与编码叠）。读：投影列集合 → 只打开那些 chunk；谓词 → 用统计裁剪。schema 演化：加列、可选字段，与模式迁移课的文件版。

<span class="marginnote">常见误区：把 Parquet 当成一种「压缩格式」。它其实是布局与编码的合同——先按列重排、再套 RLE/字典等编码，压缩器（snappy/zstd）只是叠在编码之上的可选项。列存之所以压得狠，根源在布局让同列的相似值挨在一起，不在压缩器本身。</span>

与事务：文件通常不可变，更新靠写新文件+元数据提交（湖仓表格式后课）。本课单个文件内部。

```mermaid
flowchart TD
  F["文件 footer"] --> RG["row group"]
  RG --> CC["column chunk"]
  CC --> PG["page + 统计"]
  PRED["谓词"] --> SKIP["跳过 group / page"]
```

## 机制

向量化扫描直接吃 page 解成列向量。并行：按 row group 切给工人，exchange 前已是列批次。小文件问题：每个文件一个 footer，调度税高——后课表格式用 compaction 合文件，与 LSM 同构。

<span class="marginnote">数字实例：把 1 TB 写成 10 万个 10 MB 小文件，调度器就要打开 10 万个 footer、排 10 万个任务，名字节点/对象存储的元数据请求先把队列占满——算没开始，调度税已经吃掉大半。合成 128 MB–1 GB 的大文件正是 compaction 干的事。</span>

```mermaid
flowchart TD
  Q["查询：只取两列 + 过滤条件"] --> M["读各文件 footer 与统计"]
  M --> PR["裁剪掉不满足统计的 row group"]
  PR --> SCH["剩余 row group 按块切给工人"]
  SCH --> W1["工人 A：解两列 page 为列向量"]
  SCH --> W2["工人 B：解两列 page 为列向量"]
  W1 --> EX["列批次交给上层 exchange"]
  W2 --> EX
```

类型系统：十进制、时间戳精度不一致会在引擎间产生静默错，合同要钉。

<span class="marginnote">「静默错」是最危险的一类：不报错、不崩溃，只是算出来的数悄悄不对。例如一个引擎把时间戳截到秒、另一个保留毫秒，JOIN 两边的键就对不上——报表数值差了一截，却查不出任何异常。</span>

## 边界

本课不把 Iceberg/Delta 的快照日志写完。也不讲位图索引在库内的实现——后课。ORC 的索引流、Parquet 的 bloom 模块点名存在。

后课默认：分析交换用 Parquet/ORC 一类列文件。zone map 与跳过索引：min/max 与更一般的跳过结构，库内表也用。

开放格式把「页」从一种引擎的缓冲池里解放出来。

## 小结

- Parquet/ORC 标准化 row group、列块、编码与统计。
- 谓词用统计跳过；文件不可变，事务在元数据层。
- zone map 下一课：把跳过从文件合同收回库内表。
- 出处：Apache Parquet；Apache ORC；Dremel。
