---
title: 谓词下推
date: 2026-09-08
section: cs
---

# 谓词下推

<div class="epigraph">
<p>能在扫描或第一次连接前就丢掉的行，不要拖到树顶再过滤；下推是改写里最常用的一条。</p>
<footer>—— 据 Ullman 对选择下推；Chaudhuri；Ramakrishnan and Gehrke</footer>
</div>

[上一课](/cs/query-rewrite)说先用代数律把树变瘦。本课不重列结合律。缺口是那一课最重要的实例：把 `WHERE`/`HAVING` 能下到的谓词推向叶子或更低的连接，使中间基数按[直方图](/cs/histogram-card)变小。外连接与相关子查询会挡住部分下推。

## 问题

查询改写留下「下推与结合」。若优化器只交换连接顺序、不把选择贴到扫描，索引与堆扫描仍读全集。谓词下推：选择穿过投影（保留需要的列）、穿过内连接。缺口是**这一条律的机制与挡板**，不是代价搜索本身。

<span class="marginnote">下推到存储引擎（仅读需要的列与行）是同一思想的物理版。易失函数、随机() 不能无约束下推。</span>

<span class="marginnote">术语翻译：谓词下推就是把 WHERE 里的过滤条件「搬」到尽量靠近数据存取的位置——先过滤、再连接，而不是先连接出一个巨型中间表再过滤。</span>

## 方法

从树顶收集合取项，按引用的属性集合贴到最深仍合法的节点。不能穿过改变 NULL 填充的外连接侧。物化视图匹配是另一类改写，后课视图。本课不把全部 IDB 规则写成 Datalog 教材。

```mermaid
flowchart TD
  TOP["顶层过滤"] --> PUSH["推向扫描或低连接"]
  PUSH --> SMALL["中间结果变小"]
  BLOCK["外连接 / 易失函数"] --> STOP["停在挡板之上"]
```

## 机制

下推直接兑现[选择投影连接的顺序](/cs/algebra-reorder)里画过的那棵瘦树。物理上，叶子扫描可以变成索引条件。语义在内连接与无 NULL 坑时与原 SQL 一致；挡板存在是为了不把 unknown 变成错行。

以 `users JOIN orders` 且 WHERE 带两个条件为例，看一个合取项如何按引用的表被拆开、各自贴回扫描：

```mermaid
flowchart TD
  W["WHERE u.country='CN' AND o.amount>100"] --> SPLIT["按谓词引用的表拆开"]
  SPLIT --> P1["country='CN' 贴到 users 扫描"]
  SPLIT --> P2["amount>100 贴到 orders 扫描"]
  P1 --> F1["users：100 万行 过滤到 2 万行"]
  P2 --> F2["orders：1000 万行 过滤到 50 万行"]
  F1 --> J["连接只处理过滤后的小结果"]
  F2 --> J
```

<span class="marginnote">数字实例：假设 orders 有 1000 万行、amount&gt;100 能筛掉 95%，下推后连接环节只需面对 50 万行；不下推的话这 1000 万行要全部先参与连接，内存与时间都被中间结果吃掉。</span>

## 边界

本课不把谓词下推到另一台分片节点的网络协议写完（后课分片）。模式太宽导致的更新异常，下一课范式。

后课默认：过滤尽量靠近数据。表太宽是设计问题，不是再下推一次能修好。

<span class="marginnote">常见误区：初学者容易以为任何 WHERE 条件都能无脑下推。遇到外连接的保留侧，或 random()、now() 这类每行求值结果不同的易失函数，提前过滤会改变查询语义，优化器必须把它们停在挡板之上。</span>

## 小结

- 选择尽量下推以减小中间结果。
- 外连接与易失性是挡板。
- 范式与分解下一课。
- 出处：Ullman；Chaudhuri；Ramakrishnan and Gehrke。
