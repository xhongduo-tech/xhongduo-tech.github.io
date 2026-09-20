---
title: HNSW / IVF
date: 2026-09-08
section: llm
---

# HNSW / IVF

<div class="epigraph">
<p>精确近邻随库规模线性扫。近似近邻用图或倒排粗量化换召回–延迟曲线。曲线是协议，不是「开了 HNSW 就无损」。</p>
<footer>—— Malkov &amp; Yashunin HNSW；Jégou 等 IVF 粗量化倒排</footer>
</div>

[上一课](/llm/rag-context-compression)减少进入阅读器的 token。检索库本身仍可能是千万级向量。本课写 ANN 的两条默认结构：HNSW（图）与 IVF（倒排文件）。缺口是召回率对 $k$、efSearch、nprobe 的依赖。后课 PQ 在向量内部再压缩，常与 IVF 联用。

## 问题

精确 kNN 扫全库，延迟不可接受。HNSW 建多层可导航小世界图，搜索沿边贪心 + 候选集。IVF 先把空间分成粗心，查询只扫近的若干列表。二者都是近似：可能丢掉真正的邻居。RAG 里丢掉的若是唯一支撑块，忠实度直接掉。必须在**自己的查询分布**上画 Recall–延迟曲线，不能抄基准数字。

过滤（权限、租户）与图遍历交互：先过滤后搜或边搜边滤，实现错了会漏召回或漏出无权限文档。

<span class="marginnote">「粗心」（centroid，粗量化中心）可以想象成把一座城市划成几个大区：查询先判断自己落在哪个区，只翻这个区的电话簿。nprobe 就是「除本区外再看几个邻区」——多看几个不容易漏人，但每多看一个就多扫一份列表，延迟线性涨。</span>

<span class="marginnote">建索引的超参（M、efConstruction、粗心个数）改的是图/列表质量，搜时的 efSearch / nprobe 改的是预算。两套都要进模型卡式的索引卡。</span>

## 方法

选结构：内存允许、要高召回 → HNSW；库极大、可接受较低召回或配合 PQ → IVF。调参：固定延迟预算扫 efSearch 或 nprobe，看 Recall@$k$。$k$ 应大于最终送给阅读器的块数，给融合与压缩留候选。增量插入：HNSW 可动态加边，IVF 列表 unbalanced 时要重建。

```mermaid
flowchart TD
  QV["查询向量"] --> HNSW["HNSW 图遍历"]
  QV --> IVF["IVF 选列表再扫"]
  HNSW --> CAND["近似 Top-n"]
  IVF --> CAND
```

## 机制

HNSW 的高层边是长距离高速公路，底层是精细邻居。efSearch 增大等于允许更宽的候选，逼近精确。IVF 的误差来自查询落在错误粗心附近——边界点会被切错列表，nprobe 增大即多探几个心。两者都把「真实近邻被剪掉」的概率换成延迟。这个概率在难查询（近邻很近、易混）上更高，平均召回会掩盖切片。

```mermaid
flowchart TD
  S["从顶层入口出发"] --> G1["顶层：稀疏长边，快速跳远"]
  G1 --> G2["逐层下降：图变密，步距变短"]
  G2 --> G3["底层：精细邻居，贪心逼近目标"]
  G3 --> E["候选集宽度由 efSearch 控制"]
  E --> R["返回近似 Top-k"]
```

<span class="marginnote">数字实例：一千万条向量的库，efSearch=64 时每次查询只实际碰几千条，比全库扫描少约三个数量级；代价是 Recall@10 可能只有 0.95 而不是 1.0。把 efSearch 拉大一倍召回往往只涨一两个点，延迟却近乎翻倍——你的 SLA 定在哪一格，要在自己的查询分布上量出来。</span>

## 边界

ANN 不修复坏嵌入。空间本身塌了，图再好也搜不到。下一课 PQ：把向量量化进码本，内存再降一截，误差再加一层。

<span class="marginnote">初学者容易以为权限过滤就是「先搜出 Top-10，再删掉没权限的」。若某租户被过滤掉 8 条，最终只剩 2 条可用，回答质量已经受损；正确做法是把过滤条件带进遍历（边搜边滤）或扩大候选 k，保证过滤后的数量仍然够用。</span>

## 小结

- HNSW / IVF 用近似换延迟，必须报自己分布上的 Recall–延迟。
- 搜时预算与建时超参都要版本化。
- 权限过滤与遍历实现属于正确性，不只是性能。
- 出处：Malkov & Yashunin HNSW；Jégou 等 IVF。
