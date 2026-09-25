---
title: 目录协议的扩展性
date: 2026-09-08
section: cs
---

# 目录协议的扩展性

<div class="epigraph">
<p>窥探要求每个缺失让所有 cache 听见；目录记下谁可能有副本，只向那几个节点发点对点请求，流量随共享集而不是随核数涨。</p>
<footer>—— 据 Censier and Feautrier；Hennessy and Patterson, CA:AQA 整理</footer>
</div>

[上一课](/cs/moesi-mesif) 在广播总线上加了 O/F。核数到几十上百时，即使 F 减少重复数据，**地址事务仍要人人听**。本课不重画五态。缺口是**目录：每行（或每组）一份 sharer 向量 / 指针，把一致性从广播改成端到端。**

## 问题

窥探带宽 ∝ 核数 × miss 率。互连变成广播介质后，[MLP](/cs/mlp-memory-parallelism) 越高越堵。缺口不是再加一个 Forward 态，而是**用存储换广播：目录项指出可能的副本集合。** 目录本身要分区到 LLC slice 或 memory controller，否则又成为单点。

<span class="marginnote">完整位向量：每核一位，精确但随核数线性涨。稀疏目录：少量指针，溢出则广播或退化为粗向量。这是面积与无效流量的交易。</span>

<span class="marginnote">数字实例：64 核机器上某行只被 3 个核共享。窥探作废要广播给全部 64 个核（63 份纯属陪听）；目录查 sharer 向量「…00101」只点对点发 2 份 invalidate、收 2 个 ack。请求量从 63 降到 2，这就是「流量随共享集而不是随核数涨」的具体含义。</span>

## 方法

读 miss：向 home 目录要数据与权限；目录把请求者加入 sharer，若存在 owner 则转发。写 miss：目录向所有 sharer 发 invalidate，收集 ack，再给请求者 M。替换：通知目录删 sharer，脏则写回。

```mermaid
flowchart TD
  REQ["核 miss"] --> HOME["该行的目录 home"]
  HOME --> FWD["转发到 owner / 记忆体"]
  HOME --> INV["点对点作废 sharer"]
  INV --> ACK["收集 ack 再授权"]
```

## 机制

延迟从「一跳广播」变成「到 home 再可能到 owner」，跳数增加，但带宽可扩展。与 [MOESI](/cs/moesi-mesif) 组合：目录里存 owner 指针而不是只存 S 集合。伪共享表现为同一目录项上的频繁 inv，协议正确、性能差。

目录存储与 LLC 标签可共存（in-cache directory）：行在 LLC 则目录免费，被踢出则要写回或记 overflow。

```mermaid
flowchart TD
  subgraph SN["窥探：广播"]
    M1["任一核 miss"] --> BUS["总线广播"]
    BUS --> C1["核 0 被迫听"]
    BUS --> C2["核 1 被迫听"]
    BUS --> C3["……其余 61 核也被迫听"]
  end
  subgraph DIRN["目录：点对点"]
    M2["核 5 miss"] --> HOME2["home 查 sharer 位向量"]
    HOME2 -->|"只发共享核"| P1["核 2 invalidate"]
    HOME2 --> P2["核 7 invalidate"]
  end
```

这张图回答「目录到底省掉了什么」：同样的写 miss，窥探让 63 个不相干核都花监听带宽，目录只打扰向量里为 1 的那几位。代价是多一跳（先到 home 再到 owner/sharer）和每行一份向量存储——用延迟和面积换回带宽的可扩展性。

<span class="marginnote">术语翻译：sharer 向量就是这行缓存的「名单」——一串二进制位，第 $i$ 位是 1 表示「核 $i$ 手里可能有副本」，作废时照名单点名，不搞全班广播。</span>

<span class="marginnote">常见误区：初学者容易把目录想成「集群门口的一张总表」。实际上目录按地址切片分散到各个 LLC slice / 内存控制器上（每条地址固定归属一个 slice），否则这张表自己就成了比总线更糟的单点——这正是「home 必须分区」的意思。</span>

## 边界

本课不把某种 NoC 拓扑定成唯一 home 映射，后课 mesh 会决定 slice 怎么放。也不处理死锁的虚通道，那是 NoC 路由课。原子与 acquire/release 建立在「行已按目录拿到合适态」之上，随后几课。

后课默认：大规模用目录；共享集溢出才广播。同一行上无关变量互踢，是下一课伪共享。

## 小结

- 目录用 sharer 集合代替广播监听，换存储与多跳延迟。
- home 必须分区，否则单点。
- 行粒度上的虚假争用是下一课伪共享。
- 出处：Censier and Feautrier；Hennessy and Patterson, *CA:AQA*。
