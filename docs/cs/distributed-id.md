---
title: 分布式 ID
date: 2026-09-08
section: cs
---

# 分布式 ID

<div class="epigraph">
<p>唯一标识不必来自中心自增。雪花一类：时间 + 机器 + 序列。有序、可追溯、无单点；代价是时钟与机器编号的规格。</p>
<footer>—— 据 Twitter Snowflake 公开设计；UUID 标准；Lamport 时间戳对照整理</footer>
</div>

上一课[Tail at Scale](/cs/tail-at-scale)警告中心热点。缺口是**生成主键**：数据库 `AUTO_INCREMENT` 是单点尾延迟与故障点。本课钉 ID 方案与时钟依赖。后课地理复制会让「时间序」更假。

## 问题

要求组合：全局唯一、大致时间可排序、高 QPS、故障隔离。UUID v4：唯一性好，无序，索引随机写碎。UUID v1：含时间与 MAC，隐私与时钟。Snowflake：毫秒时间戳、worker id、序列。worker id 来自配置或 ZK/etcd 租约——又把小 CP 请回来，但只在启动。

缺口：时钟回拨使时间戳段重用，破唯一。要等或用[HLC](/cs/hlc)或拒绝。多机房 worker id 空间要划片，否则碰撞。

<span class="marginnote">Snowflake 是工程博客级源头，不是 TODS。数据库号段（一次取一千）是折中：中心少热点，ID 仍大致单调。</span>

## 方法

选：无序唯一用随机；要局部单调用雪花或号段。号段：每个应用实例从 RSM 拿一段，本地发完再拿——段边界可有洞（实例崩）。洞通常可接受，连续当规格则必须中心。

```mermaid
flowchart TD
  CLK["时间"] --> SF["雪花 ID"]
  WID["worker 编号"] --> SF
  SEQ["序列"] --> SF
  RSM["号段"] --> LOCAL["本地发放"]
```

不要用墙钟毫秒当唯一：同毫秒多机碰撞。

## 机制

与线性一致计数器：中心 `INCR` 唯一且单调，QPS 受单日志限制，尾差。雪花把唯一性拆到时间维，单调只在同一 worker 且时钟不回拨。跨 worker 比较大小 ≈ 比较时间，受 $\varepsilon$ 限制——不能当因果。

本课不写区块链 nonce 挖矿。也不写证件号合规。

热行：以 ID 为分区键且 ID 时间单调，则写总打最新分区（Mongo 等经典坑）。要哈希，或在高位加 shard 随机。这是热键在主键上的投影。

## 边界

本课不规定 64 位布局。后课默认：规模 ID 不用单点自增；雪花要声明回拨策略与 worker 分配。地理复制下一课让「谁的毫秒」更乱，跨区 ID 更要划 worker 空间。滚动升级时 worker 重编号不得碰撞。

唯一是安全；有序是性能与产品。两件事分开买。

## 小结

- 中心自增唯一但成尾与单点。
- 雪花：时间+机器+序列；回拨与 worker 分配是规格。
- 单调 ID 当分区键会写热分区。
- 出处：Snowflake 设计；UUID；对照 Lamport 钟。
