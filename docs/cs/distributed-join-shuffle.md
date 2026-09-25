---
title: shuffle 与 broadcast join
date: 2026-09-08
section: cs
---

# shuffle 与 broadcast join

<div class="epigraph">
<p>等值连接在分片上要让同键碰面：shuffle 按键重分区，broadcast 把小侧复制到每一片。网络字节是代价模型的新项。</p>
<footer>—— 据 Graefe exchange；DeWitt 并行连接；分布式 Grace 同构</footer>
</div>

[上一课](/cs/shard-key-rebalance)把行放在片上。本课不选用户 id。缺口是连接：若两表已按连接键同切（collocated），工人本地连，无网络。否则要 [exchange](/cs/parallel-exchange) 的网络版：shuffle join 或 broadcast join。单机 Grace 的分区文件换成网络块。

## 问题

大对大：两边按连接键 hash 送到 $N$ 个工人，正确性同 Grace。小对大：广播小表，大表本地扫，避免动大表。缺口是估计：小表估错会变成广播一张大表打爆内存——基数误差的分布式形态。倾斜：单键打到一工人，同单机倾斜。

半连接归约：先传键再拉匹配行，Bernstein-Chiu 思想，减 payload。延迟物化：只 shuffle 键+位置，载荷留在存储节点——存算分离后课。

<span class="marginnote">并行数据库教科书连接。Spark shuffle 是同一算子在批处理引擎。本课数据库执行器。</span>

<span class="marginnote">数字实例看 broadcast 的代价模型：小表 10 MB、200 个工人，广播要发 $10\text{ MB}\times200=2$ GB 进网络，换回大表一次不动；若优化器把小表估成 10 MB 实际是 50 GB，广播就成了 10 TB 的灾难——「估错小表」在这里被网络放大 200 倍，这正是基数估计直接决定计划生死的原因。</span>

## 方法

优化器比较：colocated $\lt$ broadcast $\lt$ shuffle（通常）。强制提示可钉。bloom 过滤器随广播或 shuffle 前过滤。外连接：broadcast 方向受限（只能广播被保留侧的对面等），与外连接下推防火墙同类。

```mermaid
flowchart TD
  COL["同键同切"] --> LOC["本地连接"]
  SMALL["小表"] --> BC["广播到各大表片"]
  BIG["两表都大"] --> SH["按键 shuffle"]
```

## 机制

RTO 无关；这是查询时流量。计划回归：统计变化让昨日 colocated 今日 shuffle。网络校准必须进代价模型，否则 DP 以为哈希内存连接免费。bloom 随广播或 shuffle 前过滤，减 payload，思想同 LSM 布隆但走的是连接键。

事务：分布式连接一般读快照，写仍按分片事务。连接本身不提交两片写。外连接限制 broadcast 方向。倾斜使某一工人收到大部分键，并行名存实亡——要拆热键或加盐，与单机 Grace 相同病。

```mermaid
flowchart TD
  subgraph RAW["原始按键 shuffle：键 user_1 占 90% 行"]
    K1["user_1 的 9000 万行"] --> W0["工人 0：被压垮"]
    K2["其余键合计 1000 万行"] --> W1["工人 1–199：闲着"]
  end
  subgraph SALT["加盐后：热键拆 8 份"]
    S1["user_1#0 … user_1#7 各约 1125 万行"] --> W2["工人 0–7：均匀"]
    S3["其余键照旧"] --> W3["工人 8–199"]
    S2["join 键拼上随机后缀，两侧同规则"] -.-> S1
  end
```

这张图回答「倾斜怎么让并行名存实亡、加盐怎么救」：hash 只认键，不认键的流行度——一个顶流 user 的行全部涌向同一个工人，199 台机器围观一台干活。加盐把热键拼上随机后缀拆成多份分摊，代价是连接条件要两侧都按同一规则改写，且聚合要再来一轮合并。

<span class="marginnote">术语翻译：shuffle（洗牌）就是连接前的「重新分堆」——每行按连接键的哈希值决定去几号工人，保证同键的行必然落进同一堆，两边才能就地连接。它回答的是「让谁和谁碰面」，网络字节数就是这次碰面的路费。</span>

<span class="marginnote">常见误区：初学者容易以为「工人加一倍，连接就快一倍」。shuffle 阶段的网络流量近似随工人数平方级增长（$N$ 个工人各向 $N$ 个工人发），而倾斜键又卡死单点——并行度上限由最热的那个键决定。先看键分布再谈加机器，否则只是花钱把瓶颈挪个地方。</span>

## 边界

本课不讲 Percolator 锁。也不把 MapReduce shuffle 当 SQL 连接的定义——后课 MR 是另一套。图遍历的多跳 shuffle 更重，图库课。

后课默认：跨片等值连接靠 colocated / broadcast / shuffle 三选。Percolator：在 Bigtable 上用锁表做跨行事务。

网络是第二磁盘。估错大小等于估错 I/O。

## 小结

- 同切本地连；否则广播小表或按键 shuffle。
- 倾斜与估错会打爆单工人。
- Percolator 下一课：宽列上的分布式事务。
- 出处：Graefe；DeWitt；Bernstein-Chiu 半连接。
