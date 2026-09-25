---
title: Dynamo 与反熵
date: 2026-09-08
section: cs
---

# Dynamo 与反熵

<div class="epigraph">
<p>始终可写：斜向法定人数把字节先落到提示节点，再用 Merkle 反熵与读修复把分叉收回来。</p>
<footer>—— 据 DeCandia et al., Dynamo: Amazon's Highly Available Key-value Store, SOSP 2007 整理</footer>
</div>

上一课[Quorum](/cs/quorum-rw)的严格相交在分区下会拒绝写。Dynamo 选[PACELC](/cs/cap-pacelc)的 A 与 L。缺口是**提示移交与修复**：写没到「该在的」副本时，系统如何还不丢、如何终收敛。本课不重写 $R+W\gt n$ 的证明。后课 LWW 是收敛时的一种打结。

## 问题

一致性哈希环上每个键 $N$ 个偏好节点。协调者写 $W$ 个、读 $R$ 个。节点失败时 **sloppy quorum**：写进环上后面的活节点，并记 hinted handoff，对方回来再推。偏好集暂时不相交，线性一致放弃。缺口：愈合后如何发现不一致——不能靠全量扫描每次。

反熵：副本两两比较键的摘要。Merkle 树让子树哈希不同才下降，带宽随差异量而不是全量。读修复：读时发现版本分叉，把合并结果写回。二者一起逼近[最终一致](/cs/eventual-consistency)的收敛。

<span class="marginnote">SOSP 2007 论文把 N/R/W、向量时钟、sloppy、Merkle、gossip 成员放在同一系统里。本课抓反熵与提示，成员 gossip 点名。</span>

<span class="marginnote">sloppy quorum 直译「将就的法定人数」：本该落在 N 个偏好节点上的写，节点挂了就「将就」写到环上下一个活节点，先落住再说。类比快递：收件地址没人签收，先放邻居家并留张纸条（提示），住户回来再从邻居取走——这张纸条就是 hinted handoff。</span>

## 方法

写路径：协调者（可以是负载均衡选的任意节点）向偏好节点并行写，凑满 $W$（含提示）。读路径：凑 $R$，比版本。后台：反熵会话、提示重放、节点上环后的数据迁入。

```mermaid
flowchart TD
  PUT["PUT"] --> SQ["sloppy W"]
  SQ --> HINT["提示节点"]
  HINT --> HH["handoff 回家"]
  GET["GET"] --> RR["读修复"]
  AE["Merkle 反熵"] --> CONV["收敛"]
  RR --> CONV
```

成员变化用 gossip，不必每次写先跑共识配置——这与链式复制的配置主相反。

## 机制

向量时钟标并发写：不可比则保留多版本交给应用（购物车合并）。单时间戳 LWW 会静默丢一边——下一课专门钉这个坑。$R=1,W=1$ 延迟最低，分叉最多；$R=W=2,N=3$ 是常见折中，仍不是 ABD。

<span class="marginnote">N/R/W 代个数：N=3、W=2 表示写只要 3 个副本里 2 个确认，坏一个照样能写；R=2 同理。但 sloppy 模式下两次写可能落到不相交的节点组上，读两次也拼不齐全历史——严格 quorum 的「R+W 大于 N 必相交」保证被松掉了，反熵后台就是给这个缺口清账的。</span>

反熵不是立即：窗口内读可旧。运维看到的「不一致」是时间与故障的函数。Merkle 树深度与键分区粒度影响比较轮次；树本身要与哈希环分段对齐，否则比较的不是同一键集。

```mermaid
flowchart TD
  HA["副本 A 的 Merkle 树根哈希"] --> CMP{"根哈希相同？"}
  HB["副本 B 的 Merkle 树根哈希"] --> CMP
  CMP -->|相同| SAME["子树全同，跳过这段键空间"]
  CMP -->|不同| DOWN["下降一层，比较子树哈希"]
  DOWN --> LEAF["定位到少数不同的键，只同步这些键"]
```

<span class="marginnote">Merkle 树比较可以类比「对答案」：先对总分（根哈希），总分一样整卷免查；不一样再按大题对分，只把分数不同的几道小题翻出来细看。同步的数据量取决于差异多少而不是全量——这就是正文「带宽随差异量」的意思。</span>

本课不把 Cassandra/Riak 当新定理，它们是 Dynamo 论文的后裔。也不把对象存储的跨 AZ 异步复制写成 hinted handoff。

## 边界

本课不写 CRDT 半格，不展开一致性哈希虚节点（数据结构课已有）。后课默认：AP 键值 = sloppy + 版本 + 反熵；严格相交的寄存器不要冒充 Dynamo。冲突如何自动消掉是 LWW 与 CRDT 的分界。

始终可写把安全债务推到愈合。债务用版本与反熵还，不用 CAP 口号还。

## 小结

- Sloppy quorum 与提示移交换可用性，放弃读写相交。
- Merkle 反熵与读修复驱动收敛。
- 并发写靠版本向量检出，合并规则另课。
- 出处：DeCandia et al., SOSP 2007。
