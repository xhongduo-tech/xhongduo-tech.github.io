---
title: 流量工程
date: 2026-09-08
section: cs
---

# 流量工程

<div class="epigraph">
<p>最短 IGP 度量会把流量堆到最宽的树上；流量工程用另一套约束（带宽、时延、着色）把需求矩阵搬到链路上，使热点下降。</p>
<footer>—— 据 RFC 3272 综述；OSPF-TE / RSVP-TE 实践；Kurose and Ross TE 节整理</footer>
</div>

[上一课](/cs/internet-ixp) 画出 AS 图。域内同样会挤：ECMP 哈希仍可能打满一条。缺口是**流量工程**：与「最短路」分离的显式路由或度量调整。本课不把 MPLS 标签栈写完，下一课才是 MPLS。

## 问题

OSPF 代价若只按接口带宽设，所有需求仍可能共享同一条低代价边。TE：测量需求 → 算约束最短路或多商品流近似 → 用 RSVP-TE 建 LSP，或调 IGP 度量，或用后课 SR。目标常是最大链路利用率最小，不是单流跳数最小。与 HOL：那是微秒级结构；TE 是分钟到小时的矩阵。

<span class="marginnote">术语翻译：流量工程就是给路网装导航——不把路修宽，而是让车流按「哪条路还空着」分流；优化目标常是「最挤的那条路别太挤」，而不是让每辆车跳数最少。</span>

不要把 TE 写成「买更大的口」唯一解：口更大后最短树仍可能挤。

<span class="marginnote">RFC 3272。Fortz–Thorup 等讨论度量优化。本课不解整数多商品流。</span>

### 最短树会堆热点

度量、LSP 或 SR 把需求矩阵搬开。过频调度量会抖 IGP。域间 TE 是属性，不是 Dijkstra。

## 方法

对照：调度量（全局副作用）vs 显式 LSP（局部）。画：需求矩阵 → 约束 → 转发路径。IXP 出口选择是域间 TE：MED、AS-PATH prepend、更长前缀（谨慎，RPKI maxLength）。

<span class="marginnote">数字实例：两条 10G 链路用 ECMP 分流，但哈希只认五元组——一旦六成流量来自同一条大象流，哈希再均匀，那一条照样被打满。这就是「ECMP 不是 TE」的原因。</span>

```mermaid
flowchart TD
  DEM["需求矩阵"] --> CST["带宽/时延约束"]
  CST --> PATH["显式或调度量"]
  PATH --> UTIL["降热点利用率"]
```

## 机制

BGP 决策仍选出口；TE 决定出口之后域内怎么走。卫星链路带宽贵，TE 更要避开误码高的入口。切片 5G 是无线侧预留；这里是 IP/MPLS 预留，对象类似、层不同。

ECMP 是无状态分流，TE 可以有状态；后课会对比 SR 的无状态中间节点。

<span class="marginnote">常见误区：以为扩容就能消热点。带宽加大后最短路径树还是那棵树，需求仍往同一条边上堆——往往先把流量搬开再谈扩容，反而更省钱。</span>

```mermaid
flowchart TD
  FA["流 A：8 Gbps"] --> L1{"都选中同一条<br/>10 Gbps 最短路"}
  FB["流 B：7 Gbps<br/>也选同一条最短路"] --> L1
  L1 --> HOT["合计 15 Gbps<br/>利用率 150%，排队丢包"]
  HOT --> TE["TE 按约束重算<br/>把 B 引上空余链路"]
  TE --> OK["A 原路利用率 80%<br/>B 绕行链路利用率 70%<br/>两条都不炸"]
```

## 边界

本课不引入 PCE 的全部 PCEP。MPLS 是下一课。后课默认：TE 把约束放进选路，IGP 最短只是其中一个解。

过频调度量会引起 IGP 与 BGP 震荡，像人为抖动。

下一课[MPLS](/cs/mpls)。

## 小结

- 最短 IGP 树不等于均衡负载。
- TE 用度量、LSP 或后课 SR 搬流量。
- 域间 TE 是政策属性，不是 Dijkstra。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 3272；OSPF-TE 实践。
