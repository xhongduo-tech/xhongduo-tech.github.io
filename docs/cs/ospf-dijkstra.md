---
title: Dijkstra 在 OSPF
date: 2026-09-08
section: cs
---

# Dijkstra 在 OSPF

<div class="epigraph">
<p>链路状态库是同一张带权图；每台路由器以自己为源跑 Dijkstra，下一跳才写进 FIB。</p>
<footer>—— 据 Dijkstra 最短路；Moy, RFC 2328 OSPF Version 2；Kurose and Ross</footer>
</div>

[上一课](/cs/link-state)给出洪泛邻接与「各节点独立计算」。本课不重写 LSA 洪泛可靠性。缺口是计算本身：图课的 [Dijkstra](/cs/dijkstra) 已经会单源最短路；这里要接到 OSPF 的代价、区域直觉和「下一跳是邻居而不是整条路径」。本课不把 BGP 政策提前。

## 问题

链路状态通告描述「我连谁、代价多少」。若只停在「有一张图」，还不会转发。OSPF：从 LSDB 建图，节点是路由器与网络，边权是接口代价。以本机为源跑 Dijkstra，得到到每个前缀的出口邻居。区域把洪泛范围切开，骨干连通各区域，避免全网一张图过大。缺口是**把最短路树写成转发表**。

<span class="marginnote">代价常与带宽相关，由管理设定，不是时延测量。ECMP 在多条等代价时分流，仍要求下一跳是邻居。</span>

## 方法

同步 LSDB（可靠洪泛、序号）。算 SPF 树。把叶子网络的前缀安装为「下一跳 = 树上来的第一跳邻居」，再交给 [LPM](/cs/lpm)。区域内与区域间前缀来源不同，但查找规则仍是最长前缀。本课不把全部 LSA 类型表当必背。

```mermaid
flowchart TD
  LSDB["链路状态库"] --> G["带权图"]
  G --> SPF["本机为源的 Dijkstra"]
  SPF --> FIB["下一跳写入 FIB"]
```

## 机制

每台路由器计算自己的树，但仍应得到相容的转发：同一张无向代价图、同一算法。与距离向量对照：坏消息随拓扑洪泛，不必数到无穷。域内 IGP 仍是一个管理域的最短路；跨域政策不是代价最小，下一课 BGP。

## 边界

本课不引入 IS-IS TLV 细节当第二条主干，不把流量工程的约束最短路写成 OSPF 正文。多区域虚拟链路是例外，不展开。自治系统之间如何交换路径，下一课 BGP 直觉。

后课默认：一个 AS 内部可以用 Dijkstra 填下一跳。互联网的边界路由换政策语言。

## 小结

- OSPF 在同步图上跑 Dijkstra，安装邻居下一跳。
- 区域缩小洪泛，不改变 LPM。
- 跨 AS 不是最短路，下一课 BGP。
- 出处：RFC 2328；Dijkstra；Kurose and Ross。
