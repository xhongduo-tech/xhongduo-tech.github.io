---
title: DiffServ
date: 2026-09-08
section: cs
---

# DiffServ

<div class="epigraph">
<p>DSCP 把包分到少量行为聚集，每跳 PHB（EF/AF/BE）执行；可扩展，但不在互联网上承诺端到端预留。</p>
<footer>—— 据 RFC 2474 / 2475 DiffServ；RFC 3246 EF；RFC 2597 AF 整理</footer>
</div>

[WFQ](/cs/wfq-fair-queue) 每流不扩展。[令牌桶](/cs/token-bucket-shaping) 可按类。[上一课](/cs/token-bucket-shaping) 留下着色。缺口是 **DiffServ 架构**：标记、PHB、边界信任。本课不把 TSO 写完。

## 问题

IntServ/RSVP 每流预留在核心炸状态。DiffServ：边缘分类打 DSCP，核心只看 6 比特做 EF（低延迟）、AF（分档丢弃）、BE。无端到端硬合同，除非域内 SLA。与 5G 切片对照：切片是硬隔离实例，DSCP 是软着色。PFC 用 802.1p，要在边缘映射到 DSCP。

<span class="marginnote">术语翻译：PHB（每跳行为）就是「每一台路由器对这类包承诺怎么对待」——先发给谁、拥塞时先丢谁。它只约束单台路由器的本地行为，像酒店每层各自的服务标准，拼起来并不自动等于一条端到端保证。</span>

不要把 EF 写成 URLLC 保证：没有资源预留则 EF 也会挤。

<span class="marginnote">RFC 2475 框架。核心「简」是可扩展理由。本课不把 64 个码点背完。</span>

### 域内着色

少量 PHB 可扩展，无全球预留。IXP 常重标记。EF 无资源也会挤。主机乱打 EF 会被边缘改写。

<span class="marginnote">直觉类比：EF 是救护车道——延迟最低，但要交警（资源规划）配合，车一多照样堵；AF 是分舱位的民航——票分三档，超售时先请低舱位改签；BE 是候机大厅——人人可坐，拥塞时排队最没怨言权。</span>

## 方法

画：边缘分类+桶 → DSCP → 核心 PHB。对照 EVPN/QoS：租户类在覆盖内外要映射。IXP 通常不信你的 DSCP，会重置。

```mermaid
flowchart TD
  EDGE["边缘分类整形"] --> DS["DSCP"]
  DS --> PHB["每跳 EF/AF/BE"]
  UNTRUST["不信任域"] --> RE["重标记"]
```

## 机制

AQM 可按 AF 颜色丢。ECN 与 DSCP 正交（ECN 用 TOS 低 2 位历史纠缠，实现要小心）。BGP 不传 DSCP 政策。RoCE 依赖边缘信任的优先级，跨域失效。

上面那张图画「包从边缘到核心的架构流向」；这张回答更细的问题：同一个 AF 类里，绿黄红三档在拥塞时究竟谁先被丢。

```mermaid
flowchart TD
  IN["AF 类的包进入队列"] --> CLR["边缘按承诺速率标色 绿黄红"]
  CLR --> Q{"现在拥塞吗"}
  Q -- "没拥塞" --> FWD["全部正常转发"]
  Q -- "拥塞" --> DROP["AQM 按丢弃优先级选丢"]
  DROP --> R1["先丢 红 超速最多的"]
  DROP --> R2["其次丢 黄"]
  DROP --> R3["最后才丢 绿"]
```

<span class="marginnote">数字实例：DSCP 是 IP 头旧 TOS 字段的前 6 比特，共 64 个码点。常用如 EF=46（二进制 101110）；AF 分四类、每类三个丢弃档；默认 0 就是普通尽力而为。所谓「打标」只是改 6 个比特，核心路由器查表执行，不必记任何一条流。</span>

与 Saltzer：DiffServ 是性能，不是正确性。

## 边界

本课不引入 RSVP-TE 与 DiffServ-TE 的全部。TSO/LRO 是下一课。后课默认：公网尽力而为；DSCP 在单一信任域内有意义。

主机随意打 EF 会被边缘改写。

下一课[TSO / LRO](/cs/tso-lro)。

## 小结

- 少量 PHB 换可扩展 QoS。
- 合同在域边界，不在全球。
- 与切片、PFC 映射但不等价。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 2474/2475；RFC 3246。
