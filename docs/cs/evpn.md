---
title: EVPN
date: 2026-09-08
section: cs
---

# EVPN

<div class="epigraph">
<p>用 BGP 分发 MAC/IP 与 VNI，替代覆盖网上的洪泛学习；同一套控制面还能做 IRB 与多归。</p>
<footer>—— 据 RFC 7432 BGP MPLS-Based Ethernet VPN；RFC 8365 EVPN overlay 整理</footer>
</div>

[上一课](/cs/vxlan-overlay) 数据面仍可能洪泛。[iBGP 与 RR](/cs/ibgp-route-reflector) 已能传前缀。缺口是 **EVPN 路由类型**：Type 2 MAC/IP、Type 3 组播隧道、Type 1 多归以太网段。本课不把 GRE 头写完。

## 问题

VXLAN 无控制面像透明桥：未知就复制。EVPN：VTEP 把学到的 MAC 打成 BGP NLRI，RR 反射，对端装表——二层学习从数据面迁到控制面，像「MAC 的 BGP」。多归：主机双连两台叶子，以太网段 ESI 防环（DF 选举），对照 RSTP 但在覆盖上。IRB：同一 EVPN 里做 L2 与 L3 网关，减少绕核心。

不要把 EVPN 写成替代全球 BGP 互联网：这是数据中心/VPN 地址族。

<span class="marginnote">RFC 7432 先对 MPLS，RFC 8365 对 VXLAN。Type 5 前缀路由做 L3VPN 风格。本课不把每类 NLRI 字段背完。</span>

### MAC 走 BGP

控制面分发替代洪泛学习。ESI/DF 处理多归。这是 VPN/DC 地址族，不是互联网 eBGP 替代。

## 方法

画：本地学习 → Type 2 通告 → 远端表。对照 MAC 老化：BGP 撤回替代未知洪泛。与 RPKI 无关（不同 AF）。

```mermaid
flowchart TD
  LEARN["本地 MAC"] --> T2["EVPN Type2"]
  T2 --> RR["BGP RR"]
  RR --> REM["远端 VTEP 表"]
  ESI["ESI 多归"] --> DF["指定转发者"]
```

方法止于选定对象与对照；机制才说它如何嵌入已有分层与主干课。

## 机制

收敛：MAC 移动要序列号，防双活脑裂。这比 STP TCN 更像 BGP 前缀移动。底层仍 ECMP；EVPN 别把同一流钉死到死掉的 VTEP。IXP 一般不用 EVPN 给公网对等。

BUM：Type 3 建 ingress-replication 或组播组，收口上一课的洪泛。

## 边界

本课不引入 EVPN E-Tree 的全部。GRE 与隧道是下一课。后课默认：覆盖的 MAC 用 BGP 分发；数据面可以是 VXLAN 或 MPLS。

控制面规模仍受 RR 与表项容量约束，不是无限租户。

上一课留下的缺口在本课收口；「EVPN」进入后课词汇表后只引用。文献用来钉对象与边界，不把本课写成该主题的独立综述。下一课[GRE 与隧道](/cs/gre-tunnels)。

## 小结

- EVPN 用 BGP 传 MAC/IP，收敛洪泛。
- ESI/DF 处理多归，替代大 STP 域。
- 与 VXLAN 数据面搭配最常见。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 7432；RFC 8365。
