---
title: MPLS
date: 2026-09-08
section: cs
---

# MPLS

<div class="epigraph">
<p>入口把前缀或流量类映射成标签，中间只交换标签；把 TE 与 VPN 从「每跳最长前缀」里解放出来。</p>
<footer>—— 据 RFC 3031 MPLS 架构；RFC 5036 LDP；RFC 3209 RSVP-TE 整理</footer>
</div>

[上一课](/cs/traffic-engineering) 要显式路径。缺口是**数据面怎么走显式路径而不改 IP 头**：MPLS 标签栈、LDP 与 RSVP-TE。本课不把段路由 SID 写完。

## 问题

IP 每跳 LPM，路径随 IGP 变。MPLS：ingress 压入标签，transit 查 ILM 换标签，egress 弹出。PHP 可在倒数第二跳弹出。TE LSP 用 RSVP 预留带宽；LDP 跟 IGP 拓扑走，不单独 TE。VPN（后课对照 EVPN）用两层标签：外层走隧道，内层选 VRF。

<span class="marginnote">LPM（最长前缀匹配）就是每一跳都拿目的 IP 与路由表里所有前缀比对，选掩码最长的那条。它是 IP 转发的本职，但也意味着路径由每跳各自的路由表决定；MPLS 入口一次定夺、中间照标签走，显式路径才有了着落。</span>

不要把标签当成 MAC：标签是本跳局部的，MAC 是接口烧录或学习来的。

<span class="marginnote">RFC 3031。标签 20 比特。流量工程与 VPN 是 MPLS 的两大理由，不是「比 IP 更快的查找」神话——TCAM 后课会说 IP 也能线速。</span>

### 标签是局部 FEC

中间不查用户 IP。LDP 跟随 IGP，RSVP-TE 带带宽。VPN 常用两层标签。更快查找不是引入 MPLS 的理由。

## 方法

画：IP 包 → 压标签 → 交换 → 弹标签 → IP。对照 VLAN：VLAN 是广播域着色，MPLS 是转发等价类。与 GRE 后课对照：GRE 是 IP-in-IP 风格隧道，中间仍 LPM。

```mermaid
flowchart TD
  ING["入口压入"] --> SW["中间交换标签"]
  SW --> EGR["出口弹出"]
  FEC["转发等价类"] --> ING
```

## 机制

BGP 下一跳可达可变成「到下一跳的 LSP」，加速收敛（PIC）是工程。IXP 上通常仍是 IP 对等，MPLS 多在 AS 内。PFC 可在 MPLS 以太网上按 TC 映射，与有线课衔接。

失败：LSP 断了要 make-before-break 或 IGP 绕路，否则黑洞，像 RR 藏备选。

```mermaid
flowchart LR
  PKT["IP 包: 目的 10.1.1.0/24"] --> ING["入口 LER: 查 FEC 压入标签 100"]
  ING --> T1["中间 LSR: ILM 把 100 换成 200"]
  T1 --> PEN["倒数第二跳: PHP 提前弹出标签"]
  PEN --> EGR["出口 LER: 收到裸 IP, 正常转发"]
  T1 --> NOTE["全程未对该 IP 做最长前缀匹配"]
  NOTE --> EGR
```

<span class="marginnote">PHP（penultimate hop popping，倒数第二跳弹出）就是让弹出标签的活儿由倒数第二个路由器提前干，出口路由器收到时已经是干净 IP 包。省的是出口那一跳「弹完再查」的双重查表——一两微秒在核心网里也是钱。</span>

## 边界

本课不引入 GMPLS 光层。分段路由是下一课。后课默认：MPLS 用标签换路径与 VPN 上下文；中间不查 IP。

「MPLS 已死」口号忽略大量运营商 VPN；SR 是演化不是瞬间替换。

<span class="marginnote">可以把标签想象成行李托运条：入口柜台贴上「中转北京→伦敦」，中间每一站只扫条换条，不需要拆开行李看你到底要去哪条街（目的 IP）。标签本站生效、用完即换，MAC 地址则是网卡出厂就带着的身份证，两者完全不是一类东西。</span>

下一课[分段路由](/cs/segment-routing)。

## 小结

- 标签转发实现 FEC 与显式路径。
- LDP 跟随 IGP；RSVP-TE 做带宽 LSP。
- 中间跳不必 LPM 该用户 IP。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 3031；RFC 3209；RFC 5036。
