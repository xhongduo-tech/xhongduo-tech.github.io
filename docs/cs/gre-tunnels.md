---
title: GRE 与隧道
date: 2026-09-08
section: cs
---

# GRE 与隧道

<div class="epigraph">
<p>GRE 把任意载荷装进另一张 IP 头，中间只转发外层；简单、无状态，校验、MTU 与安全都要另配。</p>
<footer>—— 据 RFC 2784 GRE；RFC 2890 密钥与序号；RFC 7676 IPv6 上的 GRE 整理</footer>
</div>

[VXLAN](/cs/vxlan-overlay) 与 [MPLS](/cs/mpls) 已是隧道。[上一课](/cs/evpn) 是控制面。缺口是**最简 IP 隧道**：GRE、何时用、与 IPsec/NVGRE 对照。本课不把 IGMP 写完。

## 问题

要把非 IP 或私网 IP 穿过只懂公网的核心：外层源/目的是隧道端点，内层原样。GRE 无内置拥塞与加密；序号可选。中间 ECMP 只看见外层五元组，熵不足则极化——VXLAN 用 UDP 源端口，GRE 常靠外层 IP 与 GRE 头。递归：隧道口的 MTU 必须小于底层，否则内层 PMTUD 黑洞（后课）。

```mermaid
flowchart TD
  A["内层包 1500 字节"] --> B["加 4 字节 GRE 头"]
  B --> C["加 20 字节外层 IP 头"]
  C --> D{"超过底层链路 MTU 1500？"}
  D -- "是" --> E["中间路由器丢包且不回 ICMP"]
  D -- "否（隧道口 MTU 已调小）" --> F["内层先分段，正常转发"]
  E --> G["大包卡死，小包通：PMTUD 黑洞"]
```

不要把 GRE 当成 VXLAN 替代租户隔离：没有 VNI 标准语义，除非你自己编 key。

<span class="marginnote">RFC 2784。PPTP 等历史用法不展开。IPsec 隧道模式是安全对照，不是 GRE 的升级补丁。</span>

<span class="marginnote">数字实例：底层以太网 MTU 是 1500 字节，GRE 头至少 4 字节、外层 IP 头 20 字节，所以隧道口的有效 MTU 只剩 1476。若不把隧道口 MTU 调小，内层 1500 字节的包封装后变成 1524 字节，中间必丢。</span>

<span class="marginnote">直觉类比：ECMP 分箱就像快递分拣只扫外层条码。所有 GRE 包的外层地址都是同一对隧道端点，分拣机看到的「条码」几乎一样，容易把所有包扔进同一条传送带——这就是「熵不足则极化」，某些链路挤爆、其余闲置。</span>

### 无状态封装

中间只看外层 IP。无内置加密与 VNI 语义。Keepalive 是实现特性。递归 MTU 是黑洞温床。

## 方法

对照：GRE / VXLAN / MPLS / IP-in-IP。画：内层包 → GRE → 外层 IP → 核心 LPM。与蜂窝 GTP 同类：锚点隧道。

```mermaid
flowchart TD
  INN["内层报文"] --> GRE["GRE 头"]
  GRE --> OUT["外层 IP"]
  OUT --> CORE["中间只看外层"]
```

## 机制

OSPF 可在隧道上建邻接（小心递归与 MTU）。流量工程可把 GRE 当一条逻辑边，度量人工设。RSTP 不应把隧道当二层边除非你做了桥——那会把环带进核心。卫星链路上 GRE 只加头税，不减 RTT。

Keepalive 是实现特性，标准 GRE 无强制 Hello。

<span class="marginnote">常见误区：初学者以为 GRE 隧道像一条保持着的有状态连接，断了会有协议来报修。实际上 RFC 2784 的 GRE 完全无状态，两端不交换任何心跳； keepalive 是各家设备厂商自己加的实现特性，能不能用要看具体平台。</span>

## 边界

本课不引入 WireGuard 的加密握手。多播 IGMP/PIM 是下一课。后课默认：GRE 是无状态封装；隔离与安全要另加。

把整个互联网当 GRE 网是 NAT 穿越的权宜，不是架构目标。

下一课[多播 IGMP / PIM](/cs/multicast-igmp-pim)。

## 小结

- GRE：外层 IP 转运，中间无内层状态。
- 熵与 MTU 是隧道共同税。
- 不提供 VPN 密钥语义或 VNI。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 2784；RFC 2890。
