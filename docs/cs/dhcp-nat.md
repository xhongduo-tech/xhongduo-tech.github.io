---
title: DHCP 与 NAT
date: 2026-09-08
section: cs
---

# DHCP 与 NAT

<div class="epigraph">
<p>DHCP 把地址与默认路由租给主机；NAT 让私网多台主机复用少量公网地址，改写 IP 与端口。</p>
<footer>—— 据 RFC 2131, Dynamic Host Configuration Protocol；RFC 3022, Traditional IP Network Address Translator 整理</footer>
</div>

[上一课](/cs/icmp)假定主机已有地址。[IP 编址](/cs/icmp)要每接口一个 IP；手工配置不能规模化，IPv4 公网地址也不够每人一台。缺口是两件配套：**DHCP 租约**与 **NAT 复用**。本课不把 IPv6 地址空间当已解决。

## 问题

主机启动时还没有 IP，不能先打开一个「配置 TCP」。DHCP：发现、提供、请求、确认，在 UDP 广播/中继上完成，交出地址、掩码、网关、DNS（DNS 课再用）。NAT：边界路由器改写私网源地址为公网，用端口区分会话，回包再改回。缺口不是 LPM 公式，而是地址的**获得与改写**。

本课不把 CGNAT 的全部端口耗尽策略写完。

<span class="marginnote">私网段 RFC 1918：`10/8`、`172.16/12`、`192.168/16`。NAT 破坏「每个地址全球可达」，与端到端论证紧张：新会话难从外向内建。</span>

<span class="marginnote">直觉类比：DHCP 像「酒店前台办入住」——你走进大堂广播一句「要房」（Discover），前台递上一张房卡（Offer），你确认要这间（Request），前台登记生效（Ack）。退房时间就是租约，到期不续，房卡（IP）就作废。</span>

## 方法

DHCP 客户端在 UDP 68，服务器 67；租约到期续租。NAT 表：(协议, 私网 IP, 端口) ↔ (公网 IP, 端口)，对 TCP/UDP 有状态；ICMP 用查询 ID 等。TTL 与校验和随改写重算。ALG 曾修补 FTP 等嵌地址的协议，脆弱，主干不依赖。

<span class="marginnote">数字实例：家里三台设备同时访问 `203.0.113.7:443`。NAT 把它们分别改写成 `198.51.100.1:40001`、`:40002`、`:40003` 发出去——公网地址只有一个，靠不同源端口区分是谁的会话；回包按端口对号入座，再改回各自的私网地址。</span>

```mermaid
flowchart TD
  HOST["无地址主机"] --> DHCP["DHCP 租约"]
  PRIV["私网分组"] --> NAT["改写地址端口"]
  NAT --> WAN["公网"]
```

## 机制

DHCP 填好[子网](/cs/ip-subnet)课的配置，主机才能 ARP 网关。NAT 让 [LPM](/cs/lpm) 在公网只看见少量前缀，却让传输层五元组在边界被改写——后课 TCP 必须容忍，或应用改用中继。端到端：NAT 是中间盒状态，不是 Saltzer 意义上的「端」。这是后课 QUIC 与 NAT 超时、以及 IPv6 推动的背景。

上面那张图画「主机与分组整体往哪流」；这张拆开 DHCP 那一格：一台主机开机时 IP 究竟是经过哪四步对话拿到的，之后又怎么续。

```mermaid
flowchart TD
  C["客户端广播 DISCOVER 我还没有地址"] --> S["服务器回 OFFER 提议一个地址"]
  S --> C2["客户端广播 REQUEST 我要这个"]
  C2 --> S2["服务器 ACK 正式租给你 并带网关 DNS"]
  S2 --> USE["客户端配置好 开始上网"]
  USE --> REN["租约过半 发 REQUEST 续租"]
```

<span class="marginnote">常见误区：以为电脑「有 IP 就能被外网直接访问」。私网地址（如 `192.168.1.5`）在公网路由表里根本不可达，外网主动连进来会卡在 NAT——查不到对应表项就无处转发。所以 P2P 联机才需要打洞或中继来帮忙。</span>

与[进程](/cs/process-image)无关：改写发生在路由器，主机套接字仍绑定私网地址。

## 边界

本课不把 DHCP 欺骗写成攻击教程。不引入 IPsec 穿越 NAT 的全部封装。IPv6 用 SLAAC/DHCPv6，对照下一课；不要把 NAT66 当 IPv6 的必经之路。

中继代理让服务器不在广播域内。NAT 对分片重组的依赖是实现地雷，主干假定不分片或先重组再改写。

租约续期失败则地址应停止使用；NAT 表项超时则内网会话静默断开。

后课默认：IPv4 常靠 DHCP 与 NAT 撑着。地址长度与邻居发现如何换一套，下一课 IPv6 对照。

## 小结

- DHCP（RFC 2131）租地址与网关；NAT（RFC 3022）复用公网地址。
- NAT 改写五元组，削弱端到端可达。
- IPv6 用更大地址空间对照，下一课。
- 出处：RFC 2131；RFC 3022；RFC 1918；Kurose and Ross。
