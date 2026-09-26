---
title: 路径 MTU 发现
date: 2026-09-08
section: cs
---

# 路径 MTU 发现

<div class="epigraph">
<p>把 DF 置位，让太小的一跳用 ICMP 报告；过滤 ICMP 就把发现变成黑洞，主机只能靠探测或保守 1280/576。</p>
<footer>—— 据 RFC 1191；RFC 8201 IPv6 PMTUD；RFC 4821 分组化 PLPMTUD 整理</footer>
</div>

[巨帧](/cs/jumbo-mtu) 与 [GRE](/cs/gre-tunnels)、[VXLAN](/cs/vxlan-overlay)、[IPv6 过渡](/cs/ipv6-transition) 都吃 MTU。[上一课](/cs/ipv6-transition) 留下隧道头税。缺口是**沿路最小 MTU 怎么发现**。本课不把 traceroute 写完。

## 问题

源不知道路径。IPv4 PMTUD：DF=1，路由器丢弃并 ICMP Fragmentation Needed。IPv6 无源分片，只能这样。黑洞：中间丢大包且不回 ICMP（防火墙「安全」），TCP 卡住。PLPMTUD：传输层用探测与丢失推断，不依赖 ICMP。MSS 钳制后课会在边缘傻切，是权宜。

不要把 PMTUD 写成一次性：路由变化、ECMP 不同成员，最小 MTU 会变。

<span class="marginnote">RFC 4821 针对黑洞。IPv6 最小链路 MTU 1280。本课不把每个 ICMP 代码背完。</span>

<span class="marginnote">DF 就是 Don't Fragment（不许分片）标志位：源在包上贴一张「不许半路拆散」的纸条。路由器装不下又不敢拆，只能把包扔掉并回一张 ICMP 通知——「这段路只有这么宽」。PMTUD 的全部机制就建立在这张回执上。</span>

<span class="marginnote">数字实例：以太网 MTU 1500，扣掉 IP 头 20 和 TCP 头 20，TCP 的 MSS 是 1460；中间套一层 VXLAN 隧道再吃掉约 50 字节头税后，路径有效 MTU 掉到 1450——源仍按 1500 发的包就会在隧道口被丢，这正是「头税」引爆 MTU 问题的典型场景。</span>

### ICMP 被滤则黑洞

IPv6 必须发现。PLPMTUD 用传输探测。ECMP 与隧道使缓存失效。统一 MTU 可省掉发现。

## 方法

画：发大包 → ICMP → 降 MTU → 缓存。对照分片：中间分片与端到端发现，Saltzer 偏向端。隧道口应发 ICMP 或预先降低。

```mermaid
flowchart TD
  BIG["DF 大包"] --> DROP["一跳太小"]
  DROP --> ICMP["ICMP 通告 MTU"]
  ICMP --> CACHE["主机缓存"]
  DROP --> BLK["无 ICMP 则黑洞"]
```

## 机制

卫星与 5G 切片不改算法，只改 RTT，探测更慢。交换机巨帧不一致是二层版同一 bug。多播很少做 PMTUD，常用保守长度。安全：ICMP 可被伪造来缩小 MTU 做攻击，主机要合理性检查。

与容量无关：MTU 摊头税，不提高 $C$。

ICMP 被滤时，PLPMTUD 凭什么还能发现路径上限？两种发现方式的分岔如下。

```mermaid
flowchart TD
  Q["怎么探出路径能扛多大的包？"] --> PM["经典 PMTUD：靠路由器报信"]
  Q --> PL["PLPMTUD：靠自己的丢包推断"]
  PM --> P1["发 DF 大包，等 ICMP 带回该跳 MTU"]
  P1 --> P2["按通告值缩包并缓存"]
  PL --> L1["从小尺寸起逐步加大探测包"]
  L1 --> L2["探测包丢了：退回上次确认过的值"]
  L2 --> L3["全程不依赖 ICMP，防火墙滤掉也不怕"]
```

<span class="marginnote">常见误区：以为防火墙拦掉 ICMP 更安全。拦掉 Fragmentation Needed 这一类报错，等于把报信人灭口——大包悄悄消失、连接莫名卡死（黑洞），故障既难复现又不该发生。安全策略应放行这一类 ICMP，或直接改用 PLPMTUD。</span>

## 边界

本课不引入 IPv6 原子分片扩展头的全部争议。TTL 与 traceroute 是下一课。后课默认：路径 MTU 靠 ICMP 或传输探测；过滤 ICMP 要给发现留洞。

数据中心可全路径统一 9000，省掉发现，这是运营契约。

下一课[TTL 与 traceroute](/cs/ttl-traceroute)。

## 小结

- DF + ICMP 发现最小 MTU；IPv6 必须。
- ICMP 被滤则黑洞，用 PLPMTUD。
- 隧道与 ECMP 使缓存失效。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 1191；RFC 8201；RFC 4821。
