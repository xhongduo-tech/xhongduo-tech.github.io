---
title: GeoDNS
date: 2026-09-08
section: cs
---

# GeoDNS

<div class="epigraph">
<p>权威按解析器地址估用户位置，返回近的 A/AAAA；TTL 与缓存位置决定这份「地理」有多真。</p>
<footer>—— 据 RFC 7871 EDNS Client Subnet；CDN 实践；主干任播课对照整理</footer>
</div>

[DNS TTL](/cs/dns-cache-ttl) 让答案被记住。[任播](/cs/anycast-edge) 是路由侧近。[上一课](/cs/doh-dot) 把递归器集中，位置更假。缺口是 **GeoDNS**：按源 IP 或 ECS 选答案。本课不把 SMTP 写完。

## 问题

全球用户打同一主机名，要近的边缘。[CDN 直觉](/cs/cdn-intuition) 已点名。权威看递归器 IP 猜用户，误差大（公共解析器）。ECS 把客户端前缀带给权威，提高精度、泄露位置。任播：同一地址，路由选近；GeoDNS：不同地址。两者常叠。RPKI 管那些地址的源 AS。

不要把 GeoDNS 写成 GPS。

<span class="marginnote">RFC 7871。隐私与缓存碎片是代价。本课不点名厂商。</span>

### 不是 GPS

按递归器或 ECS 估位置。任播是同址路由。TTL 约束切换。公共 DoH 使无 ECS 更钝。

## 方法

对照：GeoDNS / 任播 / HTTP 302。画：查询源 → 地图 → 近记录。TTL 短才能切失败节点，与缓存课衔接。DoH 大解析器使无 ECS 时更钝。

```mermaid
flowchart TD
  Q["查询"] --> SRC["递归器或 ECS"]
  SRC --> MAP["地理/拓扑图"]
  MAP --> ANS["近的地址"]
```

## 机制

负载均衡后课可再在地址集合上做。BGP 劫持会把「近」变成攻击者。测量 ping 后课验证是否真近。H3 HTTPS RR 也可按地返回不同目标。

缓存：同一递归器后的多国用户共享假位置，是公共 DNS 的经典坑。

<span class="marginnote">直觉类比：GeoDNS 像连锁快餐的「最近门店」告示牌——但它看的不是你手里拿着的地址，而是给你带话的那位朋友（递归解析器）住在哪。朋友住得越远、越集中（如 8.8.8.8），告示牌指的路就越不准；ECS 相当于朋友替你报出你家所在的街区号。</span>

<span class="marginnote">数字实例：TTL 设 86400 秒（一天），权威换了答案后，全球各处缓存最长还要再服务旧地址一整天，故障切换就是一天的瘫痪；TTL 设 30 秒则切换近乎实时，但解析请求量放大——TTL 就是在「切换速度」与「缓存省流量」之间拨的旋钮。</span>

```mermaid
flowchart TD
  U["用户 203.0.113.9"] --> R{"递归器是谁"}
  R -->|"公共 DNS 无 ECS"| GUESS["按递归器位置猜误差大"]
  R -->|"带 ECS 前缀"| FINE["按用户前缀选近节点"]
  GUESS --> CACHE["答案缓存 TTL 秒"]
  FINE --> CACHE
  CACHE --> SERVE["后续用户共用这份答案"]
```

## 边界

本课不引入 IP 地理库的全部误差分析。SMTP/IMAP 是下一课。后课默认：GeoDNS 用查询源近似位置；ECS 更准更露。

把 TTL 设很长，故障切换要等过期。

下一课[SMTP / IMAP](/cs/smtp-imap)。

## 小结

- 按解析器/ECS 选近地址。
- 与任播互补：一名多址 vs 同址路由。
- 公共解析器钝化地理。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 7871；CDN 实践。
