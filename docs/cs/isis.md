---
title: IS-IS
date: 2026-09-08
section: cs
---

# IS-IS

<div class="epigraph">
<p>IS-IS 同样洪泛链路状态、同样跑 SPF，但 PDU 走链路层，层级是 L1/L2，TLV 让它后来比 OSPFv2 更容易带新地址族。</p>
<footer>—— 据 ISO/IEC 10589；RFC 1195 把 IS-IS 用于 IP；RFC 5308 IPv6 整理</footer>
</div>

[上一课](/cs/ospf-areas-lsa) 钉死 OSPF 区域。运营商骨干常跑 IS-IS。缺口是**对照**：不是再学一遍 Dijkstra，而是 CLNS 出身的两级层级、LSP 与 TLV。本课不把 RIP 计时器写完。

## 问题

OSPF 是 IP 上的协议，依赖 IP 可达来传 LSA（邻接除外）。IS-IS Hello/LSP 可直接封装在二层，地址族可后加 TLV——历史上加 IPv6、SR 比 OSPFv2 补丁轻松。L1 像非骨干区域，L2 像骨干；L1/L2 路由器相当于 ABR。度量原 6 比特，宽度量后才适合 TE。

<span class="marginnote">术语翻译：TLV 就是「类型-长度-值」三件套——先报这是什么（Type）、再报有多长（Length）、最后放内容（Value）；路由器遇到不认识的类型按 Length 跳过即可，这正是它后来能无痛加 IPv6 和 SR 的原因。</span>

不要把 IS-IS 写成「二层路由替代 STP」：它算的是 IP（或 CLNS）下一跳，不是 MAC 洪泛树。

<span class="marginnote">ISO 10589。RFC 1195 引入 IP。DIS 类似 DR 但广播网上选出。本课不背 NET 地址编码。</span>

### 二层 PDU 仍算 IP 下一跳

不是 STP 替代。L1/L2 对应区域分层，TLV 方便加地址族与 SID。与 OSPF 选谁是生态，SPF 同构。

## 方法

对照表：邻接、层级、扩展性、BFD 配套。画：L1 区 → L1/L2 → L2 骨干。与 OSPF 一样可 ECMP。承载 SR 前缀 SID 是后课，这里只承认 TLV 可扩展。

```mermaid
flowchart TD
  L1["L1 LSP 区内"] --> L12["L1/L2 边界"]
  L12 --> L2["L2 骨干"]
  TLV["TLV 扩展"] --> AF["IPv4/IPv6/SR"]
```

## 机制

主干链路状态课的洪泛可靠性（序号、老化）两边同构。选择哪一个往往是运维传统与 TE/SR 生态，不是容量公式。卫星高延迟链路上 Hello 死亡计时要放宽，与介质无关的协议定时器问题。

<span class="marginnote">为什么重要：卫星链路单程就有几百毫秒抖动，若照搬地面默认的「10 秒 Hello、30 秒判死」，一次抖动就可能误判邻居死亡，触发全网 SPF 重算——定时器必须随介质放宽。</span>

DIS 与 OSPF DR：都是压住广播网全网状开销的机制，细节不同——OSPF 让非 DR 路由器只与 DR/BDR 完全邻接，IS-IS 邻接仍全网状、靠伪节点收敛 LSP 描述；对象同类。

<span class="marginnote">直觉类比：广播网选出的 DIS 像班会里的「记录员」——大家彼此仍都认识，但纪要按伪节点一份来写：每台在 LSP 里只报一条到伪节点的链路，伪节点统一列出全员；10 台设备两两直连要描述 45 条边，经伪节点只需每台 1 条加一份 10 人名单。</span>

一条 LSP 里 TLV 如何实现向前兼容：

```mermaid
flowchart LR
  LSP["一个 IS-IS LSP"] --> H["固定头部"]
  LSP --> T1["TLV: IS 邻居"]
  LSP --> T2["TLV: IPv4 前缀"]
  LSP --> T3["TLV: IPv6 前缀（后加）"]
  LSP --> T4["TLV: SR SID（后加）"]
  T3 --> OLD["老路由器按 Length 跳过，照常转发"]
```

## 边界

本课不引入 IS-IS 多拓扑的全部。RIP 对照是下一课。后课默认：域内链路状态有 OSPF 与 IS-IS 两套互操作栈，SPF 相同。

不要同时在同一链路上跑两套无必要的 IGP。

下一课[RIP 对照](/cs/rip)。

## 小结

- IS-IS：二层 PDU、L1/L2、TLV 扩展。
- 计算仍是 SPF，不是距离向量。
- 与 OSPF 分工是工程生态，不是对错。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：ISO 10589；RFC 1195；RFC 5308。
