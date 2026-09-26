---
title: Nagle 与延迟 ACK
date: 2026-09-08
section: cs
---

# Nagle 与延迟 ACK

<div class="epigraph">
<p>小段太密会浪费头开销；Nagle 等一个 ACK 再发新的小段。接收方延迟 ACK 想捎带，两套等待叠在一起就会卡住。</p>
<footer>—— 据 Nagle, RFC 896；RFC 1122 对延迟 ACK；Kurose and Ross</footer>
</div>

[上一课](/cs/tcp-window)用 `rwnd` 防止淹没对端。本课不重讲滑动窗口。缺口是交互式与 RPC 式负载：一次写几个字节，每个都独立成段，头比载荷大；另一方面接收方为了把 ACK 与反向数据合并而等待。两者各自合理，合在一起造成 **ACK 死锁式延迟**。拥塞窗口下一课才出现。

## 问题

Nagle：若有未确认数据，就把后续小写缓存，直到 ACK 到来或凑满 MSS。延迟 ACK：不立刻确认每个段，等一小段时间或等到两个段。若发送方等 ACK 才发下一小段，接收方等第二段才 ACK，双方都在等。缺口是**把流控窗口里的发送时机写清**，不是新的序号规则。

<span class="marginnote">术语翻译：Nagle 是「凑满一车再发车」——上一车还没到站确认，新的小件先攒在仓库；延迟 ACK 是「收件不急着签收，等下一件一起签」。两个策略各自都在省路费（报文头开销），但班车等签收、签收等班车，就互相锁死了。</span>

<span class="marginnote">关闭 Nagle（TCP_NODELAY）是应用在「延迟敏感、载荷小」时的选择，不是协议错误。延迟 ACK 也有上限，避免无限等。</span>

## 方法

实现按 RFC 896 与主机需求（RFC 1122）组合：大块传输几乎不受 Nagle 影响（很快满 MSS）；请求–响应当两端都小段时最疼。修复是打破其中一侧等待：禁 Nagle，或更及时的 ACK。本课不把应用层缓冲策略写成框架教程。

```mermaid
flowchart TD
  SMALL["小写"] --> NAGLE["有未确认则缓存"]
  DEL["延迟 ACK"] --> WAIT["等第二段或超时"]
  NAGLE --> STALL["双方等待"]
  WAIT --> STALL
```

## 机制

Nagle 减的是报文率，保护的是共享路径上的头开销；延迟 ACK 减的是纯 ACK 报文。它们优化的是性能，不改变 RFC 793 的可靠性。[端到端](/cs/layering-e2e) 字节流语义仍在；变的是何时离开 Nagle 缓冲区。后课拥塞控制会再叠一层「飞行中能有多少 MSS」。

<span class="marginnote">数字实例：设 RTT 为 10 ms，接收方延迟 ACK 定时器为 40 ms。一次小请求–小响应本应 10 ms 往返，现在发送方的小响应被 Nagle 扣住等 ACK，接收方又等到定时器超时才确认——一次往返被拉长到约 40 ms 以上。每秒本可完成约 100 次交互，现在只剩二十几次。</span>

<span class="marginnote">常见误区：初学者以为 TCP_NODELAY 关掉 Nagle 会丢数据或破坏可靠性——实际上它只改变小段「何时离开本机」，字节流的有序、可靠、流控语义一点不变；代价只是网络上小报文变多、头开销变大。</span>

```mermaid
flowchart TD
  REQ["请求段发出"] --> ARR["到达接收方"]
  ARR --> DACK["延迟 ACK：等反向数据捎带"]
  DACK --> NAG["Nagle：有未确认数据，扣住小响应"]
  NAG --> T["延迟 ACK 定时器超时"]
  T --> A["纯 ACK 发出"]
  A --> S["Nagle 解锁，小响应才发"]
```

## 边界

本课不引入全部 Cork 与自动 Corking 的平台细节。网络队列满时即使 rwnd 很大也不能发，下一课用 `cwnd`。

后课默认：小段可能被故意推迟。瓶颈队列的公平与稳定是拥塞控制的缺口。

## 小结

- Nagle 合并小段；延迟 ACK 合并确认。
- 两套等待叠加会放大 RTT。
- 保护网络队列是下一课拥塞控制。
- 出处：RFC 896；RFC 1122；Kurose and Ross。
