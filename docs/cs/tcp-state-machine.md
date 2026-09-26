---
title: TCP 状态机
date: 2026-09-08
section: cs
---

# TCP 状态机

<div class="epigraph">
<p>连接是状态机：LISTEN、握手三态、ESTABLISHED、四次挥手各态；错误的迁移比错一个 cwnd 更容易把套接字卡死。</p>
<footer>—— 据 RFC 9293 第 3.3 节；Stevens, TCP/IP Illustrated 状态图整理</footer>
</div>

主干[三次握手](/cs/tcp-handshake) 与 [TIME_WAIT](/cs/tcp-time-wait) 已拆过。[上一课](/cs/mss-clamping) 发生在 SYN 上。缺口是**整图**：11 个状态怎么迁，ABORT vs 优雅关。本课不把 SYN cookie 写完。

## 问题

应用看 ESTABLISHED 与 close。内核还要 SYN-SENT、SYN-RECEIVED、FIN-WAIT-1/2、CLOSING、CLOSE-WAIT、LAST-ACK、TIME_WAIT。同时打开、同时关闭是边角。RST 从多数态回到 CLOSED。半开：一边已关或崩溃，另一边还 ESTABLISHED——保活后课。窗口缩放只在 SYN 协商，状态错了不能补选项。

不要把状态机写成拥塞控制；cwnd 是 ESTABLISHED 里的变量。

<span class="marginnote">RFC 9293 图 5。本课不把每条错误迁移画成考试填空，只钉主路径与 TIME_WAIT 理由（已在主干）。</span>

### 拥塞变量栖在 ESTABLISHED

应用 close 不等于 CLOSED。RST 中止；TIME_WAIT 防旧段。同时打开是规范允许的边角。

## 方法

画主路径：LISTEN → SYN-RCVD → EST → 主动关 FIN-WAIT → TIME_WAIT。对照 QUIC：连接 ID 迁移后课，状态更少暴露。

```mermaid
flowchart TD
  LI["LISTEN"] --> SR["SYN-RECEIVED"]
  SR --> ES["ESTABLISHED"]
  ES --> FW["FIN-WAIT"]
  FW --> TW["TIME_WAIT"]
  ES --> RST["RST 到 CLOSED"]
```

<span class="marginnote">数字实例：`netstat` 里成千上万个 CLOSE-WAIT，几乎都是应用代码忘了调 close——内核在等应用表态，等不到就永远停在这个态，文件描述符随连接一起泄漏。</span>

## 机制

SYN 泛洪把 SYN-RCVD 队列填满，下一课 cookies。LACP 与路由变化不改状态机，只改路径。抓包排障：看标志位对状态，比看窗口更先。incast 不改状态，只改拥塞变量。

同时打开：双方 SYN-SENT → SYN-RCVD → EST，序号交叉，规范允许。

上面的主路径是服务器视角。拆开拆除过程，主动关闭与被动关闭两侧走的是两组不同状态：

```mermaid
flowchart TD
  ES["ESTABLISHED"] --> ACT["主动方发 FIN"]
  ES --> PAS["被动方收 FIN, 回 ACK"]
  ACT --> FW1["FIN-WAIT-1"]
  FW1 --> FW2["FIN-WAIT-2, 等对方 FIN"]
  FW2 --> TW["TIME_WAIT"]
  PAS --> CW["CLOSE-WAIT, 应用不 close 就停在这"]
  CW --> LA["LAST-ACK, 发出 FIN"]
  LA --> CLS["CLOSED"]
  TW --> CLS
```

<span class="marginnote">常见误区：容易以为"状态"是两端共享的一个值。实际上每端各持一份：主动关的一端走 FIN-WAIT 与 TIME_WAIT，被动关的一端走 CLOSE-WAIT 与 LAST-ACK，同一时刻两边可以处于完全不同的状态。</span>

## 边界

本课不引入 TCP 快速打开的全部 cookie。SYN cookies 是下一课。后课默认：套接字生命周期 = 这份状态机；应用 close 不等于立刻 CLOSED。

把 TIME_WAIT 当泄漏而全局关，会制造旧段串连接。

<span class="marginnote">直觉类比：优雅关闭像双方道完别再散场；RST 则是把灯直接拉灭——不管对方话有没有说完，两端立刻回 CLOSED，之后迟到的发言自然无人接收。</span>

下一课[SYN cookies](/cs/syn-cookies)。

## 小结

- 主路径：听、握、传、挥、TIME_WAIT。
- RST 中止；半开留给保活课。
- 拥塞变量栖在 ESTABLISHED。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9293；Stevens。
