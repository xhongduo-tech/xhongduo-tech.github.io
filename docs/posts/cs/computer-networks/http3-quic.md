---
title: HTTP/3 与 QUIC：基于 UDP 的可靠传输
date: 2026-08-07
---

# HTTP/3 与 QUIC：基于 UDP 的可靠传输

<div class="epigraph">
<p>当 TCP 成了瓶颈，工程师们做了一件大胆的事：抛弃 TCP，在 UDP 之上重新发明一个「更好的 TCP」。</p>
<footer>—— 引自已故现代网络协议的工程叙事</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§6.7 ｜ 2026-08-07</p>
</div>

## 为什么从 HTTP/3 与 QUIC 开始

HTTP/2 解决了 HTTP 层的队头阻塞，但 **TCP 层的队头阻塞**还在：一个 TCP 段丢失，后面的段都要等重传。而且 TCP 的握手太慢（三次握手 + TLS 握手）。**QUIC（Quick UDP Internet Connections）** 的答案惊世骇俗：**放弃 TCP，在 UDP 之上自己实现可靠传输。**<span class="marginnote"><strong>QUIC 为什么用 UDP</strong>：因为 TCP 是「内核里的老顽固」——想改 TCP（比如加多路复用、快握手）要动所有操作系统、所有中间设备，几乎不可能。而 <strong>UDP 是个「白板」——应用层想怎么包就怎么包</strong>。QUIC 在 UDP 之上实现了「自己的 TCP」：可靠、有序、流多路复用、加密内置。<strong>「TCP 改不动，就绕过它」</strong>是 QUIC 的设计哲学。</span>

这一节讲：**TCP 的队头阻塞与握手慢、QUIC 的三大特性、HTTP/3 是什么。**

## 1 TCP 的两个「死结」

HTTP/2 之上，TCP 还有两个难以逾越的问题：<span class="marginnote"><strong>① TCP 层队头阻塞</strong>：TCP 保证「字节流有序交付」——一旦一个段丢失，接收方缓冲区里后面的数据「到齐才交付」，后面的流全被堵住。HTTP/2 的多路复用是「逻辑上的并行」，但底层 TCP 仍是一条「按序管道」——<strong>一个丢包堵住所有流</strong>。<strong>② 握手慢</strong>：TCP 三次握手（1 RTT）+ TLS 握手（1-2 RTT）——新建连接至少要 2-3 个 RTT 才能发第一个字节。<strong>「一条按序管道 + 两轮握手」</strong>是 TCP 的两大死结。</span>

**TCP 队头阻塞**：TCP 按序交付，一个段丢、后面全堵。
**握手慢**：TCP 三次握手 + TLS 握手，2–3 RTT 才发首个字节。

**辨析｜易错点：** **HTTP/2 的多路复用是「应用层的并行」，TCP 层仍是「一条按序管道」**——所以一个 TCP 丢包会让所有 HTTP/2 流都遭殃。**「HTTP/2 解决的是『响应排队』，没解决『字节按序』」**是理解 QUIC 动机的关键。

## 2 QUIC 的三大特性

**QUIC** 针对 TCP 的痛点，给出三大革新：<span class="marginnote"><strong>① 流级独立</strong>：QUIC 里有多个独立的「流」，每个流独立按序交付——一个流丢包，只阻塞那个流，不影响其他流。这从根上消灭了「一个丢包堵所有」的队头阻塞。<strong>② 0-RTT/1-RTT 握手</strong>：QUIC 把加密与传输握手合并，老连接复用可做到 0-RTT（第一个请求就带上数据）——比 TCP+TLS 快 1-2 个 RTT。<strong>③ 连接迁移</strong>：QUIC 连接用「连接 ID」标识而非「IP+端口」——手机从 Wi-Fi 切到 4G，IP 变了，连接不断。<strong>「流独立、快握手、连号不变」</strong>是 QUIC 的三张名片。</span>

1. **流级独立**：每个流独立有序交付，丢包只堵自己的流。
2. **0-RTT 握手**：加密内置、老连接复用，第一请求就带数据。
3. **连接迁移**：用连接 ID 而非 IP 标识连接，换网络不断线。

**辨析｜易错点：** **QUIC 的「连接迁移」是移动互联网的福音**——TCP 连接靠「IP + 端口」标识，IP 一变连接就断；QUIC 用「连接 ID」，IP 变了连接照旧。**「TCP 认地址，QUIC 认 ID」**是对两者连接标识差异的一句话。另一个易错点：**QUIC 的可靠传输是自己在应用层实现的**——它在 UDP 之上做了「自己的 TCP」：序号、确认、重传、拥塞控制全有。**「QUIC 是一个跑在 UDP 上的自定义可靠协议」**。

## 3 QUIC 内置 TLS：安全不是附加

QUIC 的设计有一个深刻变化：**加密不是「加一层」，而是「内置」**。传统 HTTPS = HTTP + TCP + TLS（加密在传输层之上）；QUIC 把 TLS 揉进了协议本身。<span class="marginnote"><strong>QUIC 内置加密的意义</strong>：① <strong>握手合并</strong>——传输握手与 TLS 握手一次完成，省 RTT；② <strong>首部加密</strong>——QUIC 的几乎所有字段（包括流 ID、长度）都加密，只有极少字段明文——<strong>中间设备看不到流的信息，隐私与抗审查性更强</strong>。<strong>「TCP 时代加密是外套，QUIC 时代加密是身体的一部分」</strong>。</span>

**握手合并**：传输 + 加密握手一次完成。
**首部加密**：流 ID、长度等字段都加密，中间设备不可见。
**安全性**：加密成为协议的内置属性，而非附加层。

**辨析｜易错点：** **QUIC 的「全加密」让网络设备（防火墙、QoS）更难看懂流量**——这是它的安全优势，也是网络管理者的挑战。**「加密的代价是可见性，收益是隐私与抗审查」**是 QUIC 引发的治理讨论。

## 4 HTTP/3：就是「HTTP over QUIC」

**HTTP/3**：在 QUIC 之上运行的 HTTP 版本——它继承了 HTTP/2 的「语义与帧」，但把底层从 TCP 换成了 QUIC。<span class="marginnote"><strong>HTTP/3 = HTTP 语义 + QUIC 传输</strong>。版本演进回顾：<strong>HTTP/1.1</strong>`（文本、串行、TCP）→ <strong>HTTP/2</strong>`（二进制分帧、多路复用、TCP，TCP 队头阻塞仍在）→ <strong>HTTP/3</strong>（多路复用 + QUIC，队头阻塞基本消除）。<strong>「HTTP/3 是 HTTP 语义在 QUIC 上的重新落地」</strong>——今天的 Chrome、Edge、Safari 都已默认支持 HTTP/3。</span>

| 版本 | 传输层 | 多路复用 | 队头阻塞 |
| --- | --- | --- | --- |
| HTTP/1.1 | TCP | 无 | HTTP 层 |
| HTTP/2 | TCP | HTTP 层 | TCP 层 |
| HTTP/3 | QUIC（UDP） | 流级 | 基本消除 |

**辨析｜易错点：** **HTTP/3 不是「HTTP/2 + 加密」**——它是「HTTP/2 的语义 + QUIC 的传输」，是一次传输层的彻底更换。**「HTTP/2 改的是应用层组织，HTTP/3 改的是传输层底座」**。另一个易错点：**HTTP/3 仍然叫「HTTP」**——方法、状态码、首部语义与之前完全兼容，只是「跑的方式」变了。

## 5 小结

- **TCP 两大死结**：按序交付的队头阻塞、慢握手。
- **QUIC**：基于 UDP 的自定义可靠协议——流独立、0-RTT、连接迁移。
- **流级独立**：一个流丢包只堵自己，消灭「一丢全堵」。
- **内置 TLS**：握手合并、首部加密，加密成为协议本身的一部分。
- **HTTP/3**：HTTP 语义跑在 QUIC 上——队头阻塞基本消除。
- **演进主线**：HTTP/1.1 串行 → HTTP/2 并行（TCP 内）→ HTTP/3 并行（QUIC 上）。

在下一节，我们将看 Web 的「就近服务」——**内容分发网络（CDN）的工作原理**。
