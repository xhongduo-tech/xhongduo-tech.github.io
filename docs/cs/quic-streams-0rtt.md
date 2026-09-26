---
title: QUIC 流与 0-RTT
date: 2026-09-08
section: cs
---

# QUIC 流与 0-RTT

<div class="epigraph">
<p>QUIC 在 UDP 上做加密运输：多流免 TCP 队头阻塞，TLS 1.3 进握手；0-RTT 用上次票再发数据，有重放代价。</p>
<footer>—— 据 RFC 9000 QUIC；RFC 9001 使用 TLS；RFC 8446 TLS 1.3 整理</footer>
</div>

主干[QUIC 对照](/cs/quic-contrast) 已给动机。[HTTP/2](/cs/http2) 多流仍落在一条 TCP 上。[上一课](/cs/sctp) 对照多流。缺口是 **QUIC 流与 0-RTT**：stream ID、加密、早数据。本课不把连接迁移写完。

## 问题

TCP+TLS：两次握手 RTT，且 TCP 丢包挡住所有 HTTP/2 流。QUIC：运输与加密一体，握手 1-RTT，会话票可 0-RTT 发请求。流独立丢失恢复（包号与流偏移分离）。0-RTT 数据可被重放，只适合幂等——后课重试会收回。UDP 被中间限速是部署税。

不要把 0-RTT 写成「无密钥」。

<span class="marginnote">RFC 9000。拥塞仍在连接级，避免每流 AIMD 加倍。本课不把帧类型列完。</span>

### 运输与 TLS 一体

多流免 TCP HOL。0-RTT 有重放，只适合幂等。拥塞在连接级。UDP 被限则回退。

## 方法

对照：TCP HOL vs QUIC 流。画：TLS 密钥 → 包保护 → 多 STREAM 帧。与 SYN cookies：QUIC Retry 抗泛洪。

```mermaid
flowchart TD
  TLS["TLS 1.3 进 QUIC"] --> KEY["包保护"]
  KEY --> ST["多流"]
  TKT["会话票"] --> Z["0-RTT 早数据"]
  Z --> REP["重放风险"]
```

## 机制

TSO 变成 UDP GSO。ECN 计数在 QUIC 里。PMTUD 在用户态。BBR 常与 QUIC 同部署。SCTP 多流无 Web 加密集成。窗口是流控窗口+拥塞窗口两层，类似 HTTP/2 但加密。

握手失败回退 TCP 是浏览器政策，不是协议强制。

```mermaid
flowchart TD
  F1["流1: 图片块丢失"] --> W["还在等重传"]
  F2["流2: HTML 文本完好"] --> Q{"同一传输层?"}
  F1 -->|"TCP"| H["全部流卡住"]
  F2 -->|"TCP"| H
  F1 -->|"QUIC"| OK["流2 照常送达应用"]
  F2 -->|"QUIC"| OK
```

<span class="marginnote">术语翻译：队头阻塞（HOL blocking）就是「一队人过安检，第一个人包里查出违禁品，后面所有人都得等着」——TCP 只认字节序号，不知道里面装了几条 HTTP 流，一个字节丢了全体重排队。</span>

<span class="marginnote">数字实例：一条网页要 3 个流，丢包率 1% 时每个流平均要重传 1 次，TCP 下 3 次重传串行排队；QUIC 下 3 条流独立恢复，最坏也只多等一个 RTT，这就是流 ID 与包号分离设计的收益。</span>

<span class="marginnote">常见误区：0-RTT 不是「跳过加密」。密钥照样要用上次会话票派生，只是省掉了完整握手那一个来回；代价是这批早数据没有抗重放保护，银行转账这种非幂等请求绝不能放进 0-RTT。</span>

## 边界

本课不引入 DATAGRAM 扩展全文。QUIC 连接迁移是下一课。后课默认：QUIC 多流+集成 TLS；0-RTT 仅幂等。

公司防火墙「UDP 危险」会把用户打回 TCP HOL。

下一课[QUIC 连接迁移](/cs/quic-migration)。

## 小结

- 多流消除运输队头阻塞。
- 1-RTT 握手，0-RTT 有重放。
- 拥塞在连接级，流控在流级。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9000；RFC 9001；RFC 8446。
