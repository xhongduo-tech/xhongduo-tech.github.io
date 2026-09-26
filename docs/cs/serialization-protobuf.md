---
title: 序列化：JSON / Protobuf
date: 2026-09-08
section: cs
---

# 序列化：JSON / Protobuf

<div class="epigraph">
<p>JSON 是自描述文本，便于调试与浏览器；Protobuf 是带 schema 的二进制，省带宽与 CPU。都是应用编码，不是 HTTP。</p>
<footer>—— 据 RFC 8259 JSON；Protobuf 语言指南整理</footer>
</div>

[上一课](/cs/rest-grpc) 假定有一种编码。[内容协商](/cs/content-negotiation) 可选 `application/json`。缺口是**编码本身**：字段、版本、与报文成帧。本课结束 HTTP 与 Web 协议课序。

## 问题

进程内存结构不能直接上[套接字](/cs/socket-api)。JSON：Unicode 文本、对象图、无原生二进制（用 base64）。Protobuf：字段号、varint、向前兼容靠未识别字段保留。gRPC 默认后者。压缩可再叠 gzip，但 Protobuf 已密，收益小。schema 演化：加可选字段易，改语义难——与 DB 模式后课不同栏，这里只钉报文。

不要把 Protobuf 写成加密。

<span class="marginnote">RFC 8259。Protobuf 是 Google 开文档的 IDL。本课不把每种 IDL（Thrift/Avro）列完。</span>

<span class="marginnote">数字实例：varint 把整数按 7 位一段编码，每段最高位当「还有下一字节」的旗子。数字 300 = 二进制 100101100，拆成 10 与 44，线上写成两个字节——小整数通常只占 1 字节，这是它比定长 4 字节 int 省带宽的原因。</span>

### 编码不等于成帧

JSON 自描述；Protobuf 靠字段号。无长度前缀仍会粘包。schema 演化属于合同。

## 方法

对照：可读 vs 紧凑；有无 schema。画：结构 → 字节 → 解析。与线路码对照：一层在 PHY，一层在应用。粘包后课：Protobuf 消息仍要长度前缀或 gRPC 帧。

```mermaid
flowchart TD
  OBJ["内存对象"] --> JSON["自描述文本"]
  OBJ --> PB["字段号二进制"]
  PB --> WIRE["再进 HTTP/gRPC 帧"]
```

## 机制

CPU：JSON 解析在 10G 上可成瓶颈，TSO 帮不上。QUIC 0-RTT 重放对非幂等 POST+JSON 危险。CDN 可缓存 GET+JSON，难缓存 gRPC。IPv6 过渡不影响编码。

安全：JSON 炸弹、Protobuf 递归深度是解析器边界。

上面那张图画的是「对象怎么变成字节」；下面这张回答「加了新字段，旧服务为什么还能解析」——向前兼容就发生在字段号对不上的那一刻。

```mermaid
flowchart TD
  A["读到字段头: 字段号+类型"] --> B{"旧 reader 认识此字段号?"}
  B -- "认识" --> C["按类型读值填进结构"]
  B -- "不认识" --> D["按长度跳过 并保留原始字节"]
  D --> E["写回时原样带上"]
  C --> F["继续下一段 直到消息结束"]
  E --> F
```

<span class="marginnote">常见误区：把「能解析」当「安全」。几 KB 的 JSON 可以展开成上 GB 的嵌套对象（JSON 炸弹），Protobuf 也有递归深度限制——解析器必须设深度与大小上限，否则一个恶意报文就能吃光内存。</span>

## 边界

本课不引入 ASN.1 的全部。DNS 缓存与 TTL 是下一课序第一课。后课默认：JSON 易互通；Protobuf 要 schema 与帧。

无长度的 JSON 流会粘包，不能只靠 TCP。

<span class="marginnote">术语翻译：「字段号」是 Protobuf 给每个字段编的整数编号，线上传的是编号不是字段名——所以改名不影响编码；但改编号等于换了一个新字段，旧读者会把新数据当「不认识的字段」跳过，数据悄悄丢失。</span>

下一课[DNS 缓存与 TTL](/cs/dns-cache-ttl)。

## 小结

- JSON 自描述；Protobuf 靠字段号与 schema。
- 编码不等于运输，还要成帧。
- 演化规则属于合同，不是语法边角。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 8259；Protobuf 文档。
