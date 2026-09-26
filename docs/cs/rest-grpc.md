---
title: REST 与 gRPC
date: 2026-09-08
section: cs
---

# REST 与 gRPC

<div class="epigraph">
<p>REST 用资源 URI 与 HTTP 方法约束接口；gRPC 用 HTTP/2 流跑 Protobuf RPC。都是应用合同，不是新的运输层。</p>
<footer>—— 据 Fielding 博士论文 REST；gRPC 超 HTTP/2 实践；RFC 9110 整理</footer>
</div>

[HTTP 语义](/cs/http-semantics) 给方法。[上一课](/cs/sse-server-push) 是一种推。缺口是**两种常见 API 风格**：REST 资源 vs gRPC 过程调用。本课不把 Protobuf 编码写完，下一课写。

## 问题

浏览器与公开 API 适合 REST：GET 可缓存、幂等合同清晰。微服务内部适合 gRPC：存根、流式 RPC、强类型。gRPC 默认 H2，浏览器要 grpc-web。REST 不是「用了 JSON 就 REST」：要统一接口与超媒体约束，课上钉常见子集。错误：REST 用状态码，gRPC 用 status 与 trailer。

不要把 gRPC 写成 QUIC 专属；H3 上的 gRPC 在演进。

<span class="marginnote">Fielding 2000。gRPC 文档。本课对照，不教框架安装。</span>

### 合同不是运输

REST 面向资源与缓存；gRPC 面向 RPC 与流。浏览器与公开 API 偏好前者。LB 要不要终止 H2 因此分叉。

## 方法

对照：资源/方法 vs 服务/方法；缓存；浏览器；流。画：客户端 → HTTP → 资源；客户端 → H2 流 → RPC。与 WS：全双工消息 vs RPC 流。

```mermaid
flowchart TD
  REST["资源 + HTTP 方法"] --> CACHE["可缓存 GET"]
  GRPC["Protobuf RPC"] --> H2["HTTP/2 流"]
  GRPC --> TYP["强类型存根"]
```

## 机制

负载均衡：REST 可 L7 看路径；gRPC 看服务与方法，要 H2 感知 LB。Cookie 少用于 gRPC，多用 token 头。超时与重试：REST 靠幂等头，gRPC 有 deadline，后课幂等会收。内容协商在 REST 常见，gRPC 固定编码。

<span class="marginnote">术语翻译：REST 的「统一接口」就是把系统里的一切说成名词（资源）加几个标准动词（GET/PUT/POST/DELETE）；gRPC 则是把接口写成一句句过程调用（`OrderService.Pay`）。前者像自动售货机的固定按钮，后者像打电话给柜员点单——按钮人人会用，柜员能办复杂业务但要先拿「菜单」（.proto 文件）对齐。</span>

<span class="marginnote">常见误区：初学者容易以为「用了 JSON 和 HTTP 就是 REST」。Fielding 的 REST 要求统一接口、无状态、超媒体等约束；`POST /getUserById` 这种把动词写进路径的 RPC 风格接口，即使走 HTTP+JSON，也不是 REST——判断标准是资源建模与方法语义，不是报文格式。</span>

与端到端：两种都要应用层校验，HTTP 成功不等于业务成功。

```mermaid
flowchart TD
  REQ["同一次调用"] --> Q{"接口暴露给谁?"}
  Q -- "浏览器/第三方/CDN" --> R["REST: GET 可缓存, 状态码报错"]
  Q -- "内部微服务高频调用" --> G["gRPC: 强类型存根 + deadline"]
  R --> L1["普通 HTTP 负载均衡即可"]
  G --> L2["需 H2 感知 LB, 浏览器要 grpc-web"]
```

这张图回答的问题是「同一个业务调用，选型时先问什么」：分叉点是「接口暴露给谁」。面向浏览器与公网时，缓存与通用性优先，REST 几乎是默认；内部服务之间高频互调时，存根类型安全、流式与截止时间才值回引入成本。

<span class="marginnote">数字实例：同一笔订单查询，JSON 文本报文约 500 字节，同样的字段用 Protobuf 编码常压到 100-200 字节；内部每秒百万次调用时，带宽与反序列化 CPU 的差距乘上一百万——这就是微服务内部偏爱 gRPC 最直接的账，而公开 API 损失的可缓存性与浏览器兼容往往比这点开销贵。</span>

## 边界

本课不引入 GraphQL 的全部。序列化 JSON/Protobuf 是下一课。后课默认：REST 面向资源与缓存；gRPC 面向 RPC 与流。

公开互联网 API 强推 gRPC 会挡浏览器与缓存。

下一课[序列化：JSON / Protobuf](/cs/serialization-protobuf)。

## 小结

- REST：资源、方法、缓存合同。
- gRPC：H2 上的 RPC 与流。
- LB、浏览器、缓存需求不同。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：Fielding REST；gRPC over HTTP/2。
