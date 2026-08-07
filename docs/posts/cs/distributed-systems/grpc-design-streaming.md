---
title: gRPC 的设计与流式调用
date: 2026-08-07
---

# gRPC 的设计与流式调用

<div class="epigraph">
<p>把服务当作接口来定义，让客户端的体验像调用本地函数一样自然，同时让不同语言、不同团队之间协作无间。</p>
<footer>—— gRPC 官方设计理念（gRPC Concepts, grpc.io）</footer>
</div>

<div class="article-byline">
<p>第三级 · 分布式系统 ｜ DDIA 第4章 / gRPC 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 gRPC 开始

前两篇我们理解了 RPC 的抽象与序列化格式。但 RPC 框架要真正好用，还差几块拼图：跨语言、跨平台、双向流、跨进程的错误处理。**gRPC** 是谷歌 2015 年开源、如今已成事实标准的 RPC 框架，它把三样东西焊在一起：**Protobuf**（接口与编码）、**HTTP/2**（传输层）、**stub 代码生成**（开发体验）。<span class="marginnote">gRPC 名字里的 RPC 前面那个 g 曾代表「Google」，如今官方解释为「gRPC Remote Procedure Calls」——一个递归缩写，玩笑里透着工程团队对协议的认真。它诞生于谷歌内部 Stubby 框架（2001 年至今服务着谷歌几乎所有内部调用）的教训总结。</span>

## 1 gRPC 的四个核心设计选择

- **IDL 用 Protobuf**：`.proto` 文件声明 service 与 message，`protoc` 插件生成客户端与服务端骨架。接口即契约，生成代码即实现。
- **传输层用 HTTP/2**：多路复用（一个连接并发多个请求）、二进制帧、头部压缩（HPACK）、**双向流**。相比 HTTP/1.1 的「一连接一请求」，HTTP/2 让 RPC 能高效长连接复用。
- **默认使用 Protobuf 二进制编码**：紧凑、快，但也因此失去了 curl 直接调试的便利——gRPC 提供 `grpcurl` 与 reflection 服务弥补。
- **契约先行的开发流程**：先写 `.proto`，再实现服务端与客户端。契约是团队间、服务间的第一公民，接口改动需要走兼容规则。<span class="marginnote"><strong>辨析｜易错点：</strong>gRPC 不等于「gRPC-Web」。浏览器里跑的是 gRPC-Web 或 Connect，因为浏览器无法直接使用 HTTP/2 的 trailers 与全双工流。凡是在浏览器/移动端调用 gRPC，都要注意这条边界。</span>

## 2 四种调用形态：一元、服务端流、客户端流、双向流

gRPC 最亮眼的特性是**流式调用**。`.proto` 里可以声明四种 service 方法形态：

- **一元（Unary）**：`rpc GetUser(UserId) returns (User)`——一次请求一次响应，最传统的 RPC。
- **服务端流（Server Streaming）**：`rpc Watch(UserId) returns (stream Event)`——客户端发一次，服务端持续推多个结果。典型：订阅行情、日志 tail、进度推送。
- **客户端流（Client Streaming）**：`rpc Upload(stream Chunk) returns (Status)`——客户端持续发，服务端最后统一回一个响应。典型：上传大文件、批处理任务。
- **双向流（Bidirectional Streaming）**：`rpc Chat(stream Msg) returns (stream Msg)`——两端同时发，消息顺序各自独立保证。典型：聊天室、实时代理转发。

流式调用的底层是 HTTP/2 的多路复用：同一个连接上可同时跑成百上千个流，每个流独立维护自己的消息序列。<span class="marginnote">理解双向流的语义要点：两个方向的流是<strong>相互独立</strong>的——服务端可以边读边写，不必等客户端发完。但 gRPC 默认不保证不同流的消息全局有序，业务需要时要在消息里自带序号或时间戳，这正好预告后面「逻辑时钟」章节的动机。</span>

## 3 连接管理与超时：channel 与 deadline

gRPC 的客户端抽象叫 **channel**：它代表到服务端的一簇连接，自动处理连接池、负载均衡、重连。channel 之上每次调用可设 **deadline（截止时间）**——与「超时」不同，deadline 是绝对时间点，客户端会把它通过 HTTP/2 传播到服务端，服务端据此提前终止慢处理。<span class="marginnote">deadline 传播是 gRPC 处理「部分失败」的独门武器：调用链 A→B→C 时，A 的 deadline 会沿链路传到 C，任何一个环节超时，整条链路都能尽早失败，而不是各自傻等自己的超时。这与分布式追踪里的上下文传播是同一条通道。</span>

gRPC 还内建**重试与负载均衡**：客户端可配置对特定错误码（如 `UNAVAILABLE`）重试，支持 round-robin、least-request、ring-hash 等均衡策略。注意重试与流式调用的配合——已发出部分数据的流重试会复杂得多，这直接关系后面「幂等性」的讨论。

## 4 公式解析：一元调用的延迟分解

把 gRPC 一元调用在 HTTP/2 上的代价写清楚：

$$
T_{\text{unary}} = 2T_{\text{RTT}} + t_{\text{ser}} + t_{\text{deser}} + t_{\text{proc}} + t_{\text{hp}}
$$

逐项拆解：

- $2T_{\text{RTT}}$：HTTP/2 上请求 + 响应各一个往返。注意 HTTP/2 多路复用后**无握手重复开销**——建立连接时一次 TLS 握手分摊到后续所有调用上。
- $t_{\text{ser}} + t_{\text{deser}}$：Protobuf 序列化与反序列化，规模线性、常数很小。
- $t_{\text{proc}}$：服务端业务逻辑。
- $t_{\text{hp}}$：HTTP/2 帧与 HPACK 头部压缩的固定开销——极小，但每个流都有。

这条式子点出 gRPC 的性能优势来源：**HTTP/2 把每个请求的固定开销压到接近零，真正的成本只剩网络往返与业务逻辑**。若业务是「一次调用拉 1000 条小数据」，用服务端流替代一元循环，能省掉 999 次往返，这正是流式调用的性能价值。<span class="marginnote">工程启示：gRPC 的性能瓶颈几乎总是「调用次数 × 网络 RTT」，而不是序列化或传输层。设计接口时优先减少往返：批量、流式、fan-in 聚合，都比优化单次编码更划算。</span>

## 5 小结

- gRPC = **Protobuf（契约）+ HTTP/2（传输）+ stub 生成（体验）** 的三件套。
- 四种调用形态：一元、服务端流、客户端流、双向流；流的顺序各自独立，跨流全局有序要靠业务自己保证。
- **channel** 管理连接池与负载均衡；**deadline** 是绝对截止时间并沿调用链传播，让整条链路尽早失败。
- 内建重试、均衡、鉴权、追踪等横切能力，是现代 RPC 框架的标配。
- 性能优势来自 HTTP/2 的低固定开销；接口设计优先减少往返，流式与批量是主要手段。

在下一节，我们走出「请求-响应」的范式，看看异步解耦的另一条路——**消息传递模型**：点对点、发布订阅与消息队列。
