---
title: 流式输出：SSE 与 Token 流协议
date: 2026-08-07
---

# 流式输出：SSE 与 Token 流协议

<div class="epigraph">
<p>让用户看到第一个字的时间，比看到最后一句话的时间更重要。</p>
<footer>—— 交互设计共识（借自响应式系统实践）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ SSE 规范（WHATWG）与 OpenAI 流式协议 ｜ 2026-08-07</p>
</div>

## 为什么从流式输出开始

LLM 生成一个长回答要几秒到几十秒。如果等服务全算完再一次性返回，用户的体感是「卡住」；如果边生成边推送，用户看到第一个字只需要几百毫秒（TTFT）。**流式输出（streaming）**是 LLM 服务的基本功：它把「一次请求」变成「一个持续推送的流」，客户端逐段渲染，体验从「等待」变成「对话」。<span class="marginnote">本专题《TTFT、TPOT》会把延迟指标拆开。<strong>流式改变的是用户可感知的延迟</strong>——TTFT 决定了「第一印象」，TPOT 决定「打字速度」。流式协议是这一切的载体。</span>

本篇讲 SSE（Server-Sent Events）协议、OpenAI 流式的增量格式、流式在工程上的坑（背压、超时、中断恢复）。

## 1 SSE：一个 HTTP 连接的持续推送

**SSE（Server-Sent Events，服务器推送事件）**是一种基于 HTTP 的简单流式协议：服务器在一个 HTTP 连接上，持续向客户端推送多个消息，直到连接关闭。格式极简：

```
data: {"choices": [{"delta": {"content": "你"}}]}

data: {"choices": [{"delta": {"content": "好"}}]}

data: [DONE]
```

每条消息以 `data:` 开头，消息之间用空行分隔，最后一条结束符是 `[DONE]`。<span class="marginnote">SSE 与 WebSocket 的区别：<strong>SSE 是「单向」服务器推送（客户端不需要发消息），WebSocket 是「双向」全双工</strong>。LLM 流式只需服务器单向推，SSE 足够且更简单——走普通 HTTP、天然被各种代理与 CDN 支持。</span>

SSE 的关键特性：**同一个连接上可以推无数条消息**。HTTP 响应头 `Content-Type: text/event-stream` 告诉客户端「这不是普通响应，是流」，之后服务器就持续写数据。

## 2 OpenAI 流式的增量格式

OpenAI 兼容的流式响应，每条 SSE 消息是一个「增量 JSON」——包含**新生成的那一段**，而不是全量：

```json
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"好"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

三个关键点：

**`delta` 是增量**：第一条通常是 `delta.role`（角色声明），中间是 `delta.content`（每段的文本增量），最后 `delta` 变成 `{}`（空对象，仅带 `finish_reason`）。<span class="marginnote">客户端解析流式的核心逻辑：<strong>把每条消息的 <code>delta.content</code> 追加到缓冲区</strong>。增量而非全量，是流式协议的灵魂——全量会让「打字机效果」失效。finish_reason 在最后一条：非流式时它在响应里；流式时它作为「收尾消息」的字段出现，客户端要等它确认「生成结束」。`[DONE]` 是流的终止：客户端遇到 `[DONE]` 就关闭连接、结束渲染。</span>

## 3 流式背后的引擎机制

流式不是「API 层做做样子」，它要穿透整个推理栈：

1. **引擎的逐 token 输出**：推理引擎（vLLM 等）的调度器天然是逐 token 的——每个 decode 步产生一个新 token。流式只是「把每个 token 即时推给用户」而不是攒起来。
2. **增量生成器**：服务端把引擎的输出包装成「增量流」，每个 token 转成一条 SSE 消息。
3. **首 token 特殊处理**：第一个 token 之前，先推「角色声明」消息；`role` 与 `model` 字段在流头出现一次。

**辨析｜易错点：`max_tokens` 与流式。** 流式时客户端照样要传 `max_tokens`——引擎按它约束生成长度，但**长度到了不会「突然断开」，而是发一条带 `finish_reason: "length"` 的收尾消息**。客户端如果只处理 `delta.content` 而忽略 `finish_reason`，会误以为「模型说完但内容被截断是 bug」，实际是超长截断的**正常信号**。

## 4 工程三坑：背压、超时与中断

流式把「网络」和「生成」耦合在一起，带来三个工程坑：

**背压（backpressure）**：客户端消费速度慢于生成速度时，TCP 缓冲会被填满，服务器写入阻塞——**生成暂停，全链路被拖慢**。解法：引擎侧的输出缓冲要有界，必要时丢弃或合并增量（把「字字推送」降级为「句句推送」）。<span class="marginnote">背压本质是「<strong>慢消费者拖累快生产者</strong>」。LLM 服务宁可「合并增量」也不让生成被网络阻塞——生成停了，GPU 就空转了。</span>
- **超时**：流式长连接会被中间代理（Nginx、负载均衡器）的 idle 超时杀掉——如果一段时间没有数据（如模型在想），连接被断开，客户端收到「连接重置」。解法：**周期性发心跳（注释行 `: ping`）**，或者调大代理的 `read_timeout`。
- **中断恢复**：网络断了，流怎么续？SSE 是「断了一了百了」——客户端只能重新发起请求。**幂等性设计**（请求带 `request_id`，服务端缓存已完成部分）是高级玩法，多数服务选择「重发整个请求」的朴素策略。

## 5 公式解析：TTFT 与流式的体验收益

流式对体验的收益用「可感知延迟」度量。设生成 $N$ 个 token，每 token 生成时间 $t_{\text{per-token}}$，首 token 延迟 $T_{\text{TTFT}}$。

- **第一步，写非流式体验**：用户等待 $T_{\text{TTFT}} + N \cdot t_{\text{per-token}}$ 才看到**任何**内容。$N=500$、$t_{\text{per-token}}=50\text{ms}$ 时，等 25 秒才看到第一字。
- **第二步，写流式体验**：用户在第 $T_{\text{TTFT}}$ 秒看到第一字，之后每个 $t_{\text{per-token}}$ 看到新字。**感知延迟从「全部生成完」变为「第一个 token 生成完」**。
- **第三步，比体感**：非流式 25 秒的等待 vs 流式 0.5 秒出第一个字 + 25 秒「打字」——**同样的总时长，体感从「卡死」变成「对话中」**。这就是流式协议的全部价值：它不改变总生成时间，但彻底改变用户的等待体验。

## 6 小结

- **SSE 是单向 HTTP 推送**：`data:` 前缀 + 空行分隔 + `[DONE]` 结束，走普通 HTTP 兼容性好。
- **增量而非全量**：每条消息的 `delta.content` 是新增段，`finish_reason` 在收尾消息出现。
- **流式穿透整个栈**：引擎逐 token 输出 → 服务端增量生成器 → SSE 推送。
- **三大工程坑**：背压（合并增量）、超时（心跳 + 调大代理超时）、中断（重新请求）。
- **流式改变体感不改变总量**：从「等 25 秒看全量」变成「0.5 秒见第一字 + 持续打字」。

在下一节，我们走出单实例，看请求如何被分发——**推理服务的负载均衡策略**。
