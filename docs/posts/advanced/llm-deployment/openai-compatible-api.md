---
title: OpenAI 兼容 API 的设计与实现
date: 2026-08-07
---

# OpenAI 兼容 API 的设计与实现

<div class="epigraph">
<p>兼容是最大的互操作性——用户换后端，代码一行不改。</p>
<footer>—— API 设计共识（源自 OpenAI API 生态）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ OpenAI API 规范与 vLLM/本地推理服务文档 ｜ 2026-08-07</p>
</div>

## 为什么从 OpenAI 兼容 API 开始

大模型部署的最终形态是「把模型变成别人能调的 API」。现实世界的客户端生态——LangChain、AutoGen、各类 Agent 框架、SaaS 工具——几乎都基于 **OpenAI 的 API 规范**编写。如果你部署的推理服务能提供「OpenAI 兼容」的接口，所有这些工具开箱即用，无需任何适配。<span class="marginnote">vLLM、SGLang、llama.cpp、TensorRT-LLM 等服务<strong>全部提供 OpenAI 兼容端点</strong>——它是事实上的行业标准。理解它的设计，等于理解「一个推理服务如何对用户呈现」。</span>

本篇讲 OpenAI API 的请求/响应结构、Chat Completions 与 Completions 的差异、实现兼容层的关键点，以及参数映射的坑。

## 1 API 的核心结构

OpenAI 兼容 API 的核心是**Chat Completions**：一个 HTTP POST 到 `/v1/chat/completions`，请求体是 JSON，核心字段：

| 字段 | 含义 | 典型值 |
| --- | --- | --- |
| model | 模型标识 | gpt-4o / Qwen2.5-72B-Instruct |
| messages | 对话消息列表 | `[{"role":"user","content":"你好"}]` |
| max_tokens | 最大生成长度 | 1024 |
| temperature | 采样温度 | 0.7 |
| top_p | 核采样 | 0.9 |
| stream | 是否流式 | false |
| stop | 停止词 | `["\n", "###"]` |
| tools / tool_choice | 工具调用 | — |

响应是 JSON，核心字段：`choices`（生成结果，含 `message` 与 `finish_reason`）、`usage`（token 统计：`prompt_tokens`、`completion_tokens`、`total_tokens`）、`id`、`created`。<span class="marginnote">`finish_reason` 是容易被忽略但极其重要的字段：它告诉客户端「为什么停了」——`stop`（命中停止词）、`length`（达到 max_tokens）、`tool_calls`（要调工具）、`content_filter`（内容过滤）等。<strong>Agent 框架靠它决定下一步</strong>。</span>

Completions（纯文本补全）是更老的接口：输入 `prompt` 字符串、输出 `text`，没有消息结构。现在大多数场景用 Chat Completions，但很多部署仍保留补全端点做兼容。

## 2 实现兼容层的三个层次

实现一个 OpenAI 兼容端点，从浅到深有三个层次：

**协议层**：正确解析请求 JSON、按规范返回响应 JSON，字段名、类型严格对齐。这是「能通」的基础——客户端校验字段类型，缺字段或类型错会直接抛错。<span class="marginnote">最容易踩的坑是<strong>可选字段的默认值语义</strong>：`temperature=0` 表示贪心解码、`temperature` 与 `top_p` 同时存在时的相互作用、`stop` 与引擎的 `stop_token_ids` 的关系。</span>
- **参数映射层**：把 OpenAI 语义映射到引擎参数。`max_tokens` → 引擎的生成上限；`temperature` / `top_p` → 采样器的温度与核参数；`stop` → 停止条件集合。**这里常被忽略的映射**：`frequency_penalty` / `presence_penalty`（频率/存在惩罚，OpenAI 特有，引擎要模拟实现或如实降级）。
- **行为层**：复现「OpenAI 的行为语义」——如 `stream=true` 时每条 SSE 的增量格式、SSE `chunk` 的标准结构、错误响应的 HTTP 状态码与 `error` 结构。行为不对，客户端逻辑会出错（例如流式解析器等不到 `[DONE]`）。

**辨析｜易错点：兼容 ≠ 逐字段复刻。** 你的引擎不支持某参数（如 `response_format` 的某些模式），正确做法是**返回明确错误**或忽略并记录，而不是「假装支持然后给错数据」。**静默错误比显式错误更危险**——客户端以为拿到了想要的结果，实际是错的。

## 3 流式与工具调用的协议细节

两个「现代 Agent 必需」的协议细节值得单独讲：

**流式（streaming）**：`stream=true` 时，响应不是一次 JSON，而是 **SSE（Server-Sent Events）**——每生成一个增量 token，发一条 `data: {...}` 事件，最后发 `data: [DONE]`。每条增量里的 `delta.content` 是「新增加的那段文本」，客户端拼接即得完整输出。流式的关键是**增量而非全量**：每次只发新增部分，客户端才能做到打字机效果与「边生成边处理」。（详见下一篇。）

**工具调用（tool calling）**：模型要调工具时，`finish_reason` 为 `tool_calls`，`message.tool_calls` 里是结构化的参数（`arguments` 是 JSON 字符串）。**实现层要做两件事**：把引擎的「输出 JSON」映射成标准的 `tool_calls` 结构；把客户端的 `tools` 定义转换成模型的提示与约束（常见做法是拼进 system prompt 或走 JSON Schema 约束解码）。

## 4 公式解析：token 统计与计费基础

`usage` 字段是 API 的「计量表」，也是计费（见《API 网关》篇）的基础：

$$\text{cost} = p \cdot \text{prompt\_tokens} + c \cdot \text{completion\_tokens}$$

- **第一步，读两个数**：`prompt_tokens` 是输入 token 数（含 system、历史、工具定义），`completion_tokens` 是输出 token 数。引擎在返回前统计，统计口径必须**与计费方一致**——vLLM 的 usage 统计基于 tokenizer 的实际切分。
- **第二步，读隐含的坑**：`prompt_tokens` 的统计口径（含不含聊天模板自动拼的 `<|im_start|>` 等特殊 token）直接影响账单。**同一请求在不同引擎的 token 计数可能有 ±10% 差异**——做计费系统时不能假设各引擎口径一致。
- **第三步，读上限约束**：`max_model_len` 是引擎的硬约束。超出时返回 400 错误（`error` 结构体）。**API 层要把这个约束翻译成清晰的错误**，而不是让引擎内部崩掉。

## 5 小结

- **OpenAI 兼容 API 是事实标准**：vLLM、SGLang、llama.cpp 等全部提供，客户端生态开箱即用。
- **核心是 Chat Completions**：`model`、`messages`、`max_tokens`、`stream` 等字段构成请求协议；`id`、`choices`、`usage` 构成响应协议。
- **实现分三层**：协议层（字段对齐）、参数映射层（语义映射）、行为层（流式/工具调用的行为语义）。
- **兼容 ≠ 假装支持**：不支持的参数要显式报错，静默错误最危险。
- **usage 是计费基础**：token 统计口径与引擎强相关，跨引擎不能假设一致。

在下一节，我们把流式输出挖到底——**流式输出：SSE 与 Token 流协议**。
