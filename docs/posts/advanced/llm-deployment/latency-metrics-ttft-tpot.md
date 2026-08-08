---
title: TTFT、TPOT、端到端延迟的定义与测量
date: 2026-08-07
---

# TTFT、TPOT、端到端延迟的定义与测量

<div class="epigraph">
<p>延迟不是一个数字，是一组数字——每一段都有它的意义。</p>
<footer>—— 延迟剖析实践（借自可观测性社区）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ vLLM 性能文档与延迟剖析实践 ｜ 2026-08-07</p>
</div>

## 为什么从延迟指标开始

谈 LLM 推理性能，最常被问的是「延迟多少」——但「延迟」这个词太模糊了。一个流式请求的体验由**首 token 延迟（TTFT）**与**逐 token 速度（TPOT）**共同决定，两个指标指向完全不同的优化方向。不懂分层测量，就不知道「慢」到底慢在哪一段。<span class="marginnote">本专题《服务监控》把 TTFT/TPOT 列为黄金指标；本篇把它们<strong>定义清楚、测量规范</strong>——监控是持续盯，压测是主动测，本篇两者都覆盖。</span>

本篇讲三个延迟指标的精确定义、各自的优化方向、以及测量时的规范（并发、预热、分布）。

## 1 三个延迟指标的精确定义

**TTFT（Time To First Token，首 token 延迟）**：请求发出到**第一个 token** 到达的时间。包含：网络传输 + 排队等待 + prefill 计算 + 首个 decode 步。**它决定用户「等多久看到第一个字」**。<span class="marginnote">TTFT 对「对话式」体验最重要：<strong>1 秒内的 TTFT 让对话像「即时」</strong>，超过 5 秒用户就开始烦躁。长 prompt 的 TTFT 由 prefill 主导。</span>
**TPOT（Time Per Output Token，每输出 token 时间）**：相邻两个输出 token 之间的时间间隔。它反映 **decode 阶段的速度**——纯访存瓶颈的体现。<span class="marginnote">TPOT 的别称还有 ITL（Inter-Token Latency）。<strong>TPOT ≈ 生成速度的倒数</strong>：TPOT 30ms ≈ 每秒 33 token，50ms ≈ 每秒 20 token。</span>
- **端到端延迟（end-to-end latency）**：请求发出到**全部生成完成**的时间。与 TTFT、TPOT、生成 token 数 $N$ 的关系：

$$\text{E2E} \approx \text{TTFT} + (N-1) \cdot \text{TPOT}$$

**端到端延迟不是独立指标，是前两者的合成**——同样的 E2E，可能「TTFT 短 + 生成慢」也可能「TTFT 长 + 生成快」，体验完全不同。这就是必须拆开看的原因。

## 2 各指标指向什么优化方向

三个指标指向三套完全不同的优化：

| 指标 | 瓶颈 | 优化手段 |
| --- | --- | --- |
| TTFT | prefill 计算、排队、调度 | 加大 prefill 算力、Prefix Caching、调度优先级、PD 分离 |
| TPOT | decode 访存、batch 大小 | KV 量化、FlashDecoding、控制 batch、投机解码 |
| E2E | 前两者之和 + 生成长度 | 缩短输出（prompt 优化）、提升 TTFT 与 TPOT |

**TTFT 和 TPOT 的优化常常冲突**：加大 batch 提升吞吐（TPOT 变好）、但每个请求的排队变长（TTFT 变差）。**这就是为什么压测必须同时报告两个指标**——只报 E2E 会把「TTFT 暴涨但 TPOT 极优」的异常服务误判为健康。

## 3 测量规范：并发、预热与分布

测量延迟最常犯的错是「单请求测一次就拿去发布」。规范的做法：

- **并发要真实**：单并发（batch=1）测出的是「纯单请求延迟」，与真实并发下的延迟完全不同——并发让 batch 变大、排队出现。**压测要覆盖目标并发档位**（如 1、8、32、64 并发）。
- **预热（warm-up）**：引擎刚启动时 CUDA kernel 首次加载、内存分配慢。**先发一批请求预热**（通常几十到几百个），再开始统计——否则冷启动延迟会污染数据。
- **看分布不看均值**：报告 P50、P95、P99（见监控篇），而不是平均。**P99 决定用户最差体验，是 SLA 的锚**。
- **固定输入长度**：TTFT 与 prompt 长度强相关。压测要**固定/分档** prompt 长度（如 512、2048、8192 token），分档报告，不能混在一起。

**辨析｜易错点：TPOT 的测量窗口。** TPOT 在流式响应里测「相邻 token 的时间差」。但引擎可能批量发 token（一个 chunk 里多个 token），直接测「每条 SSE 的时间差」会把 TPOT 误判成 chunk 间隔。**正确做法是除以每条消息里的 token 数**，或用引擎内部的逐 token 时间戳。

## 4 公式解析：E2E 延迟的分解

把一个流式请求的延迟完整分解：

$$T_{\text{E2E}} = T_{\text{net,req}} + T_{\text{queue}} + T_{\text{prefill}} + T_{\text{TTFT,decode}} + (N-1) \cdot T_{\text{TPOT}} + T_{\text{net,stream}}$$

- **第一步，读排队项 $T_{\text{queue}}$**：请求在调度器里等待的时间。并发越高，$T_{\text{queue}}$ 越大——**它是并发与延迟矛盾的根源**。
- **第二步，读 prefill 项 $T_{\text{prefill}}$**：首 token 前的计算。长 prompt 时主导 TTFT。**Prefix Caching 命中后 $T_{\text{prefill}}$ 可忽略**。
- **第三步，读 decode 项 $(N-1)T_{\text{TPOT}}$**：生成部分。$N$ 越大占比越高——长回答时端到端延迟几乎全在生成上。**对长回答，优化 TPOT 比优化 TTFT 更值**：$N=1000$ 时 TPOT 省 10ms 省 10 秒，TTFT 省 100ms 只省 0.1 秒。

## 5 小结

- **TTFT = 首 token 延迟**：prefill + 排队主导，决定「第一印象」，对话体验的关键。
- **TPOT = 每 token 时间**：decode 速度，访存瓶颈体现，≈ 生成速度的倒数。
- **E2E = TTFT + (N-1)·TPOT**：是合成指标，不是独立指标，必须拆开看。
- **优化方向冲突**：大 batch 提 TPOT 但伤 TTFT，压测必须同时报告两个指标。
- **测量规范**：真实并发、预热、看分位数、固定输入长度分档；TPOT 要按 chunk 内 token 数折算。

在下一节，我们把「并发 vs 性能」的关系系统化——**吞吐、QPS 与并发的关系曲线**。
